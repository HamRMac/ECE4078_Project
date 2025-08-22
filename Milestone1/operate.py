# teleoperate the robot and perform SLAM
# will be extended in following milestones for system integration

# basic python packages
import numpy as np
import cv2 
import os, sys
import time

# import utility functions
#sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
from slam.joint_optimiser import JointOptimiser2D


class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        self.map_id = 0
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.07, cube_depth=0.08) # size of the ARUCO markers 

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False,
                        'optimise_now': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        # Robot params
        self.left_wheel_cov = 1
        self.right_wheel_cov = 1
        # Joint (bundle) optimizer support
        self.joint_optimiser = JointOptimiser2D()
        self._joint_opt_frame_counter = 0
        self._joint_opt_interval = None  # run automatically every N collected frames (>=2 markers)

        if (self._joint_opt_interval == None):
            print(f"Automatic optiisation is disabled")
        else:
            print(f"Automatic optiisation is activated every {self._joint_opt_interval} frames")

    # wheel control
    def control(self):    
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        # running in sim
        if args.ip == 'localhost':
            drive_meas = measure.Drive(lv, rv, dt)
        # running on physical robot (right wheel reversed)
        else:
            drive_meas = measure.Drive(lv, -rv, dt, left_cov=self.left_wheel_cov, right_cov=self.right_wheel_cov)
        self.control_clock = time.time()
        return drive_meas
        
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)
            # Collect frame for joint optimization (only when SLAM running)
            self._collect_joint_opt_frame(lms)

            # Periodic automatic optimization
            if (self._joint_opt_interval is not None and
                self._joint_opt_frame_counter > 0 and 
                self._joint_opt_frame_counter % self._joint_opt_interval == 0):
                self._run_joint_optimisation(auto=True)

            # Manual trigger
            if self.command['optimise_now']:
                self._run_joint_optimisation(auto=False)
                self.command['optimise_now'] = False

    def _collect_joint_opt_frame(self, lms):
        """Accumulate a frame (robot pose + >=2 marker observations) for joint optimisation."""
        if not lms or len(lms) < 2:
            return
        pose = self.ekf.robot.state.flatten()  # [x, y, theta]
        obs = []
        for lm in lms:
            try:
                vec = lm.position.flatten()  # body-frame 2D vector
            except Exception:
                continue
            obs.append((int(lm.tag), vec))
        if len(obs) < 2:
            return
        self.joint_optimiser.add_frame(pose, obs)
        self._joint_opt_frame_counter += 1

    def _run_joint_optimisation(self, auto=False):
        self.notification = f'Optimising map please wait...'
        cam_poses, marker_map = self.joint_optimiser.optimise()
        if not marker_map:
            if not auto:
                print('[JointOpt] Not enough data to optimize (need frames with >=2 markers).')
            return
        # Update EKF landmark estimates with optimized values (keep covariance as-is)
        updated = 0
        for idx, tag in enumerate(self.ekf.taglist):
            if tag in marker_map:
                self.ekf.markers[:, idx] = marker_map[tag].reshape(2)
                updated += 1
        if auto:
            print(f'[JointOpt][Auto] Optimised map with {len(marker_map)} markers; updated {updated}.')
            self.notification = f'Optimised map (auto)'
        else:
            print(f'[JointOpt][Manual] Optimised map with {len(marker_map)} markers; updated {updated}.')
            self.notification = f'Optimised map (manual)'
        # (Optional) could reset optimiser to start a new batch
        # self.joint_optimiser = JointOptimiser2D()

    # save images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            print("WARN: Dividing scale by 2 for LH")
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(slam=self.ekf, map_id=self.map_id)
            self.notification = f'Map {self.map_id} is saved'
            self.map_id += 1
            self.command['output'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad)) # M2
        self.put_caption(canvas, caption='Detector (M2)',
                         position=(h_pad, 240+2*v_pad)) # M3
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))

        # --- Statistics Panel (right side) ---
        try:
            # Clear panel area to prevent text overdraw artifacts
            panel_x_clear = 700  # left boundary of panel
            panel_w_clear = canvas.get_width() - panel_x_clear
            if panel_w_clear > 0:
                panel_rect = pygame.Rect(panel_x_clear, 0, panel_w_clear, canvas.get_height())
                pygame.draw.rect(canvas, (0, 0, 0), panel_rect)
            # Optional vertical separator line
            pygame.draw.line(canvas, (60, 60, 60), (panel_x_clear, 0), (panel_x_clear, canvas.get_height()), 2)
            # Panel geometry
            panel_x = 700  # original width boundary
            panel_pad_x = panel_x + 10
            panel_pad_y = v_pad
            line_h = 24

            # Header with units note (only once)
            header = STAT_FONT.render('Stats (x,y m; θ deg)', False, (255, 255, 0))
            canvas.blit(header, (panel_pad_x, panel_pad_y))
            y_cursor = panel_pad_y + line_h + 4

            # Frame count with >=2 markers for joint optimiser
            frame_count = self._joint_opt_frame_counter
            fc_text = STAT_FONT.render(f'Frames (>=2 mk): {frame_count}', False, text_colour)
            canvas.blit(fc_text, (panel_pad_x, y_cursor))
            y_cursor += line_h

            # Robot pose
            rx, ry, rth = self.ekf.robot.state.flatten()
            rth_deg = np.rad2deg(rth)
            pose_text = STAT_FONT.render(f'Robot: x={rx:.2f} y={ry:.2f} θ={rth_deg:.1f}', False, text_colour)
            canvas.blit(pose_text, (panel_pad_x, y_cursor))
            y_cursor += line_h

            # Marker positions (estimated)
            # Iterate over known tags and their positions
            if hasattr(self.ekf, 'taglist') and hasattr(self.ekf, 'markers'):
                # Pair each tag with its column index, then sort by tag (ascending)
                tag_index_pairs = list(enumerate(self.ekf.taglist))
                # taglist elements are expected ints; if not, attempt int conversion for sorting
                try:
                    tag_index_pairs.sort(key=lambda p: int(p[1]))
                except Exception:
                    tag_index_pairs.sort(key=lambda p: p[1])
                for idx, tag in tag_index_pairs:
                    if idx < self.ekf.markers.shape[1]:
                        mx, my = self.ekf.markers[:, idx]
                        m_text = STAT_FONT.render(f'M{int(tag)}: {mx:.2f},{my:.2f}', False, (180, 220, 255))
                        canvas.blit(m_text, (panel_pad_x, y_cursor))
                        y_cursor += line_h
                        if y_cursor > 620:  # Prevent overflow; truncate if panel too small
                            more_text = STAT_FONT.render('...more', False, (255, 100, 100))
                            canvas.blit(more_text, (panel_pad_x, y_cursor))
                            break
        except Exception as e:
            err_text = STAT_FONT.render('Stats Err', False, (255, 0, 0))
            canvas.blit(err_text, (panel_pad_x, v_pad))
        # --- End Statistics Panel ---
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            ########### replace with your M1 codes ###########
            # Note: self.command['motion'] = [forward speed, turning speed]
            cmd = [0, 0]  # reset the command
            # Add/subtract to the relevant element on key press/release
            # if event is any of the direction keys
            # up, down, left, right, w, a, s, d
            if event.type in (pygame.KEYUP, pygame.KEYDOWN) and event.key in (pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s, pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d):
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        self.command['motion'][0] += 1
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        self.command['motion'][0] -= 1
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        self.command['motion'][1] += 1
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        self.command['motion'][1] -= 1
                elif event.type == pygame.KEYUP:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        self.command['motion'][0] -= 1
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        self.command['motion'][0] += 1
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        self.command['motion'][1] -= 1
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        self.command['motion'][1] += 1
            ####################################################
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.command['output'] = True
            # manual joint optimisation trigger (press 'o')
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_o:
                self.command['optimise_now'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    args, _ = parser.parse_known_args()

    print("Operate.py V2.0 (Optimiser Version)")
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    # Readable system font for statistics panel (fallback to default if unavailable)
    try:
        STAT_FONT = pygame.font.SysFont('Arial', 20)
    except Exception:
        STAT_FONT = pygame.font.Font(None, 22)
    
    # Extend width to add a statistics panel on the right (keep original area unchanged)
    STATS_PANEL_WIDTH = 300
    ORIGINAL_WIDTH = 700
    width, height = ORIGINAL_WIDTH + STATS_PANEL_WIDTH, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and (event.key not in(pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s, pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d)): # Squashes a bug (immediately detects a keyup without a keydown, pibot constantly travels with no keys pressed)
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)

    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        # Print the robot pose (one line)
        # Timestamped with fixed spacing
        x, y, th = operate.ekf.robot.state
        '''
        #print(f"Robot Pose ({pygame.time.get_ticks()} ms): x: {x[0]}, y: {y[0]}, t: {np.rad2deg(th[0])}")
        if hasattr(operate.ekf, 'taglist') and hasattr(operate.ekf, 'markers'):
            # Pair each tag with its column index, then sort by tag (ascending)
            tag_index_pairs = list(enumerate(operate.ekf.taglist))
            # taglist elements are expected ints; if not, attempt int conversion for sorting
            try:
                tag_index_pairs.sort(key=lambda p: int(p[1]))
            except Exception:
                tag_index_pairs.sort(key=lambda p: p[1])
            for idx, tag in tag_index_pairs:
                if idx < operate.ekf.markers.shape[1]:
                    mx, my = operate.ekf.markers[:, idx]
                    print(f"Tag {tag} ({pygame.time.get_ticks()} ms): x: {mx}, y: {my}")
         '''
