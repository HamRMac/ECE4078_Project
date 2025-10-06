"""Operate module (Milestone2)

Merged functionality from Milestone1 and Milestone2:
 - Teleoperation & SLAM (EKF + ArUco)
 - Joint (bundle) optimisation of robot poses & marker map
 - YOLO object detection (triggered or continuous)
 - Enhanced keyboard controls (supports arrows & WASD, map saving, optimisation, detection)
 - Statistics side panel (robot pose & marker positions)

Keys:
    Movement:  UP/W, DOWN/S, LEFT/A, RIGHT/D (press/release accumulative), SPACE stop
    SLAM:      ENTER start/pause (needs >=2 markers to pause), R reset (double press)
    Map:       M or S save map (with incremental map_id)
    Image:     I save raw camera image
    Detect:    P run detection, N save last detection snapshot
    Optimise:  O run joint optimisation now (needs accumulated frames with >=2 markers)
    Quit:      ESC or window close
"""

import os, sys, time
import cv2
import numpy as np
import math

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
import pygame                       # python package for GUI
import shutil                       # python package for file operations

import logging

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
from slam.joint_optimiser import JointOptimiser2D

from perception.fruit_ranger import (
    FruitRanger,
    bbox_ratio_ok,
)

from util.StandardValues import TARGET_HEIGHTS_DICT, TARGET_TYPES

# import YOLO components 
from YOLO.detector import Detector

log = logging.getLogger(__name__)

class Operate:
    def __init__(self, args):
        # Dataset folder fresh each run
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # Data / robot handle
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # SLAM init
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07, cube_depth=0.08)
        self.ekf_last_time = time.monotonic()

        # Joint (bundle) optimisation support
        self.joint_optimiser = JointOptimiser2D()
        self._joint_opt_frame_counter = 0
        self._joint_opt_interval = None  # set to an int N to enable auto optimisation every N frames
        if self._joint_opt_interval is None:
            print("[JointOpt] Automatic optimisation disabled")
        else:
            print(f"[JointOpt] Automatic optimisation every {self._joint_opt_interval} frames")

        # Data recording
        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')

        # Create a fruit ranger instance
        self.fruit_ranger = FruitRanger(pixel_centroid_sigma_px=2.0,
                                        pixel_height_sigma_px=3.0,
                                        range_scale_beta=0.02,
                                        ekf_weight_gamma=1.0)

        # Commands / state flags
        self.command = {'motion': [0, 0],
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False,
                        'optimise_now': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.saved_inference = False
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.map_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.pred_notifier = False

        # Dictionary to hold currently-detected objects (for display in SLAM)
        self.current_objects = {}

        # Timer (5 min)
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()

        # Images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.detector_output = np.zeros([240, 320], dtype=np.uint8)
        if args.yolo_model == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.yolo_model)
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')

        # Wheel noise params (tunable if required)
        self.left_wheel_cov = 1
        self.right_wheel_cov = 1

        # Detector FPS throttling (non-blocking)
        # Default 1 FPS, configurable via --detector_fps
        try:
            self.detector_fps = float(getattr(args, 'detector_fps', 1.0))
        except Exception:
            self.detector_fps = 1.0
        self._detector_min_interval = 0.0 if self.detector_fps <= 0 else (1.0 / self.detector_fps)
        self._last_detect_ts = 0.0

    # wheel control
    def control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(self.command['motion'])
        if self.data is not None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        # Physical robot (right wheel rotation accounted for) with covariance
        drive_meas = measure.Drive(lv, rv, dt, left_cov=self.left_wheel_cov, right_cov=self.right_wheel_cov)
        self.control_clock = time.time()
        return drive_meas

    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()

        if self.data is not None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self,drive_meas_ctrl):
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
        elif self.ekf_on:
            '''
            # Get time
            time_now = time.monotonic()
            dt = time_now - self.ekf_last_time
            l_vel, r_vel = self.pibot.get_wheel_velocity(prefer_measured=True)
            print(f"Time: {time_now} -> {l_vel}, {r_vel}")
            self.ekf_last_time = time_now
            drive_meas = measure.Drive(l_vel, r_vel, dt, left_cov=self.left_wheel_cov, right_cov=self.right_wheel_cov)
            '''
            l_vel, r_vel, dt = self.pibot.get_wheel_velocity_diff()
            drive_meas = measure.Drive(l_vel, -r_vel, dt, left_cov=self.left_wheel_cov, right_cov=self.right_wheel_cov)
            # print(f"{time.monotonic()} -> dt: {dt:.2f} w./ {l_vel:.2f}, {r_vel:.2f}")
            
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)
            # Joint optimisation data collection
            self._collect_joint_opt_frame(lms)
            # Auto optimisation
            if (self._joint_opt_interval is not None and
                self._joint_opt_frame_counter > 0 and
                self._joint_opt_frame_counter % self._joint_opt_interval == 0):
                self._run_joint_optimisation(auto=True)
            # Manual trigger
            if self.command['optimise_now']:
                self._run_joint_optimisation(auto=False)
                self.command['optimise_now'] = False

    def _collect_joint_opt_frame(self, lms):
        if not lms or len(lms) < 2:
            return
        pose = self.ekf.robot.state.flatten()
        obs = []
        for lm in lms:
            try:
                vec = lm.position.flatten()
            except Exception:
                continue
            obs.append((int(lm.tag), vec))
        if len(obs) < 2:
            return
        self.joint_optimiser.add_frame(pose, obs)
        self._joint_opt_frame_counter += 1

    def _run_joint_optimisation(self, auto=False):
        self.notification = 'Optimising map please wait...'
        cam_poses, marker_map = self.joint_optimiser.optimise()
        if not marker_map:
            if not auto:
                print('[JointOpt] Not enough data (need frames with >=2 markers).')
            return
        updated = 0
        for idx, tag in enumerate(self.ekf.taglist):
            if tag in marker_map:
                self.ekf.markers[:, idx] = marker_map[tag].reshape(2)
                updated += 1
        if auto:
            print(f'[JointOpt][Auto] Optimised {len(marker_map)} markers; updated {updated}.')
            self.notification = 'Optimised map (auto)'
        else:
            print(f'[JointOpt][Manual] Optimised {len(marker_map)} markers; updated {updated}.')
            self.notification = 'Optimised map (manual)'

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            # Non-blocking FPS throttle
            now = time.time()
            if self._detector_min_interval > 0 and (now - self._last_detect_ts) < self._detector_min_interval:
                return
            # need to convert the colour before passing to YOLO
            yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

            pose_now = self.ekf.robot.state.flatten().tolist()

            self.detector_output, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)
            self._last_detect_ts = now

            # covert the colour back for display purpose
            self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_RGB2BGR)

            # self.command['inference'] = False     # uncomment this if you do not want to continuously predict
            self.file_output = (yolo_input_img, self.ekf)
            self.saved_inference = False

            # Reset current objects
            self.current_objects = {}
            
            # Process detection results
            for detection in self.detector_output: # Grab each bbox
                target_class = detection[0]
                bbox = detection[1]  # [x, y, w, h] in px (top-left origin for your detector)

                # Accept only known classes
                if target_class not in TARGET_HEIGHTS_DICT:
                    continue

                # Aspect-ratio sanity filter (±15%)
                #if not bbox_ratio_ok(target_class, bbox, TARGET_HEIGHTS_DICT, tol=0.15):
                #    continue

                # Estimate range/bearing
                true_height = TARGET_HEIGHTS_DICT[target_class]
                est = self.fruit_ranger.from_bbox_height(bbox, true_height)
                print(f"{time.monotonic()}:\n Detection: {target_class} @({bbox}) -> {est['r']:.2f}m, {np.rad2deg(est['theta']):.1f}°" if est is not None else " -> Estimation failed")
                if est is None:
                    continue
                
                # Calculate global position of the target
                r = float(est['r']); th = float(est['theta'])
                rx, ry, rth = float(pose_now[0]), float(pose_now[1]), float(pose_now[2])
                wx = rx + r * math.cos(rth + th)
                wy = ry + r * math.sin(rth + th)

                print(f"   -> Global pos: ({wx:.2f}, {wy:.2f})")

                # Add current object to dictionary
                if target_class not in self.current_objects:
                    self.current_objects[target_class] = []
                self.current_objects[target_class].append((wx, wy))

            # self.notification = f'{len(self.detector_output)} target type(s) detected'

    # save raw images taken by the camera
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
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        # Save SLAM map (with incremental ID)
        if self.command['output']:
            self.output.write_map(slam=self.ekf, map_id=self.map_id)
            self.notification = f'Map {self.map_id} is saved'
            self.map_id += 1
            self.command['output'] = False
        # Save detector inference snapshot
        if self.command['save_inference']:
            if self.file_output is not None:
                    self.pred_fname = self.output.write_image(self.file_output[0], self.file_output[1]) 
                    if not self.saved_inference:
                        self.notification = f'New Prediction saved -> {self.pred_fname}'
                    else:
                        self.notification = f'Duplicate Prediction saved -> {self.pred_fname}'
                    self.saved_inference = True
            else:
                self.notification = 'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # SLAM panel
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
                                            not_pause=self.ekf_on,
                                            draw_subgrid=True,
                                            grid_spacing_m=0.9,
                                            subgrid_spacing_m=0.3,
                                            grid_at_origin=False,
                                            current_objects=self.current_objects # For testing
                                            )
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # Detector panel
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240 + 2*v_pad))

        # Captions
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector', position=(h_pad, 240 + 2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        # Notifications
        notifiation = TEXT_FONT.render(self.notification, False, text_colour)
        canvas.blit(notifiation, (h_pad + 10, 596))

        # Countdown
        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain) % 2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))

        # --- Statistics Panel (right side) ---
        try:
            panel_x_clear = 700
            panel_w_clear = canvas.get_width() - panel_x_clear
            if panel_w_clear > 0:
                panel_rect = pygame.Rect(panel_x_clear, 0, panel_w_clear, canvas.get_height())
                pygame.draw.rect(canvas, (0, 0, 0), panel_rect)
            pygame.draw.line(canvas, (60, 60, 60), (panel_x_clear, 0), (panel_x_clear, canvas.get_height()), 2)
            panel_x = 700
            panel_pad_x = panel_x + 10
            panel_pad_y = v_pad
            line_h = 24

            header = STAT_FONT.render('Stats (x,y m; θ deg)', False, (255, 255, 0))
            canvas.blit(header, (panel_pad_x, panel_pad_y))
            y_cursor = panel_pad_y + line_h + 4

            frame_count = self._joint_opt_frame_counter
            fc_text = STAT_FONT.render(f'Frames (>=2 mk): {frame_count}', False, text_colour)
            canvas.blit(fc_text, (panel_pad_x, y_cursor))
            y_cursor += line_h

            rx, ry, rth = self.ekf.robot.state.flatten()
            rth = wrap_pi(rth)
            rth_deg = np.rad2deg(rth)
            pose_text = STAT_FONT.render(f'Robot: x={rx:.2f} y={ry:.2f} θ={rth_deg:.1f}', False, text_colour)
            canvas.blit(pose_text, (panel_pad_x, y_cursor))
            y_cursor += line_h

            if hasattr(self.ekf, 'taglist') and hasattr(self.ekf, 'markers'):
                tag_index_pairs = list(enumerate(self.ekf.taglist))
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
                        if y_cursor > 620:
                            more_text = STAT_FONT.render('...more', False, (255, 100, 100))
                            canvas.blit(more_text, (panel_pad_x, y_cursor))
                            break
        except Exception:
            err_text = STAT_FONT.render('Stats Err', False, (255, 0, 0))
            canvas.blit(err_text, (panel_x + 10, v_pad))
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
        canvas.blit(caption_surface, (position[0], position[1] - 25))

    # Keyboard teleoperation (merged enhanced version)
    def update_keyboard(self):
        for event in pygame.event.get():
            # Movement keys (press/release increments)
            if event.type in (pygame.KEYDOWN, pygame.KEYUP) and event.key in (pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s, pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d):
                delta = 1 if event.type == pygame.KEYDOWN else -1
                if event.key in (pygame.K_UP, pygame.K_w):
                    self.command['motion'][0] += delta
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.command['motion'][0] -= delta
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    self.command['motion'][1] += delta
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.command['motion'][1] -= delta
            # Stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # Save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # Save SLAM map (M or S)
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_m, pygame.K_s):
                self.command['output'] = True
            # Joint optimisation manual trigger
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_o:
                self.command['optimise_now'] = True
            # Reset SLAM map (double press R)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm += 1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # SLAM run/pause toggle
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
                    self.notification = 'SLAM is running' if self.ekf_on else 'SLAM is paused'
            # Detection trigger
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # Save detection snapshot
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # Optimisation (alternate O) already handled
            # Quit conditions
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()


def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2*np.pi) - np.pi


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/V5.pt')
    parser.add_argument("--detector_fps", type=float, default=1.0,
                        help="Max detector runs per second (non-blocking throttle). 0 or negative disables throttling.")
    args, _ = parser.parse_known_args()

    print("----- PiBot Final Demo (Manual Control) -----")
    print(f"Connecting to PiBot at {args.ip}:{args.port}...")
    print(f"Using calibration files from {args.calib_dir}")
    if args.save_data:
        print("Data saving is enabled.")
    if args.play_data:
        print("Data playback is enabled.")
    print(f"Using YOLO model: {args.yolo_model}")
    try:
        print(f"Detector FPS limit: {args.detector_fps} fps")
    except Exception:
        pass

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    try:
        STAT_FONT = pygame.font.SysFont('Arial', 20)
    except Exception:
        STAT_FONT = pygame.font.Font(None, 22)

    STATS_PANEL_WIDTH = 300
    ORIGINAL_WIDTH = 700
    width, height = ORIGINAL_WIDTH + STATS_PANEL_WIDTH, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 Final Demo Manual Mode')
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
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
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
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
