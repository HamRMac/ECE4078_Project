# M3 - Autonomous fruit searching
# Level 1: Semi-auto with waypoints

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import logging

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi
import util.measure as measure

# Import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# Import navigation components
from navigation.controller import ControllerManager
from planning.astar import AStarPlanner
from gui.og_viewer import OGViewer
from planning.grid_map import GridMap

# Module logger
log = logging.getLogger(__name__)

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])
        aruco_true_pos_id = np.empty([10, 3])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                    aruco_true_pos_id[9][0] = x
                    aruco_true_pos_id[9][1] = y
                    aruco_true_pos_id[9][2] = 10
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
                    aruco_true_pos_id[marker_id][0] = x
                    aruco_true_pos_id[marker_id][1] = y
                    aruco_true_pos_id[marker_id][2] = int(marker_id + 1)
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos, aruco_true_pos_id


def read_search_list(list_path):
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open(list_path, 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits

    @output ordered list of positions of the target fruit
    """
    search_poses = []
    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        # Only print coordinates if present in provided map
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                try:
                    search_poses.append([fruit_true_pos[i][0],fruit_true_pos[i][1]])
                    print('{}) {} at [{}, {}]'.format(
                        n_fruit,
                        fruit,
                        np.round(fruit_true_pos[i][0], 1),
                        np.round(fruit_true_pos[i][1], 1)))
                except Exception:
                    # In minimal map there may be no fruit positions
                    print('{}) {}'.format(n_fruit, fruit))
        n_fruit += 1

    return search_poses


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically
def drive_to_point(waypoint, robot_pose, controller_kind: str = "ttg"):
    # imports camera / wheel calibration parameters
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    # Control parameters
    ctrl_rate_hz = 10.0
    dt_loop = 1.0 / ctrl_rate_hz
    max_duration = 20.0  # seconds safety timeout

    # Select controller
    ctrl_mgr = ControllerManager(controller_kind)

    t0 = time.time()
    arrived = False

    while True:
        # Safety timeout
        if time.time() - t0 > max_duration:
            print("[drive_to_point] Timeout; stopping.")
            break

        # Refresh pose from EKF
        pose = get_robot_pose(ppi, aruco_det, ekf)
        fwd_cmd, turn_cmd, fwd_tick, turn_tick, done = ctrl_mgr.compute(pose, waypoint)
        if done:
            arrived = True
            break
        ppi.set_velocity([fwd_cmd, turn_cmd], tick=fwd_tick, turning_tick=turn_tick, time=0)

        time.sleep(dt_loop)

    # Stop safely
    ppi.set_velocity([0, 0])
    if arrived:
        log.info("Arrived at waypoint (%.2f, %.2f)", waypoint[0], waypoint[1])
    else:
        log.info("Stopped before arrival at waypoint (%.2f, %.2f)", waypoint[0], waypoint[1])


def get_robot_pose(penguin_pi, aruco_detector, ekf):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # Dummy robot_pose
    robot_pose = [0.0,0.0,0.0] # will be replaced by EKF state below

    # Get the image
    img = penguin_pi.get_image()
    _ = aruco_detector.detect_marker_positions(img)

    # Reset the EKF
    # Attempt to predict the location using last drive command and dt
    now = time.time()
    if not hasattr(get_robot_pose, "_last_t"):
        get_robot_pose._last_t = now
    dt = max(1e-3, now - get_robot_pose._last_t)
    get_robot_pose._last_t = now

    # Use last commanded wheel velocities from PenguinPi
    try:
        l_vel, r_vel = penguin_pi.wheel_vel
    except Exception:
        l_vel, r_vel = 0.0, 0.0
    # Match M2 convention: invert right wheel for physical robot
    if getattr(penguin_pi, 'ip', '') == 'localhost':
        drive_meas = measure.Drive(l_vel, r_vel, dt)
    else:
        drive_meas = measure.Drive(l_vel, -r_vel, dt)
    ekf.predict(drive_meas)
    # Get any visible arucos and then update EKF
    lms, _ = aruco_detector.detect_marker_positions(img)
    ekf.update(lms)

    # Read pose from EKF robot state
    try:
        rs = ekf.robot.state.flatten()
        robot_pose = [float(rs[0]), float(rs[1]), float(rs[2])]
    except Exception:
        robot_pose = [0.0, 0.0, 0.0]
    # Debug pose log
    try:
        log.debug("Pose x=%.3f y=%.3f th=%.2f dt=%.3f lv=%.1f rv=%.1f lms=%d", robot_pose[0], robot_pose[1], robot_pose[2], dt, l_vel, r_vel, len(lms))
    except Exception:
        pass

    ####################################################

    return robot_pose

def calc_waypoint(search_pose, robot_pose, d=0):
    # Returns waypoint for next fruit from current position, needs the pose of the next target fruit, current robot pose and the distance from the centre of the fruit the waypoint should be
    # Should be run once at the start of the run and once again every time a successive waypoint is reached
    th = np.arctan2(robot_pose[1]-search_pose[1],robot_pose[0]-search_pose[0])

    x_waypoint = search_pose[0] + d * np.cos(th)
    y_waypoint = search_pose[1] + d * np.sin(th)

    waypoint = [x_waypoint, y_waypoint]

    return waypoint
    


# wheel and camera calibration for SLAM
def init_ekf(datadir, ip):
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

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M3_prac_map_min.txt', help='Path to true map file (full/part/min)')
    parser.add_argument("--shopping_list", type=str, default='M3_prac_shopping_list.txt', help='Path to shopping list file')
    parser.add_argument("--calib_dir", type=str, default='calibration/param/', help='Directory containing calibration files')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--controller", type=str, default='ttg', choices=['ttg','ppc','rhp'], help='Controller type: turn-then-go (ttg), pure pursuit (ppc), or receding horizon (rhp)')
    parser.add_argument("--no_run", action='store_true', help='Only load map, world model, and occupancy grid; do not start autonomy')
    parser.add_argument("--log", type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='Logging level')
    parser.add_argument("--level", type=int, default=1)
    args, _ = parser.parse_known_args()

    # Configure root logging early
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format='[%(levelname)s] %(name)s: %(message)s')
    log.info("auto_fruit_search starting (ip=%s, port=%s, controller=%s, no_run=%s)", args.ip, args.port, args.controller, args.no_run)

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    log.info("Loading map file: %s", args.map)
    fruits_list, fruits_true_pos, aruco_true_pos, aruco_true_pos_id = read_true_map(args.map)
    # read shopping list
    search_list = read_search_list(args.shopping_list)
    try:
        search_poses = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    except Exception:
        log.info("Loaded shopping list (positions not available in minimal map).")

    # Build world model and occupancy grid
    try:
        # Fixed arena 2.4 x 2.4 m centered at origin => [-1.4, -1.4] .. [1.4, 1.4]
        grid = GridMap(res=0.02, margin=0.0, robot_radius=0.09,
                       inflation_margin=0.05, boundary_margin=0.01,
                       arena_bounds_wm=(-1.4, -1.4, 1.4, 1.4))
        grid.build_from_aruco(aruco_true_pos)
        log.info("[WM] Occupancy grid built: size=%s res=%.3f m", str(grid.size), grid.res)
    except Exception as e:
        log.exception("[WM] Building occupancy grid failed: %s", e)

    # Initialise the EKF functions
    log.info("Initialising EKF using calibration dir: %s", args.calib_dir)
    ekf = init_ekf(args.calib_dir, args.ip)
    # Pre-seed EKF with known ArUco positions from the provided map (Level 4/minimal map)
    try:
        ekf.seed_from_map_file(args.map, initial_covariance=1e-10, only_aruco=True)
        log.info("[EKF] Seeded ArUco landmarks from map: %s", args.map)
    except Exception as e:
        log.warning("[EKF] Seeding from map failed: %s", e)
    # Create ArUco detector
    aruco_det = aruco.aruco_detector(ekf.robot, marker_length=0.07)
    log.info("ArUco detector initialised (marker_length=0.07)")

    # Interactive OG viewer (PyGame) always enabled. Closing window ends program.
    def _get_pose():
        # In dry-run mode (no_run or localhost), keep pose at origin; otherwise query EKF
        if args.no_run or args.ip == 'localhost':
            return [0.0, 0.0, 0.0]
        return get_robot_pose(ppi, aruco_det, ekf)
    
    if args.level == 2 or args.level == 3:
        next_waypoint = calc_waypoint(search_poses[0], _get_pose(), 0.1)
        print(next_waypoint)

    viewer = OGViewer(grid=grid,
                      planner=AStarPlanner(),
                      get_pose_fn=_get_pose,
                      window_scale=4,
                      fps=15,
                      controller_kind=args.controller,
                      dry_run=(args.no_run or args.ip == 'localhost'),
                      ARUCO_locations=aruco_true_pos_id,
                      ppi=ppi)
    # Start the viewer and run until closed
    log.info("Launching OGViewer GUI")
    viewer.run()
    log.info("OGViewer closed; exiting program")
    sys.exit(0)