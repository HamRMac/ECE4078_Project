# This is the original M3 level 3 code for the most part. Might need to uncomment a through things.


# M3 - Autonomous fruit searching
# Level 1: Semi-auto wit    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 hereh waypoints

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import logging
import threading
from collections import defaultdict, deque

from YOLO.detector import Detector

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
from gui.pibot_gui import PiBotGUI
from planning.grid_map import GridMap

# Import state machine
from state_machine.state_machine import PiBotFruitSearchSM
from runtime.world_model import WorldModel
from runtime.robot_commander import RobotCommander
from runtime.runner import Runner
from runtime.intents import SetGoal, SwitchMode

# Module logger
log = logging.getLogger(__name__)

# --- Colored logging formatter (ANSI) ---
class _ColoredFormatter(logging.Formatter):
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[32m",          # green
        logging.INFO: "\033[0m",            # default
        logging.WARNING: "\033[33m",        # yellow/amber
        logging.ERROR: "\033[31m",          # red
        logging.CRITICAL: "\033[1;31m",     # bold red
    }

    def __init__(self, fmt: str, use_color: bool = True):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record):
        msg = super().format(record)
        if self.use_color:
            color = self.COLORS.get(record.levelno, self.RESET)
            return f"{color}{msg}{self.RESET}"
        return msg

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
            #else:
            #    fruit_list.append(key[:-2])
            #    if len(fruit_true_pos) == 0:
            #        fruit_true_pos = np.array([[x, y]])
            #    else:
            #        fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return aruco_true_pos, aruco_true_pos_id


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
        pose = get_robot_pose(penguinpiInstance, aruco_detector, ekfInstance)
        fwd_cmd, turn_cmd, fwd_tick, turn_tick, done = ctrl_mgr.compute(pose, waypoint)
        if done:
            arrived = True
            break
        penguinpiInstance.set_velocity([fwd_cmd, turn_cmd], tick=fwd_tick, turning_tick=turn_tick, time=0)

        time.sleep(dt_loop)

    # Stop safely
    penguinpiInstance.set_velocity([0, 0])
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

    # Prefer measured wheel velocities from encoders (fallback to commands)
    try:
        l_vel, r_vel = penguin_pi.get_wheel_velocity(prefer_measured=True)
    except Exception:
        l_vel, r_vel = 0.0, 0.0
    drive_meas = measure.Drive(l_vel, r_vel, dt)
    motion_type = ekf.predict(drive_meas)
    # Get any visible arucos and then update EKF
    lms, _ = aruco_detector.detect_marker_positions(img)
    ekf.update(lms, motion_type=motion_type)

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


###### Live target estimation helpers and thread ######
TARGET_TYPES = ['orange','lemon','lime','tomato','capsicum','potato','pumpkin','garlic']


def estimate_pose(camera_matrix, obj_info, robot_pose, use_fusion=False, cam_height_m=0.20, cam_pitch_rad=0.0):
    """Estimate world (x,y) for a single detected object bbox.

    Parameters
    - camera_matrix: 3x3 intrinsics
    - obj_info: [label, [x,y,w,h]] or (label, bbox)
    - robot_pose: [x,y,theta]
    - use_fusion: whether to fuse bbox-depth with ground-ray
    - cam_height_m, cam_pitch_rad: camera mounting

    Returns: dict {'x': float, 'y': float} or None on failure
    """
    try:
        label, bbox = obj_info[0], obj_info[1]
    except Exception:
        return None

    # Ensure bbox is in (x,y,w,h) top-left format
    x, y, w, h = [float(v) for v in bbox]

    # Convert to center pixel coordinates
    cx_pix = x + w / 2.0
    cy_pix = y + h / 2.0

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    # Size lookup: use first entry as nominal height (m)
    target_dimensions_dict = {
        'orange':[0.07,0.07,0.073],'lemon':[0.078,0.053,0.050],'pear':[0.076,0.074,0.110],
        'tomato':[0.065,0.065,0.060],'capsicum':[0.076,0.074,0.090],'potato':[0.095,0.065,0.070],
        'pumpkin':[0.080,0.080,0.080],'garlic':[0.065,0.060,0.070],
    }
    true_sizes = target_dimensions_dict.get(label, target_dimensions_dict['tomato'])
    true_height = float(true_sizes[0])

    # Depth from bbox height
    bbox_h_pix = max(1.0, float(h))
    Z_bbox = (fx * true_height) / bbox_h_pix



    Z_fused = Z_bbox
    Z_ground = None
    if use_fusion:
        # Estimate depth by intersecting bottom pixel with ground plane (approximate)
        y_bottom = y + h
        v = (y_bottom - cy) / fy
        # approximate formula for depth along optical axis to ground intersection
        denom = np.cos(cam_pitch_rad) * v + np.sin(cam_pitch_rad)
        if abs(denom) > 1e-6:
            Z_ground = (cam_height_m) / denom
            # clamp to reasonable range
            if Z_ground <= 0 or Z_ground > 10.0:
                Z_ground = None

    # Fuse with logistic weight if both available
    if use_fusion and Z_ground is not None:
        z0 = 0.6
        k = 8.0
        w_bbox = 1.0 / (1.0 + np.exp(k * (Z_bbox - z0)))
        w_bbox = float(np.clip(w_bbox, 0.05, 0.95))
        Z_fused = w_bbox * Z_bbox + (1.0 - w_bbox) * Z_ground

    # Back-project center pixel into camera coords
    Xc = (cx_pix - cx) * (Z_fused / fx)
    Yc = (cy_pix - cy) * (Z_fused / fy)
    Zc = float(Z_fused)

    # Map camera coords to robot planar coords (assume camera aligned): robot_x forward = Z, robot_y left = -X
    r_x = Zc
    r_y = -Xc

    rx, ry, rth = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    c, s = np.cos(rth), np.sin(rth)
    world_x = rx + c * r_x - s * r_y
    world_y = ry + s * r_x + c * r_y

    return {'x': float(world_x), 'y': float(world_y)}


def merge_estimations(target_pose_dict, dist_thresh=0.25, max_per_type=3):
    """Greedy clustering per-class. Input: {label: [(x,y), ...], ...}
    Returns dict mapping label_index -> {'x':..., 'y':...}
    """
    out = {}
    for label, pts in target_pose_dict.items():
        pts = [tuple(p) for p in pts]
        clusters = []  # each cluster is list of pts
        for p in pts:
            placed = False
            for cl in clusters:
                # dist to cluster centroid
                cx = sum(q[0] for q in cl) / len(cl)
                cy = sum(q[1] for q in cl) / len(cl)
                if ( (p[0]-cx)**2 + (p[1]-cy)**2 )**0.5 <= dist_thresh:
                    cl.append(p)
                    placed = True
                    break
            if not placed:
                clusters.append([p])

        # average clusters and cap
        clusters = sorted(clusters, key=lambda c: -len(c))[:max_per_type]
        for i, cl in enumerate(clusters):
            cx = sum(q[0] for q in cl) / len(cl)
            cy = sum(q[1] for q in cl) / len(cl)
            out[f"{label}_{i}"] = {'x': float(cx), 'y': float(cy)}

    return out


class LiveTargetEstimator(threading.Thread):
    def __init__(self, yolo_model_path, get_image_fn, get_pose_fn, camera_matrix, use_fusion=False, cam_height_m=0.20, cam_pitch_rad=0.0, fps=4):
        super().__init__(daemon=True)
        self._model_path = yolo_model_path
        self.get_image_fn = get_image_fn
        self.get_pose_fn = get_pose_fn
        self.camera_matrix = camera_matrix
        self.use_fusion = use_fusion
        self.cam_height_m = cam_height_m
        self.cam_pitch_rad = cam_pitch_rad
        self.fps = fps
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._per_class = defaultdict(lambda: deque(maxlen=12))
        self.latest = {}
        # initialize detector (best-effort)
        try:
            self._detector = Detector(self._model_path)
        except Exception:
            # fallback: assume static method
            self._detector = Detector

    def run(self):
        log.info("LiveTargetEstimator thread running (fusion=%s)", self.use_fusion)
        period = 1.0 / max(1.0, float(self.fps))
        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                img = self.get_image_fn()
                if img is None:
                    time.sleep(period)
                    continue

                # Detector may return (detections, annotated_img) or detections only
                try:
                    dets = self._detector.detect_single_image(img)
                except Exception:
                    # try as instance method
                    dets = self._detector.detect_single_image(img)

                # Handle various return shapes
                if isinstance(dets, tuple) or isinstance(dets, list) and len(dets) >= 2 and isinstance(dets[0], list):
                    # common patterns: (detections, img) or detections list
                    if isinstance(dets[0], list):
                        detections = dets[0]
                    else:
                        detections = dets
                else:
                    detections = dets

                pose = self.get_pose_fn()
                per_class_tmp = defaultdict(list)
                for det in detections:
                    try:
                        label = det[0]
                        bbox = det[1]
                    except Exception:
                        continue
                    if label not in TARGET_TYPES:
                        continue
                    est = estimate_pose(self.camera_matrix, (label, bbox), pose, use_fusion=self.use_fusion, cam_height_m=self.cam_height_m, cam_pitch_rad=self.cam_pitch_rad)
                    if est is None:
                        continue
                    per_class_tmp[label].append((est['x'], est['y']))

                # append to deques
                for lab, lst in per_class_tmp.items():
                    for p in lst:
                        self._per_class[lab].append(p)

                # merge and store latest
                agg_in = {lab: list(self._per_class[lab]) for lab in self._per_class}
                merged = merge_estimations(agg_in)
                with self._lock:
                    self.latest = merged

            except Exception:
                log.exception("LiveTargetEstimator loop error")

            # sleep remainder
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

        log.info("LiveTargetEstimator thread exiting")

    def get_latest(self):
        with self._lock:
            return dict(self.latest)

    def stop(self):
        self._stop_event.set()

# -------------------------------
# Structured entrypoint helpers
# -------------------------------

def _parse_args():
    parser = argparse.ArgumentParser("Fruit searching")
    # For Level 3, default to the L3 map; only accept levels 3 or 4
    parser.add_argument("--map", type=str, default='james_house_l3_map.txt', help='Path to map file (L3 default: james_house_l3_map.txt)')
    parser.add_argument("--shopping_list", type=str, default='M3_prac_shopping_list.txt', help='Path to shopping list file')
    parser.add_argument("--calib_dir", type=str, default='calibration/param/', help='Directory containing calibration files')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--controller", type=str, default='ttg', choices=['ttg','ppc','rhp'], help='Controller type')
    parser.add_argument("--no_run", action='store_true', help='Only load map and grid; do not start autonomy')
    parser.add_argument("--log", type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='Logging level')
    parser.add_argument("--model", type=str, default='models/PiBotAiMk2_V5.pt', help='YOLO model path (optional)')
    parser.add_argument("--use_fusion", action='store_true', help='Fuse bbox-height with bottom-pixel ground-ray for fruit range')
    # Pose smoothing (control/GUI)
    parser.add_argument("--pose_smoothing", action='store_true', help='Enable EMA + rate-limited smoothed pose for control/GUI')
    parser.add_argument("--pose_alpha_pos", type=float, default=0.2, help='EMA alpha for x/y')
    parser.add_argument("--pose_alpha_yaw", type=float, default=0.2, help='EMA alpha for yaw')
    parser.add_argument("--pose_rate_xy", type=float, default=0.05, help='Max per-cycle xy step (m)')
    parser.add_argument("--pose_rate_yaw", type=float, default=10.0, help='Max per-cycle yaw step (deg)')
    # ArUco EKF gating/adaptive noise
    parser.add_argument("--aruco_gate", type=float, default=11.34, help='Mahalanobis gating threshold (chi2) for 2D aruco updates')
    parser.add_argument("--aruco_kd", type=float, default=1.0, help='Adaptive R scale by range: (1 + kd*range^2)')
    parser.add_argument("--level", type=int, default=3, choices=[3,4], help='Logic level: 3 or 4 (default 3)')
    parser.add_argument("--interactive_gui", action='store_true', help='Allow GUI clicks to set manual goals (Runner executes)')
    return parser.parse_known_args()


def _configure_logging(level: str):
    # Decide whether to use colors (TTY and not disabled by NO_COLOR)
    use_color = hasattr(sys.stderr, 'isatty') and sys.stderr.isatty() and (os.environ.get('NO_COLOR') is None)

    root = logging.getLogger()
    # Clear existing handlers to avoid duplicate logs if reconfigured
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(getattr(logging, (level or 'INFO').upper(), logging.INFO))
    fmt = '[%(levelname)s] [%(threadName)s] %(name)s: %(message)s'
    handler = logging.StreamHandler()
    handler.setFormatter(_ColoredFormatter(fmt, use_color=use_color))
    root.addHandler(handler)


def _init_penguinpi(args):
    log.info("Connecting to PenguinPi (ip=%s, port=%s)", args.ip, args.port)
    pibot = PenguinPi(args.ip, args.port)
    if args.ip != 'localhost':
        try:
            pibot.start_encoder_monitor(rate_hz=10.0)
        except Exception as e:
            log.warning("Encoder polling disabled (%s)", e)
    return pibot


def _load_map_and_shopping(args):
    """Load map and shopping list.

    For Level 3: expects keys like 'lemon_0' and 'aruco1_0'. Fruit keys are normalised by stripping the trailing
    '_0' to match shopping list entries. Returns also a dict mapping fruit -> (x,y) for quick lookup.
    """
    log.info("Loading map file: %s", args.map)
    fruits_list, fruits_true_pos, aruco_true_pos, aruco_true_pos_id = read_true_map(args.map)
    search_list = read_search_list(args.shopping_list)

    # Build fruit -> (x, y) dict (names already stripped by read_true_map)
    known_targets = {}
    try:
        for i, name in enumerate(fruits_list):
            if i < len(fruits_true_pos):
                known_targets[str(name)] = (float(fruits_true_pos[i][0]), float(fruits_true_pos[i][1]))
    except Exception:
        pass

    search_poses = []
    try:
        search_poses = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    except Exception:
        log.info("Loaded shopping list (positions not available in minimal map).")

    return (fruits_list, fruits_true_pos, aruco_true_pos, aruco_true_pos_id, search_list, search_poses, known_targets)


def _build_grid_from_aruco(aruco_true_pos: np.ndarray) -> GridMap:
    grid = GridMap(res=0.02, margin=0.0, robot_radius=0.09,
                   inflation_margin=0.05, boundary_margin=0.01,
                   arena_bounds_wm=(-1.4, -1.4, 1.4, 1.4))
    grid.build_from_aruco(aruco_true_pos)
    log.info("[WM] Occupancy grid built: size=%s res=%.3f m", str(grid.size), grid.res)
    return grid


def _init_ekf_and_aruco(args):
    log.info("Initialising EKF using calibration dir: %s", args.calib_dir)
    ekfInstance = init_ekf(args.calib_dir, args.ip)
    try:
        ekfInstance.seed_from_map_file(args.map, initial_covariance=1e-4, only_aruco=True)
        log.info("[EKF] Seeded ArUco landmarks from map: %s", args.map)
    except Exception as e:
        log.warning("[EKF] Seeding from map failed: %s", e)
    # Apply robust update settings
    try:
        setattr(ekfInstance, 'aruco_gate', float(args.aruco_gate))
        setattr(ekfInstance, 'aruco_kd', float(args.aruco_kd))
    except Exception:
        pass
    aruco_det = aruco.aruco_detector(ekfInstance.robot, marker_length=0.07)
    log.info("ArUco detector initialised (marker_length=0.07)")
    return ekfInstance, aruco_det


def _init_perception(args, ekfInstance):
    yoloDetectorInstance = None
    fruitRangerInstance = None
    target_dims_Dict = None
    try:
        camK = ekfInstance.robot.camera_matrix
        from perception.fruit_ranger import FruitRanger
        from TargetPoseEst import TARGET_DIMENSIONS_DICT as TARGET_DIMS
        fruitRangerInstance = FruitRanger(camera_matrix=camK)
        target_dims_Dict = TARGET_DIMS
        if args.model:
            from YOLO.detector import Detector
            # Verify the model exists
            if not os.path.exists(args.model):
                log.error("YOLO model file not found: %s", args.model)
                yoloDetectorInstance = None
            else:
                yoloDetectorInstance = Detector(args.model, 384)
                log.info("YOLO model loaded: %s", args.model)
    except Exception as e:
        log.warning("Live detection initialisation issue: %s", e)
    return yoloDetectorInstance, fruitRangerInstance, target_dims_Dict


def _make_pose_fn(args, penguinpiInstance, aruco_detector, ekfInstance):
    # Optional smoothed pose wrapper (EMA + rate limits)
    class _SmoothedPose:
        def __init__(self, alpha_pos: float, alpha_yaw: float, max_xy: float, max_yaw_deg: float):
            self.alpha_p = float(max(0.0, min(1.0, alpha_pos)))
            self.alpha_y = float(max(0.0, min(1.0, alpha_yaw)))
            self.max_xy = float(max_xy)
            self.max_yaw = float(max_yaw_deg) * np.pi / 180.0
            self.prev = None

        @staticmethod
        def _wrap(theta):
            return (theta + np.pi) % (2*np.pi) - np.pi

        def step(self, raw):
            x, y, th = float(raw[0]), float(raw[1]), float(raw[2])
            if self.prev is None:
                self.prev = (x, y, th)
                return [x, y, th]
            px, py, pth = self.prev
            # EMA
            sx = self.alpha_p * x + (1 - self.alpha_p) * px
            sy = self.alpha_p * y + (1 - self.alpha_p) * py
            dth = self._wrap(th - pth)
            sth = self._wrap(pth + self.alpha_y * dth)
            # Rate limit
            dx = np.clip(sx - px, -self.max_xy, self.max_xy)
            dy = np.clip(sy - py, -self.max_xy, self.max_xy)
            dth_rl = np.clip(self._wrap(sth - pth), -self.max_yaw, self.max_yaw)
            out = (px + dx, py + dy, self._wrap(pth + dth_rl))
            self.prev = out
            return [out[0], out[1], out[2]]

    smoother = None
    if bool(args.pose_smoothing):
        smoother = _SmoothedPose(args.pose_alpha_pos, args.pose_alpha_yaw, args.pose_rate_xy, args.pose_rate_yaw)

    def _get_pose():
        if args.no_run or args.ip == 'localhost':
            pose = [0.0, 0.0, 0.0]
        else:
            pose = get_robot_pose(penguinpiInstance, aruco_detector, ekfInstance)
        if smoother is not None:
            try:
                return smoother.step(pose)
            except Exception:
                return pose
        return pose
    return _get_pose


def _init_state_machine():
    try:
        sm = PiBotFruitSearchSM()
        return sm
    except Exception:
        return None


def _load_target_fruits_dict(list_path: str):
    targets = {}
    try:
        with open(list_path, "r") as f:
            for idx, line in enumerate(f):
                targets[idx] = {"collected": False, "fruit": line.strip()}
    except Exception:
        pass
    return targets


def main():
    args, _ = _parse_args()
    _configure_logging(args.log)
    log.info("auto_fruit_search starting (ip=%s, port=%s, controller=%s, no_run=%s)", args.ip, args.port, args.controller, args.no_run)
    if args.no_run or args.ip == 'localhost':
        log.warning("Running in no_run / localhost mode! No robot will be controlled!")

    # 1) Robot I/O
    penguinpiInstance = _init_penguinpi(args)

    # 2) Map + shopping list (+ known targets for L3)
    fruits_list, fruits_true_pos, aruco_true_pos, aruco_true_pos_id, search_list, search_poses, known_targets = _load_map_and_shopping(args)

    # 3) Occupancy grid
    gridMapInstance = _build_grid_from_aruco(aruco_true_pos)

    # 4) EKF + ArUco
    ekfInstance, aruco_det = _init_ekf_and_aruco(args)

    # 5) Perception
    yoloDetectorInstance, fruitRangerInstance, target_dims_Dict = _init_perception(args, ekfInstance)

    # 6) Pose callback
    _get_pose = _make_pose_fn(args, penguinpiInstance, aruco_det, ekfInstance)

    # 7) (Optional) Waypoint demo (kept disabled as before)
    # if (args.level == 2 or args.level == 3) and search_poses:
    #     next_waypoint = calc_waypoint(search_poses[0], _get_pose(), 0.1)
    #     print(next_waypoint)

    # 8) State machine + targets (if used by GUI)
    stateMachineInstance = _init_state_machine()
    _ = _load_target_fruits_dict("ECE4078_Project/Milestone3/M3_prac_shopping_list.txt")

    # 9) Runtime wiring: WorldModel, Runner (L3/L4), GUI
    from queue import Queue
    world = WorldModel()
    intents_q: Queue = Queue()
    commander = RobotCommander(penguinpiInstance)

    # Runner pose function is same EKF-based callback
    from pibot_actions import PiBotActions
    actions = PiBotActions(penguinpiInstance, calib_dir=args.calib_dir)
    # Choose runner based on level
    if int(args.level) == 3:
        from runtime.runnerl3 import RunnerL3
        runner = RunnerL3(commander=commander,
                          ekf=ekfInstance,
                          aruco_det=aruco_det,
                          grid=gridMapInstance,
                          planner=AStarPlanner(clearance_weight=1.5,
                                                clearance_power=3.0,
                                                clearance_epsilon=0.02*0.5,
                                                min_prune_clearance=0.06,
                                                clearance_mode='static_dynamic'),
                          world=world,
                          get_pose_fn=_get_pose,
                          intents_q=intents_q,
                          controller_kind=args.controller,
                          hz=10.0,
                          drive_enabled=not (args.no_run or args.ip == 'localhost'),
                          # Level 3 SM is optional; RunnerL3 manages a simple flow
                          state_machine=None,
                          actions=actions,
                          detector=yoloDetectorInstance,
                          fruit_ranger=fruitRangerInstance,
                          target_dims=target_dims_Dict,
                          aruco_positions=aruco_true_pos,
                          shopping_list=search_list,
                          known_targets=known_targets)
    else:
        runner = Runner(commander=commander,
                        ekf=ekfInstance,
                        aruco_det=aruco_det,
                        grid=gridMapInstance,
                        planner=AStarPlanner(clearance_weight=0.6, clearance_power=2.0, clearance_epsilon=0.02*0.5, min_prune_clearance=0.10),
                        world=world,
                        get_pose_fn=_get_pose,
                        intents_q=intents_q,
                        controller_kind=args.controller,
                        hz=10.0,
                        drive_enabled=not (args.no_run or args.ip == 'localhost'),
                        state_machine=stateMachineInstance,
                        actions=actions,
                        detector=yoloDetectorInstance,
                        fruit_ranger=fruitRangerInstance,
                        target_dims=target_dims_Dict,
                        aruco_positions=aruco_true_pos,
                        shopping_list=search_list)
    runner.start()

    # Providers for GUI (display-only)
    def _plan_provider():
        return world.get_plan()

    def _detections_provider():
        return world.get_detections()

    def _status_provider():
        return world.get_status()

    # Sector overlay provider for GUI (via WorldModel)
    def _sector_provider():
        try:
            return world.get_sectors()
        except Exception:
            return None

    # Intent sink from GUI clicks (wrap to SetGoal)
    def _intent_sink(gx: float, gy: float):
        intents_q.put(SetGoal(gx, gy))

    def _mode_sink(mode: str):
        intents_q.put(SwitchMode(mode))

    guiInstance = PiBotGUI(
        grid=gridMapInstance,
        ppi=penguinpiInstance,
        planner=AStarPlanner(),
        state_machine=stateMachineInstance,
        detector=yoloDetectorInstance,
        fruit_ranger=fruitRangerInstance,
        controller_kind=args.controller,
        get_pose_fn=_get_pose,
        get_frame_fn=penguinpiInstance.get_image,
        window_scale=4,
        fps=15,
        dry_run=True,  # GUI never controls motors
        ARUCO_locations=aruco_true_pos_id,
        target_dims=target_dims_Dict,
        interactive=bool(args.interactive_gui),
        intent_sink=_intent_sink if args.interactive_gui else None,
        plan_provider=_plan_provider,
        status_provider=_status_provider,
        detections_provider=_detections_provider,
        sector_provider=_sector_provider,
        mode_sink=_mode_sink,
        targets_provider=world.get_targets_info,
    )

    log.info("Launching PiBotGUI (display-only)")
    try:
        guiInstance.run()
    finally:
        log.info("Shutting down runner and stopping robot")
        try:
            runner.stop()
            runner.join(timeout=2.0)
            commander.stop()
        except Exception:
            pass
    log.info("PiBotGUI closed; exiting program")


if __name__ == "__main__":
    sys.exit(main() or 0)
