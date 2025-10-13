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
import threading
from collections import defaultdict, deque

from typing import List, Tuple

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

from slam.aruco_detector import aruco_detector

# Import state machine
from runtime.world_model import WorldModel
from runtime.robot_commander import RobotCommander
from runtime.runner_finaldemo import RunnerFinal
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
    # For now just grabs the marker coords.

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
        num_arucos = len(gt_dict)
        aruco_true_pos = np.empty([num_arucos, 2])
        aruco_true_pos_id = np.empty([num_arucos, 3])
        log.info(f"Loading {num_arucos} arucos")

        rounding = 4

        # remove unique id of targets of the same type
        for idx, key in enumerate(gt_dict):
            if key.startswith('aruco'):
                x = np.round(gt_dict[key]['x'], rounding)
                y = np.round(gt_dict[key]['y'], rounding)
                aruco_id = int(str(key).split("_")[0].replace("aruco",""))
                # Write the idless version
                aruco_true_pos[idx][0] = x
                aruco_true_pos[idx][1] = y
                # Write the version with the id
                aruco_true_pos_id[idx][0] = x
                aruco_true_pos_id[idx][1] = y
                aruco_true_pos_id[idx][2] = int(aruco_id)
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

def get_robot_pose(penguin_pi: PenguinPi, aruco_detector: aruco_detector, ekf: EKF) -> Tuple[List[float], float]:
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
    ekf.predict(drive_meas)
    # Get any visible arucos and then update EKF
    lms, _ = aruco_detector.detect_marker_positions(img)
    ekf.update(lms)

    # --- Track time since last seen ArUco ---
    time_of_last_aruco = time.monotonic()

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

    return robot_pose, time_of_last_aruco


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
TARGET_TYPES = ['orange','lemon','pear','tomato','capsicum','potato','pumpkin','garlic']


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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_target_list(s):
    """Parse a comma-separated list of integers into a sorted list."""
    if not s:
        return None
    try:
        values = [int(x.strip()) for x in s.split(',') if x.strip()]
        return sorted(values)
    except ValueError:
        raise argparse.ArgumentTypeError("visit_targets must be a comma-separated list of integers")

def _parse_args():
    parser = argparse.ArgumentParser("Fruit searching")
    # For Level 3, default to the L3 map; only accept levels 3 or 4
    parser.add_argument("--map", type=str, default='map.txt', help='Path to map file (Default: map.txt)')
    parser.add_argument("--shopping_list", type=str, default='shopping_list.txt', help='Path to shopping list file')
    parser.add_argument("--calib_dir", type=str, default='calibration/param/', help='Directory containing calibration files')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--controller", type=str, default='ttg', choices=['ttg','ppc','rhp'], help='Controller type')
    parser.add_argument("--no_run", action='store_true', help='Only load map and grid; do not start autonomy')
    parser.add_argument("--log", type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='Logging level')
    parser.add_argument("--model", type=str, default='models/PiBotAiMk2_V5.pt', help='YOLO model path (optional)')
    parser.add_argument("--use_fusion", action='store_true', help='Fuse bbox-height with bottom-pixel ground-ray for fruit range')
    parser.add_argument("--update_targets", type=str2bool, nargs='?', const=True, default=True, help="Whether to update target positions from live detections")
    parser.add_argument("--obstacle_size_large", type=float, default=0.08, help='Size of un-updated obstacles (m)')
    parser.add_argument("--obstacle_size_target", type=float, default=None, help='Size of the target obstacles (m)')
    parser.add_argument("--obstacle_size_small", type=float, default=0.06, help='Size of updated obstacles (m)')
    parser.add_argument("--inflation_margin", type=float, default=0.05, help='Exclusion zone inflation margin (m)')
    parser.add_argument("--reached_thresh_m", type=float, default=0.25, help='Threshold to consider a target "reached" (m)')
    parser.add_argument("--max_approach_attempts", type=int, default=6, help='Maximum attempts to approach a target')
    parser.add_argument("--boundary_margin", type=float, default=0.01, help='Boundary margin for the arena edge (m)')
    parser.add_argument("--visit_targets",type=parse_target_list,default=None,help="DEBUG USE ONLY: Comma-separated list of target IDs to visit, e.g. '1,2,4,5'")
    # Pose smoothing (control/GUI)
    parser.add_argument("--pose_smoothing", action='store_true', help='Enable EMA + rate-limited smoothed pose for control/GUI')
    parser.add_argument("--pose_alpha_pos", type=float, default=0.2, help='EMA alpha for x/y')
    parser.add_argument("--pose_alpha_yaw", type=float, default=0.2, help='EMA alpha for yaw')
    parser.add_argument("--pose_rate_xy", type=float, default=0.05, help='Max per-cycle xy step (m)')
    parser.add_argument("--pose_rate_yaw", type=float, default=10.0, help='Max per-cycle yaw step (deg)')
    # ArUco EKF gating/adaptive noise
    parser.add_argument("--aruco_gate", type=float, default=11.34, help='Mahalanobis gating threshold (chi2) for 2D aruco updates')
    parser.add_argument("--aruco_kd", type=float, default=1.0, help='Adaptive R scale by range: (1 + kd*range^2)')
    parser.add_argument("--level", type=int, default=1, choices=[1,2], help='Demo level: 1 or 2 (default 1)')
    parser.add_argument("--interactive_gui", action='store_true', help='Allow GUI clicks to set manual goals (Runner executes)')
    parser.add_argument("--disable_scans", action='store_true', help='Disable scans')
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

import json

def _load_map_and_shopping(args):
    """Load map, shopping list, and target positions from lab_output/targets.txt.

    Returns:
        aruco_true_pos: np.ndarray of shape (10,2)
        aruco_true_pos_id: np.ndarray of shape (10,3)
        search_list: list of fruit names (from shopping list)
        target_positions: dict[int, dict] e.g. {0: {"class": "orange", "pos": (x, y)}, ...}
    """
    log.info("Loading map file: %s", args.map)
    aruco_true_pos, aruco_true_pos_id = read_true_map(args.map)

    log.info("Loading shopping list: %s", args.shopping_list)
    search_list = read_search_list(args.shopping_list)

    # --- Load fruit targets ---
    targets_path = "lab_output/targets.txt"
    log.info("Loading target positions from: %s", targets_path)
    with open(targets_path, "r") as f:
        targets_raw = json.load(f)

    # Build target_positions in order of appearance
    target_positions = {}
    for idx, (class_name, coords) in enumerate(targets_raw.items()):
        pos = (np.round(float(coords["x"]), 4), np.round(float(coords["y"]), 4))
        # If in level 1 keep the class, if in level 2 set to None
        if args.level == 1:
            target_positions[idx+1] = {"class": class_name, "pos": pos}
        else:
            target_positions[idx+1] = {"class": None, "pos": pos}

    log.info("Loaded %d target positions", len(target_positions))
    log.info("Loaded shopping list: %s", search_list)

    return aruco_true_pos, aruco_true_pos_id, search_list, target_positions



def _build_grid_from_aruco(args, aruco_true_pos: np.ndarray) -> GridMap:
    grid = GridMap(res=0.02, margin=0.0, robot_radius=0.09,
                   inflation_margin=args.inflation_margin, boundary_margin=args.boundary_margin,
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
    def _get_pose():
        if args.no_run or args.ip == 'localhost':
            pose, time_since_last_aruco = [0.0, 0.0, 0.0], None
        else:
            pose, time_since_last_aruco = get_robot_pose(penguinpiInstance, aruco_detector, ekfInstance)
        return pose, time_since_last_aruco
    return _get_pose


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
    aruco_true_pos, aruco_true_pos_id, shopping_list, known_targets = _load_map_and_shopping(args)

    # 3) Occupancy grid
    gridMapInstance = _build_grid_from_aruco(args, aruco_true_pos)

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
    _ = _load_target_fruits_dict("ECE4078_Project/Milestone3/M3_prac_shopping_list.txt")

    # 9) Runtime wiring: WorldModel, Runner (L3/L4), GUI
    from queue import Queue
    world = WorldModel()
    intents_q: Queue = Queue()
    commander = RobotCommander(penguinpiInstance)

    # Runner pose function is same EKF-based callback
    from pibot_actions import PiBotActions
    actions = PiBotActions(penguinpiInstance, calib_dir=args.calib_dir)
    
    # Define planner
    PLANNER_OPTS = {
        "clearance_weight": 1.5,
        "clearance_power": 3.0,
        "clearance_epsilon": 0.01,   # 0.02 * 0.5
        "min_prune_clearance": 0.06,
        "clearance_mode": "static_dynamic",
    }
    planner = AStarPlanner(**PLANNER_OPTS)
    drive_enabled = (args.ip != "localhost") and (not args.no_run)

    # Define runner
    runner = RunnerFinal(
        # External components
        commander=commander,
        ekf=ekfInstance,
        aruco_det=aruco_det,
        grid=gridMapInstance,
        planner=planner,
        world=world,
        get_pose_fn=_get_pose,
        intents_q=intents_q,  # unused for L3

        # Runtime controls
        controller_kind=args.controller,
        hz=10.0,
        drive_enabled=drive_enabled,

        # Capabilities
        actions=actions,
        detector=yoloDetectorInstance,
        fruit_ranger=fruitRangerInstance,

        # World data
        target_dims=target_dims_Dict,
        aruco_positions=aruco_true_pos,
        shopping_list=shopping_list,
        target_positions=known_targets,   # renamed from known_targets
        reached_thresh_m=float(args.reached_thresh_m),
        max_approach_attempts=args.max_approach_attempts,
        obstacle_size_target=float(args.obstacle_size_target) if args.obstacle_size_target is not None else float(args.obstacle_size_small),

        # Debug / overrides
        visit_targets=args.visit_targets,  # None or list of target IDs

        update_targets=bool(args.update_targets),

        obstacle_sizes={
            "undetected": float(args.obstacle_size_large),
            "detected": float(args.obstacle_size_small),
        },

        disable_scans=bool(args.disable_scans),
    )
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
        detector=yoloDetectorInstance,
        aruco_detector=aruco_det,
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
        threshold_m=args.reached_thresh_m,
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
