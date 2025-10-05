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

from pibot_actions import PiBotActions

# Module logger
log = logging.getLogger(__name__)

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

_configure_logging("DEBUG")

# Define constants
args = argparse.Namespace()
args.calib_dir = "calibration/param/"
args.model = "YOLO/model/V5.pt"
args.ip = "192.168.50.1"
args.port = 8080
args.no_run = False
args.pose_smoothing = True
args.pose_alpha_pos = 0.2
args.pose_alpha_yaw = 0.2
args.pose_rate_xy = 0.05
args.pose_rate_yaw = 10.0

# Define EKF
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

def init_perception(args, ekfInstance):
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

def init_penguinpi(args):
    log.info("Connecting to PenguinPi (ip=%s, port=%s)", args.ip, args.port)
    pibot = PenguinPi(args.ip, args.port)
    # Start encoder polling on physical robot to obtain measured wheel speeds
    if args.ip != 'localhost':
        try:
            pibot.start_encoder_monitor(rate_hz=10.0)
        except Exception as e:
            log.warning("Encoder polling disabled (%s)", e)
    return pibot

penguinpiInstance = init_penguinpi(args)
ekfInstance = init_ekf(args.calib_dir, args.ip)
aruco_det = aruco.aruco_detector(ekfInstance.robot, marker_length=0.07)
yoloDetectorInstance, fruitRangerInstance, target_dims_Dict = init_perception(args, ekfInstance)
actions = PiBotActions(penguinpiInstance, calib_dir=args.calib_dir)

def get_robot_pose(penguin_pi: PenguinPi, aruco_detector: aruco.aruco_detector, ekf: EKF):
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

    # Prefer measured wheel velocities from PenguinPi encoders
    try:
        l_vel, r_vel = penguin_pi.get_wheel_velocity(prefer_measured=True)
    except Exception:
        l_vel, r_vel = 0.0, 0.0
    drive_meas = measure.Drive(l_vel, r_vel, dt)
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
        return pose
    return _get_pose

get_pose = _make_pose_fn(args, penguinpiInstance, aruco_det, ekfInstance)

last_tick = actions.turn_to_heading(goal_heading_rad=0, get_pose_fn=get_pose, turning_tick=50)
penguinpiInstance.set_velocity([0, 0], turning_tick=last_tick, time=0)

actions.turn_to_heading(
    goal_heading_rad=np.deg2rad(0),
    get_pose_fn=get_pose,
    turning_tick=25
    )
penguinpiInstance.set_velocity([0, 0], tick=10, time=0)

