import threading
import time
import logging
import math
from typing import Optional, Tuple, List, Dict, Literal
from collections import Counter

import numpy as np

from navigation.controller import ControllerManager
from planning.astar import AStarPlanner
from planning.grid_map import GridMap
from planning.visibility_helper import compute_safety_mask
from .world_model import WorldModel
from .robot_commander import RobotCommander
from pibot_actions import PiBotActions

from slam.ekf import EKF
from slam.aruco_detector import aruco_detector
from YOLO.detector import Detector

from perception.fruit_ranger import FruitRanger

log = logging.getLogger(__name__)


class RunnerFinal(threading.Thread):
    """Final Demo mission runner (focused logic).

    Assumptions:
    - Robot starts at the centre of the arena, facing +X
    - Arena is square, with known size
    - We have a known map of ArUco positions and the robot can see at least one at start
    - We have a known map of all possible fruit positions (unclassified)
    - We have a shopping list of fruit types to collect
    - Apart from the obstacles, the arena is otherwise free drivable space
    Behaviour:
    - Robot checks for closest unclassified fruit position
    - If robot is within line of sight of the fruit position it will turn and face it
    - The robot will use the detector to try and classify the fruit
    - If the detected fruit is close to the location of the unclassified fruit, it is marked as classified with the detected type
    - If the fruit is on the shopping list, the robot will plan and drive to a standoff point near the fruit (and report what fruit it is going to collect)
    - If the fruit is not on the shopping list, the robot will detect the next closest unclassified fruit and repeat
    - If the robot cannot see any unclassified fruit, it will navigate to the closest one
    - Once the robot has collected all fruit on the shopping list, it will report success and stop
    """

    def __init__(self,
                 commander: RobotCommander,
                 ekf: EKF,
                 aruco_det: aruco_detector,
                 grid: GridMap,
                 planner: Optional[AStarPlanner],
                 world: WorldModel,
                 get_pose_fn,
                 intents_q,  # unused for L3
                 controller_kind: str = "ttg",
                 hz: float = 10.0,
                 drive_enabled: bool = True,
                 state_machine=None,  # unused for L3
                 actions: PiBotActions = None,
                 detector=None,
                 fruit_ranger=None,
                 target_dims=None,
                 aruco_positions: np.ndarray = None,
                 shopping_list: Optional[List[str]] = None,
                 target_positions: Optional[Dict[int, Tuple[float, float]]] = None,
                 update_targets: Optional[bool] = True,
                 obstacle_sizes: Optional[Dict[str, float]] = None,
                 ):
        super().__init__(daemon=True, name="RunnerFinal")

        # External components
        self.cmd: RobotCommander = commander
        self.ekf: EKF = ekf
        self.aruco: aruco_detector = aruco_det
        self.grid: GridMap = grid
        self.planner: AStarPlanner = planner or AStarPlanner()
        self.world: WorldModel = world
        self.get_pose_fn = get_pose_fn
        self.ctrl: ControllerManager = ControllerManager(controller_kind)
        self._stop: threading.Event = threading.Event()
        self.actions: PiBotActions = actions
        self.detector: Optional[Detector] = detector
        self.fruit_ranger: Optional[FruitRanger] = fruit_ranger
        self.target_dims: Optional[Dict[str, Tuple[float, float, float]]] = target_dims

        # Settings
        self._drive_enabled = bool(drive_enabled)

        # Provided data
        self.aruco_positions = aruco_positions
        # This target_positions is in the order by which they will be reached
        self.target_positions: Dict[int, Dict] = target_positions or {}  # {1: {class:"",pos:(x,y)}}
        self.shopping_list: List[str] = list(shopping_list or [])
        self.update_targets = bool(update_targets)
        self.obstacle_sizes = obstacle_sizes if isinstance(obstacle_sizes, dict) else {
            "undetected": 0.10,  # larger
            "detected": 0.05,  # smaller
        }

        # Generated data
        self.all_obstacles_world_dict = {}
        self.all_targets_world_dict = {}

        # Planning state
        self._goal: Optional[Tuple[float, float]] = None
        self._plan_waypoints: List[Tuple[float, float]] = []
        self._wp_idx: int = 0
        self._period = 1.0 / max(1.0, float(hz))
        self._xtrack_thresh: float = 0.05
        self._just_replanned: bool = False

        self._target_mode: Literal['KNOWN_TARGETS','CHECK_ALL'] = 'KNOWN_TARGETS'
        
        '''
        # Ordered route derived from shopping list
        self._route = ()
        
        for name in self.shopping_list:
            if name in self.known_targets:
                self._route.append((name, self.known_targets[name]))
        '''

    # ---------------- Small helpers ----------------
    def stop(self):
        self._stop.set()
        self.cmd.stop()

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx, dy = (a[0] - b[0]), (a[1] - b[1])
        return float((dx * dx + dy * dy) ** 0.5)

    def _apply_static_target_exclusions(self, positions: List[Tuple[float, float]], fruit_radius_m: float = 0.01):
        import cv2
        if self.grid.static_layer is None or self.grid.size is None:
            return
        # For L3 request: use a small fixed buffer around targets (~1 cm)
        inflate_r = float(fruit_radius_m)
        rc = max(1, int(math.ceil(inflate_r / float(self.grid.res))))
        for (x, y) in positions or []:
            r, c = self.grid.world_to_grid(float(x), float(y))
            cv2.circle(self.grid.static_layer, (c, r), rc, color=255, thickness=-1)
        self.grid._clearance_cache = None  # invalidate cache

    def _plan_from_current(self) -> bool:
        pose = self.get_pose_fn()
        if self._goal is None:
            return False
        pr = self.planner.plan(self.grid, (pose[0], pose[1]), (self._goal[0], self._goal[1]))
        if pr is None:
            return False
        self._plan_waypoints = list(pr.pruned_world if pr.pruned_world else pr.path_world)
        self._wp_idx = 0
        self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
        self.world.set_status(action='drive', progress=f"0/{len(self._plan_waypoints)}")
        self._just_replanned = True
        return True

    def _maybe_replan(self, pose):
        if not self._plan_waypoints or self._goal is None:
            return
        # If robot's current cell is occupied, keep following the existing plan instead of replanning.
        # This prevents a replan loop where the planner repeatedly shifts the start to the nearest free cell.
        try:
            occ = self.grid.combined()
            r, c = self.grid.world_to_grid(float(pose[0]), float(pose[1]))
            if int(occ[r, c]) != 0:
                # Inside an exclusion/occupied zone: do not trigger replans based on cross-track error.
                return
        except Exception:
            pass
        try:
            xtrack = AStarPlanner.cross_track_error((pose[0], pose[1]), self._plan_waypoints)
            if xtrack > self._xtrack_thresh:
                self._plan_from_current()
        except Exception:
            pass

    def _drive_step(self, pose) -> bool:
        if not self._plan_waypoints:
            return False
        self._wp_idx = min(self._wp_idx, len(self._plan_waypoints) - 1)
        wp = self._plan_waypoints[self._wp_idx]
        fwd_cmd, turn_cmd, fwd_tick, turn_tick, done = self.ctrl.compute(pose, wp)
        if self._drive_enabled:
            self.cmd.set_velocity([fwd_cmd, turn_cmd], tick=fwd_tick, turning_tick=turn_tick, time=0)
        if done:
            if self._wp_idx < len(self._plan_waypoints) - 1:
                if self._just_replanned:
                    self._just_replanned = False
                self._wp_idx += 1
                self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
            else:
                self.cmd.stop()
                self.world.set_status(action='arrived')
                self._plan_waypoints = []
                self.world.clear_plan()
                return True
        return False

    def _plan_approach_to_target(self, target_xy: Tuple[float, float], radius_m: float = 0.20) -> bool:
        """Plan to a standoff point on the circle of radius 'radius_m' around the target.
        For this radius, evaluate all candidate approach points and pick the shortest successful path.
        """
        pose = self.get_pose_fn()
        rx, ry = float(pose[0]), float(pose[1])
        tx, ty = float(target_xy[0]), float(target_xy[1])
        base_th = math.atan2(ty - ry, tx - rx)
        Ks = 24
        angles = [((base_th + 2 * math.pi * i / Ks + math.pi) % (2 * math.pi)) - math.pi for i in range(Ks)]
        # Evaluate all candidates at this radius; keep best by A* grid cost
        occ = self.grid.combined()
        best = None  # (cost, PlanResult, (wx, wy))
        for ang in angles:
            wx = tx + radius_m * math.cos(ang)
            wy = ty + radius_m * math.sin(ang)
            r, c = self.grid.world_to_grid(wx, wy)
            if int(occ[r, c]) != 0:
                continue
            pr = self.planner.plan(self.grid, (rx, ry), (wx, wy))
            if pr is None:
                continue
            cost = float(pr.cost)
            if best is None or cost < best[0]:
                best = (cost, pr, (wx, wy))
        if best is None:
            return False
        # Install the best plan directly
        _, pr_best, goal_xy = best
        self._goal = (float(goal_xy[0]), float(goal_xy[1]))
        self._plan_waypoints = list(pr_best.pruned_world if pr_best.pruned_world else pr_best.path_world)
        self._wp_idx = 0
        self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
        self.world.set_status(action='drive', progress=f"0/{len(self._plan_waypoints)}")
        self._just_replanned = True
        return True

    def _find_nearest_free_xy_around(self, target_xy: Tuple[float, float], max_radius_cells: int = 60) -> Optional[Tuple[float, float]]:
        """Find nearest free grid cell around the given world (x,y) and return its world coords."""
        try:
            occ = self.grid.combined()
            H, W = occ.shape
        except Exception:
            return None
        tr, tc = self.grid.world_to_grid(float(target_xy[0]), float(target_xy[1]))
        best = None
        best_d2 = 1e12
        for rad in range(1, int(max_radius_cells) + 1):
            r0 = max(0, tr - rad); r1 = min(H - 1, tr + rad)
            c0 = max(0, tc - rad); c1 = min(W - 1, tc + rad)
            found = False
            for r in range(r0, r1 + 1):
                for c in (c0, c1):
                    if int(occ[r, c]) == 0:
                        d2 = (r - tr) * (r - tr) + (c - tc) * (c - tc)
                        if d2 < best_d2:
                            best_d2 = d2; best = (r, c); found = True
            for c in range(c0, c1 + 1):
                for r in (r0, r1):
                    if int(occ[r, c]) == 0:
                        d2 = (r - tr) * (r - tr) + (c - tc) * (c - tc)
                        if d2 < best_d2:
                            best_d2 = d2; best = (r, c); found = True
            if found and best is not None:
                wx, wy = self.grid.grid_to_world(best[0], best[1])
                return (float(wx), float(wy))
        return None

    def _plan_best_approach_to_target(self, target_xy: Tuple[float, float]) -> bool:
        """Ensure we generate a path to get as close as possible to the target.
        Strategy:
        - Try multiple standoff radii (0.20 → 0.50m) using _plan_approach_to_target.
        - If none succeed, find nearest free cell around target and plan to it.
        - If still none, sample farther radii (0.55 → 1.0m) to at least move closer.
        Returns True if a plan is installed.
        """
        # Try expanding radii close to the target
        for r in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            if self._plan_approach_to_target(target_xy, radius_m=r):
                return True
        # Try nearest free around target
        nf = self._find_nearest_free_xy_around(target_xy, max_radius_cells=80)
        if nf is not None:
            self._goal = nf
            if self._plan_from_current():
                return True
        # Try farther standoff rings
        for r in [0.55, 0.65, 0.80, 1.0, 1.2, 1.5]:
            if self._plan_approach_to_target(target_xy, radius_m=r):
                return True
        return False

    # ----- Beeline helpers (static + dynamic LOS, ignore dark) -----
    @staticmethod
    def _supercover_line(p0: Tuple[int, int], p1: Tuple[int, int]):
        r0, c0 = p0; r1, c1 = p1
        dr = r1 - r0; dc = c1 - c0
        sr = 1 if dr > 0 else -1 if dr < 0 else 0
        sc = 1 if dc > 0 else -1 if dc < 0 else 0
        dr = abs(dr); dc = abs(dc)
        r, c = r0, c0
        cells = [(r, c)]
        if dr == 0 and dc == 0:
            return cells
        if dc >= dr:
            err = dc // 2
            for _ in range(dc):
                c += sc
                err += dr
                if err >= dc:
                    err -= dc
                    r += sr
                    cells.append((r, c - sc))
                cells.append((r, c))
        else:
            err = dr // 2
            for _ in range(dr):
                r += sr
                err += dc
                if err >= dr:
                    err -= dr
                    c += sc
                    cells.append((r - sr, c))
                cells.append((r, c))
        return cells

    def _segment_free_static_dynamic(self, a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> bool:
        try:
            occ_sd = np.maximum(self.grid.static_layer, self.grid.dynamic_layer)
            H, W = occ_sd.shape
            a_rc = self.grid.world_to_grid(float(a_xy[0]), float(a_xy[1]))
            b_rc = self.grid.world_to_grid(float(b_xy[0]), float(b_xy[1]))
            for r, c in self._supercover_line(a_rc, b_rc):
                if r < 0 or r >= H or c < 0 or c >= W:
                    return False
                if int(occ_sd[r, c]) != 0:
                    return False
            return True
        except Exception:
            return False

    def _segment_avoids_dark(self, a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> bool:
        """True only if safety_layer is marked safe (0) along segment."""
        try:
            safe = self.grid.safety_layer
            if safe is None:
                return False
            H, W = safe.shape
            a_rc = self.grid.world_to_grid(float(a_xy[0]), float(a_xy[1]))
            b_rc = self.grid.world_to_grid(float(b_xy[0]), float(b_xy[1]))
            for r, c in self._supercover_line(a_rc, b_rc):
                if r < 0 or r >= H or c < 0 or c >= W:
                    return False
                if int(safe[r, c]) != 0:
                    return False
            return True
        except Exception:
            return False

    def _beeline_permitted(self, pose_xy: Tuple[float, float], target_xy: Tuple[float, float], stop_radius_m: float) -> Tuple[bool, Tuple[float, float]]:
        """Permit beeline only if within 0.5m immediately after a scan and path avoids darkness.

        Returns (ok, goal_xy) where goal is shortened by stop_radius.
        """
        rx, ry = float(pose_xy[0]), float(pose_xy[1])
        tx, ty = float(target_xy[0]), float(target_xy[1])
        dx, dy = (tx - rx), (ty - ry)
        dist = math.hypot(dx, dy)
        if dist > 0.50:
            return (False, (tx, ty))
        if (time.time() - float(self._last_scan_time)) > 3.0:
            return (False, (tx, ty))
        step = max(0.0, dist - stop_radius_m)
        if step <= 1e-3:
            gx, gy = tx, ty
        else:
            ux, uy = dx / max(1e-9, dist), dy / max(1e-9, dist)
            gx, gy = rx + ux * step, ry + uy * step
        if not self._segment_avoids_dark((rx, ry), (gx, gy)):
            return (False, (gx, gy))
        if not self._segment_free_static_dynamic((rx, ry), (gx, gy)):
            return (False, (gx, gy))
        return (True, (gx, gy))

    def _attempt_beeline_pose(self, target_name: str, target_xy: Tuple[float, float], stop_radius_m: float = 0.25) -> bool:
        """Direct LOS crawl using pose, gated by recent scan and safe path."""
        pose = self.get_pose_fn()
        rx, ry = float(pose[0]), float(pose[1])
        ok, goal = self._beeline_permitted((rx, ry), target_xy, stop_radius_m)
        if not ok:
            return False
        gx, gy = goal
        tx, ty = float(target_xy[0]), float(target_xy[1])
        # Install line for GUI (red) and crawl
        self._goal = (gx, gy)
        self._plan_waypoints = [(rx, ry), (gx, gy)]
        self._wp_idx = 0
        self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx, color='red')
        self.world.set_status(mode='AUTO', sm_state='FinalDemo', action='beeline_pose', target=target_name)
        t0 = time.time()
        timeout = 8.0  # safety cap for straight drive
        while not self._stop.is_set() and self._plan_waypoints:
            pose = self.get_pose_fn()
            self.world.set_pose(pose)
            self._drive_step(pose)  # do not call _maybe_replan during beeline
            # proximity check
            if self._dist((pose[0], pose[1]), (tx, ty)) <= stop_radius_m:
                return True
            if (time.time() - t0) > timeout:
                break
            time.sleep(self._period)
        return self._dist(self.get_pose_fn()[:2], (tx, ty)) <= stop_radius_m

    def _attempt_beeline_yolo(self, target_name: str, target_xy: Tuple[float, float], stop_radius_m: float = 0.25) -> bool:
        """Use detector to center the target in view and slow-crawl forward until in range.

        Fallback: returns False if detector or required metadata unavailable, or if no
        matching detection is found within a short time window.
        """
        if self.detector is None or self.fruit_ranger is None or not isinstance(self.target_dims, dict):
            return False
        dims = self.target_dims.get(str(target_name))
        if not isinstance(dims, (list, tuple)) or len(dims) < 3:
            return False
        true_h = float(dims[2])
        # Control params for slow crawl
        turn_tick_base = 12
        fwd_tick_crawl = 18
        dt = 0.10
        timeout = 10.0
        not_found_patience = 12  # ~1.2s
        missing = 0
        t0 = time.time()
        cx_img = 160.0
        f = 320.0
        try:
            K = getattr(self.fruit_ranger, 'camera_matrix', None)
            if K is not None:
                f = float(K[0, 0])
                cx_img = float(K[0, 2])
        except Exception:
            pass

        # Gate beeline by scan/clearance first
        try:
            pose0 = self.get_pose_fn()
            ok_gate, goal = self._beeline_permitted((float(pose0[0]), float(pose0[1])), target_xy, stop_radius_m)
        except Exception:
            ok_gate, goal = (False, target_xy)
        if not ok_gate:
            return False
        self.world.set_status(mode='AUTO', sm_state='FinalDemo', action='beeline_yolo', target=target_name)

        while not self._stop.is_set():
            # Acquire frame
            frame = None
            try:
                frame = self.actions.ppi.get_image() if (self.actions is not None) else None
            except Exception:
                frame = None
            if frame is None:
                return False
            # Ensure BGR for detector
            try:
                import cv2
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception:
                frame_bgr = frame

            # Detect and filter for target class
            try:
                det_out = self.detector.detect_single_image(frame_bgr)
                detections = det_out[0] if isinstance(det_out, (list, tuple)) else det_out
            except Exception:
                detections = []
            # pick best by largest area
            cand = None
            best_area = -1.0
            for d in detections or []:
                try:
                    lab = str(d[0])
                    if lab != str(target_name):
                        continue
                    x, y, w, h = [float(v) for v in d[1]]
                    area = w * h
                    if area > best_area:
                        best_area = area
                        cand = (x, y, w, h)
                except Exception:
                    continue

            if cand is None:
                missing += 1
                if missing > not_found_patience or (time.time() - t0) > timeout:
                    return False
                # small search turn to try reacquire
                if self._drive_enabled:
                    self.cmd.set_velocity([0, 1], turning_tick=10, time=0)
                time.sleep(dt)
                continue

            missing = 0
            # Centering control (proportional on pixel error)
            x, y, w, h = cand
            cx = x + w / 2.0
            err_px = float(cx_img - cx)
            # approx angle ~ atan(err/f) ≈ err/f for small
            ang = err_px / max(1.0, f)
            turn_dir = 0
            turn_tick = 0
            if abs(ang) > 0.01:
                turn_dir = 1 if ang > 0 else -1
                turn_tick = int(min(18, max(8, abs(ang) * 200)))

            # Range estimate from bbox height
            est = None
            try:
                est = self.fruit_ranger.from_bbox_height([x, y, w, h], true_h)
            except Exception:
                est = None
            in_range = False
            if est is not None:
                in_range = float(est.get('r', 1e9)) <= (stop_radius_m + 0.02)
            else:
                # fallback: EKF distance
                pose = self.get_pose_fn()
                in_range = self._dist((pose[0], pose[1]), target_xy) <= stop_radius_m

            if in_range:
                if self._drive_enabled:
                    self.cmd.stop()
                return True

            # Issue slow-crawl command
            if self._drive_enabled:
                self.cmd.set_velocity([1, turn_dir], tick=fwd_tick_crawl, turning_tick=turn_tick if turn_tick>0 else 10, time=0)

            if (time.time() - t0) > timeout:
                break
            time.sleep(dt)

        if self._drive_enabled:
            self.cmd.stop()
        return False

    def _scan_and_update(self):
        # Tunables
        CLOSE_DET_RADIUS_M = 0.8   # "close to the robot" (consistent with earlier usage)
        ASSIGN_THRESH_M    = 0.30  # 30 cm

        # 1) Execute scan (drive-based) if available
        if self.actions is not None and self._drive_enabled:
            try:
                self.actions.scan(step_angle_deg=30.0,
                                detector=self.detector,
                                fruit_ranger=self.fruit_ranger,
                                target_dims=self.target_dims,
                                get_pose_fn=self.get_pose_fn,
                                turning_tick=40,
                                pause_s=1.0)
            except Exception as e:
                log.warning("Scan failed: %s", e)

        # Record scan time (used elsewhere)
        self._last_scan_time = time.time()

        # 2) Gather detections (position + class) and keep only those close to the robot
        try:
            raw_dets = getattr(self.actions, 'current_obj_positions', []) or []
        except Exception:
            raw_dets = []

        rx, ry, _ = self.get_pose_fn()

        close_dets = []
        for d in raw_dets:
            try:
                pos = d.get('position')
                if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                    continue
                wx, wy = float(pos[0]), float(pos[1])

                # Keep only detections close to the robot
                if self._dist((rx, ry), (wx, wy)) > CLOSE_DET_RADIUS_M:
                    continue

                # Normalise label key
                lab = d.get('class') if ('class' in d) else d.get('label')
                if lab is None or lab == '':
                    continue
                lab = str(lab)

                close_dets.append({"class": lab, "position": (wx, wy)})
            except Exception:
                continue

        # 3) For each detection, identify the closest object (targets + obstacles) and update its entry
        if (self.all_targets_world_dict or self.all_obstacles_world_dict) and close_dets and self.update_targets:
            def _iter_entries():
                # tag which dict each entry comes from so we can update the right one
                for k, v in self.all_targets_world_dict.items():
                    yield ("targets", k, v)
                for k, v in self.all_obstacles_world_dict.items():
                    yield ("obstacles", k, v)

            for det in close_dets:
                dpos = det["position"]  # (wx, wy)
                try:
                    which, key, entry = min(
                        _iter_entries(),
                        key=lambda t: self._dist((float(t[2]["x"]), float(t[2]["y"])), dpos)
                    )
                except ValueError:
                    # both dicts empty
                    break

                ex, ey = float(entry["x"]), float(entry["y"])
                if self._dist((ex, ey), dpos) <= ASSIGN_THRESH_M:
                    disp = str(det["class"])
                    if which == "targets":
                        if self.all_targets_world_dict[key]["disp_name"] is None:
                            self.all_targets_world_dict[key]["disp_name"] = disp
                        self.all_targets_world_dict[key]["x"] = float(dpos[0])
                        self.all_targets_world_dict[key]["y"] = float(dpos[1])
                        self.all_targets_world_dict[key]["updated_by_scan"] = True
                    else:  # obstacles
                        self.all_obstacles_world_dict[key]["disp_name"] = disp
                        self.all_obstacles_world_dict[key]["x"] = float(dpos[0])
                        self.all_obstacles_world_dict[key]["y"] = float(dpos[1])
                        self.all_obstacles_world_dict[key]["updated_by_scan"] = True

        # 4) (Optional) publish detections to the world model for GUI
        try:
            self.world.set_detections(close_dets)
        except Exception:
            pass

        # 5) Rebuild ONLY the dynamic layer (no safety layer updates)
        try:
            self.update_dynamic_layer_with_targets()
        except Exception as e:
            log.debug("Dynamic layer update skipped: %s", e)

    def _shopping_list_complete(self) -> bool:
        # Build required counts from shopping_list
        need = Counter(self.shopping_list or [])
        if not need:
            return False  # no shopping list to satisfy
        # What we have classified so far
        have = Counter()
        for info in self.target_positions.values():
            c = info.get("class")
            if c:
                have[c.split("_")[0]] += 1
        # Complete when for every wanted class we have at least that many classified
        return all(have[k] >= v for k, v in need.items())

    def _all_classified(self) -> bool:
        return all(info.get("class") is not None for info in self.target_positions.values())

    def _all_collected_in_order(self, target_order: list[int], collected_ids: set[str]) -> bool:
        return all(str(tid) in collected_ids for tid in target_order)

    def _next_known_targets(self, target_order: list[int], collected_ids: set[str]) -> int | None:
        # First id in order not yet collected and with a known position
        for tid in self.target_order:
            if str(tid) not in collected_ids:
                info = self.target_positions.get(tid, {})
                if info.get("pos") is not None:
                    return tid
        return None

    # ---------------- Main loop ----------------
    def run(self):
        log.info("FinalDemo Runner starting!")

        # Allow the robot to go anywhere (no dark zones)
        self.grid.mark_all_safe()

        # Set Target Order
        # First see if we have classes assigned
        all_targets = [key for key in self.target_positions.keys()]
        target_order = []
        for want in self.shopping_list:
            for tid, info in self.target_positions.items():
                if info.get("class").split("_")[0] == want and tid in all_targets:
                    target_order.append(tid)
                    break
        # Check if we added any targets. If not, just add all targets in order
        if not target_order:
            target_order = all_targets.copy()
            target_order.sort()
            self._target_mode = 'CHECK_ALL'

        print(target_order)
        total_targets = len(target_order)

        # Collate all the obstacles into two dictionaries
        # self.all_targets_world_dict contains all the targets
        self.all_targets_world_dict = {
            tid: {
                "disp_name": info.get("class").split("_")[0] if info.get("class") is not None else None,
                "x": float(info["pos"][0]),
                "y": float(info["pos"][1]),
                "updated_by_scan": False if self.update_targets else True
            }
            for tid, info in self.target_positions.items()
            if tid in target_order
        }
        # self.all_obstacles_world_dict contains any fruit that isn't a target
        self.all_obstacles_world_dict = {
            tid: {
                "disp_name": info.get("class").split("_")[0] if info.get("class") is not None else None,
                "x": float(info["pos"][0]),
                "y": float(info["pos"][1]),
                "updated_by_scan": False if self.update_targets else True
            }
            for tid, info in self.target_positions.items()
            if tid not in target_order
        }

        # Create a list of collected targets
        collected_targets = []

        # Append known targets to dynamic exclusion map
        try:
            self.update_dynamic_layer_with_targets()
        except Exception:
            pass

        # Publish targets info to WorldModel for GUI overlay
        try:
            # Extract Positions
            positions = {
                str(k): (float(info["pos"][0]), float(info["pos"][1]))
                for k, info in self.target_positions.items()
            }
            remaining = positions.copy()

            # Set the targets info in the world model
            self.world.set_targets_info(
                targets = self.all_targets_world_dict,
                active = -1,
                collected = collected_targets
            )
            # Initial status
            self.world.set_status(mode='AUTO', sm_state='FinalDemo', action='init', progress=f"0/{total_targets}")
        except Exception:
            pass
        
        # -- Main loop
        # if self._target_mode == 'KNOWN_TARGETS' --> go in order of target_order until all collected
        # if self._target_mode == 'CHECK_ALL' --> go in order of target_order until all classified OR shopping list complete

        # For each target in target_order
        for target_index in target_order:
            current_target_index = -1
            # If we are in CHECK_ALL
            if self._target_mode == 'CHECK_ALL':
                # 1) Choose closest unclassified target
                wx, wy, _ = self.get_pose_fn()

                unclassified = [
                    (tid, info) for tid, info in self.target_positions.items()
                    if info.get("class") is None and info.get("pos") is not None
                ]
                if not unclassified:
                    log.warning("No unclassified targets with known positions; stopping.")
                    break

                def sqdist(p):
                    dx, dy = p[0] - wx, p[1] - wy
                    return dx*dx + dy*dy

                # tie-break by tid for determinism
                idx, info = min(unclassified, key=lambda kv: (sqdist(kv[1]["pos"]), kv[0]))
                txy = (float(info["pos"][0]), float(info["pos"][1]))
                name = str(idx)
                current_target_index = idx

                # Create progress string
                progress_str = f"{len(collected_targets) + 1}/{total_targets}"

                if self._stop.is_set():
                    break

                # 2) Mark this as the active target in the world model
                try:
                    wm = self.world.get_targets_info()
                    self.world.set_targets_info(
                        targets=self.all_targets_world_dict,
                        active=idx,
                        collected=collected_targets
                    )
                    self.world.set_status(mode='AUTO', sm_state='FinalDemo', action='scan',
                                        target=name, progress=progress_str)
                except Exception:
                    pass

                log.info("🎯 Heading to closest unclassified target id=%d: %s at (%.2f, %.2f) [%s]",
                        idx, name, txy[0], txy[1], progress_str)
            
            # Otherwise we know the target order so we know where we need to go to next
            if self._target_mode == 'KNOWN_TARGETS':
                # Get the location and class of the current target
                current_target = self.all_targets_world_dict[target_index]
                txy = (current_target.get("x"), current_target.get("y"))
                name = current_target.get("disp_name") if current_target.get("disp_name") is not None else str(target_index)
                current_target_index = target_index

                progress_str = f"{len(collected_targets) + 1}/{total_targets}"

                # Set gui status
                try:
                    wm = self.world.get_targets_info()
                    self.world.set_targets_info(
                        targets=self.all_targets_world_dict,
                        active=target_index,
                        collected=collected_targets
                    )
                    self.world.set_status(mode='AUTO', sm_state='FinalDemo', action='scan',
                                        target=name, progress=progress_str)
                except Exception:
                    pass

                # Set status
                log.info("🎯 Heading to known target id=%d: %s at (%.2f, %.2f) [%s]",
                        target_index, name, txy[0], txy[1], progress_str)

            attempt = 0
            while not self._stop.is_set():
                # 1) Scan
                log.info("Starting scan before approaching %s (attempt %d)", name, attempt + 1)
                self.world.set_status(mode='AUTO', sm_state='FinalDemo', action='scan', target=name,
                                       progress=progress_str)
                self._scan_and_update()

                # Refresh target coords in case the scan updated them
                current_target = self.all_targets_world_dict[current_target_index]
                new_txy = (current_target.get("x"), current_target.get("y"))
                print(current_target)
                if self._dist(txy, new_txy) > 0.03:  # 3 cm hysteresis to avoid thrashing
                    log.info("Target %s moved from (%.2f, %.2f) to (%.2f, %.2f); replanning",
                            str(current_target_index), txy[0], txy[1], new_txy[0], new_txy[1])
                    txy = new_txy
                    self._goal = txy

                pose = self.get_pose_fn()
                self.world.set_pose(pose)
                dist = self._dist((pose[0], pose[1]), txy)
                if dist <= 0.25:
                    print(f"🎯 <-- Reached {name}")
                    time.sleep(2.0)
                    # Update targets info: mark as collected
                    try:
                        wm = self.world.get_targets_info()
                        collected_targets.append(current_target_index)
                        self.world.set_targets_info(
                            targets=self.all_targets_world_dict,
                            active=-1,
                            collected=collected_targets
                        )
                        self.world.set_status(mode='AUTO', sm_state='FinalDemo', action='reached', target=name,
                                               progress=progress_str)
                    except Exception:
                        pass
                    break  # next target

                # 2) Plan: get as close as possible
                planned = self._plan_best_approach_to_target(txy)
                if not planned:
                    log.info("No path found yet towards %s; rescanning", name)
                    self.world.set_status(mode='AUTO', sm_state='FinalDemo', action='replan', target=name,
                                           progress=progress_str)
                    time.sleep(0.5)
                    attempt += 1
                    continue
                # 4) Drive this plan
                done_this_target = False
                while not self._stop.is_set() and self._plan_waypoints:
                    t0 = time.time()
                    pose = self.get_pose_fn()
                    self.world.set_pose(pose)
                    self._maybe_replan(pose)
                    self._drive_step(pose)
                    try:
                        total = max(1, len(self._plan_waypoints) - 1)
                        self.world.set_status(mode='AUTO', sm_state='FinalDemo', action='drive', target=name,
                                               progress=f"{min(self._wp_idx,total)}/{total}")
                    except Exception:
                        pass
                    dist = self._dist((pose[0], pose[1]), txy)
                    if dist <= 0.25:
                        print(f"🎯 <-- Reached {name}")
                        self.cmd.stop()
                        time.sleep(2.0)
                        self._plan_waypoints = []
                        # mirror collection update here to avoid re-scanning and double-reporting
                        try:
                            wm = self.world.get_targets_info()
                            collected_targets.append(current_target_index)
                            self.world.set_targets_info(
                                targets=self.all_targets_world_dict,
                                active=-1,
                                collected=collected_targets
                            )
                            self.world.set_status(mode='AUTO', sm_state='FinalDemo', action='reached', target=name,
                                                progress=progress_str)
                        except Exception:
                            pass

                        done_this_target = True
                        break
                    dt = time.time() - t0
                    if dt < self._period:
                        time.sleep(self._period - dt)
                # Loop back to scan again if not within threshold
                if done_this_target:
                    break

        self.world.set_status(mode='AUTO', sm_state='FinalDemo', action='done')
        print("Reached all targets")
        self.cmd.stop()

    def update_dynamic_layer_with_targets(self):
        targets = getattr(self, 'all_targets_world_dict', {}) or {}
        obstacles = getattr(self, 'all_obstacles_world_dict', {}) or {}

        objects = list(targets.values()) + list(obstacles.values())

        # Build positions and radii, defaulting updated_by_scan to False when missing
        positions = []
        radii = []
        for obj in objects:
            try:
                x = float(obj["x"])
                y = float(obj["y"])
            except Exception:
                continue  # skip malformed entries
            positions.append((x, y))
            is_fresh = bool(obj.get("updated_by_scan", False))
            radii.append(self.obstacle_sizes["detected"] if is_fresh else self.obstacle_sizes["undetected"])

        # Push to the grid (safe to call with empty lists)
        self.grid.set_dynamic_fruits_sizes(positions=positions, fruit_radii_m=radii)

