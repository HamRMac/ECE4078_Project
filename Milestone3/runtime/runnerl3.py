import threading
import time
import logging
import math
from typing import Optional, Tuple, List, Dict

import numpy as np

from navigation.controller import ControllerManager
from planning.astar import AStarPlanner
from planning.grid_map import GridMap
from planning.visibility_helper import compute_safety_mask
from .world_model import WorldModel
from .robot_commander import RobotCommander
from pibot_actions import PiBotActions

log = logging.getLogger(__name__)


class RunnerL3(threading.Thread):
    """Level 3 mission runner (focused logic).

    Behaviour:
    - Perform a scan and clustering (as in Level 4) and update the grid safety layer.
      When clustering/detections exist, drop any detection that is at roughly the same
      location as a known target (<= 0.15 m).
    - Append known target locations to the static exclusion map (as with ArUCO inflation).
    - Compute a route using known target positions in shopping-list order. For each target,
      plan to a 0.20 m standoff, retry once if needed, print on success, and wait briefly.
    - After all targets processed, print "Reached all targets" and stop the robot.
    """

    def __init__(self,
                 commander: RobotCommander,
                 ekf, aruco_det,
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
                 known_targets: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__(daemon=True, name="RunnerL3")
        self.cmd = commander
        self.ekf = ekf
        self.aruco = aruco_det
        self.grid = grid
        self.planner = planner or AStarPlanner()
        self.world = world
        self.get_pose_fn = get_pose_fn
        self.ctrl = ControllerManager(controller_kind)
        self._stop = threading.Event()
        self._drive_enabled = bool(drive_enabled)
        self.actions = actions
        self.detector = detector
        self.fruit_ranger = fruit_ranger
        self.target_dims = target_dims
        # Use provided ArUco positions for safety mask updates
        self.aruco_positions = aruco_positions
        self.shopping_list: List[str] = list(shopping_list or [])
        self.known_targets: Dict[str, Tuple[float, float]] = dict(known_targets or {})

        # Planning state
        self._goal: Optional[Tuple[float, float]] = None
        self._plan_waypoints: List[Tuple[float, float]] = []
        self._wp_idx: int = 0
        self._period = 1.0 / max(1.0, float(hz))
        self._xtrack_thresh: float = 0.05
        self._just_replanned: bool = False

        # Ordered route derived from shopping list
        self._route: List[Tuple[str, Tuple[float, float]]] = []
        for name in self.shopping_list:
            if name in self.known_targets:
                self._route.append((name, self.known_targets[name]))

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
        Tries angles biased towards direction from robot.
        """
        rx, ry, _ = self.get_pose_fn()
        tx, ty = float(target_xy[0]), float(target_xy[1])
        base_th = math.atan2(ty - ry, tx - rx)
        Ks = 24
        angles = [((base_th + 2 * math.pi * i / Ks + math.pi) % (2 * math.pi)) - math.pi for i in range(Ks)]
        order = []
        mid = Ks // 2
        for i in range(Ks):
            j = (i // 2) * (-1 if i % 2 else 1)
            order.append((mid + j) % Ks)
        occ = self.grid.combined()
        for idx in order:
            ang = angles[idx]
            wx = tx + radius_m * math.cos(ang)
            wy = ty + radius_m * math.sin(ang)
            r, c = self.grid.world_to_grid(wx, wy)
            if int(occ[r, c]) != 0:
                continue
            self._goal = (wx, wy)
            if self._plan_from_current():
                return True
        return False

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

    def _attempt_beeline(self, target_name: str, target_xy: Tuple[float, float], stop_radius_m: float = 0.25) -> bool:
        """Try a direct LOS crawl toward the target, ignoring dark cells.

        Returns True if we executed a beeline and ended within stop_radius_m; False otherwise.
        """
        log.info("Attempting beeline to %s at (%.2f, %.2f)", target_name, target_xy[0], target_xy[1])
        pose = self.get_pose_fn()
        rx, ry = float(pose[0]), float(pose[1])
        tx, ty = float(target_xy[0]), float(target_xy[1])
        dx, dy = (tx - rx), (ty - ry)
        dist = math.hypot(dx, dy)
        if dist <= stop_radius_m:
            return True
        # Goal point: stop short by stop_radius_m
        step = max(0.0, dist - stop_radius_m)
        if step <= 1e-3:
            gx, gy = tx, ty
        else:
            ux, uy = dx / dist, dy / dist
            gx, gy = rx + ux * step, ry + uy * step
        # Require LOS against static+dynamic
        if not self._segment_free_static_dynamic((rx, ry), (gx, gy)):
            return False
        # Install single-waypoint plan and crawl
        self._goal = (gx, gy)
        self._plan_waypoints = [(gx, gy)]
        self._wp_idx = 0
        self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
        self.world.set_status(mode='AUTO', sm_state='L3', action='beeline', target=target_name)
        log.debug("Beeline to %s at (%.2f, %.2f) from (%.2f, %.2f)", target_name, tx, ty, rx, ry)
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

    def _scan_and_update(self):
        # Execute scan (drive-based) if available
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

        # Read clusters
        all_dets = []
        try:
            all_dets = getattr(self.actions, 'current_obj_positions', []) or []
        except Exception:
            all_dets = []

        # Level 3 policy:
        # - GUI detections exclude any item whose class/label is in the shopping list
        #   and exclude near-known-target duplicates.
        # - Keep-clear uses ALL detections irrespective of class.
        shopping = set(str(s) for s in (self.shopping_list or []))
        gui_dets = []
        keepclear_positions = []
        for d in all_dets:
            try:
                pos = d.get('position')
                if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                    continue
                wx, wy = float(pos[0]), float(pos[1])
                # Label check for filtering
                lab = d.get('class') if ('class' in d) else d.get('label')
                lab = str(lab) if lab is not None else ''
                if lab in shopping:
                    continue
                if any(self._dist((wx, wy), txy) <= 0.15 for txy in self.known_targets.values()):
                    continue
                # Keep-clear set
                keepclear_positions.append((wx, wy))
                gui_dets.append(d)
            except Exception:
                continue
        self.world.set_detections(gui_dets)

        # Update visibility-based safety and dynamic fruit obstacles (use keep-clear set)
        try:
            fruit_positions = list(keepclear_positions)
            if self.aruco_positions is not None and self.grid.size is not None:
                safe = compute_safety_mask(self.grid,
                                           robot_pose=self.get_pose_fn(),
                                           aruco_positions=self.aruco_positions,
                                           fruit_positions=fruit_positions,
                                           marker_length=0.07,
                                           fruit_radius=0.05,
                                           fov_deg=360.0,
                                           max_distance=0.8,
                                           step_cells=2)
                self.grid.apply_safety_mask(safe)
            if fruit_positions:
                self.grid.set_dynamic_fruits(fruit_positions, fruit_radius_m=0.05)
        except Exception as e:
            log.debug("Safety/dynamic update skipped: %s", e)

    # ---------------- Main loop ----------------
    def run(self):
        log.info("RunnerL3 starting; targets: %s", ", ".join([n for n, _ in self._route]) or '<none>')

        # Append known targets to static exclusion map
        try:
            self._apply_static_target_exclusions([xy for _, xy in self._route], fruit_radius_m=0.05)
        except Exception:
            pass

        # Publish targets info to WorldModel for GUI overlay
        try:
            order = list(self.shopping_list or [])
            remaining = {name: (float(xy[0]), float(xy[1])) for name, xy in self._route}
            positions = {str(k): (float(v[0]), float(v[1])) for k, v in self.known_targets.items()}
            self.world.set_targets_info(order=order,
                                        remaining=remaining,
                                        collected=[],
                                        seen_not_collected=list(remaining.keys()),
                                        unseen=[n for n in order if n not in remaining],
                                        active=None,
                                        positions=positions)
            # Initial status
            self.world.set_status(mode='AUTO', sm_state='L3', action='init', progress=f"0/{len(self._route)}")
        except Exception:
            pass

        # Visit targets in order with scan → approach loop
        for idx, (name, txy) in enumerate(self._route):
            log.info("Heading to target %d/%d: %s at (%.2f, %.2f)", idx + 1, len(self._route), name, txy[0], txy[1])
            if self._stop.is_set():
                break
            # Mark this as the active target in the world model
            try:
                info = self.world.get_targets_info()
                self.world.set_targets_info(order=info.get('order', []),
                                            remaining=info.get('remaining', {}),
                                            collected=info.get('collected', []),
                                            seen_not_collected=info.get('seen_not_collected', []),
                                            unseen=info.get('unseen', []),
                                            active=name,
                                            positions=info.get('positions', {}))
                self.world.set_status(mode='AUTO', sm_state='L3', action='scan', target=name,
                                       progress=f"{idx+1}/{len(self._route)}")
            except Exception:
                pass

            attempt = 0
            while not self._stop.is_set():
                # 1) Scan
                log.info("Starting scan before approaching %s (attempt %d)", name, attempt + 1)
                self.world.set_status(mode='AUTO', sm_state='L3', action='scan', target=name,
                                       progress=f"{idx+1}/{len(self._route)}")
                self._scan_and_update()
                pose = self.get_pose_fn()
                self.world.set_pose(pose)
                dist = self._dist((pose[0], pose[1]), txy)
                if dist <= 0.25:
                    print(f"Reached {name}")
                    time.sleep(2.0)
                    # Update targets info: mark as collected
                    try:
                        info = self.world.get_targets_info()
                        remaining = dict(info.get('remaining', {}))
                        if name in remaining:
                            del remaining[name]
                        collected = list(info.get('collected', []))
                        if name not in collected:
                            collected.append(name)
                        order = list(info.get('order', []))
                        seen = [n for n in remaining.keys()]
                        unseen = [n for n in order if (n not in remaining) and (n not in collected)]
                        self.world.set_targets_info(order=order,
                                                    remaining=remaining,
                                                    collected=collected,
                                                    seen_not_collected=seen,
                                                    unseen=unseen,
                                                    active=None,
                                                    positions=info.get('positions', {}))
                        self.world.set_status(mode='AUTO', sm_state='L3', action='reached', target=name,
                                               progress=f"{idx+1}/{len(self._route)}")
                    except Exception:
                        pass
                    break  # next target
                # 2) Beeline if LOS free (static+dynamic only)
                if self._attempt_beeline(name, txy, stop_radius_m=0.25):
                    print(f"Reached {name}")
                    time.sleep(2.0)
                    try:
                        info = self.world.get_targets_info()
                        remaining = dict(info.get('remaining', {}))
                        if name in remaining:
                            del remaining[name]
                        collected = list(info.get('collected', []))
                        if name not in collected:
                            collected.append(name)
                        order = list(info.get('order', []))
                        seen = [n for n in remaining.keys()]
                        unseen = [n for n in order if (n not in remaining) and (n not in collected)]
                        self.world.set_targets_info(order=order,
                                                    remaining=remaining,
                                                    collected=collected,
                                                    seen_not_collected=seen,
                                                    unseen=unseen,
                                                    active=None,
                                                    positions=info.get('positions', {}))
                        self.world.set_status(mode='AUTO', sm_state='L3', action='reached', target=name,
                                               progress=f"{idx+1}/{len(self._route)}")
                    except Exception:
                        pass
                    break

                # 3) Plan: get as close as possible
                planned = self._plan_best_approach_to_target(txy)
                if not planned:
                    log.info("No path found yet towards %s; rescanning", name)
                    self.world.set_status(mode='AUTO', sm_state='L3', action='replan', target=name,
                                           progress=f"{idx+1}/{len(self._route)}")
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
                        self.world.set_status(mode='AUTO', sm_state='L3', action='drive', target=name,
                                               progress=f"{min(self._wp_idx,total)}/{total}")
                    except Exception:
                        pass
                    dist = self._dist((pose[0], pose[1]), txy)
                    if dist <= 0.25:
                        print(f"Reached {name}")
                        time.sleep(2.0)
                        self._plan_waypoints = []
                        self.cmd.stop()
                        # mirror collection update here to avoid re-scanning and double-reporting
                        try:
                            info = self.world.get_targets_info()
                            remaining = dict(info.get('remaining', {}))
                            if name in remaining:
                                del remaining[name]
                            collected = list(info.get('collected', []))
                            if name not in collected:
                                collected.append(name)
                            order = list(info.get('order', []))
                            seen = [n for n in remaining.keys()]
                            unseen = [n for n in order if (n not in remaining) and (n not in collected)]
                            self.world.set_targets_info(order=order,
                                                        remaining=remaining,
                                                        collected=collected,
                                                        seen_not_collected=seen,
                                                        unseen=unseen,
                                                        active=None,
                                                        positions=info.get('positions', {}))
                            self.world.set_status(mode='AUTO', sm_state='L3', action='reached', target=name,
                                                   progress=f"{idx+1}/{len(self._route)}")
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

        self.world.set_status(mode='AUTO', sm_state='L3', action='done')
        print("Reached all targets")
        self.cmd.stop()
