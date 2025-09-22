import threading
import time
import logging
import math
from typing import Optional, Tuple, List

import math

from navigation.controller import ControllerManager
from planning.astar import AStarPlanner
from planning.grid_map import GridMap
from planning.visibility_helper import compute_safety_mask
from planning.sector_planner import SectorExplorer
import numpy as np
from .world_model import WorldModel
from .robot_commander import RobotCommander
from state_machine.state_machine import PiBotFruitSearchSMLevel3
from pibot_actions import PiBotActions

log = logging.getLogger(__name__)


class RunnerL3(threading.Thread):
    """Mission runner for level 3: consumes intents, updates pose, plans, and drives the robot."""

    def __init__(self,
                 commander: RobotCommander,
                 ekf, aruco_det,
                 grid: GridMap,
                 planner: Optional[AStarPlanner],
                 world: WorldModel,
                 get_pose_fn,
                 intents_q,
                 controller_kind: str = "ttg",
                 hz: float = 10.0,
                 drive_enabled: bool = True,
                 state_machine: PiBotFruitSearchSMLevel3 = None,
                 actions: PiBotActions=None,
                 detector=None,
                 fruit_ranger=None,
                 target_dims=None,
                 aruco_positions: np.ndarray = None,
                 shopping_list: Optional[List[str]] = None):
        super().__init__(daemon=True, name="Runner")
        log.info("Runner initialized")
        self.cmd = commander
        self.ekf = ekf
        self.aruco = aruco_det
        self.grid = grid
        self.planner = planner or AStarPlanner()
        self.world = world
        self.get_pose_fn = get_pose_fn
        self.q = intents_q
        self.ctrl = ControllerManager(controller_kind)
        self._stop = threading.Event()
        self._drive_enabled = bool(drive_enabled)
        self.sm = state_machine
        self.actions = actions
        self.detector = detector
        self.fruit_ranger = fruit_ranger
        self.target_dims = target_dims
        self.aruco_positions = aruco_positions
        self.shopping_list: List[str] = list(shopping_list or [])
        self._goal: Optional[Tuple[float, float]] = None
        self._plan_waypoints: List[Tuple[float, float]] = []
        self._wp_idx: int = 0
        self._period = 1.0 / max(1.0, float(hz))
        self._last_plan_time: float = 0.0
        self._xtrack_thresh: float = 0.15
        self._replan_period_s: float = 0
        self._last_plan_by_goal: dict[Tuple[float, float], List[Tuple[float, float]]] = {}
        # Sector exploration state
        self._sector_explorer = SectorExplorer(rows=3, cols=3, min_clearance_m=0.10)
        self._searched_sectors: set[tuple[int, int]] = set()
        self._next_sector_idx: Optional[Tuple[int, int]] = None
        self._next_scan_point: Optional[Tuple[float, float]] = None
        # Track if a plan was just (re)generated to avoid spinning at the very first waypoint
        self._just_replanned: bool = False
        # Compiled map of remaining targets (fruit -> (x,y))
        self._remaining_targets: dict[str, Tuple[float, float]] = {}
        self._collected_targets: set[str] = set()
        # Current target tracking
        self._current_target_name: Optional[str] = None
        self._current_target_xy: Optional[Tuple[float, float]] = None
        self._approach_radius_m: float = 0.20
        # Final scan flow flags
        self._final_scan_requested: bool = False
        self._switch_to_targets: bool = False
        # Post-escape desired goal
        self._post_escape_goal: Optional[Tuple[float, float]] = None

        # modes: 'IDLE' | 'MANUAL_WAYPOINTS' | 'AUTO'
        # Start in IDLE; GUI can send SwitchMode('AUTO') (e.g., press 'S') to begin SM control
        self.mode = 'IDLE'

    def stop(self):
        self._stop.set()
        self.cmd.stop()

    def _handle_intents(self):
        try:
            while True:
                intent = self.q.get_nowait()
                name = intent.__class__.__name__
                if name == 'SetGoal':
                    self._goal = (float(intent.x), float(intent.y))
                    self.mode = 'MANUAL_WAYPOINTS'
                    self.world.set_status(mode=self.mode, action='plan', sm_state='Manual')
                    self._plan_from_current()
                elif name == 'CancelGoal':
                    self._goal = None
                    self._plan_waypoints = []
                    self._wp_idx = 0
                    self.mode = 'IDLE'
                    self.world.clear_plan()
                    self.world.set_status(mode=self.mode, action='idle')
                elif name == 'SwitchMode':
                    self.mode = str(intent.mode)
                    self.world.set_status(mode=self.mode)
                else:
                    pass
        except Exception:
            # empty queue
            pass

    def _plan_from_current(self):
        pose = self.get_pose_fn()
        if self._goal is None:
            return
        log.info("Planning to goal (%.2f, %.2f)", self._goal[0], self._goal[1])
        desired_goal = (float(self._goal[0]), float(self._goal[1]))
        pr = self.planner.plan(self.grid, (pose[0], pose[1]), desired_goal)
        if pr is None:
            # If we failed and are starting inside an occupied cell, attempt an escape-to-free-cell plan first
            try:
                occ = self.grid.combined()
                s_rc = self.grid.world_to_grid(pose[0], pose[1])
                start_blocked = int(occ[s_rc[0], s_rc[1]]) != 0
            except Exception:
                start_blocked = False
            if start_blocked:
                esc_xy = self._find_nearest_free_xy(s_rc)
                if esc_xy is not None:
                    log.info("Planning escape to nearest free cell @ (%.2f, %.2f) before re-attempting goal",
                             esc_xy[0], esc_xy[1])
                    # Store post-escape desired goal and replan to escape point
                    self._post_escape_goal = desired_goal
                    self._goal = esc_xy
                    self.world.set_status(action='escape', progress='to_free')
                    pr = self.planner.plan(self.grid, (pose[0], pose[1]), self._goal)
                    if pr is None:
                        log.warning("Escape planning also failed; aborting plan cycle")
                        self._plan_waypoints = []
                        self.world.clear_plan()
                        self.world.set_status(action='plan_failed')
                        return False
                else:
                    log.warning("Start is blocked and no nearby free cell found; aborting plan")
                    self._plan_waypoints = []
                    self.world.clear_plan()
                    self.world.set_status(action='plan_failed')
                    return False
            else:
                self._plan_waypoints = []
                self.world.clear_plan()
                self.world.set_status(action='plan_failed')
                log.warning("Plan failed to goal (%.2f, %.2f)", desired_goal[0], desired_goal[1])
                return False
        self._plan_waypoints = list(pr.pruned_world if pr.pruned_world else pr.path_world)
        self._wp_idx = 0
        self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
        self.world.set_status(action='drive', progress=f"0/{len(self._plan_waypoints)}")
        log.info("Plan OK: %d waypoints (pruned=%s)", len(self._plan_waypoints), "yes" if pr.pruned_world else "no")
        # Record last plan time and cache plan by goal for fallback
        self._last_plan_time = time.time()
        try:
            key = (round(self._goal[0], 3), round(self._goal[1], 3))
            self._last_plan_by_goal[key] = list(self._plan_waypoints)
        except Exception:
            pass
        # Mark as just replanned so we can skip initial spin
        self._just_replanned = True
        return True

    def _find_nearest_free_xy(self, start_rc: Tuple[int, int], max_radius_cells: int = 30) -> Optional[Tuple[float, float]]:
        """Find nearest free cell around start_rc and return its world (x,y)."""
        try:
            occ = self.grid.combined()
            H, W = occ.shape
        except Exception:
            return None
        sr, sc = int(start_rc[0]), int(start_rc[1])
        best = None
        best_d2 = 1e12
        for rad in range(1, int(max_radius_cells) + 1):
            r0 = max(0, sr - rad); r1 = min(H - 1, sr + rad)
            c0 = max(0, sc - rad); c1 = min(W - 1, sc + rad)
            found = False
            for r in range(r0, r1 + 1):
                for c in (c0, c1):
                    if int(occ[r, c]) == 0:
                        d2 = (r - sr) * (r - sr) + (c - sc) * (c - sc)
                        if d2 < best_d2:
                            best_d2 = d2; best = (r, c); found = True
            for c in range(c0, c1 + 1):
                for r in (r0, r1):
                    if int(occ[r, c]) == 0:
                        d2 = (r - sr) * (r - sr) + (c - sc) * (c - sc)
                        if d2 < best_d2:
                            best_d2 = d2; best = (r, c); found = True
            if found and best is not None:
                wx, wy = self.grid.grid_to_world(best[0], best[1])
                return (float(wx), float(wy))
        return None

    # ---------------- Target approach helpers ----------------
    def _select_next_target(self) -> Optional[Tuple[str, Tuple[float, float]]]:
        """Pick the next target to approach based on shopping list order and availability."""
        if not self._remaining_targets:
            return None
        for name in list(self.shopping_list or []):
            if name in self._remaining_targets:
                return name, self._remaining_targets[name]
        # Fallback: any available target
        for k, v in self._remaining_targets.items():
            return k, v
        return None

    def _plan_approach_to_target(self, name: str, target_xy: Tuple[float, float], radius_m: float = 0.20) -> bool:
        """Plan to a standoff point around target within radius_m using A*.

        Samples candidate waypoints on a circle around the target and tries to plan to each until one succeeds.
        """
        if self.grid.size is None or self.grid.bounds_wm is None:
            return False
        rx, ry, _ = self.get_pose_fn()
        tx, ty = float(target_xy[0]), float(target_xy[1])
        # Sample K candidates around the circle, biased towards direction from robot
        dx, dy = tx - rx, ty - ry
        base_th = math.atan2(dy, dx)
        Ks = 16
        cand_angles = [((base_th + 2 * math.pi * i / Ks + math.pi) % (2 * math.pi)) - math.pi for i in range(Ks)]
        # Interleave angles to try closest to base_th first
        order = []
        mid = Ks // 2
        for i in range(Ks):
            j = (i // 2) * (-1 if i % 2 else 1)
            idx = (mid + j) % Ks
            if idx not in order:
                order.append(idx)
        occ = self.grid.combined()
        best_pr = None
        best_wp = None
        for idx in order:
            ang = cand_angles[idx]
            wx = tx + radius_m * math.cos(ang)
            wy = ty + radius_m * math.sin(ang)
            # Check within bounds and free
            r, c = self.grid.world_to_grid(wx, wy)
            if int(occ[r, c]) != 0:
                continue
            pr = self.planner.plan(self.grid, (rx, ry), (wx, wy))
            if pr is not None:
                best_pr = pr
                best_wp = (wx, wy)
                break
        if best_pr is None or best_wp is None:
            log.info("Approach planning failed for %s; no reachable standoff at %.2fm", name, radius_m)
            return False
        # Apply plan
        self._goal = best_wp
        self._plan_waypoints = list(best_pr.pruned_world if best_pr.pruned_world else best_pr.path_world)
        self._wp_idx = 0
        self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
        self.world.set_status(action='drive_to_target', target=name, progress=f"0/{len(self._plan_waypoints)}")
        self._last_plan_time = time.time()
        try:
            key = (round(self._goal[0], 3), round(self._goal[1], 3))
            self._last_plan_by_goal[key] = list(self._plan_waypoints)
        except Exception:
            pass
        self._just_replanned = True
        return True

    def _maybe_replan(self, pose):
        if self._goal is None or not self._plan_waypoints:
            return
        # Periodic replan
        need_replan = (self._replan_period_s != 0) and ((time.time() - self._last_plan_time) >= self._replan_period_s)
        # Cross-track error based replan
        try:
            xtrack = AStarPlanner.cross_track_error((pose[0], pose[1]), self._plan_waypoints)
            if xtrack > self._xtrack_thresh:
                log.info("Cross-track %.3f > %.3f; triggering replanning", xtrack, self._xtrack_thresh)
                need_replan = True
        except Exception:
            pass
        if not need_replan:
            return
        # Attempt replan
        prev_plan = list(self._plan_waypoints)
        prev_wp_idx = self._wp_idx
        success = self._plan_from_current()
        if not success:
            # Fallback to last-known plan towards this goal if available
            try:
                key = (round(self._goal[0], 3), round(self._goal[1], 3))
                cached = self._last_plan_by_goal.get(key)
                if cached:
                    self._plan_waypoints = list(cached)
                    # Keep closest future waypoint as current
                    self._wp_idx = min(prev_wp_idx, len(self._plan_waypoints) - 1)
                    self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
                    self.world.set_status(action='drive', progress=f"{self._wp_idx}/{len(self._plan_waypoints)-1}")
                    log.info("Replan failed; using cached plan to goal (%.2f, %.2f)", self._goal[0], self._goal[1])
                else:
                    # Restore previous plan if no cache
                    self._plan_waypoints = prev_plan
                    self._wp_idx = min(prev_wp_idx, len(self._plan_waypoints) - 1)
                    if self._plan_waypoints:
                        self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
                        self.world.set_status(action='drive')
            except Exception:
                pass

    def _drive_step(self, pose):
        if not self._plan_waypoints:
            return
        self._wp_idx = min(self._wp_idx, len(self._plan_waypoints) - 1)
        wp = self._plan_waypoints[self._wp_idx]
        fwd_cmd, turn_cmd, fwd_tick, turn_tick, done = self.ctrl.compute(pose, wp)
        if self._drive_enabled:
            self.cmd.set_velocity([fwd_cmd, turn_cmd], tick=fwd_tick, turning_tick=turn_tick, time=0)
        else:
            # Reflect disabled driving in status; no velocity command sent
            self.world.set_status(action='drive_disabled')
        log.debug("drive_step: wp_idx=%d/%d cmd=(%d,%d) ticks=(%d,%d)", self._wp_idx, max(0, len(self._plan_waypoints)-1), fwd_cmd, turn_cmd, fwd_tick, turn_tick)
        if done:
            if self._wp_idx < len(self._plan_waypoints) - 1:
                # Perform a quick 360° spin at each waypoint, except immediately after a replan at the first wp
                do_spin = not (self._wp_idx == 0 and self._just_replanned)
                '''
                try:
                    if do_spin and self.actions is not None and self._drive_enabled:
                        self.world.set_status(action='spin', progress=f"wp{self._wp_idx}")
                        
                        self.actions.localise_scan(step_angle_deg=45.0,
                                                   get_pose_fn=self.get_pose_fn,
                                                   turning_tick=50,
                                                   pause_s=0.2)
                        
                except Exception:
                    pass
                '''
                # Clear the just-replanned guard after first advancement opportunity
                if self._just_replanned:
                    self._just_replanned = False

                self._wp_idx += 1
                self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
                self.world.set_status(progress=f"{self._wp_idx}/{len(self._plan_waypoints)-1}")
                log.debug("Reached waypoint; advancing to %d", self._wp_idx)
            else:
                # Final waypoint reached: optional final spin as well
                '''
                try:
                    if self.actions is not None and self._drive_enabled:
                        self.world.set_status(action='spin_final')
                        self.actions.localise_scan(step_angle_deg=45.0,
                                                   get_pose_fn=self.get_pose_fn,
                                                   turning_tick=50,
                                                   pause_s=0.2)
                except Exception:
                    pass
                '''
                self.cmd.stop()
                self.world.set_status(action='arrived')
                self._plan_waypoints = []
                self.world.clear_plan()
                # If we were escaping to a free cell, immediately replan to the original desired goal
                if self._post_escape_goal is not None:
                    self._goal = self._post_escape_goal
                    self._post_escape_goal = None
                    log.info("Escape complete; replanning to original goal (%.2f, %.2f)", self._goal[0], self._goal[1])
                    self._plan_from_current()
                # Only drop to IDLE if in manual mode; in AUTO, let SM decide next step
                if self.mode == 'MANUAL_WAYPOINTS':
                    self.mode = 'IDLE'
                log.info("Arrived at goal.")

    # ---------------- AUTO (SM-driven) ----------------
    def _auto_step(self, pose):
        if self.sm is None:
            return
        try:
            sm_state = self.sm.current_state.id.lower()
            log.debug("SM state: %s", sm_state)
        except Exception:
            sm_state = 'scan'
        
        # EXECUTES IN STATE "Scan"
        if sm_state == 'scan':
            # Call PiBotActions.scan if allowed; otherwise simulate dwell
            self.world.set_status(mode='AUTO', sm_state='scan', action='scan')
            log.info("SM: Scan → scan action")
            if self.actions is not None and self._drive_enabled:
                try:
                    # conservative scan parameters; pass detector/ranger/target_dims if available
                    self.actions.scan(
                        step_angle_deg=30.0,
                        detector=self.detector,
                        fruit_ranger=self.fruit_ranger,
                        target_dims=self.target_dims,
                        get_pose_fn=self.get_pose_fn,
                        turning_tick=40,
                        pause_s=1.0,
                    )
                    # Publish detections (clustered) to world model if available
                    try:
                        dets = getattr(self.actions, 'current_obj_positions', []) or []
                        self.world.set_detections(dets)
                        log.info("Scan complete: %d clustered objects", len(dets))
                        # Update observed-free mask in grid using visibility helper
                        fruit_positions = []
                        for d in dets:
                            pos = d.get('position')
                            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                                fruit_positions.append((float(pos[0]), float(pos[1])))
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
                            # Update dynamic obstacles from current fruit positions with buffered radius
                            if fruit_positions:
                                self.grid.set_dynamic_fruits(fruit_positions, fruit_radius_m=0.05)
                            log.info("Applied safety mask (observed safe cells: %d) and dynamic fruit obstacles (%d)", int(np.count_nonzero(safe)), len(fruit_positions))
                        # After scan, report nearby shopping-list targets from clustered results (<=0.5m)
                        try:
                            pose_now = self.get_pose_fn()
                            rx, ry, rth = float(pose_now[0]), float(pose_now[1]), float(pose_now[2])
                            shopping = set(self.shopping_list or [])
                            nearby = []
                            for item in (dets or []):
                                try:
                                    label = item.get('class')
                                    if shopping and (label not in shopping):
                                        continue
                                    pos = item.get('position')
                                    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                                        continue
                                    wx = float(pos[0]); wy = float(pos[1])
                                    dx, dy = wx - rx, wy - ry
                                    dist = float((dx*dx + dy*dy) ** 0.5)
                                    if dist <= 0.5:
                                        heading_deg = (math.degrees(math.atan2(dy, dx) - rth) + 180.0) % 360.0 - 180.0
                                        nearby.append((str(label), dist, heading_deg))
                                except Exception:
                                    continue
                            if nearby:
                                msg = ", ".join([f"{lab}: d={d:.2f}m, hdg={h:+.0f}°" for (lab, d, h) in nearby])
                                log.info("Nearby targets: %s", msg)
                        except Exception:
                            pass
                    except Exception as e:
                        log.warning("Visibility update failed: %s", e)
                    # Mark current sector as searched AFTER completing the scan
                    try:
                        pose_now = self.get_pose_fn()
                        ix, iy = self._sector_explorer.xy_to_sector_idx(self.grid, float(pose_now[0]), float(pose_now[1]))
                        self._searched_sectors.add((ix, iy))
                        # Publish overlay info to world model
                        self.world.set_sectors(rows=self._sector_explorer.rows,
                                               cols=self._sector_explorer.cols,
                                               searched=list(self._searched_sectors),
                                               next_idx=self._next_sector_idx,
                                               next_point=self._next_scan_point)
                        log.info("Marked sector (%d,%d) as searched", ix, iy)
                        # If we requested a final scan and we are in the center sector, signal next transition
                        center_ix = self._sector_explorer.cols // 2
                        center_iy = self._sector_explorer.rows // 2
                        if self._final_scan_requested and ix == center_ix and iy == center_iy:
                            self._switch_to_targets = True
                            self._final_scan_requested = False
                            log.info("Final center-sector scan done; will proceed to compile targets")
                    except Exception as e:
                        log.warning("Failed to mark searched sector: %s", e)
                except Exception:
                    pass
            else:
                log.info("Scan disabled (drive off). Sleeping for 0.5 seconds")
                time.sleep(0.5)
            # Transition to next state
            try:
                self.sm.T_scan_to_calculate_next_safe_point()
            except Exception:
                pass
        # EXECUTES IN STATE "ApproachTarget"
        elif sm_state == 'approach_target':
            # Use PiBotActions to approach the current queued target if available
            log.info("SM: ApproachFruit → approach_current action")
            if self.actions is not None and hasattr(self.actions, 'approach_current'):
                try:
                    res = self.actions.approach_current()
                    if res is None:
                        log.info("approach_current: no target to approach")
                    else:
                        log.info("approach_current: result=%s", res)
                    try:
                        self.sm.T_approach_fruit_to_navigate_to_fruit()
                    except Exception:
                        pass
                except Exception as e:
                    log.warning("Runner: approach_current failed: %s", e)
            else:
                # Fallback: plan to a placeholder goal
                self._goal = (1.0, 0.0)
                self._plan_from_current()
                try:
                    self.sm.T_approach_fruit_to_navigate_to_fruit()
                except Exception:
                    pass

        # EXECUTES IN STATE "SitNextToCloseFruit"
        elif sm_state == 'sit_next_to_close_fruit':
            # Placeholder: sit next to fruit at (1,0)
            self.world.set_status(mode='AUTO', sm_state='sit_next_to_close_fruit', action='sit_next_to_fruit')
            if self.actions is not None:
                try:
                    # Use the PiBotActions convenience method which pops the target
                    ok = self.actions.collect_current(duration_s=2.1)
                    if ok:
                        log.info("SM: SitNextToCloseFruit → collected current target")
                    else:
                        log.info("SM: SitNextToCloseFruit → no target to collect")
                except Exception as e:
                    log.warning("Runner: collect_current failed: %s", e)
            else:
                # Fallback: sleep to simulate collection
                time.sleep(2.1)

        # EXECUTES IN STATE "GoBackToScanPoint"
        elif sm_state == 'go_back_to_scan_point':
            # Attempt to return from current using PiBotActions if available
            log.info("SM: GoBackToScanPoint → return_from_current action")
            if self.actions is not None and hasattr(self.actions, 'return_from_current'):
                try:
                    ok = self.actions.return_from_current()
                    if ok:
                        log.info("return_from_current: returned by %.3f m", getattr(self.actions, 'last_forward', 0.0))
                    else:
                        log.info("return_from_current: nothing to return from; planning to (0,0)")
                        self._goal = (0.0, 0.0)
                        self._plan_from_current()
                    try:
                        self.sm.T_go_back_to_scan_point_to_scan()
                    except Exception:
                        pass
                except Exception as e:
                    log.warning("Runner: return_from_current failed: %s", e)
            else:
                # Fallback: plan back to origin
                self._goal = (0.0, 0.0)
                self._plan_from_current()
                try:
                    self.sm.T_go_back_to_scan_point_to_scan()
                except Exception:
                    pass



        # EXECUTES IN STATE "CalculateNextSafePoint"
        elif sm_state == 'calculate_next_safe_point':
            # If final scan at origin has been completed, proceed to target evaluation
            if self._switch_to_targets:
                log.info("Final scan complete; transitioning to CheckForRemainingTargets")
                self._switch_to_targets = False
                try:
                    self.sm.T_calculate_next_safe_point_to_check_for_remaining_targets()
                except Exception:
                    pass
                return
            self.world.set_status(mode='AUTO', sm_state='calculate_next_safe_point', action='choose_safe_point')
            log.info("SM: CalculateNextSafePoint → selecting sector-based safe point (safest-first, plan-validated)")
            pose_now = self.get_pose_fn()
            tried: set[tuple[int, int]] = set()
            chosen_goal: Optional[Tuple[float, float]] = None
            chosen_idx: Optional[Tuple[int, int]] = None
            chosen_info = None
            # Iterate through sectors by priority until we find one we can plan to
            for _ in range(self._sector_explorer.rows * self._sector_explorer.cols):
                try:
                    pick = self._sector_explorer.pick_next_target(self.grid, excluded=(self._searched_sectors | tried))
                except Exception as e:
                    log.warning("Sector selection failed: %s", e)
                    pick = None
                if pick is None:
                    break
                goal, sector_idx, info = pick
                # Try to plan to this goal
                pr = None
                try:
                    pr = self.planner.plan(self.grid, (pose_now[0], pose_now[1]), (goal[0], goal[1]))
                except Exception:
                    pr = None
                if pr is None:
                    log.info("Sector %s goal (%.2f, %.2f) unreachable now; trying next.", str(sector_idx), goal[0], goal[1])
                    tried.add(tuple(sector_idx))
                    continue
                # Accept this sector and plan
                chosen_goal = (float(goal[0]), float(goal[1]))
                chosen_idx = tuple(sector_idx)
                chosen_info = info
                # Apply plan to world/runner
                self._goal = chosen_goal
                self._plan_waypoints = list(pr.pruned_world if pr.pruned_world else pr.path_world)
                self._wp_idx = 0
                self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
                self.world.set_status(action='drive', progress=f"0/{len(self._plan_waypoints)}")
                self._last_plan_time = time.time()
                try:
                    key = (round(self._goal[0], 3), round(self._goal[1], 3))
                    self._last_plan_by_goal[key] = list(self._plan_waypoints)
                except Exception:
                    pass
                # New plan installed; skip spin at first waypoint
                self._just_replanned = True
                break

            if chosen_goal is None:
                # No viable sector: plan to origin for a final scan, then go to targets
                self._final_scan_requested = True
                self._goal = (0.0, 0.0)
                log.info("No scannable sectors remain → planning to origin (0,0) for final scan")
                self._plan_from_current()
            else:
                # Defer exclusion until scan completes, and publish overlay
                self._next_sector_idx = chosen_idx
                self._next_scan_point = chosen_goal
                self.world.set_sectors(rows=self._sector_explorer.rows,
                                       cols=self._sector_explorer.cols,
                                       searched=list(self._searched_sectors),
                                       next_idx=self._next_sector_idx,
                                       next_point=self._next_scan_point)
                if chosen_info is not None:
                    log.info("Selected sector %s (dark=%.2f free=%d/%d) → goal @ (%.2f, %.2f)",
                             str(chosen_idx), float(chosen_info.dark_fraction), int(chosen_info.free_cells), int(chosen_info.total_cells), chosen_goal[0], chosen_goal[1])

            try:
                self.sm.T_calculate_next_safe_point_to_navigate_to_safe_point()
            except Exception:
                pass
        
        # EXECUTES IN STATE "NavigateToSafePoint"
        elif sm_state == 'navigate_to_safe_point':
            self.world.set_status(mode='AUTO', sm_state='navigate_to_safe_point', action='begin_drive')
            # If we have a plan, drive a step; otherwise consider arrival
            if self._plan_waypoints:
                # Periodic and cross-track replanning while following
                self._maybe_replan(pose)
                self._drive_step(pose)
            else:
                # arrived or cannot plan; go back to spin
                log.info("SM: NavigateToSafePoint → Scan")
                try:
                    self.sm.T_navigate_to_safe_point_to_scan()
                except Exception:
                    pass

        # EXECUTES IN STATE "GoToTargetFruit"
        elif sm_state == 'go_to_target_fruit':
            self.world.set_status(mode='AUTO', sm_state='go_to_target_fruit', action='plan_or_drive')
            # If we don't have a current target, select one
            if self._current_target_name is None or self._current_target_xy is None:
                sel = self._select_next_target()
                if sel is None:
                    log.info("No remaining targets to go to; returning to scan")
                    try:
                        self.sm.T_go_to_target_fruit_to_sit_next_to_target_fruit()
                    except Exception:
                        pass
                else:
                    self._current_target_name, self._current_target_xy = sel[0], sel[1]
                    log.info("Selected target '%s' @ (%.2f, %.2f)", self._current_target_name, self._current_target_xy[0], self._current_target_xy[1])
            # If no plan present, plan an approach
            if not self._plan_waypoints and self._current_target_name and self._current_target_xy:
                ok = self._plan_approach_to_target(self._current_target_name, self._current_target_xy, radius_m=self._approach_radius_m)
                if not ok:
                    log.info("Could not plan to target '%s'; skipping", self._current_target_name)
                    # Remove this target and reselect next time
                    try:
                        if self._current_target_name in self._remaining_targets:
                            del self._remaining_targets[self._current_target_name]
                    except Exception:
                        pass
                    self._current_target_name = None
                    self._current_target_xy = None
            # Drive if plan exists
            if self._plan_waypoints:
                self._maybe_replan(pose)
                self._drive_step(pose)
            else:
                # No plan; fall back
                time.sleep(0.1)
            # If arrived (plan cleared by _drive_step), transition to sit next
            if not self._plan_waypoints and self._current_target_name:
                try:
                    self.sm.T_go_to_target_fruit_to_sit_next_to_target_fruit()
                except Exception:
                    pass

        # EXECUTES IN STATE "SitNextToTargetFruit"
        elif sm_state == 'sit_next_to_target_fruit':
            self.world.set_status(mode='AUTO', sm_state='sit_next_to_target_fruit', action='collect_target')
            name = self._current_target_name or 'fruit'
            if self.actions is not None and hasattr(self.actions, 'collect_current'):
                try:
                    # Use specific class if available
                    self.actions.collect_current(duration_s=2.0, collection_class=name)
                except Exception:
                    pass
            else:
                time.sleep(2.0)
            # Remove this target from remaining and clear current
            try:
                if name in self._remaining_targets:
                    del self._remaining_targets[name]
                self._collected_targets.add(str(name))
            except Exception:
                pass
            self._current_target_name = None
            self._current_target_xy = None
            # Loop to next target if any, else return to scan
            if self._remaining_targets:
                # Publish updated target status
                try:
                    order = list(self.shopping_list or [])
                    collected = sorted(list(self._collected_targets))
                    seen_not_collected = sorted(list(self._remaining_targets.keys()))
                    unseen = [f for f in order if (f not in self._remaining_targets) and (f not in self._collected_targets)]
                    self.world.set_targets_info(order=order,
                                                remaining=self._remaining_targets,
                                                collected=collected,
                                                seen_not_collected=seen_not_collected,
                                                unseen=unseen)
                except Exception:
                    pass
                try:
                    self.sm.T_sit_next_to_target_fruit_to_go_to_target_fruit()
                except Exception:
                    pass
            else:
                # Publish final target status
                try:
                    order = list(self.shopping_list or [])
                    collected = sorted(list(self._collected_targets))
                    self.world.set_targets_info(order=order,
                                                remaining={},
                                                collected=collected,
                                                seen_not_collected=[],
                                                unseen=[f for f in order if f not in self._collected_targets])
                except Exception:
                    pass
                try:
                    self.sm.T_sit_next_to_target_fruit_to_end_state()
                except Exception:
                    pass
        # EXECUTES IN STATE "CheckForRemainingTargets"
        elif sm_state == 'check_for_remaining_targets':
            self.world.set_status(mode='AUTO', sm_state='check_for_remaining_targets', action='compile_targets')
            log.info("SM: CheckForRemainingTargets → compiling targets from shopping list")
            try:
                pose_now = self.get_pose_fn()
                rx, ry = float(pose_now[0]), float(pose_now[1])
            except Exception:
                rx = ry = 0.0
            targets: dict[str, Tuple[float, float]] = {}
            shopping = list(self.shopping_list or [])
            clusters = getattr(self.actions, 'current_obj_positions', []) or []
            for fruit in shopping:
                # Filter clusters for this class
                candidates = [c for c in clusters if str(c.get('class')) == str(fruit) and isinstance(c.get('position'), (list, tuple)) and len(c.get('position')) >= 2]
                if not candidates:
                    log.info(f"Skipping {fruit}")
                    continue
                # Pick nearest to current position
                best = None
                best_d2 = float('inf')
                for c in candidates:
                    wx, wy = float(c['position'][0]), float(c['position'][1])
                    d2 = (wx - rx) ** 2 + (wy - ry) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (wx, wy)
                if best is not None:
                    targets[fruit] = best
                    log.info("Target %s at (%.2f, %.2f)", fruit, best[0], best[1])
            # Build and publish status
            self._remaining_targets = targets
            order = list(self.shopping_list or [])
            collected = sorted(list(self._collected_targets))
            seen_not_collected = sorted(list(self._remaining_targets.keys()))
            unseen = [f for f in order if (f not in self._remaining_targets) and (f not in self._collected_targets)]

            # Logging summary
            log.info("Target order: %s", ", ".join(order) if order else "<none>")
            log.info("Remaining targets with positions: %s", ", ".join([f"{k}@({v[0]:.2f},{v[1]:.2f})" for k, v in self._remaining_targets.items()]) if self._remaining_targets else "<none>")
            log.info("Collected: %s", ", ".join(collected) if collected else "<none>")
            log.info("Seen (not collected): %s", ", ".join(seen_not_collected) if seen_not_collected else "<none>")
            log.info("Unseen: %s", ", ".join(unseen) if unseen else "<none>")

            try:
                self.world.set_targets_info(order=order,
                                            remaining=self._remaining_targets,
                                            collected=collected,
                                            seen_not_collected=seen_not_collected,
                                            unseen=unseen)
            except Exception:
                pass
        else:
            # Other states not yet implemented
            self.world.set_status(mode='AUTO', sm_state=sm_state, action='idle')

    def run(self):
        while not self._stop.is_set():
            t0 = time.time()
            # 1) intents
            self._handle_intents()
            # 2) pose
            pose = self.get_pose_fn()
            self.world.set_pose(pose)
            # 3) control
            if self.mode == 'MANUAL_WAYPOINTS':
                if self._plan_waypoints:
                    # Robust replanning
                    self._maybe_replan(pose)
                    self._drive_step(pose)
            elif self.mode == 'AUTO':
                self._auto_step(pose)
            # 4) sleep remainder
            dt = time.time() - t0
            if dt < self._period:
                time.sleep(self._period - dt)
        self.cmd.stop()
