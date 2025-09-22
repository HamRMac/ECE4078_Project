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
from state_machine.state_machine import PiBotFruitSearchSM
from pibot_actions import PiBotActions

log = logging.getLogger(__name__)


class Runner(threading.Thread):
    """Mission runner: consumes intents, updates pose, plans, and drives the robot."""

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
                 state_machine: PiBotFruitSearchSM = None,
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
        pr = self.planner.plan(self.grid, (pose[0], pose[1]), self._goal)
        if pr is None:
            self._plan_waypoints = []
            self.world.clear_plan()
            self.world.set_status(action='plan_failed')
            log.warning("Plan failed to goal (%.2f, %.2f)", self._goal[0], self._goal[1])
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
        # EXECUTES IN STATE "ApproachFruit"
        elif sm_state == 'approach_fruit':
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
                # Fallback to map center
                try:
                    bx0, by0, bx1, by1 = self.grid.bounds_wm  # type: ignore
                    chosen_goal = ((bx0 + bx1) * 0.5, (by0 + by1) * 0.5)
                except Exception:
                    chosen_goal = (0.0, 0.0)
                self._goal = (float(chosen_goal[0]), float(chosen_goal[1]))
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
