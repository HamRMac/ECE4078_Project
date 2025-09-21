import threading
import time
from typing import Optional, Tuple, List

from navigation.controller import ControllerManager
from planning.astar import AStarPlanner
from planning.grid_map import GridMap
from .world_model import WorldModel
from .robot_commander import RobotCommander


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
                 drive_enabled: bool = True):
        super().__init__(daemon=True)
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
        self._goal: Optional[Tuple[float, float]] = None
        self._plan_waypoints: List[Tuple[float, float]] = []
        self._wp_idx: int = 0
        self._period = 1.0 / max(1.0, float(hz))

        # modes: 'IDLE' | 'MANUAL_WAYPOINTS' | 'AUTO'
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
        pr = self.planner.plan(self.grid, (pose[0], pose[1]), self._goal)
        if pr is None:
            self._plan_waypoints = []
            self.world.clear_plan()
            self.world.set_status(action='plan_failed')
            return
        self._plan_waypoints = list(pr.pruned_world if pr.pruned_world else pr.path_world)
        self._wp_idx = 0
        self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
        self.world.set_status(action='drive', progress=f"0/{len(self._plan_waypoints)}")

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
        if done:
            if self._wp_idx < len(self._plan_waypoints) - 1:
                self._wp_idx += 1
                self.world.set_plan(self._plan_waypoints, active_idx=self._wp_idx)
                self.world.set_status(progress=f"{self._wp_idx}/{len(self._plan_waypoints)-1}")
            else:
                self.cmd.stop()
                self.world.set_status(action='arrived')
                self._plan_waypoints = []
                self.world.clear_plan()
                self.mode = 'IDLE'

    def run(self):
        while not self._stop.is_set():
            t0 = time.time()
            # 1) intents
            self._handle_intents()
            # 2) pose
            pose = self.get_pose_fn()
            self.world.set_pose(pose)
            # 3) control
            if self.mode == 'MANUAL_WAYPOINTS' and self._plan_waypoints:
                self._drive_step(pose)
            # 4) sleep remainder
            dt = time.time() - t0
            if dt < self._period:
                time.sleep(self._period - dt)
        self.cmd.stop()
