import math
import logging
from typing import Tuple

import numpy as np

from warnings import warn


log = logging.getLogger(__name__)

class BaseController:
    def __init__(self,
                 pos_tol: float = 0.05,
                 ang_tol: float = 0.2,
                 max_forward_tick: int = 80,
                 min_forward_tick: int = 20,
                 max_turn_tick: int = 25,
                 min_turn_tick: int = 10):
        self.pos_tol = pos_tol
        self.ang_tol = ang_tol
        self.max_forward_tick = max_forward_tick
        self.min_forward_tick = min_forward_tick
        self.max_turn_tick = max_turn_tick
        self.min_turn_tick = min_turn_tick

    @staticmethod
    def _wrap_pi(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def compute(self, pose: Tuple[float, float, float], goal: Tuple[float, float]) -> Tuple[int, int, int, int, bool]:
        raise NotImplementedError


class TurnThenGoController(BaseController):
    """Two-phase controller: rotate to heading, then drive straight."""

    def compute(self, pose, goal):
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        gx, gy = float(goal[0]), float(goal[1])
        dx, dy = gx - x, gy - y
        dist = float(math.hypot(dx, dy))
        if dist <= self.pos_tol:
            return 0, 0, 0, 0, True

        desired = float(math.atan2(dy, dx))
        herr = self._wrap_pi(desired - th)

        # Rotate in place if misaligned
        if abs(herr) > self.ang_tol:
            turn_dir = 1 if herr > 0 else -1
            turn_mag = min(1.0, abs(herr) / 0.8)
            turn_tick = int(min(self.max_turn_tick, max(self.min_turn_tick, turn_mag * self.max_turn_tick)))
            log.debug("TTG rotate: herr=%.3f dir=%d tick=%d", herr, turn_dir, turn_tick)
            return 0, turn_dir, 0, turn_tick, False

        # Drive forward if aligned
        allowed_speeds = range(self.min_forward_tick, self.max_forward_tick + 1, 5)
        fwd_mag = min(1.0, dist / 0.5)
        fwd_tick = int(min(self.max_forward_tick, max(self.min_forward_tick, fwd_mag * self.max_forward_tick)))

        round_speed = None
        for speed in allowed_speeds:
            if round_speed == None or abs(speed-fwd_tick) < round_speed:
                round_speed = speed
            if abs(speed - fwd_tick) > round_speed:
                break

        fwd_tick = round_speed

        log.debug("TTG forward: dist=%.3f tick=%d", dist, fwd_tick)
        return 1, 0, fwd_tick, 0, False


class PurePursuitController(BaseController):
    """Minimal pure-pursuit-like controller for a single waypoint.

    Uses heading error to command simultaneous forward and turning for smoother approach.
    """

    def __init__(self, lookahead: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.lookahead = lookahead

    def compute(self, pose, goal):
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        gx, gy = float(goal[0]), float(goal[1])
        dx, dy = gx - x, gy - y
        dist = float(math.hypot(dx, dy))
        if dist <= self.pos_tol:
            log.debug("PPC done: dist=%.3f tol=%.3f", dist, self.pos_tol)
            return 0, 0, 0, 0, True

        # Body-frame coordinates of goal (approximate lookahead behavior)
        s, c = math.sin(th), math.cos(th)
        x_b = c * dx + s * dy
        y_b = -s * dx + c * dy

        # Choose lookahead point along the line towards the goal
        Ld = max(self.lookahead, min(dist, 2 * self.lookahead))
        # Approximate curvature to the point (single waypoint case)
        kappa = (2.0 * y_b) / (Ld * Ld + 1e-6)

        # Map curvature to turning command direction and magnitude
        turn_dir = 0
        turn_tick = 0
        if abs(kappa) > 1e-3:
            turn_dir = 1 if kappa > 0 else -1
            # Heuristic mapping from curvature to turning ticks
            turn_tick = int(min(self.max_turn_tick, max(self.min_turn_tick, abs(kappa) * 200)))

        # Forward command reduced with curvature magnitude
        fwd_scale = max(0.2, 1.0 - min(1.0, abs(kappa)))
        fwd_tick = int(min(self.max_forward_tick, max(self.min_forward_tick, fwd_scale * self.max_forward_tick)))

        log.debug("PPC step: kappa=%.3f turn_dir=%d turn_tick=%d fwd_tick=%d", kappa, turn_dir, turn_tick, fwd_tick)
        return 1, turn_dir, fwd_tick, turn_tick, False


class RHPController(BaseController):
    """Stub of a receding horizon controller; uses a PPC-like fallback for now."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ppc = PurePursuitController(**kwargs)
        warn("RHPController is a stub and uses PurePursuitController as fallback.")

    def compute(self, pose, goal):
        # TODO: implement sampling-based short-horizon optimization over controls
        return self._ppc.compute(pose, goal)


class ControllerManager:
    def __init__(self, kind: str = "ttg"):
        kind = (kind or "ttg").lower()
        if kind == "ttg":
            self.ctrl = TurnThenGoController()
        elif kind == "ppc":
            self.ctrl = PurePursuitController()
        elif kind == "rhp":
            self.ctrl = RHPController()
        else:
            raise ValueError(f"Unknown controller kind: {kind}")

    def compute(self, pose, goal):
        return self.ctrl.compute(pose, goal)
