from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grid_map import GridMap
import logging

log = logging.getLogger(__name__)

Coord = Tuple[int, int]   # (row, col)
Point = Tuple[float, float]  # (x, y)


@dataclass
class PlanResult:
    path_grid: List[Coord]
    path_world: List[Point]
    pruned_world: List[Point]
    cost: float
    planned_at: float


class AStarPlanner:
    """A* planner over GridMap with 8-connectivity and Euclidean heuristic.

    - Occupied if grid value > occ_th (treat any value > occ_th as blocked).
    - Optional clearance-aware costs to bias paths away from obstacles.
    - Returns both raw grid path and pruned world waypoints (line-of-sight pruning).
    """

    def __init__(self, occupancy_threshold: int = 0,
                 clearance_weight: float = 0.0,
                 clearance_epsilon: float = 0.02,
                 clearance_power: float = 1.0,
                 min_prune_clearance: float = 0.0,
                 clearance_from_static: bool = False):
        """
        Parameters
        - occupancy_threshold: cells with value > occ_th are blocked
        - clearance_weight: 0.0 disables clearance bias (original behavior).
          Positive values increase the penalty for low-clearance cells.
        - clearance_epsilon: metres used to bound the inverse-clearance penalty.
        - clearance_power: exponent on the inverse-clearance term; >1 sharpens
          the penalty near obstacles.
        - min_prune_clearance: metres required along each pruned LOS segment.
          0.0 disables clearance checks during pruning.
        """
        self.occ_th = int(occupancy_threshold)
        self.clear_w = float(clearance_weight)
        self.clear_eps = float(clearance_epsilon)
        self.clear_pow = float(clearance_power)
        self.min_prune_clear = float(min_prune_clearance)
        self.clear_from_static = bool(clearance_from_static)

    # --------------- Core A* ---------------
    @staticmethod
    def _neighbors(H: int, W: int, p: Coord) -> List[Tuple[Coord, float]]:
        r, c = p
        nbrs: List[Tuple[Coord, float]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    nbrs.append(((rr, cc), math.hypot(dr, dc)))  # 1.0 or sqrt(2)
        return nbrs

    @staticmethod
    def _heuristic(a: Coord, b: Coord) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _reconstruct(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _is_free(self, occ: np.ndarray, p: Coord) -> bool:
        return int(occ[p[0], p[1]]) <= self.occ_th

    def plan_grid(self, grid: GridMap, start_xy: Point, goal_xy: Point) -> Optional[List[Coord]]:
        """Run A* on the combined occupancy (0 free, >occ_th blocked)."""
        log.debug("A* plan_grid start=%s goal=%s", str(start_xy), str(goal_xy))
        occ = grid.combined()
        if occ.ndim != 2:
            raise ValueError("Occupancy grid must be 2D")
        H, W = occ.shape

        s = grid.world_to_grid(*start_xy)
        g = grid.world_to_grid(*goal_xy)

        # Optional clearance map (metres); None when disabled
        clearance = None
        if self.clear_w > 0.0:
            try:
                clearance = grid.clearance_map_static() if self.clear_from_static else grid.clearance_map()
            except Exception:
                clearance = None  # fail-safe to original costs

        # Basic validity checks
        if not (0 <= s[0] < H and 0 <= s[1] < W and 0 <= g[0] < H and 0 <= g[1] < W):
            log.warning("A*: start or goal out of bounds s=%s g=%s size=(%d,%d)", str(s), str(g), H, W)
            return None
        # Allow planning when starting inside an exclusion/occupied zone; only require goal to be free.
        if not self._is_free(occ, g):
            log.info("A*: goal cell blocked; cannot plan. s_free=%s g_free=%s", self._is_free(occ, s), self._is_free(occ, g))
            return None
        if not self._is_free(occ, s):
            log.info("A*: starting inside occupied/exclusion cell; attempting to escape via free neighbors.")

        # Standard A*: allow duplicates in heap, discard stale entries on pop.
        open_heap: List[Tuple[float, float, Coord]] = []  # (f, g, node)
        came_from: Dict[Coord, Coord] = {}
        g_score = {s: 0.0}

        heapq.heappush(open_heap, (self._heuristic(s, g), 0.0, s))

        closed: set[Coord] = set()

        while open_heap:
            f_curr, g_curr, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            # Stale entry check
            if g_curr > g_score.get(current, float("inf")):
                continue

            if current == g:
                path = self._reconstruct(came_from, current)
                log.debug("A*: goal reached; path_len=%d", len(path))
                return path

            closed.add(current)

            for nb, step_cost in self._neighbors(H, W, current):
                if not self._is_free(occ, nb) or nb in closed:
                    continue
                # Optionally scale step cost based on clearance at neighbor cell
                step = step_cost
                if clearance is not None:
                    try:
                        cl = float(clearance[nb[0], nb[1]])
                        inv = 1.0 / max(self.clear_eps, cl)
                        penalty = 1.0 + self.clear_w * (inv ** max(1.0, self.clear_pow))
                        step *= penalty
                    except Exception:
                        pass
                tentative = g_curr + step
                if tentative < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    # Heuristic remains plain Euclidean for admissibility
                    f = tentative + self._heuristic(nb, g)
                    heapq.heappush(open_heap, (f, tentative, nb))

        log.info("A*: no path found")
        return None  # no path

    # --------------- LOS & Pruning ---------------
    @staticmethod
    def _supercover_line(p0: Coord, p1: Coord) -> List[Coord]:
        """Supercover line: visits all grid cells touched by the segment."""
        r0, c0 = p0
        r1, c1 = p1
        dr = r1 - r0
        dc = c1 - c0
        sr = 1 if dr > 0 else -1 if dr < 0 else 0
        sc = 1 if dc > 0 else -1 if dc < 0 else 0
        dr = abs(dr)
        dc = abs(dc)
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

    def line_free(self, occ: np.ndarray, a: Coord, b: Coord) -> bool:
        H, W = occ.shape
        for r, c in self._supercover_line(a, b):
            if r < 0 or r >= H or c < 0 or c >= W:
                return False
            if not self._is_free(occ, (r, c)):
                return False
        return True

    def prune_world(self, grid: GridMap, path_grid: List[Coord]) -> List[Point]:
        if not path_grid:
            return []
        log.debug("Pruning path of length %d", len(path_grid))
        occ = grid.combined()
        # Clearance map optionally used to ensure minimum clearance on LOS segments
        clr = None
        if self.min_prune_clear > 0.0:
            try:
                clr = grid.clearance_map()
            except Exception:
                clr = None
        pruned: List[Coord] = [path_grid[0]]
        anchor = path_grid[0]
        for i in range(1, len(path_grid)):
            nxt = path_grid[i]
            los_ok = self.line_free(occ, anchor, nxt)
            if los_ok and clr is not None:
                # Enforce minimum clearance along the supercover line
                for r, c in self._supercover_line(anchor, nxt):
                    if float(clr[r, c]) < self.min_prune_clear:
                        los_ok = False
                        break
            if not los_ok:
                prev = path_grid[i - 1]
                if prev != pruned[-1]:
                    pruned.append(prev)
                anchor = prev
        if pruned[-1] != path_grid[-1]:
            pruned.append(path_grid[-1])
        return [grid.grid_to_world(r, c) for (r, c) in pruned]

    # --------------- External API ---------------
    def plan(self, grid: GridMap, start_xy: Point, goal_xy: Point) -> Optional[PlanResult]:
        t0 = time.time()
        path_grid = self.plan_grid(grid, start_xy, goal_xy)
        if path_grid is None:
            return None
        path_world = [grid.grid_to_world(r, c) for (r, c) in path_grid]

        # Compute true cost along the grid path
        cost = 0.0
        for (r0, c0), (r1, c1) in zip(path_grid, path_grid[1:]):
            cost += math.hypot(r1 - r0, c1 - c0)

        pruned = self.prune_world(grid, path_grid)
        pr = PlanResult(
            path_grid=path_grid,
            path_world=path_world,
            pruned_world=pruned,
            cost=cost,
            planned_at=time.time(),
        )
        log.info("A* plan complete: nodes=%d pruned=%d cost=%.2f dt=%.3fs", len(path_grid), len(pruned), cost, pr.planned_at - t0)
        return pr

    # --------------- Replanning helpers ---------------
    @staticmethod
    def cross_track_error(pose_xy: Point, path_world: List[Point]) -> float:
        if not path_world:
            return float("inf")
        px, py = pose_xy
        min_d = float("inf")
        for i in range(len(path_world) - 1):
            x1, y1 = path_world[i]
            x2, y2 = path_world[i + 1]
            vx, vy = x2 - x1, y2 - y1
            wx, wy = px - x1, py - y1
            seg_len2 = vx * vx + vy * vy + 1e-9
            t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
            projx, projy = x1 + t * vx, y1 + t * vy
            d = math.hypot(px - projx, py - projy)
            if d < min_d:
                min_d = d
        return min_d

    def path_intersects_obstacles(self, grid: GridMap, path_grid: List[Coord]) -> bool:
        if not path_grid:
            return False
        occ = grid.combined()
        return any(not self._is_free(occ, rc) for rc in path_grid)

    @staticmethod
    def time_to_replan(last_plan_time: float, period_s: float = 2.5) -> bool:
        return (time.time() - last_plan_time) >= period_s
