import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

import numpy as np

from .grid_map import GridMap


@dataclass
class SectorInfo:
    idx: Tuple[int, int]
    bounds_xy: Tuple[float, float, float, float]  # (x0, x1, y0, y1)
    center_xy: Tuple[float, float]
    dark_fraction: float
    free_cells: int
    total_cells: int


class SectorExplorer:
    """Partitions the arena bounds into a fixed grid of sectors and
    computes a safe point per sector using the current occupancy.

    Darkness fraction = proportion of non-free cells in sector using GridMap.combined().
    Safe point = nearest free cell to sector center with minimum clearance (if available).
    """

    def __init__(self, rows: int = 3, cols: int = 3, min_clearance_m: float = 0.10):
        self.rows = max(1, int(rows))
        self.cols = max(1, int(cols))
        self.min_clearance = float(min_clearance_m)

    def _sector_edges(self, grid: GridMap) -> Tuple[np.ndarray, np.ndarray]:
        assert grid.bounds_wm is not None
        bx0, by0, bx1, by1 = grid.bounds_wm
        xs = np.linspace(bx0, bx1, self.cols + 1)
        ys = np.linspace(by0, by1, self.rows + 1)
        return xs, ys

    def xy_to_sector_idx(self, grid: GridMap, x: float, y: float) -> Tuple[int, int]:
        """Map a world (x,y) to sector index (ix, iy), clamped to valid range."""
        xs, ys = self._sector_edges(grid)
        # ix: last edge <= x
        ix = int(np.searchsorted(xs, float(x), side='right') - 1)
        iy = int(np.searchsorted(ys, float(y), side='right') - 1)
        ix = max(0, min(self.cols - 1, ix))
        iy = max(0, min(self.rows - 1, iy))
        return ix, iy

    @staticmethod
    def _xy_to_rc_bounds(grid: GridMap, x0: float, x1: float, y0: float, y1: float) -> Tuple[int, int, int, int]:
        assert grid.bounds_wm is not None and grid.size is not None
        bx0, by0, bx1, by1 = grid.bounds_wm
        H, W = grid.size
        # Convert world intervals to inclusive row/col index ranges
        # Columns grow with +x; rows grow as y decreases from top (by1)
        c0 = int(math.floor((x0 - bx0) / grid.res))
        c1 = int(math.floor((x1 - bx0) / grid.res))
        r0 = int(math.floor((by1 - y1) / grid.res))
        r1 = int(math.floor((by1 - y0) / grid.res))
        if c0 > c1:
            c0, c1 = c1, c0
        if r0 > r1:
            r0, r1 = r1, r0
        c0 = max(0, min(W - 1, c0))
        c1 = max(0, min(W - 1, c1))
        r0 = max(0, min(H - 1, r0))
        r1 = max(0, min(H - 1, r1))
        return r0, r1, c0, c1

    def compute_sector_stats(self, grid: GridMap) -> List[SectorInfo]:
        occ = grid.combined()
        xs, ys = self._sector_edges(grid)
        stats: List[SectorInfo] = []
        for ix in range(self.cols):
            for iy in range(self.rows):
                x0, x1 = float(xs[ix]), float(xs[ix + 1])
                y0, y1 = float(ys[iy]), float(ys[iy + 1])
                r0, r1, c0, c1 = self._xy_to_rc_bounds(grid, x0, x1, y0, y1)
                if r1 < r0 or c1 < c0:
                    total = 0
                    dark = 0
                else:
                    sl = occ[r0 : r1 + 1, c0 : c1 + 1]
                    total = int(sl.size)
                    dark = int(np.count_nonzero(sl))  # non-zero = unknown/occupied
                frac = (dark / total) if total > 0 else 1.0
                cx = (x0 + x1) * 0.5
                cy = (y0 + y1) * 0.5
                free = (total - dark)
                stats.append(
                    SectorInfo(idx=(ix, iy), bounds_xy=(x0, x1, y0, y1), center_xy=(cx, cy), dark_fraction=float(frac), free_cells=int(free), total_cells=int(total))
                )
        return stats

    def _find_safe_point_in_sector(
        self, grid: GridMap, bounds_xy: Tuple[float, float, float, float], center_xy: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        occ = grid.combined()
        clr = None
        try:
            clr = grid.clearance_map()
        except Exception:
            clr = None
        x0, x1, y0, y1 = bounds_xy
        r0, r1, c0, c1 = self._xy_to_rc_bounds(grid, x0, x1, y0, y1)
        if r1 < r0 or c1 < c0:
            return None
        sl_occ = occ[r0 : r1 + 1, c0 : c1 + 1]
        if sl_occ.size == 0:
            return None
        free_mask = (sl_occ == 0)
        if not np.any(free_mask):
            return None
        if clr is not None and self.min_clearance > 0.0:
            sl_clr = clr[r0 : r1 + 1, c0 : c1 + 1]
            free_mask = free_mask & (sl_clr >= float(self.min_clearance))
            if not np.any(free_mask):
                return None
        # Compute distances to center in world coordinates
        idxs = np.argwhere(free_mask)
        if idxs.size == 0:
            return None
        cx, cy = center_xy
        best_d2 = float("inf")
        best_xy: Optional[Tuple[float, float]] = None
        for dr, dc in idxs:
            r = r0 + int(dr)
            c = c0 + int(dc)
            wx, wy = grid.grid_to_world(r, c)
            d2 = (wx - cx) * (wx - cx) + (wy - cy) * (wy - cy)
            if d2 < best_d2:
                best_d2 = d2
                best_xy = (float(wx), float(wy))
        return best_xy

    def pick_next_target(
        self, grid: GridMap, excluded: Optional[Set[Tuple[int, int]]] = None
    ) -> Optional[Tuple[Tuple[float, float], Tuple[int, int], SectorInfo]]:
        excluded = excluded or set()
        stats = self.compute_sector_stats(grid)
        # Prefer lowest darkness (safest) among remaining sectors that have at least some free cells
        stats = [s for s in stats if s.idx not in excluded and s.total_cells > 0 and s.free_cells > 0]
        if not stats:
            return None
        stats.sort(key=lambda s: s.dark_fraction)  # ascending = safest first
        for s in stats:
            pt = self._find_safe_point_in_sector(grid, s.bounds_xy, s.center_xy)
            if pt is not None:
                return pt, s.idx, s
        return None
