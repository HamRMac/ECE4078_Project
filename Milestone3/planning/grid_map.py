import math
from typing import Tuple, Optional

import cv2
import numpy as np


class GridMap:
    """
    Occupancy Grid builder using ArUco markers and/or provided arena bounds.

    Assumptions:
    - If arena bounds are not provided, they are inferred as the rectangle covering all ArUco
      positions with a configurable margin.
    - ArUco obstacles are modelled as discs enclosing the marker cube footprint. The smallest
      enclosing disc for a square of side s has radius s / sqrt(2). We then inflate by
      (robot_radius + inflation_margin) for conservative planning.
    - Static layer: aruco obstacles + boundary inflation.
    - Dynamic layer: starts empty; intended for online obstacles later.
    """

    def __init__(
        self,
        res: float = 0.05,                # Grid cell size in metres
        margin: float = 0.4,              # Margin around inferred arena bounds
        robot_radius: float = 0.09,       # Robot radius in metres
        inflation_margin: float = 0.05,   # Extra inflation buffer in metres
        boundary_margin: float = 0.1,     # NEW: extra margin for arena edges
        arena_bounds_wm: Optional[Tuple[float, float, float, float]] = None,  # (minx, miny, maxx, maxy)
    ):
        if res <= 0:
            raise ValueError("Resolution 'res' must be positive.")
        if robot_radius < 0 or inflation_margin < 0 or boundary_margin < 0 or margin < 0:
            raise ValueError("Margins and radii must be non-negative.")

        self.res = float(res)
        self.margin = float(margin)
        self.robot_radius = float(robot_radius)
        self.inflation_margin = float(inflation_margin)
        self.boundary_margin = float(boundary_margin)

        # World bounds (minx, miny, maxx, maxy)
        self.bounds_wm: Optional[Tuple[float, float, float, float]] = arena_bounds_wm
        # Grid origin (world metres) and size (H, W)
        self.origin_wm: Optional[Tuple[float, float]] = None
        self.size: Optional[Tuple[int, int]] = None

        # Layers: 0 free, 255 occupied
        self.static_layer: Optional[np.ndarray] = None
        self.dynamic_layer: Optional[np.ndarray] = None

    # ---------------- Coordinates ----------------
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Map world metres to grid indices (row, col), clamped to image bounds.
        Uses floor so a point maps to the cell that contains it.
        """
        assert self.origin_wm is not None and self.size is not None, "Grid not built yet."
        ox, oy = self.origin_wm
        H, W = self.size

        c = int(math.floor((x - ox) / self.res))
        r = int(math.floor((y - oy) / self.res))

        # Clamp to valid index range
        c = 0 if c < 0 else (W - 1 if c >= W else c)
        r = 0 if r < 0 else (H - 1 if r >= H else r)
        return r, c

    def grid_to_world(self, r: int, c: int) -> Tuple[float, float]:
        """
        Map grid indices (row, col) to world metres at the centre of the cell.
        """
        assert self.origin_wm is not None and self.size is not None, "Grid not built yet."
        ox, oy = self.origin_wm
        x = ox + (c + 0.5) * self.res
        y = oy + (r + 0.5) * self.res
        return x, y

    # ---------------- Build ----------------
    def build_from_aruco(self, aruco_positions: np.ndarray, aruco_cube_size: float = 0.07):
        """
        Build static occupancy from ArUco positions and inflated boundary.
        aruco_positions: (N, 2) ndarray of [x, y] in metres.
        If arena bounds were provided at init, they are used as-is (no margin).
        Otherwise, bounds are inferred from markers with an added margin.
        """
        if aruco_positions.ndim != 2 or aruco_positions.shape[1] != 2:
            raise ValueError("aruco_positions must be an array of shape (N, 2).")

        if self.bounds_wm is None:
            if aruco_positions.size == 0:
                raise ValueError("aruco_positions is empty and arena bounds were not provided.")
            xs = aruco_positions[:, 0]
            ys = aruco_positions[:, 1]
            minx, maxx = float(xs.min()), float(xs.max())
            miny, maxy = float(ys.min()), float(ys.max())
            # Bounds with margin inferred from markers
            bx0 = minx - self.margin
            by0 = miny - self.margin
            bx1 = maxx + self.margin
            by1 = maxy + self.margin
            self.bounds_wm = (bx0, by0, bx1, by1)

        bx0, by0, bx1, by1 = self.bounds_wm
        if not (bx1 > bx0 and by1 > by0):
            raise ValueError("Invalid arena bounds: ensure (minx, miny) < (maxx, maxy).")
        self.origin_wm = (bx0, by0)

        # Grid size over half-open domain [bx0, bx1) x [by0, by1)
        W = int(math.ceil((bx1 - bx0) / self.res))
        H = int(math.ceil((by1 - by0) / self.res))
        if W <= 0 or H <= 0:
            raise ValueError("Computed grid size is non-positive. Check bounds and resolution.")
        self.size = (H, W)

        self.static_layer = np.zeros((H, W), dtype=np.uint8)
        self.dynamic_layer = np.zeros((H, W), dtype=np.uint8)

        # Inflate radius in metres: enclosing disc for marker square + robot + margin
        # base_r = s / sqrt(2) = 0.5 * sqrt(2) * s
        base_r = 0.5 * math.sqrt(2.0) * float(aruco_cube_size)
        inflate_r = base_r + self.robot_radius + self.inflation_margin
        r_cells = max(1, int(math.ceil(inflate_r / self.res)))

        # Draw aruco discs (skip if no markers provided)
        for (x, y) in aruco_positions:
            r, c = self.world_to_grid(float(x), float(y))
            cv2.circle(self.static_layer, (c, r), r_cells, color=255, thickness=-1)

        # Boundary inflation: mark a thick border as occupied
        b_cells = max(1, int(math.ceil(self.boundary_margin / self.res)))
        self.static_layer[0:b_cells, :] = 255
        self.static_layer[-b_cells:, :] = 255
        self.static_layer[:, 0:b_cells] = 255
        self.static_layer[:, -b_cells:] = 255

    # ---------------- Update & Query ----------------
    def add_dynamic_obstacle(self, x: float, y: float, radius: float):
        assert self.dynamic_layer is not None, "Grid not built yet."
        if radius < 0:
            raise ValueError("Obstacle radius must be non-negative.")
        r, c = self.world_to_grid(x, y)
        rc = max(1, int(math.ceil(radius / self.res)))
        cv2.circle(self.dynamic_layer, (c, r), rc, color=255, thickness=-1)

    def combined(self) -> np.ndarray:
        assert self.static_layer is not None and self.dynamic_layer is not None, "Grid not built yet."
        return np.maximum(self.static_layer, self.dynamic_layer)

    # ---------------- Visualisation ----------------
    def render(self, scale: int = 3) -> np.ndarray:
        """Return a BGR image for visualisation of the occupancy grid."""
        occ = self.combined()
        vis = cv2.cvtColor(255 - occ, cv2.COLOR_GRAY2BGR)
        if scale != 1:
            H, W = vis.shape[:2]
            vis = cv2.resize(vis, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
        # Draw metadata text
        cv2.putText(
            vis,
            f"res={self.res:.3f}m, margin={self.margin:.3f}m, robot_radius={self.robot_radius:.3f}m, inflation_margin={self.inflation_margin:.3f}m, bounds={self.bounds_wm}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )
        return vis
