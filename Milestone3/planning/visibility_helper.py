import math
from typing import List, Tuple, Optional

from .grid_map import GridMap

import numpy as np


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _line_intersects_circle(p0, p1, center, radius) -> bool:
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    c = np.array(center, dtype=float)
    d = p1 - p0
    f = p0 - c
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(f, d))
    cval = float(np.dot(f, f) - radius * radius)
    disc = b * b - 4.0 * a * cval
    if disc < 0:
        return False
    sd = math.sqrt(disc)
    t1 = (-b - sd) / (2.0 * a)
    t2 = (-b + sd) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def _segments_intersect(A, B, C, D) -> bool:
    def ccw(X, Y, Z):
        return (Z[1] - X[1]) * (Y[0] - X[0]) > (Y[1] - X[1]) * (Z[0] - X[0])
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def _line_intersects_square(p0, p1, center, half) -> bool:
    cx, cy = center
    x_min, x_max = cx - half, cx + half
    y_min, y_max = cy - half, cy + half
    edges = [((x_min, y_min), (x_max, y_min)),
             ((x_max, y_min), (x_max, y_max)),
             ((x_max, y_max), (x_min, y_max)),
             ((x_min, y_max), (x_min, y_min))]
    return any(_segments_intersect(p0, p1, e0, e1) for e0, e1 in edges)


def compute_safety_mask(grid: GridMap,
                        robot_pose: List[float],
                        aruco_positions: np.ndarray,
                        fruit_positions: List[Tuple[float, float]],
                        marker_length: float = 0.07,
                        fruit_radius: float = 0.05,
                        fov_deg: float = 360.0,
                        max_distance: float = 1.2,
                        step_cells: int = 1) -> np.ndarray:
    """Compute a grid-aligned safety mask (True = observed free) near the robot.

    - Considers LOS from robot to grid cell centers within range/FOV
    - Obstacles: ArUco squares (half=marker_length/2), fruits as circles (fruit_radius)
    - step_cells: subsample factor over grid (2 means every 2nd cell evaluated)
    """
    assert grid.size is not None and grid.bounds_wm is not None
    H, W = grid.size
    rx, ry, rth = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    fov_h = float(fov_deg) / 2.0
    safe = np.zeros((H, W), dtype=bool)

    # Evaluate subset of cells for performance
    step = max(1, int(step_cells))
    rows = range(0, H, step)
    cols = range(0, W, step)

    for r in rows:
        for c in cols:
            x, y = grid.grid_to_world(r, c)
            dx, dy = x - rx, y - ry
            dist = math.hypot(dx, dy)
            if dist > max_distance:
                continue
            ang = math.degrees(_wrap_pi(math.atan2(dy, dx) - rth))
            if abs(ang) > fov_h:
                continue
            # Assume visible; check occlusion
            visible = True
            p0 = (rx, ry); p1 = (x, y)
            # Squares (aruco)
            if aruco_positions is not None and aruco_positions.size > 0:
                half = marker_length / 2.0
                for ax, ay in aruco_positions:
                    if _line_intersects_square(p0, p1, (float(ax), float(ay)), half):
                        visible = False
                        break
            if not visible:
                continue
            # Circles (fruit)
            for (fx, fy) in (fruit_positions or []):
                if _line_intersects_circle(p0, p1, (float(fx), float(fy)), fruit_radius):
                    visible = False
                    break
            if visible:
                # Mark sampled cell safe
                if step == 1:
                    safe[r, c] = True
                else:
                    # Block-fill the step x step neighborhood starting at (r,c)
                    r1 = min(H, r + step)
                    c1 = min(W, c + step)
                    safe[r:r1, c:c1] = True

    # Simple erosion near boundaries could be added, but keep minimal now.
    return safe
