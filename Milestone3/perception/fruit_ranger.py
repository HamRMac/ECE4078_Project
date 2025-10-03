"""
Fruit range/bearing estimation utilities and helpers.

Contains:
- FruitRanger: estimate (r, theta) from bbox height, fuse multiple measurements.
- Helper functions for aspect-ratio sanity checks and arena bounds checks.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np


def is_inside_arena(x: float, y: float, bound: float) -> bool:
    x = float(x); y = float(y)
    return (-bound <= x <= bound) and (-bound <= y <= bound)


def expected_ratio_for_class(target_dimensions_dict: Dict[str, Tuple[float, float, float]],
                             target_class: str) -> float:
    """Expected bbox aspect ratio ≈ physical width/height."""
    w, _, h = target_dimensions_dict[target_class]
    return max(1e-9, w / h)


def bbox_ratio_ok(target_class: str,
                  bbox: List[float] | Tuple[float, float, float, float],
                  target_dimensions_dict: Dict[str, Tuple[float, float, float]],
                  tol: float = 0.15) -> bool:
    """Enforce |(w/h) - expected| / expected ≤ tol for a detection bbox."""
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return False
    ratio_px = w / h
    expected = expected_ratio_for_class(target_dimensions_dict, target_class)
    return abs(ratio_px - expected) / expected <= tol


class FruitRanger:
    """Compute range/bearing (and uncertainties) for fruit detections.
    Provides a height-based prior method and a stub for ground-ray back-projection.
    Also fuses multiple (r,theta) measurements into a single (r*,theta*) with 2×2 covariance.
    """
    def __init__(self,
                 pixel_centroid_sigma_px: float = 2.0,
                 pixel_height_sigma_px: float = 3.0,
                 range_scale_beta: float = 0.02,
                 ekf_weight_gamma: float = 1.0,
                 camera_matrix: np.ndarray | None = None,
                 camera_height_m: float = 0.041,
                 camera_pitch_rad: float = np.deg2rad(-5.0)
                 ) -> None:
        self.pixel_centroid_sigma_px = float(pixel_centroid_sigma_px)
        self.pixel_height_sigma_px = float(pixel_height_sigma_px)
        self.range_scale_beta = float(range_scale_beta)
        self.ekf_weight_gamma = float(ekf_weight_gamma)
        self.camera_matrix = camera_matrix
        self.camera_height_m = camera_height_m
        self.camera_pitch_rad = camera_pitch_rad

        print("FruitRanger initialised with camera height %.3f m, pitch %.3f rad" % (self.camera_height_m, self.camera_pitch_rad))

    def from_bbox_height(self,
                         bbox: List[float] | Tuple[float, float, float, float],
                         true_height_m: float) -> Optional[Dict[str, float]]:
        """Estimate r, theta and uncertainties from bbox height.
        bbox is [x, y, w, h] in pixels (top-left origin).
        Returns: {'r','theta','sigma_r','sigma_theta','x','y'} in camera frame.
        """
        if self.camera_matrix is None or true_height_m <= 0:
            return None
        try:
            x, y, w, h = [float(v) for v in bbox]
        except Exception:
            return None
        if h <= 0:
            return None

        f = float(self.camera_matrix[0, 0])
        cx = float(self.camera_matrix[0, 2]) if self.camera_matrix.shape[1] >= 3 else 160.0

        # Similar triangles range estimate along optical axis
        r = (true_height_m / h) * f

        # Bearing in camera frame; positive when bbox centre is left of principal point
        x_c = x + w / 2.0
        theta = float(np.arctan2(cx - x_c, f))

        # Uncertainty propagation
        dr_dh = abs((-f * true_height_m) / (h * h))
        sigma_r_from_h = dr_dh * self.pixel_height_sigma_px
        sigma_r = float(np.sqrt(sigma_r_from_h**2 + (self.range_scale_beta * r * r)))
        sigma_theta = float(self.pixel_centroid_sigma_px / f)

        # Cartesian in camera frame (x forward, y left)
        x_cam = float(r * np.cos(theta))
        y_cam = float(r * np.sin(theta))

        return {
            'r': r, 'theta': theta,
            'sigma_r': sigma_r, 'sigma_theta': sigma_theta,
            'x': x_cam, 'y': y_cam
        }

    def from_ground_ray(self,
                    bbox: List[float] | Tuple[float, float, float, float]) -> Optional[Dict[str, float]]:
        """Estimate r, theta using ground ray back projection from the bbox bottom.

        Assumptions:
        - Camera at height H above the ground, pitched down by camera_pitch_rad (radians).
        - Camera frame: x forward, y left. Positive theta when target is to the left.
        - Ground plane at z=0 with camera optical centre at z=H.

        Returns dict like from_bbox_height: {'r','theta','sigma_r','sigma_theta','x','y'}.
        """
        if self.camera_matrix is None or self.camera_height_m <= 0:
            return None

        try:
            x, y, w, h = [float(v) for v in bbox]
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None

        f = float(self.camera_matrix[0, 0])
        cx = float(self.camera_matrix[0, 2]) if self.camera_matrix.shape[1] >= 3 else 160.0
        cy = float(self.camera_matrix[1, 2]) if self.camera_matrix.shape[0] >= 3 else 120.0

        # Bottom midpoint pixel
        xb = x + 0.5 * w
        yb = y + h

        # Horizontal bearing (left positive)
        theta = float(np.arctan2(cx - xb, f))

        # Vertical offset from optical axis (downward positive since image y increases downward)
        v = (yb - cy) / f
        alpha = float(np.arctan(v))

        # Total depression angle from horizontal
        gamma = float(alpha + float(self.camera_pitch_rad))

        # Guard against rays parallel or upward relative to ground plane
        t = np.tan(gamma)
        if t <= 1e-9:
            return None

        # Range along ground
        r = float(self.camera_height_m / t)

        # Cartesian in camera frame
        x_cam = float(r * np.cos(theta))
        y_cam = float(r * np.sin(theta))

        # Uncertainties
        sigma_px = float(self.pixel_centroid_sigma_px)

        # Bearing uncertainty from horizontal pixel uncertainty
        sigma_theta = float(sigma_px / f)

        # Range uncertainty via propagation: dr/dgamma * dgamma/dyb * sigma_px
        # dr/dgamma = -H / sin^2(gamma)
        sin_g = np.sin(gamma)
        sin_g2 = max(sin_g * sin_g, 1e-12)
        dr_dgamma = -self.camera_height_m / sin_g2

        # d alpha / d yb = 1 / (f * (1 + v^2))
        dalpha_dyb = 1.0 / (f * (1.0 + v * v))

        # d gamma / d yb = d alpha / d yb
        dgamma_dyb = dalpha_dyb

        sigma_r_from_y = abs(dr_dgamma * dgamma_dyb) * sigma_px

        # Add scale term
        sigma_r = float(np.sqrt(sigma_r_from_y ** 2 + (self.range_scale_beta * r * r)))

        return {
            'r': r, 'theta': float(theta),
            'sigma_r': sigma_r, 'sigma_theta': sigma_theta,
            'x': x_cam, 'y': y_cam
        }


    def fuse(self,
             measurements: List[Dict[str, float]],
             ekf_pose_var: float = 0.0) -> Optional[Dict[str, object]]:
        """Fuse multiple measurements (r,theta) into (r*,theta*) with covariance in (x,y)."""
        if not measurements:
            return None

        xs, ys, ws = [], [], []
        for m in measurements:
            r = float(m['r'])
            th = float(m['theta'])
            sr = float(m['sigma_r'])
            st = float(m['sigma_theta'])

            x = r * np.cos(th)
            y = r * np.sin(th)

            # Variance proxy in x,y
            var_x = (np.cos(th) ** 2) * (sr ** 2) + (r * np.sin(th)) ** 2 * (st ** 2)
            var_y = (np.sin(th) ** 2) * (sr ** 2) + (r * np.cos(th)) ** 2 * (st ** 2)
            var_xy = var_x + var_y + self.ekf_weight_gamma * float(ekf_pose_var)

            w = 1.0 / max(var_xy, 1e-12)
            xs.append(x); ys.append(y); ws.append(w)

        ws = np.asarray(ws, dtype=float)
        if ws.sum() <= 0:
            return None
        wnorm = ws / ws.sum()
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        x_mean = float((wnorm * xs).sum())
        y_mean = float((wnorm * ys).sum())

        # Weighted sample covariance
        dx = xs - x_mean
        dy = ys - y_mean
        cov_xx = float((wnorm * dx * dx).sum())
        cov_xy = float((wnorm * dx * dy).sum())
        cov_yy = float((wnorm * dy * dy).sum())
        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)

        r_star = float(np.hypot(x_mean, y_mean))
        th_star = float(np.arctan2(y_mean, x_mean))
        return {'r': r_star, 'theta': th_star, 'x': x_mean, 'y': y_mean, 'cov': cov}

