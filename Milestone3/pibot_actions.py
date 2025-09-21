import time
import math
import logging
from typing import Optional, Callable, Dict, Any, List, Tuple

import numpy as np

# Access the robot HTTP API wrapper
import sys, os
sys.path.insert(0, f"{os.getcwd()}/util")
from util.pibot import PenguinPi
from YOLO.detector import Detector
from perception.clustering import cluster_detections_dbscan


log = logging.getLogger(__name__)


class PiBotActions:
    """Convenience wrapper around PenguinPi with calibrated motion helpers."""

    def __init__(self,
                 pibot: PenguinPi,
                 calib_dir: str = "calibration/param/") -> None:
        self.ppi = pibot
        # Load calibration to convert ticks to angular rate
        try:
            scale_path = os.path.join(calib_dir, "scale.txt")
            base_path = os.path.join(calib_dir, "baseline.txt")
            self.scale = float(np.loadtxt(scale_path, delimiter=','))
            self.baseline = float(np.loadtxt(base_path, delimiter=','))
            log.info("PiBotActions: loaded calibration scale=%.6f baseline=%.4f", self.scale, self.baseline)
        except Exception as e:
            # Fallback to sensible defaults if missing
            self.scale = 0.002  # m per tick (example)
            self.baseline = 0.08  # m
            log.warning("PiBotActions: calibration load failed (%s). Using defaults scale=%.6f baseline=%.4f",
                        e, self.scale, self.baseline)

    def _turn_time_for_angle(self, angle_deg: float, turning_tick: int) -> float:
        """Compute time (s) to rotate in place by angle_deg at given turning_tick.

        Using omega = (v_r - v_l)/baseline, with v_r = +scale*tick, v_l = -scale*tick (effective),
        so omega ≈ 2*scale*turning_tick / baseline, time = |angle| / omega.
        """
        angle_rad = math.radians(abs(float(angle_deg)))
        omega = max(1e-6, 2.0 * self.scale * float(turning_tick) / self.baseline)
        return angle_rad / omega

    def scan(self,
             step_angle_deg: float,
             turning_tick: int = 25,
             pause_s: float = 1.0,
             detector: Optional[Detector] = None,
             fruit_ranger: Optional[object] = None,
             target_dims: Optional[Dict[str, Tuple[float, float, float]]] = None,
             get_pose_fn: Optional[Callable[[], List[float]]] = None,
             edge_margin_frac: float = 0.05) -> None:
        """Rotate on the spot in increments until a full 360° scan is done.

        - step_angle_deg: per-step rotation in degrees (minimum 10°).
        - turning_tick: tick value used for turning (affects speed).
        - pause_s: pause between steps to allow sensing.
        """
        # ------------------------------
        # 1) Normalize inputs and guards
        # ------------------------------
        try:
            step = abs(float(step_angle_deg))
        except Exception:
            step = 10.0

        if step < 10.0:
            log.info("scan: requested step %.1f° < 10°. Using 10°.", step)
            step = 10.0

        # ---------------------------------------------
        # 2) Partition full rotation into equal increments
        # ---------------------------------------------
        n_steps = max(1, int(math.ceil(360.0 / step)))
        step = 360.0 / n_steps
        log.info("scan: %d steps of %.1f° (turning_tick=%d, pause=%.1fs)", n_steps, step, turning_tick, pause_s)

        duration = self._turn_time_for_angle(step, turning_tick)
        
        # ---------------------------------------------
        # 3) Rotate in-place incrementally and capture images
        # ---------------------------------------------
        # Storage for captured images, poses and detections
        self.scan_images: List[Tuple[float, np.ndarray]] = []   # (angle_deg, image_rgb)
        self.scan_poses: List[Tuple[float, List[float]]] = []   # (angle_deg, [x,y,theta])
        self.scan_results: List[Dict[str, Any]] = []            # per-frame detection summary
        self.scan_detections: List[Dict[str, Any]] = []         # flat list of detections across frames
        self.scan_positions_by_class: Dict[int, List[Tuple[float, float]]] = {}  # class_id -> [(x,y),...]
        cum_angle = 0.0
        for i in range(n_steps):
            # Rotate on the spot by +step degrees
            # Use forward=0, turning=+1. Duration computed from calibration
            try:
                self.ppi.set_velocity([0, 1], turning_tick=turning_tick, time=duration)
            except Exception as e:
                log.warning("scan: set_velocity failed at step %d/%d: %s", i+1, n_steps, e)
                # attempt to continue
            # Pause to allow sensors to settle/capture
            time.sleep(max(0.0, float(pause_s)))
            # Capture an image after turning and pausing
            try:
                img_rgb = self.ppi.get_image()
                ang_now = cum_angle + step
                self.scan_images.append((ang_now, img_rgb))
                # Capture pose if provided
                if callable(get_pose_fn):
                    try:
                        pose_now = get_pose_fn()
                    except Exception as e:
                        log.warning("scan: pose capture failed at step %d/%d: %s", i+1, n_steps, e)
                        pose_now = [0.0, 0.0, 0.0]
                    self.scan_poses.append((ang_now, pose_now))
            except Exception as e:
                log.warning("scan: image capture failed at step %d/%d: %s", i+1, n_steps, e)
            cum_angle += step

        # ------------------------------
        # 4) Stop motion to finish the scan
        # ------------------------------
        try:
            self.ppi.set_velocity([0, 0])
        except Exception:
            pass
        
        # -----------------------------------------------------------
        # 5) Batch detection over captured images (optional, preferred)
        # -----------------------------------------------------------
        if detector is not None:
            try:
                # Prepare BGR list in the same order as scan_images
                imgs_bgr = []
                for _, img_rgb in self.scan_images:
                    img_bgr = img_rgb
                    if img_bgr is not None and img_bgr.ndim == 3:
                        img_bgr = img_bgr[:, :, ::-1]
                    imgs_bgr.append(img_bgr)
                # 5a) Detection (batched if available)
                detections_per_frame: List[List[Any]] = []
                if hasattr(detector, 'detect_batch'):
                    batch_results = detector.detect_batch(imgs_bgr)
                    for (det_out, _vis) in batch_results:
                        detections_per_frame.append(det_out or [])
                elif hasattr(detector, 'detect_single_image'):
                    # Log fallback warning
                    log.warning("scan: falling back to single image detection")
                    for img_bgr in imgs_bgr:
                        try:
                            det_out, _ = detector.detect_single_image(img_bgr)
                        except Exception:
                            det_out = []
                        detections_per_frame.append(det_out or [])

                # 5b) Filtering + world position computation
                for idx, ((angle, img_rgb), det_out) in enumerate(zip(self.scan_images, detections_per_frame)):
                    frame_entry: Dict[str, Any] = {'angle': float(angle), 'detections': []}
                    H, W = (img_rgb.shape[0], img_rgb.shape[1]) if img_rgb is not None else (0, 0)
                    margin_px = max(1, int(edge_margin_frac * W)) if W > 0 else 0
                    pose_now = self.scan_poses[idx][1] if idx < len(self.scan_poses) else None
                    for item in (det_out or []):
                        try:
                            label, bbox = item[0], item[1]
                            x, y, w, h = [float(v) for v in bbox]
                        except Exception as e:
                            frame_entry['detections'].append({'error': f'bad_bbox:{e}'})
                            continue
                        if W > 0 and (x < margin_px or (x + w) > (W - margin_px)):
                            frame_entry['detections'].append({'label': label, 'bbox': [x, y, w, h], 'error': 'edge_filtered'})
                            continue
                        world_ok = False
                        error_reason = None
                        wx = wy = None
                        if fruit_ranger is not None and target_dims is not None and pose_now is not None:
                            try:
                                true_h = None
                                if isinstance(label, str) and label in target_dims:
                                    dims = target_dims[label]
                                    if isinstance(dims, (list, tuple)) and len(dims) == 3:
                                        true_h = float(dims[2])
                                if true_h is None:
                                    true_h = 0.08
                                est = fruit_ranger.from_bbox_height([x, y, w, h], true_h)
                                if est is None:
                                    error_reason = 'range_theta_failed'
                                else:
                                    r = float(est['r']); th = float(est['theta'])
                                    rx, ry, rth = float(pose_now[0]), float(pose_now[1]), float(pose_now[2])
                                    wx = rx + r * math.cos(rth + th)
                                    wy = ry + r * math.sin(rth + th)
                                    world_ok = True
                            except Exception as e:
                                error_reason = f'world_compute_failed:{e}'
                        else:
                            error_reason = 'missing_ranger_or_pose_or_dims'

                        class_id = -1
                        try:
                            names = getattr(detector.model, 'names', {})
                            if isinstance(label, int):
                                class_id = int(label)
                            else:
                                for cid, name in names.items():
                                    if name == label:
                                        class_id = int(cid)
                                        break
                        except Exception:
                            pass

                        det_record: Dict[str, Any] = {
                            'label': label,
                            'class_id': class_id,
                            'bbox': [x, y, w, h],
                            'frame_angle': float(angle)
                        }
                        if world_ok and wx is not None and wy is not None:
                            det_record['world'] = {'x': float(wx), 'y': float(wy)}
                            self.scan_positions_by_class.setdefault(class_id, []).append((float(wx), float(wy)))
                        else:
                            det_record['error'] = error_reason or 'unknown'
                        frame_entry['detections'].append(det_record)
                        self.scan_detections.append(det_record)
                    self.scan_results.append(frame_entry)
            except Exception as e:
                log.warning("scan: batched detection failed: %s", e)

        # -----------------------------------------------------------
        # 6) Cluster detections into object hypotheses (store for next stage)
        #    Include prior object hypotheses (if any) as weighted detections
        # -----------------------------------------------------------
        try:
            # Merge prior object positions as detections with weight=count
            dets_for_cluster = list(self.scan_detections)
            if hasattr(self, 'current_obj_positions') and isinstance(self.current_obj_positions, list):
                for cl in self.current_obj_positions:
                    try:
                        dets_for_cluster.append({
                            'label': cl.get('class', ''),
                            'class_id': cl.get('class_id', -1),
                            'world': {'x': float(cl['position'][0]), 'y': float(cl['position'][1])},
                            'count': int(cl.get('count', 1))
                        })
                    except Exception:
                        continue
            self.current_obj_positions = cluster_detections_dbscan(dets_for_cluster, eps_m=0.15, min_samples=1, arena_bound=None)
        except Exception as e:
            log.warning("scan: clustering failed: %s", e)
            self.current_obj_positions = []
    
    def approach_fruit(self,
                        angle_deg: float,
                        distance_m: float,
                        turning_tick: int = 25,
                        forward_tick: int = 50) -> None:
        """Rotate to target direction and move forward by distance_m at given forward_tick speed. 
            Angle_deg should be from the robot's current heading.(i.e. relative angle)

        - distance_m: distance to move in meters (positive).
        - forward_tick: tick value for forward motion (affects speed).
        """
        try:
            dist = abs(float(distance_m))
        except Exception:
            dist = 0.0

        if dist < 0.01:
            log.info("approach_fruit: requested distance %.3f m < 0.01 m. No movement.", dist)
            return

        try:
            angle = float(angle_deg)
        except Exception:
            angle = 0.0

        if angle != 0.0:
            log.info("approach_fruit: rotating to angle %.1f°", angle)
            self.ppi.set_velocity([0, 0], turning_tick=turning_tick)



        # rotate to face target direction first
        if angle != 0.0:
            duration = self._turn_time_for_angle(angle, turning_tick)
            try:
                self.ppi.set_velocity([0, 1 if angle > 0 else -1], turning_tick=turning_tick, time=duration)
            except Exception as e:
                log.warning("approach_fruit: set_velocity failed during turn: %s", e)
                # attempt to continue

        try:
            tick = int(forward_tick)
        except Exception:
            tick = 50

        if tick < 10:
            log.info("approach_fruit: requested forward_tick %d < 10. Using 10.", tick)
            tick = 10

        # Compute time to move the requested distance at given speed
        v = self.scale * float(tick)  # m/s
        duration = dist / max(1e-6, v)
        log.info("approach_fruit: moving forward %.3f m at tick=%d (v=%.3f m/s) for %.2f s",
                 dist, tick, v, duration)

        try:
            self.ppi.set_velocity([1, 0], forward_tick=tick, time=duration)
        except Exception as e:
            log.warning("approach_fruit: set_velocity failed: %s", e)

        # Ensure motors are stopped
        try:
            self.ppi.set_velocity([0, 0])
        except Exception:
            pass

    def collect_fruit(self,
                      collection_class: str = "default",
                      duration_s: float = 2.1) -> None:
        """Sit next to fruit to collect for duration_s seconds. Prints fruit collected message in gui"""
        try:
            dur = max(0.1, float(duration_s))
        except Exception:
            dur = 2.1

        log.info("collect_fruit: activating collector for %.1f s", dur)
        try:
            time.sleep(dur)
            # print fruit collected message in gui
            self.ppi.collect_fruit(collection_class) # i dont know what this does
        except Exception as e:
            log.warning("collect_fruit: collector activation failed: %s", e)


    def return_to_scan_point(self,
                             distance_m: float,
                             forward_tick: int = 50) -> None:
        """Move backwards by distance_m at given forward_tick speed.

        - distance_m: distance to move in meters (positive).
        - forward_tick: tick value for backward motion (affects speed).
        """
        try:
            dist = abs(float(distance_m))
        except Exception:
            dist = 0.0

        if dist < 0.01:
            log.info("return_to_scan_point: requested distance %.3f m < 0.01 m. No movement.", dist)
            return

        try:
            tick = int(forward_tick)
        except Exception:
            tick = 50

        if tick < 10:
            log.info("return_to_scan_point: requested forward_tick %d < 10. Using 10.", tick)
            tick = 10

        # Compute time to move the requested distance at given speed
        v = self.scale * float(tick)  # m/s
        duration = dist / max(1e-6, v)
        log.info("return_to_scan_point: moving backward %.3f m at tick=%d (v=%.3f m/s) for %.2f s",
                 dist, tick, v, duration)

        try:
            self.ppi.set_velocity([-1, 0], forward_tick=tick, time=duration)
        except Exception as e:
            log.warning("return_to_scan_point: set_velocity failed: %s", e)

        # Ensure motors are stopped
        try:
            self.ppi.set_velocity([0, 0])
        except Exception:
            pass