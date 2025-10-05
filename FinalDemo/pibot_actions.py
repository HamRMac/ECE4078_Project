import time
import math
import logging
from typing import Optional, Callable, Dict, Any, List, Tuple
from collections import deque

import numpy as np

# Access the robot HTTP API wrapper
import sys, os
sys.path.insert(0, f"{os.getcwd()}/util")
from util.pibot import PenguinPi
from YOLO.detector import Detector
from perception.clustering import cluster_detections_dbscan
from perception.fruit_ranger import FruitRanger


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

        # ---------- Added (non-breaking) ----------
        self.get_pose_fn: Optional[Callable[[], List[float]]] = None
        self.target_queue: deque = deque()          # queue of {"class","class_id","position":[x,y],"count"}
        self.last_forward: float = 0.0              # last forward distance for return
        self.dets_for_cluster: List[Dict[str, Any]] = []  # raw dets used for clustering
        self.current_obj_positions: List[Dict[str, Any]] = []  # clustered objects
        # -----------------------------------------

    def stop_robot(self):
        self.ppi.set_velocity([0,0])

    def _turn_time_for_angle(self, angle_deg: float, turning_tick: int) -> float:
        """Compute time (s) to rotate in place by angle_deg at given turning_tick.

        Using omega = (v_r - v_l)/baseline, with v_r = +scale*tick, v_l = -scale*tick (effective),
        so omega ≈ 2*scale*turning_tick / baseline, time = |angle| / omega.
        """
        angle_rad = math.radians(abs(float(angle_deg)))
        omega = max(1e-6, 2.0 * self.scale * float(turning_tick) / self.baseline)
        return angle_rad / omega

    def turn_to_heading(self,
                        goal_heading_rad: float,
                        get_pose_fn: Callable[[], List[float]],
                        turning_tick: int) -> int:
        """Turn in place until heading ≈ goal_heading_rad using EKF feedback.

        Uses time=0 non-blocking motor commands and slows down near the goal.
        Returns the last turning_tick command used (for stop).
        """
        def wrap_pi(a: float) -> float:
            return (a + math.pi) % (2.0 * math.pi) - math.pi

        if not callable(get_pose_fn):
            log.error("get_pose_fn is not defined or callable")

        min_tick = 10
        max_tick = int(max(10, turning_tick))
        ang_tol = math.radians(2.0)
        dt_cmd = 0.05
        safety_timeout = 20
        t0 = time.time()
        last_tick = -1000
        while True:
            pose_now = get_pose_fn()
            th_now = float(pose_now[2] if pose_now is not None else 0.0)
            err = wrap_pi(float(goal_heading_rad) - th_now)
            log.debug(f"scan: angle: {th_now} with goal {goal_heading_rad} --> err = {err}")
            if abs(err) <= ang_tol:
                break
            turn_dir = 1 if err > 0 else -1
            gain = min(1.0, max(0.1, abs(err) / math.pi))
            tick_cmd = int(max(min_tick, min(max_tick, gain * max_tick)))
            tick_changed_much = (abs(last_tick-tick_cmd) > 5) or ((tick_cmd == min_tick) and (last_tick != min_tick))
            log.debug(f"tick_changed_much = {tick_changed_much} ({tick_cmd} vs {last_tick})")
            last_tick = tick_cmd
            try:
                if (tick_changed_much):
                    self.ppi.set_velocity([0, turn_dir], turning_tick=tick_cmd, time=0)
            except Exception as e:
                log.warning("scan: set_velocity failed during turn_to_heading: %s", e)
                break
            # time.sleep(dt_cmd)
            if (time.time() - t0) > safety_timeout:
                log.warning("scan: heading step timeout (err=%.3f rad)", err)
                break
        
        return last_tick

    def scan(self,
             step_angle_deg: float,
             detector: Detector,
             fruit_ranger: FruitRanger,
             target_dims: Dict[str, Tuple[float, float, float]],
             get_pose_fn: Callable[[], List[float]],
             turning_tick: int = 25,
             pause_s: float = 1.0,
             edge_margin_frac: float = 0.05) -> None:
        """Rotate on the spot in increments until a full 360° scan is done.

        - step_angle_deg: per-step rotation in degrees (minimum 10°).
        - turning_tick: tick value used for turning (affects speed).
        - pause_s: pause between steps to allow sensing.
        """
        # Validate we have values for all parameters
        missing = [name for name, value in {
            "detector": detector,
            "fruit_ranger": fruit_ranger,
            "target_dims": target_dims,
            "get_pose_fn": get_pose_fn,
        }.items() if value is None]
        if missing:
            log.error(f"scan: missing required parameters {missing}")
            return

        # ------------------------------
        # 1) Normalize inputs and guards
        # ------------------------------
        try:
            step = abs(float(step_angle_deg))
        except Exception:
            step = 10.0

        if step < 10.0:
            log.warning("scan: requested step %.1f° < 10°. Using 10°.", step)
            step = 10.0

        # ---------------------------------------------
        # 2) Partition full rotation into equal increments
        # ---------------------------------------------
        n_steps = max(1, int(math.ceil(360.0 / step)))
        step = 360.0 / n_steps
        log.debug("scan: %d steps of %.1f° (turning_tick=%d, pause=%.1fs)", n_steps, step, turning_tick, pause_s)

        # ---------------------------------------------
        # 3) Rotate incrementally to each target heading using closed-loop heading
        #    control (time=0 non-blocking motor command) and slow down as we
        #    approach the target heading. Use get_pose_fn for feedback.
        # ---------------------------------------------
        # Storage for captured images, poses and detections
        self.scan_images: List[Tuple[float, np.ndarray]] = []   # (angle_deg, image_rgb)
        self.scan_poses: List[Tuple[float, List[float]]] = []   # (angle_deg, [x,y,theta])
        self.scan_results: List[Dict[str, Any]] = []            # per-frame detection summary
        self.scan_detections: List[Dict[str, Any]] = []         # flat list of detections across frames
        self.scan_positions_by_class: Dict[int, List[Tuple[float, float]]] = {}  # class_id -> [(x,y),...]
        cum_angle = 0.0

        def wrap_pi(a: float) -> float:
            return (a + math.pi) % (2.0 * math.pi) - math.pi

        min_tick = 10
        max_tick = int(max(min_turning_tick := 10, turning_tick))
        ang_tol = math.radians(2.0)  # ~2 degrees
        dt_cmd = 0.05

        for i in range(n_steps):
            # Determine current and target headings
            start_pose = None
            if callable(get_pose_fn):
                try:
                    start_pose = get_pose_fn()
                except Exception:
                    start_pose = [0.0, 0.0, 0.0]
            curr_th = float(start_pose[2] if start_pose is not None else 0.0)
            goal_th = wrap_pi(curr_th + math.radians(step))
            log.debug(f"scan: heading goal {goal_th}")

            # Closed-loop turn using helper
            last_tick = self.turn_to_heading(goal_th, get_pose_fn, turning_tick)

            # Short pause to settle, then capture EKF pose and image at this bearing
            self.ppi.set_velocity([0, 0], turning_tick=last_tick, time=0)
            time.sleep(max(0.0, float(pause_s)))
            ang_now = cum_angle + step
            # Capture pose first (closest to capture time)
            pose_now = None
            if callable(get_pose_fn):
                try:
                    pose_now = get_pose_fn()
                except Exception as e:
                    log.warning("scan: pose capture failed at step %d/%d: %s", i+1, n_steps, e)
                    pose_now = [0.0, 0.0, 0.0]
            # Capture an image after turning and pausing
            try:
                img_rgb = self.ppi.get_image()
                self.scan_images.append((ang_now, img_rgb))
                if pose_now is not None:
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
        log.debug("scan: running detection on %d frames ...", len(self.scan_images))
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
                                    log.warning("scan: missing true height for class '%s'. Using 0.08 m.", label)
                                    true_h = 0.08
                                est = fruit_ranger.from_bbox_height([x, y, w, h], true_h)
                                if est is None:
                                    error_reason = 'range_theta_failed'
                                else:
                                    r = float(est['r']); th = float(est['theta'])
                                    rx, ry, rth = float(pose_now[0]), float(pose_now[1]), float(pose_now[2])
                                    wx = rx + r * math.cos(rth + th)
                                    wy = ry + r * math.sin(rth + th)
                                    log.debug("scan: class='%s' bbox=[%.1f,%.1f,%.1f,%.1f] => r=%.3f m, θ=%.3f rad => @ (%.3f, %.3f)", label, x, y, w, h, r, th, wx, wy)

                                    if (r <= 1.0):
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
        #    Maintain a persistent, ever-growing list of raw world detections
        #    across scans; cluster over the full list each time.
        # -----------------------------------------------------------
        try:
            # Ensure persistent list exists
            if not hasattr(self, 'dets_for_cluster') or not isinstance(self.dets_for_cluster, list):
                self.dets_for_cluster = []
            # Append new world detections from this scan
            new_cnt = 0
            for det in (self.scan_detections or []):
                try:
                    w = det.get('world')
                    if w is None:
                        continue
                    x = float(w['x']); y = float(w['y'])
                    label = det.get('label', '')
                    cid = det.get('class_id', -1)
                    # store with unit weight; the growth in list size represents observation count
                    self.dets_for_cluster.append({'label': label, 'class_id': cid, 'world': {'x': x, 'y': y}, 'count': 1})
                    new_cnt += 1
                except Exception:
                    continue
            log.debug("scan: clustering over %d persisted detections (+%d this scan)", len(self.dets_for_cluster), new_cnt)
            # Cluster over the full persistent list
            self.current_obj_positions = cluster_detections_dbscan(self.dets_for_cluster, eps_m=0.25, min_samples=1, arena_bound=None)

            # Rebuild target queue from clustered objects
            self._build_queue_from_current_objs(order="fifo", pose_fn=get_pose_fn)
        except Exception as e:
            log.warning("scan: clustering failed: %s", e)
            self.current_obj_positions = []
    
    def approach_fruit(self,
                        angle_deg: float,
                        distance_m: float,
                        turning_tick: int = 25,
                        forward_tick: int = 50) -> None:
        """Rotate to target direction and move forward by distance_m at given forward_tick speed. 
            Angle_deg should be from the global frame 

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

        # interpret `angle_deg` as a global/absolute heading in degrees (world frame)
        try:
            goal_angle_deg = float(angle_deg)
        except Exception:
            goal_angle_deg = 0.0

        if not (-360.0 < goal_angle_deg < 360.0):
            # clamp to sensible range
            goal_angle_deg = ((goal_angle_deg + 180.0) % 360.0) - 180.0

        # convert to radians and normalise to [-pi, pi]
        goal_th = (math.radians(goal_angle_deg) + math.pi) % (2.0 * math.pi) - math.pi
        try:
            pose_fn = self.get_pose_fn if callable(self.get_pose_fn) else (lambda: [0.0, 0.0, 0.0])
            last_tick = self.turn_to_heading(goal_th, pose_fn, turning_tick)
            # ensure motors stopped from turning
            try:
                self.ppi.set_velocity([0, 0], turning_tick=last_tick, time=0)
            except Exception:
                pass
        except Exception as e:
            log.warning("approach_fruit: closed-loop turn failed: %s", e)

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
                      duration_s: float = 2.5) -> None:
        """Sit next to fruit to collect for duration_s seconds. Prints fruit collected message in gui"""
        try:
            dur = max(0.1, float(duration_s))
        except Exception:
            dur = 2.5

        try:
            log.info(f"collecting {collection_class}...")
            time.sleep(dur)
            # print fruit collected message in gui
            log.info(f"Collected {collection_class}!")
        except Exception as e:
            log.warning("collect_fruit: collector failed: %s", e)


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

        # Move backward without time-based rotation; ensure heading maintained
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
            # Use negative forward command to move backward
            self.ppi.set_velocity([-1, 0], forward_tick=tick, time=duration)
        except Exception as e:
            log.warning("return_to_scan_point: set_velocity failed: %s", e)

        # Ensure motors are stopped
        try:
            self.ppi.set_velocity([0, 0])
        except Exception:
            pass

    def localise_scan(self,
                      step_angle_deg: float = 45.0,
                      get_pose_fn: Optional[Callable[[], List[float]]] = None,
                      turning_tick: int = 35,
                      pause_s: float = 0.3) -> None:
        try:
            step = abs(float(step_angle_deg))
        except Exception:
            step = 45.0
        if step < 10.0:
            step = 10.0
        n_steps = max(1, int(math.ceil(360.0 / step)))
        step = 360.0 / n_steps

        def wrap_pi(a: float) -> float:
            return (a + math.pi) % (2.0 * math.pi) - math.pi

        for _ in range(n_steps):
            try:
                pose_now = get_pose_fn() if callable(get_pose_fn) else [0.0, 0.0, 0.0]
            except Exception:
                pose_now = [0.0, 0.0, 0.0]
            curr_th = float(pose_now[2] if pose_now is not None else 0.0)
            goal_th = wrap_pi(curr_th + math.radians(step))
            try:
                last_tick = self.turn_to_heading(goal_th, get_pose_fn if callable(get_pose_fn) else (lambda: [0.0, 0.0, 0.0]), int(turning_tick))
                try:
                    self.ppi.set_velocity([0, 0], turning_tick=last_tick, time=0)
                except Exception:
                    pass
            except Exception:
                break
            try:
                time.sleep(max(0.0, float(pause_s)))
            except Exception:
                pass
        try:
            self.ppi.set_velocity([0, 0])
        except Exception:
            pass

    # ======================================================================
    #                Added: dets_for_cluster + queue conveniences
    # ======================================================================

    def _to_target_dict(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalise a detection/cluster item into:
          {"class":str, "class_id":int, "position":[x,y], "count":int}
        Accepts either {"position":[x,y],...} or {"world":{"x":...,"y":...},...}.
        """
        try:
            cls = item.get("class") or item.get("label") or "fruit"
            cid = int(item.get("class_id", -1)) if item.get("class_id") is not None else -1

            if "position" in item and item["position"] is not None:
                x, y = float(item["position"][0]), float(item["position"][1])
            elif isinstance(item.get("world"), dict) and "x" in item["world"] and "y" in item["world"]:
                x, y = float(item["world"]["x"]), float(item["world"]["y"])
            else:
                return None

            cnt = int(item.get("count", 1))
            return {"class": cls, "class_id": cid, "position": [x, y], "count": cnt}
        except Exception:
            return None

    def _build_queue_from_current_objs(self, order: str = "fifo", pose_fn: Optional[Callable[[], List[float]]] = None) -> None:
        """
        Build/refresh self.target_queue from clustered objects if available,
        else fall back to raw dets_for_cluster. Optionally sort by nearest.
        """
        self.target_queue.clear()
        if pose_fn:
            self.get_pose_fn = pose_fn

        source = self.current_obj_positions if self.current_obj_positions else self.dets_for_cluster
        canon: List[Dict[str, Any]] = []
        for it in (source or []):
            td = self._to_target_dict(it)
            if td:
                canon.append(td)

        # Optional nearest-first ordering
        if order == "nearest" and callable(self.get_pose_fn) and canon:
            try:
                rx, ry, *_ = self.get_pose_fn()
                canon.sort(key=lambda t: (t["position"][0] - rx)**2 + (t["position"][1] - ry)**2)
            except Exception:
                pass

        for t in canon:
            self.target_queue.append(t)

    def refresh_queue_from_scan(self, order: str = "fifo") -> None:
        """Rebuild target queue using the last scan results (use after a re-scan)."""
        self._build_queue_from_current_objs(order=order)

    def has_targets(self) -> bool:
        return len(self.target_queue) > 0

    def peek_target(self) -> Optional[Dict[str, Any]]:
        return self.target_queue[0] if self.target_queue else None

    def pop_target(self) -> Optional[Dict[str, Any]]:
        return self.target_queue.popleft() if self.target_queue else None

    def _rel_angle_dist(self, target_xy: List[float], standoff_m: float = 0.10) -> Tuple[float, float]:
        """
        Convert world target XY to (relative angle [deg], forward distance [m]) from current pose.
        Requires self.get_pose_fn to be set (passed into scan or set later).
        """
        if not callable(self.get_pose_fn):
            raise RuntimeError("Pose function not set. Pass get_pose_fn to scan(...) or set self.get_pose_fn.")
        rx, ry, rth = [float(v) for v in self.get_pose_fn()]
        tx, ty = float(target_xy[0]), float(target_xy[1])
        dx, dy = tx - rx, ty - ry
        ang = math.degrees(math.atan2(dy, dx) - rth)
        ang = (ang + 180.0) % 360.0 - 180.0  # wrap to [-180, 180]
        dist = max(0.0, math.hypot(dx, dy) - float(standoff_m))
        return ang, dist

    def approach_current(self, turning_tick: int = 25, forward_tick: int = 50, standoff_m: float = 0.10) -> Optional[Dict[str, float]]:
        """
        Convenience: compute (angle, distance) to the first target in the queue and call approach_fruit(...).
        Stores the forward distance so return_from_current() can back out the same amount.
        """
        tgt = self.peek_target()
        if not tgt:
            return None
        angle_deg, distance_m = self._rel_angle_dist(tgt["position"], standoff_m=standoff_m)
        # compute global heading: robot heading + relative angle
        try:
            rx, ry, rth = [float(v) for v in self.get_pose_fn()]
            global_heading_deg = (math.degrees(rth) + float(angle_deg) + 180.0) % 360.0 - 180.0
        except Exception:
            global_heading_deg = float(angle_deg)
        self.last_forward = distance_m
        # Use closed-loop turning inside approach_fruit
        self.approach_fruit(angle_deg=global_heading_deg, distance_m=distance_m,
                            turning_tick=turning_tick, forward_tick=forward_tick)
        return {"angle_deg": angle_deg, "distance_m": distance_m}

    def collect_current(self, duration_s: float = 2.1, collection_class: Optional[str] = None) -> bool:
        """
        Convenience: call collect_fruit() using the provided class (if given) or the
        current target's class, then pop it from the queue.

        Parameters:
        - duration_s: seconds to run the collector
        - collection_class: optional explicit class name to pass through to logging/collector
        """
        tgt = self.peek_target()
        if not tgt:
            return False

        class_to_collect = collection_class or tgt.get("class", "default")
        log.info("collect_current: collecting class='%s' for %.2fs", class_to_collect, float(duration_s))
        self.collect_fruit(collection_class=class_to_collect, duration_s=duration_s)
        self.pop_target()
        return True

    def return_from_current(self, forward_tick: int = 50) -> bool:
        """
        Convenience: back out by the same distance used in the last approach_fruit().
        """
        d = float(self.last_forward or 0.0)
        if d <= 1e-6:
            return False
        self.return_to_scan_point(distance_m=d, forward_tick=forward_tick)
        self.last_forward = 0.0
        return True
