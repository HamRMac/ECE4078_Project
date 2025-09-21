import time
import math
import logging
from typing import Optional

import numpy as np

# Access the robot HTTP API wrapper
import sys, os
sys.path.insert(0, f"{os.getcwd()}/util")
from util.pibot import PenguinPi


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
             pause_s: float = 1.0) -> None:
        """Rotate on the spot in increments until a full 360° scan is done.

        - step_angle_deg: per-step rotation in degrees (minimum 10°).
        - turning_tick: tick value used for turning (affects speed).
        - pause_s: pause between steps to allow sensing.
        """
        try:
            step = abs(float(step_angle_deg))
        except Exception:
            step = 10.0

        if step < 10.0:
            log.info("scan: requested step %.1f° < 10°. Using 10°.", step)
            step = 10.0

        # Compute number of steps to cover 360°, use equal partition
        n_steps = max(1, int(math.ceil(360.0 / step)))
        step = 360.0 / n_steps
        log.info("scan: %d steps of %.1f° (turning_tick=%d, pause=%.1fs)", n_steps, step, turning_tick, pause_s)

        duration = self._turn_time_for_angle(step, turning_tick)
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

        # Ensure motors are stopped
        try:
            self.ppi.set_velocity([0, 0])
        except Exception:
            pass

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