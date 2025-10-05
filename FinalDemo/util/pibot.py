# access each wheel and the camera onboard of PenguinPi

import logging
import math
import numpy as np
import requests
import cv2 
import time as time_module
import urllib.request
import threading
from typing import Optional, Tuple


log = logging.getLogger(__name__)


class PenguinPi:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

        # Hardware sends right wheel in reverse compared to simulator
        self._right_wheel_inverted = (self.ip != 'localhost')

        # Commanded wheel velocities in body frame (ticks/s)
        self._wheel_vel_cmd = np.zeros(2, dtype=float)

        # Most recent measured velocities from encoder polling (ticks/s)
        self._wheel_vel_meas = np.zeros(2, dtype=float)
        self._wheel_vel_meas_time: Optional[float] = None

        # Latest encoder counts (body frame tick counts) and timestamp
        self._encoder_counts = None
        self._encoder_timestamp = None

        # Background polling infrastructure
        self._encoder_lock = threading.Lock()
        self._encoder_thread: Optional[threading.Thread] = None
        self._encoder_stop = threading.Event()
        self._encoder_rate_hz = 0.0

        # Legacy attribute (kept for compatibility)
        self.wheel_vel = [0.0, 0.0]

    ##########################################
    # Change the robot velocity here
    # tick = forward speed
    # turning_tick = turning speed
    ########################################## 
    def set_velocity(self, command: list[int], tick=50, turning_tick=20, time=0):
        # command: [forward speed, turning speed]
        assert (len(command) == 2), "Command must be a list of two elements"

        # Body-frame wheel velocities (ticks/s)
        body_l_ticks = command[0]*tick - command[1]*turning_tick
        body_r_ticks = command[0]*tick + command[1]*turning_tick

        # Convert to hardware commands (right wheel inverted on physical robot)
        hw_l = body_l_ticks
        hw_r = -body_r_ticks if self._right_wheel_inverted else body_r_ticks

        # Track commanded body-frame velocities
        cmd_mps = np.array([body_l_ticks, body_r_ticks], dtype=float)
        self._wheel_vel_cmd = cmd_mps
        self.wheel_vel = [body_l_ticks, body_r_ticks]

        # If we intentionally stop, mark measured velocity as stale immediately
        if body_l_ticks == 0.0 and body_r_ticks == 0.0:
            with self._encoder_lock:
                self._wheel_vel_meas = np.zeros(2, dtype=float)
                self._wheel_vel_meas_time = time_module.monotonic()

        if time == 0:
            requests.get(
                f"http://{self.ip}:{self.port}/robot/set/velocity?value={hw_l},{hw_r}"
            )
        else:
            assert (time > 0), "Time must be positive."
            assert (time < 30), "Time must be less than network timeout (20s)."
            requests.get(
                f"http://{self.ip}:{self.port}/robot/set/velocity?value={hw_l},{hw_r}&time={time}"
            )
        return body_l_ticks, body_r_ticks
    
    # get frame from simulated robot    
    def get_image_sim(self):
        try:
            r = requests.get(f"http://{self.ip}:{self.port}/camera/get", timeout=0.1)
            img = cv2.imdecode(np.frombuffer(r.content,np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Image retrieval timed out.")
            img = np.zeros((240,320,3), dtype=np.uint8)
        return img
    
    # get frames from the physical robot
    def get_image_physical(self):
        try:
            # the image stream URL
            url_str = f"http://{self.ip}:{self.port}/camera/get" # "http://192.168.50.1:8080/camera/get"
            #encoding = 'ISO-8859-1'
            x = urllib.request.urlopen(url=url_str)

            # size of bytes to read from the connected robot for a frame
            max_size = 2048
            result = b''
            # look for two consecutive '--frame' in the data
            i = 0
            while True:
                buf = x.fp.read(max_size)
                # (Second) frame start?
                if b'--frame' in buf:
                    i += 1
                result += buf
                if i > 1:
                    break
            next_frame_boundary = result.rfind(b'--frame')
            # get the binary data in between '--frame'
            img_bits = result[len(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'):next_frame_boundary]
            # save the binary data as image for display
            img_array = np.frombuffer(img_bits, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Image retrieval timed out.")
            img = np.zeros((240,320,3), dtype=np.uint8)
        return img

    # choose which get image function to use depending on running in simulator or on robot
    def get_image(self):
        if self.ip == 'localhost':
            return self.get_image_sim()
        else:
            return self.get_image_physical()

    # ---------------- Encoder polling ----------------
    def start_encoder_monitor(self, rate_hz: float = 10.0):
        """Start a background thread that polls wheel encoders at rate_hz."""
        rate_hz = float(rate_hz)
        if rate_hz <= 0:
            return
        with self._encoder_lock:
            self._encoder_rate_hz = rate_hz
        if self._encoder_thread and self._encoder_thread.is_alive():
            return
        self._encoder_stop.clear()
        self._encoder_thread = threading.Thread(
            target=self._encoder_poll_loop,
            args=(rate_hz,),
            name="PenguinPi-EncoderPoll",
            daemon=True,
        )
        self._encoder_thread.start()

    def stop_encoder_monitor(self, join: bool = True):
        if not self._encoder_thread:
            return
        self._encoder_stop.set()
        if join and self._encoder_thread.is_alive():
            self._encoder_thread.join(timeout=1.0)
        self._encoder_thread = None

    def get_wheel_velocity(self,
                            prefer_measured: bool = True,
                            max_staleness: float = 0.5) -> Tuple[float, float]:
        """Return wheel velocities (ticks/s) in robot body frame (left, right).

        prefer_measured: if True, use encoder-derived velocities when available and
        not older than max_staleness seconds; otherwise falls back to the last
        commanded velocity.
        """
        now = time_module.monotonic()
        if prefer_measured:
            with self._encoder_lock:
                if self._wheel_vel_meas_time is not None and (now - self._wheel_vel_meas_time) <= max_staleness:
                    return float(self._wheel_vel_meas[0]), float(self._wheel_vel_meas[1])
        # Fall back to commanded velocities
        return float(self._wheel_vel_cmd[0]), float(self._wheel_vel_cmd[1])

    def _encoder_poll_loop(self, rate_hz: float) -> None:
        period = 1.0 / max(1e-3, rate_hz)
        session = requests.Session()
        prev_counts: Optional[np.ndarray] = None
        prev_stamp: Optional[float] = None
        error_logged = False
        while not self._encoder_stop.is_set():
            start = time_module.monotonic()
            try:
                counts, stamp = self._fetch_encoder(session)
            except Exception as exc:
                if not error_logged:
                    log.warning("Encoder poll failed: %s", exc)
                    error_logged = True
                prev_counts = None
                prev_stamp = None
                counts = None
                stamp = None
            else:
                error_logged = False
            if counts is not None and stamp is not None:
                body_counts = self._apply_body_frame_counts(counts)
                with self._encoder_lock:
                    self._encoder_counts = body_counts
                    self._encoder_timestamp = stamp
                if prev_counts is not None and prev_stamp is not None:
                    dt = stamp - prev_stamp  
                    if dt > 1e-6:
                        vel = ((body_counts - prev_counts) / dt) / 10  # Divide by 10 to convert from encoder_ticks/s to command_ticks/s
                        with self._encoder_lock:
                            self._wheel_vel_meas = vel
                            self._wheel_vel_meas_time = time_module.monotonic()
                prev_counts = body_counts
                prev_stamp = stamp
            elapsed = time_module.monotonic() - start
            wait_time = max(0.0, period - elapsed)
            if self._encoder_stop.wait(wait_time):
                break

    def _apply_body_frame_counts(self, counts: np.ndarray) -> np.ndarray:
        left = float(counts[0])
        right = float(counts[1])
        if self._right_wheel_inverted:
            right = -right
        return np.array([left, right], dtype=float)

    def _fetch_encoder(self, session: requests.Session) -> Tuple[np.ndarray, float]:
        url = f"http://{self.ip}:{self.port}/robot/get/encoder"
        resp = session.get(url, timeout=0.2)
        resp.raise_for_status()
        payload = resp.text.strip()
        parts = payload.split(',') if payload else []
        if len(parts) != 2:
            raise ValueError(f"Unexpected encoder payload: '{payload}'")
        try:
            counts = np.array([float(parts[0]), float(parts[1])], dtype=float)
        except ValueError as exc:
            raise ValueError(f"Invalid encoder reading: '{payload}'") from exc
        try:
            stamp = float(resp.headers.get('X-Encoder-Monotonic'))
        except (TypeError, ValueError):
            stamp = time_module.monotonic()
        return counts, stamp
