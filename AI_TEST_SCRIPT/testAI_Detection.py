#!/usr/bin/env python3
"""Live YOLOv11 inference on MJPEG stream with dashed boxes for low-confidence detections.

Features:
  * Loads custom YOLO model (.pt) via ultralytics
  * Reads an MJPEG stream (tested with 320x240 @25fps) using OpenCV VideoCapture
  * Draws bounding boxes, class names, confidence values
  * Solid box if conf >= solid_threshold, dashed box otherwise (but >= min_conf)
  * Per-class color mapping (auto or user supplied)
  * Graceful reconnect logic if the stream drops
  * FPS overlay

Usage (examples):
  python testAI_Detection.py \
	  --model ECE4078B1_Modelv1_FixedNames.pt \
	  --url http://192.168.50.1:8080/camera/get \
	  --solid-thr 0.5 --min-conf 0.2

Optional class colors:
  --class-colors "0:#FF5555,1:#33DD33,2:#3355FF,3:#FF33AA,4:#FFAA33,5:#44FFFF,6:#AA33FF,7:#AAAAAA"

Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
import math
from typing import Dict, Tuple, List
from threading import Thread, Event, Lock

import cv2
import numpy as np
try:
	import torch
except Exception:  # pragma: no cover
	torch = None  # type: ignore

try:
	from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
	print("Failed to import ultralytics. Ensure it's installed (pip install ultralytics).", file=sys.stderr)
	raise


DEFAULT_COLOR_HEX = [
	"#ffc821",  # class 0
	"#fff8d6",  # class 1
	"#ebe20f",  # class 2
	"#fd9f21",  # class 3
	"#00dd1d",  # class 4
	"#7b5010",  # class 5
	"#e37400",  # class 6
	"#ff0000",  # class 7
]


def default_color_mapping(model_names: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
	"""Return a fixed default color mapping (BGR) for reproducibility.

	If there are more classes than predefined colors, remaining classes use auto palette.
	"""
	mapping: Dict[int, Tuple[int, int, int]] = {}
	sorted_ids = sorted(model_names.keys())
	for i, cls_id in enumerate(sorted_ids):
		if i < len(DEFAULT_COLOR_HEX):
			hexcol = DEFAULT_COLOR_HEX[i][1:]
			r = int(hexcol[0:2], 16)
			g = int(hexcol[2:4], 16)
			b = int(hexcol[4:6], 16)
			mapping[cls_id] = (b, g, r)
		else:
			# fallback using auto palette for extra classes
			auto = auto_palette({cls_id: model_names[cls_id]})
			mapping[cls_id] = auto[cls_id]
	return mapping


def parse_class_colors(arg: str | None, model_names: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
	"""Parse user-provided mapping like "0:#RRGGBB,1:#RRGGBB" into BGR tuples.
	Falls back to a deterministic default mapping if not provided; fills gaps with auto palette.
	"""
	if not arg:
		return default_color_mapping(model_names)
	mapping: Dict[int, Tuple[int, int, int]] = {}
	for item in arg.split(','):
		item = item.strip()
		if not item:
			continue
		try:
			idx_str, hexcol = item.split(':', 1)
			cls_idx = int(idx_str)
			hexcol = hexcol.strip()
			if hexcol.startswith('#'):
				hexcol = hexcol[1:]
			if len(hexcol) != 6:
				raise ValueError
			r = int(hexcol[0:2], 16)
			g = int(hexcol[2:4], 16)
			b = int(hexcol[4:6], 16)
			mapping[cls_idx] = (b, g, r)  # OpenCV uses BGR
		except ValueError:
			print(f"Warning: could not parse color segment '{item}', expected format idx:#RRGGBB", file=sys.stderr)
	# Fill any missing classes with deterministic defaults first, then auto palette if needed
	defaults = default_color_mapping(model_names)
	for i in model_names.keys():
		mapping.setdefault(i, defaults[i])
	return mapping


def auto_palette(model_names: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
	"""Generate visually distinct colors using HSV spacing."""
	n = len(model_names)
	colors: Dict[int, Tuple[int, int, int]] = {}
	for i in model_names.keys():
		h = (i / max(1, n)) % 1.0
		s = 0.75
		v = 0.95
		r, g, b = hsv_to_rgb(h, s, v)
		colors[i] = (int(b * 255), int(g * 255), int(r * 255))
	return colors


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
	i = int(h * 6)
	f = h * 6 - i
	p = v * (1 - s)
	q = v * (1 - f * s)
	t = v * (1 - (1 - f) * s)
	i = i % 6
	if i == 0:
		return v, t, p
	if i == 1:
		return q, v, p
	if i == 2:
		return p, v, t
	if i == 3:
		return p, q, v
	if i == 4:
		return t, p, v
	return v, p, q


def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_len=8, gap_len=6):
	"""Draw a dashed rectangle from pt1 to pt2."""
	x1, y1 = pt1
	x2, y2 = pt2
	# Top and bottom
	_draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_len, gap_len)
	_draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash_len, gap_len)
	# Left and right
	_draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash_len, gap_len)
	_draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_len, gap_len)


def _draw_dashed_line(img, p1, p2, color, thickness, dash_len, gap_len):
	x1, y1 = p1
	x2, y2 = p2
	length = math.hypot(x2 - x1, y2 - y1)
	if length == 0:
		return
	dx = (x2 - x1) / length
	dy = (y2 - y1) / length
	dist = 0.0
	while dist < length:
		start_x = int(x1 + dx * dist)
		start_y = int(y1 + dy * dist)
		dist_end = min(dist + dash_len, length)
		end_x = int(x1 + dx * dist_end)
		end_y = int(y1 + dy * dist_end)
		cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)
		dist += dash_len + gap_len


def put_label(img, text, x1, y1, color):
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.5
	thickness = 1
	(tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
	pad = 2
	cv2.rectangle(img, (x1, y1 - th - pad * 2), (x1 + tw + pad * 2, y1), color, -1)
	cv2.putText(img, text, (x1 + pad, y1 - pad), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def reconnect_capture(url: str, attempts: int = 5, delay: float = 2.0):
	"""Attempt to open the MJPEG stream multiple times.

	Returns an opened cv2.VideoCapture or None on failure.
	"""
	for i in range(attempts):
		cap = cv2.VideoCapture(url)
		if cap.isOpened():
			return cap
		time.sleep(delay)
	return None


class FrameGrabber:
	"""Continuously grabs frames on a background thread, always exposing the latest frame.

	Why: If inference is slower than the camera FPS (e.g. 10-12fps vs 25fps source), a naive
	loop reading a frame then running inference will accumulate latency as frames queue up.
	By grabbing on a separate thread and overwriting a single storage slot, we effectively
	*drop* intermediate frames and always process the newest available, eliminating lag and
	"snapping" artifacts when buffers flush.
	"""

	def __init__(self, url: str, reconnect_attempts: int = 5, reconnect_delay: float = 2.0, fail_reopen_threshold: int = 25):
		self.url = url
		self.reconnect_attempts = reconnect_attempts
		self.reconnect_delay = reconnect_delay
		self.fail_reopen_threshold = fail_reopen_threshold  # number of consecutive read failures before reopen
		self.cap = reconnect_capture(url, attempts=reconnect_attempts, delay=reconnect_delay)
		self.latest_frame = None  # type: ignore
		self.latest_timestamp = 0.0
		self.lock = Lock()
		self.stop_event = Event()
		self.thread: Thread | None = None
		self.consecutive_fail = 0

	def start(self):
		if self.cap is None:
			return False
		self.thread = Thread(target=self._loop, daemon=True)
		self.thread.start()
		return True

	def _loop(self):
		while not self.stop_event.is_set():
			if self.cap is None:
				# Attempt to reopen if we lost the capture
				self.cap = reconnect_capture(self.url, attempts=self.reconnect_attempts, delay=self.reconnect_delay)
				if self.cap is None:
					time.sleep(1.0)
					continue
			ret, frame = self.cap.read()
			if not ret or frame is None:
				self.consecutive_fail += 1
				if self.consecutive_fail > self.fail_reopen_threshold:
					self.cap.release()
					self.cap = None
					self.consecutive_fail = 0
					time.sleep(0.2)
				continue
			self.consecutive_fail = 0
			with self.lock:
				self.latest_frame = frame
				self.latest_timestamp = time.time()

	def read(self):
		"""Return a (frame, timestamp) tuple for the newest frame (shallow copy), or (None, 0) if none yet."""
		with self.lock:
			if self.latest_frame is None:
				return None, 0.0
			# We do not deep-copy the frame to avoid extra overhead; caller should not mutate heavily.
			return self.latest_frame.copy(), self.latest_timestamp

	def stop(self):
		self.stop_event.set()
		if self.thread and self.thread.is_alive():
			self.thread.join(timeout=1.0)
		if self.cap:
			self.cap.release()
		self.cap = None


def class_agnostic_merge(boxes: np.ndarray, confs: np.ndarray, clss: np.ndarray, iou_thr: float) -> List[int]:
	"""Perform a simple class-agnostic Non-Maximum Suppression (NMS) pass.

	Ultralytics YOLO already runs NMS, but typically class-aware; occasionally, overlapping
	detections of the same physical object with different classes (or residual duplicates)
	may remain. We apply an additional class-agnostic merge: for any pair with IoU > iou_thr,
	keep only the higher-confidence index.

	Returns list of kept indices in original order of descending confidence (already sorted in logic).
	"""
	if boxes.shape[0] <= 1 or iou_thr <= 0:
		return list(range(boxes.shape[0]))
	# Sort by confidence descending
	idxs = np.argsort(-confs)
	keep: List[int] = []
	suppressed = np.zeros(boxes.shape[0], dtype=bool)
	for i in idxs:
		if suppressed[i]:
			continue
		keep.append(i)
		# Compute IoU of this box with the rest
		xyxy_i = boxes[i]
		x1i, y1i, x2i, y2i = xyxy_i
		area_i = (x2i - x1i) * (y2i - y1i)
		for j in idxs:  # iterate in confidence order for early pruning
			if j == i or suppressed[j]:
				continue
			x1j, y1j, x2j, y2j = boxes[j]
			# Intersection
			ix1 = max(x1i, x1j)
			iy1 = max(y1i, y1j)
			ix2 = min(x2i, x2j)
			iy2 = min(y2i, y2j)
			w = max(0.0, ix2 - ix1)
			h = max(0.0, iy2 - iy1)
			inter = w * h
			if inter <= 0:
				continue
			area_j = (x2j - x1j) * (y2j - y1j)
			iou = inter / (area_i + area_j - inter + 1e-9)
			if iou > iou_thr:
				suppressed[j] = True
	return keep


def select_device(user_choice: str) -> str:
	"""Resolve device string.

	user_choice 'auto' -> prefer CUDA, then MPS (Apple), else CPU.
	Otherwise return the user string unchanged.
	"""
	if user_choice != 'auto':
		return user_choice
	if torch is not None and torch.cuda.is_available():
		return 'cuda:0'
	if torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # Apple Silicon
		return 'mps'
	return 'cpu'


def run(args):
	"""Main execution loop.

	1. Load YOLO model
	2. Start frame grabber thread (latest frame only)
	3. For each iteration: pull newest frame, run inference, apply class-agnostic merge, draw annotations
	4. Display with FPS overlay; allow quitting with 'q'
	"""
	model = YOLO(args.model)
	device = select_device(args.device)
	try:
		# ultralytics YOLO exposes .to(device) for underlying torch model
		model.to(device)  # type: ignore[attr-defined]
		print(f"Using device: {device}")
	except Exception as e:  # pragma: no cover
		print(f"Warning: could not move model to {device}: {e}")
	names = model.model.names if hasattr(model, 'model') else getattr(model, 'names', {})  # type: ignore
	if not isinstance(names, dict):
		names = {i: str(n) for i, n in enumerate(names)}
	colors = parse_class_colors(args.class_colors, names)

	grabber = FrameGrabber(args.url)
	if not grabber.start():
		print(f"ERROR: Cannot open stream {args.url}")
		return 1

	window = "YOLO Live"
	cv2.namedWindow(window, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(window, 800, 600)

	last_time = time.time()
	fps = 0.0

	try:
		while True:
			# Obtain the latest available frame (may skip intermediate frames -> reduced latency)
			frame, ts = grabber.read()
			if frame is None:  # Not yet ready
				time.sleep(0.005)
				continue

			# Inference (note: we pass the frame directly; YOLO handles numpy array)
			results = model(frame, verbose=False)[0]
			if results.boxes is not None and results.boxes.xyxy is not None:
				boxes_obj = results.boxes
				xyxy = boxes_obj.xyxy.cpu().numpy()
				confs = boxes_obj.conf.cpu().numpy()
				clss = boxes_obj.cls.cpu().numpy().astype(int)

				# Filter by confidence threshold first
				mask = confs >= args.min_conf
				xyxy = xyxy[mask]
				confs = confs[mask]
				clss = clss[mask]

				# Additional class-agnostic merge to remove overlapping duplicates across classes
				keep_indices = class_agnostic_merge(xyxy, confs, clss, args.merge_iou)
				for idx in keep_indices:
					x1, y1, x2, y2 = xyxy[idx]
					conf = confs[idx]
					cls = clss[idx]
					name = names.get(cls, str(cls))
					color = colors.get(cls, (255, 255, 255))
					x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
					if conf >= args.solid_thr:
						cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
					else:
						draw_dashed_rect(frame, (x1i, y1i), (x2i, y2i), color, 2)
					label = f"{name} {conf*100:.1f}%"
					put_label(frame, label, x1i, y1i if y1i > 20 else y1i + 20, color)

			# FPS (smoothed)
			now = time.time()
			dt = now - last_time
			if dt > 0:
				fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
			last_time = now
			cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
			cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

			cv2.imshow(window, frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
	finally:
		grabber.stop()
		cv2.destroyAllWindows()
	return 0


def build_argparser():
	p = argparse.ArgumentParser(description="Live YOLO inference on MJPEG stream with dashed low-confidence boxes")
	p.add_argument('--model', default='ECE4078B1_Modelv1_FixedNames.pt', help='Path to YOLO model .pt file')
	p.add_argument('--url', default='http://192.168.50.1:8080/camera/get', help='MJPEG stream URL')
	p.add_argument('--solid-thr', type=float, default=0.5, dest='solid_thr', help='Confidence threshold for solid box (>=)')
	p.add_argument('--min-conf', type=float, default=0.25, dest='min_conf', help='Minimum confidence to display')
	p.add_argument('--class-colors', type=str, default=None, help='Comma list like "0:#FF0000,1:#00FF00" for custom colors')
	p.add_argument('--merge-iou', type=float, default=0.5, dest='merge_iou', help='Class-agnostic IoU threshold to merge overlapping boxes (<=0 disables)')
	p.add_argument('--device', type=str, default='auto', help="Device: 'auto', 'cpu', 'cuda:0', 'mps', etc.")
	return p


if __name__ == '__main__':
	exit_code = run(build_argparser().parse_args())
	sys.exit(exit_code)
