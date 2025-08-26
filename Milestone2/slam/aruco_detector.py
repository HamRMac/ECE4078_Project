# detect ARUCO markers and estimate their positions
import numpy as np
import cv2
import os, sys

sys.path.insert(0, "{}/util".format(os.getcwd()))
import util.measure as measure

class aruco_detector:

    # Tags of markers that are used
    KNOWN_TAGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __init__(self, robot, marker_length=0.07, cube_depth=None):
        self.camera_matrix = robot.camera_matrix
        self.distortion_params = robot.camera_dist

        self.marker_length = marker_length
        # Depth of physical cube carrying the marker; if None assume same as marker side length
        self.cube_depth = cube_depth if cube_depth is not None else marker_length
        self.aruco_params = cv2.aruco.DetectorParameters() # updated to work with newer OpenCV
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100) # updated to work with newer OpenCV

    # ---------------- Internal helpers ----------------
    def _tvec_to_body2d_centered(self, rvec, tvec):
        """Convert a single marker pose (rvec,tvec) into 2D body-frame position of the cube CENTER.

        OpenCV returns tvec to the center of the tag plane. For a cube we shift along the marker
        plane normal (negative, into the cube) by cube_depth/2 to reach the cube's geometric center.
        We then project to the robot body frame 2D using [z ; -x] convention used elsewhere.
        """
        # Rotation matrix to get marker normal in camera frame
        R, _ = cv2.Rodrigues(rvec)
        n_cam = R[:, 2]              # Marker +Z axis points from marker plane toward camera
        center_tvec = tvec.reshape(3, 1) - (self.cube_depth / 2.0) * n_cam.reshape(3, 1)
        # Map to 2D robot body frame: forward = camera Z, left = -camera X
        lm_bff2d = np.array([[center_tvec[2, 0]], [-center_tvec[0, 0]]])
        return lm_bff2d

    def _aggregate_measurements(self, ids, rvecs, tvecs):
        """Aggregate per-id measurements to 2D body frame positions using cube-centered offset."""
        measurements = []
        if ids is None:
            return measurements
        seen_ids = []
        for idx, idi in enumerate(ids.flatten()):
            if idi in seen_ids or idi not in self.KNOWN_TAGS:
                continue
            seen_ids.append(idi)
            # All detections of this same id (in case of duplicates)
            mask = (ids.flatten() == idi)
            # Average cube-centered positions of duplicates
            lm_positions = []
            for j, cond in enumerate(mask):
                if not cond: continue
                lm_positions.append(self._tvec_to_body2d_centered(rvecs[j], tvecs[j]))
            if not lm_positions:
                continue
            lm_bff2d = np.mean(np.hstack(lm_positions), axis=1, keepdims=True)
            measurements.append(measure.Marker(lm_bff2d, int(idi)))
        return measurements
    
    def detect_marker_positions(self, img):
        # Perform detection (legacy path)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params)
        if ids is None:
            return [], img
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.distortion_params)
        # Compute marker cube-centered positions (replaces previous incorrect planar radial offset)
        measurements = self._aggregate_measurements(ids, rvecs, tvecs)
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)
        return measurements, img_marked

    def detect(self, img): # A new implementation using the OpenCV Detector API
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return [], img
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.distortion_params)
        # Use same cube-centered aggregation as legacy path
        measurements = self._aggregate_measurements(ids, rvecs, tvecs)
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)
        return measurements, img_marked
