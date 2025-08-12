# detect ARUCO markers and estimate their positions
import numpy as np
import cv2
import os, sys

sys.path.insert(0, "{}/util".format(os.getcwd()))
import util.measure as measure

class aruco_detector:

    # Tags of markers that are used
    KNOWN_TAGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __init__(self, robot, marker_length=0.07):
        self.camera_matrix = robot.camera_matrix
        self.distortion_params = robot.camera_dist

        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters() # updated to work with newer OpenCV
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100) # updated to work with newer OpenCV
    
    def detect_marker_positions(self, img):
        # Perform detection
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.distortion_params)
        # rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.distortion_params) # use this instead if you got a value error

        if ids is None:
            return [], img

        # Compute the marker positions
        measurements = []
        seen_ids = []
        for i in range(len(ids)):
            idi = ids[i,0]
            # Some markers appear multiple times but should only be handled once.
            if idi in seen_ids:
                continue
            elif idi not in self.KNOWN_TAGS:
                continue
            else:
                seen_ids.append(idi)

            lm_tvecs = tvecs[ids==idi].T
            lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
            lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)

            lm_measurement = measure.Marker(lm_bff2d, idi)
            measurements.append(lm_measurement)
        
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked

    def detect(self, img): # A new implementation using the OpenCV Detector API
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
        measurements = []

        if ids is None or len(ids) == 0:
            return measurements, img

        # Pose (per marker)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.distortion_params)

        # Group by id, convert tvecs to robot body frame 2D: [z; -x], avg if multi-detection
        seen_ids = []
        for idi in ids.flatten():
            if idi in seen_ids: continue
            seen_ids.append(idi)

            lm_tvecs = tvecs[ids.flatten()==idi].reshape(-1,3).T
            lm_bff2d = np.vstack([ lm_tvecs[2,:], -lm_tvecs[0,:] ]).mean(axis=1, keepdims=True)
            measurements.append(measure.Marker(lm_bff2d, int(idi)))

        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)
        return measurements, img_marked
