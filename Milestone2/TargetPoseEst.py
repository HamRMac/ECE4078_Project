# estimate the pose of target objects detected
import numpy as np
import json
import os
import ast
import cv2
from typing import List, Dict, Tuple, Optional
from YOLO.detector import Detector
from sklearn.cluster import DBSCAN


# nominal true sizes (width, depth, height) in metres
TARGET_DIMENSIONS_DICT: Dict[str, Tuple[float, float, float]] = {
    'orange': (0.07, 0.07, 0.073),
    'lemon': (0.078, 0.053, 0.05),
    'pear': (0.076, 0.074, 0.11),
    'tomato': (0.065, 0.065, 0.06),
    'capsicum': (0.076, 0.074, 0.09),
    'potato': (0.095, 0.065, 0.07),
    'pumpkin': (0.08, 0.08, 0.08),
    'garlic': (0.065, 0.06, 0.07),
    'lime': (0.074, 0.052, 0.05),
}


class FruitRanger:
    """Compute range/bearing (and uncertainties) for fruit detections.

    Provides a prior-from-height method and a stub for ground-ray. Also fuses
    multiple (r,theta) measurements into a single (r*,theta*) with a 2x2
    covariance in Cartesian camera-frame coordinates.
    """

    def __init__(self,
                 pixel_centroid_sigma_px: float = 2.0,
                 pixel_height_sigma_px: float = 3.0,
                 range_scale_beta: float = 0.02,
                 ekf_weight_gamma: float = 1.0) -> None:
        self.pixel_centroid_sigma_px = float(pixel_centroid_sigma_px)
        self.pixel_height_sigma_px = float(pixel_height_sigma_px)
        self.range_scale_beta = float(range_scale_beta)
        self.ekf_weight_gamma = float(ekf_weight_gamma)

    def from_bbox_height(self,
                         camera_matrix: np.ndarray,
                         bbox: List[float] | Tuple[float, float, float, float],
                         true_height_m: float) -> Optional[Dict[str, float]]:
        """Estimate r, theta and uncertainties from bbox height.

        bbox is [x, y, w, h] in pixels (top-left origin).
        Returns dict with keys: r, theta, sigma_r, sigma_theta, x, y (camera frame)
        or None if invalid inputs.
        """
        if camera_matrix is None:
            return None
        try:
            x, y, w, h = [float(v) for v in bbox]
        except Exception:
            return None

        f = float(camera_matrix[0, 0])
        cx = float(camera_matrix[0, 2]) if camera_matrix.shape[1] >= 3 else 160.0

        if h <= 0 or true_height_m <= 0:
            return None

        # range from height prior
        r = (true_height_m / h) * f

        # centroid location
        x_c = x + w / 2.0
        # bearing: positive = left of centre (cx - x_c)
        theta = float(np.arctan2(cx - x_c, f))

        # uncertainty propagation
        dr_dh = abs((-f * true_height_m) / (h * h))
        sigma_r_from_h = dr_dh * self.pixel_height_sigma_px
        sigma_r = float(np.sqrt(sigma_r_from_h ** 2 + (self.range_scale_beta * r * r)))
        sigma_theta = float(self.pixel_centroid_sigma_px / f)

        # Cartesian camera frame (x forward, y left)
        x_cam = float(r * np.cos(theta))
        y_cam = float(r * np.sin(theta))

        return {
            'r': float(r),
            'theta': float(theta),
            'sigma_r': sigma_r,
            'sigma_theta': sigma_theta,
            'x': x_cam,
            'y': y_cam,
        }

    def from_ground_ray(self,
                        camera_matrix: np.ndarray,
                        bbox: List[float] | Tuple[float, float, float, float],
                        camera_height_m: float,
                        camera_pitch_rad: float) -> Dict:
        """Stub for ground-ray method: back-project bottom pixel to ground.
        Not implemented yet.
        """
        raise NotImplementedError("Ground-ray method not implemented")

    def fuse(self,
             measurements: List[Dict[str, float]],
             ekf_pose_var: float = 0.0) -> Optional[Dict[str, object]]:
        """Fuse multiple measurements (r,theta) into (r*,theta*) and cov.

        measurements: list of dicts from from_bbox_height
        ekf_pose_var: scalar variance proxy from EKF robot pose (adds to denom)
        Returns dict with keys: r, theta, x, y, cov (2x2 numpy)
        """
        if not measurements:
            return None

        xs = []
        ys = []
        ws = []

        for m in measurements:
            r = float(m['r'])
            theta = float(m['theta'])
            sr = float(m['sigma_r'])
            st = float(m['sigma_theta'])

            # Cartesian
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # conservative variance proxies for x,y
            var_x = (np.cos(theta) ** 2) * (sr ** 2) + ((r * np.sin(theta)) ** 2) * (st ** 2)
            var_y = (np.sin(theta) ** 2) * (sr ** 2) + ((r * np.cos(theta)) ** 2) * (st ** 2)
            var_xy = var_x + var_y
            weight = 1.0 / (var_xy + self.ekf_weight_gamma * float(ekf_pose_var) + 1e-12)

            xs.append(x)
            ys.append(y)
            ws.append(weight)

        ws = np.array(ws, dtype=float)
        if ws.sum() <= 0:
            return None

        wnorm = ws / ws.sum()
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        x_mean = float((wnorm * xs).sum())
        y_mean = float((wnorm * ys).sum())

        # weighted covariance (2x2)
        dx = xs - x_mean
        dy = ys - y_mean
        cov_xx = float((wnorm * dx * dx).sum())
        cov_xy = float((wnorm * dx * dy).sum())
        cov_yy = float((wnorm * dy * dy).sum())
        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)

        r_star = float(np.hypot(x_mean, y_mean))
        theta_star = float(np.arctan2(y_mean, x_mean))

        return {
            'r': r_star,
            'theta': theta_star,
            'x': x_mean,
            'y': y_mean,
            'cov': cov,
        }


# list of target fruits and vegs types
# Make sure the names are the same as the ones used in your YOLO model
TARGET_TYPES = ['orange', 'lemon', 'lime', 'tomato', 'capsicum', 'potato', 'pumpkin', 'garlic']


def estimate_pose(camera_matrix, obj_info, robot_pose):
    """
    function:
        estimate the pose of a target based on size and location of its bounding box and the corresponding robot pose
    input:
        camera_matrix: list, the intrinsic matrix computed from camera calibration (read from 'param/intrinsic.txt')
            |f_x, s,   c_x|
            |0,   f_y, c_y|
            |0,   0,   1  |
            (f_x, f_y): focal length in pixels
            (c_x, c_y): optical centre in pixels
            s: skew coefficient (should be 0 for PenguinPi)
        obj_info: list, an individual bounding box in an image (generated by get_bounding_box, [label,[x,y,width,height]])
        robot_pose: list, pose of robot corresponding to the image (read from 'lab_output/images.txt', [x,y,theta])
    output:
        target_pose: dict, prediction of target pose
    """
    # read in camera matrix (from camera calibration results)
    focal_length = camera_matrix[0][0]

    # there are 8 possible types of fruits and vegs
    ######### Replace with your codes #########
    # TODO: measure actual sizes of targets [width, depth, height] and update the dictionary of true target dimensions
    target_dimensions_dict = {'orange': [0.07,0.07,0.073], 'lemon': [0.078,0.053,0.05], 
                              'pear': [0.076,0.074,0.11], 'tomato': [0.065,0.065,0.06], 
                              'capsicum': [0.076,0.074,0.09], 'potato': [0.095,0.065,0.07], 
                              'pumpkin': [0.08,0.08,0.08], 'garlic': [0.065,0.06,0.07]}  # 'lime': [0.074,0.052,0.05]
    #########

    # estimate target pose using bounding box and robot pose
    target_class = obj_info[0]     # get predicted target label of the box
    target_box = obj_info[1]       # get bounding box measures: [x,y,width,height]
    true_height = target_dimensions_dict[target_class][2]   # look up true height of by class label

    # compute pose of the target based on bounding box info, true object height, and robot's pose
    pixel_height = target_box[3]
    pixel_center = target_box[0]
    distance = true_height/pixel_height * focal_length  # estimated distance between the robot and the centre of the image plane based on height
    # training image size 320x240p
    image_width = 320 # change this if your training image is in a different size (check details of pred_0.png taken by your robot)
    x_shift = image_width/2 - pixel_center              # x distance between bounding box centre and centreline in camera view
    theta = np.arctan(x_shift/focal_length)     # angle of object relative to the robot
    ang = theta + robot_pose[2]     # angle of object in the world frame
    
   # relative object location
    distance_obj = distance/np.cos(theta) # relative distance between robot and object
    x_relative = distance_obj * np.cos(theta) # relative x pose
    y_relative = distance_obj * np.sin(theta) # relative y pose
    relative_pose = {'x': x_relative, 'y': y_relative}
    #print(f'relative_pose: {relative_pose}')

    # location of object in the world frame using rotation matrix
    delta_x_world = x_relative * np.cos(robot_pose[2]) - y_relative * np.sin(robot_pose[2])
    delta_y_world = x_relative * np.sin(robot_pose[2]) + y_relative * np.cos(robot_pose[2])
    # add robot pose with delta target pose
    target_pose = {'y': (robot_pose[1]+delta_y_world)[0],
                   'x': (robot_pose[0]+delta_x_world)[0]}
    #print(f'delta_x_world: {delta_x_world}, delta_y_world: {delta_y_world}')
    #print(f'target_pose: {target_pose}')

    return target_pose


def merge_estimations(target_pose_dict):
    """
    function:
        merge estimations of the same target
    input:
        target_pose_dict: dict, generated by estimate_pose
    output:
        target_est: dict, target pose estimations after merging
    """
    target_est: Dict[str, Dict[str, float]] = {}

    if not target_pose_dict:
        return target_est

    # collect per-class points
    per_class: Dict[str, List[Tuple[float, float]]] = {}
    for key, pos in target_pose_dict.items():
        # expect keys like 'tomato_0' -> base label before last underscore
        label = key.rsplit('_', 1)[0]
        x = float(pos['x'])
        y = float(pos['y'])
        per_class.setdefault(label, []).append((x, y))

    # cluster per class using DBSCAN and keep up to 3 largest clusters
    for label, pts in per_class.items():
        pts_arr = np.array(pts, dtype=float)
        if pts_arr.shape[0] == 0:
            continue
        if pts_arr.shape[0] == 1:
            # single detection -> keep as is
            target_est[f"{label}_0"] = {'x': float(pts_arr[0, 0]), 'y': float(pts_arr[0, 1])}
            continue

        # DBSCAN parameters: eps in metres, min_samples=1 to ensure singletons form their own cluster
        clustering = DBSCAN(eps=0.25, min_samples=1).fit(pts_arr)
        labels = clustering.labels_
        unique_labels = sorted(set(labels), key=lambda l: -sum(labels == l))

        clusters = []
        for ul in unique_labels:
            members = pts_arr[labels == ul]
            if members.size == 0:
                continue
            centroid = members.mean(axis=0)
            clusters.append((len(members), centroid))

        # sort clusters by size (desc) and cap to 3
        clusters.sort(key=lambda t: -t[0])
        for i, (_size, centroid) in enumerate(clusters[:3]):
            target_est[f"{label}_{i}"] = {'x': float(centroid[0]), 'y': float(centroid[1])}

    return target_est


# main loop
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))     # get current script directory (TargetPoseEst.py)

    # read in camera matrix
    fileK = f'{script_dir}/calibration/param/intrinsic.txt'
    camera_matrix = np.loadtxt(fileK, delimiter=',')

    # init YOLO model
    model_path = f'{script_dir}/YOLO/model/yolov8_model.pt'
    yolo = Detector(model_path)

    # instantiate ranger
    ranger = FruitRanger(pixel_centroid_sigma_px=2.0,
                         pixel_height_sigma_px=3.0,
                         range_scale_beta=0.02,
                         ekf_weight_gamma=1.0)

    # create a dictionary of all the saved images with their corresponding robot pose
    image_poses = {}
    with open(f'{script_dir}/lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # estimate pose of targets in each image
    target_pose_dict = {}
    detected_type_list: List[str] = []

    # measurements buffered per class (to fuse across detections)
    meas_by_class: Dict[str, List[Dict]] = {}

    for image_path in image_poses.keys():
        input_image = cv2.imread(image_path)
        bounding_boxes, bbox_img = yolo.detect_single_image(input_image)
        robot_pose = image_poses[image_path]

        # reset per-image measurements (optional: keep across frames by moving this)
        meas_by_class.clear()

        for detection in bounding_boxes:
            target_class = detection[0]
            bbox = detection[1]

            # basic checks: class in target list
            if target_class not in TARGET_TYPES:
                continue

            # look up true height (use third entry if available)
            true_height = TARGET_DIMENSIONS_DICT.get(target_class, TARGET_DIMENSIONS_DICT['tomato'])[2]

            # compute measurement from bbox height prior
            m = ranger.from_bbox_height(camera_matrix, bbox, true_height)
            if m is None:
                continue

            meas_by_class.setdefault(target_class, []).append(m)

            # fuse now (could also fuse after accumulating many frames)
            ekf_pose_var = 0.0
            fused = ranger.fuse(meas_by_class[target_class], ekf_pose_var=ekf_pose_var)
            if fused is None:
                continue

            # fused x,y are in camera frame (x forward, y left)
            x_cam = float(fused['x'])
            y_cam = float(fused['y'])

            # rotate into world using robot pose
            th = float(robot_pose[2])
            dx = x_cam * np.cos(th) - y_cam * np.sin(th)
            dy = x_cam * np.sin(th) + y_cam * np.cos(th)

            pos = {
                'x': float(robot_pose[0] + dx),
                'y': float(robot_pose[1] + dy)
            }

            # arena check (reuse same ±1.35 m bounds as elsewhere if needed)
            # keep existing filters (assuming caller has implemented them)

            occurrence = detected_type_list.count(target_class)
            target_pose_dict[f'{target_class}_{occurrence}'] = pos
            detected_type_list.append(target_class)

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = {}
    target_est = merge_estimations(target_pose_dict)
    print(target_est)
    # save target pose estimations
    with open(f'{script_dir}/lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo, indent=4)

    print('Estimations saved!')
