# estimate the pose of target objects detected
import numpy as np
import json
import os
import ast
import cv2

import matplotlib
matplotlib.use("TkAgg")  # Set backend to TkAgg

from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MultipleLocator
import matplotlib.image as mpimg

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

# Default to None
yolo = None

# list of target fruits and vegs types
# Make sure the names are the same as the ones used in your YOLO model
TARGET_TYPES = ['orange', 'lemon', 'pear', 'tomato', 'capsicum', 'potato', 'pumpkin', 'garlic']
ARENA_SIZE  = 2.7
ARENA_BOUND = ARENA_SIZE / 2.0  # ±1.35 m


def load_map(map_source):
    """
    Load a SLAM map either from a filepath or a pre-loaded dict.

    Expected format (JSON/dict):
      {
        "taglist": [tag_id0, tag_id1, ...],
        "map": [[x0, x1, ...], [y0, y1, ...]]
      }

    Returns: (taglist, positions) where positions is a list of (x, y) in the
    same index order as taglist.
    """
    if map_source is None:
        return None, None
    if isinstance(map_source, str) and os.path.exists(map_source):
        with open(map_source, 'r') as f:
            data = json.load(f)
    elif isinstance(map_source, dict):
        data = map_source
    else:
        raise ValueError("map_source must be a filepath or a dict matching the SLAM map format")

    taglist = data.get('taglist', [])
    m = data.get('map', [])
    if not isinstance(taglist, list) or not isinstance(m, list) or len(m) != 2:
        raise ValueError("Invalid map format: expected 'taglist' and 'map' with shape [2, N]")
    xs, ys = m[0], m[1]
    if len(xs) != len(ys) or len(xs) != len(taglist):
        raise ValueError("Invalid map lengths: |xs|, |ys|, and |taglist| must match")
    positions = [(float(xs[i]), float(ys[i])) for i in range(len(taglist))]
    return taglist, positions


def plot_targets_and_markers(target_est, taglist, positions, out_path,
                             marker_png_dir, fruit_diameter_m=0.08):
    """
    Plot estimated targets and SLAM markers in world coordinates.

    - target_est: dict like {"class_idx": {"x": float, "y": float}, ...}
    - taglist: list of tag ids
    - positions: list of (x, y) matching taglist indices
    - out_path: file to save the figure
    - marker_png_dir: directory containing lm_{tag}.png and lm_unknown.png
    - fruit_diameter_m: used to size marker icons to match fruit size
    """
    if taglist is None or positions is None:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-ARENA_BOUND, ARENA_BOUND)
    ax.set_ylim(-ARENA_BOUND, ARENA_BOUND)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Targets and Markers (world frame)')

    # Plot targets as circles with diameter matching fruit_diameter_m
    fruit_radius = fruit_diameter_m / 2.0
    for key, pose in (target_est or {}).items():
        x, y = float(pose['x']), float(pose['y'])
        circ = plt.Circle((x, y), fruit_radius, color='tab:red', alpha=0.6)
        ax.add_patch(circ)
        ax.text(x, y + fruit_radius + 0.02, key, ha='center', va='bottom', fontsize=8)

    # Plot markers using PNG icons at the same extent size as fruit
    for idx, tag in enumerate(taglist):
        x, y = positions[idx]
        png_path = os.path.join(marker_png_dir, f"lm_{tag}.png")
        if not os.path.exists(png_path):
            png_path = os.path.join(marker_png_dir, "lm_unknown.png")
        try:
            img = mpimg.imread(png_path)
            half = fruit_radius
            ax.imshow(img, extent=[x - half, x + half, y - half, y + half],
                      origin='center', zorder=3)
        except Exception:
            # Fallback to a square marker if image fails to load
            ax.plot(x, y, marker='s', color='tab:blue')
        ax.text(x, y - fruit_radius - 0.02, f"LM {tag}", ha='center', va='top', fontsize=8)

    # Arena boundary
    ax.add_patch(plt.Rectangle((-ARENA_BOUND, -ARENA_BOUND), ARENA_SIZE, ARENA_SIZE,
                               fill=False, linewidth=1.5, color='k', alpha=0.5))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

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
    image_width = 384 # change this if your training image is in a different size (check details of pred_0.png taken by your robot)
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


def merge_estimations(target_pose_dict, eps=0.15, min_samples=2):

    print("Using clustering distance of: " + str(eps) + "m")

    """
    function:
        merge estimations of the same target
    input:
        target_pose_dict: dict, generated by estimate_pose
    output:
        target_est: dict, target pose estimations after merging

    need to add outter bound of 2.7m
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
    print(f'Clustering with eps={eps}, min_samples={min_samples}')
    for label, pts in per_class.items():
        pts_arr = np.array(pts, dtype=float)
        if pts_arr.shape[0] == 0:
            continue
        
        print(f'Clustering class {label}: pts_arr.shape={pts_arr.shape}')
        # DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_arr)
        labels = clustering.labels_
        #print(f'  labels: {labels}')
        unique_labels = sorted(set(labels), key=lambda l: -sum(labels == l))

        clusters = []
        for ul in unique_labels:
            if ul < 0:
                # Skip noise points
                continue
            # Get members of this cluster
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_file', type=str, default=None,
                        help='Path to SLAM map JSON (default: lab_output/slam.txt if exists)')
    parser.add_argument('--no_plot', action='store_true', help='Disable plotting of markers/targets')
    args, _ = parser.parse_known_args()

    # Adding clustering distance (cm)
    parser.add_argument('-dist', type=float, default=0.15,
                        help='Clustering distance in m (default: 15)')
    args, _ = parser.parse_known_args()

    clustering_distance = args.dist


    script_dir = os.path.dirname(os.path.abspath(__file__))     # get current script directory (TargetPoseEst.py)

    # read in camera matrix
    fileK = f'{script_dir}/calibration/param/intrinsic.txt'
    camera_matrix = np.loadtxt(fileK, delimiter=',')

    # init YOLO model
    model_path = f'{script_dir}/YOLO/model/V5.pt'
    print(f'>> Loading YOLO model from {model_path}')
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

    # estimate pose of targets using batched detection for speed
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
            th = (robot_pose[2])
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
    target_est = merge_estimations(target_pose_dict, eps=clustering_distance, min_samples=3)
    print(target_est)
    # save target pose estimations
    with open(f'{script_dir}/lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo, indent=4)

    # Optional: load a provided SLAM map and plot markers
    if not args.no_plot:
        taglist, positions = (None, None)
        map_path = args.map_file
        if map_path is None:
            candidate = f'{script_dir}/lab_output/slam.txt'
            map_path = candidate if os.path.exists(candidate) else None
        try:
            if map_path is not None:
                taglist, positions = load_map(map_path)
                out_plot = f'{script_dir}/lab_output/targets_map.png'
                marker_png_dir = f'{script_dir}/pics/8bit'
                plot_targets_and_markers(target_est, taglist, positions, out_plot, marker_png_dir)
                print(f'Map plot saved to {out_plot}')
        except Exception as e:
            print(f'Warning: map plotting skipped: {e}')

    print('Estimations saved!')

    # --------------------
    # Plot detections in arena (2.4m x 2.4m), AR=1
    # - Pre-clustering points: reduced opacity
    # - Post-clustering points: full opacity
    # - Colors per class from yolo.class_colour
    # --------------------
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-ARENA_BOUND, ARENA_BOUND)
        ax.set_ylim(-ARENA_BOUND, ARENA_BOUND)
        ax.set_aspect('equal', adjustable='box')  # ensure AR = 1
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Fruit Positions: pre (transparent) vs post (solid) clustering')

        # Grid: small 0.3 m (minor), large 0.9 m (major)
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax.grid(which='major', linestyle='-', linewidth=0.8, color='0.7')
        ax.grid(which='minor', linestyle=':', linewidth=0.5, color='0.85')

        # Helper to convert OpenCV BGR (0-255) to Matplotlib RGB (0-1)
        def bgr_to_rgb01(bgr):
            return (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)

        # Plot pre-clustering detections with reduced opacity
        pre_by_class = {}
        for key, pose in target_pose_dict.items():
            cls = key.split('_')[0]
            pre_by_class.setdefault(cls, []).append((pose['x'], pose['y']))

        for cls, pts in pre_by_class.items():
            color_rgb = bgr_to_rgb01(yolo.class_colour.get(cls, (128, 128, 128)))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            # Pre-cluster points: small size, transparent
            ax.scatter(xs, ys, c=[color_rgb], alpha=0.35, s=25, edgecolors='none')

        # Prepare post-clustering points grouped by class
        post_by_class = {}
        for key, pose in target_est.items():
            cls = key.split('_')[0]
            post_by_class.setdefault(cls, []).append((pose['x'], pose['y']))

        # Draw cluster outlines (convex hull per DBSCAN cluster of pre-cluster points)
        eps, min_samples = 0.15, 2  # keep in sync with merge_estimations defaults
        for cls, pts in pre_by_class.items():
            if len(pts) < 1:
                continue
            pts_np = np.array(pts, dtype=np.float32)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_np).labels_
            color_rgb = bgr_to_rgb01(yolo.class_colour.get(cls, (128, 128, 128)))

            for cid in np.unique(labels):
                if cid == -1:
                    continue  # skip noise
                cluster_pts = pts_np[labels == cid]
                if len(cluster_pts) < 3:
                    # Not enough points for a hull: emphasize with a ring marker
                    ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=80,
                               facecolors='none', edgecolors=color_rgb, linewidths=1.0)
                    continue
                hull = cv2.convexHull(cluster_pts.reshape(-1, 1, 2))
                hull = hull.squeeze()
                hx = np.r_[hull[:, 0], hull[0, 0]]
                hy = np.r_[hull[:, 1], hull[0, 1]]
                ax.plot(hx, hy, color=color_rgb, linewidth=1.2)
                ax.fill(hull[:, 0], hull[:, 1], color=color_rgb, alpha=0.10)

        # Load pixel art icons
        pixel_art_dir = os.path.join(script_dir, 'pixel_art')
        pixel_icons = {}
        if os.path.isdir(pixel_art_dir):
            for fname in os.listdir(pixel_art_dir):
                if not fname.lower().endswith('.png'):
                    continue
                key = os.path.splitext(fname)[0].lower()
                try:
                    img = mpimg.imread(os.path.join(pixel_art_dir, fname))
                    # Ensure image is in RGBA or RGB float format
                    pixel_icons[key] = img
                except Exception as _:
                    pass

        # Overlay pixel art at final (post-cluster) locations; fallback to dot if missing
        ICON_SIZE_PX = 10  # target visual size for the longest image side

        def add_icon(ax, xy, img, size_px=ICON_SIZE_PX, z=5):
            h, w = img.shape[:2]
            scale = size_px / float(max(h, w))
            oi = OffsetImage(img, zoom=scale)
            ab = AnnotationBbox(oi, xy, frameon=False, pad=0.0, box_alignment=(0.5, 0.5),
                                annotation_clip=True, zorder=z)
            ax.add_artist(ab)

        # Draw cluster outlines first (already drawn above), then icons on top
        for cls, pts in post_by_class.items():
            icon = pixel_icons.get(cls.lower())
            if icon is None:
                # Fallback to solid dots for classes without an icon
                color_rgb = bgr_to_rgb01(yolo.class_colour.get(cls, (64, 64, 64)))
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.scatter(xs, ys, c=[color_rgb], alpha=1.0, s=50, edgecolors='black', linewidths=0.5, zorder=4)
                continue

            for (x, y) in pts:
                add_icon(ax, (x, y), icon)

        plt.show()
    except Exception as e:
        print(f'Plotting failed: {e}')
