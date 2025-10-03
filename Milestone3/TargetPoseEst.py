# estimate the pose of target objects detected
import numpy as np
import json
import os
import ast
import cv2
from typing import List, Dict, Tuple, Optional
# plotting (added for visualising detections)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from YOLO.detector import Detector
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable

# -----------------------------
# Config / constants
# -----------------------------
TARGET_TYPES = ['orange', 'lemon', 'pear', 'tomato', 'capsicum', 'potato', 'pumpkin', 'garlic']
ARENA_SIZE  = 2.7
ARENA_BOUND = ARENA_SIZE / 2.0  # ±1.35 m

# nominal true sizes (width, depth, height) in metres
TARGET_DIMENSIONS_DICT: Dict[str, Tuple[float, float, float]] = {
    'orange':  (0.07,  0.07,  0.073),
    'lemon':   (0.078, 0.053, 0.05),
    'pear':    (0.076, 0.074, 0.11),
    'tomato':  (0.065, 0.065, 0.06),
    'capsicum':(0.076, 0.074, 0.09),
    'potato':  (0.095, 0.065, 0.07),
    'pumpkin': (0.08,  0.08,  0.08),
    'garlic':  (0.065, 0.06,  0.07),
    'lime':    (0.074, 0.052, 0.05),
}

# -----------------------------
# Helpers (imported)
# -----------------------------
from perception.fruit_ranger import (
    FruitRanger,
    is_inside_arena,
    expected_ratio_for_class,
    bbox_ratio_ok,
)

# FruitRanger class moved to perception/fruit_ranger.py

# -----------------------------
# Map/plot utilities
# -----------------------------
def load_map(map_source):
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
    if taglist is None or positions is None:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-ARENA_BOUND, ARENA_BOUND)
    ax.set_ylim(-ARENA_BOUND, ARENA_BOUND)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('Targets and Markers (world frame)')

    fruit_radius = fruit_diameter_m / 2.0
    for key, pose in (target_est or {}).items():
        x, y = float(pose['x']), float(pose['y'])
        circ = plt.Circle((x, y), fruit_radius, color='tab:red', alpha=0.6)
        ax.add_patch(circ)
        ax.text(x, y + fruit_radius + 0.02, key, ha='center', va='bottom', fontsize=8)

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
            ax.plot(x, y, marker='s', color='tab:blue')
        ax.text(x, y - fruit_radius - 0.02, f"LM {tag}", ha='center', va='top', fontsize=8)


# -----------------------------
# Legacy height-only world pose (kept for ref; not used in main)
# -----------------------------
def estimate_pose(camera_matrix, obj_info, robot_pose):
    focal_length = camera_matrix[0][0]
    target_dimensions_dict = {'orange': [0.07,0.07,0.073], 'lemon': [0.078,0.053,0.05],
                              'pear': [0.076,0.074,0.11], 'tomato': [0.065,0.065,0.06],
                              'capsicum': [0.076,0.074,0.09], 'potato': [0.095,0.065,0.07],
                              'pumpkin': [0.08,0.08,0.08], 'garlic': [0.065,0.06,0.07]}
    target_class = obj_info[0]
    target_box = obj_info[1]
    true_height = target_dimensions_dict[target_class][2]
    pixel_height = target_box[3]; pixel_center = target_box[0]
    distance = true_height/pixel_height * focal_length
    image_width = 384
    x_shift = image_width/2 - pixel_center
    theta = np.arctan(x_shift/focal_length)
    distance_obj = distance/np.cos(theta)
    x_relative = distance_obj * np.cos(theta)
    y_relative = distance_obj * np.sin(theta)
    delta_x_world = x_relative * np.cos(robot_pose[2]) - y_relative * np.sin(robot_pose[2])
    delta_y_world = x_relative * np.sin(robot_pose[2]) + y_relative * np.cos(robot_pose[2])
    target_pose = {'y': (robot_pose[1]+delta_y_world)[0], 'x': (robot_pose[0]+delta_x_world)[0]}
    return target_pose

# -----------------------------
# Merging (DBSCAN) with arena filter
# -----------------------------
def merge_estimations(target_pose_dict: Dict[str, Dict[str, float]],
                      eps: float = 0.25, min_samples: int = 1,
                      max_clusters_per_class: int = 3) -> Dict[str, Dict[str, float]]:
    """DBSCAN per class, arena-filtered, keep up to 3 largest clusters."""
    target_est: Dict[str, Dict[str, float]] = {}
    if not target_pose_dict:
        return target_est

    # Collect per class
    per_class: Dict[str, List[Tuple[float, float]]] = {}
    for key, pos in target_pose_dict.items():
        label = key.rsplit('_', 1)[0]
        x, y = float(pos['x']), float(pos['y'])
        if is_inside_arena(x, y):
            per_class.setdefault(label, []).append((x, y))

    # Cluster and keep top-K
    for label, pts in per_class.items():
        pts_arr = np.array(pts, dtype=float)
        if pts_arr.shape[0] == 0:
            continue
        if pts_arr.shape[0] == 1:
            target_est[f"{label}_0"] = {'x': float(pts_arr[0, 0]), 'y': float(pts_arr[0, 1])}
            continue

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_arr).labels_
        clusters = []
        for cid in np.unique(labels):
            if cid == -1:
                continue
            members = pts_arr[labels == cid]
            if members.size == 0:
                continue
            centroid = members.mean(axis=0)
            clusters.append((len(members), centroid))
        clusters.sort(key=lambda t: -t[0])
        for i, (_n, c) in enumerate(clusters[:max_clusters_per_class]):
            target_est[f"{label}_{i}"] = {'x': float(c[0]), 'y': float(c[1])}
    return target_est

# -----------------------------
# Main loop
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_file', type=str, default=None,
                        help='Path to SLAM map JSON (default: lab_output/slam.txt if exists)')
    parser.add_argument('--no_plot', action='store_true', help='Disable plotting of markers/targets')
    args, _ = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Camera intrinsics
    fileK = f'{script_dir}/calibration/param/intrinsic.txt'
    camera_matrix = np.loadtxt(fileK, delimiter=',')

    # YOLO
    model_path = f'{script_dir}/YOLO/model/af8.pt'
    print(f'>> Loading YOLO model from {model_path}')
    yolo = Detector(model_path)

    # FruitRanger
    ranger = FruitRanger(pixel_centroid_sigma_px=2.0,
                         pixel_height_sigma_px=3.0,
                         range_scale_beta=0.02,
                         ekf_weight_gamma=1.0)

    # Load image→pose index
    image_poses: Dict[str, np.ndarray] = {}
    with open(f'{script_dir}/lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # Prepare images (batched detection)
    target_pose_dict: Dict[str, Dict[str, float]] = {}
    detected_type_list: List[str] = []

    image_paths = list(image_poses.keys())
    images = [cv2.imread(p) for p in image_paths]

    MAX_BATCH = 8
    batch_results = yolo.detect_images(images, max_batch=MAX_BATCH)

    # Optional: fuse measurements across ALL images per class
    meas_by_class: Dict[str, List[Dict[str, float]]] = {}

    for idx, (bounding_boxes, bbox_img) in enumerate(
        tqdm(batch_results, total=len(batch_results), desc="Processing detections")
    ):
        image_path = image_paths[idx]
        robot_pose = image_poses[image_path]

        for detection in bounding_boxes:
            target_class = detection[0]
            bbox = detection[1]  # [x, y, w, h] in px (top-left origin for your detector)

            # Accept only known classes
            if target_class not in TARGET_TYPES or target_class not in TARGET_DIMENSIONS_DICT:
                continue

            # Aspect-ratio sanity filter (±15%)
            if not bbox_ratio_ok(target_class, bbox, TARGET_DIMENSIONS_DICT, tol=0.15):
                continue

            true_height = TARGET_DIMENSIONS_DICT[target_class][2]
            m = ranger.from_bbox_height(camera_matrix, bbox, true_height)
            if m is None:
                continue

            # Collect and fuse per class (across frames)
            meas_by_class.setdefault(target_class, []).append(m)

            # If you want per-image fusion instead, move meas_by_class.clear() here at each image
            ekf_pose_var = 0.0  # TODO: set to trace(P_robot_xy) if EKF pose covariance available
            fused = ranger.fuse(meas_by_class[target_class], ekf_pose_var=ekf_pose_var)
            if fused is None:
                continue

            # Camera → world
            x_cam, y_cam = float(fused['x']), float(fused['y'])
            th = float(robot_pose[2])
            dx = x_cam * np.cos(th) - y_cam * np.sin(th)
            dy = x_cam * np.sin(th) + y_cam * np.cos(th)
            pos = {'x': float(robot_pose[0] + dx), 'y': float(robot_pose[1] + dy)}

            # Arena filter
            if not is_inside_arena(pos['x'], pos['y']):
                continue

            # Record
            occurrence = detected_type_list.count(target_class)
            target_pose_dict[f'{target_class}_{occurrence}'] = pos
            detected_type_list.append(target_class)

    # Merge with DBSCAN → ≤3 clusters per class
    target_est = merge_estimations(target_pose_dict, eps=0.25, min_samples=1, max_clusters_per_class=3)
    print(target_est)

    # Save results
    out_json = f'{script_dir}/lab_output/targets.txt'
    with open(out_json, 'w') as fo:
        json.dump(target_est, fo, indent=4)
    print(f'Estimations saved to {out_json}')

    # Optional: plot with SLAM map
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

    # Visual comparison: pre vs post clustering
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-ARENA_BOUND, ARENA_BOUND)
        ax.set_ylim(-ARENA_BOUND, ARENA_BOUND)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title('Fruit Positions: pre (transparent) vs post (solid)')
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax.grid(which='major', linestyle='-', linewidth=0.8, color='0.7')
        ax.grid(which='minor', linestyle=':', linewidth=0.5, color='0.85')

        def bgr_to_rgb01(bgr):
            return (bgr[2]/255.0, bgr[1]/255.0, bgr[0]/255.0)

        # Pre-cluster points
        pre_by_class: Dict[str, List[Tuple[float,float]]] = {}
        for key, pose in target_pose_dict.items():
            cls = key.split('_')[0]
            pre_by_class.setdefault(cls, []).append((pose['x'], pose['y']))

        for cls, pts in pre_by_class.items():
            color_rgb = bgr_to_rgb01(getattr(yolo, 'class_colour', {}).get(cls, (128, 128, 128)))
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.scatter(xs, ys, c=[color_rgb], alpha=0.35, s=25, edgecolors='none')

        # Post-cluster points
        post_by_class: Dict[str, List[Tuple[float,float]]] = {}
        for key, pose in target_est.items():
            cls = key.split('_')[0]
            post_by_class.setdefault(cls, []).append((pose['x'], pose['y']))

        # Draw cluster outlines (optional hulls)
        eps, min_samples = 0.15, 2
        for cls, pts in pre_by_class.items():
            if len(pts) < 1:
                continue
            pts_np = np.array(pts, dtype=np.float32)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_np).labels_
            color_rgb = bgr_to_rgb01(getattr(yolo, 'class_colour', {}).get(cls, (128, 128, 128)))
            for cid in np.unique(labels):
                if cid == -1: continue
                cluster_pts = pts_np[labels == cid]
                if len(cluster_pts) < 3:
                    ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=80,
                               facecolors='none', edgecolors=color_rgb, linewidths=1.0)
                    continue
                hull = cv2.convexHull(cluster_pts.reshape(-1, 1, 2)).squeeze()
                hx = np.r_[hull[:, 0], hull[0, 0]]
                hy = np.r_[hull[:, 1], hull[0, 1]]
                ax.plot(hx, hy, color=color_rgb, linewidth=1.2)
                ax.fill(hull[:, 0], hull[:, 1], color=color_rgb, alpha=0.10)

        # Overlay icons or dots for final estimates
        pixel_art_dir = os.path.join(script_dir, 'pixel_art')
        pixel_icons = {}
        if os.path.isdir(pixel_art_dir):
            for fname in os.listdir(pixel_art_dir):
                if fname.lower().endswith('.png'):
                    key = os.path.splitext(fname)[0].lower()
                    try:
                        pixel_icons[key] = mpimg.imread(os.path.join(pixel_art_dir, fname))
                    except Exception:
                        pass

        ICON_SIZE_PX = 10
        def add_icon(ax, xy, img, size_px=ICON_SIZE_PX, z=5):
            h, w = img.shape[:2]
            scale = size_px / float(max(h, w))
            oi = OffsetImage(img, zoom=scale)
            ab = AnnotationBbox(oi, xy, frameon=False, pad=0.0, box_alignment=(0.5, 0.5),
                                annotation_clip=True, zorder=z)
            ax.add_artist(ab)

        for cls, pts in post_by_class.items():
            icon = pixel_icons.get(cls.lower())
            color_rgb = bgr_to_rgb01(getattr(yolo, 'class_colour', {}).get(cls, (64, 64, 64)))
            if icon is None:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                ax.scatter(xs, ys, c=[color_rgb], alpha=1.0, s=50, edgecolors='black', linewidths=0.5, zorder=4)
            else:
                for (x, y) in pts:
                    add_icon(ax, (x, y), icon)

        plt.show()
    except Exception as e:
        print(f'Plotting failed: {e}')
