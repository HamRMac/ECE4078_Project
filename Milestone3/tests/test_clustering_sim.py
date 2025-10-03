import os
import sys
import random
import math
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Ensure local package imports resolve when running as a script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from perception.clustering import cluster_detections_dbscan


# Colour map per class label for plotting
CLASS_COLOUR = {
    'orange': (1.0, 0.65, 0.0),
    'lemon': (1.0, 1.0, 0.0),
    'lime': (0.0, 1.0, 0.0),
    'tomato': (1.0, 0.0, 0.0),
    'capsicum': (1.0, 0.0, 1.0),
    'potato': (0.7, 0.7, 0.0),
    'pumpkin': (1.0, 0.55, 0.0),
    'garlic': (0.7, 0.0, 0.7),
}


def make_detections(true_objs: List[Dict[str, Any]],
                    scan_pos: Tuple[float, float],
                    n_samples_range=(1, 3),
                    base_sigma=0.01,
                    k_sigma=0.02,
                    seed=None) -> List[Dict[str, Any]]:
    """Simulate detections from a given scan position.

    - Noise sigma = base_sigma + k_sigma * distance_to_scan (small dependence).
    - Returns flat list of detection dicts with 'label','class_id','world','count'.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    sx, sy = map(float, scan_pos)
    dets: List[Dict[str, Any]] = []
    for obj in true_objs:
        label = obj['label']
        cid = obj['class_id']
        tx, ty = obj['position']
        dist = math.hypot(tx - sx, ty - sy)
        sigma = base_sigma + k_sigma * dist
        # Sample 1..3 observations with Gaussian noise
        n = random.randint(n_samples_range[0], n_samples_range[1])
        for _ in range(n):
            nx = np.random.normal(tx, sigma)
            ny = np.random.normal(ty, sigma)
            dets.append({
                'label': label,
                'class_id': cid,
                'world': {'x': float(nx), 'y': float(ny)},
                'count': 1
            })
    return dets


def plot_stage(true_objs, samples, clusters, prev_clusters=None, title="stage", out_dir="tests_output", scan_points=None):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)

    # Plot scan points as black X, with older scans lower opacity
    if scan_points:
        n = len(scan_points)
        for i, (sx, sy) in enumerate(scan_points):
            # progressive opacity: oldest ~0.2 .. newest 1.0
            alpha = 0.2 + 0.8 * ((i + 1) / n)
            ax.plot(sx, sy, marker='x', color='black', markersize=10, mew=2, alpha=alpha)

    # Plot true positions as triangles
    for obj in true_objs:
        label = obj['label']
        tx, ty = obj['position']
        c = CLASS_COLOUR.get(label, (0.2, 0.2, 0.2))
        ax.plot(tx, ty, marker='^', color=c, markersize=8, label=f"true:{label}")

    # Plot previous cluster centroids as plus
    if prev_clusters:
        for cl in prev_clusters:
            label = cl['class']
            x, y = cl['position']
            c = CLASS_COLOUR.get(label, (0.2, 0.2, 0.2))
            ax.plot(x, y, marker='+', color=c, markersize=10, mew=2)

    # Plot samples as faint dots
    for det in samples:
        label = det['label']
        w = det.get('world', None)
        if not w:
            continue
        x, y = w['x'], w['y']
        c = CLASS_COLOUR.get(label, (0.4, 0.4, 0.4))
        ax.plot(x, y, marker='o', color=c, alpha=0.3, markersize=4)

    # Plot cluster centroids as solid X
    for cl in clusters:
        label = cl['class']
        x, y = cl['position']
        c = CLASS_COLOUR.get(label, (0.2, 0.2, 0.2))
        ax.plot(x, y, marker='x', color=c, markersize=10, mew=2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title.replace(' ', '_')}.png"), dpi=150)
    plt.close()


def main():
    # True object set
    true_objs = [
        {'label': 'apple', 'class_id': 1, 'position': (1.15, 0.95)},
        {'label': 'apple', 'class_id': 1, 'position': (0.9, 1.11)},
        {'label': 'tomato', 'class_id': 2, 'position': (0.20, -0.50)},
        {'label': 'orange', 'class_id': 3, 'position': (-0.70, 0.80)},
    ]

    # Scan locations (robot poses not required here; only distance affects noise)
    scan1 = (0.0, 0.0)
    scan2 = (0.8, 0.6)

    # Stage 1: sample around scan1 and cluster
    dets1 = make_detections(true_objs, scan_pos=scan1, seed=42)
    clusters1 = cluster_detections_dbscan(dets1, eps_m=0.15, min_samples=1, arena_bound=1.35)
    plot_stage(true_objs, dets1, clusters1, prev_clusters=None, title="stage 1", out_dir="tests_output", scan_points=[scan1])

    # Stage 2: new samples around scan2, merge with previous clusters (as weighted prior)
    dets2 = make_detections(true_objs, scan_pos=scan2, seed=43)
    # Convert previous clusters to detections with weight=count
    prior_as_dets = []
    for cl in clusters1:
        prior_as_dets.append({
            'label': cl['class'],
            'class_id': cl['class_id'],
            'world': {'x': cl['position'][0], 'y': cl['position'][1]},
            'count': cl['count']
        })
    dets_combined = dets2 + prior_as_dets
    clusters2 = cluster_detections_dbscan(dets_combined, eps_m=0.15, min_samples=1, arena_bound=1.35)
    plot_stage(true_objs, dets2, clusters2, prev_clusters=clusters1, title="stage 2", out_dir="tests_output", scan_points=[scan1, scan2])

    # Print results as JSON
    print(json.dumps({
        'stage1': clusters1,
        'stage2': clusters2
    }, indent=2))


if __name__ == "__main__":
    main()
