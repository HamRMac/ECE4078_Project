from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.cluster import DBSCAN

from .fruit_ranger import is_inside_arena


def cluster_detections_dbscan(detections: List[Dict[str, Any]],
                              eps_m: float = 0.15,
                              min_samples: int = 1,
                              arena_bound: Optional[float] = None) -> List[Dict[str, Any]]:
    """Cluster world positions (per-class) with DBSCAN and return merged results.

    Parameters
    - detections: list of detection dicts; each should contain at least:
        {'label': <str|int>, 'class_id': <int|str>, 'world': {'x': float, 'y': float}}
        Detections without 'world' are ignored for clustering.
    - eps_m: maximum distance (m) to join points in the same cluster (default 0.15m)
    - min_samples: DBSCAN min_samples (default 1)
    - arena_bound: if provided, discard clusters outside valid workspace (|x|,|y| <= bound)

    Returns
    - A JSON-serialisable list of dicts:
        {'class': <str>, 'class_id': <int|str>, 'position': [x,y], 'count': <int>}
    """
    if not detections:
        return []

    # Group positions by class (use both class_id and label for robustness)
    # Store tuples (x, y, weight) per class group
    groups: Dict[Tuple[Any, str], List[Tuple[float, float, float]]] = {}
    for det in detections:
        try:
            label = det.get('label', '')
            cid = det.get('class_id', -1)
            w = det.get('world', None)
            if w is None:
                continue
            x = float(w['x']); y = float(w['y'])
            wt = float(det.get('count', 1.0))  # optional weight; default 1
            if not np.isfinite(wt) or wt <= 0:
                wt = 1.0
        except Exception:
            continue
        key = (cid, str(label))
        groups.setdefault(key, []).append((x, y, wt))

    merged: List[Dict[str, Any]] = []
    for (cid, label), pts in groups.items():
        if not pts:
            continue
        # Split into coordinates and weights
        arr = np.asarray(pts, dtype=float)
        X = arr[:, :2]
        W = arr[:, 2]
        try:
            db = DBSCAN(eps=eps_m, min_samples=min_samples).fit(X, sample_weight=W)
            lbls = db.labels_
        except Exception:
            lbls = -np.ones((len(X),), dtype=int)

        unique_lbls = set(lbls.tolist())
        if -1 in unique_lbls:
            unique_lbls.remove(-1)
        for k in (unique_lbls if unique_lbls else []):
            idx = np.where(lbls == k)[0]
            if idx.size == 0:
                continue
            cluster_pts = X[idx]
            cluster_wts = W[idx]
            # Weighted centroid
            wsum = float(cluster_wts.sum()) if cluster_wts.size > 0 else 0.0
            if wsum <= 0:
                cx = float(cluster_pts[:, 0].mean())
                cy = float(cluster_pts[:, 1].mean())
                count_val = int(cluster_pts.shape[0])
            else:
                cx = float((cluster_pts[:, 0] * cluster_wts).sum() / wsum)
                cy = float((cluster_pts[:, 1] * cluster_wts).sum() / wsum)
                count_val = int(round(wsum))
            if arena_bound is not None and not is_inside_arena(cx, cy, bound=arena_bound):
                continue
            merged.append({
                'class': str(label),
                'class_id': cid,
                'position': [round(cx, 3), round(cy, 3)],
                'count': count_val
            })

        # If all noise, keep singletons
        if not unique_lbls and len(X) > 0:
            for i in range(len(X)):
                cx, cy = float(X[i, 0]), float(X[i, 1])
                wt = float(W[i]) if i < len(W) and np.isfinite(W[i]) else 1.0
                if arena_bound is not None and not is_inside_arena(cx, cy, bound=arena_bound):
                    continue
                merged.append({
                    'class': str(label),
                    'class_id': cid,
                    'position': [round(cx, 3), round(cy, 3)],
                    'count': int(round(wt))
                })

    return merged
