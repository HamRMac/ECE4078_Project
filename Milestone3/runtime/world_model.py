import threading
from typing import Any, Dict, List, Optional, Tuple


class WorldModel:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pose: List[float] = [0.0, 0.0, 0.0]
        self._plan: Optional[Dict[str, Any]] = None  # {'waypoints':[(x,y),...], 'active_idx':int}
        self._detections: List[Dict[str, Any]] = []
        self._status: Dict[str, Any] = {"mode": "IDLE", "sm_state": "Init", "action": "", "progress": ""}
        # Sector overlay info for GUI
        self._sectors: Dict[str, Any] = {
            "rows": 3,
            "cols": 3,
            "searched": [],            # list of (ix, iy)
            "next_idx": None,          # (ix, iy) or None
            "next_point": None,        # (x, y) or None
        }
        # Targets info for GUI (shopping list status)
        self._targets_info: Dict[str, Any] = {
            "order": [],               # shopping list order
            "remaining": {},           # fruit -> (x,y)
            "collected": [],           # list of fruit names
            "seen_not_collected": [],  # list of fruit names
            "unseen": [],              # list of fruit names
            "active": None,            # currently active target name
            "positions": {},           # all known target positions: fruit -> (x,y)
        }

    # Pose
    def set_pose(self, pose_xyz: List[float]) -> None:
        with self._lock:
            self._pose = [float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2])]

    def get_pose(self) -> List[float]:
        with self._lock:
            return list(self._pose)

    # Plan
    def set_plan(self, waypoints: List[Tuple[float, float]], active_idx: int = 0) -> None:
        with self._lock:
            self._plan = {"waypoints": [(float(x), float(y)) for (x, y) in waypoints], "active_idx": int(active_idx)}

    def clear_plan(self) -> None:
        with self._lock:
            self._plan = None

    def get_plan(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return None if self._plan is None else {"waypoints": list(self._plan["waypoints"]), "active_idx": int(self._plan["active_idx"])}

    # Detections
    def set_detections(self, dets: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._detections = list(dets)

    def get_detections(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._detections)

    # Status
    def set_status(self, **kwargs) -> None:
        with self._lock:
            self._status.update(kwargs)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    # Sectors overlay
    def set_sectors(self, rows: int, cols: int,
                    searched: List[Tuple[int, int]],
                    next_idx: Optional[Tuple[int, int]],
                    next_point: Optional[Tuple[float, float]]) -> None:
        with self._lock:
            self._sectors = {
                "rows": int(rows),
                "cols": int(cols),
                "searched": [ (int(ix), int(iy)) for (ix, iy) in (searched or []) ],
                "next_idx": None if next_idx is None else (int(next_idx[0]), int(next_idx[1])),
                "next_point": None if next_point is None else (float(next_point[0]), float(next_point[1])),
            }

    def get_sectors(self) -> Dict[str, Any]:
        with self._lock:
            # Return a shallow copy
            s = dict(self._sectors)
            s["searched"] = list(self._sectors.get("searched", []))
            return s

    # Targets info
    def set_targets_info(self, order: List[str], remaining: Dict[str, Tuple[float, float]],
                         collected: List[str], seen_not_collected: List[str], unseen: List[str],
                         active: Optional[str] = None,
                         positions: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        with self._lock:
            self._targets_info = {
                "order": list(order or []),
                "remaining": {str(k): (float(v[0]), float(v[1])) for k, v in (remaining or {}).items()},
                "collected": list(collected or []),
                "seen_not_collected": list(seen_not_collected or []),
                "unseen": list(unseen or []),
                "active": None if active is None else str(active),
                "positions": {str(k): (float(v[0]), float(v[1])) for k, v in (positions or {}).items()},
            }

    def get_targets_info(self) -> Dict[str, Any]:
        with self._lock:
            info = dict(self._targets_info)
            info["remaining"] = dict(self._targets_info.get("remaining", {}))
            return info
