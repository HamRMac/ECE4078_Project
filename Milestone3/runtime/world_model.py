import threading
from typing import Any, Dict, List, Optional, Tuple


class WorldModel:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pose: List[float] = [0.0, 0.0, 0.0]
        self._plan: Optional[Dict[str, Any]] = None  # {'waypoints':[(x,y),...], 'active_idx':int}
        self._detections: List[Dict[str, Any]] = []
        self._status: Dict[str, Any] = {"mode": "IDLE", "sm_state": "Init", "action": "", "progress": ""}

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

