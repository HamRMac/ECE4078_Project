import threading
from typing import List


class RobotCommander:
    """Serialize motor commands; thin wrapper over PenguinPi.set_velocity."""

    def __init__(self, ppi) -> None:
        self._ppi = ppi
        self._lock = threading.Lock()

    def set_velocity(self, command: List[int], tick: int = 50, turning_tick: int = 20, time: float = 0.0):
        with self._lock:
            return self._ppi.set_velocity(command, tick=tick, turning_tick=turning_tick, time=time)

    def stop(self):
        try:
            with self._lock:
                self._ppi.set_velocity([0, 0])
        except Exception:
            pass

