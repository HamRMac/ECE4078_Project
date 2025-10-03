from dataclasses import dataclass


@dataclass
class SetGoal:
    x: float
    y: float


@dataclass
class CancelGoal:
    reason: str = "user"


@dataclass
class SwitchMode:
    mode: str  # 'AUTO' | 'MANUAL_WAYPOINTS' | 'IDLE'

