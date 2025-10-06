from typing import List, Dict, Tuple, Optional

# nominal true sizes (width, depth, height) in metres
TARGET_HEIGHTS_DICT: Dict[str, float] = {
    'orange':  0.073,
    'lemon':   0.05,
    'pear':    0.11,
    'tomato':  0.06,
    'capsicum':0.09,
    'potato':  0.07,
    'pumpkin': 0.08,
    'garlic':  0.07,
    'lime':    0.05,
}

TARGET_TYPES = [f for f in TARGET_HEIGHTS_DICT.keys()]