import random
import numpy as np

# Assume we have:
# - `bbox` = (x_min, y_min, x_max, y_max)
# - `image` = the cropped region of interest (ROI) corresponding to bbox
# - `is_tomato` and `is_capsicum` are initial predictions from a model

# bbox dimensions and height
width = bbox[2] - bbox[0]
height = bbox[3] - bbox[1]

aspect_ratio = width / height

# --- Function to sample and average color ---
def average_color(image, num_samples=50):
    h, w, _ = image.shape
    samples = [image[random.randint(0, h-1), random.randint(0, w-1)] for _ in range(num_samples)]
    return np.mean(samples, axis=0)  # returns [R, G, B]

avg_color = average_color(image)
r, g, b = avg_color

# --- Classification logic ---
if is_tomato:
    if 0.9 <= aspect_ratio <= 1.1:  # roughly square
        label = "tomato"
    elif aspect_ratio < 0.9:  # tall rectangle
        print("Model error corrected: capsicum" )
        label = "capsicum"

elif is_capsicum:
    if g > r and g > b:  # mostly green
        if aspect_ratio > 1.2:  # wide rectangle
            print("Model error corrected: capsicum" )
            label = "pumpkin"
        else:  # tall
            label = "capsicum"


print("Detected:", label)
