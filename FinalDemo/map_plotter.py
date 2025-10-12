import json
import matplotlib.pyplot as plt

# --- Load the data ---
with open("slam_markers.txt", "r") as f:
    markers = json.load(f)

with open("lab_output/targets.txt", "r") as f:
    targets = json.load(f)

# --- Extract coordinates ---
marker_x = [v["x"] for v in markers.values()]
marker_y = [v["y"] for v in markers.values()]

target_x = [v["x"] for v in targets.values()]
target_y = [v["y"] for v in targets.values()]

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 8))

# Plot ArUco markers
ax.scatter(marker_x, marker_y, color="blue", label="Markers", s=100)
for name, v in markers.items():
    ax.text(v["x"], v["y"], name, color="blue", fontsize=9, ha='right', va='bottom')

# Plot targets
ax.scatter(target_x, target_y, color="green", label="Targets", s=100, marker='s')
for name, v in targets.items():
    ax.text(v["x"], v["y"], name, color="green", fontsize=9, ha='left', va='bottom')

# Set axes labels
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Markers and Targets Map")

# Set gradations every 0.3 m
ax.xaxis.set_major_locator(plt.MultipleLocator(0.3))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.3))

# Grid
ax.grid(True, which='major', linestyle='--', alpha=0.7)

ax.legend()
ax.set_aspect('equal')  # Keep aspect ratio square

plt.show()
