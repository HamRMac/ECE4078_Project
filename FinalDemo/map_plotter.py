import json
import matplotlib.pyplot as plt

# Load map from file
with open("true_map_backup.txt", "r") as f:
    map_data = json.load(f)

# Separate markers and fruits
markers = {k:v for k,v in map_data.items() if k.startswith("aruco")}
fruits = {k:v for k,v in map_data.items() if not k.startswith("aruco")}

plt.figure(figsize=(10,10))

# Plot markers
for name, pos in markers.items():
    plt.scatter(pos["x"], pos["y"], c='blue', marker='s', s=100)
    plt.text(pos["x"]+0.02, pos["y"]+0.02, name, color='blue')

# Plot fruits
for name, pos in fruits.items():
    plt.scatter(pos["x"], pos["y"], c='green', marker='o', s=100)
    plt.text(pos["x"]+0.02, pos["y"]+0.02, name, color='green')

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("ArUco Markers and Fruits Map")
plt.grid(True)
plt.axis('equal')
plt.show()
