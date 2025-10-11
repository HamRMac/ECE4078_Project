import json

# Load slam.txt
with open("lab_output/slam.txt", "r") as f:
    slam_data = json.load(f)

tags = slam_data["taglist"]
map_data = slam_data["map"]
x_coords = map_data[0]
y_coords = map_data[1]

# Build the true_map
true_map = {}
for i, tag in enumerate(tags):
    key = f"aruco{tag}_0"
    true_map[key] = {"x": x_coords[i], "y": y_coords[i]}

# Save to JSON
with open("slam_markers.txt", "w") as f:
    json.dump(true_map, f, indent=2)

print("Converted slam.txt to slam_markers.txt in true map format!")
