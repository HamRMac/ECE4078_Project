import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from matplotlib.patches import Wedge
from scipy.signal import convolve2d

# --- Load ground truth ---
with open("true_map.txt", "r") as f:
    data = json.load(f)

# --- Obstacles ---
marker_x, marker_y = [], []
fruit_x, fruit_y = [], []
marker_diameter_m = 0.07
fruit_diameter_m = 0.10
dot_spacing = 0.025
dot_diameter_m = 0.005

for name, coords in data.items():
    if "aruco" in name:
        marker_x.append(coords["x"])
        marker_y.append(coords["y"])
    else:
        fruit_x.append(coords["x"])
        fruit_y.append(coords["y"])

padding = 0.1
all_x = marker_x + fruit_x
all_y = marker_y + fruit_y
xmin, xmax = min(all_x)-padding, max(all_x)+padding
ymin, ymax = min(all_y)-padding, max(all_y)+padding

# --- Grid ---
x_grid = np.arange(xmin, xmax, dot_spacing)
y_grid = np.arange(ymin, ymax, dot_spacing)
xx, yy = np.meshgrid(x_grid, y_grid)
xx = xx.flatten()
yy = yy.flatten()

# --- Obstacles list ---
obstacles = [{'type':'square','x':x,'y':y,'half':marker_diameter_m/2} for x,y in zip(marker_x, marker_y)]
obstacles += [{'type':'circle','x':x,'y':y,'radius':fruit_diameter_m/2} for x,y in zip(fruit_x, fruit_y)]

# --- Helper functions ---
def line_intersects_circle(p0, p1, center, radius):
    p0,p1,center = np.array(p0), np.array(p1), np.array(center)
    d = p1-p0
    f = p0-center
    a = np.dot(d,d)
    b = 2*np.dot(f,d)
    c = np.dot(f,f)-radius**2
    disc = b*b - 4*a*c
    if disc < 0: return False
    disc = np.sqrt(disc)
    t1 = (-b-disc)/(2*a)
    t2 = (-b+disc)/(2*a)
    return 0<=t1<=1 or 0<=t2<=1

def ccw(A,B,C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
def segments_intersect(A,B,C,D): return ccw(A,C,D)!=ccw(B,C,D) and ccw(A,B,C)!=ccw(A,B,D)

def line_intersects_square(p0, p1, center, half):
    cx,cy=center
    x_min,x_max = cx-half,cx+half
    y_min,y_max = cy-half,cy+half
    edges = [((x_min,y_min),(x_max,y_min)),
             ((x_max,y_min),(x_max,y_max)),
             ((x_max,y_max),(x_min,y_max)),
             ((x_min,y_max),(x_min,y_min))]
    return any(segments_intersect(p0,p1,e0,e1) for e0,e1 in edges)

def diameter_to_scatter(diameter_m, ax):
    fig = ax.get_figure()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_pts = bbox.width*72
    height_pts = bbox.height*72
    xlim,ylim = ax.get_xlim(),ax.get_ylim()
    scale = (width_pts/(xlim[1]-xlim[0]) + height_pts/(ylim[1]-ylim[0]))/2
    return (diameter_m*scale)**2

# --- Robot / visibility ---
heading_rad = 0
light_pos = (0,0)
fov_deg = 360
fov_half = fov_deg/2
max_distance = 1.2
visibility = np.ones_like(xx)*-1  # -1 = outside, 0 = blocked, 1 = visible

for i,(x,y) in enumerate(zip(xx,yy)):
    dx,dy = x-light_pos[0],y-light_pos[1]
    distance = np.hypot(dx,dy)
    if distance>max_distance: continue
    angle = (np.arctan2(dy,dx)-heading_rad + np.pi)%(2*np.pi)-np.pi
    if abs(np.rad2deg(angle))<=fov_half:
        visibility[i]=1
        for obs in obstacles:
            if obs['type']=='circle' and line_intersects_circle(light_pos,(x,y),(obs['x'],obs['y']),obs['radius']):
                visibility[i]=0
                break
            elif obs['type']=='square' and line_intersects_square(light_pos,(x,y),(obs['x'],obs['y']),obs['half']):
                visibility[i]=0
                break

# --- Visibility mask ---
vis_grid = visibility.reshape(len(y_grid), len(x_grid))
dark_mask = vis_grid==0
outside_mask = vis_grid==-1
black_map = dark_mask | outside_mask
visible_mask = vis_grid==1

# --- Red border ---
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
edges_x = convolve2d(black_map.astype(float), sobel_x, mode='same', boundary='fill', fillvalue=0)
edges_y = convolve2d(black_map.astype(float), sobel_y, mode='same', boundary='fill', fillvalue=0)
edge_map = np.hypot(edges_x, edges_y) > 0
border_mask = binary_dilation(edge_map, iterations=4)
red_border_mask = border_mask & ~black_map

# --- Safety mask: all visible points not part of red border ---

visible_mask = (vis_grid == 1).flatten()
red_border_mask = (red_border_mask).flatten()

safety_mask = visible_mask & (~red_border_mask)

# --- Sector analysis ---
sector_size = 0.8
num_sectors_x = num_sectors_y = 3
sector_x_edges = [xmin+i*sector_size for i in range(num_sectors_x+1)]
sector_y_edges = [ymin+i*sector_size for i in range(num_sectors_y+1)]
sector_dark_fraction = np.zeros((num_sectors_y,num_sectors_x))
for ix in range(num_sectors_x):
    for iy in range(num_sectors_y):
        x0,x1 = sector_x_edges[ix], sector_x_edges[ix+1]
        y0,y1 = sector_y_edges[iy], sector_y_edges[iy+1]
        mask = (xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
        dots = visibility[mask]
        sector_dark_fraction[iy,ix] = np.sum(dots!=1)/len(dots) if len(dots)>0 else 0

# --- Plot ---
plt.figure(figsize=(8,8))
ax = plt.gca()
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
dot_size = diameter_to_scatter(dot_diameter_m, ax)
marker_size = diameter_to_scatter(marker_diameter_m, ax)
fruit_size = diameter_to_scatter(fruit_diameter_m, ax)

# Black/white points based on visibility
ax.scatter(xx[black_map.flatten()], yy[black_map.flatten()], s=dot_size, c='black', marker='o', edgecolors='none')
ax.scatter(xx[visible_mask.flatten()], yy[visible_mask.flatten()], s=dot_size, c='white', marker='o', edgecolors='none')
# Red border
ax.scatter(xx[red_border_mask.flatten()], yy[red_border_mask.flatten()], s=dot_size*1.5, c='red', marker='o', edgecolors='none', alpha=0.8)
# Safety mask (all visible points not in red border)
ax.scatter(xx[safety_mask], yy[safety_mask], s=dot_size, c='lightblue', marker='o', edgecolors='none', alpha=0.6)

# Obstacles
ax.scatter(marker_x, marker_y, s=marker_size, c='blue', marker='s', alpha=0.6, edgecolors='k')
ax.scatter(fruit_x, fruit_y, s=fruit_size, c='red', marker='o', alpha=0.6, edgecolors='k')

# Sector dark fractions
for ix in range(num_sectors_x):
    for iy in range(num_sectors_y):
        cx,cy = (sector_x_edges[ix]+sector_x_edges[ix+1])/2, (sector_y_edges[iy]+sector_y_edges[iy+1])/2
        fraction = sector_dark_fraction[iy,ix]
        offset_x = 0.0  # meters to move right (positive) or left (negative)
        offset_y = 0.15  # meters to move up (positive) or down (negative)
        ax.text(cx + offset_x, cy + offset_y, f"{fraction:.2f}", ha='center', va='center', fontsize=10, fontweight='bold', color='black')



# Robot FOV
fov_patch = Wedge(light_pos, xmax,
                  np.rad2deg(heading_rad-np.deg2rad(fov_half)),
                  np.rad2deg(heading_rad+np.deg2rad(fov_half)),
                  alpha=0.1, color='green')
ax.add_patch(fov_patch)
ax.plot(light_pos[0], light_pos[1], 'ko', markersize=8)

# --- Sector centres & nearest safety points ---
for ix in range(num_sectors_x):
    for iy in range(num_sectors_y):
        cx, cy = (sector_x_edges[ix]+sector_x_edges[ix+1])/2, (sector_y_edges[iy]+sector_y_edges[iy+1])/2
        ax.plot(cx, cy, 'kx', markersize=10, markeredgewidth=2)  # black cross

        # Mask for points in sector that are safe
        sector_mask = (xx >= sector_x_edges[ix]) & (xx < sector_x_edges[ix+1]) & \
                      (yy >= sector_y_edges[iy]) & (yy < sector_y_edges[iy+1])
        mask = sector_mask & safety_mask
        if np.any(mask):
            sector_x_coords = xx[mask]
            sector_y_coords = yy[mask]
            distances = np.hypot(sector_x_coords - cx, sector_y_coords - cy)
            min_idx = np.argmin(distances)
            closest_x = sector_x_coords[min_idx]
            closest_y = sector_y_coords[min_idx]
            ax.scatter(closest_x, closest_y, c='green', s=80, marker='x', linewidths=2)

# Sector boundaries
for x_edge in sector_x_edges: ax.plot([x_edge,x_edge],[ymin,ymax], color='gray', linestyle='--', linewidth=1)
for y_edge in sector_y_edges: ax.plot([xmin,xmax],[y_edge,y_edge], color='gray', linestyle='--', linewidth=1)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.axis('equal')
plt.show()
