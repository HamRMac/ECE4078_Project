# TEST_OPTIMISER_COMPLEX.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from slam.joint_optimiser import JointOptimiser2D
from os import makedirs

# ---------- utils ----------
def R(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def save_csv(path, arr, cols):
    df = pd.DataFrame(arr, columns=cols)
    df.to_csv(path, index=False)
    print(f"Saved {path}")

# ---------- world generation ----------
def generate_loop_trajectory(n=60, radius=1.2, center=(1.0, 1.0)):
    """
    Smooth closed loop with heading tangent to the path.
    Param t in [0, 2pi).
    """
    cx, cy = center
    ts = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = cx + radius*np.cos(ts) + 0.25*np.cos(2*ts)           # small deformation so it is not a circle
    y = cy + radius*np.sin(ts) - 0.15*np.sin(3*ts)
    # heading is the path tangent
    dx = -radius*np.sin(ts) - 0.5*np.sin(2*ts)
    dy =  radius*np.cos(ts) - 0.45*np.cos(3*ts)
    th = np.arctan2(dy, dx)
    poses = np.stack([x, y, th], axis=1)
    return poses

def generate_markers_grid(xmin=-0.5, xmax=2.5, ymin=0.0, ymax=2.2, nx=7, ny=5, jitter=0.08, seed=0):
    """
    Grid of markers with small jitter, return dict id -> [mx, my]
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    markers = {}
    mid = 0
    for yy in ys:
        for xx in xs:
            p = np.array([xx, yy]) + rng.normal(0, jitter, size=2)
            markers[mid] = p
            mid += 1
    return markers

# ---------- simulation ----------
def simulate_frames(poses_gt, markers_gt, rng,
                    max_range=1.6, meas_sigma=0.01,
                    ekf_pos_sigma=0.06, ekf_yaw_sigma_deg=4.0, ekf_yaw_bias_deg=6.0):
    """
    Build EKF-like noisy poses (initial guesses) and ARUCO body-frame observations
    computed from the TRUE trajectory. Keep frames with >= 2 true-visible markers.
    Returns:
      poses_ekf (N x 3), frames=[(ekf_pose_j, [(id, b_true_ij), ...])], kept_idx
    """
    N = len(poses_gt)

    # 1) Noisy EKF trajectory (initial guess)
    poses_ekf = poses_gt.copy()
    poses_ekf[:, 0:2] += rng.normal(0, ekf_pos_sigma, size=(N, 2))
    poses_ekf[:, 2]   += np.deg2rad(ekf_yaw_bias_deg) + rng.normal(0, np.deg2rad(ekf_yaw_sigma_deg), size=N)

    # 2) Build observations USING TRUE POSES
    frames = []
    kept_idx = []
    for j in range(N):
        x_true, y_true, th_true = poses_gt[j]
        P_true = np.array([x_true, y_true])

        obs = []
        for mid, Mi in markers_gt.items():
            v = Mi - P_true
            if np.linalg.norm(v) <= max_range:     # visibility from TRUE pose
                b_true = R(th_true).T @ v + meas_sigma * rng.standard_normal(2)
                obs.append((mid, b_true))

        if len(obs) >= 2:
            frames.append((poses_ekf[j].copy(), obs))   # store EKF pose with true-based obs
            kept_idx.append(j)

    return poses_ekf, frames, np.array(kept_idx)

def ekf_only_map(frames):
    """
    Simple baseline: average world predictions using EKF pose per frame.
    M_i ≈ avg_j (P_j + R(th_j) b_ij)
    """
    sums, counts = {}, {}
    for pose, obs in frames:
        x,y,th = pose
        for mid, b in obs:
            est = np.array([x,y]) + R(th) @ b
            sums[mid]   = sums.get(mid, np.zeros(2)) + est
            counts[mid] = counts.get(mid, 0) + 1
    out = {mid: sums[mid]/counts[mid] for mid in sums.keys()}
    return out

# ---------- main ----------
def main():
    rng = np.random.default_rng(7)

    # True world
    poses_gt = generate_loop_trajectory(n=64, radius=1.2, center=(1.0, 1.1))
    markers_gt = generate_markers_grid(nx=8, ny=6, xmin=-0.3, xmax=2.6, ymin=0.1, ymax=2.2, jitter=0.06, seed=2)

    # Build observations from EKF-like poses
    poses_ekf, frames, kept = simulate_frames(poses_gt, markers_gt, rng)

    # Run optimiser
    opt = JointOptimiser2D()
    opt.OPT_PRIOR_ENABLED = True
    opt._OPT_PRIOR_SIGMAS  = (0.7, 0.7, 40.0)  # loose prior to kill gauge without bias
    for pose, obs in frames:
        opt.add_frame(pose, obs)
    cam_opt, map_opt = opt.optimise()

    # EKF-only baseline map
    map_ekf = ekf_only_map(frames)

    # ---------- Collect arrays and save ----------
    # True poses for the kept frames (so shapes match)
    poses_true_kept = poses_gt[kept]

    # Build marker arrays on common id set
    ids_sorted = sorted(map_ekf.keys() & map_opt.keys() & set(markers_gt.keys()))
    M_true = np.array([markers_gt[i] for i in ids_sorted])
    M_ekf  = np.array([map_ekf[i]     for i in ids_sorted])
    M_opt  = np.array([map_opt[i]     for i in ids_sorted])

    poses_opt = np.array(cam_opt)             # (n_kept, 3)
    poses_ekf_kept = poses_ekf[kept]          # (n_kept, 3)

    # Save CSVs
    optimiser_tests_location = "optimiser_test_results"
    # make the test location dir if it doesn't exist
    makedirs(optimiser_tests_location, exist_ok=True)
    save_csv(optimiser_tests_location + "/true_poses.csv", poses_true_kept, ["x", "y", "theta"])
    save_csv(optimiser_tests_location + "/ekf_poses.csv", poses_ekf_kept, ["x", "y", "theta"])
    save_csv(optimiser_tests_location + "/opt_poses.csv", poses_opt, ["x", "y", "theta"])
    save_csv(optimiser_tests_location + "/markers_true.csv", np.c_[ids_sorted, M_true], ["id", "mx", "my"])
    save_csv(optimiser_tests_location + "/markers_ekf.csv",  np.c_[ids_sorted, M_ekf],  ["id", "mx", "my"])
    save_csv(optimiser_tests_location + "/markers_opt.csv",  np.c_[ids_sorted, M_opt],  ["id", "mx", "my"])

    # ---------- Plots ----------
    # 1) Map view
    plt.figure()
    # Markers
    plt.scatter(M_true[:,0], M_true[:,1], marker='o', label='Markers GT')
    plt.scatter(M_ekf[:,0],  M_ekf[:,1],  marker='x', label='Markers EKF-only')
    plt.scatter(M_opt[:,0],  M_opt[:,1],  marker='^', label='Markers EKF+Optim')
    for mid, p in zip(ids_sorted, M_true):
        plt.text(p[0], p[1], str(mid))

    # Poses
    plt.plot(poses_true_kept[:,0], poses_true_kept[:,1], linestyle='--', label='True poses (kept)')
    plt.plot(poses_ekf_kept[:,0],  poses_ekf_kept[:,1],  label='EKF poses')
    plt.plot(poses_opt[:,0],       poses_opt[:,1],       label='Optim poses')

    # Arrows for first and last
    for (x,y,th) in [poses_true_kept[0], poses_true_kept[-1]]:
        d = 0.25*np.array([np.cos(th), np.sin(th)])
        plt.arrow(x, y, d[0], d[1], head_width=0.05, length_includes_head=True)
    for (x,y,th) in [poses_opt[0], poses_opt[-1]]:
        d = 0.25*np.array([np.cos(th), np.sin(th)])
        plt.arrow(x, y, d[0], d[1], head_width=0.05, length_includes_head=True)

    plt.axis('equal')
    plt.title("Complex loop: GT vs EKF vs EKF+Optim — markers and poses")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Headings over index
    plt.figure()
    plt.plot(np.degrees(poses_true_kept[:,2]), label="theta GT")
    plt.plot(np.degrees(poses_ekf_kept[:,2]),  label="theta EKF")
    plt.plot(np.degrees(poses_opt[:,2]),       label="theta Optim")
    plt.xlabel("frame index (kept)")
    plt.ylabel("heading (deg)")
    plt.title("Headings along the kept frames")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Optional: marker errors summary
    err_ekf = np.linalg.norm(M_ekf - M_true, axis=1)
    err_opt = np.linalg.norm(M_opt - M_true, axis=1)
    print(f"Marker RMSE EKF-only:   {np.sqrt(np.mean(err_ekf**2)):.3f} m")
    print(f"Marker RMSE EKF+Optim:  {np.sqrt(np.mean(err_opt**2)):.3f} m")

if __name__ == "__main__":
    main()
