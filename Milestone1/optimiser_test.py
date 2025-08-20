# optimiser_test.py
import numpy as np
import matplotlib.pyplot as plt
from slam.joint_optimiser import JointOptimiser2D

def R(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def build_world():
    # Ground truth saved frames (the *first saved frame* is index 0, not necessarily origin)
    poses_gt = np.array([
        [0.30, 0.00, np.deg2rad(  0)],
        [0.80, 0.25, np.deg2rad( 10)],
        [1.30, 0.45, np.deg2rad( 18)],
        [1.75, 0.60, np.deg2rad( 22)],
    ])
    # Ground truth markers (IDs arbitrary but fixed)
    markers_gt = {
        1: np.array([ 0.10,  1.00]),
        2: np.array([ 0.90,  1.10]),
        3: np.array([ 1.80,  0.70]),
        4: np.array([-0.30,  0.85]),
    }
    return poses_gt, markers_gt

def simulate_frames(poses_gt, markers_gt, rng, max_range=1.7, meas_sigma=0.02):
    """Return EKF-like noisy poses and per-frame ARUCO body-frame observations b_ij."""
    # EKF pose guess with drift/bias
    yaw_bias = np.deg2rad(5.0)
    poses_ekf = poses_gt.copy()
    poses_ekf[:, 0:2] += rng.normal(0, 0.07, size=(len(poses_gt), 2))  # 7 cm
    poses_ekf[:, 2]   += yaw_bias + rng.normal(0, np.deg2rad(5.0), size=len(poses_gt))

    frames = []
    for (xj, yj, thj) in poses_ekf:
        Pj = np.array([xj, yj])
        obs = []
        for mid, Mi in markers_gt.items():
            v = Mi - Pj
            if np.linalg.norm(v) <= max_range:
                b = R(thj).T @ v + meas_sigma * rng.standard_normal(2)  # body vector with noise
                # In your pipeline b ≈ [Tz, -Tx]; here b is already 2D body vector
                obs.append((mid, b))
        if len(obs) >= 2:
            frames.append((np.array([xj, yj, thj]), obs))
    return poses_ekf, frames

def ekf_only_map(frames):
    """Simple averaging baseline: M_i ≈ avg_j (P_j + R(th_j) b_ij)."""
    sums, counts = {}, {}
    for pose, obs in frames:
        x, y, th = pose
        for mid, b in obs:
            est = np.array([x, y]) + R(th) @ b
            sums[mid]   = sums.get(mid, np.zeros(2)) + est
            counts[mid] = counts.get(mid, 0) + 1
    out = {mid: sums[mid] / counts[mid] for mid in sums.keys()}
    return out

def main():
    rng = np.random.default_rng(0)
    poses_gt, markers_gt = build_world()
    poses_ekf, frames = simulate_frames(poses_gt, markers_gt, rng)

    # Run optimiser
    opt = JointOptimiser2D()
    opt.OPT_PRIOR_ENABLED = True
    opt._OPT_PRIOR_SIGMAS  = (0.5, 0.5, 30.0)  # loose prior to kill gauge
    for pose, obs in frames:
        opt.add_frame(pose, obs)
    cam_opt, map_opt = opt.optimise()

    # EKF-only baseline
    map_ekf = ekf_only_map(frames)

    # Gather arrays for plotting
    mk_ids_sorted = sorted(markers_gt.keys())
    M_gt  = np.array([markers_gt[i] for i in mk_ids_sorted])
    M_ekf = np.array([map_ekf[i]     for i in mk_ids_sorted])
    M_opt = np.array([map_opt[i]     for i in mk_ids_sorted])

    poses_opt = np.array(cam_opt)

    # --- Plot markers and poses ---
    plt.figure()
    # markers
    plt.scatter(M_gt[:,0],  M_gt[:,1],  marker='o', label='Markers GT')
    plt.scatter(M_ekf[:,0], M_ekf[:,1], marker='x', label='Markers EKF-only')
    plt.scatter(M_opt[:,0], M_opt[:,1], marker='^', label='Markers EKF+Optim')
    for mid, p in zip(mk_ids_sorted, M_gt):
        plt.text(p[0], p[1], str(mid))

    # poses
    plt.plot(poses_ekf[:,0], poses_ekf[:,1], label='EKF poses')
    plt.scatter(poses_ekf[:,0], poses_ekf[:,1])
    plt.plot(poses_opt[:,0],  poses_opt[:,1],  label='Optim poses')
    plt.scatter(poses_opt[:,0],  poses_opt[:,1])

    # draw heading arrows for first and last EKF/OPT poses
    for (x,y,th) in [poses_ekf[0], poses_ekf[-1]]:
        d = 0.25 * np.array([np.cos(th), np.sin(th)])
        plt.arrow(x, y, d[0], d[1], head_width=0.05, length_includes_head=True)
    for (x,y,th) in [poses_opt[0], poses_opt[-1]]:
        d = 0.25 * np.array([np.cos(th), np.sin(th)])
        plt.arrow(x, y, d[0], d[1], head_width=0.05, length_includes_head=True)

    plt.axis('equal')
    plt.title("True vs EKF vs EKF+Optimised — markers and poses")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
