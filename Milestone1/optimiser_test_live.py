# TEST_OPTIMISER_LIVE.py
# Live-style demo: as new frames arrive (>=2 markers each), run the optimiser
# for a few iterations and plot progress. Produces an animation and a metrics CSV.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from slam.joint_optimiser import JointOptimiser2D
from os import makedirs
from tqdm import tqdm

method = "dense"

# ---------- small helpers ----------
def R(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s,  c]])

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rmse(A, B):
    if len(A) == 0:
        return np.nan
    A = np.asarray(A); B = np.asarray(B)
    return float(np.sqrt(np.mean((A - B) ** 2)))

# ---------- world generation ----------
def generate_loop_trajectory(n=100, radius=1.2, center=(1.0, 1.1)):
    cx, cy = center
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = cx + radius*np.cos(t) + 0.25*np.cos(2*t)
    y = cy + radius*np.sin(t) - 0.15*np.sin(3*t)
    dx = -radius*np.sin(t) - 0.5*np.sin(2*t)
    dy =  radius*np.cos(t) - 0.45*np.cos(3*t)
    th = np.arctan2(dy, dx)
    return np.stack([x, y, th], axis=1)

def generate_markers_grid(xmin=-0.3, xmax=2.6, ymin=0.1, ymax=2.3, nx=8, ny=6, jitter=0.06, seed=2):
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

# ---------- data generator (true-based obs, EKF as guesses) ----------
def simulate_inputs(poses_gt, markers_gt, rng,
                    max_range=1.7, meas_sigma=0.015,
                    ekf_pos_sigma=0.07, ekf_yaw_sigma_deg=4.0, ekf_yaw_bias_deg=6.0):
    N = len(poses_gt)

    # EKF guesses
    poses_ekf = poses_gt.copy()
    poses_ekf[:, 0:2] += rng.normal(0, ekf_pos_sigma, size=(N, 2))
    poses_ekf[:, 2]   += np.deg2rad(ekf_yaw_bias_deg) + rng.normal(0, np.deg2rad(ekf_yaw_sigma_deg), size=N)

    # For each step, build observations using TRUE pose
    frames_stream = []  # each entry: (ekf_pose_j, [(id, b_true_ij), ...])
    for j in range(N):
        xT, yT, thT = poses_gt[j]
        P_true = np.array([xT, yT])
        obs = []
        for mid, Mi in markers_gt.items():
            v = Mi - P_true
            if np.linalg.norm(v) <= max_range:
                b = R(thT).T @ v + meas_sigma * rng.standard_normal(2)
                obs.append((mid, b))
        frames_stream.append((poses_ekf[j].copy(), obs))
    return poses_ekf, frames_stream

# ---------- live run ----------
def run_live_demo():
    rng = np.random.default_rng(7)
    poses_gt = generate_loop_trajectory(n=120, radius=1.25, center=(1.05, 1.1))
    markers_gt = generate_markers_grid()

    poses_ekf, frames_stream = simulate_inputs(poses_gt, markers_gt, rng)

    opt = JointOptimiser2D()
    opt.GAUGE_MODE = "fix_pose0"
    print(f"Running with gauge mode set to {opt.GAUGE_MODE}")
    #opt.OPT_PRIOR_SIGMAS  = (0.7, 0.7, 30.0)  # loose prior just to fix gauge

    # Snapshots for animation and metrics
    snaps = []   # each: dict with keys: step, poses_true, poses_ekf, poses_opt, M_true, M_ekf, M_opt
    metrics = [] # rows for CSV

    # Maintain growing list of frames with >=2 markers
    kept_frames = []

    def ekf_only_map(frames):
        sums, counts = {}, {}
        for pose, obs in frames:
            x, y, th = pose
            for mid, b in obs:
                est = np.array([x, y]) + R(th) @ b
                sums[mid]   = sums.get(mid, np.zeros(2)) + est
                counts[mid] = counts.get(mid, 0) + 1
        out = {mid: sums[mid]/counts[mid] for mid in sums.keys()}
        return out

    for j, (pose_guess, obs) in enumerate(tqdm(frames_stream, desc="Frames")):
        # Save a frame only if it sees >= 2 markers
        if len(obs) >= 2:
            kept_frames.append((pose_guess, obs))
            opt.add_frame(pose_guess, obs)

            # Online optimise a little each time (few iterations)
            cam_opt, map_opt = opt.optimise(max_iters=8, lambda_init=1e-2, tol=1e-6, method="schur")

            # EKF-only cumulative map using the same kept_frames
            map_ekf = ekf_only_map(kept_frames)

            # Build arrays on the common id set
            common_ids = sorted(set(map_ekf.keys()) & set(map_opt.keys()) & set(markers_gt.keys()))
            M_true = np.array([markers_gt[i] for i in common_ids])
            M_ekf  = np.array([map_ekf[i]     for i in common_ids])
            M_opt  = np.array([map_opt[i]     for i in common_ids])

            # Poses up to current step, only those we actually kept
            kept_indices = [k for k in range(j+1) if len(frames_stream[k][1]) >= 2]
            poses_true_kept = poses_gt[kept_indices]
            poses_ekf_kept  = poses_ekf[kept_indices]
            poses_opt_arr   = np.array(cam_opt)

            # Metrics
            e_ekf = rmse(M_ekf, M_true)
            e_opt = rmse(M_opt, M_true)
            metrics.append([len(kept_frames), e_ekf, e_opt])

            snaps.append(dict(
                step=len(kept_frames),
                poses_true=poses_true_kept,
                poses_ekf=poses_ekf_kept,
                poses_opt=poses_opt_arr,
                M_true=M_true, M_ekf=M_ekf, M_opt=M_opt,
                ids=common_ids
            ))

    # Save metrics
    optimiser_tests_location = "optimiser_test_results"
    makedirs(optimiser_tests_location, exist_ok=True)
    df = pd.DataFrame(metrics, columns=["step", "rmse_markers_ekf", "rmse_markers_opt"])
    df.to_csv(optimiser_tests_location + "/live_metrics.csv", index=False)
    print("Saved live_metrics.csv")

    # ---------- animation ----------
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_title("Live optimiser demo")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    scat_true = ax.scatter([], [], marker='o', label='Markers GT')
    scat_ekf  = ax.scatter([], [], marker='x', label='Markers EKF-only')
    scat_opt  = ax.scatter([], [], marker='^', label='Markers EKF+Optim')

    line_true, = ax.plot([], [], linestyle='--', label='True poses (kept)')
    line_ekf,  = ax.plot([], [], label='EKF poses')
    line_opt,  = ax.plot([], [], label='Optim poses')

    ax.legend(loc="best")

    # Set a fixed view box that encloses the arena
    all_markers = np.stack(list(markers_gt.values()))
    xmin, ymin = all_markers.min(axis=0) - 0.3
    xmax, ymax = all_markers.max(axis=0) + 0.3
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    def init():
        return scat_true, scat_ekf, scat_opt, line_true, line_ekf, line_opt

    def update(k):
        s = snaps[k]
        # markers
        scat_true.set_offsets(s["M_true"])
        scat_ekf.set_offsets(s["M_ekf"])
        scat_opt.set_offsets(s["M_opt"])
        # poses
        line_true.set_data(s["poses_true"][:,0], s["poses_true"][:,1])
        line_ekf.set_data(s["poses_ekf"][:,0],  s["poses_ekf"][:,1])
        line_opt.set_data(s["poses_opt"][:,0],  s["poses_opt"][:,1])
        ax.set_title(f"Live optimiser demo — step {s['step']} (frames kept)")
        return scat_true, scat_ekf, scat_opt, line_true, line_ekf, line_opt

    anim = FuncAnimation(fig, update, frames=len(snaps), init_func=init, blit=False, interval=200)
    # Save as GIF (PillowWriter comes with matplotlib)
    anim.save(optimiser_tests_location + "/optimiser_live.gif", writer=PillowWriter(fps=5))
    print("Saved optimiser_live.gif")
    plt.close(fig)

    # Also dump a static final figure
    last = snaps[-1]
    plt.figure()
    plt.scatter(last["M_true"][:,0], last["M_true"][:,1], marker='o', label='Markers GT')
    plt.scatter(last["M_ekf"][:,0],  last["M_ekf"][:,1],  marker='x', label='Markers EKF-only')
    plt.scatter(last["M_opt"][:,0],  last["M_opt"][:,1],  marker='^', label='Markers EKF+Optim')
    plt.plot(last["poses_true"][:,0], last["poses_true"][:,1], linestyle='--', label='True poses (kept)')
    plt.plot(last["poses_ekf"][:,0],  last["poses_ekf"][:,1],  label='EKF poses')
    plt.plot(last["poses_opt"][:,0],  last["poses_opt"][:,1],  label='Optim poses')
    plt.axis("equal"); plt.legend(); plt.title(f"Final snapshot {method}")
    plt.tight_layout(); plt.savefig(optimiser_tests_location + f"/optimiser_live_final_{method}.png")
    print(f"Saved optimiser_live_final_{method}.png")

if __name__ == "__main__":
    method = "schur"
    print("Running live optimiser demo (v3)...")
    run_live_demo()