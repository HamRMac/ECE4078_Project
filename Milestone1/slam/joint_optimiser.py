"""joint_optimiser.py

2D joint (bundle-like) optimisation of camera poses and ARUCO marker positions.

We build a factor graph from frames (images) that observe >=2 markers. Each frame j
has an initial pose (X_j, Y_j, theta_j). Each observation supplies a relative 2D
body-frame vector b_ij derived from ArUco tvec: b_ij = [Tz, -Tx].

Residual for observation of marker i in frame j:
    r_ij = M_i - ( P_j + R(theta_j) @ b_ij ) ; shape (2,)

Stack all residuals to build Y. State S stacks all camera poses first (3 per frame)
then marker positions (2 per unique marker):
    S = [ X_0,Y_0,theta_0, X_1,Y_1,theta_1, ... , M_0x,M_0y, M_1x,M_1y, ... ]

Jacobian blocks:
  ∂r/∂M_i = I_2
  ∂r/∂P_j(X,Y) = -I_2
  ∂r/∂theta_j = - d(R*b)/dθ, with dR/dθ = [[-sinθ,-cosθ],[cosθ,-sinθ]]

Levenberg–Marquardt update:
  Solve (J^T J + λ diag(J^T J)) δ = - J^T r
  S <- S + δ  (accept if error decreases, else increase λ and retry)

Usage Example:
    opt = JointOptimiser2D()
    opt.add_frame(pose_init, [(marker_id, bff_vec2), ...])
    cam_poses, marker_map = opt.optimise()

Returned:
  cam_poses: list of (X,Y,theta)
  marker_map: dict marker_id -> np.array([x,y])

Robustness notes: This is a minimal implementation for small graphs. No robust
loss or covariance weighting included (can be extended later).
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict
from warnings import warn

class JointOptimiser2D:
    def __init__(self):
        self.frames: List[dict] = []  # each: {"pose": np.array([x,y,theta]), "obs":[(id, vec2), ...]}
        self._marker_ids: set[int] = set()
        
        # Gauge handling: "none", "prior_full", "prior_yaw", "fix_pose0"
        self._GAUGE_MODE: str = "prior_full"

        if self._GAUGE_MODE not in ["none", "prior_full", "prior_yaw", "fix_pose0"]:
            raise ValueError(f"Unknown GAUGE_MODE: {self._GAUGE_MODE}")

        self._YAW_PRIOR_DEG: float = 5.0  # used only when GAUGE_MODE == "prior_yaw"
        self._OPT_PRIOR_SIGMAS: tuple[float, float, float] = (0.50, 0.50, 30.0)  # (meters, meters, degrees)
        # /\ Avoid super tight priors unless you are certain the EKF’s first saved pose is very accurate.
        self._pose0_ref = None

    @property
    def GAUGE_MODE(self):
        return self._GAUGE_MODE

    @property
    def YAW_PRIOR_DEG(self):
        return self._YAW_PRIOR_DEG

    @property
    def OPT_PRIOR_SIGMAS(self):
        return self._OPT_PRIOR_SIGMAS
    
    @GAUGE_MODE.setter
    def GAUGE_MODE(self, value: str):
        """
        Any of none, prior_full, prior_yaw, fix_pose0   
        """
        if value not in ["none", "prior_full", "prior_yaw", "fix_pose0"]:
            raise ValueError(f"Unknown GAUGE_MODE: {value}")
        self._GAUGE_MODE = value

    @YAW_PRIOR_DEG.setter
    def YAW_PRIOR_DEG(self, value: float):
        if self._GAUGE_MODE != "prior_yaw":
            warn("YAW_PRIOR_DEG is only used when GAUGE_MODE is prior_yaw.")
        self._YAW_PRIOR_DEG = value

    @OPT_PRIOR_SIGMAS.setter
    def OPT_PRIOR_SIGMAS(self, value: tuple[float, float, float]):
        if len(value) != 3:
            raise ValueError(f"Invalid OPT_PRIOR_SIGMAS: {value}")
        if self._GAUGE_MODE != "prior_full":
            warn("OPT_PRIOR_SIGMAS is only used when GAUGE_MODE is prior_full.")
        self._OPT_PRIOR_SIGMAS = value

    def add_frame(self, pose_initial: np.ndarray, observations: List[Tuple[int, np.ndarray]]):
        """Add a frame if it has >=2 unique marker observations.

        pose_initial: shape (3,) or (3,1)
        observations: list of (marker_id, body_frame_vec2) where vec2 shape (2,) or (2,1)
        """
        # Extract each visible marker's body-frame vector from the frame
        # also make sure we only keep 1 observation per marker
        uniq = {}
        for mid, v in observations:
            if mid not in uniq:
                uniq[mid] = np.asarray(v).reshape(2)
        # Ensure we only frames with 2+ markers otherwise the optimiser
        # cannot constrain the frame
        if len(uniq) < 2:
            return  # discard single-marker frames

        # Add the frame and the pose to the optimiser's global copy
        pose = np.asarray(pose_initial).reshape(3).copy()
        obs_list = [(mid, uniq[mid]) for mid in sorted(uniq.keys())]
        for mid in uniq.keys():
            self._marker_ids.add(mid)
        self.frames.append({"pose": pose, "obs": obs_list})

        # Pin the first frame's pose if enabled
        if len(self.frames) == 1 and self._GAUGE_MODE in ("prior_full", "prior_yaw", "fix_pose0"):
            self._pose0_ref = np.array(pose, dtype=float)

    def optimise(self,
                 max_iters: int = 25,
                 lambda_init: float = 1e-2,
                 tol: float = 1e-6,
                 method: str = "dense"):
        """Run joint optimisation. Returns (camera_poses, marker_map).

        camera_poses: list of np.array([x,y,theta]) following input frame order
        marker_map: dict of marker_id -> np.array([x,y])
        """
        # If we don't have enough info return just poses from the frames
        if len(self.frames) == 0 or len(self._marker_ids) == 0:
            return [f["pose"].copy() for f in self.frames], {}

        # Build marker lookup index
        marker_ids = sorted(self._marker_ids)
        m_index = {mid: i for i, mid in enumerate(marker_ids)}

        # Initialise marker positions and number of times we see the marker
        marker_init = {mid: np.zeros(2) for mid in marker_ids}
        marker_counts = {mid: 0 for mid in marker_ids}

        # Estimate initial marker positions
        for f in self.frames:
            x, y, th = f["pose"] # Extract the pose from the frame
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]]) # Build rotation matrix
            for mid, b in f["obs"]: # For each marker observation in the frame
                est = np.array([x, y]) + R @ b # Estimate its location
                marker_init[mid] += est # Accumulate estimates
                marker_counts[mid] += 1 # Accumulate counts
        
        # For each marker that has been seen at least once
        # average the estimates
        for mid in marker_ids:
            if marker_counts[mid] > 0:
                marker_init[mid] /= marker_counts[mid]

        nF = len(self.frames)
        nM = len(marker_ids)
        # Build the state vector (camera poses + marker positions)
        S = np.zeros(self._n_pose_vars(nF) + 2*nM)
        # For each frame, add the pose to the state vector (first 3 rows)
        for j, f in enumerate(self.frames):
            off = self._pose_offset(j)
            if off is None:
                continue  # pose 0 is fixed, not stored in S
            S[off:off+3] = f["pose"]
        # For each marker, add its estimated position to the state vector
        for mid in marker_ids:
            k = m_index[mid]
            m_off = self._marker_offset(k, nF)
            S[m_off:m_off+2] = marker_init[mid]

        # Initialise damping parameter for Levenberg–Marquardt
        lamb = lambda_init
        prev_err = np.inf  # Track previous error for convergence
        for itr in range(max_iters):
            # --- Compute residuals and build block list ---
            r_list = []  # List of 2D residuals for all observations
            blocks = []  # List of (frame_idx, marker_id, body_frame_vec)
            for j, f in enumerate(self.frames):
                # Extract current pose estimate for frame j
                if self._fix_pose0() and j == 0:
                    xj, yj, thj = self._pose0_ref
                else:
                    off = self._pose_offset(j)
                    xj, yj, thj = S[off:off+3]
                c, s = np.cos(thj), np.sin(thj)
                Rj = np.array([[c, -s], [s, c]])  # Rotation matrix for frame
                for mid, b in f["obs"]:
                    k = m_index[mid]  # Marker index
                    m_off = self._marker_offset(k, nF) # Marker offset
                    Mk = S[m_off:m_off+2]  # Marker position estimate
                    pred = np.array([xj, yj]) + Rj @ b  # Predicted marker position in world
                    r = Mk - pred  # Residual: difference between estimated and predicted
                    r_list.append(r)
                    blocks.append((j, mid, b))  # Store for Jacobian construction
            # Stack all residuals into a single vector
            r_vec = np.concatenate(r_list, axis=0)
            # Evaluate current error using measurement-only residuals
            r_meas = r_vec.copy()  # keep measurement-only residual for error checks
            err = float(r_meas @ r_meas)  # Total squared error

            # If error increases dramatically, break (divergence safeguard)
            if err > prev_err + 1e6:
                break

            # --- Choose LM step implementation ---
            method_l = method.lower()
            if method_l not in ("dense", "schur"):
                raise ValueError(f"Unknown method: {method}")

            if method_l == "dense":
                # --- Build Jacobian matrix (dense path) ---
                nObs = len(blocks)
                dim  = S.size
                J = np.zeros((2*nObs, dim))
                for obs_i, (frame_idx, mid, b) in enumerate(blocks):
                    row = 2*obs_i
                    k = m_index[mid]
                    m_off = self._marker_offset(k, nF)

                    # wrt marker
                    J[row:row+2, m_off:m_off+2] = np.eye(2)

                    # wrt frame (if variable)
                    f_off = self._pose_offset(frame_idx)
                    if f_off is not None:
                        xj, yj, thj = S[f_off:f_off+3]
                        c, s = np.cos(thj), np.sin(thj)
                        dR_dth = np.array([[-s, -c],[c, -s]])
                        J[row:row+2, f_off:f_off+2] = -np.eye(2)
                        J[row:row+2, f_off+2]       = - (dR_dth @ b)

                # Append prior rows only if pose0 is in the state
                r_vec_aug = r_vec
                if self._GAUGE_MODE == "prior_full" and self._pose0_ref is not None and not self._fix_pose0():
                    J, r_vec_aug = self._add_pose0_prior_full(J, r_vec_aug, S, self._pose0_ref, self._OPT_PRIOR_SIGMAS)
                elif self._GAUGE_MODE == "prior_yaw" and self._pose0_ref is not None and not self._fix_pose0():
                    J, r_vec_aug = self._add_pose0_prior_yaw(J, r_vec_aug, S, self._pose0_ref, self._YAW_PRIOR_DEG)

                # --- Levenberg–Marquardt update (dense) ---
                JTJ = J.T @ J
                g   = J.T @ r_vec_aug
                A   = JTJ + lamb * np.diag(np.diag(JTJ))
                try:
                    delta = -np.linalg.solve(A, g)
                except np.linalg.LinAlgError:
                    delta = -np.linalg.pinv(A) @ g

            else:
                # --- Schur-complement LM step (poses-only reduced system) ---
                use_prior = (self._GAUGE_MODE in ("prior_full", "prior_yaw")
                            and self._pose0_ref is not None
                            and not self._fix_pose0())
                delta = self._lm_step_schur(S, m_index, marker_ids, nF, lamb,
                                            use_prior=use_prior, prior_mode=self._GAUGE_MODE)


            # If update step is very small, stop (converged)
            if np.linalg.norm(delta) < tol:
                break
            
            # --- Evaluate new state ---
            S_new = S + delta
            self._wrap_all_pose_angles(S_new, nF)

            # Recompute measurement-only residuals at S_new
            r_new_list = []
            for j, f in enumerate(self.frames):
                if self._fix_pose0() and j == 0:
                    xj, yj, thj = self._pose0_ref
                else:
                    off = self._pose_offset(j)
                    xj, yj, thj = S_new[off:off+3]
                c, s = np.cos(thj), np.sin(thj)
                Rj = np.array([[c, -s], [s, c]])
                for mid, b in f["obs"]:
                    k = m_index[mid]
                    m_off = self._marker_offset(k, nF)
                    Mk = S_new[m_off:m_off+2]
                    pred = np.array([xj, yj]) + Rj @ b
                    r_new_list.append(Mk - pred)

            err_new = float(np.concatenate(r_new_list) @ np.concatenate(r_new_list))

            # Accept or reject update based on error
            if err_new < err:
                # Accept: update state, decrease damping
                S = S_new
                prev_err = err_new
                lamb = max(lamb * 0.5, 1e-6)
            else:
                # Reject: increase damping, possibly break if too large
                lamb = min(lamb * 2.0, 1e6)
                if lamb > 1e5:
                    break
        
        # Wrap angles after the main loop
        self._wrap_all_pose_angles(S, nF)

        # --- Extract results from optimised state vector ---
        cam_poses = []
        for j in range(nF):
            off = self._pose_offset(j)
            if off is None:  # pose 0 fixed, not in S
                cam_poses.append(np.array(self._pose0_ref, dtype=float))
            else:
                cam_poses.append(S[off:off+3].copy())

        # Markers: use helper to be correct for all gauge modes
        marker_map = {}
        for mid in marker_ids:
            k = m_index[mid]
            m_off = self._marker_offset(k, nF)   # <-- important
            marker_map[mid] = S[m_off:m_off+2].copy()

        # Return corrected poses and markers
        return cam_poses, marker_map

    def _lm_step_schur(self, S, m_index, marker_ids, nF, lamb,
                   use_prior: bool, prior_mode: str):
        """
        One LM step via Schur complement:
        1) Build reduced normal equations over poses only.
        2) Solve for Δposes.
        3) Back-substitute Δmarkers.
        Returns: full delta vector matching S.shape.
        """
        pose_dim = self._n_pose_vars(nF)
        A   = np.zeros((pose_dim, pose_dim))  # reduced Hessian over poses
        gp  = np.zeros(pose_dim)              # reduced gradient over poses

        # Group observations by marker: for each marker k, store list of (pose_off, Jp_ij, r_ij)
        obs_by_marker = {m_index[mid]: [] for mid in marker_ids}

        # Pass 1: accumulate per-pose blocks and stash per-marker triplets
        for j, f in enumerate(self.frames):
            # pose at current linearization point
            if self._fix_pose0() and j == 0:
                xj, yj, thj = self._pose0_ref
                f_off = None
            else:
                f_off = self._pose_offset(j)
                xj, yj, thj = S[f_off:f_off+3]
            c, s = np.cos(thj), np.sin(thj)
            Rj = np.array([[c, -s],[s, c]])
            dR_dth = np.array([[-s, -c],[c, -s]])

            for mid, b in f["obs"]:
                k = m_index[mid]
                m_off = self._marker_offset(k, nF)
                Mk = S[m_off:m_off+2]
                pred = np.array([xj, yj]) + Rj @ b
                r = Mk - pred  # 2-vector

                # Jp block (2x3) for this observation: [-I2 | -(dR_dth@b)]
                Jp = np.zeros((2,3))
                Jp[:, 0:2] = -np.eye(2)
                Jp[:, 2]   = -(dR_dth @ b)

                # Accumulate into reduced pose system for this single obs (diagonal block)
                if f_off is not None:
                    A[f_off:f_off+3, f_off:f_off+3] += Jp.T @ Jp
                    gp[f_off:f_off+3]               += Jp.T @ r

                # Stash triplet (we'll form Schur coupling across obs of the same marker)
                obs_by_marker[k].append((f_off, Jp, r))

        # Levenberg damping on the reduced pose system
        A += lamb * np.diag(np.diag(A))

        # Add priors directly to the reduced system if pose0 is present
        if use_prior and not self._fix_pose0():
            if prior_mode == "prior_full":
                sx, sy, sdeg = self._OPT_PRIOR_SIGMAS
                sth = np.deg2rad(sdeg)
                W = np.diag([1.0/sx, 1.0/sy, 1.0/sth])
                W2  = W @ W
                dth = self._wrap_pi(S[2] - self._pose0_ref[2])
                rp  = np.array([S[0]-self._pose0_ref[0], S[1]-self._pose0_ref[1], dth])
                A[0:3, 0:3] += W2
                gp[0:3]     += W2 @ rp
            elif prior_mode == "prior_yaw":
                sth = np.deg2rad(self._YAW_PRIOR_DEG)
                w2 = 1.0/(sth**2)
                dth = self._wrap_pi(S[2] - self._pose0_ref[2])
                A[2,2] += w2
                gp[2]  += w2 * dth

        # Schur reduction: subtract marker coupling terms
        # With Jm = I2, each marker's H_mm = count_i * I2 and Cinv = (1/count_i) * I2
        S_p = A.copy()
        b_p = gp.copy()

        for k, obs_list in obs_by_marker.items():
            if not obs_list:
                continue
            count_i = len(obs_list)
            Cinv = 1.0 / max(count_i, 1)

            # gm_i = sum r_ij (2-vector)
            gm_i = np.sum([r for (_,_,r) in obs_list], axis=0)

            # Vector term: b_p -= Jp^T Cinv gm_i
            for (off_a, Jp_a, _) in obs_list:
                if off_a is not None:
                    b_p[off_a:off_a+3] -= Cinv * (Jp_a.T @ gm_i)

            # Matrix term: S_p -= Jp_a^T Cinv Jp_b for all obs pairs (a,b) that are on poses
            for (off_a, Jp_a, _) in obs_list:
                if off_a is None:
                    continue
                a_sl = slice(off_a, off_a+3)
                JaT = Jp_a.T
                for (off_b, Jp_b, _) in obs_list:
                    if off_b is None:
                        continue
                    b_sl = slice(off_b, off_b+3)
                    S_p[a_sl, b_sl] -= Cinv * (JaT @ Jp_b)

        # Solve reduced system for Δposes
        try:
            delta_p = -np.linalg.solve(S_p, b_p)
        except np.linalg.LinAlgError:
            delta_p = -np.linalg.pinv(S_p) @ b_p

        # Back-substitute Δmarkers: dm_i = Cinv*(gm_i - sum_j Jp_ij Δp_j)
        delta = np.zeros_like(S)
        # scatter pose deltas
        for j in range(nF):
            off = self._pose_offset(j)
            if off is not None:
                delta[off:off+3] = delta_p[off:off+3]

        for k, obs_list in obs_by_marker.items():
            if not obs_list:
                continue
            count_i = len(obs_list)
            Cinv = 1.0 / max(count_i, 1)
            gm_i = np.sum([r for (_,_,r) in obs_list], axis=0)
            accum = np.zeros(2)
            for (off_j, Jp_j, _) in obs_list:
                if off_j is not None:
                    accum += Jp_j @ delta_p[off_j:off_j+3]
            dm_i = Cinv * (gm_i - accum)
            m_off = self._marker_offset(k, nF)
            delta[m_off:m_off+2] = dm_i

        return delta

    
    def _wrap_pi(self, a: float) -> float:
        """
        Small utility function to wrap angles to the range [-pi, pi].
        a: float The angle in radians to wrap.
        return: float The wrapped angle in radians.
        """
        return (a + np.pi) % (2*np.pi) - np.pi
    
    def _fix_pose0(self) -> bool:
        return self._GAUGE_MODE == "fix_pose0"

    def _n_pose_vars(self, nF: int) -> int:
        return 3*nF if not self._fix_pose0() else 3*(nF-1)

    def _pose_offset(self, j: int) -> int | None:
        """
        Return column offset for pose j in S, or None if j==0 and pose0 is fixed.
        """
        if self._fix_pose0() and j == 0:
            return None
        return 3*j if not self._fix_pose0() else 3*(j-1)

    def _marker_offset(self, k: int, nF: int) -> int:
        return self._n_pose_vars(nF) + 2*k

    def _wrap_all_pose_angles(self, S: np.ndarray, nF: int) -> None:
        start = 0 if not self._fix_pose0() else 1
        for j in range(start, nF):
            off = self._pose_offset(j)
            S[off+2] = self._wrap_pi(S[off+2])

    def _add_pose0_prior_full(self, J, r, S, pose0_ref, sigmas_xy_deg):
        sx, sy, sdeg = sigmas_xy_deg
        sth = np.deg2rad(sdeg)
        W = np.diag([1.0/sx, 1.0/sy, 1.0/sth])

        Jp = np.zeros((3, J.shape[1]), dtype=J.dtype)
        Jp[:, 0:3] = np.eye(3)  # pose 0 columns when not fixing pose 0

        dth = self._wrap_pi(S[2] - pose0_ref[2])
        rp  = np.array([S[0]-pose0_ref[0], S[1]-pose0_ref[1], dth], dtype=J.dtype)

        return np.vstack([J, W @ Jp]), np.hstack([r, (W @ rp)])

    def _add_pose0_prior_yaw(self, J, r, S, pose0_ref, yaw_sigma_deg):
        sth = np.deg2rad(yaw_sigma_deg)
        W = np.array([[1.0/sth]])

        Jp = np.zeros((1, J.shape[1]), dtype=J.dtype)
        Jp[0, 2] = 1.0  # theta_0 column when not fixing pose 0

        dth = self._wrap_pi(S[2] - pose0_ref[2])
        rp  = np.array([dth], dtype=J.dtype)

        return np.vstack([J, W @ Jp]), np.hstack([r, (W @ rp)])

__all__ = ["JointOptimiser2D"]
