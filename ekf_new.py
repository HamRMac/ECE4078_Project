import numpy as np
from mapping_utils import MappingUtils
import cv2
import math
import time
import pygame
from robot import Robot
import logging

log = logging.getLogger(__name__)

class EKF:
    # Implementation of an EKF for SLAM
    # The state is ordered as [x; y; theta; l1x; l1y; ...; lnx; lny]

    ##########################################
    # Utility
    # Add outlier rejection here
    ##########################################

    def __init__(self, robot: Robot):
        log.info("Initialising ekf.py V2.0")
        # State components
        self.robot = robot
        self.markers = np.zeros((2,0))
        self.taglist = []

        # Covariance matrix & landmark init
        self.P = np.zeros((3,3))
        self.init_lm_cov = 1e3  # Initial landmark covariance
        self.robot_init_state = None
        # Track whether we've locked the first landmark (reference) while stationary
        self.first_marker_locked = False
        # Store initial robot pose (will be zeros initially)
        self.robot_init_state = self.robot.state.copy()
        # Movement threshold (meters & radians) below which we consider the robot stationary
        self._stationary_lin_thresh = 1e-4  # 0.1 mm
        self._stationary_ang_thresh = 1e-4  # ~0.006 degrees
        # Static covariance for the robot's motion model
        # Added each time predict is called
        self.static_covariance = 5e-3
        # Robust ArUco update settings (can be overridden by caller)
        self.aruco_gate: float = 11.34  # chi2 gate (2D) ~99%
        self.aruco_kd: float = 1.0      # adaptive R scale: (1 + kd*range^2)

        # Graphics assets
        self.lm_pics = []
        for i in range(1, 11):
            f_ = f'./pics/8bit/lm_{i}.png'
            self.lm_pics.append(pygame.image.load(f_))
        f_ = f'./pics/8bit/lm_unknown.png'
        self.lm_pics.append(pygame.image.load(f_))
        self.pibot_pic = pygame.image.load(f'./pics/8bit/pibot_top.png')

    ##########################################
    # Seeding known landmarks (for known maps)
    ##########################################
    def seed_landmarks(self, tag_positions: dict, initial_covariance: float = 1e-10):
        """Pre-seed EKF with known ArUco landmark world positions.

        Parameters
        - tag_positions: dict[int -> tuple(float,float)] mapping tag id to (x, y)
        - initial_covariance: variance to set for each landmark coordinate. Use a
          very small number (e.g., 1e-10 .. 1e-8) to indicate strong certainty.

        Notes
        - This switches EKF into a localisation-only mode for those markers.
        - Landmark covariance is set on the diagonal; cross-covariances remain zero.
        - Any existing landmarks are cleared and replaced by this seed.
        """
        # Reset landmark containers
        self.markers = np.zeros((2, 0))
        self.taglist = []

        # Start from current robot covariance and expand for landmarks
        P_robot = self.P[0:3, 0:3] if self.P.shape[0] >= 3 else np.zeros((3, 3))
        P = P_robot

        for tag in sorted(tag_positions.keys()):
            xy = np.array(tag_positions[tag], dtype=float).reshape(2, 1)
            self.taglist.append(int(tag))
            self.markers = np.concatenate((self.markers, xy), axis=1)
            # Expand P by 2 dims and set small variance for the landmark
            P = np.concatenate((P, np.zeros((P.shape[0], 2))), axis=1)
            P = np.concatenate((P, np.zeros((2, P.shape[1]))), axis=0)
            P[-2, -2] = float(initial_covariance)
            P[-1, -1] = float(initial_covariance)

        self.P = 0.5 * (P + P.T)  # enforce symmetry

    def seed_from_map_file(self, fname: str, initial_covariance: float = 1e-10, only_aruco: bool = True):
        """Load marker positions from a true map JSON and seed landmarks.

        Accepts maps like M3_prac_map_min.txt (markers only) or full/part maps.
        - only_aruco=True: ignore all non-'aruco' entries.
        """
        import json
        tag_positions = {}
        with open(fname, 'r') as fd:
            gt = json.load(fd)
            for key, v in gt.items():
                if key.startswith('aruco'):
                    # Extract numeric id between 'aruco' and first '_'
                    try:
                        rest = key[5:]
                        num_str = rest.split('_')[0]
                        tag_id = int(num_str)
                    except Exception:
                        continue
                    tag_positions[tag_id] = (float(v['x']), float(v['y']))
                elif not only_aruco:
                    # Skip fruits/vegs when only_aruco=True
                    pass
        if tag_positions:
            self.seed_landmarks(tag_positions, initial_covariance)
        else:
            print('[EKF] seed_from_map_file: No aruco tags found to seed')

    def reset(self):
        self.robot.state = np.zeros((3, 1))
        self.markers = np.zeros((2,0))
        self.taglist = []
        # Covariance matrix
        self.P = np.zeros((3,3))
        self.init_lm_cov = 1e3
        self.robot_init_state = self.robot.state.copy()
        self.first_marker_locked = False

    def number_landmarks(self):
        return int(self.markers.shape[1])

    def get_state_vector(self):
        state = np.concatenate(
            (self.robot.state, np.reshape(self.markers, (-1,1), order='F')), axis=0)
        return state
    
    def set_state_vector(self, state):
        self.robot.state = state[0:3,:]
        self.markers = np.reshape(state[3:,:], (2,-1), order='F')
    
    def save_map(self, fname="slam_map.txt"):
        if self.number_landmarks() > 0:
            utils = MappingUtils(self.markers, self.P[3:,3:], self.taglist)
            utils.save(fname)

    def recover_from_pause(self, measurements):
        if not measurements:
            return False
        else:
            lm_new = np.zeros((2,0))
            lm_prev = np.zeros((2,0))
            tag = []
            for lm in measurements:
                if lm.tag in self.taglist:
                    lm_new = np.concatenate((lm_new, lm.position), axis=1)
                    tag.append(int(lm.tag))
                    lm_idx = self.taglist.index(lm.tag)
                    lm_prev = np.concatenate((lm_prev,self.markers[:,lm_idx].reshape(2, 1)), axis=1)
            if int(lm_new.shape[1]) > 2:
                R,t = self.umeyama(lm_new, lm_prev)
                theta = math.atan2(R[1][0], R[0][0])
                self.robot.state[:2]=t[:2]
                self.robot.state[2]=theta
                log.info("EKF recovered pose from %d landmarks", int(lm_new.shape[1]))
                return True
            else:
                return False
        
    ##########################################
    # EKF functions
    # Tune your SLAM algorithm here
    # ########################################

    # The prediction step of EKF
    def predict(self, raw_drive_meas):
        # Retrieve the required matricies

        v, w = self.robot.convert_wheel_speeds(raw_drive_meas.left_speed, raw_drive_meas.right_speed)
        eps = 1e-3
        if abs(v) < eps and abs(w) > eps:
            mode = "Pure Rotation"
            print("Pure Rotation")
            Q = self.predict_covariance(raw_drive_meas)
            # x and y can't change much when rotating
            Q[0,0] *= 0.01  # x position uncertainty
            Q[1,1] *= 0.01  # y position uncertainty
        elif abs(v) > eps and abs(w) < eps:
            mode = "Pure Translation"
            print("Pure Translation")
            Q = self.predict_covariance(raw_drive_meas)
            Q[2,2] *= 0.01  # heading uncertainty
            # theta can't change much when translating
        else:
            mode = "Stationary"
            print("Stationary or Arcing")
            # The robot must not drive in an arc as it will treat this as stationary at present.
            Q = self.predict_covariance(raw_drive_meas)
            Q[0:3,0:3] *= 0.01
            # Only small motion allowed in theta or x and y


        F = self.state_transition(raw_drive_meas)
        #Q = self.predict_covariance(raw_drive_meas)

        # Advance robot state only (landmarks fixed)
        self.robot.drive(raw_drive_meas) # This now becomes x_{k|k-1}

        # Store predicted state
        x_pred = self.get_state_vector()
        self.set_state_vector(x_pred)

        # Propagate the covariance
        self.P = F @ self.P @ F.T + Q # < The Q here is uncertainty
        
        # Enforce symmetry to correct for roundoff errors
        self.P = 0.5*(self.P + self.P.T)

    # The update step of EKF
    def update(self, measurements):
        print(time.time())

        if not measurements:
            return

        # Filter out measurements for unknown tags (not present in the EKF state)
        known_meas = [lm for lm in measurements if lm.tag in self.taglist]
        if not known_meas:
            return

        # Per-measurement gating and adaptive R
        inliers = []
        idxs = []
        Rs = []
        for lm in known_meas:
            try:
                idx = self.taglist.index(lm.tag)
            except ValueError:
                continue
            # Predicted measurement and Jacobian for this tag
            zhat_i = self.robot.measure(self.markers, [idx]).reshape((2,1), order='F')
            H_i = self.robot.derivative_measure(self.markers, [idx])
            z_i = lm.position.reshape((2,1))
            R_i = lm.covariance.copy()
            # Adaptive R based on range (norm of body-frame measurement)
            try:
                rng = float(np.linalg.norm(z_i))
                scale = 1.0 + max(0.0, float(self.aruco_kd)) * (rng * rng)
                R_i = R_i * float(scale)
            except Exception:
                pass
            # Innovation gating
            y_i = z_i - zhat_i
            S_i = H_i @ self.P @ H_i.T + R_i
            try:
                d2 = float(y_i.T @ np.linalg.inv(S_i) @ y_i)
            except Exception:
                d2 = 0.0
            if d2 <= float(self.aruco_gate):
                inliers.append((z_i, H_i, R_i))
                idxs.append(idx)
                Rs.append(R_i)
            else:
                # reject outlier silently
                continue

        if not inliers:
            return

        # Stack inliers
        z = np.concatenate([zi for (zi, _, _) in inliers], axis=0)
        H = np.concatenate([Hi for (_, Hi, _) in inliers], axis=0)
        Rm = np.zeros((2*len(inliers), 2*len(inliers)))
        for i, R_i in enumerate(Rs):
            Rm[2*i:2*i+2, 2*i:2*i+2] = R_i

        x = self.get_state_vector()
        y = z - self.robot.measure(self.markers, idxs).reshape((-1,1), order='F')
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        x_new = x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.set_state_vector(x_new)


    def state_transition(self, raw_drive_meas):
        n = self.number_landmarks()*2 + 3
        F = np.eye(n)
        F[0:3,0:3] = self.robot.derivative_drive(raw_drive_meas)
        return F
    
    def predict_covariance(self, raw_drive_meas):
        n = self.number_landmarks()*2 + 3
        Q = np.zeros((n,n))
        Q[0:3,0:3] = self.robot.covariance_drive(raw_drive_meas) + self.static_covariance*np.eye(3)
        return Q
    
    # Helper: determine if robot has moved since initialisation
    def _robot_is_stationary(self):
        if self.robot_init_state is None:
            return True
        dpos = np.linalg.norm(self.robot.state[0:2] - self.robot_init_state[0:2])
        dth = abs(float(self.robot.state[2] - self.robot_init_state[2]))
        return (dpos < self._stationary_lin_thresh) and (dth < self._stationary_ang_thresh)

    def add_landmarks(self, measurements):
        if not measurements:
            return

        th = self.robot.state[2]
        robot_xy = self.robot.state[0:2,:]
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])


        # Add new landmarks to the state
        for lm in measurements:
            if lm.tag in self.taglist:
                # ignore known tags
                continue
            
            lm_bff = lm.position

            lm_inertial = robot_xy + R_theta @ lm_bff

            self.taglist.append(int(lm.tag))
            self.markers = np.concatenate((self.markers, lm_inertial), axis=1)

            # Expand covariance matrix for new landmark
            self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
            self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)

            # Default: large initial uncertainty (squared because init_lm_cov is std-like)
            new_cov_val = self.init_lm_cov**2

            # If this is the FIRST landmark added while robot is stationary, lock it with tiny covariance
            if (len(self.taglist) == 1  # we just appended this landmark
                and not self.first_marker_locked
                and self._robot_is_stationary()):
                new_cov_val = 1e-12  # extremely high certainty ~ effectively fixed reference
                self.first_marker_locked = True
                # Print lock info: id and estimated position (rounded to 2 decimals)
                pos = lm_inertial.flatten()
                pos_str = f"({pos[0]:.2f}, {pos[1]:.2f})"
                log.info("[EKF] Locked marker %s at %s as reference (high certainty)", lm.tag, pos_str)

            self.P[-2,-2] = new_cov_val
            self.P[-1,-1] = new_cov_val

            # (Optional) enforce symmetry
            self.P = 0.5*(self.P + self.P.T)

    ##########################################
    ##########################################
    ##########################################

    @staticmethod
    def umeyama(from_points, to_points):

    
        assert len(from_points.shape) == 2, \
            "from_points must be a m x n array"
        assert from_points.shape == to_points.shape, \
            "from_points and to_points must have the same shape"
        
        N = from_points.shape[1]
        m = 2
        
        mean_from = from_points.mean(axis = 1).reshape((2,1))
        mean_to = to_points.mean(axis = 1).reshape((2,1))
        
        delta_from = from_points - mean_from # N x m
        delta_to = to_points - mean_to       # N x m
        
        cov_matrix = delta_to @ delta_from.T / N
        
        U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
        cov_rank = np.linalg.matrix_rank(cov_matrix)
        S = np.eye(m)
        
        if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
            S[m-1, m-1] = -1
        elif cov_rank < m-1:
            raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
        
        R = U.dot(S).dot(V_t)
        t = mean_to - R.dot(mean_from)
    
        return R, t

    # Plotting functions
    # ------------------
    @ staticmethod
    def to_im_coor(xy, res, m2pixel):
        w, h = res
        x, y = xy
        x_im = int(-x*m2pixel+w/2.0)
        y_im = int(y*m2pixel+h/2.0)
        return (x_im, y_im)

    def draw_slam_state(self, res=(320, 500), not_pause=True,
                    draw_grid=True, grid_spacing_m=1.0,
                    draw_subgrid=False, subgrid_spacing_m=0.25, subgrid_alpha=0.3,
                    grid_at_origin: bool = False):
        """
        Draw the SLAM state visualization.
        If grid_at_origin is True, the grid is centered at the world origin (0,0).
        Otherwise, the grid is placed as in the original logic (robot-centric).
        """
        # scale: meters -> pixels
        m2pixel = 100

        # background
        bg_rgb = np.array([213, 213, 213] if not_pause else [120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((res[1], res[0], 3)) * bg_rgb.astype(np.uint8)

        # --- GRID (under everything) ---
        if draw_grid or draw_subgrid:
            def draw_grid_lines(dst, origin_uv, spacing_m, color=(170,170,170), thickness=1):
                spacing_px = max(1, int(round(spacing_m * m2pixel)))
                w, h = res[0], res[1]
                u0, v0 = int(origin_uv[0]), int(origin_uv[1])

                # verticals to the right
                u = u0
                while u < w:
                    cv2.line(dst, (u, 0), (u, h-1), color, thickness)
                    u += spacing_px
                # verticals to the left
                u = u0 - spacing_px
                while u >= 0:
                    cv2.line(dst, (u, 0), (u, h-1), color, thickness)
                    u -= spacing_px
                # horizontals downward
                v = v0
                while v < h:
                    cv2.line(dst, (0, v), (w-1, v), color, thickness)
                    v += spacing_px
                # horizontals upward
                v = v0 - spacing_px
                while v >= 0:
                    cv2.line(dst, (0, v), (w-1, v), color, thickness)
                    v -= spacing_px

            # Determine grid origin in image coordinates
            if grid_at_origin:
                # Always center grid at world origin (0,0)
                origin_uv = self.to_im_coor((0, 0), res, m2pixel)
            else:
                # Default: center grid at robot's current position
                robot_xy = self.robot.state[:2, 0] if self.robot.state.shape == (3, 1) else self.robot.state[:2]
                origin_uv = self.to_im_coor((-robot_xy[0], -robot_xy[1]), res, m2pixel)

            # main grid (solid grey)
            if draw_grid:
                draw_grid_lines(canvas, origin_uv, grid_spacing_m, color=(170,170,170), thickness=1)

            # subgrid (overlay with lower opacity)
            if draw_subgrid:
                overlay = canvas.copy()
                draw_grid_lines(overlay, origin_uv, subgrid_spacing_m, color=(170,170,170), thickness=1)
                cv2.addWeighted(overlay, subgrid_alpha, canvas, 1.0 - subgrid_alpha, 0, dst=canvas)

        # --- Pose/landmarks in robot-centric frame ---
        lms_xy = self.markers[:2, :]
        robot_xy = self.robot.state[:2, 0].reshape((2, 1))
        lms_xy = lms_xy - robot_xy
        robot_xy = robot_xy * 0
        robot_theta = self.robot.state[2, 0]

        # robot covariance ellipse (in canvas space)
        start_point_uv = self.to_im_coor((0, 0), res, m2pixel)
        p_robot = self.P[0:2, 0:2]
        axes_len, angle = self.make_ellipse(p_robot)
        canvas = cv2.ellipse(canvas, start_point_uv,
                            (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                            angle, 0, 360, (0, 30, 56), 1)

        # landmark ellipses
        if self.number_landmarks() > 0:
            for i in range(self.markers.shape[1]):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                Plmi = self.P[3+2*i:3+2*(i+1), 3+2*i:3+2*(i+1)]
                axes_len, angle = self.make_ellipse(Plmi)
                canvas = cv2.ellipse(canvas, coor_,
                                    (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                                    angle, 0, 360, (244, 69, 96), 1)

        # convert to pygame surface (your original transforms)
        surface = pygame.surfarray.make_surface(np.rot90(canvas))
        surface = pygame.transform.flip(surface, True, False)

        # robot sprite
        surface.blit(self.rot_center(self.pibot_pic, robot_theta*57.3),
                    (start_point_uv[0]-15, start_point_uv[1]-15))

        # landmark sprites
        if self.number_landmarks() > 0:
            for i in range(self.markers.shape[1]):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                try:
                    surface.blit(self.lm_pics[self.taglist[i]-1], (coor_[0]-5, coor_[1]-5))
                except IndexError:
                    surface.blit(self.lm_pics[-1], (coor_[0]-5, coor_[1]-5))

        # --- ORIGIN MARKERS ON TOP OF EVERYTHING ---
        # Compute world-origin image coordinate explicitly so arrows are anchored at WORLD origin
        # (independent of whether the grid was drawn robot-centric or world-centric)
        origin_uv = self.to_im_coor((0.0, 0.0), res, m2pixel)  # world origin (0,0) in image coords

        arrow_len_m = 0.2  # 20 cm arrows
        # black '+' at world origin (arms in world coordinates for consistency)
        plus_len_m = 0.05
        p1 = self.to_im_coor((-plus_len_m, 0.0), res, m2pixel)
        p2 = self.to_im_coor((+plus_len_m, 0.0), res, m2pixel)
        p3 = self.to_im_coor((0.0, -plus_len_m), res, m2pixel)
        p4 = self.to_im_coor((0.0, +plus_len_m), res, m2pixel)
        pygame.draw.line(surface, (0, 0, 0), p1, p2, 2)
        pygame.draw.line(surface, (0, 0, 0), p3, p4, 2)

        # Draw axes arrows anchored at the WORLD origin (origin_uv) rather than robot position
        # red +x arrow (world +X)
        x_tip = self.to_im_coor((arrow_len_m, 0.0), res, m2pixel)
        pygame.draw.line(surface, (255, 0, 0), origin_uv, x_tip, 3)
        # green +y arrow (world +Y)
        y_tip = self.to_im_coor((0.0, arrow_len_m), res, m2pixel)
        pygame.draw.line(surface, (0, 200, 0), origin_uv, y_tip, 3)

        return surface


    @staticmethod
    def rot_center(image, angle):
        """rotate an image while keeping its center and size"""
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image       

    @staticmethod
    def make_ellipse(P):
        e_vals, e_vecs = np.linalg.eig(P)
        idx = e_vals.argsort()[::-1]   
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        alpha = np.sqrt(4.605)
        axes_len = e_vals*2*alpha
        if abs(e_vecs[1, 0]) > 1e-3:
            angle = np.arctan(e_vecs[0, 0]/e_vecs[1, 0])
        else:
            angle = 0
        return (axes_len[0], axes_len[1]), angle
