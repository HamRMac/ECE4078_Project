import numpy as np

class Robot:
    def __init__(self, wheels_width, wheels_scale, camera_matrix, camera_dist):
        print("Initialising robot.py V1.0")
        # State is a vector of [x,y,theta]'
        self.state = np.zeros((3,1))
        
        # Wheel parameters
        self.wheels_width = wheels_width  # The distance between the left and right wheels
        self.wheels_scale = wheels_scale  # The scaling factor converting ticks/s to m/s

        # Camera parameters
        self.camera_matrix = camera_matrix  # Matrix of the focal lengths and camera centre
        self.camera_dist = camera_dist  # Distortion coefficients
    
    def drive(self, drive_meas):
        # left_speed and right_speed are the speeds in ticks/s of the left and right wheels.
        # dt is the length of time to drive for

        # Compute the linear and angular velocity
        linear_velocity, angular_velocity = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        # Apply the velocities
        dt = drive_meas.dt
        if angular_velocity == 0:
            self.state[0] += np.cos(self.state[2]) * linear_velocity * dt
            self.state[1] += np.sin(self.state[2]) * linear_velocity * dt
        else:
            th = self.state[2]
            self.state[0] += linear_velocity / angular_velocity * (np.sin(th+dt*angular_velocity) - np.sin(th))
            self.state[1] += -linear_velocity / angular_velocity * (np.cos(th+dt*angular_velocity) - np.cos(th))
            self.state[2] += dt*angular_velocity

    def measure(self, markers, idx_list):
        # Markers are 2d landmarks in a 2xn structure where there are n landmarks.
        # The index list tells the function which landmarks to measure in order.
        
        # Construct a 2x2 rotation matrix from the robot angle
        th = self.state[2]
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        robot_xy = self.state[0:2,:]

        measurements = []
        for idx in idx_list:
            marker = markers[:,idx:idx+1]
            marker_bff = Rot_theta.T @ (marker - robot_xy)
            measurements.append(marker_bff)

        # Stack the measurements in a 2xm structure.
        markers_bff = np.concatenate(measurements, axis=1)
        return markers_bff
    
    def convert_wheel_speeds(self, left_speed, right_speed):
        # Convert to m/s
        left_speed_m = left_speed * self.wheels_scale
        right_speed_m = right_speed * self.wheels_scale

        # Compute the linear and angular velocity
        linear_velocity = (left_speed_m + right_speed_m) / 2.0
        angular_velocity = (right_speed_m - left_speed_m) / self.wheels_width
        
        return linear_velocity, angular_velocity

    # Derivatives and Covariance
    # --------------------------

    def derivative_drive(self, drive_meas):
        """
        Jacobian of the differential-drive motion model wrt robot state x=[x,y,theta]^T.
        Returns DFx = ∂f/∂x (3x3) for the one-step update used in drive().
        """
        # Define identity matrix
        DFx = np.eye(3)

        # Get the required information (states, dt)
        v, w = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)
        dt = drive_meas.dt
        th = float(self.state[2])

        # Compute the relevant jacobian depending on if we are turning or not
        if w < 1e-9:
            # Simple non turning model
            # x' = x + v cosθ dt ; y' = y + v sinθ dt ; θ' = θ
            DFx[0,2] = -v * np.sin(th) * dt
            DFx[1,2] =  v * np.cos(th) * dt
        else:
            # Turning model with turning radius
            th2 = th + w*dt
            # arc Jacobian wrt θ (v, w are treated as inputs here)
            DFx[0,2] = (v/w) * (np.cos(th2) - np.cos(th))
            DFx[1,2] = (v/w) * (np.sin(th2) - np.sin(th))

        return DFx

    def derivative_measure(self, markers, idx_list):
        # Compute the derivative of the markers in the order given by idx_list w.r.t. robot and markers
        n = 2*len(idx_list)
        m = 3 + 2*markers.shape[1]

        DH = np.zeros((n,m))

        robot_xy = self.state[0:2,:]
        th = self.state[2]        
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        DRot_theta = np.block([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])

        for i in range(n//2):
            j = idx_list[i]
            # i identifies which measurement to differentiate.
            # j identifies the marker that i corresponds to.

            lmj_inertial = markers[:,j:j+1]
            # lmj_bff = Rot_theta.T @ (lmj_inertial - robot_xy)

            # robot xy DH
            DH[2*i:2*i+2,0:2] = - Rot_theta.T
            # robot theta DH
            DH[2*i:2*i+2, 2:3] = DRot_theta.T @ (lmj_inertial - robot_xy)
            # lm xy DH
            DH[2*i:2*i+2, 3+2*j:3+2*j+2] = Rot_theta.T

            # print(DH[i:i+2,:])

        return DH
    
    def covariance_drive(self, drive_meas):
        """
        Compute the process noise covariance Q for the current timestep
        by mapping wheel rotation rate noise into (x, y, theta) uncertainty
        using the motion model Jacobians.
        """
        J_wheelVel2bodyVel = np.array([
            [self.wheels_scale/2, self.wheels_scale/2],
            [-self.wheels_scale/self.wheels_width, self.wheels_scale/self.wheels_width]
        ])
        
        v, w = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)
        dt = drive_meas.dt
        th = float(self.state[2])
        th2 = th + w*dt             # Next theta

        # Derivative of x,y,theta w.r.t. lin_vel, ang_vel
        J_bodyVel2Pose = np.zeros((3,2))
        
        if abs(w) < 1e-9:
            # straight-line limits (use Taylor series)
            J_bodyVel2Pose[0,0] = np.cos(th) * dt                     # ∂x/∂v
            J_bodyVel2Pose[1,0] = np.sin(th) * dt                     # ∂y/∂v
            J_bodyVel2Pose[2,0] = 0.0                                 # ∂θ/∂v
            J_bodyVel2Pose[0,1] = -0.5 * v * np.sin(th) * dt**2       # ∂x/∂ω
            J_bodyVel2Pose[1,1] =  0.5 * v * np.cos(th) * dt**2       # ∂y/∂ω
            J_bodyVel2Pose[2,1] = dt                                  # ∂θ/∂ω
        else:
            # turning (arc)
            J_bodyVel2Pose[0,0] = (np.sin(th2) - np.sin(th)) / w
            J_bodyVel2Pose[1,0] = -(np.cos(th2) - np.cos(th)) / w
            J_bodyVel2Pose[2,0] = 0.0

            # ∂x/∂ω and ∂y/∂ω (closed form)
            J_bodyVel2Pose[0,1] = v * (w*dt*np.cos(th2) - (np.sin(th2) - np.sin(th))) / (w**2)
            J_bodyVel2Pose[1,1] = v * (w*dt*np.sin(th2) + (np.cos(th2) - np.cos(th))) / (w**2)
            J_bodyVel2Pose[2,1] = dt

        # Jacobian that converts wheel velocities to robot pose
        Jac = J_bodyVel2Pose @ J_wheelVel2bodyVel

        # Map the wheel covariance to the pose covariance
        cov_wheels = np.diag((drive_meas.left_cov, drive_meas.right_cov))
        cov = Jac @ cov_wheels @ Jac.T

        # If we find it is a bit jittery we can uncomment this line: Make covariance matrix symmetric
        # cov = 0.5 * (cov + cov.T)

        # Return the pose covariance
        return cov