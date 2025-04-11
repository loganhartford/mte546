import numpy as np

class EKF:
    def __init__(self, P, Q, R, x, dt, Vc=299792458):
        self.P = P
        self.Q = Q
        self.R = R
        self.x = x
        self.dt = dt
        self.Vc = Vc
        self.K = None

    def predict(self, u, event_str, prob):
        if event_str == "usual stuff":
            prob = 1

        # Adaptive Q
        Q = self.Q.copy()
        Q = Q * prob

        self.A = self.jacobian_A(u)       # Compute state transition Jacobian
        self.motion_model(u)              # Predict state using motion model
        self.H = self.jacobian_H()         # Measurement Jacobian (constant for h(x) = [x, y, z, phi])
        self.P = self.A @ self.P @ self.A.T + Q  # Covariance prediction

    def update(self, z, event_str, prob):
        if event_str == "usual stuff":
            prob = 1
        
        # Adaptive R
        meas_norm = np.linalg.norm(z)
        R = self.R.copy()
        R = R * meas_norm / prob  # Adjust measurement noise covariance based on measurement norm
        
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)  # Kalman gain
        self.x = self.x + self.K @ (z - self.measurement_model())                        # State update
        I = np.eye(self.A.shape[0])
        self.P = (I - self.K @ self.H) @ self.P @ (I - self.K @ self.H).T + self.K @ R @ self.K.T  # Covariance update

    def motion_model(self, u):
        dt = self.dt
        a, wx, wy, wz = u
        x, y, z, v, phi, theta, psi = self.x
        
        R_mat = self.rotation_matrix(phi, theta, psi)
        D = v * dt + 0.5 * a * dt**2
        
        # Position
        pos_update = R_mat[:, 0] * D
        x_new = x + pos_update[0]
        y_new = y + pos_update[1]
        z_new = z + pos_update[2]
        
        # Velocity
        v_new = v + a / np.sqrt(1 - (v / self.Vc)**2)
        
        # Orientation update using the provided Jacobian J:
        phi_new = phi + wx + np.sin(phi) * np.tan(theta) * wy + np.cos(phi) * np.tan(theta) * wz
        theta_new = theta + np.cos(phi) * wy - np.sin(phi) * wz
        psi_new = psi + (np.sin(phi) / np.cos(theta)) * wy + (np.cos(phi) / np.cos(theta)) * wz
        
        self.x = np.array([x_new, y_new, z_new, v_new, phi_new, theta_new, psi_new])

    def rotation_matrix(self, phi, theta, psi):
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        
        R = np.array([
            [cphi * ctheta,      cphi * stheta * spsi - sphi * cpsi,    cphi * stheta * cpsi + sphi * spsi],
            [sphi * ctheta,      sphi * stheta * spsi + cphi * cpsi,     sphi * stheta * cpsi - cphi * spsi],
            [-stheta,             ctheta * spsi,           ctheta * cpsi]
        ])
        return R

    def jacobian_A(self, u):
        dt = self.dt
        a, wx, wy, wz = u
        x, y, z, v, phi, theta, psi = self.x
        D = v * dt + 0.5 * a * dt**2
        Vc = self.Vc

        A = np.zeros((7,7))

        # Posistion
        # x
        A[0, 0] = 1
        A[0, 3] = dt * np.cos(phi) * np.cos(theta)
        A[0, 4] = -np.sin(phi) * np.cos(theta) * D
        A[0, 5] = -np.cos(phi) * np.sin(theta) * D

        # y
        A[1, 1] = 1
        A[1, 3] = dt * np.sin(phi) * np.cos(theta)
        A[1, 4] = np.cos(phi) * np.cos(theta) * D
        A[1, 5] = -np.sin(phi) * np.sin(theta) * D

        # z
        A[2, 2] = 1
        A[2, 3] = - dt * np.sin(theta)
        A[2, 4] = 0
        A[2, 5] = -np.cos(theta) * D

        # Velocity
        A[3, 3] = 1 + a * v / (Vc**2 * np.sqrt(1 - (v / Vc)**2))

        # Orientation
        # phi
        A[4, 4] = 1 + np.cos(phi) * np.tan(theta) * wy - np.sin(phi) * np.tan(theta) * wz
        A[4, 5] = (np.sin(phi) / (np.cos(theta)**2)) * wy + (np.cos(phi) / (np.cos(theta)**2)) * wz

        # theta
        A[5, 4] = -np.sin(phi) * wy - np.cos(phi) * wz
        A[5, 5] = 1

        # psi
        A[6, 4] = (np.cos(phi) / np.cos(theta)) * wy - (np.sin(phi) / np.cos(theta)) * wz
        A[6, 5] = (np.sin(phi) * np.sin(theta) / (np.cos(theta)**2)) * wy + (np.cos(phi) * np.sin(theta) / (np.cos(theta)**2)) * wz
        A[6, 6] = 1

        return A

    def jacobian_H(self):
        H = np.zeros((4, 7))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 4] = 1
        return H

    def measurement_model(self):
        x, y, z, v, phi, theta, psi = self.x
        return np.array([x, y, z, phi])
    
    def get_state(self):
        return self.x
