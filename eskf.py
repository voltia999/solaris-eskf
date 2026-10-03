import numpy as np
from quaternion import (rotation_matrix_hamilton, quat_product, quat_from_rotvec, quat_normalize,
                        jacobian_q_dtheta, jacobian_rotT_vec_q)
from linalg_utils import skew
from sensors import Sensor

class ESKF():

    def __init__(self, sigma_a, sigma_w, sigma_aw, sigma_ww, mag_ref=None):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0, 0, 0])
        self.b_a = np.zeros(3)
        self.b_g = np.zeros(3)
        self.g = np.array([0, 0, -9.81])
        # Earth magnetic field in NED, from WMM/IGRF at the launch site
        self.mag_ref = None if mag_ref is None else np.asarray(mag_ref, dtype=float)

        self.P = np.eye(15) * 1e3
        
        self.sigma_a = sigma_a
        self.sigma_w = sigma_w
        self.sigma_aw = sigma_aw #accel bias random walk
        self.sigma_ww = sigma_ww #gyro bias random walk

        
        self.F_i = np.zeros((15, 12))
        self.F_i[3:15, 0:12] = np.eye(12)


    def _predict(self, a_meas, w_meas, dt):
        self.R = rotation_matrix_hamilton(self.q)
        a = a_meas - self.b_a
        w = w_meas - self.b_g
        theta = w * dt

        Phi = self._transition_matrix(a, theta, dt)
        Q_i = self._process_noise(dt)

        self.p = self.p + self.v * dt
        self.v = self.v + (self.R @ a - self.g) * dt
        self.q = quat_normalize(quat_product(self.q, quat_from_rotvec(theta)))

        self.P = Phi @ self.P @ Phi.T + self.F_i @ Q_i @ self.F_i.T

    def update_gps(self, p_meas, sigma_gps):
        # h(x) = p
        h_x = self.p

        H_x = np.zeros((3, 16))
        H_x[:, 0:3] = np.eye(3)
        H = self._observation_jacobian(H_x)

        V = np.diag(np.broadcast_to(sigma_gps, 3) ** 2)

        self._update(p_meas, h_x, H, V)

    def update_mag(self, m_meas, sigma_mag):
        # h(x) = R(q)^T r: Earth field (NED) seen in body axes
        if self.mag_ref is None:
            raise ValueError("mag_ref (Earth magnetic field in NED) is not set")

        h_x = rotation_matrix_hamilton(self.q).T @ self.mag_ref

        H_x = np.zeros((3, 16))
        H_x[:, 6:10] = jacobian_rotT_vec_q(self.q, self.mag_ref)
        H = self._observation_jacobian(H_x)

        V = np.diag(np.broadcast_to(sigma_mag, 3) ** 2)

        self._update(m_meas, h_x, H, V)

    def update_accel(self, a_meas, sigma_a):

        h_x = rotation_matrix_hamilton(self.q).T @ self.g + self.b_a

        H_x = np.zeros((3,16))
        H_x[:, 6:10] = jacobian_rotT_vec_q(self.q, self.g)
        H_x[:, 10:13] = np.eye(3)
        H = self._observation_jacobian(H_x)

        V = np.diag(np.broadcast_to(sigma_a, 3) ** 2)
        self._update(a_meas, h_x, H, V)

    def update_baro(self, z_meas, sigma_baro):

        h_x = self.p[2:3]

        H_x = np.zeros((1, 16))
        H_x[0, 2] = 1.0
        H = self._observation_jacobian(H_x)

        V = np.array([[sigma_baro ** 2]])

        self._update(np.atleast_1d(z_meas), h_x, H, V)

    def run(self, a_meas, w_meas, dt, sensors: Sensor):
        self._predict(a_meas=a_meas, w_meas=w_meas, dt=dt)

        for s in sensors:
            if s.has_new_data():
                getattr(self, f"update_{s.kind}")(s.data, s.sigma)
                s.last_ts = s.ts



    def _update(self, y, h_x, H, V):
        # gain and innovation 
        S = H @ self.P @ H.T + V
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ (y - h_x)

        # covariance, Joseph form 
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ V @ K.T

        # injection 
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = quat_normalize(quat_product(self.q, quat_from_rotvec(dx[6:9])))
        self.b_a = self.b_a + dx[9:12]
        self.b_g = self.b_g + dx[12:15]

        # reset
        G = np.eye(15)
        G[6:9, 6:9] = np.eye(3) - skew(0.5 * dx[6:9])
        self.P = G @ self.P @ G.T

    def _transition_matrix(self, a, theta, dt):
        I3 = np.eye(3)
        Phi = np.eye(15)

        Phi[0:3, 3:6] = I3 * dt
        Phi[3:6, 6:9] = -self.R @ skew(a) * dt
        Phi[3:6, 9:12] = -self.R * dt
        Phi[6:9, 6:9] = rotation_matrix_hamilton(quat_from_rotvec(theta)).T
        Phi[6:9, 12:15] = -I3 * dt

        return Phi

    def _process_noise(self, dt):
        I3 = np.eye(3)
        Q_i = np.zeros((12, 12))

        Q_i[0:3, 0:3] = self.sigma_a ** 2 * dt ** 2 * I3
        Q_i[3:6, 3:6] = self.sigma_w ** 2 * dt ** 2 * I3
        Q_i[6:9, 6:9] = self.sigma_aw ** 2 * dt * I3
        Q_i[9:12, 9:12] = self.sigma_ww ** 2 * dt * I3

        return Q_i

    def _error_jacobian(self):
        X_dx = np.zeros((16, 15))

        X_dx[0:6, 0:6] = np.eye(6)
        X_dx[6:10, 6:9] = jacobian_q_dtheta(self.q)
        X_dx[10:16, 9:15] = np.eye(6)

        return X_dx

    def _observation_jacobian(self, H_x):
        return H_x @ self._error_jacobian()





