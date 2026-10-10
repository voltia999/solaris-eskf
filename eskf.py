import numpy as np
from quaternion import (rotation_matrix_hamilton, quat_product, quat_from_rotvec, quat_normalize,
                        jacobian_q_dtheta, jacobian_rotT_vec_q)
from linalg_utils import skew
from sensors import Sensor

class ESKF():
    """Error-State Kalman Filter for inertial navigation (docs/ESKF.pdf).

    Conventions: NED navigation frame, Hamilton quaternions, R(q) rotates body to
    NED and the attitude error is local: q_t = q ⊗ δq{δθ}.

    Nominal state x = [p, v, q, b_a, b_g] (16) and error state
    δx = [δp, δv, δθ, δb_a, δb_g] (15), in that order in every matrix.

    Attributes:
        p: Position in NED (m).
        v: Velocity in NED (m/s).
        q: Attitude quaternion [w, x, y, z], body to NED.
        b_a: Accelerometer bias (m/s²).
        b_g: Gyroscope bias (rad/s).
        g: Gravity reaction in NED, [0, 0, -9.81]: what the accelerometer reads at rest.
        mag_ref: Earth magnetic field in NED, or None if the magnetometer is not used.
        accel_threshold: Max | ‖a_m‖ - g | to use the accelerometer as a gravity
            measurement (sec. 4.5.2).
        P: Error-state covariance (15x15).
        sigma_a, sigma_w: Accelerometer (m/s²) and gyroscope (rad/s) white noise std.
        sigma_aw, sigma_ww: Accelerometer and gyroscope bias random walk std
            (K from the Allan variance, appendix C).
        F_i: Jacobian of the error dynamics w.r.t. the process noise (eq. 37).
    """

    def __init__(self, sigma_a, sigma_w, sigma_aw, sigma_ww, mag_ref=None, accel_threshold=0.5):
        """
        Args:
            sigma_a: Accelerometer white noise std (m/s²), N·√BW.
            sigma_w: Gyroscope white noise std (rad/s), N·√BW.
            sigma_aw: Accelerometer bias random walk std (m/s²/√s), K.
            sigma_ww: Gyroscope bias random walk std (rad/s/√s), K.
            mag_ref: Earth magnetic field in NED at the launch site (WMM/IGRF), in the
                same units as the magnetometer measurement. Required by update_mag.
            accel_threshold: Max | ‖a_m‖ - g | (m/s²) to accept an accelerometer update.
        """
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0, 0, 0])
        self.b_a = np.zeros(3)
        self.b_g = np.zeros(3)
        self.g = np.array([0, 0, -9.81])
        # Earth magnetic field in NED, from WMM/IGRF at the launch site
        self.mag_ref = None if mag_ref is None else np.asarray(mag_ref, dtype=float)
        # max | ||a_m|| - g | (m/s^2) to use the accel as a gravity measurement (sec. 4.5.2)
        self.accel_threshold = accel_threshold

        self.P = np.eye(15) * 1e3

        self.sigma_a = sigma_a
        self.sigma_w = sigma_w
        self.sigma_aw = sigma_aw #accel bias random walk
        self.sigma_ww = sigma_ww #gyro bias random walk


        self.F_i = np.zeros((15, 12))
        self.F_i[3:15, 0:12] = np.eye(12)


    def _predict(self, a_meas, w_meas, dt):
        """Propagate the nominal state and the error covariance one IMU sample.

        Nominal state with forward Euler and zero-order hold (eq. 21-27); covariance
        with P = Φ P Φᵀ + F_i Q_i F_iᵀ (eq. 39). Φ is evaluated at the state before
        the step.

        Args:
            a_meas: Accelerometer measurement, specific force in body axes (m/s²).
            w_meas: Gyroscope measurement in body axes (rad/s).
            dt: Time since the previous IMU sample (s).
        """
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

    def update_gps(self, p_meas, v_meas, sigma_p, sigma_v):
        """Correct the state with a GPS position and velocity (sec. 4.3).

        The measurement must already be in the NED frame of the filter (LLA to NED
        conversion in appendix A).

        Args:
            p_meas: Position in NED (m), shape (3,).
            v_meas: Velocity in NED (m/s), shape (3,).
            sigma_p: Position noise std (m), scalar or per axis (3,).
            sigma_v: Velocity noise std (m/s), scalar or per axis (3,).
        """
        self._fuse([self._meas_gps(np.concatenate([p_meas, v_meas]), (sigma_p, sigma_v))])

    def update_mag(self, m_meas, sigma_mag):
        """Correct the attitude with a magnetometer measurement (sec. 4.1).

        Model h(x) = R(q)ᵀ mag_ref: the Earth field seen in body axes (the PDF writes
        R(q) in eq. 53; the transpose is the NED-to-body rotation). The measurement
        must be calibrated (hard/soft iron removed) and in the units of mag_ref.

        Applies the measurement on its own. In run it is only used together with an
        accepted accelerometer measurement (sec. 4.5.3).

        Args:
            m_meas: Magnetic field in body axes, shape (3,).
            sigma_mag: Noise std, scalar or per axis (3,).

        Raises:
            ValueError: If mag_ref is not set.
        """
        self._fuse([self._meas_mag(m_meas, sigma_mag)])

    def update_accel(self, a_meas, sigma_a):
        """Correct roll, pitch and b_a using the accelerometer as a gravity sensor (sec. 4.2).

        Model h(x) = R(q)ᵀ g + b_a, only valid without linear acceleration. The
        measurement is skipped when | ‖a_meas‖ - g | >= accel_threshold (sec. 4.5.2):
        boost, coast (free fall reads ~0) or strong horizontal acceleration. Yaw is
        not observable from gravity.

        Args:
            a_meas: Accelerometer measurement in body axes (m/s²), shape (3,).
            sigma_a: Noise std (m/s²), scalar or per axis (3,). Usually larger than
                the IMU sigma_a, to absorb small unmodelled accelerations.

        Returns:
            True if the measurement was used, False if it was rejected.
        """
        meas = self._meas_accel(a_meas, sigma_a)
        if meas is None:
            return False
        self._fuse([meas])
        return True

    def update_baro(self, z_meas, sigma_baro):
        """Correct the vertical position with a barometer measurement (sec. 4.4).

        Args:
            z_meas: NED z (m, positive down), already converted from pressure
                (appendix B): z = -(altitude - altitude_ref). Scalar or shape (1,).
            sigma_baro: Noise std (m).
        """
        self._fuse([self._meas_baro(z_meas, sigma_baro)])

    def run(self, a_meas, w_meas, dt, sensors: Sensor):
        """Run one filter step: predict with the IMU, then fuse the new measurements.

        Sec. 4.5.3: the sensors with a measurement newer than the last one used are
        stacked into a single update (eq. 75-77, V block diagonal), and the
        magnetometer is only used if the accelerometer is used in the same step (it
        has new data and passes the threshold of sec. 4.5.2). Every new measurement is
        marked as used, also the rejected ones, so it is not applied in a later step.

        Args:
            a_meas: Accelerometer measurement in body axes (m/s²).
            w_meas: Gyroscope measurement in body axes (rad/s).
            dt: Time since the previous IMU sample (s).
            sensors: Iterable of Sensor. Each kind must have a _meas_<kind> method
                taking (data, sigma).

        Returns:
            Kinds of the measurements fused in this step, in the given order.
        """
        self._predict(a_meas=a_meas, w_meas=w_meas, dt=dt)

        meas = {}
        for s in sensors:
            if s.has_new_data():
                m = getattr(self, f"_meas_{s.kind}")(s.data, s.sigma)
                if m is not None:
                    meas[s.kind] = m
                s.last_ts = s.ts

        # the magnetometer alone does not observe roll and pitch (sec. 4.5.3)
        if "accel" not in meas:
            meas.pop("mag", None)

        if meas:
            self._fuse(list(meas.values()))
        return list(meas)

    def _meas_gps(self, data, sigma):
        """GPS measurement terms, h(x) = [p; v] (eq. 67-69).

        Args:
            data: [p, v] in NED (m, m/s), shape (6,).
            sigma: (sigma_p, sigma_v), each scalar or per axis (3,).

        Returns:
            (y, h(x), H_x, V) with H_x w.r.t. the nominal state, shape (6, 16).
        """
        sigma_p, sigma_v = sigma
        h_x = np.concatenate([self.p, self.v])

        H_x = np.zeros((6, 16))
        H_x[:3, 0:3] = np.eye(3)
        H_x[3:, 3:6] = np.eye(3)

        V = np.diag(np.concatenate([np.broadcast_to(sigma_p, 3), np.broadcast_to(sigma_v, 3)]) ** 2)
        return np.asarray(data, dtype=float), h_x, H_x, V

    def _meas_mag(self, m_meas, sigma_mag):
        """Magnetometer measurement terms, h(x) = R(q)ᵀ mag_ref (sec. 4.1).

        Returns:
            (y, h(x), H_x, V), H_x shape (3, 16).

        Raises:
            ValueError: If mag_ref is not set.
        """
        # h(x) = R(q)^T r: Earth field (NED) seen in body axes
        if self.mag_ref is None:
            raise ValueError("mag_ref (Earth magnetic field in NED) is not set")

        h_x = rotation_matrix_hamilton(self.q).T @ self.mag_ref

        H_x = np.zeros((3, 16))
        H_x[:, 6:10] = jacobian_rotT_vec_q(self.q, self.mag_ref)

        V = np.diag(np.broadcast_to(sigma_mag, 3) ** 2)
        return m_meas, h_x, H_x, V

    def _meas_accel(self, a_meas, sigma_a):
        """Accelerometer measurement terms, h(x) = R(q)ᵀ g + b_a (sec. 4.2).

        Returns:
            (y, h(x), H_x, V), H_x shape (3, 16), or None if the measurement is
            rejected by the threshold (sec. 4.5.2).
        """
        # only valid without linear acceleration: skipped when its norm is far from g
        if abs(np.linalg.norm(a_meas) - np.linalg.norm(self.g)) >= self.accel_threshold:
            return None

        h_x = rotation_matrix_hamilton(self.q).T @ self.g + self.b_a

        H_x = np.zeros((3, 16))
        H_x[:, 6:10] = jacobian_rotT_vec_q(self.q, self.g)
        H_x[:, 10:13] = np.eye(3)

        V = np.diag(np.broadcast_to(sigma_a, 3) ** 2)
        return a_meas, h_x, H_x, V

    def _meas_baro(self, z_meas, sigma_baro):
        """Barometer measurement terms, h(x) = p_z (eq. 71-73).

        Returns:
            (y, h(x), H_x, V), H_x shape (1, 16).
        """
        h_x = self.p[2:3]

        H_x = np.zeros((1, 16))
        H_x[0, 2] = 1.0

        V = np.array([[sigma_baro ** 2]])
        return np.atleast_1d(z_meas), h_x, H_x, V

    def _fuse(self, measurements):
        """Stack several measurements into a single update (eq. 75-77).

        All the terms are evaluated at the same nominal state, before any correction.
        The noise of different sensors is independent, so V is block diagonal.

        Args:
            measurements: List of (y, h(x), H_x, V) from the _meas_<kind> methods.
        """
        ys, hs, Hs, Vs = zip(*measurements)
        m = sum(len(y) for y in ys)
        V = np.zeros((m, m))
        i = 0
        for Vk in Vs:
            n = Vk.shape[0]
            V[i:i + n, i:i + n] = Vk
            i += n

        H = self._observation_jacobian(np.vstack(Hs))
        self._update(np.concatenate(ys), np.concatenate(hs), H, V)

    def _update(self, y, h_x, H, V):
        """Kalman correction, error injection and reset (sec. 3.6).

        Gain and covariance in Joseph form (eq. 41-43), injection into the nominal
        state (eq. 48) and covariance reset with G = I - [½δθ]× (eq. 49-51).

        Args:
            y: Measurement, shape (m,).
            h_x: Predicted measurement h(x) at the nominal state, shape (m,).
            H: Observation Jacobian w.r.t. the error state, shape (m, 15).
            V: Measurement noise covariance, shape (m, m).
        """
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
        """Error-state transition matrix Φ for one step (sec. 3.5.2).

        Second-order (Δt²) expansion of eq. 32 without the gravity error state, which
        is not estimated (15 states instead of 18). Uses self.R, which _predict sets
        before calling it.

        Args:
            a: Bias-corrected specific force in body axes (m/s²).
            theta: Bias-corrected rotation vector of the step, (w_m - b_g)·dt (rad).
            dt: Step (s).

        Returns:
            Φ, shape (15, 15).
        """
        I3 = np.eye(3)
        Phi = np.eye(15)
        R_a = self.R @ skew(a)

        Phi[0:3, 3:6] = I3 * dt
        Phi[0:3, 6:9] = -0.5 * R_a * dt ** 2
        Phi[0:3, 9:12] = -0.5 * self.R * dt ** 2
        Phi[3:6, 6:9] = -R_a * dt
        Phi[3:6, 9:12] = -self.R * dt
        Phi[3:6, 12:15] = 0.5 * R_a * dt ** 2
        Phi[6:9, 6:9] = rotation_matrix_hamilton(quat_from_rotvec(theta)).T
        Phi[6:9, 12:15] = -I3 * dt

        return Phi

    def _process_noise(self, dt):
        """Process noise covariance Q_i for one step (eq. 36).

        White noise enters as impulses of variance σ²Δt² (velocity and angle);
        the bias random walks as σ²Δt.

        Args:
            dt: Step (s).

        Returns:
            Q_i, shape (12, 12), ordered [accel, gyro, accel bias, gyro bias].
        """
        I3 = np.eye(3)
        Q_i = np.zeros((12, 12))

        Q_i[0:3, 0:3] = self.sigma_a ** 2 * dt ** 2 * I3
        Q_i[3:6, 3:6] = self.sigma_w ** 2 * dt ** 2 * I3
        Q_i[6:9, 6:9] = self.sigma_aw ** 2 * dt * I3
        Q_i[9:12, 9:12] = self.sigma_ww ** 2 * dt * I3

        return Q_i

    def _error_jacobian(self):
        """Jacobian of the true state w.r.t. the error state, X_δx (eq. 45-47).

        Identity except for the attitude block Q_δθ = ∂(q ⊗ δq)/∂δθ.

        Returns:
            X_δx, shape (16, 15).
        """
        X_dx = np.zeros((16, 15))

        X_dx[0:6, 0:6] = np.eye(6)
        X_dx[6:10, 6:9] = jacobian_q_dtheta(self.q)
        X_dx[10:16, 9:15] = np.eye(6)

        return X_dx

    def _observation_jacobian(self, H_x):
        """Chain rule H = H_x · X_δx (eq. 44).

        Args:
            H_x: Jacobian of h w.r.t. the nominal state, shape (m, 16).

        Returns:
            H w.r.t. the error state, shape (m, 15).
        """
        return H_x @ self._error_jacobian()





