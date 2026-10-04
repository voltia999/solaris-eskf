import numpy as np
import pytest
from pytest import approx
from eskf import ESKF
from linalg_utils import skew
from quaternion import (quat_to_euler, quat_product, quat_conjugate, quat_from_rotvec, rotvec_from_quat,
                        rotation_matrix_hamilton, jacobian_rotT_vec_q)

A_REST = np.array([0, 0, -9.81])


def make_eskf(sigma_a=0.0, sigma_w=0.0, sigma_aw=0.0, sigma_ww=0.0):
    return ESKF(sigma_a=sigma_a, sigma_w=sigma_w, sigma_aw=sigma_aw, sigma_ww=sigma_ww)


# nominal propagation 

def test_rest():
    eskf = make_eskf()
    dt = 0.01
    T = 10

    for _ in range(round(T/dt)):
        eskf._predict(a_meas=A_REST, w_meas=np.zeros(3), dt=dt)

    assert eskf.p == approx(np.zeros(3))
    assert eskf.v == approx(np.zeros(3))
    assert eskf.q == approx((1.0, 0.0, 0.0, 0.0))

def test_rotation():
    eskf = make_eskf()
    w_meas = np.array([0, 0, 0.5])
    dt = 0.01
    T = 1

    for _ in range(round(T/dt)):
        eskf._predict(a_meas=A_REST, w_meas=w_meas, dt=dt)

    assert quat_to_euler(eskf.q) == approx((0.0, 0.0, 0.5))

def test_quat_norm():
    eskf = make_eskf()
    w_meas = np.array([0.3, -0.2, 0.5])
    dt = 0.01
    T = 100

    for _ in range(round(T/dt)):
        eskf._predict(a_meas=A_REST, w_meas=w_meas, dt=dt)

    assert np.linalg.norm(eskf.q) == approx(1.0)


#covariance propagation

def run_noisy(steps=500, dt=0.01):
    eskf = make_eskf(sigma_a=0.05, sigma_w=0.01, sigma_aw=1e-3, sigma_ww=1e-4)
    eskf.P = np.eye(15) * 1e-4
    rng = np.random.default_rng(0)
    traces = [np.trace(eskf.P)]

    for _ in range(steps):
        eskf._predict(a_meas=A_REST + rng.normal(size=3), w_meas=rng.normal(size=3) * 0.3, dt=dt)
        traces.append(np.trace(eskf.P))

    return eskf, np.array(traces)

def test_covariance_symmetric():
    eskf, _ = run_noisy()
    np.testing.assert_allclose(eskf.P, eskf.P.T, atol=1e-12)

def test_covariance_positive_semidefinite():
    eskf, _ = run_noisy()
    assert np.all(np.linalg.eigvalsh(eskf.P) >= -1e-12)

def test_covariance_grows():
    _, traces = run_noisy()
    assert np.all(np.diff(traces) > 0)

def test_covariance_gyro_noise_only():
    # at rest, P0 = 0, only gyro white noise: var(dtheta) = N * sigma_w^2 * dt^2
    sigma_w = 0.01
    dt = 0.01
    N = 1000
    eskf = make_eskf(sigma_w=sigma_w)
    eskf.P = np.zeros((15, 15))

    for _ in range(N):
        eskf._predict(a_meas=A_REST, w_meas=np.zeros(3), dt=dt)

    np.testing.assert_allclose(np.diag(eskf.P)[6:9], N * sigma_w ** 2 * dt ** 2)


#transition matrix vs finite differences

def set_state(eskf, p, v, q, b_a, b_g):
    eskf.p, eskf.v, eskf.q, eskf.b_a, eskf.b_g = p.copy(), v.copy(), q.copy(), b_a.copy(), b_g.copy()

def inject(p, v, q, b_a, b_g, dx):
    return (p + dx[0:3], v + dx[3:6], quat_product(q, quat_from_rotvec(dx[6:9])),
            b_a + dx[9:12], b_g + dx[12:15])

def error_between(nominal, perturbed):
    return np.concatenate([
        perturbed.p - nominal.p,
        perturbed.v - nominal.v,
        rotvec_from_quat(quat_product(quat_conjugate(nominal.q), perturbed.q)),
        perturbed.b_a - nominal.b_a,
        perturbed.b_g - nominal.b_g,
    ])

def test_transition_matrix_finite_differences():
    rng = np.random.default_rng(1)
    dt = 0.01
    state = (rng.normal(size=3), rng.normal(size=3), quat_from_rotvec(rng.normal(size=3)),
             rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.01)
    a_meas = A_REST + rng.normal(size=3)
    w_meas = rng.normal(size=3)

    nominal = make_eskf()
    set_state(nominal, *state)
    nominal.R = rotation_matrix_hamilton(nominal.q)
    Phi = nominal._transition_matrix(a_meas - nominal.b_a, (w_meas - nominal.b_g) * dt, dt)
    nominal._predict(a_meas, w_meas, dt)

    eps = 1e-6
    for i in range(15):
        dx = np.zeros(15)
        dx[i] = eps
        perturbed = make_eskf()
        set_state(perturbed, *inject(*state, dx))
        perturbed._predict(a_meas, w_meas, dt)

        np.testing.assert_allclose(error_between(nominal, perturbed), Phi @ dx, atol=1e-10,
                                   err_msg=f"column {i} of Phi")


# magnetometer update

MAG_REF = np.array([26.0, -0.5, 37.0])

def test_mag_observation_jacobian_finite_differences():
    rng = np.random.default_rng(2)
    eskf = make_eskf()
    eskf.mag_ref = MAG_REF
    eskf.q = quat_from_rotvec(rng.normal(size=3))

    H_x = np.zeros((3, 16))
    H_x[:, 6:10] = jacobian_rotT_vec_q(eskf.q, MAG_REF)
    H = eskf._observation_jacobian(H_x)

    h = lambda q: rotation_matrix_hamilton(q).T @ MAG_REF
    eps = 1e-7
    for i in range(3):
        dtheta = np.zeros(3)
        dtheta[i] = eps
        dh = (h(quat_product(eskf.q, quat_from_rotvec(dtheta))) - h(eskf.q)) / eps
        np.testing.assert_allclose(H[:, 6 + i], dh, atol=1e-5)

def test_mag_corrects_yaw():
    # truth: at rest, identity orientation; estimate starts with 0.3 rad of yaw error
    # roll/pitch are assumed known (small P): one field vector only fixes 2 of the 3 angles
    eskf = make_eskf(sigma_w=1e-3, sigma_ww=1e-5)
    eskf.mag_ref = MAG_REF
    eskf.q = quat_from_rotvec(np.array([0, 0, 0.3]))
    eskf.P = np.diag([1e-2] * 6 + [1e-6, 1e-6, 0.5 ** 2] + [1e-4] * 3 + [1e-6] * 3)

    for k in range(500):
        eskf._predict(a_meas=A_REST, w_meas=np.zeros(3), dt=0.01)
        if k % 10 == 0:
            eskf.update_mag(MAG_REF, sigma_mag=0.5)

    assert quat_to_euler(eskf.q)[2] == approx(0.0, abs=1e-2)

def test_mag_requires_reference():
    eskf = make_eskf()
    with pytest.raises(ValueError):
        eskf.update_mag(MAG_REF, sigma_mag=0.5)


# barometer update

def test_baro_corrects_only_z():
    eskf = make_eskf()
    eskf.p = np.array([5.0, -3.0, 10.0])
    eskf.P = np.eye(15) * 1e-2
    eskf.P[2, 2] = 100.0

    eskf.update_baro(z_meas=0.0, sigma_baro=0.5)

    assert eskf.p[0:2] == approx([5.0, -3.0])
    assert abs(eskf.p[2]) < 0.5
    assert eskf.P[2, 2] < 0.5 ** 2

def test_baro_accepts_array_measurement():
    eskf = make_eskf()
    eskf.update_baro(z_meas=np.array([1.0]), sigma_baro=0.5)
    assert eskf.p.shape == (3,)


# gps update

def test_gps_linear_kf():
    eskf = make_eskf()
    eskf.p = np.random.normal(size=3)
    eskf.v = np.random.normal(size=3)
    A = np.random.normal(size=(15, 15))
    eskf.P = A @ A.T + np.eye(15)

    p_meas = eskf.p + np.random.normal(size=3)
    v_meas = eskf.v + np.random.normal(size=3)
    sigma_p, sigma_v = 0.7, 0.2

    x0 = np.concatenate([eskf.p, eskf.v])
    P0 = eskf.P.copy()
    y = np.concatenate([p_meas, v_meas])

    H = np.zeros((6,15))
    H[:, 0:6] = np.eye(6)
    V = np.diag([sigma_p**2] * 3 + [sigma_v**2] * 3)

    K = P0 @ H.T @ np.linalg.inv(H @ P0 @ H.T + V)
    dx = K @ (y - x0)
    I_KH = np.eye(15) - K @ H
    P_kf = I_KH @ P0 @ I_KH.T + K @ V @ K.T

    eskf.update_gps(p_meas, v_meas, sigma_p, sigma_v)

    np.testing.assert_allclose(eskf.p, x0[0:3] + dx[0:3], atol=1e-12)
    np.testing.assert_allclose(eskf.v, x0[3:6] + dx[3:6], atol=1e-12)

    G = np.eye(15)
    G[6:9, 6:9] = np.eye(3) - skew(0.5 * dx[6:9])
    np.testing.assert_allclose(eskf.P, G @ P_kf @ G.T, atol=1e-12)

def test_gps_sigma_v_large_ignores_velocity():
    eskf = make_eskf()
    eskf.P = np.eye(15)
    eskf.update_gps(p_meas=np.full(3, 1.0), v_meas=np.full(3, 1.0),
                     sigma_p=0.01, sigma_v=1e4)

    assert eskf.p == approx(np.full(3, 1.0), abs=1e-3)
    assert eskf.v == approx(np.zeros(3), abs=1e-6)

def test_gps_sigma_p_large_ignores_position():
    eskf = make_eskf()
    eskf.P = np.eye(15)
    eskf.update_gps(p_meas=np.full(3, 1.0), v_meas=np.full(3, 1.0), sigma_p=1e4, sigma_v=0.01)

    assert eskf.p == approx(np.zeros(3), abs=1e-6)
    assert eskf.v == approx(np.full(3, 1.0), abs=1e-3)

def test_gps_scalar_and_per_axis_sigma_match():
      rng = np.random.default_rng(3)
      p_meas, v_meas = rng.normal(size=3), rng.normal(size=3)
      a, b = make_eskf(), make_eskf()

      a.update_gps(p_meas, v_meas, sigma_p=0.5, sigma_v=0.1)
      b.update_gps(p_meas, v_meas, sigma_p=np.full(3, 0.5), sigma_v=np.full(3, 0.1))

      assert a.p == approx(b.p)
      assert a.v == approx(b.v)
      np.testing.assert_allclose(a.P, b.P)

def test_gps_velocity_corrects_tilt():
    eskf = make_eskf(sigma_a=1e-2, sigma_w=1e-3)
    eskf.q = quat_from_rotvec(np.array([0.1, 0.0, 0.0]))
    eskf.P = np.diag([1e-4] * 6 + [0.2 ** 2] * 3 + [1e-6] * 6)

    for k in range(1000):
          eskf._predict(a_meas=A_REST, w_meas=np.zeros(3), dt=0.01)
          if k % 10 == 0:
              eskf.update_gps(np.zeros(3), np.zeros(3), sigma_p=0.5, sigma_v=0.05)

    roll, pitch, _ = quat_to_euler(eskf.q)
    assert roll == approx(0.0, abs=1e-2)
    assert pitch == approx(0.0, abs=1e-2)

def test_gps_at_rest_does_not_observe_yaw():
    eskf = make_eskf(sigma_a=1e-2, sigma_w=1e-3)
    eskf.q = quat_from_rotvec(np.array([0.0, 0.0, 0.3]))
    eskf.P = np.diag([1e-4] * 6 + [0.2 ** 2] * 3 + [1e-6] * 6)
    P_yaw0 = eskf.P[8, 8]

    for k in range(1000):
        eskf._predict(a_meas=A_REST, w_meas=np.zeros(3), dt=0.01)
        if k % 10 == 0:
            eskf.update_gps(np.zeros(3), np.zeros(3), sigma_p=0.5, sigma_v=0.05)

    assert quat_to_euler(eskf.q)[2] == approx(0.3, abs=1e-3)
    assert eskf.P[8, 8] >= P_yaw0


# accel update

def test_accel_finite_differences():
    rng = np.random.default_rng(4)
    eskf = make_eskf()
    state = (rng.normal(size=3), rng.normal(size=3), quat_from_rotvec(rng.normal(size=3)),
               rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.01)
    set_state(eskf, *state)

    H_x = np.zeros((3, 16))
    H_x[:, 6:10] = jacobian_rotT_vec_q(eskf.q, eskf.g)
    H_x[:, 10:13] = np.eye(3)
    H = eskf._observation_jacobian(H_x)

    h = lambda p, v, q, b_a, b_g: rotation_matrix_hamilton(q).T @ eskf.g + b_a
    eps = 1e-7
    for i in range(15):
        dx = np.zeros(15)
        dx[i] = eps
        dh = (h(*inject(*state, dx)) - h(*state)) / eps
        np.testing.assert_allclose(H[:, i], dh, atol=1e-5, err_msg=f"column {i} of H")

def test_accel_corrects_pitch_roll():
    eskf = make_eskf()
    eskf.q = quat_from_rotvec(np.array([0.1, 0.0, 0.0]))
    eskf.P = np.diag([1e-4] * 6 + [0.2 ** 2] * 3 + [1e-6] * 6)

    for i in range(1000):
        eskf._predict(a_meas=A_REST, w_meas=np.zeros(3), dt=0.01)
        if i % 10 == 0:
            eskf.update_accel(a_meas=A_REST, sigma_a=0.05)

    roll, pitch, _ = quat_to_euler(eskf.q)
    assert roll == approx(0.0, abs=1e-2)
    assert pitch == approx(0.0, abs=1e-2)

def test_accel_corrects_pitch_roll():
    eskf = make_eskf()
    eskf.q = quat_from_rotvec(np.array([0.0, 0.0, 0.3]))
    eskf.P = np.diag([1e-4] * 6 + [0.2 ** 2] * 3 + [1e-6] * 6)
    P_yaw0 = eskf.P[8, 8]

    for i in range(1000):
        eskf._predict(a_meas=A_REST, w_meas=np.zeros(3), dt=0.01)
        if i % 10 == 0:
            eskf.update_accel(a_meas=A_REST, sigma_a=0.05)

    assert quat_to_euler(eskf.q)[2] == approx(0.3, abs=1e-3)
    assert eskf.P[8, 8] >= P_yaw0

def test_accel_tilt_and_horizontal_bias_are_indistinguishable():
      eskf = make_eskf()
      H_x = np.zeros((3, 16))
      H_x[:, 6:10] = jacobian_rotT_vec_q(eskf.q, eskf.g)
      H_x[:, 10:13] = np.eye(3)
      H = eskf._observation_jacobian(H_x)

      roll_col, b_ay_col = H[:, 6], H[:, 10]
      assert np.linalg.matrix_rank(np.column_stack([roll_col, b_ay_col])) == 1

