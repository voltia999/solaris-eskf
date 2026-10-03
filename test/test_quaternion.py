import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quaternion import (
    quat_product,
    rotation_matrix_hamilton,
    quat_conjugate,
    quat_normalize,
    quat_from_rotvec,
    rotvec_from_quat,
    jacobian_q_dtheta,
    quat_to_euler,
    jacobian_rotT_vec_q,
)

S2 = np.sqrt(2) / 2
IDENTITY = np.array([1.0, 0, 0, 0])
Q_Z90 = np.array([S2, 0, 0, S2])
Q_Y90 = np.array([S2, 0, S2, 0])
Q_X90 = np.array([S2, S2, 0, 0])
Q_111 = np.array([0.5, 0.5, 0.5, 0.5])


def random_quats(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return [quat_normalize(rng.normal(size=4)) for _ in range(n)]


# --- quat_product ---

def test_product_integers():
    q1 = np.array([1, 2, 3, 4])
    q2 = np.array([5, 6, 7, 8])
    np.testing.assert_allclose(quat_product(q1, q2), [-60, 12, 30, 24])
    np.testing.assert_allclose(quat_product(q2, q1), [-60, 20, 14, 32])


def test_product_rotations():
    np.testing.assert_allclose(quat_product(Q_Z90, Q_X90), [0.5, 0.5, 0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(quat_product(Q_X90, Q_Z90), [0.5, 0.5, -0.5, 0.5], atol=1e-12)


@pytest.mark.parametrize("q", random_quats(10))
def test_product_identity(q):
    np.testing.assert_allclose(quat_product(q, IDENTITY), q)
    np.testing.assert_allclose(quat_product(IDENTITY, q), q)


@pytest.mark.parametrize("q", random_quats(10))
def test_product_inverse(q):
    np.testing.assert_allclose(quat_product(q, quat_conjugate(q)), IDENTITY, atol=1e-12)


# --- rotation_matrix_hamilton ---

@pytest.mark.parametrize("q, R_expected", [
    (IDENTITY, np.eye(3)),
    (Q_Z90, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    (Q_Y90, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    (Q_111, [[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
])
def test_rotation_matrix_known(q, R_expected):
    np.testing.assert_allclose(rotation_matrix_hamilton(q), R_expected, atol=1e-12)


@pytest.mark.parametrize("q", random_quats())
def test_rotation_matrix_is_rotation(q):
    R = rotation_matrix_hamilton(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


@pytest.mark.parametrize("q", random_quats())
def test_rotation_matrix_vs_scipy(q):
    R_scipy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    np.testing.assert_allclose(rotation_matrix_hamilton(q), R_scipy, atol=1e-12)


def test_rotation_matrix_composition():
    q1, q2 = random_quats(2, seed=1)
    np.testing.assert_allclose(
        rotation_matrix_hamilton(quat_normalize(quat_product(q1, q2))),
        rotation_matrix_hamilton(q1) @ rotation_matrix_hamilton(q2),
        atol=1e-12,
    )


@pytest.mark.parametrize("q", random_quats(10))
def test_rotation_matrix_matches_sandwich(q):
    # R(q) v == vector part of q ⊗ [0, v] ⊗ q*
    v = np.array([0.3, -1.2, 2.0])
    rotated = quat_product(quat_product(q, np.append(0.0, v)), quat_conjugate(q))
    np.testing.assert_allclose(rotation_matrix_hamilton(q) @ v, rotated[1:], atol=1e-12)


def test_rotation_matrix_rejects_non_unit():
    with pytest.raises(ValueError):
        rotation_matrix_hamilton(np.array([1.0, 2, 3, 4]))


# --- quat_conjugate / quat_normalize ---

def test_conjugate():
    np.testing.assert_allclose(quat_conjugate(np.array([1, 2, 3, 4])), [1, -2, -3, -4])


def test_normalize():
    q = quat_normalize(np.array([1.0, 2, 3, 4]))
    assert np.linalg.norm(q) == pytest.approx(1.0)
    np.testing.assert_allclose(q, np.array([1, 2, 3, 4]) / np.sqrt(30))


# --- quat_from_rotvec / rotvec_from_quat ---

def test_from_rotvec_known():
    np.testing.assert_allclose(quat_from_rotvec(np.array([0, 0, np.pi / 2])), Q_Z90, atol=1e-12)


def test_from_rotvec_zero():
    q = quat_from_rotvec(np.zeros(3))
    assert not np.any(np.isnan(q))
    np.testing.assert_allclose(q, IDENTITY)


def test_rotvec_from_identity():
    rv = rotvec_from_quat(IDENTITY)
    assert not np.any(np.isnan(rv))
    np.testing.assert_allclose(rv, np.zeros(3))


@pytest.mark.parametrize("theta", [
    np.zeros(3),
    np.array([1e-10, -2e-10, 3e-10]),
    np.array([0.01, -0.02, 0.005]),
    np.array([0, 0, np.pi / 2]),
    np.array([0.3, -1.1, 0.7]),
    np.array([0, 0, np.pi - 1e-6]),
])
def test_rotvec_roundtrip(theta):
    q = quat_from_rotvec(theta)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    np.testing.assert_allclose(rotvec_from_quat(q), theta, atol=1e-12)


@pytest.mark.parametrize("q", random_quats(10))
def test_rotvec_double_cover(q):
    np.testing.assert_allclose(rotvec_from_quat(-q), rotvec_from_quat(q), atol=1e-12)


@pytest.mark.parametrize("q", random_quats(10))
def test_rotvec_vs_scipy(q):
    rv_scipy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()
    np.testing.assert_allclose(rotvec_from_quat(q), rv_scipy, atol=1e-12)


# --- jacobian_q_dtheta ---

@pytest.mark.parametrize("q", random_quats(10))
def test_jacobian_finite_differences(q):
    eps = 1e-7
    J_num = np.zeros((4, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        J_num[:, i] = (quat_product(q, quat_from_rotvec(d)) - q) / eps
    np.testing.assert_allclose(jacobian_q_dtheta(q), J_num, atol=1e-6)


# --- quat_to_euler ---

@pytest.mark.parametrize("q, euler_expected", [
    (IDENTITY, [0, 0, 0]),
    (Q_X90, [np.pi / 2, 0, 0]),
    (quat_from_rotvec(np.array([0, 0.5, 0])), [0, 0.5, 0]),
    (Q_Z90, [0, 0, np.pi / 2]),
])
def test_euler_known(q, euler_expected):
    np.testing.assert_allclose(quat_to_euler(q), euler_expected, atol=1e-12)


def test_euler_gimbal_lock():
    # pitch = 90°: roll and yaw are not unique, only check pitch and no NaN
    euler = quat_to_euler(Q_Y90)
    assert not np.any(np.isnan(euler))
    assert euler[1] == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("q", random_quats())
def test_euler_vs_scipy(q):
    euler_scipy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("ZYX")[::-1]
    np.testing.assert_allclose(quat_to_euler(q), euler_scipy, atol=1e-9)


# --- jacobian_rotT_vec_q ---

@pytest.mark.parametrize("q", random_quats(10))
def test_jacobian_rotT_vec_finite_differences(q):
    # q + eps is not unit, so R^T r is computed with the bilinear formula valid for any q
    def rotT_vec(q, r):
        w, x, y, z = q
        R = np.array([
            [w*w + x*x - y*y - z*z, 2 * (x*y - w*z), 2 * (x*z + w*y)],
            [2 * (x*y + w*z), w*w - x*x + y*y - z*z, 2 * (y*z - w*x)],
            [2 * (x*z - w*y), 2 * (y*z + w*x), w*w - x*x - y*y + z*z],
        ])
        return R.T @ r

    r = np.array([26.0, -0.5, 37.0])
    eps = 1e-7
    J_num = np.zeros((3, 4))
    for i in range(4):
        dq = np.zeros(4)
        dq[i] = eps
        J_num[:, i] = (rotT_vec(q + dq, r) - rotT_vec(q, r)) / eps
    np.testing.assert_allclose(jacobian_rotT_vec_q(q, r), J_num, atol=1e-4)
