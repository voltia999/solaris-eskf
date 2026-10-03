import numpy as np

def quat_product(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    result_w = np.array([q1[0] * q2[0] - q1[1:].T @ q2[1:]],)
    result_v = [q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])]
    result = np.append(result_w, result_v)
    return result

def rotation_matrix_hamilton(q: np.ndarray) -> np.ndarray:

    if not np.isclose(np.linalg.norm(q), 1.0,atol=1e-9):
        raise ValueError("the quaternion must be unitary")
    
    R1 = [1 - 2 *  (q[2] ** 2 + q[3] ** 2), 
        2 * (q[1] * q[2] - q[0] * q[3]), 
        2 * (q[1] * q[3] + q[0] * q[2])]
    R2 = [2 * (q[1] * q[2] + q[0] * q[3]),
        1 - 2 *  (q[1] ** 2 + q[3] ** 2), 
        2 * (q[2] * q[3] - q[0] * q[1])]
    R3 = [2 * (q[1] * q[3] - q[0] * q[2]), 
        2 * (q[2] * q[3] + q[0] * q[1]),
        1 - 2 *  (q[1] ** 2 + q[2] ** 2)]
    
    return np.array([R1, R2, R3])

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_normalize(q:np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)

def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)

    if theta < 1e-8:
        return quat_normalize(np.append(1.0, rotvec / 2))

    u = rotvec / theta
    q_w = np.cos(theta / 2)
    q_v = u * np.sin(theta / 2)

    return np.append(q_w, q_v)

def rotvec_from_quat(q: np.ndarray) -> np.ndarray:

    if q[0] < 0:
        q = -q

    q_v = q[1:]
    s = np.linalg.norm(q_v)

    if s < 1e-12:
        return 2 * q_v

    theta = 2 * np.arctan2(s, q[0])
    u = q_v / s

    return u * theta

def jacobian_q_dtheta(q:np.ndarray) -> np.ndarray:
    R1 = [-q[1], -q[2], -q[3]]
    R2 = [q[0], -q[3], q[2]]
    R3 = [q[3], q[0], -q[1]]
    R4 = [-q[2], q[1], q[0]]
    return 0.5 * np.array([R1, R2, R3, R4])

def quat_to_euler(q: np.ndarray) -> np.ndarray:
    # ZYX (yaw-pitch-roll), returns [roll, pitch, yaw] in rad
    w, x, y, z = q

    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))

    return np.array([roll, pitch, yaw])
def jacobian_rotT_vec_q(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    # d(R(q)^T r)/dq, 3x4 (eq. 56; eq. 63 is the case r = [0, 0, g])
    w, x, y, z = q
    rx, ry, rz = r
    R1 = [rx * w + ry * z - rz * y, rx * x + ry * y + rz * z, -rx * y + ry * x - rz * w, -rx * z + ry * w + rz * x]
    R2 = [-rx * z + ry * w + rz * x, rx * y - ry * x + rz * w, rx * x + ry * y + rz * z, -rx * w - ry * z + rz * y]
    R3 = [rx * y - ry * x + rz * w, rx * z - ry * w - rz * x, rx * w + ry * z - rz * y, rx * x + ry * y + rz * z]
    return 2 * np.array([R1, R2, R3])
