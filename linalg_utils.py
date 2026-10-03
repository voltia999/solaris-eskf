import numpy as np

def skew(x:np.ndarray) -> np.ndarray:
    R1 = [0, -x[2], x[1]]
    R2 = [x[2], 0, -x[0]]
    R3 = [-x[1], x[0], 0]
    return np.array([R1, R2, R3])