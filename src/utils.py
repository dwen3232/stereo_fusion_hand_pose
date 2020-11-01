import numpy

def K_matrix(fx, fy, tx, ty):
    return np.array([
        [fx, 0, tx],
        [0, ty, ty],
        [0, 0, 1]
    ])

def Q_matrix(fx, fy, tx, ty, baseline):
    return np.array([
        [1, 0, 0, -tx],
        [0, 1, 0, -ty],
        [0, 0, 0, fx],
        [0, 0, 1 / baseline, 0]
    ])

def P_matrix(fx, fy, tx, ty, baseline):
    return np.array([
        [fx, 0, tx, 0],
        [0, fy, ty, 0],
        [0, 0, 0, fx * baseline],
        [0, 0, 1, 0]
    ])