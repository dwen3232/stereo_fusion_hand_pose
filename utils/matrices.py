import numpy as np

def K_matrix(fx, fy, tx, ty):
    return np.array([
        [fx, 0, tx],
        [0, ty, ty],
        [0, 0, 1]
    ])

# used for UVD -> XYZ
def Q_matrix(fx, fy, tx, ty, baseline):
    return np.array([
        [1, 0, 0, -tx],
        [0, 1, 0, -ty],
        [0, 0, 0, fx],
        [0, 0, 1 / baseline, 0]
    ])

# used for XYZ -> UVD
def P_matrix(fx, fy, tx, ty, baseline):
    return np.array([
        [fx, 0, tx, 0],
        [0, fy, ty, 0],
        [0, 0, 0, fx * baseline],
        [0, 0, 1, 0]
    ])

def apply_homogeneous_transform(coords, transformation):
    # coords in shape (4, N) or (3, N)
    # returns UVD in homogeneous form
    assert transformation.shape == (4, 4), "transformation must be 4x4"
    assert coords.shape[0] == 3 or coords.shape[0] == 4, "coords must be length 3 or 4 vectors"
    if coords.shape[0] == 3:
        coords = np.append(coords, np.ones((1, coords.shape[1])), axis=0)
    return transformation.dot(coords)

def reduce_homogeneous(homo_coords):
    reduced = homo_coords / homo_coords[-1, :]
    return reduced[:-1]


