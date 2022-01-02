"""
Math utility functions based on lectures from Prof. Daniel Cremers' Multiple View Geometry course.
"""
import numpy as np


def eulerToRotationMatrix(rXrYrZDegrees):
    rx, ry, rz = rXrYrZDegrees
    wx = col((1, 0, 0))
    wy = col((0, 1, 0))
    wz = col((0, 0, 1))

    Rx = exp(np.radians(rx) * skew(wx))
    Ry = exp(np.radians(ry) * skew(wy))
    Rz = exp(np.radians(rz) * skew(wz))
    R = Rz @ Ry @ Rx
    return R


def col(v):
    # create a column vector out of a list / tuple
    return np.array(v).reshape(-1, 1)


def exp(wHat: np.ndarray):
    # exponential mapping of a skew symmetric matrix so(3) onto
    #   the rotation matrix group SO(3), uses Rodrigues' formula
    w = unskew(wHat)
    wNorm = np.linalg.norm(w)
    I = np.eye(3)
    if np.isclose(wNorm, 0):
        term1 = 0
    else:
        term1 = (wHat / wNorm) * np.sin(wNorm)
    if np.isclose(wNorm, 0):
        term2 = 0
    else:
        term2 = ((wHat @ wHat) / (wNorm**2)) * (1 - np.cos(wNorm))
    R = I + term1 + term2
    return R


def skew(v):
    # converts vector into a skew symmetric matrix, aka 'hat' operator
    # only works for v of shape (3,1)
    validateShape(v.shape, (3,1))
    a = v[:,0]
    vHat = np.array([
        [    0, -a[2],  a[1]],
        [ a[2],     0, -a[0]],
        [-a[1],  a[0],     0],
    ])
    return vHat


def unskew(vHat):
    validateShape(vHat.shape, (3,3))
    return np.array([vHat[2,1], vHat[0,2], vHat[1,0]])


def validateShape(inputShape, requiredShape):
    for i in range(len(inputShape)):
        if requiredShape[i] is not None and requiredShape[i] != inputShape[i]:
            raise ValueError(f"Expected shape {requiredShape}, got {inputShape}")


def stack(A):
    # stacks the columns of a matrix into a single column
    return col(A.T.ravel())


def unstack(As):
    # unstacks the column vector into a squre matrix
    Nsquared = As.size
    N = int(np.sqrt(Nsquared))
    return As.reshape((N, N)).T


def homog(v):
    # v is Nx2 or Nx3
    if isinstance(v, tuple) or isinstance(v, list):
        v = np.array(v).reshape(-1, len(v))
    vh = np.hstack((v, np.ones((v.shape[0], 1))))
    return vh


def normalize(A):
    return A / A.ravel()[-1]


def poseFromRT(R, T):
    world_M_camera = np.eye(4)
    world_M_camera[:3,:3] = R
    if isinstance(T, np.ndarray):
        T = T.ravel()
    world_M_camera[:3,3] = T
    return world_M_camera


def project(A, cameraPose, X_0):
    """
    A             -- the intrinsic parameter matrix
    cameraPose    -- the camera pose in world
    X_0           -- the 3D points (homogeneous) in the world
    xp            -- x' the projected 2D points in the camera (homogeneous)

    Π0            -- standard projection matrix
    """
    Π_0 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])

    # λ*x' = 𝐾 * Π₀ * g * 𝑋₀
    g = np.linalg.inv(cameraPose)
    lambdaxp = (A @ Π_0 @ g @ X_0.T).T
    xp = lambdaxp / col(lambdaxp[:, -1])
    return xp

