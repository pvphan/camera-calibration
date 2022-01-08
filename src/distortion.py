import numpy as np

from __context__ import src
from src import mathutils as mu


def distortPointsFisheye(x: np.ndarray, distortionCoeffients: tuple):
    raise NotImplementedError()


def distortPoints(x: np.ndarray, distortionCoeffients: tuple):
    """
    Inputs:
        x -- normalized points (undistorted), (N,2)
        k -- distortion coefficients, (5,) or (2,)

    Outputs:
        xd -- distorted normalized points (N,2)
    """
    if len(distortionCoeffients) == 2:
        k1, k2 = distortionCoeffients
        p1 = p2 = k3 = 0
    elif len(distortionCoeffients) == 5:
        k1, k2, p1, p2, k3 = distortionCoeffients
    else:
        raise ValueError(f"Invalid distortion coefficient length {len(distortionCoeffients)}: "
                f"{distortionCoeffients}")
    r = np.linalg.norm(x, axis=1)

    xn = x[:,0]
    yn = x[:,1]

    radialComponent = (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    tangentialComponentX = (2 * p1 * xn * yn + p2 * (r**2 + 2 * xn**2))
    tangentialComponentY = (p1 * (r**2 + 2 * yn**2) + 2 * p2 * xn * yn)

    xd = radialComponent * xn + tangentialComponentX
    yd = radialComponent * yn + tangentialComponentY

    normalizedDistortedPointsNx2 = np.hstack((mu.col(xd), mu.col(yd)))
    return normalizedDistortedPointsNx2


def projectWithDistortion(A, X, k):
    """
    Input:
        A -- intrinsic matrix
        X -- Nx3 points in camera coordinates
        k -- distortion coefficients

    Output:
        distortedPointsInSensor -- Nx2 matrix

    x -- normalized points in camera
    xd -- distorted normalized points in camera
    """
    x = mu.projectStandard(X)
    xd = distortPoints(x, k)
    Ap = A[:2,:3]
    distortedPointsInSensor = (Ap @ mu.hom(xd).T).T
    return distortedPointsInSensor

