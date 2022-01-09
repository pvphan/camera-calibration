import enum
import numpy as np
import sympy

from __context__ import src
from src import mathutils as mu


class DistortionModel(enum.Enum):
    RadialTangential = "radtan"
    FishEye = "fisheye"


def distortPointsFisheye(x: np.ndarray, k: tuple):
    raise NotImplementedError()


def distortPoints(x: np.ndarray, k: tuple, isSymbolic=False):
    """
    Inputs:
        x -- normalized points (undistorted), (N,2)
        k -- distortion coefficients, (5,)

    Outputs:
        xd -- distorted normalized points (N,2)
    """
    k1, k2, p1, p2, k3 = k

    if isSymbolic:
        xn = x[0,0]
        yn = x[0,1]
        r = sympy.sqrt(xn**2 + yn**2)
    else:
        xn = x[:,0]
        yn = x[:,1]
        r = np.linalg.norm(x, axis=1)

    radialComponent = (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    tangentialComponentX = (2 * p1 * xn * yn + p2 * (r**2 + 2 * xn**2))
    tangentialComponentY = (p1 * (r**2 + 2 * yn**2) + 2 * p2 * xn * yn)

    xd = radialComponent * xn + tangentialComponentX
    yd = radialComponent * yn + tangentialComponentY

    return np.hstack((mu.col(xd), mu.col(yd)))


def projectWithDistortion(A, X, k, isSymbolic=False):
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
    xd = distortPoints(x, k, isSymbolic=isSymbolic)
    Ap = A[:2,:3]
    distortedPointsInSensor = (Ap @ mu.hom(xd).T).T
    return distortedPointsInSensor

