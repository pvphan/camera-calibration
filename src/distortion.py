import numpy as np

from __context__ import src
from src import mathutils as mu


def distortPoints_old(normalizedPointsNx2: np.ndarray, distortionCoeffients: tuple):
    """https://euratom-software.github.io/calcam/html/intro_theory.html#rectilinear-lens-distortion-model"""
    if len(distortionCoeffients) == 2:
        k1, k2 = distortionCoeffients
        p1 = p2 = k3 = 0
    elif len(distortionCoeffients) == 5:
        k1, k2, p1, p2, k3 = distortionCoeffients
    else:
        raise ValueError(f"Invalid distortion coefficient length {len(distortionCoeffients)}: "
                f"{distortionCoeffients}")
    r = np.linalg.norm(normalizedPointsNx2, axis=1)

    xn = normalizedPointsNx2[:,0]
    yn = normalizedPointsNx2[:,1]

    radialComponent = (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    tangentialComponentX = (2 * p1 * xn * yn + p2 * (r**2 + 2 * xn**2))
    tangentialComponentY = (p1 * (r**2 + 2 * yn**2) + 2 * p2 * xn * yn)

    xd = radialComponent * xn + tangentialComponentX
    yd = radialComponent * yn + tangentialComponentY

    normalizedDistortedPointsNx2 = np.hstack((mu.col(xd), mu.col(yd)))
    return normalizedDistortedPointsNx2


def distortPoints(x: np.ndarray, k: tuple):
    """
    Inputs:
        x -- normalized points (undistorted), (N,2)
        k -- distortion coefficients, (5,)

    Outputs:
        xd -- distorted normalized points (N,2)
    """
    if len(k) == 2:
        k1, k2 = k
        p1 = p2 = k3 = 0
    elif len(k) == 5:
        k1, k2, p1, p2, k3 = k
    else:
        raise ValueError(f"Invalid distortion coefficient length {len(k)}: {k}")
    r = np.linalg.norm(x, axis=1)
    D = k1 * r**2 + k2 * r**4
    xd_x = x[:,0] * (1 + D)
    xd_y = x[:,1] * (1 + D)
    xd = np.hstack((mu.col(xd_x), mu.col(xd_y)))
    return xd


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


def estimateDistortion(A: np.ndarray, allDetections: list):
    """
    Input:
        A -- estimated intrinsic matrix
        allDetections -- a list of tuples, where the tuples are the measurements
                (measuredPointsInSensor, measuredPointsInBoard)

    Output:
        k -- the distortion model, made up of (k1, k2)^T

    Notes:
        We have M views, each with N points
        Formulate the problem as the linear system:

            D * k = Ddot

        where D is (2MN, 2) and and Ddot is (2MN, 1)

        We'll solve this linear system by taking the pseudo-inverse of D and
        left-multiplying it with Ddot.

        k = pinv(D) * Ddot
    """
    k = None
    return k
