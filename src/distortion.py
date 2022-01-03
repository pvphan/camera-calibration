import numpy as np

from __context__ import src
from src import mathutils as mu


def distortPoints(normalizedPointsNx2: np.ndarray, distortionCoeffients: tuple):
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
