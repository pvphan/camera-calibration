import numpy as np

from __context__ import src
from src import mathutils as mu


def distort(normalizedPointsNx2: np.ndarray, distortionCoeffients: tuple):
    """https://euratom-software.github.io/calcam/html/intro_theory.html#rectilinear-lens-distortion-model"""
    k1, k2, p1, p2, k3 = distortionCoeffients
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


def distortSimple(normalizedPointsNx2: np.ndarray, distortionCoeffients: tuple):
    k1, k2 = distortionCoeffients
    r = np.linalg.norm(normalizedPointsNx2, axis=1)

    xn = normalizedPointsNx2[:,0]
    yn = normalizedPointsNx2[:,1]

    radialComponent = (1 + k1 * r**2 + k2 * r**4)

    xd = radialComponent * xn
    yd = radialComponent * yn

    normalizedDistortedPointsNx2 = np.hstack((mu.col(xd), mu.col(yd)))
    return normalizedDistortedPointsNx2


def estimateDistortion():
    return k
