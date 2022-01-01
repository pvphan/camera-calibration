"""
Core calibration functions.
"""
import numpy as np

from __context__ import src
from src import mathutils as mu


def project(K, cameraPose, X_0):
    # K             -- the intrinsic parameter matrix
    # cameraPose    -- the camera pose in world
    # X_0           -- the 3D points (homogeneous) in the world
    # xp            -- x' the projected 2D points in the camera (homogeneous)

    # Π0            -- standard projection matrix
    Π_0 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])

    # λ*x' = 𝐾 * Π₀ * g * 𝑋₀
    g = np.linalg.inv(cameraPose)
    lambdaxp = (K @ Π_0 @ g @ X_0.T).T
    xp = lambdaxp / mu.col(lambdaxp[:, -1])
    return xp


# https://euratom-software.github.io/calcam/html/intro_theory.html#rectilinear-lens-distortion-model
def distort(normalizedPointsNx2: np.ndarray, distortionCoeffients: tuple):
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


def computeHomography(x, X):
    # Estimate homography using DLT
    pass
