"""
Core calibration functions.
"""
import numpy as np

from __context__ import src
from src import mathutils as mu


def project(K, cameraPose, X_0):
    """
    K             -- the intrinsic parameter matrix
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
    lambdaxp = (K @ Π_0 @ g @ X_0.T).T
    xp = lambdaxp / mu.col(lambdaxp[:, -1])
    return xp


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


def computeHomography(x, X):
    """
    Estimate homography using DLT

    Rearrange the model points, X, and the sensor points, x, into the following
    formulation:

        M * h = 0

    M represents the model and sensor point correspondences
    h is a vector representation of the homography H we are trying to find:
        (h11, h12, h13, h21, h22, h23, h31, h32, h33).T
    """
    N = x.shape[0]
    M = np.zeros((2*N, 9))
    for i in range(N):
        Xi, Yi = X[i][:2]
        ui, vi = x[i][:2]
        M[2*i,:] =   (-Xi, -Yi, -1,   0,   0,  0, ui * Xi, ui * Yi, ui)
        M[2*i+1,:] = (  0,   0,  0, -Xi, -Yi, -1, vi * Xi, vi * Yi, vi)
    U, S, V_T = np.linalg.svd(M)
    h = V_T[-1]
    H = h.reshape(3,3) / h[-1]
    return H
