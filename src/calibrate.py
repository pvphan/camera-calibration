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

    Inputs:
        X -- 2D model points (3rd dimension ignored)
        x -- 2D points in sensor

    Output:
        H -- homography matrix which relates x and X

    Rearrange into the formulation:

        M * h = 0

    M represents the model and sensor point correspondences
    h is a vector representation of the homography H we are trying to find:
        h = (h11, h12, h13, h21, h22, h23, h31, h32, h33).T
    """
    N = x.shape[0]
    M = np.zeros((2*N, 9))
    for i in range(N):
        Xi, Yi = X[i][:2]
        ui, vi = x[i][:2]
        M[2*i,:]   = (-Xi, -Yi, -1,   0,   0,  0, ui * Xi, ui * Yi, ui)
        M[2*i+1,:] = (  0,   0,  0, -Xi, -Yi, -1, vi * Xi, vi * Yi, vi)
    U, S, V_T = np.linalg.svd(M)
    h = V_T[-1]
    H = h.reshape(3,3) / h[-1]
    return H


def computeIntrinsicMatrix(Hs: list):
    """
    Compute the intrinsic matrix from a set of homographies using the closed form solution.

    Inputs:
        Hs -- list of homographies

    Output:
        K -- intrinsic camera matrix

    From the Burger paper, use equations 96 - 105 to solve for vector b = (B0, B1, B2, B3, B4, B5)^T
    and then compute alpha, beta, gamma, principal_x, principal_y. Store them in a matrix and return.

    H = [h0 h1 h2] = lambda * A * [r0 r1 t]

    By leveraging that r0 and r1 are orthonormal, we get:

        h0^T * (A^-1)^T * A^-1 * h1 = 0
        h0^T * (A^-1)^T * A^-1 * h0 = h1^T * (A^-1)^T * A^-1 * h1

    B = (A^-1)^T * A^-1, where B = [B0 B1 B3]
                                   [B1 B2 B4]
                                   [B3 B4 B5]

    simplifying:
        h0^T * B * h1 = 0
        h0^T * B * h0 - h1^T * B * h1 = 0

    letting b = (B0, B1, B2, B3, B4, B5)^T

    we reformulate the h^T * B * h form:
        hp^T * B * hq = vecpq(H) * b

        with vec(H, p, q) = (
                H0p * H0q,
                H0p * H1q + H1p * H0q,
                H1p * H1q,
                H2p * H0q + H0p * H2q,
                H2p * H1q + H1p * H2q,
                H2p * H2q,
            )

    so we can rewrite our system of equations for a single homography as:

        [        vec(H, 0, 1)        ] * b = 0
        [vec(H, 0, 0) - vec(H, 1, 1) ]

    Now we can stack these terms in the left matrix for each homography to create
    a matrix V of size (2*N, 6) and then solve for b with SVD.

        V * b = 0
    """
    N = len(Hs)
    V = np.zeros((2*N, 6))
    for i in range(N):
        H = Hs[i]
        V[2*i,:]   = vecHomog(H, 0, 1)
        V[2*i+1,:] = vecHomog(H, 0, 0) - vecHomog(H, 1, 1)
    U, S, V_T = np.linalg.svd(V)
    b = V_T[-1]
    B0, B1, B2, B3, B4, B5 = b

    # eqs 104, 105
    w = B0*B2*B5 - B1**2*B5 - B0*B4**2 + 2*B1*B3*B4 - B2*B3**2
    d = B0*B2 - B1**2

    alpha = np.sqrt(w / (d * B0))               # eq 99
    beta = np.sqrt(w / d**2 * B0)               # eq 100
    gamma = np.sqrt(w / (d**2 * B0) * B1)       # eq 101
    uc = (B1*B4 - B2*B3) / d                    # eq 102
    vc = (B1*B3 - B0*B4) / d                    # eq 103

    K = np.array([
        [alpha, gamma, uc],
        [    0,  beta, vc],
        [    0,     0,  1],
    ])
    return K


def vecHomog(H: np.ndarray, p: int, q: int):
    """
    Creates a vector of shape (1, 6) made up of components of H based on the
    indices p and q which represent columns of the homography H. This format
    allows the product to be used in homogenous form to solve for the values
    of a matrix which is a product of the intrinsic parameters (B).
    """
    values = (
        H[0,p] * H[0,q],
        H[0,p] * H[1,q] + H[1,p] * H[0,q],
        H[1,p] * H[1,q],
        H[2,p] * H[0,q] + H[0,p] * H[2,q],
        H[2,p] * H[1,q] + H[1,p] * H[2,q],
        H[2,p] * H[2,q],
    )
    return np.array(values).reshape(1, 6)
