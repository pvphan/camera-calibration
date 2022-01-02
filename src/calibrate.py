"""
Core calibration functions.
"""
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
    Compute the intrinsic matrix from a set of homographies

    Inputs:
        Hs -- list of homographies

    Output:
        A -- intrinsic camera matrix

    From the Burger paper, use equations 96 - 98 to solve for vector b = (B0, B1, B2, B3, B4, B5)^T
    and then compute the intrinsic matrix and return.

    Notes:
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
        V[2*i,:]   = vecHomography(H, 0, 1)
        V[2*i+1,:] = vecHomography(H, 0, 0) - vecHomography(H, 1, 1)
    U, S, V_T = np.linalg.svd(V)
    b = V_T[-1]
    A = computeIntrinsicMatrixFrombCholesky(b)
    return A


def vecHomography(H: np.ndarray, p: int, q: int):
    """
    Input:
        H -- 3x3 homography matrix
        p -- first column index
        q -- second column index

    Output:
        v -- 1x6 vector containing components of H based on the
             indices p and q which represent columns of the homography H

    Notes:
        This format of v allows the product to be used in homogenous
        form to solve for the values
        of a matrix which is a product of the intrinsic parameters (B).

        Implements equation 96 of Burger
    """
    values = (
        H[0,p] * H[0,q],
        H[0,p] * H[1,q] + H[1,p] * H[0,q],
        H[1,p] * H[1,q],
        H[2,p] * H[0,q] + H[0,p] * H[2,q],
        H[2,p] * H[1,q] + H[1,p] * H[2,q],
        H[2,p] * H[2,q],
    )
    v = np.array(values).reshape(1, 6)
    return v


def computeIntrinsicMatrixFrombClosedForm(b):
    """
    Computes the intrinsic matrix from the vector b using the closed
    form solution given in Burger, equations 99 - 104.

    Input:
        b -- vector made up of (B0, B1, B2, B3, B4, B5)^T

    Output:
        A -- intrinsic matrix
    """
    B0, B1, B2, B3, B4, B5 = b

    # eqs 104, 105
    w = B0*B2*B5 - B1**2*B5 - B0*B4**2 + 2*B1*B3*B4 - B2*B3**2
    d = B0*B2 - B1**2

    alpha = np.sqrt(w / (d * B0))               # eq 99
    beta = np.sqrt(w / d**2 * B0)               # eq 100
    gamma = np.sqrt(w / (d**2 * B0) * B1)       # eq 101
    uc = (B1*B4 - B2*B3) / d                    # eq 102
    vc = (B1*B3 - B0*B4) / d                    # eq 103

    A = np.array([
        [alpha, gamma, uc],
        [    0,  beta, vc],
        [    0,     0,  1],
    ])
    return A


def computeIntrinsicMatrixFrombCholesky(b):
    """
    Computes the intrinsic matrix from the vector b using
    Cholesky decomposition.

    Input:
        b -- vector made up of (B0, B1, B2, B3, B4, B5)^T

    Output:
        A -- intrinsic matrix

    Notes:
        Recall,

        B = (A^-1)^T * A^-1, where B = [B0 B1 B3]
                                       [B1 B2 B4]
                                       [B3 B4 B5]
        let L = (A^-1)^T
        then B = L * L^T
        L = Chol(B)

        and
        A = (L^T)^-1
    """
    B0, B1, B2, B3, B4, B5 = b
    B = np.array([
        [B0, B1, B3],
        [B1, B2, B4],
        [B3, B4, B5],
    ])
    L = np.linalg.cholesky(B)
    A = np.linalg.inv(L.T)
    A /= A[2,2]
    return A


def computeExtrinsics(Hs: list, A: np.ndarray):
    """
    Input:
        Hs -- list of homographies
        A -- intrinsic matrix

    Output:
        transformsWorldToCamera -- list of transform matrices from world to camera
    """
    Ainv = np.linalg.inv(A)
    transformsWorldToCamera = []
    for H in Hs:
        h0 = H[:,0]
        h1 = H[:,1]
        h2 = H[:,2]

        lmbda = 1 / np.linalg.norm(Ainv @ h0)

        r0 = lmbda * Ainv @ h0
        r1 = lmbda * Ainv @ h1
        r2 = np.cross(r0, r1)

        t = lmbda * Ainv @ h2

        # Q is not in SO(3)
        Q = np.hstack((mu.col(r0), mu.col(r1), mu.col(r2)))

        R = approximateRotationMatrix(Q)
        transformWorldToCamera = mu.poseFromRT(R, t)
        transformsWorldToCamera.append(transformWorldToCamera)
    return transformsWorldToCamera


def approximateRotationMatrix(Q: np.ndarray):
    """
    Input:
        Q -- a 3x3 matrix which is close to a rotation matrix

    Output:
        R -- a 3x3 rotation matrix which is in SO(3)

    Method from Zhang paper, Appendix C

    Notes:
        minimize R in frobenius_norm(R - Q) subject to R^T * R = I

        frobenius_norm(R - Q) = trace((R - Q)^T * (R - Q))
                              = 3 + trace(Q^T * Q) - 2*trace(R^T * Q)

        so equivalently, maximize trace(R^T * Q)

        let
            U, S, V^T = svd(Q)

        we define
            Z = V^T * R^T * U
        ==>
            trace(R^T * Q)
            trace(R^T * U * S * V^T)
            trace(V^T * R^T * U * S)
            trace(Z * S)
    """
    U, S, V_T = np.linalg.svd(Q)
    R = U @ V_T
    return R
