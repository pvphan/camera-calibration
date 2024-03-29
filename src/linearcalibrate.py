import numpy as np

from __context__ import src
from src import mathutils as mu


def estimateHomographies(allDetections: list):
    """
    Input:
        allDetections -- list of tuples (one for each view).
                Each tuple is (Xa, Xb), a set of sensor points
                and model points respectively

    Output:
        Hs -- list of homographies, one for each view
    """
    Hs = []
    for Xa, Xb in allDetections:
        H = estimateHomography(Xa[:,:2], Xb[:,:2])
        Hs.append(H)
    return Hs


def estimateHomography(Xa: np.ndarray, Xb: np.ndarray):
    """
    Estimate homography using DLT
    Inputs:
        Xa -- 2D points in sensor
        Xb -- 2D model points
    Output:
        aHb -- homography matrix which relates the model plane (points Xb)
                to the sensor plane (points Xa)
    Rearrange into the formulation:
        M * h = 0
    M represents the model and sensor point correspondences
    h is a vector representation of the homography aHb we are trying to find:
        h = (h11, h12, h13, h21, h22, h23, h31, h32, h33).T

    Prior to constructing M, the points Xa and Xb need to be 'normalized'
    so that the results of SVD are more well behaved.
    """
    mu.validateShape(Xa.shape, (None, 2))
    mu.validateShape(Xb.shape, (None, 2))
    Na = computeNormalizationMatrix(Xa)
    Nb = computeNormalizationMatrix(Xb)
    N = Xa.shape[0]
    M = np.zeros((2*N, 9))
    for i in range(N):
        ui, vi = mu.unhom(Na @ mu.hom(Xa[i]))
        Xi, Yi = mu.unhom(Nb @ mu.hom(Xb[i]))
        M[2*i,:]   = (-Xi, -Yi, -1,   0,   0,  0, ui * Xi, ui * Yi, ui)
        M[2*i+1,:] = (  0,   0,  0, -Xi, -Yi, -1, vi * Xi, vi * Yi, vi)
    U, S, V_T = np.linalg.svd(M)
    h = V_T[-1]
    Hp = h.reshape(3,3)
    H = np.linalg.inv(Na) @ Hp @ Nb
    H /= H[2,2]
    return H


def computeNormalizationMatrix(X):
    """
    Compute a matrix M which maps a set of points X to their 'normalized'
    form Xnorm, i.e.

        Xnorm = unhom(M * hom(X))

    where the mean Euclidean distance of the of points in Xnorm is sqrt(2)
    and the centroid of the points is the origin.

    For more on why this is necessary, see 'Multiple View Geometry in Computer Vision,
    2nd edition', Hartley & Zisserman, §4.4.4, pg 108.
    """
    Xmean = np.mean(X, axis=0)
    M1 = np.array([
        [1, 0, -Xmean[0]],
        [0, 1, -Xmean[1]],
        [0, 0, 1],
    ])
    Xshifted = X - Xmean
    Xmagnitudes = np.linalg.norm(Xshifted, axis=1)
    meanMagnitude = np.mean(Xmagnitudes)
    scaleFactor = np.sqrt(2) / meanMagnitude
    M2 = np.array([
        [scaleFactor, 0, 0],
        [0, scaleFactor, 0],
        [0, 0, 1],
    ])
    M = M2 @ M1
    return M


def computeIntrinsicMatrix(Hs: list):
    """
    Compute the intrinsic matrix from a list of homographies

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
    b = tuple(V_T[-1])

    A = computeIntrinsicMatrixFrombCholesky(b)
    if np.sum(np.isnan(A)) > 0:
        raise ValueError(f"Computed intrinsic matrix contains NaN: \n{A}")
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


def computeIntrinsicMatrixFrombClosedFormBurger(b):
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

    α = np.sqrt(w / (d * B0))          # eq 99
    β = np.sqrt(w / d**2 * B0)         # eq 100
    γ = np.sqrt(w / (d**2 * B0)) * B1  # eq 101
    uc = (B1*B4 - B2*B3) / d           # eq 102
    vc = (B1*B3 - B0*B4) / d           # eq 103

    A = np.array([
        [α, γ, uc],
        [0, β, vc],
        [0, 0,  1],
    ])
    return A


def computeIntrinsicMatrixFrombClosedFormZhang(b):
    """
    Computes the intrinsic matrix from the vector b using the closed
    form solution given in Burger, equations 99 - 104.

    Input:
        b -- vector made up of (B0, B1, B2, B3, B4, B5)^T

    Output:
        A -- intrinsic matrix
    """
    B = matrixBfromVector(b)
    B11 = B[0,0]
    B12 = B[1,0]
    B13 = B[2,0]
    B22 = B[1,1]
    B23 = B[1,2]
    B33 = B[2,2]

    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    λ = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    α = np.sqrt(λ / B11)
    β = np.sqrt((λ * B11) / (B11 * B22 - B12**2))
    γ = -B12 * α**2 * β / λ
    u0 = γ * v0 / β - B13 * α**2 / λ

    A = np.array([
        [α, γ, u0],
        [0, β, v0],
        [0, 0,  1],
    ])
    return A


def matrixBfromVector(b):
    B0, B1, B2, B3, B4, B5 = b
    B = np.array([
        [B0, B1, B3],
        [B1, B2, B4],
        [B3, B4, B5],
    ])
    return B


def computeIntrinsicMatrixFrombCholesky(b: tuple):
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
    # ensure B is positive semi-definite
    sign = +1
    if B0 < 0 or B2 < 0 or B5 < 0:
        sign = -1
    B = sign * np.array([
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
        worldToCameraTransforms -- list of transform matrices from world to camera
    """
    Ainv = np.linalg.inv(A)
    worldToCameraTransforms = []
    for H in Hs:
        h0 = H[:,0]
        h1 = H[:,1]
        h2 = H[:,2]

        λ = np.linalg.norm(Ainv @ h0)

        r0 = (1 / λ) * Ainv @ h0
        r1 = (1 / λ) * Ainv @ h1
        r2 = np.cross(r0, r1)

        t = (1 / λ) * Ainv @ h2

        # Q is not in SO(3)
        Q = np.hstack((mu.col(r0), mu.col(r1), mu.col(r2)))

        # R is in SO(3)
        R = approximateRotationMatrix(Q)
        transformWorldToCamera = mu.poseFromRT(R, t)
        worldToCameraTransforms.append(transformWorldToCamera)
    return worldToCameraTransforms


def approximateRotationMatrix(Q: np.ndarray):
    """
    Input:
        Q -- a 3x3 matrix which is close to a rotation matrix

    Output:
        R -- a 3x3 rotation matrix which is in SO(3)

    Method from Zhang paper, Appendix C

    Notes:
        Find R which minimizes the frobenius_norm(R - Q) subject to R^T * R = I

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

