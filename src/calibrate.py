"""
Core calibration functions.
"""
import time

import numpy as np

from __context__ import src
from src import distortion
from src import jacobian
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
        H -- homography matrix which relates Xa and Xb

    Rearrange into the formulation:

        M * h = 0

    M represents the model and sensor point correspondences
    h is a vector representation of the homography H we are trying to find:
        h = (h11, h12, h13, h21, h22, h23, h31, h32, h33).T
    """
    mu.validateShape(Xa.shape, (None, 2))
    mu.validateShape(Xb.shape, (None, 2))
    N = Xa.shape[0]
    M = np.zeros((2*N, 9))
    for i in range(N):
        ui, vi = Xa[i][:2]
        Xi, Yi = Xb[i][:2]
        M[2*i,:]   = (-Xi, -Yi, -1,   0,   0,  0, ui * Xi, ui * Yi, ui)
        M[2*i+1,:] = (  0,   0,  0, -Xi, -Yi, -1, vi * Xi, vi * Yi, vi)
    U, S, V_T = np.linalg.svd(M)
    h = V_T[-1]
    Hp = h.reshape(3,3)
    H = Hp / Hp[2,2]
    return H


def estimateHomographyWithNormalization(Xa, Xb):
    # doesn't pass the unit test, also equation in the summary doesn't match
    #   the derivation from Burger section 3.2
    mu.validateShape(Xa.shape, (None, 2))
    mu.validateShape(Xb.shape, (None, 2))
    Na = getNormalizationMatrix(Xa)
    Nb = getNormalizationMatrix(Xb)
    N = Xa.shape[0]
    M = np.zeros((2*N, 9))
    for i in range(N):
        ui, vi = mu.unhom(Na @ mu.hom(Xa[i][:2]))
        Xi, Yi = mu.unhom(Nb @ mu.hom(Xb[i][:2]))
        M[2*i,:]   = (-Xi, -Yi, -1,   0,   0,  0, ui * Xi, ui * Yi, ui)
        M[2*i+1,:] = (  0,   0,  0, -Xi, -Yi, -1, vi * Xi, vi * Yi, vi)
    U, S, V_T = np.linalg.svd(M)
    h = V_T[-1]
    Hp = h.reshape(3,3)
    H = np.linalg.inv(Nb) @ Hp @ Na
    H /= H[2,2]
    return H


def getNormalizationMatrix(X):
    mu.validateShape(X.shape, (None, 2))
    x = X[:,0]
    y = X[:,1]
    xbar = np.mean(x)
    ybar = np.mean(y)
    sigmax2 = np.mean((x - xbar)**2)
    sigmay2 = np.mean((y - ybar)**2)

    sx = np.sqrt(2/sigmax2)
    sy = np.sqrt(2/sigmay2)
    N_X = np.array([
        [sx,  0, -sx*xbar],
        [ 0, sy, -sy*ybar],
        [ 0,  0,        1],
    ])
    return N_X


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
    b = V_T[-1]

    # could use the 'closed form' solution here instead (also implemented),
    #   but using Cholesky decomposition was more interesting
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

    α = np.sqrt(w / (d * B0))          # eq 99
    β = np.sqrt(w / d**2 * B0)         # eq 100
    γ = np.sqrt(w / (d**2 * B0) * B1)  # eq 101
    uc = (B1*B4 - B2*B3) / d           # eq 102
    vc = (B1*B3 - B0*B4) / d           # eq 103

    A = np.array([
        [α, γ, uc],
        [0, β, vc],
        [0, 0,  1],
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

        λ = 1 / np.linalg.norm(Ainv @ h0)

        r0 = λ * Ainv @ h0
        r1 = λ * Ainv @ h1
        r2 = np.cross(r0, r1)

        t = λ * Ainv @ h2

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


def estimateDistortion(A: np.ndarray, allDetections: list, allBoardPosesInCamera: list):
    """
    Input:
        A -- estimated intrinsic matrix
        allDetections -- a list of tuples, where the tuples are the measurements
                (measuredPointsInSensor, measuredPointsInBoard)
        allBoardPosesInCamera -- a list of all board poses in camera, corresponding
                to each view

    Output:
        k -- the distortion model, made up of (k1, k2, p1, p2, k3)

    Notes:
        We have M views, each with N points
        Formulate the problem as the linear system:

            D * k = Ddot

        where D is (2MN, 5) and and Ddot is (2MN, 1)

        for each 2 rows in D:
            [(uij - uc) * rij**2, (uij - uc) * rij**4]
            [(uij - vc) * rij**2, (uij - vc) * rij**4]

        for each 2 rows in Ddot:
            (udotij - uij)
            (vdotij - uij)

        We'll solve this linear system by taking the pseudo-inverse of D and
        left-multiplying it with Ddot.

        Solution: k = pinv(D) * Ddot
    """
    fx = A[0,0]
    fy = A[1,1]
    uc = A[0,2]
    vc = A[1,2]
    D = np.empty((0,5))
    Ddot = np.empty((0,1))

    shouldRunVectorized = False
    if shouldRunVectorized:
        raise NotImplementedError("Haven't vectorized 5 parameter distortion yet")
        # ~15x speedup over the unvectorized loop version below
        for i, ((Udot, bX), cMb) in enumerate(zip(allDetections, allBoardPosesInCamera)):
            cXi = mu.transform(cMb, bX)
            xi = mu.projectStandard(cXi)
            ri = np.linalg.norm(xi, axis=1)
            r = np.hstack((mu.col(ri)**2, mu.col(ri)**4))

            # U is the projected sensor points without distortion
            U = mu.project(A, np.eye(4), cXi)
            Di1 = (U - (uc, vc)).reshape(-1, 1)
            Di = np.tile(Di1, (1,2))
            Di[::2,:] = Di1[::2,:] * r
            Di[1::2,:] = Di1[1::2,:] * r
            D = np.vstack((D, Di))

            # Udot is the measured sensor points with distortion
            Ddoti = (Udot - U).reshape(-1, 1)
            Ddot = np.vstack((Ddot, Ddoti))

    else:
        # keeping the unvectorized version for posterity. it's also easier to read
        for i, ((Udot, bX), cMb) in enumerate(zip(allDetections, allBoardPosesInCamera)):
            for j, (udot, bXij) in enumerate(zip(Udot, bX)):
                # rij is computed from the normalized sensor coordinate, which is computed by
                #   projecting the 3D model point to camera coordinates using the standard
                #   projection matrix (f=1)
                cXij = mu.transform(cMb, bXij)
                xij = mu.projectStandard(cXij)
                rij = np.linalg.norm(xij)

                # the measured sensor points with distortion
                udotij, vdotij = udot

                # the projected sensor points without distortion
                u, v = mu.project(A, np.eye(4), cXij)

                xn, yn = xij.ravel()
                Dij = np.array([
                    [
                        (u - uc) * rij**2,
                        (u - uc) * rij**4,
                        fx * (2 * xn * yn),
                        fy * (rij**2 + 2 * xn**2),
                        (u - uc) * rij**6,
                    ],
                    [
                        (v - vc) * rij**2,
                        (v - vc) * rij**4,
                        fx * (rij**2 + 2 * yn**2),
                        fy * (2 * xn * yn),
                        (v - vc) * rij**6,
                    ],
                ])
                D = np.vstack((D, Dij))

                Ddotij = np.array([
                    [udotij - u],
                    [vdotij - v],
                ])
                Ddot = np.vstack((Ddot, Ddotij))
    k = np.linalg.pinv(D) @ Ddot
    return tuple(k.ravel())


def estimateCalibrationParameters(allDetections):
    """
    Input:
        allDetections -- list of tuples (one for each view).
                Each tuple is (Xa, Xb), a set of sensor points
                and model points respectively

    Output:
        Ainitial -- initial estimate of intrinsic matrix
        Winitial -- initial estimate of world-to-camera transforms
        kInitial -- initial estimate of distortion coefficients
    """
    Hs = estimateHomographies(allDetections)
    Ainitial = computeIntrinsicMatrix(Hs)
    Winitial = computeExtrinsics(Hs, Ainitial)
    kInitial = estimateDistortion(Ainitial, allDetections, Winitial)
    return Ainitial, Winitial, kInitial


def refineCalibrationParameters(Ainitial, Winitial, kInitial, allDetections):
    """
    Input:
        Ainitial -- initial estimate of intrinsic matrix
        Winitial -- initial estimate of world-to-camera transforms
        kInitial -- initial estimate of distortion coefficients
        allDetections -- list of tuples (one for each view).
                Each tuple is (Xa, Xb), a set of sensor points
                and model points respectively

    Output:
        Arefined -- refined estimate of intrinsic matrix
        Wrefined -- refined estimate of world-to-camera transforms
        kRefined -- refined estimate of distortion coefficients

    Uses Levenberg-Marquardt to solve non-linear optimization. Jacobian matrices
        are compute by jacobian.py
    """
    shouldPrint = True
    maxIters = 50
    Pt = composeParameterVector(Ainitial, Winitial, kInitial)
    allModelPoints = [modelPoints for sensorPoints, modelPoints in allDetections]
    ydot = getSensorPoints(allDetections)

    jac = jacobian.ProjectionJacobian(distortion.DistortionModel.RadialTangential)

    ts = time.time()
    # Levenberg-Marquardt
    λ = 1e-3
    for iter in range(maxIters):
        J = jac.compute(Pt, allModelPoints)

        JTJ = J.T @ J
        diagJTJ = np.diag(np.diagonal(JTJ))
        y = projectAllPoints(Pt, allModelPoints)

        # compute residuum
        r = ydot.reshape(-1, 1) - y.reshape(-1, 1)
        Δ = np.linalg.inv(JTJ + λ*diagJTJ) @ J.T @ r

        # evaluate if Pt + Δ reduces the error or not
        Pt_error = computeReprojectionError(Pt, allDetections)
        Pt1_error = computeReprojectionError(Pt + Δ, allDetections)

        if Pt1_error < Pt_error:
            Pt += Δ
            λ /= 10
        else:
            λ *= 10

        if shouldPrint:
            printIterationStats(iter, ts, Pt, Pt_error)

        if λ < 1e-150 or Pt_error < 1e-12:
            break

    Arefined, Wrefined, kRefined = decomposeParameterVector(Pt)
    return Arefined, Wrefined, kRefined


def printIterationStats(iter, ts, Pt, Pt_error):
    At, Wt, kt = decomposeParameterVector(Pt)
    print(f"\niter {iter}: ({time.time() - ts:0.3f}s), error = {Pt_error:0.3f}")
    print(f"A:\n{At}")
    print(f"k:\n{kt}")


def computeReprojectionError(P, allDetections):
    allModelPoints = [modelPoints for sensorPoints, modelPoints in allDetections]
    y = projectAllPoints(P, allModelPoints)
    ydot = getSensorPoints(allDetections)
    totalError = np.sum(np.linalg.norm(ydot - y, axis=1)**2)
    return totalError


def getSensorPoints(allDetections):
    allSensorPoints = [sensorPoints for sensorPoints, modelPoints in allDetections]
    y = np.empty((0, 2))
    for sensorPoints in allSensorPoints:
        y = np.vstack((y, sensorPoints))
    return y


def projectAllPoints(P, allModelPoints):
    A, W, k = decomposeParameterVector(P)
    ydot = np.empty((0, 2))
    for wP, cMw in zip(allModelPoints, W):
        cP = mu.transform(cMw, wP)
        yidot = distortion.projectWithDistortion(A, cP, k)
        ydot = np.vstack((ydot, yidot))
    return ydot


def composeParameterVector(A, W, k):
    """
    Input:
        A -- intrinsic matrix
        W -- world-to-camera transforms
        k -- distortion coefficients

    Output:
        P -- vector of all calibration parameters, intrinsic and all M views extrinsic:
            P = (α, β, γ, uc, uv, k1, k2, p1, p2, k3,
                    ρ0x, ρ0y, ρ0z, t0x, t0y, t0z,
                    ρ1x, ρ1y, ρ1z, t1x, t1y, t1z,
                    ...,
                    ρM-1x, ρM-1y, ρM-1z, tM-1x, tM-1y, tM-1z,
                    ...)^T
    """
    α = A[0,0]
    β = A[1,1]
    γ = A[0,1]
    uc = A[0,2]
    vc = A[1,2]
    k1, k2, p1, p2, k3 = k

    P = mu.col([α, β, γ, uc, vc, k1, k2, p1, p2, k3])

    for cMw in W:
        R = cMw[:3,:3]
        t = cMw[:3,3]
        ρix, ρiy, ρiz = mu.rotationMatrixToEuler(R)
        tix, tiy, tiz = t
        P = np.vstack((P, mu.col([ρix, ρiy, ρiz, tix, tiy, tiz])))
    return P


def decomposeParameterVector(P):
    """
    Input:
        P -- vector of all calibration parameters, intrinsic and all M views extrinsic:
            P = (α, β, γ, uc, uv, k1, k2, p1, p2, k3,
                    ρ0x, ρ0y, ρ0z, t0x, t0y, t0z,
                    ρ1x, ρ1y, ρ1z, t1x, t1y, t1z,
                    ...,
                    ρM-1x, ρM-1y, ρM-1z, tM-1x, tM-1y, tM-1z,
                    ...)^T

    Output:
        A -- intrinsic matrix
        W -- world-to-camera transforms
        k -- distortion coefficients
    """
    if isinstance(P, np.ndarray):
        P = P.ravel()
    poseStartIndex = 10
    numPoseParams = 6
    α, β, γ, uc, vc, k1, k2, p1, p2, k3 = P[:poseStartIndex]
    A = np.array([
        [α, γ, uc],
        [0, β, vc],
        [0, 0,  1],
    ])
    W = []
    for i in range(poseStartIndex, len(P), numPoseParams):
        ρix, ρiy, ρiz, tix, tiy, tiz = P[i:i+numPoseParams]
        R = mu.eulerToRotationMatrix((ρix, ρiy, ρiz))
        t = (tix, tiy, tiz)
        W.append(mu.poseFromRT(R, t))

    k = (k1, k2, p1, p2, k3)
    return A, W, k
