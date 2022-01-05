import numpy as np

from __context__ import src
from src import mathutils as mu


def distortPoints(normalizedPointsNx2: np.ndarray, distortionCoeffients: tuple):
    """
    Inputs:
        x -- normalized points (undistorted), (N,2)
        k -- distortion coefficients, (5,) or (2,)

    Outputs:
        xd -- distorted normalized points (N,2)
    """
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


def distortPointsSimple(x: np.ndarray, k: tuple):
    """
    Inputs:
        x -- normalized points (undistorted), (N,2)
        k -- distortion coefficients, (2,)

    Outputs:
        xd -- distorted normalized points (N,2)
    """
    if len(k) == 2:
        k1, k2 = k
    else:
        raise ValueError(f"Invalid distortion coefficient length {len(k)}: {k}")
    r = np.linalg.norm(x, axis=1)
    D = k1 * r**2 + k2 * r**4
    xd_x = x[:,0] * (1 + D)
    xd_y = x[:,1] * (1 + D)
    xd = np.hstack((mu.col(xd_x), mu.col(xd_y)))
    #xd = x * mu.col((1 + D))
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


def estimateDistortion(A: np.ndarray, allDetections: list, allBoardPosesInCamera: list):
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

        for each 2 rows in D:
            [(udotij - uc) * rij**2, (udotij - uc) * rij**4]
            [(vdotij - vc) * rij**2, (vdotij - vc) * rij**4]

        for each 2 rows in Ddot:
            (uij - udotij)
            (vij - vdotij)
    """
    uc = A[0,2]
    vc = A[1,2]
    D = np.empty((0,2))
    Ddot = np.empty((0,1))
    for i, ((U, bX), cMb) in enumerate(zip(allDetections, allBoardPosesInCamera)):
        for j, (udot, bXij) in enumerate(zip(U, bX)):
            # rij is computed from the normalized image coordinate, which is computed by
            #   projecting the 3D model point to camera coordinates using the standard
            #   projection (f=1)
            cXij = mu.transform(cMb, bXij)
            xij = mu.projectStandard(cXij)
            rij = np.linalg.norm(xij)

            # the measured image points with distortion
            udotij, vdotij = udot

            # the projected image points without distortion
            u, v = mu.project(A, np.eye(4), cXij)

            Dij = np.array([
                [(u - uc) * rij**2, (u - uc) * rij**4],
                [(v - vc) * rij**2, (v - vc) * rij**4],
            ])
            D = np.vstack((D, Dij))

            Ddotij = np.array([
                [udotij - u],
                [vdotij - v],
            ])
            Ddot = np.vstack((Ddot, Ddotij))
    k = np.linalg.pinv(D) @ Ddot
    return tuple(k.ravel())
