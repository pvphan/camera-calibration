import time

import numpy as np

from __context__ import src
from src import distortion
from src import linearcalibrate
from src import jacobian
from src import mathutils as mu


class Calibrator:
    def __init__(self, distortionModel: distortion.DistortionModel):
        self._distortionModel = distortionModel
        self._jac = jacobian.ProjectionJacobian(self._distortionModel)

    def estimateCalibrationParameters(self, allDetections):
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
        Hs = linearcalibrate.estimateHomographies(allDetections)
        Ainitial = linearcalibrate.computeIntrinsicMatrix(Hs)
        Winitial = linearcalibrate.computeExtrinsics(Hs, Ainitial)
        kInitial = self._distortionModel.estimateDistortion(Ainitial, allDetections, Winitial)
        return Ainitial, Winitial, kInitial

    def refineCalibrationParameters(self, Ainitial, Winitial, kInitial, allDetections,
            maxIters=50, shouldPrint=False):
        """
        Input:
            Ainitial -- initial estimate of intrinsic matrix
            Winitial -- initial estimate of world-to-camera transforms
            kInitial -- initial estimate of distortion coefficients
            allDetections -- list of tuples (one for each view).
                    Each tuple is (Xa, Xb), a set of sensor points
                    and model points respectively
            jac -- Jacobian computation class

        Output:
            Arefined -- refined estimate of intrinsic matrix
            Wrefined -- refined estimate of world-to-camera transforms
            kRefined -- refined estimate of distortion coefficients

        Uses Levenberg-Marquardt to solve non-linear optimization. Jacobian matrices
            are compute by jacobian.py
        """
        Pt = composeParameterVector(Ainitial, Winitial, kInitial)
        allModelPoints = [modelPoints for sensorPoints, modelPoints in allDetections]
        ydot = getSensorPoints(allDetections)

        ts = time.time()
        # Levenberg-Marquardt
        λ = 1e-3
        for iter in range(maxIters):
            J = self._jac.compute(Pt, allModelPoints)

            JTJ = J.T @ J
            diagJTJ = np.diag(np.diagonal(JTJ))
            y = self._projectAllPoints(Pt, allModelPoints)

            # compute residuum
            r = ydot.reshape(-1, 1) - y.reshape(-1, 1)
            Δ = np.linalg.inv(JTJ + λ*diagJTJ) @ J.T @ r

            # evaluate if Pt + Δ reduces the error or not
            Pt_error = self._computeReprojectionError(Pt, allDetections)
            Pt1_error = self._computeReprojectionError(Pt + Δ, allDetections)

            if Pt1_error < Pt_error:
                Pt += Δ
                λ /= 10
            else:
                λ *= 10

            if shouldPrint:
                printIterationStats(iter, ts, Pt, min(Pt1_error, Pt_error))

            if λ < 1e-150 or Pt_error < 1e-12:
                break

        Arefined, Wrefined, kRefined = decomposeParameterVector(Pt)
        return Pt_error, Arefined, Wrefined, kRefined

    def _computeReprojectionError(self, P, allDetections):
        allModelPoints = [modelPoints for sensorPoints, modelPoints in allDetections]
        y = self._projectAllPoints(P, allModelPoints)
        ydot = getSensorPoints(allDetections)
        totalError = np.sum(np.linalg.norm(ydot - y, axis=1)**2)
        return totalError

    def _projectAllPoints(self, P, allModelPoints):
        A, W, k = decomposeParameterVector(P)
        ydot = np.empty((0, 2))
        for wP, cMw in zip(allModelPoints, W):
            cP = mu.transform(cMw, wP)
            yidot = self._distortionModel.projectWithDistortion(A, cP, k)
            ydot = np.vstack((ydot, yidot))
        return ydot


def printIterationStats(iter, ts, Pt, error):
    At, Wt, kt = decomposeParameterVector(Pt)
    print(f"\niter {iter}: ({time.time() - ts:0.3f}s), error={error:0.3f}")
    print(f"A:\n{At}")
    print(f"k:\n{kt}")


def getSensorPoints(allDetections):
    allSensorPoints = [sensorPoints for sensorPoints, modelPoints in allDetections]
    y = np.empty((0, 2))
    for sensorPoints in allSensorPoints:
        y = np.vstack((y, sensorPoints))
    return y


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
