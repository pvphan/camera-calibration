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
        self._jac = None

    def calibrate(self, allDetections, maxIters=50):
        """
        Input:
            allDetections -- list of tuples (one for each view).
                    Each tuple is (Xa, Xb), a set of sensor points
                    and model points respectively

        Output:
            sse -- final sum squared error
            Afinal -- intrinsic calibration matrix, (3,3)
            Wfinal -- list of world-to-camera transforms
            kFinal -- distortion coefficient tuple
        """
        Ainitial, Winitial, kInitial = self.estimateCalibrationParameters(
                allDetections)
        sse, Afinal, Wfinal, kFinal = self.refineCalibrationParameters(
                Ainitial, Winitial, kInitial, allDetections,
                maxIters=maxIters, shouldPrint=True)
        return sse, Afinal, Wfinal, kFinal

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
        Hsref = self._refineHomographies(Hs, allDetections)
        Ainitial = linearcalibrate.computeIntrinsicMatrix(Hsref)
        Winitial = linearcalibrate.computeExtrinsics(Hsref, Ainitial)
        kInitial = self._distortionModel.estimateDistortion(Ainitial, allDetections, Winitial)
        return Ainitial, Winitial, kInitial

    def _refineHomographies(self, Hs, allDetections):
        homographyJac = jacobian.HomographyJacobian()
        Hsref = []
        for H, detections in zip(Hs, allDetections):
            sensorPoints, modelPoints = detections
            Href = self._refineHomography(H, sensorPoints, modelPoints, homographyJac)
            Hsref.append(Href)
        return Hsref

    def _refineHomography(self, H, sensorPoints, modelPoints, jac):
        """
        Use Levenberg-Marquardt nonlinear optimization to refine the homography

        TODO: refactor to DRY out LM (pass in value function)
        """
        ydot = sensorPoints

        ts = time.time()
        λ = 1e-3
        maxIters = 20
        shouldPrint = True
        Pt = H.ravel()
        for iter in range(maxIters):
            J = jac.compute(Pt, modelPoints)

            JTJ = J.T @ J
            diagJTJ = np.diag(np.diagonal(JTJ))
            Ht = Pt.reshape(3,3)
            y = self._projectPointsHomography(Ht, modelPoints)

            # compute residuum
            r = ydot.reshape(-1, 1) - y.reshape(-1, 1)
            Δ = (np.linalg.inv(JTJ + λ*diagJTJ) @ J.T @ r).ravel()

            # evaluate if Pt + Δ reduces the error or not
            Pt_error = self._computeTotalError(ydot, y)

            Pt1 = Pt + Δ
            Ht1 = Pt1.reshape(3,3)
            yt1 = self._projectPointsHomography(Ht1, modelPoints)
            Pt1_error = self._computeTotalError(ydot, yt1)

            if Pt1_error < Pt_error:
                Pt += Δ
                λ /= 10
            else:
                λ *= 10

            if λ < 1e-150 or Pt_error < 1e-12:
                break

        Href = Pt.reshape(3,3)
        Href /= Href[2,2]
        return Href

    def _projectPointsHomography(self, H, modelPoints):
        y = mu.unhom((H @ mu.hom(modelPoints[:,:2]).T).T)
        return y

    def refineCalibrationParameters(self, Ainitial, Winitial, kInitial, allDetections,
            maxIters=100, shouldPrint=False):
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
        self._initializeJacobian()
        Pt = self._composeParameterVector(Ainitial, Winitial, kInitial)
        allModelPoints = [modelPoints for sensorPoints, modelPoints in allDetections]
        ydot = getSensorPoints(allDetections)

        ts = time.time()
        λ = 1e-3
        for iter in range(maxIters):
            J = self._jac.compute(Pt, allModelPoints)

            JTJ = J.T @ J
            diagJTJ = np.diag(np.diagonal(JTJ))
            y = self.projectAllPoints(Pt, allModelPoints)

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
                self._printIterationStats(iter, ts, Pt, min(Pt1_error, Pt_error))

            if λ < 1e-150 or Pt_error < 1e-12:
                break

        Arefined, Wrefined, kRefined = self._decomposeParameterVector(Pt)
        return Pt_error, Arefined, Wrefined, kRefined

    def _initializeJacobian(self):
        # this takes ~7 sec, so only do this once and only when required
        if self._jac is None:
            self._jac = jacobian.ProjectionJacobian(self._distortionModel)

    def _computeReprojectionError(self, P, allDetections):
        allModelPoints = [modelPoints for sensorPoints, modelPoints in allDetections]
        y = self.projectAllPoints(P, allModelPoints)
        ydot = getSensorPoints(allDetections)
        totalError = self._computeTotalError(ydot, y)
        return totalError

    def _computeTotalError(self, ydot, y):
        totalError = np.sum(np.linalg.norm(ydot - y, axis=1)**2)
        return totalError

    def projectAllPoints(self, P, allModelPoints):
        A, W, k = self._decomposeParameterVector(P)
        ydot = np.empty((0, 2))
        for wP, cMw in zip(allModelPoints, W):
            cP = mu.transform(cMw, wP)
            yidot = self._distortionModel.projectWithDistortion(A, cP, k)
            ydot = np.vstack((ydot, yidot))
        return ydot

    def _composeParameterVector(self, A, W, k):
        """
        Input:
            A -- intrinsic matrix
            W -- world-to-camera transforms
            k -- distortion coefficients

        Output:
            P -- vector of all calibration parameters, intrinsic and all M views extrinsic:
                P = (α, β, γ, uc, uv, k[...],
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

        P = mu.col([α, β, γ, uc, vc] + list(k))

        for cMw in W:
            R = cMw[:3,:3]
            t = cMw[:3,3]
            ρix, ρiy, ρiz = mu.rotationMatrixToEuler(R)
            tix, tiy, tiz = t
            P = np.vstack((P, mu.col([ρix, ρiy, ρiz, tix, tiy, tiz])))
        return P

    def _decomposeParameterVector(self, P):
        """
        Input:
            P -- vector of all calibration parameters, intrinsic and all M views extrinsic:
                P = (α, β, γ, uc, uv, k[...],
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
        numIntrinsicParams = 5
        numPoseParams = 6
        if isinstance(P, np.ndarray):
            P = P.ravel()
        numDistortionParameters = len(self._distortionModel.getDistortionSymbols())
        poseStartIndex = numIntrinsicParams + numDistortionParameters
        α, β, γ, uc, vc = P[:numIntrinsicParams]
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

        k = P[numIntrinsicParams:numIntrinsicParams+numDistortionParameters]
        return A, W, k

    def _printIterationStats(self, iter, ts, Pt, error):
        At, Wt, kt = self._decomposeParameterVector(Pt)
        print(f"\niter {iter}: ({time.time() - ts:0.3f}s), error={error:0.3f}")
        print(f"A:\n{At}")
        print(f"k:\n{kt}")


def getSensorPoints(allDetections):
    allSensorPoints = [sensorPoints for sensorPoints, modelPoints in allDetections]
    y = np.empty((0, 2))
    for sensorPoints in allSensorPoints:
        y = np.vstack((y, sensorPoints))
    return y

