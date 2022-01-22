import enum
import numpy as np
import sympy

from __context__ import src
from src import mathutils as mu
from src import symbolic


class DistortionModel:
    def getProjectionExpression(self):
        """
        Creates the base expression for point projection (u, v) from P vector symbols
        and a single world point wP = (X, Y, Z), with
                P = (α, β, γ, uc, uv, k[...], ρx, ρy, ρz, tx, ty, tz)
        """
        isSymbolic = True
        α, β, γ, uc, vc = self.getIntrinsicSymbols()
        A = np.array([
            [α, γ, uc],
            [0, β, vc],
            [0, 0,  1],
        ])
        ρx, ρy, ρz, tx, ty, tz = symbolic.getExtrinsicSymbols()
        R = mu.eulerToRotationMatrix((ρx, ρy, ρz), isSymbolic=isSymbolic)
        cMw = np.array([
            [R[0,0], R[0,1], R[0,2], tx],
            [R[1,0], R[1,1], R[1,2], ty],
            [R[2,0], R[2,1], R[2,2], tz],
            [     0,      0,      0,  1],
        ])
        X, Y, Z = symbolic.getModelPointSymbols()
        wPHom = mu.col((X, Y, Z, 1))
        cPHom = (cMw @ wPHom).T
        cP = mu.unhom(cPHom)
        k = self.getDistortionSymbols()
        uvExpr = self.projectWithDistortion(A, cP, k, isSymbolic=isSymbolic)
        return uvExpr

    def projectWithDistortion(self, A, X, k, isSymbolic=False):
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
        xd = self.distortPoints(x, k, isSymbolic=isSymbolic)
        Ap = A[:2,:3]
        distortedPointsInSensor = (Ap @ mu.hom(xd).T).T
        return distortedPointsInSensor

    def getIntrinsicSymbols(self):
        return tuple(sympy.symbols("α β γ uc vc"))

    def getDistortionSymbols(self):
        raise NotImplementedError()

    def distortPoints(self, x: np.ndarray, k: tuple, isSymbolic=False):
        raise NotImplementedError()

    def estimateDistortion(self, A: np.ndarray, allDetections: list, allBoardPosesInCamera: list):
        raise NotImplementedError()


class RadialTangentialModel(DistortionModel):
    def getDistortionSymbols(self):
        return tuple(sympy.symbols("k1 k2 p1 p2 k3"))

    def distortPoints(self, x: np.ndarray, k: tuple, isSymbolic=False):
        """
        Inputs:
            x -- normalized points (undistorted), (N,2)
            k -- distortion coefficients, (5,)

        Outputs:
            xd -- distorted normalized points (N,2)
        """
        k1, k2, p1, p2, k3 = k

        if isSymbolic:
            xn = x[0,0]
            yn = x[0,1]
            r = sympy.sqrt(xn**2 + yn**2)
        else:
            xn = x[:,0]
            yn = x[:,1]
            r = np.linalg.norm(x, axis=1)

        radialComponent = (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
        tangentialComponentX = (2 * p1 * xn * yn + p2 * (r**2 + 2 * xn**2))
        tangentialComponentY = (p1 * (r**2 + 2 * yn**2) + 2 * p2 * xn * yn)

        xd = radialComponent * xn + tangentialComponentX
        yd = radialComponent * yn + tangentialComponentY

        return np.hstack((mu.col(xd), mu.col(yd)))

    def estimateDistortion(self, A: np.ndarray, allDetections: list, allBoardPosesInCamera: list):
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
        numDistortionParameters = len(self.getDistortionSymbols())
        D = np.empty((0,numDistortionParameters))
        Ddot = np.empty((0,1))

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
                        fx * (rij**2 + 2 * xn**2),
                        (u - uc) * rij**6,
                    ],
                    [
                        (v - vc) * rij**2,
                        (v - vc) * rij**4,
                        fy * (rij**2 + 2 * yn**2),
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


class FisheyeModel(DistortionModel):
    def getDistortionSymbols(self):
        return tuple(sympy.symbols("k1 k2 k3 k4"))

    def distortPoints(self, x: np.ndarray, k: tuple, isSymbolic=False):
        k1, k2, k3, k4 = k

        if isSymbolic:
            xn = x[0,0]
            yn = x[0,1]
            r = sympy.sqrt(xn**2 + yn**2)
            θ = sympy.atan(r)
        else:
            xn = x[:,0]
            yn = x[:,1]
            r = np.linalg.norm(x, axis=1)
            θ = np.arctan(r)

        radialComponent = (θ / r) * (1 + k1 * θ**2 + k2 * θ**4 + k3 * θ**6 + k4 * θ**8)

        xd = radialComponent * xn
        yd = radialComponent * yn

        return np.hstack((mu.col(xd), mu.col(yd)))

    def estimateDistortion(self, A: np.ndarray, allDetections: list, allBoardPosesInCamera: list):
        fx = A[0,0]
        fy = A[1,1]
        uc = A[0,2]
        vc = A[1,2]
        numDistortionParameters = len(self.getDistortionSymbols())
        D = np.empty((0,numDistortionParameters))
        Ddot = np.empty((0,1))

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
                θij = np.arctan(rij)

                Dij = np.array([
                    [
                        fx * (u - uc) * (θij/rij) * θij**2,
                        fx * (u - uc) * (θij/rij) * θij**4,
                        fx * (u - uc) * (θij/rij) * θij**6,
                        fx * (u - uc) * (θij/rij) * θij**8,
                    ],
                    [
                        fy * (v - vc) * (θij/rij) * θij**2,
                        fy * (v - vc) * (θij/rij) * θij**4,
                        fy * (v - vc) * (θij/rij) * θij**6,
                        fy * (v - vc) * (θij/rij) * θij**8,
                    ],
                ])
                D = np.vstack((D, Dij))

                Ddotij = np.array([
                    [udotij - u],
                    [vdotij - v],
                ])
                Ddot = np.vstack((Ddot, Ddotij))
        try:
            k = np.linalg.pinv(D) @ Ddot
        except np.linalg.LinAlgError as e:
            print("Estimating distortion failed, returning 0s")
            return tuple([0] * numDistortionParameters)
        return tuple(k.ravel())

