import numpy as np
import sympy

from __context__ import src
from src import distortion
from src import mathutils as mu


class ProjectionJacobian:
    _numExtrinsicParamsPerView = 6
    def __init__(self, distortionModel: distortion.DistortionModel):
        if distortionModel == distortion.DistortionModel.RadialTangential:
            self._uvExpr = createExpressionIntrinsicProjectionRadTan()
            self._intrinsicSymbols = getRadTanSymbols()
            self._intrinsicJacobianBlockExpr = self._createJacobianBlockExpression(self._intrinsicSymbols)
            self._extrinsicSymbols = getExtrinsicSymbols()
            self._extrinsicJacobianBlockExpr = self._createJacobianBlockExpression(self._extrinsicSymbols)
        else:
            raise NotImplementedError("Only radial-tangential distortion supported currently")

    def _createIntrinsicsJacobianBlock(self, intrinsicValues, extrinsicValues, modelPoint):
        intrinsicBlock = np.zeros(shape=self._intrinsicJacobianBlockExpr.shape)
        valuesDict = dict(zip(self._intrinsicSymbols, intrinsicValues))
        #valuesDict.update(dict(zip(self._extrinsicSymbols, extrinsicValues)))
        insertModelPoints(valuesDict, modelPoint)
        for i, symbol in enumerate(self._intrinsicSymbols):
            du = self._intrinsicJacobianBlockExpr[0,i].evalf(subs=valuesDict)
            print(du)
            dv = self._intrinsicJacobianBlockExpr[1,i].evalf(subs=valuesDict)
            intrinsicBlock[0,i] = du
            intrinsicBlock[1,i] = dv
        return intrinsicBlock

    def _createExtrinsicsJacobianBlock(self, intrinsicValues, extrinsicValues, modelPoint):
        extrinsicBlock = np.zeros(shape=self._extrinsicJacobianBlockExpr.shape)
        valuesDict = dict(zip(self._extrinsicSymbols, extrinsicValues))
        insertModelPoints(valuesDict, modelPoint)
        for i, symbol in enumerate(self._extrinsicSymbols):
            du = self._extrinsicJacobianBlockExpr[0,i].evalf(subs=valuesDict)
            dv = self._extrinsicJacobianBlockExpr[1,i].evalf(subs=valuesDict)
            extrinsicBlock[0,i] = du
            extrinsicBlock[1,i] = dv
        return extrinsicBlock

    def _createJacobianBlockExpression(self, derivativeSymbols):
        uExpr, vExpr = self._uvExpr.ravel()
        uExprs = []
        vExprs = []
        for i, paramSymbol in enumerate(derivativeSymbols):
            uExprs.append(sympy.diff(uExpr, paramSymbol))
            vExprs.append(sympy.diff(vExpr, paramSymbol))

        jacobianBlockExpr = np.array([uExprs, vExprs])
        return jacobianBlockExpr

    def compute(self, P, allModelPoints):
        M = len(allModelPoints)
        a = P[:-M * self._numExtrinsicParamsPerView]
        MN = sum([modelPoints.shape[0] for modelPoints in allModelPoints])
        L = len(self._intrinsicSymbols)
        K = L + self._numExtrinsicParamsPerView * M
        J = np.zeros((2*MN, K))

        indexJ = 0
        for i in range(M):
            wX = allModelPoints[i]
            N = len(wX)
            # populate intrinsic columns
            J[indexJ:indexJ + 2*N, :L] = None

            # populate extrinsic blocks

        return J


def insertModelPoints(valuesDict, modelPoint):
    X0, Y0, Z0 = sympy.symbols("X0 Y0 Z0")
    valuesDict[X0] = modelPoint[0]
    valuesDict[Y0] = modelPoint[1]
    valuesDict[Z0] = modelPoint[2]


def createExpressionIntrinsicProjectionRadTan():
    α, β, γ, uc, vc, k1, k2, p1, p2, k3 = getRadTanSymbols()
    A = np.array([
        [α, γ, uc],
        [0, β, vc],
        [0, 0,  1],
    ])
    ρx, ρy, ρz, tx, ty, tz = getExtrinsicSymbols()
    R = mu.eulerToRotationMatrix((ρx, ρy, ρz), isSymbolic=True)
    cMw = np.array([
        [R[0,0], R[0,1], R[0,2], tx],
        [R[1,0], R[1,1], R[1,2], ty],
        [R[2,0], R[2,1], R[2,2], tz],
        [     0,      0,      0,  1],
    ])
    X0, Y0, Z0 = sympy.symbols("X0 Y0 Z0")
    wP = mu.col((X0, Y0, Z0, 1))
    cPHom = (cMw @ wP).T
    X0 = mu.unhom(cPHom)
    k = (k1, k2, p1, p2, k3)
    uvExpr = distortion.projectWithDistortion(A, X0, k, isSymbolic=True)
    return uvExpr


def getRadTanSymbols():
    return sympy.symbols("α β γ uc vc k1 k2 p1 p2 k3")


def getExtrinsicSymbols():
    return sympy.symbols("ρx ρy ρz tx ty tz")
