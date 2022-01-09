import numpy as np
import sympy

from __context__ import src
from src import distortion


class ProjectionJacobian:
    _numExtrinsicParamsPerView = 6
    def __init__(self, distortionModel: distortion.DistortionModel):
        if distortionModel == distortion.DistortionModel.RadialTangential:
            self._uvExpr = createExpressionIntrinsicProjectionRadTan()
            intrinsicSymbols = getRadTanSymbols()
            self._intrinsicJacobianBlockExpr = self._createJacobianBlockExpression(intrinsicSymbols)
            extrinsicSymbols = getExtrinsicSymbols()
            self._extrinsicJacobianBlockExpr = self._createJacobianBlockExpression(extrinsicSymbols)
        else:
            raise NotImplementedError("Only radial-tangential distortion supported currently")

    def createIntrinsicsJacobianBlock(self):
        self._intrinsicJacobianBlockExpr

    def createExtrinsicsJacobianBlock(self):
        self._extrinsicJacobianBlockExpr

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
        N = len(allModelPoints)
        a = P[:-N * self._numExtrinsicParamsPerView]
        print(a)


def createExpressionIntrinsicProjectionRadTan():
    α, β, γ, uc, vc, k1, k2, p1, p2, k3 = getRadTanSymbols()
    A = np.array([
        [α, γ, uc],
        [0, β, vc],
        [0, 0,  1],
    ])
    X, Y, Z = sympy.symbols("X Y Z")
    X0 = np.array([[X, Y, Z]])
    k = (k1, k2, p1, p2, k3)
    uvExpr = distortion.projectWithDistortion(A, X0, k, isSymbolic=True)
    return uvExpr


def getRadTanSymbols():
    return sympy.symbols("α β γ uc vc k1 k2, p1, p2, k3")


def getExtrinsicSymbols():
    return sympy.symbols("ρx ρy ρz tx ty tz")
