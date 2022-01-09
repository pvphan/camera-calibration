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
            self._intrinsicSymbols = getIntrinsicRadTanSymbols()
            self._intrinsicJacobianBlockExpr = self._createJacobianBlockExpression(self._intrinsicSymbols)
            self._intrinsicJacobianBlockFunctions, self._intrinsicJacobianBlockInputSymbols = \
                    self._createJacobianBlockFunctions(self._intrinsicJacobianBlockExpr)

            self._extrinsicSymbols = getExtrinsicSymbols()
            self._extrinsicJacobianBlockExpr = self._createJacobianBlockExpression(self._extrinsicSymbols)
            self._extrinsicJacobianBlockFunctions, self._extrinsicJacobianBlockInputSymbols = \
                    self._createJacobianBlockFunctions(self._extrinsicJacobianBlockExpr)
        else:
            raise NotImplementedError("Only radial-tangential distortion supported currently")

    def _createJacobianBlockFunctions(self, jacobianBlockExpr):
        inputSymbols = []
        jacobianBlockFunctions = []
        for i in range(jacobianBlockExpr.shape[0]):
            rowInputSymbols = []
            rowJacobianBlockFunctions = []
            for j in range(jacobianBlockExpr.shape[1]):
                expression = jacobianBlockExpr[i, j]
                lambdaFunction, orderedInputSymbosl = createLambdaFunction(expression)
                rowJacobianBlockFunctions.append(lambdaFunction)
                rowInputSymbols.append(orderedInputSymbosl)

            jacobianBlockFunctions.append(rowJacobianBlockFunctions)
            inputSymbols.append(rowInputSymbols)
        return np.array(jacobianBlockFunctions), np.array(inputSymbols)

    def _createIntrinsicsJacobianBlock(self, intrinsicValues, extrinsicValues, modelPoint):
        valuesDict = self._createValuesDict(intrinsicValues, extrinsicValues, modelPoint)
        intrinsicBlock = self._computeJacobianBlock(valuesDict,
                self._intrinsicJacobianBlockFunctions,
                self._intrinsicJacobianBlockInputSymbols)
        return intrinsicBlock

    def _createExtrinsicsJacobianBlock(self, intrinsicValues, extrinsicValues, modelPoint):
        valuesDict = self._createValuesDict(intrinsicValues, extrinsicValues, modelPoint)
        extrinsicBlock = self._computeJacobianBlock(valuesDict,
                self._extrinsicJacobianBlockFunctions,
                self._extrinsicJacobianBlockInputSymbols)
        return extrinsicBlock

    def _computeJacobianBlock(self, valuesDict, functionBlock, inputSymbolsBlock):
        blockValues = np.zeros(shape=functionBlock.shape)
        for i in range(functionBlock.shape[0]):
            for j in range(functionBlock.shape[1]):
                blockValues[i,j] = evaluate(functionBlock[i,j],
                        inputSymbolsBlock[i,j], valuesDict)
        return blockValues

    def _createValuesDict(self, intrinsicValues, extrinsicValues, modelPoint):
        """
        Input:
            intrinsicValues -- α, β, γ, uc, uv, k1, k2, p1, p2, k3
            extrinsicValues -- ρx, ρy, ρz, tx, ty, tz
            modelPoint -- X0, Y0, Z0

        Output:
            valuesDict -- dictionary with items (key=symbol, value=value)
        """
        valuesDict = dict(zip(self._intrinsicSymbols, intrinsicValues))
        valuesDict.update(dict(zip(self._extrinsicSymbols, extrinsicValues)))
        insertModelPoints(valuesDict, modelPoint)
        return valuesDict

    def _createJacobianBlockExpression(self, derivativeSymbols):
        """
        Input:
            derivativeSymbols -- the symbols with which to take the partial derivative
                    of the projection expression (left-to-right along column dimension)

        Output:
            jacobianBlockExpr -- matrix containing the expressions for the partial
                    derivative of the point projection expression wrt the corresponding
                    derivative symbol
        """
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
    X0, Y0, Z0 = getModelPointSymbols()
    valuesDict[X0] = modelPoint[0]
    valuesDict[Y0] = modelPoint[1]
    valuesDict[Z0] = modelPoint[2]


def createExpressionIntrinsicProjectionRadTan():
    """
    Creates the base expression for point projection (u, v) from P vector symbols
    and a single world point wP = (X0, Y0, Z0), with
            P = (α, β, γ, uc, uv, k1, k2, p1, p2, k3, ρx, ρy, ρz, tx, ty, tz)
    """
    isSymbolic = True
    α, β, γ, uc, vc, k1, k2, p1, p2, k3 = getIntrinsicRadTanSymbols()
    A = np.array([
        [α, γ, uc],
        [0, β, vc],
        [0, 0,  1],
    ])
    ρx, ρy, ρz, tx, ty, tz = getExtrinsicSymbols()
    R = mu.eulerToRotationMatrix((ρx, ρy, ρz), isSymbolic=isSymbolic)
    cMw = np.array([
        [R[0,0], R[0,1], R[0,2], tx],
        [R[1,0], R[1,1], R[1,2], ty],
        [R[2,0], R[2,1], R[2,2], tz],
        [     0,      0,      0,  1],
    ])
    X0, Y0, Z0 = getModelPointSymbols()
    wPHom = mu.col((X0, Y0, Z0, 1))
    cPHom = (cMw @ wPHom).T
    cP = mu.unhom(cPHom)
    k = (k1, k2, p1, p2, k3)
    uvExpr = distortion.projectWithDistortion(A, cP, k, isSymbolic=isSymbolic)
    return uvExpr


def getIntrinsicRadTanSymbols():
    return tuple(sympy.symbols("α β γ uc vc k1 k2 p1 p2 k3"))


def getExtrinsicSymbols():
    return tuple(sympy.symbols("ρx ρy ρz tx ty tz"))


def getModelPointSymbols():
    return tuple(sympy.symbols("X0 Y0 Z0"))


def createLambdaFunction(expression):
    orderedSymbols = tuple(expression.atoms(sympy.Symbol))
    f = sympy.lambdify(orderedSymbols, expression, "numpy")
    return f, orderedSymbols


def evaluate(f, orderedSymbols, valuesDict):
    eps = 1e-100
    orderedInputs = [valuesDict[symbol] if np.abs(valuesDict[symbol]) > eps else eps
            for symbol in orderedSymbols]
    output = f(*orderedInputs)
    return output
