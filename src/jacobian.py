import numpy as np
import sympy

from __context__ import src
from src import distortion
from src import mathutils as mu


class ProjectionJacobian:
    """
    Uses sympy to compute expressions and evaluate the partial derivatives
    of the projection function with respect to the variables in the
    parameter vector P (intrinsics, distortion, extrinsics).
    """
    _numExtrinsicParamsPerView = 6
    def __init__(self, distortionModel: distortion.DistortionModel):
        if distortionModel == distortion.DistortionModel.RadialTangential:
            self._uvExpr = createExpressionIntrinsicProjectionRadTan()
            self._intrinsicSymbols = getIntrinsicRadTanSymbols()
        elif distortionModel == distortion.DistortionModel.FishEye:
            raise NotImplementedError("Fish eye distortion model not yet supported")
        else:
            raise NotImplementedError("Unknown distortion model: {distortionModel}")

        self._intrinsicJacobianBlockExpr = self._createJacobianBlockExpression(self._intrinsicSymbols)
        self._intrinsicJacobianBlockFunctions, self._intrinsicJacobianBlockInputSymbols = \
                self._createJacobianBlockFunctions(self._intrinsicJacobianBlockExpr)

        self._extrinsicSymbols = getExtrinsicSymbols()
        self._extrinsicJacobianBlockExpr = self._createJacobianBlockExpression(self._extrinsicSymbols)
        self._extrinsicJacobianBlockFunctions, self._extrinsicJacobianBlockInputSymbols = \
                self._createJacobianBlockFunctions(self._extrinsicJacobianBlockExpr)

    def _createJacobianBlockFunctions(self, jacobianBlockExpr):
        """
        Input:
            jacobianBlockExpr -- (2, T) matrix of the sympy expressions of the partial
                    derivatives of the projection equation

        Output:
            jacobianBlockFunctions -- (2, T) matrix of lambda functions to evaluate
                    the values given inputs
            inputSymbols -- (2, T) matrix of ordered tuples for the input symbols
                    so that values are assigned in the correct order during evaluation

        T is 6 for the extrinsic case, and 10 for the radial-tangential intrinsic case.
        """
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
        return (np.array(jacobianBlockFunctions, dtype=object),
                np.array(inputSymbols, dtype=object))

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
        """
        Evaluates the values of J (general purpose)

        Input:
            valuesDict -- dictionary with items (key=symbol, value=value)
            functionBlock -- (2, T) matrix of callable functions to compute the values of
                    the Jacobian for that specific block
            inputSymbolsBlock -- (2, T) ordered symbols so values are assigned correctly in
                    functionBlock

        Output:
            blockValues -- (2, T) matrix block of the Jacobian J
        """
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
        """
        Input:
            P -- parameter value vector (intrinsics, distortion, extrinsics)
            allModelPoints -- model points which are projected into the camera

        Output:
            J -- Jacobian matrix made up of the partial derivatives of the
                    projection function wrt each of the input parameters in P
        """
        if isinstance(P, np.ndarray):
            P = P.ravel()
        M = len(allModelPoints)
        intrinsicValues = P[:-M * self._numExtrinsicParamsPerView]
        MN = sum([modelPoints.shape[0] for modelPoints in allModelPoints])
        L = len(self._intrinsicSymbols)
        K = L + self._numExtrinsicParamsPerView * M
        J = np.zeros((2*MN, K))

        rowIndexJ = 0
        for i in range(M):
            modelPoints = allModelPoints[i]
            intrinsicStartCol = L+i*self._numExtrinsicParamsPerView
            intrinsicEndCol = intrinsicStartCol + self._numExtrinsicParamsPerView
            extrinsicValues = P[intrinsicStartCol:intrinsicEndCol]
            N = len(modelPoints)
            colIndexJ = L+i*self._numExtrinsicParamsPerView

            # TODO: try to broadcast this
            for j, modelPoint in enumerate(modelPoints):
                intrinsicBlock = self._createIntrinsicsJacobianBlock(
                        intrinsicValues, extrinsicValues, modelPoint)
                extrinsicBlock = self._createExtrinsicsJacobianBlock(
                        intrinsicValues, extrinsicValues, modelPoint)

                J[rowIndexJ:rowIndexJ + 2, :L] = intrinsicBlock
                J[rowIndexJ:rowIndexJ + 2, colIndexJ:colIndexJ+self._numExtrinsicParamsPerView] = \
                        extrinsicBlock
                rowIndexJ += 2
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
    # need a very small number instead of zero to avoid divide-by-zero errors
    eps = 1e-100
    orderedInputs = [valuesDict[symbol] if np.abs(valuesDict[symbol]) > eps else eps
            for symbol in orderedSymbols]
    output = f(*orderedInputs)
    return output
