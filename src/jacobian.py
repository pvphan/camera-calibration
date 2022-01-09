import numpy as np
import sympy

from __context__ import src
from src import distortion
from src import mathutils as mu


class ProjectionJacobian:
    def __init__(self, distortionModel: distortion.DistortionModel):
        # TODO: write expressions for projection
        #       set these up to be evaluated
        pass

    def createIntrinsicsJacobianBlock():
        pass

    def createExtrinsicsJacobianBlock():
        pass

    def compute(P, allModelPoints):
        pass


def createExpressionIntrinsicProjection():
    α, β, γ, uc, vc, k1, k2, p1, p2, k3 = sympy.symbols("α β γ uc vc k1 k2, p1, p2, k3")
    A = np.array([
        [α, γ, uc],
        [0, β, vc],
        [0, 0,  1],
    ])
    X, Y, Z = sympy.symbols("X Y Z")
    X0 = np.array([[X, Y, Z]])
    k = (k1, k2, p1, p2, k3)
    expr = distortion.projectWithDistortion(A, X0, k, isSymbolic=True)
    return expr

