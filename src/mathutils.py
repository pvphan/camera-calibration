"""
Math utility functions based on lectures from Prof. Daniel Cremers' Multiple View Geometry course.
"""
import math
import numpy as np
import sympy


def radians(angleDegrees):
    return angleDegrees / 180.0 * np.pi


def rotationMatrixToEuler(R):
    """
    Compute the Euler angles from a rotation matrix

    http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    """
    R11, R12, R13, R21, R22, R23, R31, R32, R33 = R.ravel()
    if not (np.isclose(R31, +1) or np.isclose(R31, -1)):
        θ = -np.arcsin(R31)
        ψ = np.arctan2(R32/np.cos(θ), R33/np.cos(θ))
        ϕ = np.arctan2(R21/np.cos(θ), R11/np.cos(θ))
    else:
        ϕ = 0
        if np.isclose(R31, -1):
            θ = np.pi/2
            ψ = ϕ + np.arctan2(R12, R13)
        else:
            θ = -np.pi/2
            ψ = -ϕ + np.arctan2(-R12, -R13)

    return tuple(np.degrees((ψ, θ, ϕ)))


def eulerToRotationMatrix(rXrYrZDegrees, isSymbolic=False):
    rx, ry, rz = rXrYrZDegrees
    wx = col((1, 0, 0))
    wy = col((0, 1, 0))
    wz = col((0, 0, 1))

    if isSymbolic:
        Rx = exp(radians(rx) * skew(wx), isSymbolic=isSymbolic)
        Ry = exp(radians(ry) * skew(wy), isSymbolic=isSymbolic)
        Rz = exp(radians(rz) * skew(wz), isSymbolic=isSymbolic)
    else:
        Rx = exp(np.radians(rx) * skew(wx))
        Ry = exp(np.radians(ry) * skew(wy))
        Rz = exp(np.radians(rz) * skew(wz))
    R = Rz @ Ry @ Rx
    return R


def col(v):
    # create a column vector out of a list / tuple
    return np.array(v).reshape(-1, 1)


def exp(wHat: np.ndarray, isSymbolic=False):
    # exponential mapping of a skew symmetric matrix so(3) onto
    #   the rotation matrix group SO(3), uses Rodrigues' formula
    w = unskew(wHat)
    if isSymbolic:
        wNorm = sympy.sqrt(w[0]**2 + w[1]**2 + w[2]**2)
        I = np.eye(3)
        term1 = (wHat / wNorm) * sympy.sin(wNorm)
        term2 = ((wHat @ wHat) / (wNorm**2)) * (1 - sympy.cos(wNorm))
    else:
        wNorm = np.linalg.norm(w)

        I = np.eye(3)
        if np.isclose(wNorm, 0):
            term1 = 0
        else:
            term1 = (wHat / wNorm) * np.sin(wNorm)
        if np.isclose(wNorm, 0):
            term2 = 0
        else:
            term2 = ((wHat @ wHat) / (wNorm**2)) * (1 - np.cos(wNorm))
    R = I + term1 + term2
    return R


def skew(v):
    # converts vector into a skew symmetric matrix, aka 'hat' operator
    # only works for v of shape (3,1)
    validateShape(v.shape, (3,1))
    a = v[:,0]
    vHat = np.array([
        [    0, -a[2],  a[1]],
        [ a[2],     0, -a[0]],
        [-a[1],  a[0],     0],
    ])
    return vHat


def unskew(vHat):
    validateShape(vHat.shape, (3,3))
    return np.array([vHat[2,1], vHat[0,2], vHat[1,0]])


def validateShape(inputShape, requiredShape):
    for i in range(len(inputShape)):
        if requiredShape[i] is not None and requiredShape[i] != inputShape[i]:
            raise ValueError(f"Expected shape {requiredShape}, got {inputShape}")


def stack(A):
    # stacks the columns of a matrix into a single column
    return col(A.T.ravel())


def unstack(As):
    # unstacks the column vector into a squre matrix
    Nsquared = As.size
    N = int(np.sqrt(Nsquared))
    return As.reshape((N, N)).T


def hom(v):
    if isinstance(v, tuple) or isinstance(v, list):
        v = np.array(v).reshape(-1, len(v))
    elif len(v.shape) == 1:
        return np.append(v, 1)
    # v is Nx2 or Nx3
    vh = np.hstack((v, np.ones((v.shape[0], 1))))
    return vh


def unhom(vHom):
    if len(vHom.shape) == 1:
        v = vHom[:-1] / vHom[-1]
    elif len(vHom.shape) == 2:
        v = vHom[:,:-1] / col(vHom[:,-1])
    else:
        raise ValueError(f"Unexpected input shape for unhom: {vHom.shape}\n{vHom}")
    return v


def normalize(A):
    return A / A.ravel()[-1]


def poseFromRT(R, T):
    world_M_camera = np.eye(4)
    world_M_camera[:3,:3] = R
    if isinstance(T, np.ndarray):
        T = T.ravel()
    world_M_camera[:3,3] = T
    return world_M_camera


def project(A, wMc, wX):
    """
    Inputs:
        A   -- the intrinsic parameter matrix
        wMc -- the camera pose in world
        wX  -- the 3D points in the world

    Outputs:
        xp  -- x' the projected 2D points in the camera
    """
    validateShape(A.shape, (3,3))
    validateShape(wMc.shape, (4,4))
    validateShape(wX.shape, (None,3))

    cMw = np.linalg.inv(wMc)
    X = transform(cMw, wX)
    x = projectStandard(X)
    u = unhom((A @ hom(x).T).T)
    return u


def projectStandard(X):
    """
    Inputs:
        X -- Nx3 points in camera coordinates

    Outputs:
        x -- Nx2 normalized projected 2D points

    Π0            -- standard projection matrix
    """
    validateShape(X.shape, (None,3))

    Π0 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])
    x = unhom((Π0 @ hom(X).T).T)
    return x


def transform(b_M_a, aX):
    """
    Inputs:
        b_M_a -- rigid body transform from coordinates of a to b
        aX -- matrix of Nx3 points in coordinates of a

    Outputs:
        bX -- matrix of Nx3 points in coordinates of b
    """
    validateShape(b_M_a.shape, (4,4))
    validateShape(aX.shape, (None,3))

    bX = unhom((b_M_a @ hom(aX).T).T)
    return bX
