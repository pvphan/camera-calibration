import numpy as np

from __context__ import src
from src import calibrate
from src import jacobian


def calibrateCamera(allDetections: list[tuple[np.ndarray, np.ndarray]],
        maxIters=50) -> tuple[np.ndarray, tuple]:
    """
    Computes the intrinsic matrix and distortion coefficients from a
        set of detections.

    Input:
        allDetections -- list of tuples (one for each view).
                Each tuple is (Xa, Xb), a set of sensor points
                and model points respectively

    Output:
        Afinal -- intrinsic calibration matrix, (3,3)
        kFinal -- distortion coefficient tuple, (5,)
    """
    jac = jacobian.createJacRadTan()
    Ainitial, Winitial, kInitial = calibrate.estimateCalibrationParameters(
            allDetections)
    Afinal, _, kFinal = calibrate.refineCalibrationParameters(
            Ainitial, Winitial, kInitial, allDetections, jac,
            maxIters=maxIters, shouldPrint=True)
    return Afinal, kFinal

