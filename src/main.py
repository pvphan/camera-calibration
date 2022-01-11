"""
The user interface for intrinsic calibration
"""
import numpy as np

from __context__ import src
from src import calibrate
from src import jacobian


def calibrateCamera(allDetections: list[tuple[np.ndarray, np.ndarray]],
        maxIters=50) -> tuple[float, np.ndarray, list[np.ndarray], tuple]:
    """
    Computes the intrinsic matrix and distortion coefficients from a
        set of detections.

    Input:
        allDetections -- list of tuples (one for each view).
                Each tuple is (Xa, Xb), a set of sensor points
                and model points respectively

    Output:
        sse -- final sum squared error
        Afinal -- intrinsic calibration matrix, (3,3)
        Wfinal -- list of world-to-camera transforms
        kFinal -- distortion coefficient tuple of length 5
    """
    jac = jacobian.createJacRadTan()
    Ainitial, Winitial, kInitial = calibrate.estimateCalibrationParameters(
            allDetections)
    sse, Afinal, Wfinal, kFinal = calibrate.refineCalibrationParameters(
            Ainitial, Winitial, kInitial, allDetections, jac,
            maxIters=maxIters, shouldPrint=True)
    return sse, Afinal, Wfinal, kFinal

