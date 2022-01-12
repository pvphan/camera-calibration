"""
The user interface for intrinsic calibration
"""
import numpy as np

from __context__ import src
from src import calibrate
from src import distortion


def calibrateCamera(allDetections: list, distortionType: str, maxIters=50) -> tuple:
    """
    Computes the intrinsic matrix, distortion coefficients,
        and board poses in camera coordinates from a set of detections.

    Input:
        allDetections -- list of tuples (one for each view).
                Each tuple is (Xa, Xb), a set of sensor points
                and model points respectively
        distortionType -- one of ["radtan", "fisheye"]

    Output:
        sse -- final sum squared error
        Afinal -- intrinsic calibration matrix, (3,3)
        Wfinal -- list of world-to-camera transforms
        kFinal -- distortion coefficient tuple of length 5
    """
    if distortionType == "radtan":
        distortionModel = distortion.RadialTangentialModel()
    elif distortionType == "fisheye":
        distortionModel = distortion.FisheyeModel()
    else:
        raise ValueError(f"Distortion type: {distortionType} unknown")
    calibrator = calibrate.Calibrator(distortionModel)
    sse, Afinal, Wfinal, kFinal = calibrator.calibrate(allDetections, maxIters)
    return sse, Afinal, Wfinal, kFinal

