from __context__ import src
from src import calibrate
from src import jacobian


def calibrateCamera(allDetections: list, maxIters=50):
    """
    Input:
        allDetections -- list of tuples (one for each view).
                Each tuple is (Xa, Xb), a set of sensor points
                and model points respectively
    """
    jac = jacobian.createJacRadTan()
    Ainitial, Winitial, kInitial = calibrate.estimateCalibrationParameters(
            allDetections)
    Arefined, Wrefined, kRefined = calibrate.refineCalibrationParameters(
            Ainitial, Winitial, kInitial, jac, maxIters=maxIters, shouldPrint=True)
    return Arefined, Wrefined, kRefined

