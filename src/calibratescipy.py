"""
Currently unused refinement using SciPy, kept for posterity.
"""
from scipy.optimize import curve_fit


def refineCalibrationParametersSciPy(Ainitial, Winitial, kInitial, allDetections):
    """
    Input:
        Ainitial -- initial estimate of intrinsic matrix
        Winitial -- initial estimate of world-to-camera transforms
        kInitial -- initial estimate of distortion coefficients
        allDetections -- list of tuples (one for each view).
                Each tuple is (Xa, Xb), a set of sensor points
                and model points respectively

    Output:
        Arefined -- refined estimate of intrinsic matrix
        Wrefined -- refined estimate of world-to-camera transforms
        kRefined -- refined estimate of distortion coefficients

    Uses SciPy non-linear optimization to solve.
    """
    ydata = np.empty((0,1))
    xdataIndex = np.empty((0,4))
    for i, (sensorPoints, modelPoints) in enumerate(allDetections):
        ydata = np.vstack((ydata, sensorPoints.reshape(-1, 1)))
        indexCol = np.tile(i, (modelPoints.shape[0], 1))
        modelPointsWithIndex = np.hstack((modelPoints.reshape(-1, 3), indexCol))
        xdataIndex = np.vstack((xdataIndex, modelPointsWithIndex))

    p0 = composeParameterVector(Ainitial, Winitial, kInitial)
    P, Pcovariance = curve_fit(valueFunctionSciPy, xdataIndex, ydata.ravel(), p0, method='lm')
    Arefined, Wrefined, kRefined = decomposeParameterVector(P)
    return Arefined, Wrefined, kRefined


def valueFunctionSciPy(xdataIndex, *P):
    """
    The function to minimize to refine all calibration parameters.

    Input:
        xdataIndex -- vector of model points with the view index
                appended to the end, (M*N, 4)
        P -- vector of parameters made up of A, W, k

    Output:
        u -- the expected measurements given the input x and the
                calibration parameters P
    """
    A, W, k = decomposeParameterVector(P)
    dataIndices = xdataIndex[:,3].astype(int)
    maxIndex = np.max(dataIndices)
    xdata = []
    for i in range(maxIndex+1):
        slicei = np.s_[dataIndices == i]
        xdatai = xdataIndex[slicei,:3]
        xdata.append(xdatai)

    ydot = np.empty((0, 1))
    for cMw, wP in zip(W, xdata):
        cP = mu.transform(cMw, wP)
        udot = distortion.projectWithDistortion(A, cP, k)
        ydot = np.vstack((ydot, udot.reshape(-1, 1)))
    return ydot.ravel()



