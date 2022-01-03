import numpy as np

from __context__ import src
from src import checkerboard
from src import distortion
from src import mathutils as mu


class VirtualCamera:
    def __init__(self, intrinsicMatrix: np.ndarray, distortionVector: tuple,
            imageWidth: int, imageHeight: int):
        self._intrinsicMatrix = intrinsicMatrix
        self._distortionVector = distortionVector
        self._imageWidth = imageWidth
        self._imageHeight = imageHeight

    def getGroundTruthIntrinsicMatrix(self):
        return self._intrinsicMatrix

    def getGroundTruthDistortionVector(self):
        return self._distortionVector

    def getImageWidth(self):
        return self._imageWidth

    def getImageHeight(self):
        return self._imageHeight

    def measureDetectedPoints(self, checkerboard: checkerboard.Checkerboard,
            boardPoseInCamera: np.ndarray):
        cornerPointsInBoard = checkerboard.getCornerPositions()
        camera_M_board = boardPoseInCamera
        cornerPointsInCamera = mu.transform(camera_M_board, cornerPointsInBoard)
        normalizedPoints = mu.unhom(mu.projectStandard(cornerPointsInCamera))

        distortedNormalizedPoints = distortion.distortPoints(normalizedPoints,
                self._distortionVector)
        n = 34
        if normalizedPoints.shape[0] > 34:
            print(cornerPointsInCamera[34])
            print(normalizedPoints[34])
            print(distortedNormalizedPoints[34])

        measuredPoints = (self._intrinsicMatrix @ mu.hom(distortedNormalizedPoints).T).T[:,:2]
        pointInImageSlice = np.s_[
                (measuredPoints[:,0] > 0) & (measuredPoints[:,0] < self._imageWidth)
                & (measuredPoints[:,1] > 0) & (measuredPoints[:,1] < self._imageHeight)
        ]
        print(self._distortionVector, self._imageWidth, self._imageHeight)
        return measuredPoints[pointInImageSlice], cornerPointsInBoard[pointInImageSlice]

