import numpy as np

from __context__ import src
from src import checkerboard
from src import distortion
from src import mathutils as mu


class VirtualCamera:
    """
    An ideal camera with radial-tangential distortion.
    """
    def __init__(self, intrinsicMatrix: np.ndarray, distortionVector: tuple,
            imageWidth: int, imageHeight: int):
        self._intrinsicMatrix = intrinsicMatrix
        self._distortionVector = distortionVector
        self._imageWidth = imageWidth
        self._imageHeight = imageHeight

    def getIntrinsicMatrix(self):
        return self._intrinsicMatrix

    def getDistortionVector(self):
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
        distortedPointsInSensor = distortion.projectWithDistortion(self._intrinsicMatrix,
                cornerPointsInCamera, self._distortionVector)

        pointInImageSlice = np.s_[
                (distortedPointsInSensor[:,0] > 0) & (distortedPointsInSensor[:,0] < self._imageWidth)
                & (distortedPointsInSensor[:,1] > 0) & (distortedPointsInSensor[:,1] < self._imageHeight)
        ]
        return distortedPointsInSensor[pointInImageSlice], cornerPointsInBoard[pointInImageSlice]

