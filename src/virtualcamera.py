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

    def measureBoardPoints(self, checkerboard: checkerboard.Checkerboard,
            boardPoseInCamera: np.ndarray):
        wP = checkerboard.getCornerPositions()
        cMw = boardPoseInCamera
        return self.measurePoints(cMw, wP)

    def measurePoints(self, cMw, wP):
        cP = mu.transform(cMw, wP)
        u = distortion.projectWithDistortion(self._intrinsicMatrix,
                cP, self._distortionVector)

        pointInImageSlice = np.s_[
                (u[:,0] > 0) & (u[:,0] < self._imageWidth)
                & (u[:,1] > 0) & (u[:,1] < self._imageHeight)
        ]
        return u[pointInImageSlice], wP[pointInImageSlice]

