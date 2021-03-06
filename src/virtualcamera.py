import numpy as np

from __context__ import src
from src import checkerboard
from src import distortion
from src import mathutils as mu
from src import noise


class VirtualCamera:
    def __init__(self, intrinsicMatrix: np.ndarray, distortionVector: tuple,
            distortionModel: distortion.DistortionModel, imageWidth: int, imageHeight: int,
            noiseModel: noise.NoiseModel):
        self._intrinsicMatrix = intrinsicMatrix
        self._distortionVector = distortionVector
        self._distortionModel = distortionModel
        self._imageWidth = imageWidth
        self._imageHeight = imageHeight
        self._noiseModel = noiseModel

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
        u, pointInImageSlice = self._measurePoints(cMw, wP)
        numCorners = wP.shape[0]
        ids = np.arange(numCorners)
        return ids[pointInImageSlice], u[pointInImageSlice], wP[pointInImageSlice]

    def _measurePoints(self, cMw, wP):
        cP = mu.transform(cMw, wP)
        u = self._distortionModel.projectWithDistortion(self._intrinsicMatrix,
                cP, self._distortionVector)

        if self._noiseModel is not None:
            u = self._noiseModel.applyNoise(u)

        pointInImageSlice = np.s_[
                (u[:,0] > 0) & (u[:,0] < self._imageWidth)
                & (u[:,1] > 0) & (u[:,1] < self._imageHeight)
                & (cP[:,2] > 0)
        ]
        return u, pointInImageSlice

