"""
Generates synthetic datasets to test calibration code.
"""

import numpy as np


class Checkerboard:
    def __init__(self, numCornersWidth, numCornersHeight, spacing):
        self._cornerPositions = self._createCornerPositions(
                numCornersWidth, numCornersHeight, spacing)

    def _createCornerPositions(self, numCornersWidth,
            numCornersHeight, spacing) -> np.ndarray:
        cornerPositions = []
        for j in range(numCornersHeight):
            for i in range(numCornersWidth):
                x = i * spacing
                y = j * spacing
                cornerPositions.append((x, y, 0))
        return np.array(cornerPositions).reshape(-1, 3)

    def getCornerPositions(self) -> np.ndarray:
        return self._cornerPositions


class VirtualCamera:
    def __init__(self, intrinsicMatrix: np.ndarray, distortionVector: tuple):
        self._intrinsicMatrix = intrinsicMatrix
        self._distortionVector = distortionVector

    def getGroundTruthIntrinsicMatrix(self):
        return self._intrinsicMatrix

    def getGroundTruthDistortionVector(self):
        return self._distortionVector

    def measureDetectedPoints(self, checkerboard: Checkerboard, boardPoseInCamera: np.ndarray):
        """
        """
        pass


class Dataset:
    def __init__(self, checkerboard: Checkerboard, virtualCamera: VirtualCamera):
        self._checkerboard = checkerboard
        self._virtualCamera = virtualCamera

    def getCornerDetectionsInSensorCoordinates(self, numViews: int):
        boardCornerPositions = self._checkerboard.getCornerPositions()
        numBoardCorners = boardCornerPositions.shape[0]
        for viewIndex in range(numViews):
            np.random.seed(viewIndex)
            # perturb the board pose randomly
            # choose a random point on the board
            cornerIndexToPointAt = np.random.choice(numBoardCorners)

            self._computeCameraPoseInBoard(cornerIndexToPointAt)

            # record the corner detections
            self._virtualCamera.measureDetectedPoints(self._checkerboard, boardPoseInCamera)

    def _computeCameraPoseInBoard(self, cornerIndexToPointAt, rotationEulerAngles):
        # align camera to point facing the board, axis aligned
        # random rotations
        # position camera a set distance from that point
        pass
