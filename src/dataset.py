"""
Generates synthetic datasets to test calibration code.
"""

import numpy as np

from __context__ import src
from src import checkerboard
from src import mathutils as mu
from src import virtualcamera


class Dataset:
    _minDistanceFromBoard = 0.3
    _maxDistanceFromBoard = 0.8
    _rollPitchBounds = (-20, +20)
    _yawBounds = (-180, +180)
    def __init__(self, checkerboard: checkerboard.Checkerboard,
            virtualCamera: virtualcamera.VirtualCamera, numViews: int):
        self._checkerboard = checkerboard
        self._virtualCamera = virtualCamera
        self._allDetections = self._computeDetections(numViews)

    def getCornerDetectionsInSensorCoordinates(self):
        return self._allDetections

    def _computeDetections(self, numViews: int):
        boardCornerPositions = self._checkerboard.getCornerPositions()
        numBoardCorners = boardCornerPositions.shape[0]
        allDetections = []
        for viewIndex in range(numViews):
            np.random.seed(viewIndex)
            cornerIndexToPointAt = np.random.choice(numBoardCorners)
            rx = np.random.uniform(*self._rollPitchBounds)
            ry = np.random.uniform(*self._rollPitchBounds)
            rz = np.random.uniform(*self._yawBounds)
            distanceFromBoard = np.random.uniform(self._minDistanceFromBoard,
                    self._maxDistanceFromBoard)
            rotationEulerAngles = (rx, ry, rz)
            boardPositionToAimAt = boardCornerPositions[cornerIndexToPointAt]
            cameraPoseInBoard = self._computeCameraPoseInBoard(
                    boardPositionToAimAt, rotationEulerAngles, distanceFromBoard)

            boardPoseInCamera = np.linalg.inv(cameraPoseInBoard)
            measuredPointsInSensor, measuredPointsInBoard = (
                    self._virtualCamera.measureDetectedPoints(self._checkerboard,
                            boardPoseInCamera))
            allDetections.append((measuredPointsInSensor, measuredPointsInBoard))
        return allDetections

    def _computeCameraPoseInBoard(self, boardPositionToAimAt, rotationEulerAngles,
            distanceFromBoard):
        cameraPerturbationRotation = mu.eulerToRotationMatrix(rotationEulerAngles)
        cameracoincident_M_cameraperturb = mu.poseFromRT(cameraPerturbationRotation, (0, 0, 0))
        cameraFacingBoardRotation = mu.eulerToRotationMatrix((180, 0, 0))
        board_M_cameracoincident = mu.poseFromRT(
                cameraFacingBoardRotation, boardPositionToAimAt)
        cameraperturb_M_camera = mu.poseFromRT(np.eye(3), (0, 0, -distanceFromBoard))
        board_M_camera = (board_M_cameracoincident
                @ cameracoincident_M_cameraperturb
                @ cameraperturb_M_camera)
        return board_M_camera
