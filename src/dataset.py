"""
Generates synthetic datasets to test calibration code.
"""

import numpy as np

from __context__ import src
from src import mathutils as mu
from src import calibrate


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

    def measureDetectedPoints(self, checkerboard: Checkerboard, boardPoseInCamera: np.ndarray):
        cornerPointsInBoard = checkerboard.getCornerPositions()
        camera_M_board = boardPoseInCamera
        cornerPointsInCamera = (camera_M_board @ mu.homog(cornerPointsInBoard).T).T
        projectedPointsInSensor = calibrate.project(
                self._intrinsicMatrix, np.eye(4), cornerPointsInCamera)
        measuredPoints = projectedPointsInSensor[:,:2]
        pointInImageSlice = np.s_[
                (measuredPoints[:,0] > 0) & (measuredPoints[:,0] < self._imageWidth)
                & (measuredPoints[:,1] > 0) & (measuredPoints[:,1] < self._imageHeight)
        ]
        return measuredPoints[pointInImageSlice], cornerPointsInBoard[pointInImageSlice]


class Dataset:
    _minDistanceFromBoard = 0.3
    _maxDistanceFromBoard = 0.8
    _rollPitchBounds = (-20, +20)
    _yawBounds = (-180, +180)
    def __init__(self, checkerboard: Checkerboard, virtualCamera: VirtualCamera):
        self._checkerboard = checkerboard
        self._virtualCamera = virtualCamera

    def getCornerDetectionsInSensorCoordinates(self, numViews: int):
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
            self._computeCameraPoseInBoard(boardPositionToAimAt, rotationEulerAngles,
                    distanceFromBoard)

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
