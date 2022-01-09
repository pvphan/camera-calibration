"""
Generates synthetic datasets to test calibration code.
"""
import json
import os

import numpy as np

from __context__ import src
from src import checkerboard
from src import mathutils as mu
from src import virtualcamera
from src import visualize


class Dataset:
    _minDistanceFromBoard = 0.3
    _maxDistanceFromBoard = 0.8
    _rollPitchBounds = (-20, +20)
    _yawBounds = (-180, +180)
    def __init__(self, checkerboard: checkerboard.Checkerboard,
            virtualCamera: virtualcamera.VirtualCamera, numViews: int):
        self._checkerboard = checkerboard
        self._virtualCamera = virtualCamera
        self._allDetections, self._allBoardPosesInCamera = self._computeDetections(numViews)

    def getCornerDetectionsInSensorCoordinates(self):
        return self._allDetections

    def getAllBoardPosesInCamera(self):
        return self._allBoardPosesInCamera

    def getIntrinsicMatrix(self):
        return self._virtualCamera.getIntrinsicMatrix()

    def getDistortionVector(self):
        return self._virtualCamera.getDistortionVector()

    def writeDatasetImages(self, outputFolderPath):
        os.makedirs(outputFolderPath, exist_ok=True)
        w = self._virtualCamera.getImageWidth()
        h = self._virtualCamera.getImageHeight()
        for i, (measuredPointsInSensor, measuredPointsInBoard) in enumerate(self._allDetections):
            outputPath = os.path.join(outputFolderPath, f"{i:03d}.png")
            visualize.writeDetectionsImage(measuredPointsInSensor, w, h, outputPath)

    def _computeDetections(self, numViews: int):
        boardCornerPositions = self._checkerboard.getCornerPositions()
        numBoardCorners = boardCornerPositions.shape[0]
        allDetections = []
        allBoardPosesInCamera = []
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
            allBoardPosesInCamera.append(boardPoseInCamera)
            measuredPointsInSensor, measuredPointsInBoard = (
                    self._virtualCamera.measureDetectedPoints(self._checkerboard,
                            boardPoseInCamera))
            allDetections.append((measuredPointsInSensor, measuredPointsInBoard))
        return allDetections, allBoardPosesInCamera

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

    def exportDetections(self, filePath):
        allDetections = self.getCornerDetectionsInSensorCoordinates()
        detectionsDict = {"views": []}

        for sensorPoints, modelPoints in allDetections:
            view = {
                "sensorPoints": sensorPoints.tolist(),
                "modelPoints": modelPoints.tolist(),
            }
            detectionsDict["views"].append(view)

        with open(filePath, "w") as f:
            f.write(json.dumps(detectionsDict))


def createSyntheticDataset(A, width, height, k):
    checkerBoard = checkerboard.Checkerboard(9, 6, 0.100)
    virtualCamera = virtualcamera.VirtualCamera(A, k, width, height)
    numViews = 10
    dataSet = Dataset(checkerBoard, virtualCamera, numViews)
    return dataSet


def createDetectionsFromPath(filePath):
    with open(filePath, "r") as f:
        detectionsDict = json.load(f)
    allDetections = []
    for view in detectionsDict["views"]:
        sensorPoints = np.array(view["sensorPoints"]).reshape(-1, 2)
        modelPoints = np.array(view["modelPoints"]).reshape(-1, 3)
        allDetections.append((sensorPoints, modelPoints))
    return allDetections

