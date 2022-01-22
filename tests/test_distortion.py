import unittest

import cv2
import numpy as np

from __context__ import src
from src import dataset
from src import distortion
from src import mathutils as mu


class TestCommon(unittest.TestCase):
    def assertAllClose(self, A, B, atol=1e-9):
        self.assertTrue(np.allclose(A, B, atol=atol),
                f"\n{A} \n != \n {B}")


class TestRadialTangentialModel(TestCommon):
    @classmethod
    def setUpClass(cls):
        cls.pointsInWorld = np.array([
            [1, -1, 0.4],
            [-1, 1, 0.4],
            [0.3, 0.1, 2.0],
            [0.3, -0.1, 2.0],
            [-0.8, 0.4, 1.2],
            [-0.8, 0.2, 1.2],
        ])
        cls.distortionModel = distortion.RadialTangentialModel()
        fx = 1380.5
        fy = 1410.2
        A = np.array([
            [fx, 0, 715.9],
            [0, fy, 539.3],
            [0, 0, 1],
        ], dtype=np.float64)
        k = (-0.5, 0.2, 0.005, -0.03, 0.05)
        width, height = 1440, 1080
        noiseModel = None
        cls.syntheticDataset = dataset.createSyntheticDatasetRadTan(
                A, width, height, k, noiseModel)

    def test_createExpressionIntrinsicProjection(self):
        expr = self.distortionModel.getProjectionExpression()
        self.assertNotEqual(str(expr), "None")
        self.assertEqual(expr.shape, (1,2))

    def test_distortPoints(self):
        k1 = -0.5
        k2 = 0.2
        p1 = 0
        p2 = 0
        k3 = 0
        k = (k1, k2, p1, p2, k3)

        normalizedPointsNx2 = mu.projectStandard(self.pointsInWorld)
        distortedPoints = self.distortionModel.distortPoints(normalizedPointsNx2, k)

        self.assertEqual(distortedPoints.shape, normalizedPointsNx2.shape)
        self.assertEqual(normalizedPointsNx2.shape, (distortedPoints.shape[0], 2))
        self.assertFalse(np.allclose(normalizedPointsNx2, distortedPoints))

    def test_projectWithDistortion(self):
        chosenIndex = 8
        realisticDataset = dataset.createRealisticRadTanDataset()
        allDetections = realisticDataset.getCornerDetectionsInSensorCoordinates()
        allBoardPoses = realisticDataset.getAllBoardPosesInCamera()
        wP = allDetections[chosenIndex][1]
        cMw = allBoardPoses[chosenIndex]
        cP = mu.transform(cMw, wP)
        A = realisticDataset.getIntrinsicMatrix()
        k = realisticDataset.getDistortionVector()

        projectedPoints = self.distortionModel.projectWithDistortion(A, cP, k)

        rvec = (0, 0, 0)
        tvec = (0, 0, 0)
        projectedPointsOpencv = cv2.projectPoints(cP, rvec, tvec, A, k)[0].reshape(-1, 2)

        self.assertEqual(projectedPoints.shape, (cP.shape[0], 2))
        self.assertFalse(np.isnan(np.sum(projectedPoints)))
        self.assertAllClose(projectedPointsOpencv, projectedPoints)

    def test_estimateDistortion(self):
        dataSet = self.syntheticDataset
        A = dataSet.getIntrinsicMatrix()
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        allBoardPosesInCamera = dataSet.getAllBoardPosesInCamera()
        kExpected = dataSet.getDistortionVector()

        kComputed = self.distortionModel.estimateDistortion(A, allDetections, allBoardPosesInCamera)

        self.assertAllClose(kExpected, kComputed)


class TestFisheyeModel(TestCommon):
    @classmethod
    def setUpClass(cls):
        cls.pointsInWorld = np.array([
            [1, -1, 0.4],
            [-1, 1, 0.4],
            [0.3, 0.1, 2.0],
            [0.3, -0.1, 2.0],
            [-0.8, 0.4, 1.2],
            [-0.8, 0.2, 1.2],
        ])
        cls.distortionModel = distortion.FisheyeModel()
        cls.A = np.array([
            [1410.2, 0, 715.8],
            [0, 1410.2, 539.3],
            [0, 0, 1],
        ], dtype=np.float64)
        cls.k = (-0.126, 0.004, 0.0, 0.0)
        noiseModel = None
        width, height = 1440, 1080
        cls.syntheticDataset = dataset.createSyntheticDatasetFisheye(
                cls.A, width, height, cls.k, noiseModel)

    def test_createExpressionIntrinsicProjection(self):
        expr = self.distortionModel.getProjectionExpression()
        self.assertNotEqual(str(expr), "None")
        self.assertEqual(expr.shape, (1,2))

    def test_distortPoints(self):
        normalizedPointsNx2 = mu.projectStandard(self.pointsInWorld)
        distortedPoints = self.distortionModel.distortPoints(normalizedPointsNx2, self.k)

        self.assertEqual(distortedPoints.shape, normalizedPointsNx2.shape)
        self.assertEqual(normalizedPointsNx2.shape, (distortedPoints.shape[0], 2))
        self.assertFalse(np.allclose(normalizedPointsNx2, distortedPoints))

    def test_projectWithDistortion(self):
        isSymbolic = False
        projectedPoints = self.distortionModel.projectWithDistortion(self.A, self.pointsInWorld, self.k,
                isSymbolic=isSymbolic)

        rvec = (0, 0, 0)
        tvec = (0, 0, 0)
        projectedPointsOpencv = cv2.fisheye.projectPoints(
                self.pointsInWorld.reshape(-1, 1, 3), rvec, tvec,
                self.A, self.k)[0].reshape(-1, 2)
        self.assertAllClose(projectedPointsOpencv, projectedPoints)

        self.assertEqual(projectedPoints.shape, (self.pointsInWorld.shape[0], 2))
        self.assertFalse(np.isnan(np.sum(projectedPoints)))

    def test_estimateDistortion(self):
        dataSet = self.syntheticDataset
        A = dataSet.getIntrinsicMatrix()
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        allBoardPosesInCamera = dataSet.getAllBoardPosesInCamera()
        kExpected = dataSet.getDistortionVector()

        #width = dataSet.getImageWidth()
        #height = dataSet.getImageHeight()
        #imageSize = (width, height)
        #objectPoints = [xyz.reshape(-1, 1, 3) for uv, xyz in allDetections]
        #imagePoints = [uv.reshape(-1, 1, 2) for uv, xyz in allDetections]

        #retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        #        objectPoints, imagePoints,
        #        imageSize, None, None)

        kComputed = self.distortionModel.estimateDistortion(A, allDetections, allBoardPosesInCamera)

        #self.assertAllClose(kExpected, kComputed)


if __name__ == "__main__":
    unittest.main()

