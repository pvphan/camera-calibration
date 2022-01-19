import unittest
import warnings
from unittest.mock import MagicMock

import numpy as np

from __context__ import src
from src import linearcalibrate
from src import dataset
from src import mathutils as mu

import cv2


class TestLinearCalibrate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pointsInWorld = np.array([
            [1, -1, 0.4, 1],
            [-1, 1, 0.4, 1],
            [0.3, 0.1, 2.0, 1],
            [0.3, -0.1, 2.0, 1],
            [-0.8, 0.4, 1.2, 1],
            [-0.8, 0.2, 1.2, 1],
        ])
        H1 = np.array([
            [400, 10, 320],
            [20, 400, 240],
            [0, 0, 1],
        ])
        H2 = np.array([
            [300, 15, 320],
            [20, 300, 240],
            [0, 0, 1],
        ])
        H3 = np.array([
            [200, 15, 120],
            [0, 200, 340],
            [0, 0, 1],
        ])
        cls.Hs = [H1, H2, H3]

        A = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])

        width, height = 640, 480
        kExpected = (-0.5, 0.2, 0.07, -0.03, 0.05)
        noiseModel = None
        cls.syntheticDataset = dataset.createSyntheticDatasetRadTan(
                A, width, height, kExpected, noiseModel)
        cls.numIntrinsicParams = 10
        cls.numExtrinsicParamsPerView = 6

    def test_estimateHomography(self):
        numPoints = 10
        X = generateRandomPointsInFrontOfCamera(numPoints)
        X[:,2] = 1
        Hexpected = np.array([
            [410, 10, 320],
            [20, 385, 240],
            [0, 0, 1],
        ])
        x = (Hexpected @ X.T).T
        x = (x / mu.col(x[:,2]))[:,:2]

        Hcomputed = linearcalibrate.estimateHomography(x, X[:,:2])
        #print(Hcomputed - cv2.findHomography(X[:,:2], x)[0])

        self.assertEqual(Hcomputed.shape, (3,3))
        self.assertAllClose(Hcomputed, Hexpected)

    def test_estimateHomography2(self):
        x, X = getExampleData()

        Hcomputed = linearcalibrate.estimateHomography(x, X[:,:2])
        Hexpected, mask = cv2.findHomography(X[:,:2], x)
        print(Hcomputed)
        print(Hexpected)
        print(Hcomputed - Hexpected)
        breakpoint()

        self.assertEqual(Hcomputed.shape, (3,3))

    def test_estimateHomographies(self):
        Aexpected = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1],
        ])
        width, height = 640, 480
        k = (0, 0, 0, 0, 0)
        noiseModel = None
        dataSet = dataset.createSyntheticDatasetRadTan(
                Aexpected, width, height, k, noiseModel)
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()

        Hs = linearcalibrate.estimateHomographies(allDetections)

        self.assertEqual(len(Hs), len(allDetections))
        self.assertEqual(Hs[0].shape, (3,3))

    def test_vecHomog(self):
        expectedShape = (1, 6)
        H = np.array([
            [400, 10, 320],
            [20, 400, 240],
            [0, 0, 1],
        ])

        v1 = linearcalibrate.vecHomography(H, 0, 0)
        v2 = linearcalibrate.vecHomography(H, 0, 1)
        v3 = linearcalibrate.vecHomography(H, 1, 1)

        self.assertEqual(v1.shape, expectedShape)
        self.assertEqual(v2.shape, expectedShape)
        self.assertEqual(v3.shape, expectedShape)

    def test_approximateRotationMatrix(self):
        Q = np.array([
            [0.95, 0, 0],
            [0, 1, -0.05],
            [0, 0, 1.05],
        ])

        R = linearcalibrate.approximateRotationMatrix(Q)

        self.assertAlmostEqual(np.linalg.det(R), 1)

    def test_computeExtrinsics(self):
        A = np.array([
            [420, 0, 327],
            [0, 415, 243],
            [0, 0, 1],
        ])
        width, height = 640, 480
        k = (0, 0, 0, 0, 0)
        noiseModel = None
        dataSet = dataset.createSyntheticDatasetRadTan(
                A, width, height, k, noiseModel)
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        Wexpected = dataSet.getAllBoardPosesInCamera()
        Hs = linearcalibrate.estimateHomographies(allDetections)

        Wcomputed = linearcalibrate.computeExtrinsics(Hs, A)

        self.assertEqual(len(Hs), len(allDetections))
        self.assertAllClose(Wexpected, Wcomputed)

    def test_computeIntrinsicMatrixFrombClosedForm(self):
        Aexpected = np.array([
            [410.05, 0, 320.2],
            [0, 395.34, 240.7],
            [0, 0, 1],
        ])
        b = createbVectorFromIntrinsicMatrix(Aexpected)

        Acomputed = linearcalibrate.computeIntrinsicMatrixFrombClosedForm(b)

        self.assertAllClose(Aexpected, Acomputed)

    def test_computeIntrinsicMatrixFrombCholesky(self):
        Aexpected = np.array([
            [410.05, 0, 320.2],
            [0, 395.34, 240.7],
            [0, 0, 1],
        ])
        b = createbVectorFromIntrinsicMatrix(Aexpected)

        Acomputed = linearcalibrate.computeIntrinsicMatrixFrombCholesky(b)

        self.assertAllClose(Aexpected, Acomputed)

    def test_computeIntrinsicMatrix(self):
        Aexpected = np.array([
            [410.05, 0, 320.2],
            [0, 395.34, 240.7],
            [0, 0, 1],
        ])
        width, height = 640, 480
        k = (0, 0, 0, 0, 0)
        noiseModel = None
        dataSet = dataset.createSyntheticDatasetRadTan(
                Aexpected, width, height, k, noiseModel)
        allDetections = dataSet.getCornerDetectionsInSensorCoordinates()
        Hs = linearcalibrate.estimateHomographies(allDetections)

        A = linearcalibrate.computeIntrinsicMatrix(Hs)

        self.assertAllClose(A, Aexpected)

    def assertAllClose(self, A, B, atol=1e-9):
        self.assertTrue(np.allclose(A, B, atol=atol),
                f"\n{A} \n != \n {B}")


def generateRandomPointsInFrontOfCamera(numPoints):
    np.random.seed(0)
    pointsInCamera = np.zeros((numPoints, 3))
    pointsInCamera[:,0] = np.random.uniform(-1, 1, numPoints)
    pointsInCamera[:,1] = np.random.uniform(-1, 1, numPoints)
    pointsInCamera[:,2] = np.random.uniform(0.5, 1.5, numPoints)
    return pointsInCamera


def createbVectorFromIntrinsicMatrix(A):
    """
    From the relation given by Burger eq 88:

        B = (A^-1)^T * A^-1, where B = [B0 B1 B3]
                                       [B1 B2 B4]
                                       [B3 B4 B5]
    """
    Ainv = np.linalg.inv(A)
    B = Ainv.T @ Ainv
    b = (B[0,0], B[0,1], B[1,1], B[0,2], B[1,2], B[2,2])
    return b


def getExampleData():
    x = np.array([
            [ 394.28674316,   93.32616425],
            [ 438.65335083,  121.87199402],
            [ 486.04974365,  152.33041382],
            [ 536.50982666,  185.19142151],
            [ 590.04205322,  220.4634552 ],
            [ 646.52337646,  258.31115723],
            [ 705.82080078,  298.36209106],
            [ 767.63867188,  340.7232666 ],
            [ 831.55096436,  385.18890381],
            [ 897.08062744,  431.41638184],
            [ 964.05334473,  479.29379272],
            [1031.67663574,  528.32397461],
            [1099.34387207,  578.13494873],
            [1166.35339355,  628.28594971],
            [1231.96838379,  678.29321289],
            [1295.75097656,  727.79058838],
            [ 467.7088623 ,   60.44608688],
            [ 516.44885254,   89.70900726],
            [ 568.3026123 ,  121.08250427],
            [ 623.11657715,  155.20370483],
            [ 681.23095703,  191.859375  ],
            [ 742.15826416,  231.13520813],
            [ 805.59417725,  272.67922974],
            [ 871.15826416,  316.65783691],
            [1076.26342773,  459.88769531],
            [1145.32836914,  510.14538574],
            [1213.62902832,  561.08306885],
            [1280.34594727,  611.96185303],
            [ 657.43585205,   89.27095032],
            [ 716.86816406,  124.69165039],
            [ 779.4128418 ,  162.80223083],
            [ 844.45715332,  203.60295105],
            [ 911.64819336,  246.71156311],
            [1121.2376709 ,  389.51382446],
            [1191.62646484,  440.05291748],
            [1260.99499512,  491.5947876 ],
            [1328.57019043,  543.46472168],
            [ 753.45935059,   57.35271454],
            [ 817.38201904,   94.27061462],
            [ 883.83746338,  134.03848267],
            [ 952.44287109,  176.41668701],
            [1022.59246826,  221.26831055],
            [1094.15100098,  268.60772705],
            [1166.10461426,  317.88900757],
            [1237.54675293,  368.69247437],
            [1307.81384277,  420.72689819],
            [1376.01721191,  473.21408081],
            [ 993.51983643,  105.71081543],
            [1064.91369629,  149.80644226],
            [1137.57067871,  196.56898499],
            [1210.47192383,  245.6917572 ],
            [1282.62805176,  296.46792603],
            [1353.61206055,  348.9644165 ],
            [1106.56567383,   78.88583374],
            [1180.02148438,  125.05670929],
            [1253.57165527,  173.70237732],
            [1326.34289551,  224.43983459],
    ])

    X = np.array([
            [0.03, 0.  , 0.  ],
            [0.06, 0.  , 0.  ],
            [0.09, 0.  , 0.  ],
            [0.12, 0.  , 0.  ],
            [0.15, 0.  , 0.  ],
            [0.18, 0.  , 0.  ],
            [0.21, 0.  , 0.  ],
            [0.24, 0.  , 0.  ],
            [0.27, 0.  , 0.  ],
            [0.3 , 0.  , 0.  ],
            [0.33, 0.  , 0.  ],
            [0.36, 0.  , 0.  ],
            [0.39, 0.  , 0.  ],
            [0.42, 0.  , 0.  ],
            [0.45, 0.  , 0.  ],
            [0.48, 0.  , 0.  ],
            [0.03, 0.03, 0.  ],
            [0.06, 0.03, 0.  ],
            [0.09, 0.03, 0.  ],
            [0.12, 0.03, 0.  ],
            [0.15, 0.03, 0.  ],
            [0.18, 0.03, 0.  ],
            [0.21, 0.03, 0.  ],
            [0.24, 0.03, 0.  ],
            [0.33, 0.03, 0.  ],
            [0.36, 0.03, 0.  ],
            [0.39, 0.03, 0.  ],
            [0.42, 0.03, 0.  ],
            [0.09, 0.06, 0.  ],
            [0.12, 0.06, 0.  ],
            [0.15, 0.06, 0.  ],
            [0.18, 0.06, 0.  ],
            [0.21, 0.06, 0.  ],
            [0.3 , 0.06, 0.  ],
            [0.33, 0.06, 0.  ],
            [0.36, 0.06, 0.  ],
            [0.39, 0.06, 0.  ],
            [0.09, 0.09, 0.  ],
            [0.12, 0.09, 0.  ],
            [0.15, 0.09, 0.  ],
            [0.18, 0.09, 0.  ],
            [0.21, 0.09, 0.  ],
            [0.24, 0.09, 0.  ],
            [0.27, 0.09, 0.  ],
            [0.3 , 0.09, 0.  ],
            [0.33, 0.09, 0.  ],
            [0.36, 0.09, 0.  ],
            [0.15, 0.12, 0.  ],
            [0.18, 0.12, 0.  ],
            [0.21, 0.12, 0.  ],
            [0.24, 0.12, 0.  ],
            [0.27, 0.12, 0.  ],
            [0.3 , 0.12, 0.  ],
            [0.15, 0.15, 0.  ],
            [0.18, 0.15, 0.  ],
            [0.21, 0.15, 0.  ],
            [0.24, 0.15, 0.  ],
    ])
    return x, X


if __name__ == "__main__":
    unittest.main()

