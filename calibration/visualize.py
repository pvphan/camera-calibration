import numpy as np
from imageio import imwrite

from calibration import calibrate


def drawCrosses(image, points, length, color):
    for point in points:
        drawCross(image, point, length, color)


def drawCross(image, point, length, color):
    h, w = image.shape[:2]
    u, v = np.rint(point).astype(int)
    rowRange = np.arange(v - length//2, v + length//2 + 1)
    rowRange = np.clip(rowRange, 0, h)
    colRange = np.arange(u - length//2, u + length//2 + 1)
    colRange = np.clip(colRange, 0, w)
    image[v, colRange] = color
    image[rowRange, u] = color


def createBlankImage(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


def main():
    pointsInWorld = np.array([
        [1, -1, 0.4, 1],
        [1, -1, 0.4, 1],
        [0.3, -0.1, 2.0, 1],
        [0.3, -0.1, 2.0, 1],
        [-0.8, 0.4, 1.2, 1],
        [-0.8, 0.4, 1.2, 1],
    ])
    w = 720
    h = 480
    K = np.array([
        [450,   0, w//2],
        [  0, 450, h//2],
        [  0,   0,    1],
    ], dtype=np.float64)
    computedPointsInCamera = calibrate.project(K, np.eye(4), pointsInWorld)
    image = createBlankImage(w, h)
    outputPath = "/tmp/output/blank.png"
    imwrite(outputPath, image)


if __name__ == "__main__":
    main()
