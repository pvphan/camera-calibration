import numpy as np
from imageio import imwrite

from context import src
from src import calibrate


def drawCrosses(image, points, length, color):
    for point in points:
        drawCross(image, point, length, color)


def drawCross(image, point, length, color):
    h, w = image.shape[:2]
    u, v = np.rint(point).astype(int)
    print(u, v)
    if 0 < u < w and 0 < v < h:
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
        [0.15, -0.1, 0.4, 1],
        [0.35, 0.1, 1.4, 1],
        [0.3, -0.1, 2.0, 1],
        [0.2, 0.1, 2.0, 1],
        [0.6, 0.4, 1.2, 1],
        [-0.6, 0.4, 1.2, 1],
    ])
    focalLength = 500
    w = 640
    h = 480
    K = np.array([
        [focalLength,   0, w//2],
        [  0, focalLength, h//2],
        [  0,   0,    1],
    ], dtype=np.float64)
    computedPointsInCamera = calibrate.project(K, np.eye(4), pointsInWorld)
    image = createBlankImage(w, h)
    length = 7
    color = (255, 255, 0)
    drawCrosses(image, computedPointsInCamera[:,:2], length, color)
    outputPath = "/tmp/output/crosses.png"
    imwrite(outputPath, image)


if __name__ == "__main__":
    main()
