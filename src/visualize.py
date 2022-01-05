"""
Convenience functions to visualize a simple scene of points in a virtual camera.
"""
import imageio
import numpy as np


def writeDetectionsImage(measuredPointsInSensor, w, h, outputPath):
    image = createDetectionsImage(measuredPointsInSensor, w, h)
    imageio.imwrite(outputPath, image)


def createDetectionsImage(measuredPointsInSensor, w, h):
    gray = (64, 64, 64)
    image = createBlankImage(w, h, color=gray)
    length = 9
    teal = (0, 255, 255)
    drawCrosses(image, measuredPointsInSensor, length, teal)
    return image


def drawCrosses(image, points, length, color):
    for point in points:
        drawCross(image, point, length, color)


def drawCross(image, point, length, color):
    h, w = image.shape[:2]
    u, v = np.rint(point).astype(int)
    if not (0 <= u < w and 0 <= v < h):
        return
    rowRange = np.arange(v - length//2, v + length//2 + 1)
    rowRange = np.clip(rowRange, 0, h-1)
    colRange = np.arange(u - length//2, u + length//2 + 1)
    colRange = np.clip(colRange, 0, w-1)
    image[v, colRange] = color
    image[rowRange, u] = color


def createBlankImage(w, h, color=None):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    if color is not None:
        image[:,:] = color
    return image
