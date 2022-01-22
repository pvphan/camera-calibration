"""
Convenience functions to visualize a simple scene of points in a virtual camera.
"""
import cv2
import imageio
import numpy as np

teal = (0, 255, 255)


def drawLine(image, pt1, pt2):
    pass


def writeDetectionsImage(ids, measuredPointsInSensor, w, h, outputPath):
    image = createDetectionsImage(ids, measuredPointsInSensor, w, h)
    imageio.imwrite(outputPath, image)


def createDetectionsImage(ids, measuredPointsInSensor, w, h, color=teal):
    gray = (64, 64, 64)
    image = createBlankImage(w, h, color=gray)
    length = 9
    drawCrosses(image, measuredPointsInSensor, length, color, ids)
    return image


def drawCrosses(image, points, length, color, ids):
    for id, point in zip(ids, points):
        drawCross(image, point, length, color, id)


def drawCross(image, point, length, color, id):
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1
    cv2.putText(image, str(id), (u, v), font, fontScale, color, thickness, cv2.LINE_AA)


def createBlankImage(w, h, color=None):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    if color is not None:
        image[:,:] = color
    return image
