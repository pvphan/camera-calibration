import numpy as np

from src import calibrate


def drawCrosses(image, points, length, color):
    for point in points:
        drawCross(image, point, length, color)


def drawCross(image, point, length, color):
    h, w = image.shape[:2]
    u, v = np.rint(point).astype(int)
    rowRange = np.arange(v - length//2, v + length//2)
    rowRange = np.clip(rowRange, 0, h)
    colRange = np.arange(u - length//2, u + length//2)
    colRange = np.clip(colRange, 0, w)
    image[v, colRange] = color
    image[rowRange, u] = color


def createBlankImage(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


def main():
    pass


if __name__ == "__main__":
    main()
