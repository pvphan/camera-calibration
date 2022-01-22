import unittest

import numpy as np

from __context__ import src
from src import visualize


class TestVisualize(unittest.TestCase):
    def test_drawCross(self):
        maxValue = 255
        w = 100
        h = 100
        image = visualize.createBlankImage(w, h)
        point = (25, 50)
        length = 7
        color = (maxValue, maxValue, maxValue)
        id = 0
        self.assertEqual(np.sum(image), 0)
        visualize.drawCross(image, point, length, color, id)
        self.assertEqual(np.sum(image), maxValue * len(color) * (2 * length - 1))

    def test_drawCrossOutOfBounds(self):
        maxValue = 255
        w = 100
        h = 100
        image = visualize.createBlankImage(w, h)
        point = (200, -50)
        length = 7
        color = (maxValue, maxValue, maxValue)
        id = 0
        self.assertEqual(np.sum(image), 0)
        visualize.drawCross(image, point, length, color, id)
        self.assertEqual(np.sum(image), 0)


if __name__ == "__main__":
    unittest.main()

