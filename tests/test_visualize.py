import unittest

import numpy as np

from calibration import visualize


class TestVisualize(unittest.TestCase):
    def testdrawCross(self):
        maxValue = 255
        w = 100
        h = 100
        image = visualize.createBlankImage(w, h)
        point = (25, 50)
        length = 7
        color = (maxValue, maxValue, maxValue)
        self.assertEqual(np.sum(image), 0)
        visualize.drawCross(image, point, length, color)
        self.assertEqual(np.sum(image), maxValue * len(color) * (2 * length - 1))


if __name__ == "__main__":
    unittest.main()

