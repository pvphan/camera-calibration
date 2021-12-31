import unittest

import numpy as np

from src import visualize


class TestVisualize(unittest.TestCase):
    def testdrawCross(self):
        image = visualize.createBlankImage(100, 100)
        point = (25, 50)
        length = 7
        color = (255, 255, 250)
        visualize.drawCross(image, point, length, color)


if __name__ == "__main__":
    unittest.main()

