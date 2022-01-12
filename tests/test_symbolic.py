import unittest

import numpy as np

from __context__ import src
from src import symbolic


class TestSymbolic(unittest.TestCase):
    def test_getModelPointSymbols(self):
        modelPointSymbols = symbolic.getModelPointSymbols()
        self.assertEqual(len(modelPointSymbols), 3)

    def test_getExtrinsicSymbols(self):
        extrinsicSymbols = symbolic.getExtrinsicSymbols()
        self.assertEqual(len(extrinsicSymbols), 6)


if __name__ == "__main__":
    unittest.main()

