"""Test module hardware_tools.extension.bresenham
"""

import unittest

import numpy as np

from hardware_tools.extensions import bresenham_slow, bresenham_fast


class TestBresenham(unittest.TestCase):
  """Test bresenham methods
  """

  def test_slow(self):
    shape = np.random.randint(10, 100, 2)
    grid = np.zeros(shape=shape, dtype=np.int32)

    x = np.array([shape[0] - 1, 0, 0, shape[0] - 1])
    y = np.array([0, 0, shape[1] - 1, 0])

    bresenham_slow.draw(x, y, grid)
    self.assertEqual(1, grid[0, 0])
    self.assertEqual(2, grid[-1, 0])
    self.assertEqual(1, grid[0, -1])

    bresenham_slow.draw(x, y, grid)
    self.assertEqual(2, grid[0, 0])
    self.assertEqual(4, grid[-1, 0])
    self.assertEqual(2, grid[0, -1])

    bresenham_slow.draw(-x, -y, grid)
    self.assertEqual(3, grid[0, 0])
    self.assertEqual(4, grid[-1, 0])
    self.assertEqual(2, grid[0, -1])

    # x = np.repeat(x, 10000)
    # y = np.repeat(y, 10000)
    # import time
    # start = time.time()
    # bresenham_slow.draw(x, y, grid)
    # elapsed = time.time() - start
    # print(f"bresenham_slow = {elapsed}")

  def test_fast(self):
    shape = np.random.randint(10, 100, 2)
    grid = np.zeros(shape=shape, dtype=np.int32)

    x = np.array([shape[0] - 1, 0, 0, shape[0] - 1])
    y = np.array([0, 0, shape[1] - 1, 0])

    bresenham_fast.draw(x, y, grid)
    self.assertEqual(1, grid[0, 0])
    self.assertEqual(2, grid[-1, 0])
    self.assertEqual(1, grid[0, -1])

    bresenham_fast.draw(x, y, grid)
    self.assertEqual(2, grid[0, 0])
    self.assertEqual(4, grid[-1, 0])
    self.assertEqual(2, grid[0, -1])

    bresenham_fast.draw(-x, -y, grid)
    self.assertEqual(3, grid[0, 0])
    self.assertEqual(4, grid[-1, 0])
    self.assertEqual(2, grid[0, -1])

    # x = np.repeat(x, 10000)
    # y = np.repeat(y, 10000)
    # import time
    # start = time.time()
    # bresenham_fast.draw(x, y, grid)
    # elapsed = time.time() - start
    # print(f"bresenham_fast = {elapsed}")
