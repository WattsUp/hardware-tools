"""Test module hardware_tools.extension.bresenham
"""

import time

import numpy as np

from hardware_tools.extensions import bresenham_slow, bresenham_fast

from tests import base


class TestBresenham(base.TestBase):
  """Test bresenham methods
  """

  def _test_draw(self, module):
    shape = self._RNG.integers(10, 100, 2)
    grid = np.zeros(shape=shape, dtype=np.int32)

    x = np.array([shape[0] - 1, 0, 0, shape[0] - 1]).astype(np.int32)
    y = np.array([0, 0, shape[1] - 1, 0]).astype(np.int32)

    module.draw(x, y, grid)
    self.assertEqual(1, grid[0, 0])
    self.assertEqual(1, grid[1, 0])
    self.assertEqual(2, grid[-1, 0])
    self.assertEqual(1, grid[0, -1])

    module.draw(x, y, grid)
    self.assertEqual(2, grid[0, 0])
    self.assertEqual(2, grid[1, 0])
    self.assertEqual(4, grid[-1, 0])
    self.assertEqual(2, grid[0, -1])

    module.draw(-x, -y, grid)
    self.assertEqual(3, grid[0, 0])
    self.assertEqual(2, grid[1, 0])
    self.assertEqual(4, grid[-1, 0])
    self.assertEqual(2, grid[0, -1])

  def test_draw(self):
    self._test_draw(bresenham_slow)
    self._test_draw(bresenham_fast)

    # Validate fast is actually faster
    shape = self._RNG.integers(10, 100, 2)
    grid = np.zeros(shape=shape, dtype=np.int32)

    x = np.array([shape[0] - 1, 0, 0, shape[0] - 1]).astype(np.int32)
    y = np.array([0, 0, shape[1] - 1, 0]).astype(np.int32)
    x = np.repeat(x, 10000)
    y = np.repeat(y, 10000)

    start = time.perf_counter()
    bresenham_slow.draw(x, y, grid)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    bresenham_fast.draw(x, y, grid)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_draw_points(self, module):
    shape = self._RNG.integers(10, 100, 2)
    grid = np.zeros(shape=shape, dtype=np.int32)

    x = np.array([shape[0] - 1, 0, 0, shape[0] - 1]).astype(np.int32)
    y = np.array([0, 0, shape[1] - 1, 0]).astype(np.int32)

    module.draw_points(x, y, grid)
    self.assertEqual(1, grid[0, 0])
    self.assertEqual(0, grid[1, 0])
    self.assertEqual(2, grid[-1, 0])
    self.assertEqual(1, grid[0, -1])

    module.draw_points(x, y, grid)
    self.assertEqual(2, grid[0, 0])
    self.assertEqual(0, grid[1, 0])
    self.assertEqual(4, grid[-1, 0])
    self.assertEqual(2, grid[0, -1])

    module.draw_points(-x, -y, grid)
    self.assertEqual(3, grid[0, 0])
    self.assertEqual(0, grid[1, 0])
    self.assertEqual(4, grid[-1, 0])
    self.assertEqual(2, grid[0, -1])

  def test_draw_points(self):
    self._test_draw_points(bresenham_slow)
    self._test_draw_points(bresenham_fast)

    # Validate fast is actually faster
    shape = self._RNG.integers(10, 100, 2)
    grid = np.zeros(shape=shape, dtype=np.int32)

    x = np.array([shape[0] - 1, 0, 0, shape[0] - 1]).astype(np.int32)
    y = np.array([0, 0, shape[1] - 1, 0]).astype(np.int32)
    x = np.repeat(x, 10000)
    y = np.repeat(y, 10000)

    start = time.perf_counter()
    bresenham_slow.draw_points(x, y, grid)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    bresenham_fast.draw_points(x, y, grid)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)
