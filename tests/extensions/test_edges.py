"""Test module hardware_tools.extension.edges
"""

import time

import numpy as np

from hardware_tools.extensions import edges_slow, edges_fast

from tests import base


class TestEdges(base.TestBase):
  """Test edges methods
  """

  def _test_get_crossing(self, module):
    axis_return = [0, 1, 2, 3, 4]
    axis_search = [10, 20, -10, 50, -50]

    value, index = module.get_crossing(axis_return, axis_search, 1, 0)
    self.assertTrue(np.isnan(value))
    self.assertEqual(0, index)

    value, index = module.get_crossing(axis_return, axis_search, 2, 0)
    self.assertAlmostEqual(1 + 2 / 3, value)
    self.assertEqual(2, index)

    value, index = module.get_crossing(axis_return, axis_search, 3, 0)
    self.assertAlmostEqual(2 + 1 / 6, value)
    self.assertEqual(3, index)

    value, index = module.get_crossing(axis_return, axis_search, 4, 0)
    self.assertAlmostEqual(3.5, value)
    self.assertEqual(4, index)

  def test_get_crossing(self):
    self._test_get_crossing(edges_slow)
    self._test_get_crossing(edges_fast)

    # Validate fast is actually faster
    n = 10000
    axis_return = [0, 1, 2, 3, 4]
    axis_search = [10, 20, -10, 50, -50]
    i = 4
    value = 0

    start = time.perf_counter()
    for _ in range(n):
      _ = edges_slow.get_crossing(axis_return, axis_search, i, value, False, n)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n):
      _ = edges_fast.get_crossing(axis_return, axis_search, i, value, False, n)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_get(self, module):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)
    t = t.tolist()
    y = y.tolist()

    edges_rise, edges_fall = module.get(t, y, 0.1, 0.0, -0.1)
    for i in range(len(edges_rise)):
      self.assertEqualWithinError(i + 0.75, edges_rise[i], 0.01)
    for i in range(len(edges_fall)):
      self.assertEqualWithinError(i + 0.25, edges_fall[i], 0.01)

  def test_get(self):
    self._test_get(edges_slow)
    self._test_get(edges_fast)

    # Validate fast is actually faster

    t = np.linspace(0, 10000, 100000)
    y = np.cos(t * 2 * np.pi)
    t = t.tolist()
    y = y.tolist()

    start = time.perf_counter()
    _ = edges_slow.get(t, y, 0.1, 0.0, -0.1)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    _ = edges_fast.get(t, y, 0.1, 0.0, -0.1)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_get_np(self, module):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)

    edges_rise, edges_fall = module.get_np(t, y, 0.1, 0.0, -0.1)
    for i in range(len(edges_rise)):
      self.assertEqualWithinError(i + 0.75, edges_rise[i], 0.01)
    for i in range(len(edges_fall)):
      self.assertEqualWithinError(i + 0.25, edges_fall[i], 0.01)

  def test_get_np(self):
    self._test_get_np(edges_slow)
    self._test_get_np(edges_fast)

    # Validate fast is actually faster
    t = np.linspace(0, 10000, 100000)
    y = np.cos(t * 2 * np.pi)

    start = time.perf_counter()
    _ = edges_slow.get_np(t, y, 0.1, 0.0, -0.1)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    _ = edges_fast.get_np(t, y, 0.1, 0.0, -0.1)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)
