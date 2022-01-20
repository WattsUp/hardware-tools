"""Test module hardware_tools.extension.edges
"""

import numpy as np

from hardware_tools.extensions import edges_slow, edges_fast

from tests import base


class TestEdges(base.TestBase):
  """Test edges methods
  """

  def test_slow(self):
    axis_return = [0, 1, 2, 3, 4]
    axis_search = [10, 20, -10, 50, -50]

    value, index = edges_slow.get_crossing(axis_return, axis_search, 1, 0)
    self.assertTrue(np.isnan(value))
    self.assertEqual(0, index)

    value, index = edges_slow.get_crossing(axis_return, axis_search, 2, 0)
    self.assertAlmostEqual(1 + 2 / 3, value)
    self.assertEqual(2, index)

    value, index = edges_slow.get_crossing(axis_return, axis_search, 3, 0)
    self.assertAlmostEqual(2 + 1 / 6, value)
    self.assertEqual(3, index)

    value, index = edges_slow.get_crossing(axis_return, axis_search, 4, 0)
    self.assertAlmostEqual(3.5, value)
    self.assertEqual(4, index)

    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)
    edges_rise, edges_fall = edges_slow.get_np(t, y, 0.1, 0.0, -0.1)
    for i in range(len(edges_rise)):
      self.assertEqualWithinError(i + 0.75, edges_rise[i], 0.01)
    for i in range(len(edges_fall)):
      self.assertEqualWithinError(i + 0.25, edges_fall[i], 0.01)

    # t = np.linspace(0, 10000, 1000000)
    # y = np.cos(t * 2 * np.pi)
    # import time
    # start = time.time()
    # _ = edges_slow.get(t, y, 0.1, 0.0, -0.1)
    # elapsed = time.time() - start
    # print(f"edges_slow = {elapsed}")
    # start = time.time()
    # _ = edges_slow.get_np(t, y, 0.1, 0.0, -0.1)
    # elapsed = time.time() - start
    # print(f"edges_slow_np = {elapsed}")

  def test_fast(self):
    axis_return = [0, 1, 2, 3, 4]
    axis_search = [10, 20, -10, 50, -50]

    value, index = edges_fast.get_crossing(axis_return, axis_search, 1, 0)
    self.assertTrue(np.isnan(value))
    self.assertEqual(0, index)

    value, index = edges_fast.get_crossing(axis_return, axis_search, 2, 0)
    self.assertAlmostEqual(1 + 2 / 3, value)
    self.assertEqual(2, index)

    value, index = edges_fast.get_crossing(axis_return, axis_search, 3, 0)
    self.assertAlmostEqual(2 + 1 / 6, value)
    self.assertEqual(3, index)

    value, index = edges_fast.get_crossing(axis_return, axis_search, 4, 0)
    self.assertAlmostEqual(3.5, value)
    self.assertEqual(4, index)

    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)
    edges_rise, edges_fall = edges_fast.get_np(t, y, 0.1, 0.0, -0.1)
    for i in range(len(edges_rise)):
      self.assertEqualWithinError(i + 0.75, edges_rise[i], 0.01)
    for i in range(len(edges_fall)):
      self.assertEqualWithinError(i + 0.25, edges_fall[i], 0.01)

    # t = np.linspace(0, 10000, 1000000)
    # y = np.cos(t * 2 * np.pi)
    # import time
    # start = time.time()
    # _ = edges_fast.get_np(t, y, 0.1, 0.0, -0.1)
    # elapsed = time.time() - start
    # print(f"edges_fast_np = {elapsed}")
