"""Test module hardware_tools.extension.intersections
"""

import time

import numpy as np

from hardware_tools.extensions import intersections_slow, intersections_fast

from tests import base


class TestIntersections(base.TestBase):
  """Test intersections methods
  """

  def _test_get(self, module):
    x0 = self._RNG.uniform(-1000, 1000)
    y0 = self._RNG.uniform(-1000, 1000)
    x1 = self._RNG.uniform(-1000, 1000)
    y1 = self._RNG.uniform(-1000, 1000)

    if x0 == x1 and y0 == y1:
      x0 = x1 + 1

    p_intersect = module.get(x0, y0, x1, y1, x0, y1, x1, y0)
    self.assertIsNotNone(p_intersect)
    self.assertEqualWithinError((x0 + x1) / 2, p_intersect[0], 0.01)
    self.assertEqualWithinError((y0 + y1) / 2, p_intersect[1], 0.01)

    p_intersect = module.get(x1, y1, x0, y0, x1, y0, x0, y1)
    self.assertIsNotNone(p_intersect)
    self.assertEqualWithinError((x0 + x1) / 2, p_intersect[0], 0.01)
    self.assertEqualWithinError((y0 + y1) / 2, p_intersect[1], 0.01)

    p_intersect = module.get(x0, y0, x1, y1, x0, y0, x1, y1)
    self.assertIsNone(p_intersect)

    p_intersect = module.get(x0, y0, x1, y1, x0, y0 + 1, x1, y1 + 0.9)
    self.assertIsNone(p_intersect)
    p_intersect = module.get(x0,
                             y0,
                             x1,
                             y1,
                             x0,
                             y0 + 1,
                             x1,
                             y1 + 0.9,
                             segments=False)
    self.assertIsNotNone(p_intersect)

    p_intersect = module.get(0, 0, 1, 1, -1, 10, 3, 2)
    self.assertIsNone(p_intersect)
    p_intersect = module.get(1, 1, 0, 0, -1, 10, 3, 2)
    self.assertIsNone(p_intersect)
    p_intersect = module.get(0, 0, 0, 1, -1, 10, 0.2, 1.1)
    self.assertIsNone(p_intersect)
    p_intersect = module.get(0, 1, 0, 0, -1, 10, 0.2, 1.1)
    self.assertIsNone(p_intersect)
    p_intersect = module.get(0, 0, 1, 1, 0, 1, 0.1, 0.9)
    self.assertIsNone(p_intersect)
    p_intersect = module.get(0, 0, 1, 1, 0.1, 0.9, 0, 1)
    self.assertIsNone(p_intersect)
    p_intersect = module.get(0, 0, 1, 1, 0.1, 1, 0.1, 0.9)
    self.assertIsNone(p_intersect)

    p_intersect = module.get(0, 0, 1, 1, 0.1, 0.9, 0.1, 1)
    self.assertIsNone(p_intersect)
    p_intersect = module.get(0, -1, 0.2, -1, 0.1, 0.9, 0.1, 1)
    self.assertIsNone(p_intersect)
    p_intersect = module.get(0, 0, 0, 1, 0, 0.5, 2, 0.5)
    self.assertIsNotNone(p_intersect)

  def test_get(self):
    self._test_get(intersections_slow)
    self._test_get(intersections_fast)

    # Validate fast is actually faster
    x0 = self._RNG.uniform(-1000, 1000)
    y0 = self._RNG.uniform(-1000, 1000)
    x1 = self._RNG.uniform(-1000, 1000)
    y1 = self._RNG.uniform(-1000, 1000)

    if x0 == x1 and y0 == y1:
      x0 = x1 + 1

    n = 10000

    start = time.perf_counter()
    for _ in range(n):
      _ = intersections_slow.get(x0, y0, x1, y1, x0, y1, x1, y0)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n):
      _ = intersections_fast.get(x0, y0, x1, y1, x0, y1, x1, y0)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_is_hitting(self, module):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)
    t = t.tolist()
    y = y.tolist()

    mask = [[(0, -1), (10, 1)]]
    self.assertTrue(module.is_hitting(t, y, mask))

    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]
    self.assertFalse(module.is_hitting(t, y, mask))

  def test_is_hitting(self):
    self._test_is_hitting(intersections_slow)
    self._test_is_hitting(intersections_fast)

    # Validate fast is actually faster
    t = np.linspace(0, 10, 10000)
    y = np.cos(t * 2 * np.pi)
    t = t.tolist()
    y = y.tolist()
    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]

    start = time.perf_counter()
    _ = intersections_slow.is_hitting(t, y, mask)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    _ = intersections_fast.is_hitting(t, y, mask)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_get_hits(self, module):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)
    t = t.tolist()
    y = y.tolist()

    mask = [[(0, 1), (0.5, -1), (1, 1)]]
    targets = [(0, 1), (0.25, 0), (0.5, -1), (0.5, -1), (0.75, 0), (1, 1)]
    targets = sorted(targets, key=lambda h: h[0])
    hits = module.get_hits(t, y, mask)
    self.assertIsInstance(hits, list)
    self.assertIsInstance(hits[0], tuple)
    hits = sorted(hits, key=lambda h: h[0])
    self.assertEqual(len(hits), len(targets))
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]
    hits = module.get_hits(t, y, mask)
    self.assertEqual(0, len(hits))

  def test_get_hits(self):
    self._test_get_hits(intersections_slow)
    self._test_get_hits(intersections_fast)

    # Validate fast is actually faster
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi * 10)
    t = t.tolist()
    y = y.tolist()
    mask = [[(0, 1), (5, -1), (10, 1)], [(0, -1), (5, 1), (10, -1)],
            [(1, -1), (2, 1), (3, -1), (4, -1)]]

    start = time.perf_counter()
    result_slow = intersections_slow.get_hits(t, y, mask)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = intersections_fast.get_hits(t, y, mask)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertEqual(len(result_slow), len(result_fast))
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_is_hitting_np(self, module):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)

    mask = [[(0, -1), (10, 1)]]
    self.assertTrue(module.is_hitting_np(t, y, mask))

    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]
    self.assertFalse(module.is_hitting_np(t, y, mask))

  def test_is_hitting_np(self):
    self._test_is_hitting_np(intersections_slow)
    self._test_is_hitting_np(intersections_fast)

    # Validate fast is actually faster
    t = np.linspace(0, 10, 10000)
    y = np.cos(t * 2 * np.pi)
    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]

    start = time.perf_counter()
    _ = intersections_slow.is_hitting_np(t, y, mask)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    _ = intersections_fast.is_hitting_np(t, y, mask)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_get_hits_np(self, module):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)

    mask = [[(0, 1), (0.5, -1), (1, 1)]]
    targets = [(0, 1), (0.25, 0), (0.5, -1), (0.5, -1), (0.75, 0), (1, 1)]
    targets = sorted(targets, key=lambda h: h[0])
    hits = module.get_hits_np(t, y, mask)
    self.assertIsInstance(hits, np.ndarray)
    self.assertIsInstance(hits[0], np.ndarray)
    hits = sorted(hits, key=lambda h: h[0])
    self.assertEqual(len(hits), len(targets))
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]
    hits = module.get_hits_np(t, y, mask)
    self.assertEqual(0, len(hits))

  def test_get_hits_np(self):
    self._test_get_hits_np(intersections_slow)
    self._test_get_hits_np(intersections_fast)

    # Validate fast is actually faster
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi * 10)
    mask = [[(0, 1), (5, -1), (10, 1)], [(0, -1), (5, 1), (10, -1)],
            [(1, -1), (2, 1), (3, -1), (4, -1)]]

    start = time.perf_counter()
    result_slow = intersections_slow.get_hits_np(t, y, mask)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = intersections_fast.get_hits_np(t, y, mask)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertEqual(result_slow.shape, result_fast.shape)
    self.assertLess(elapsed_fast, elapsed_slow)
