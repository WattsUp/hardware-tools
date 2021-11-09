"""Test module hardware_tools.extension.intersections
"""

import unittest

import numpy as np

from hardware_tools.extensions import intersections_slow, intersections_fast


class TestIntersections(unittest.TestCase):
  """Test intersections methods
  """

  def test_get_slow(self):
    x0 = np.random.uniform(-1000, 1000)
    y0 = np.random.uniform(-1000, 1000)
    x1 = np.random.uniform(-1000, 1000)
    y1 = np.random.uniform(-1000, 1000)

    if x0 == x1 and y0 == y1:
      x0 = x1 + 1

    p_intersect = intersections_slow.get(x0, y0, x1, y1, x0, y1, x1, y0)
    self.assertIsNotNone(p_intersect)
    self.assertAlmostEqual(p_intersect[0], (x0 + x1) / 2, 2)
    self.assertAlmostEqual(p_intersect[1], (y0 + y1) / 2, 2)

    p_intersect = intersections_slow.get(x1, y1, x0, y0, x1, y0, x0, y1)
    self.assertIsNotNone(p_intersect)
    self.assertAlmostEqual(p_intersect[0], (x0 + x1) / 2, 2)
    self.assertAlmostEqual(p_intersect[1], (y0 + y1) / 2, 2)

    p_intersect = intersections_slow.get(x0, y0, x1, y1, x0, y0, x1, y1)
    self.assertIsNone(p_intersect)

    p_intersect = intersections_slow.get(x0, y0, x1, y1, x0, y0 + 1, x1,
                                         y1 + 0.9)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_slow.get(x0,
                                         y0,
                                         x1,
                                         y1,
                                         x0,
                                         y0 + 1,
                                         x1,
                                         y1 + 0.9,
                                         segments=False)
    self.assertIsNotNone(p_intersect)

    p_intersect = intersections_slow.get(0, 0, 1, 1, -1, 10, 3, 2)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_slow.get(1, 1, 0, 0, -1, 10, 3, 2)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_slow.get(0, 0, 0, 1, -1, 10, 0.2, 1.1)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_slow.get(0, 1, 0, 0, -1, 10, 0.2, 1.1)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_slow.get(0, 0, 1, 1, 0, 1, 0.1, 0.9)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_slow.get(0, 0, 1, 1, 0.1, 0.9, 0, 1)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_slow.get(0, 0, 1, 1, 0.1, 1, 0.1, 0.9)
    self.assertIsNone(p_intersect)

    p_intersect = intersections_slow.get(0, 0, 1, 1, 0.1, 0.9, 0.1, 1)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_slow.get(0, -1, 0.2, -1, 0.1, 0.9, 0.1, 1)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_slow.get(0, 0, 0, 1, 0, 0.5, 2, 0.5)
    self.assertIsNotNone(p_intersect)

    # import time
    # start = time.time()
    # for _ in range(100000):
    #   _ = intersections_slow.get(x0, y0, x1, y1, x0, y1, x1, y0)
    # elapsed = time.time() - start
    # print(f"intersections_slow = {elapsed}")

  def test_get_fast(self):
    x0 = np.random.uniform(-1000, 1000)
    y0 = np.random.uniform(-1000, 1000)
    x1 = np.random.uniform(-1000, 1000)
    y1 = np.random.uniform(-1000, 1000)

    if x0 == x1 and y0 == y1:
      x0 = x1 + 1

    p_intersect = intersections_fast.get(x0, y0, x1, y1, x0, y1, x1, y0)
    self.assertIsNotNone(p_intersect)
    self.assertAlmostEqual(p_intersect[0], (x0 + x1) / 2, 2)
    self.assertAlmostEqual(p_intersect[1], (y0 + y1) / 2, 2)

    p_intersect = intersections_fast.get(x1, y1, x0, y0, x1, y0, x0, y1)
    self.assertIsNotNone(p_intersect)
    self.assertAlmostEqual(p_intersect[0], (x0 + x1) / 2, 2)
    self.assertAlmostEqual(p_intersect[1], (y0 + y1) / 2, 2)

    p_intersect = intersections_fast.get(x0, y0, x1, y1, x0, y0, x1, y1)
    self.assertIsNone(p_intersect)

    p_intersect = intersections_fast.get(x0, y0, x1, y1, x0, y0 + 1, x1,
                                         y1 + 0.9)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_fast.get(x0,
                                         y0,
                                         x1,
                                         y1,
                                         x0,
                                         y0 + 1,
                                         x1,
                                         y1 + 0.9,
                                         segments=False)
    self.assertIsNotNone(p_intersect)

    p_intersect = intersections_fast.get(0, 0, 1, 1, -1, 10, 3, 2)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_fast.get(1, 1, 0, 0, -1, 10, 3, 2)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_fast.get(0, 0, 0, 1, -1, 10, 0.2, 1.1)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_fast.get(0, 1, 0, 0, -1, 10, 0.2, 1.1)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_fast.get(0, 0, 1, 1, 0, 1, 0.1, 0.9)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_fast.get(0, 0, 1, 1, 0.1, 0.9, 0, 1)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_fast.get(0, 0, 1, 1, 0.1, 1, 0.1, 0.9)
    self.assertIsNone(p_intersect)

    p_intersect = intersections_fast.get(0, 0, 1, 1, 0.1, 0.9, 0.1, 1)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_fast.get(0, -1, 0.2, -1, 0.1, 0.9, 0.1, 1)
    self.assertIsNone(p_intersect)
    p_intersect = intersections_fast.get(0, 0, 0, 1, 0, 0.5, 2, 0.5)
    self.assertIsNotNone(p_intersect)

    # import time
    # start = time.time()
    # for _ in range(100000):
    #   _ = intersections_fast.get(x0, y0, x1, y1, x0, y1, x1, y0)
    # elapsed = time.time() - start
    # print(f"intersections_fast = {elapsed}")

  def test_hits_slow(self):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)

    mask = [[(0, -1), (10, 1)]]
    self.assertTrue(intersections_slow.is_hitting_np(t, y, mask))

    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]
    self.assertFalse(intersections_slow.is_hitting_np(t, y, mask))

    mask = [[(0, 1), (0.5, -1), (1, 1)]]
    targets = [(0, 1), (0.25, 0), (0.5, -1), (0.5, -1), (0.75, 0), (1, 1)]
    targets = sorted(targets, key=lambda h: h[0])
    hits = intersections_slow.get_hits_np(t, y, mask)
    hits = sorted(hits, key=lambda h: h[0])
    self.assertEqual(len(hits), len(targets))
    for hit, target in zip(hits, targets):
      self.assertAlmostEqual(hit[0], target[0], 2)
      self.assertAlmostEqual(hit[1], target[1], 2)

    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]
    hits = intersections_slow.get_hits_np(t, y, mask)
    self.assertEqual(0, len(hits))

    # t = np.linspace(0, 10, 10000)
    # y = np.cos(t * 2 * np.pi)
    # mask = np.sin(t[::100] * 20 * np.pi)
    # mask = [[(tp, yp) for tp, yp in zip(t, mask)]]
    # t = t.tolist()
    # y = y.tolist()
    # import time
    # start = time.time()
    # for _ in range(100):
    #   _ = intersections_slow.get_hits(t, y, mask)
    # elapsed = time.time() - start
    # print(f"intersections_slow = {elapsed}")

  def test_hits_fast(self):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)

    mask = [[(0, -1), (10, 1)]]
    self.assertTrue(intersections_fast.is_hitting_np(t, y, mask))

    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]
    self.assertFalse(intersections_fast.is_hitting_np(t, y, mask))

    mask = [[(0, 1), (0.5, -1), (1, 1)]]
    targets = [(0, 1), (0.25, 0), (0.5, -1), (0.5, -1), (0.75, 0), (1, 1)]
    targets = sorted(targets, key=lambda h: h[0])
    hits = intersections_fast.get_hits_np(t, y, mask)
    hits = sorted(hits, key=lambda h: h[0])
    self.assertEqual(len(hits), len(targets))
    for hit, target in zip(hits, targets):
      self.assertAlmostEqual(hit[0], target[0], 2)
      self.assertAlmostEqual(hit[1], target[1], 2)

    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]
    hits = intersections_fast.get_hits_np(t, y, mask)
    self.assertEqual(0, len(hits))

    # t = np.linspace(0, 10, 10000)
    # y = np.cos(t * 2 * np.pi)
    # mask = np.sin(t[::100] * 20 * np.pi)
    # mask = [[(tp, yp) for tp, yp in zip(t, mask)]]
    # t = t.tolist()
    # y = y.tolist()
    # import time
    # start = time.time()
    # for _ in range(100):
    #   _ = intersections_fast.get_hits(t, y, mask)
    # elapsed = time.time() - start
    # print(f"intersections_fast = {elapsed}")
