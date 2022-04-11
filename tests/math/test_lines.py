"""Test module hardware_tools.lines.lines
"""

import time

import numpy as np

from hardware_tools.math import lines, _lines, _lines_fb

from tests import base


class TestLines(base.TestBase):
  """Test lines methods
  """

  def test_point2d(self):
    x0 = self._RNG.uniform(-1000, 1000)
    y0 = self._RNG.uniform(-1000, 1000)
    x1 = self._RNG.uniform(-1000, 1000)
    y1 = self._RNG.uniform(-1000, 1000)

    if x0 == x1 and y0 == y1:
      x0 = x1 + 1

    p0 = lines.Point2D(x0, y0)
    p1 = lines.Point2D(x1, y1)
    self.assertEqual(p0.x, x0)
    self.assertEqual(p0.y, y0)
    self.assertEqual(str(p0), f"({x0}, {y0})")

    p_center = lines.Point2D((x0 + x1) / 2, (y0 + y1) / 2)
    self.assertTrue(p_center.in_rect(p0, p1))
    self.assertTrue(p_center.in_rect(p1, p0))
    self.assertTrue(p0.in_rect(p0, p1))
    self.assertTrue(p1.in_rect(p0, p1))
    self.assertFalse(p1.in_rect(p0, p_center))
    self.assertFalse(p1.in_rect(p_center, p0))
    self.assertFalse(p0.in_rect(p1, p_center))
    self.assertFalse(p0.in_rect(p_center, p1))

    self.assertEqual(0, lines.Point2D.orientation(p0, p_center, p1))
    self.assertEqual(0, lines.Point2D.orientation(p0, p1, p_center))
    self.assertEqual(0, lines.Point2D.orientation(p_center, p0, p1))

    p_knee = lines.Point2D(p_center.x, 2000)
    if x0 < x1:
      self.assertEqual(1, lines.Point2D.orientation(p0, p_knee, p1))
      self.assertEqual(-1, lines.Point2D.orientation(p1, p_knee, p0))
    elif x0 > x1:
      self.assertEqual(-1, lines.Point2D.orientation(p0, p_knee, p1))
      self.assertEqual(1, lines.Point2D.orientation(p1, p_knee, p0))
    else:
      p_knee = lines.Point2D(2000, p_center.y)
      if y0 < y1:
        self.assertEqual(-1, lines.Point2D.orientation(p0, p_knee, p1))
        self.assertEqual(1, lines.Point2D.orientation(p1, p_knee, p0))
      else:
        self.assertEqual(1, lines.Point2D.orientation(p0, p_knee, p1))
        self.assertEqual(-1, lines.Point2D.orientation(p1, p_knee, p0))

  def test_line2d(self):
    x0 = self._RNG.uniform(-1000, 1000)
    y0 = self._RNG.uniform(-1000, 1000)
    x1 = self._RNG.uniform(-1000, 1000)
    y1 = self._RNG.uniform(-1000, 1000)

    if x0 == x1 and y0 == y1:
      x0 = x1 + 1

    l0 = lines.Line2D(x0, y0, x1, y1)
    self.assertEqual(l0.p.x, x0)
    self.assertEqual(l0.p.y, y0)
    self.assertEqual(l0.q.x, x1)
    self.assertEqual(l0.q.y, y1)
    self.assertEqual(str(l0), f"(({x0}, {y0}), ({x1}, {y1}))")

    l1 = lines.Line2D(x0, y1, x1, y0)
    self.assertTrue(l0.intersecting(l1))
    l1 = lines.Line2D(x0 + 1, y0 + 1, x1 + 1, y1 + 1)
    self.assertFalse(l0.intersecting(l1))

    p0 = lines.Point2D((x0 + x1) / 2, (y0 + y1) / 2)
    p1 = lines.Point2D(x0 + (x1 - x0) * 2, y0 + (y1 - y0) * 2)
    self.assertTrue(l0.intersecting_points(p0, p1))
    self.assertTrue(l0.intersecting_points(p1, p0))

    p0 = lines.Point2D(x0 - (x1 - x0) / 2, y0 - (y1 - y0) / 2)
    p1 = lines.Point2D(x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2)
    l1 = lines.Line2D(x0, y0, x0, y0)
    self.assertTrue(l1.intersecting_points(p0, p1))

    l1 = lines.Line2D(x0, y1, x1, y0)
    p_intersect = l0.intersection(l1)
    self.assertIsNotNone(p_intersect)
    self.assertAlmostEqual(p_intersect.x, (x0 + x1) / 2)
    self.assertAlmostEqual(p_intersect.y, (y0 + y1) / 2)

    self.assertIsNone(l0.intersection(l0))
    l1 = lines.Line2D(x0 + 1, y0 + 1, x1 + 1, y1 + 1)
    self.assertIsNone(l0.intersection(l1))


class TestDraw(base.TestBase):
  """Test line drawing methods
  """

  def _test_draw(self, module: _lines_fb):
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
    self._test_draw(_lines_fb)
    self._test_draw(_lines)

    # Validate fast is actually faster
    shape = self._RNG.integers(10, 100, 2)
    grid = np.zeros(shape=shape, dtype=np.int32)

    x = np.array([shape[0] - 1, 0, 0, shape[0] - 1]).astype(np.int32)
    y = np.array([0, 0, shape[1] - 1, 0]).astype(np.int32)
    x = np.repeat(x, 10000)
    y = np.repeat(y, 10000)

    start = time.perf_counter()
    _lines_fb.draw(x, y, grid)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    _lines.draw(x, y, grid)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_draw_points(self, module: _lines_fb):
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
    self._test_draw_points(_lines_fb)
    self._test_draw_points(_lines)

    # Validate fast is actually faster
    shape = self._RNG.integers(10, 100, 2)
    grid = np.zeros(shape=shape, dtype=np.int32)

    x = np.array([shape[0] - 1, 0, 0, shape[0] - 1]).astype(np.int32)
    y = np.array([0, 0, shape[1] - 1, 0]).astype(np.int32)
    x = np.repeat(x, 10000)
    y = np.repeat(y, 10000)

    start = time.perf_counter()
    _lines_fb.draw_points(x, y, grid)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    _lines.draw_points(x, y, grid)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)


class TestEdges(base.TestBase):
  """Test line edges methods
  """

  def _test_crossing(self, module: _lines_fb):
    axis_return = [0, 1, 2, 3, 4]
    axis_search = [10, 20, -10, 50, -50]

    value, index = module.crossing(axis_return, axis_search, 1, 0)
    self.assertTrue(np.isnan(value))
    self.assertEqual(0, index)

    value, index = module.crossing(axis_return, axis_search, 2, 0)
    self.assertAlmostEqual(1 + 2 / 3, value)
    self.assertEqual(2, index)

    value, index = module.crossing(axis_return, axis_search, 3, 0)
    self.assertAlmostEqual(2 + 1 / 6, value)
    self.assertEqual(3, index)

    value, index = module.crossing(axis_return, axis_search, 4, 0)
    self.assertAlmostEqual(3.5, value)
    self.assertEqual(4, index)

  def test_crossing(self):
    self._test_crossing(_lines_fb)
    self._test_crossing(_lines)

    # Validate fast is actually faster
    n = 10000
    axis_return = [0, 1, 2, 3, 4]
    axis_search = [10, 20, -10, 50, -50]
    i = 4
    value = 0

    start = time.perf_counter()
    for _ in range(n):
      _ = _lines_fb.crossing(axis_return, axis_search, i, value, False, n)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n):
      _ = _lines.crossing(axis_return, axis_search, i, value, False, n)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_edges(self, module: _lines_fb):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)
    t = t.tolist()
    y = y.tolist()

    edges_rise, edges_fall = module.edges(t, y, 0.1, 0.0, -0.1)
    for i in range(len(edges_rise)):
      self.assertEqualWithinError(i + 0.75, edges_rise[i], 0.01)
    for i in range(len(edges_fall)):
      self.assertEqualWithinError(i + 0.25, edges_fall[i], 0.01)

  def test_edges(self):
    self._test_edges(_lines_fb)
    self._test_edges(_lines)

    # Validate fast is actually faster

    t = np.linspace(0, 10000, 100000)
    y = np.cos(t * 2 * np.pi)
    t = t.tolist()
    y = y.tolist()

    start = time.perf_counter()
    _ = _lines_fb.edges(t, y, 0.1, 0.0, -0.1)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    _ = _lines.edges(t, y, 0.1, 0.0, -0.1)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_edges_np(self, module: _lines_fb):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)

    edges_rise, edges_fall = module.edges_np(t, y, 0.1, 0.0, -0.1)
    for i in range(len(edges_rise)):
      self.assertEqualWithinError(i + 0.75, edges_rise[i], 0.01)
    for i in range(len(edges_fall)):
      self.assertEqualWithinError(i + 0.25, edges_fall[i], 0.01)

  def test_edges_np(self):
    self._test_edges_np(_lines_fb)
    self._test_edges_np(_lines)

    # Validate fast is actually faster
    t = np.linspace(0, 10000, 100000)
    y = np.cos(t * 2 * np.pi)

    start = time.perf_counter()
    _ = _lines_fb.edges_np(t, y, 0.1, 0.0, -0.1)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    _ = _lines.edges_np(t, y, 0.1, 0.0, -0.1)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)


class TestIntersections(base.TestBase):
  """Test line intersection methods
  """

  def _test_intersections(self, module: _lines_fb):
    x0 = self._RNG.uniform(-1000, 1000)
    y0 = self._RNG.uniform(-1000, 1000)
    x1 = self._RNG.uniform(-1000, 1000)
    y1 = self._RNG.uniform(-1000, 1000)

    if x0 == x1 and y0 == y1:
      x0 = x1 + 1

    p_intersect = module.intersection(x0, y0, x1, y1, x0, y1, x1, y0)
    self.assertIsNotNone(p_intersect)
    self.assertEqualWithinError((x0 + x1) / 2, p_intersect[0], 0.01)
    self.assertEqualWithinError((y0 + y1) / 2, p_intersect[1], 0.01)

    p_intersect = module.intersection(x1, y1, x0, y0, x1, y0, x0, y1)
    self.assertIsNotNone(p_intersect)
    self.assertEqualWithinError((x0 + x1) / 2, p_intersect[0], 0.01)
    self.assertEqualWithinError((y0 + y1) / 2, p_intersect[1], 0.01)

    p_intersect = module.intersection(x0, y0, x1, y1, x0, y0, x1, y1)
    self.assertIsNone(p_intersect)

    p_intersect = module.intersection(x0, y0, x1, y1, x0, y0 + 1, x1, y1 + 0.9)
    self.assertIsNone(p_intersect)
    p_intersect = module.intersection(x0,
                                      y0,
                                      x1,
                                      y1,
                                      x0,
                                      y0 + 1,
                                      x1,
                                      y1 + 0.9,
                                      segments=False)
    self.assertIsNotNone(p_intersect)

    p_intersect = module.intersection(0, 0, 1, 1, -1, 10, 3, 2)
    self.assertIsNone(p_intersect)
    p_intersect = module.intersection(1, 1, 0, 0, -1, 10, 3, 2)
    self.assertIsNone(p_intersect)
    p_intersect = module.intersection(0, 0, 0, 1, -1, 10, 0.2, 1.1)
    self.assertIsNone(p_intersect)
    p_intersect = module.intersection(0, 1, 0, 0, -1, 10, 0.2, 1.1)
    self.assertIsNone(p_intersect)
    p_intersect = module.intersection(0, 0, 1, 1, 0, 1, 0.1, 0.9)
    self.assertIsNone(p_intersect)
    p_intersect = module.intersection(0, 0, 1, 1, 0.1, 0.9, 0, 1)
    self.assertIsNone(p_intersect)
    p_intersect = module.intersection(0, 0, 1, 1, 0.1, 1, 0.1, 0.9)
    self.assertIsNone(p_intersect)

    p_intersect = module.intersection(0, 0, 1, 1, 0.1, 0.9, 0.1, 1)
    self.assertIsNone(p_intersect)
    p_intersect = module.intersection(0, -1, 0.2, -1, 0.1, 0.9, 0.1, 1)
    self.assertIsNone(p_intersect)
    p_intersect = module.intersection(0, 0, 0, 1, 0, 0.5, 2, 0.5)
    self.assertIsNotNone(p_intersect)

  def test_intersections(self):
    self._test_intersections(_lines_fb)
    self._test_intersections(_lines)

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
      _ = _lines_fb.intersection(x0, y0, x1, y1, x0, y1, x1, y0)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n):
      _ = _lines.intersection(x0, y0, x1, y1, x0, y1, x1, y0)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_is_hitting(self, module: _lines_fb):
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
    self._test_is_hitting(_lines_fb)
    self._test_is_hitting(_lines)

    # Validate fast is actually faster
    t = np.linspace(0, 10, 10000)
    y = np.cos(t * 2 * np.pi)
    t = t.tolist()
    y = y.tolist()
    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]

    start = time.perf_counter()
    _ = _lines_fb.is_hitting(t, y, mask)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    _ = _lines.is_hitting(t, y, mask)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_hits(self, module: _lines_fb):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)
    t = t.tolist()
    y = y.tolist()

    mask = [[(0, 1), (0.5, -1), (1, 1)]]
    targets = [(0, 1), (0.25, 0), (0.5, -1), (0.5, -1), (0.75, 0), (1, 1)]
    targets = sorted(targets, key=lambda h: h[0])
    hits = module.hits(t, y, mask)
    self.assertIsInstance(hits, list)
    self.assertIsInstance(hits[0], tuple)
    hits = sorted(hits, key=lambda h: h[0])
    self.assertEqual(len(hits), len(targets))
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]
    hits = module.hits(t, y, mask)
    self.assertEqual(0, len(hits))

  def test_hits(self):
    self._test_hits(_lines_fb)
    self._test_hits(_lines)

    # Validate fast is actually faster
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi * 10)
    t = t.tolist()
    y = y.tolist()
    mask = [[(0, 1), (5, -1), (10, 1)], [(0, -1), (5, 1), (10, -1)],
            [(1, -1), (2, 1), (3, -1), (4, -1)]]

    start = time.perf_counter()
    result_slow = _lines_fb.hits(t, y, mask)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = _lines.hits(t, y, mask)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertEqual(len(result_slow), len(result_fast))
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_is_hitting_np(self, module: _lines_fb):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)

    mask = [[(0, -1), (10, 1)]]
    self.assertTrue(module.is_hitting_np(t, y, mask))

    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]
    self.assertFalse(module.is_hitting_np(t, y, mask))

  def test_is_hitting_np(self):
    self._test_is_hitting_np(_lines_fb)
    self._test_is_hitting_np(_lines)

    # Validate fast is actually faster
    t = np.linspace(0, 10, 10000)
    y = np.cos(t * 2 * np.pi)
    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]

    start = time.perf_counter()
    _ = _lines_fb.is_hitting_np(t, y, mask)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    _ = _lines.is_hitting_np(t, y, mask)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_hits_np(self, module: _lines_fb):
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi)

    mask = [[(0, 1), (0.5, -1), (1, 1)]]
    targets = [(0, 1), (0.25, 0), (0.5, -1), (0.5, -1), (0.75, 0), (1, 1)]
    targets = sorted(targets, key=lambda h: h[0])
    hits = module.hits_np(t, y, mask)
    self.assertIsInstance(hits, np.ndarray)
    self.assertIsInstance(hits[0], np.ndarray)
    hits = sorted(hits, key=lambda h: h[0])
    self.assertEqual(len(hits), len(targets))
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    mask = [[(1, 2), (5, 1.1)], [(0.25, 0.9), (0.5, -0.9), (0.75, 0.9)],
            [(0.75, -0.9), (1, 0.9), (1.25, -0.9)]]
    hits = module.hits_np(t, y, mask)
    self.assertEqual(0, len(hits))

  def test_hits_np(self):
    self._test_hits_np(_lines_fb)
    self._test_hits_np(_lines)

    # Validate fast is actually faster
    t = np.linspace(0, 10, 1000)
    y = np.cos(t * 2 * np.pi * 10)
    mask = [[(0, 1), (5, -1), (10, 1)], [(0, -1), (5, 1), (10, -1)],
            [(1, -1), (2, 1), (3, -1), (4, -1)]]

    start = time.perf_counter()
    result_slow = _lines_fb.hits_np(t, y, mask)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = _lines.hits_np(t, y, mask)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertEqual(result_slow.shape, result_fast.shape)
    self.assertLess(elapsed_fast, elapsed_slow)
