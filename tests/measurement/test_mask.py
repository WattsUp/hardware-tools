"""Test module hardware_tools.measurement.mask
"""

from __future__ import annotations

import unittest

from hardware_tools.measurement import mask
from hardware_tools.extensions import intersections


class Derrived(mask.Mask):

  def adjust(self, factor: float) -> Derrived:
    return self


class TestMask(unittest.TestCase):
  """Test Mask
  """

  def test_init(self):

    m = Derrived()
    self.assertListEqual([], m.paths)

    paths = [(0, 1), (1, 0)]
    m = Derrived(paths=paths)
    self.assertListEqual(paths, m.paths)

    self.assertDictEqual({"paths": paths}, m.to_dict())


class TestMaskDecagon(unittest.TestCase):
  """Test MaskDecagon
  """

  def test_init(self):
    x1 = 0.25
    x2 = 0.3
    x3 = 0.4
    y1 = 0.2
    y2 = 0.15
    y3 = 0.1
    y4 = 0.3

    m = mask.MaskDecagon(x1, x2, x3, y1, y2, y3, y4)
    self.assertEqual(3, len(m.paths))

    d = {
        "type": "decagon",
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "y1": y1,
        "y2": y2,
        "y3": y3,
        "y4": y4
    }
    self.assertDictEqual(d, m.to_dict())

    t = [0, 1]
    yp = -y3 - 0.1
    y = [yp, yp]
    hits = intersections.get_hits(t, y, m.paths)
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertAlmostEqual(hit[0], target[0], 2)
      self.assertAlmostEqual(hit[1], target[1], 2)

    t = [0, 1]
    yp = 1 + y4 + 0.1
    y = [yp, yp]
    hits = intersections.get_hits(t, y, m.paths)
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertAlmostEqual(hit[0], target[0], 2)
      self.assertAlmostEqual(hit[1], target[1], 2)

    t = [0, 1]
    yp = 0.5
    y = [yp, yp]
    hits = intersections.get_hits(t, y, m.paths)
    targets = [(x1, yp), (1 - x1, yp)]
    for hit, target in zip(hits, targets):
      self.assertAlmostEqual(hit[0], target[0], 2)
      self.assertAlmostEqual(hit[1], target[1], 2)

  def test_adjust(self):
    x1 = 0.25
    x2 = 0.3
    x3 = 0.4
    y1 = 0.2
    y2 = 0.15
    y3 = 0.1
    y4 = 0.3

    m = mask.MaskDecagon(x1, x2, x3, y1, y2, y3, y4)
    self.assertEqual(3, len(m.paths))

    d = {
        "type": "decagon",
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "y1": y1,
        "y2": y2,
        "y3": y3,
        "y4": y4
    }
    self.assertDictEqual(d, m.to_dict())

    f = 0.1
    m_adjusted = m.adjust(f)

    d = {
        "type": "decagon",
        "x1": x1 * (1 - f),
        "x2": x2 * (1 - f),
        "x3": x3 * (1 - f),
        "y1": y1 * (1 - f),
        "y2": y2 * (1 - f),
        "y3": y3 * (1 - f),
        "y4": y4 * (1 - f)
    }
    self.assertDictEqual(d, m_adjusted.to_dict())

    f = -0.1
    m_adjusted = m.adjust(f)

    d = {
        "type": "decagon",
        "x1": 0.5 * (-f) + x1 * (1 + f),
        "x2": 0.5 * (-f) + x2 * (1 + f),
        "x3": 0.5 * (-f) + x3 * (1 + f),
        "y1": 0.5 * (-f) + y1 * (1 + f),
        "y2": 0.5 * (-f) + y2 * (1 + f),
        "y3": 0.5 * (-f) + y3 * (1 + f),
        "y4": 0.5 * (-f) + y4 * (1 + f)
    }
    self.assertDictEqual(d, m_adjusted.to_dict())
