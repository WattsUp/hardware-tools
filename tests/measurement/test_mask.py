"""Test module hardware_tools.measurement.mask
"""

from __future__ import annotations

from hardware_tools.measurement import mask
from hardware_tools.math import lines

from tests import base


class Derrived(mask.Mask):

  def convert_mixed_units(self, ua: float, ui: float) -> Derrived:
    m = Derrived()
    m._paths = self._paths  # pylint: disable=protected-access
    m._converted = True  # pylint: disable=protected-access
    return m

  def adjust(self, factor: float) -> Derrived:
    return self


class TestMask(base.TestBase):
  """Test Mask
  """

  def test_init(self):

    m = Derrived()
    self.assertRaises(ValueError, getattr, m, "paths")
    m.convert_mixed_units(1, 1)
    self.assertRaises(ValueError, getattr, m, "paths")
    m = m.convert_mixed_units(1, 1)
    self.assertListEqual([], m.paths)

    paths = [(0, 1), (1, 0)]
    m = Derrived(paths=paths)
    m = m.convert_mixed_units(1, 1)
    self.assertListEqual(paths, m.paths)

    self.assertDictEqual({"paths": paths}, m.to_dict())


class TestMaskHexagon(base.TestBase):
  """Test MaskHexagon
  """

  def test_init(self):
    x1 = 0.25
    x2 = 0.3
    y1 = 0.2
    yl = 0.1
    yu = 0.3

    m = mask.MaskHexagon(x1, x2, y1, yl, yu)
    m = m.convert_mixed_units(1.0, 1.0)
    self.assertEqual(3, len(m.paths))

    d = {"type": "hexagon", "x1": x1, "x2": x2, "y1": y1, "yl": yl, "yu": yu}
    self.assertDictEqual(d, m.to_dict())

    t = [0, 1]
    yp = -yl - 0.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 1 + yu + 0.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 0.5001
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(x1, yp), (1 - x1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 0.75
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))

  def test_mixed_units(self):
    x1 = mask.RealUnits(0.25)
    x2 = 0.3
    y1 = 0.2
    yl = 0.1
    yu = mask.RealUnits(0.3)

    ui = 0.5
    ua = 0.5

    m = mask.MaskHexagon(x1, x2, y1, yl, yu)
    m = m.convert_mixed_units(ua, ui)
    self.assertEqual(3, len(m.paths))

    d = {
        "type": "hexagon",
        "x1": x1 / ui,
        "x2": x2,
        "y1": y1,
        "yl": yl,
        "yu": yu / ua
    }
    self.assertDictEqual(d, m.to_dict())

    t = [0, 1]
    yp = 1 + yu * 2.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 1 + yu * 1.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(0, len(hits))

  def test_not_zero_ref(self):
    x1 = 0.25
    x2 = 0.3
    y1 = 0.2
    yl = 0.1
    yu = 0.3

    m = mask.MaskHexagon(x1, x2, y1, yl, yu, zero_ref=False)
    m = m.convert_mixed_units(1.0, 1.0)
    self.assertEqual(3, len(m.paths))

    d = {
        "type": "hexagon",
        "x1": x1,
        "x2": x2,
        "y1": 0.5 - y1,
        "yl": yl - 0.5,
        "yu": yu - 0.5
    }
    self.assertDictEqual(d, m.to_dict())

    t = [0, 1]
    yp = -yl - 0.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 1 + yu + 0.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 0.5001
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(x1, yp), (1 - x1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 0.75
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(0, len(hits))

  def test_adjust(self):
    x1 = 0.25
    x2 = 0.3
    y1 = 0.2
    yl = 0.1
    yu = 0.3

    m = mask.MaskHexagon(x1, x2, y1, yl, yu)
    m = m.convert_mixed_units(1.0, 1.0)
    self.assertEqual(3, len(m.paths))

    d = {"type": "hexagon", "x1": x1, "x2": x2, "y1": y1, "yl": yl, "yu": yu}
    self.assertDictEqual(d, m.to_dict())

    f = 0.1
    m_adjusted = m.adjust(f)

    x1_new = x1 * (1 - f)
    d = {
        "type": "hexagon",
        "x1": x1 * (1 - f),
        "x2": 0.5 - (0.5 - x1_new) * (0.5 - x2) / (0.5 - x1),
        "y1": y1 * (1 - f),
        "yl": yl * (1 - f),
        "yu": yu * (1 - f)
    }
    self.assertDictEqual(d, m_adjusted.to_dict())

    f = -0.1
    m_adjusted = m.adjust(f)

    d = {
        "type": "hexagon",
        "x1": 0.5 * (-f) + x1 * (1 + f),
        "x2": 0.5 * (-f) + x2 * (1 + f),
        "y1": 0.5 * (-f) + y1 * (1 + f),
        "yl": 0.5 * (-f) + yl * (1 + f),
        "yu": 0.5 * (-f) + yu * (1 + f)
    }
    self.assertDictEqual(d, m_adjusted.to_dict())


class TestMaskDecagon(base.TestBase):
  """Test MaskDecagon
  """

  def test_init(self):
    x1 = 0.25
    x2 = 0.3
    x3 = 0.4
    y1 = 0.2
    y2 = 0.15
    yl = 0.1
    yu = 0.3

    m = mask.MaskDecagon(x1, x2, x3, y1, y2, yl, yu)
    m = m.convert_mixed_units(1.0, 1.0)
    self.assertEqual(3, len(m.paths))

    d = {
        "type": "decagon",
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "y1": y1,
        "y2": y2,
        "yl": yl,
        "yu": yu
    }
    self.assertDictEqual(d, m.to_dict())

    t = [0, 1]
    yp = -yl - 0.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 1 + yu + 0.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 0.5001
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(x1, yp), (1 - x1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 0.75
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))

  def test_mixed_units(self):
    x1 = mask.RealUnits(0.25)
    x2 = 0.3
    x3 = 0.4
    y1 = 0.2
    y2 = 0.15
    yl = 0.1
    yu = mask.RealUnits(0.3)

    ui = 0.5
    ua = 0.5

    m = mask.MaskDecagon(x1, x2, x3, y1, y2, yl, yu)
    m = m.convert_mixed_units(ua, ui)
    self.assertEqual(3, len(m.paths))

    d = {
        "type": "decagon",
        "x1": x1 / ui,
        "x2": x2,
        "x3": x3,
        "y1": y1,
        "y2": y2,
        "yl": yl,
        "yu": yu / ua
    }
    self.assertDictEqual(d, m.to_dict())

    t = [0, 1]
    yp = 1 + yu * 2.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 1 + yu * 1.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(0, len(hits))

  def test_not_zero_ref(self):
    x1 = 0.25
    x2 = 0.3
    x3 = 0.4
    y1 = 0.2
    y2 = 0.15
    yl = 0.1
    yu = 0.3

    m = mask.MaskDecagon(x1, x2, x3, y1, y2, yl, yu, zero_ref=False)
    m = m.convert_mixed_units(1.0, 1.0)
    self.assertEqual(3, len(m.paths))

    d = {
        "type": "decagon",
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "y1": 0.5 - y1,
        "y2": 0.5 - y2,
        "yl": yl - 0.5,
        "yu": yu - 0.5
    }
    self.assertDictEqual(d, m.to_dict())

    t = [0, 1]
    yp = -yl - 0.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 1 + yu + 0.1
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(0, yp), (1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 0.5001
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(2, len(hits))
    targets = [(x1, yp), (1 - x1, yp)]
    for hit, target in zip(hits, targets):
      self.assertEqualWithinError(target[0], hit[0], 0.01)
      self.assertEqualWithinError(target[1], hit[1], 0.01)

    t = [0, 1]
    yp = 0.75
    y = [yp, yp]
    hits = lines.hits(t, y, m.paths)
    self.assertEqual(0, len(hits))

  def test_adjust(self):
    x1 = 0.25
    x2 = 0.3
    x3 = 0.4
    y1 = 0.2
    y2 = 0.15
    yl = 0.1
    yu = 0.3

    m = mask.MaskDecagon(x1, x2, x3, y1, y2, yl, yu)
    m = m.convert_mixed_units(1.0, 1.0)
    self.assertEqual(3, len(m.paths))

    d = {
        "type": "decagon",
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "y1": y1,
        "y2": y2,
        "yl": yl,
        "yu": yu
    }
    self.assertDictEqual(d, m.to_dict())

    f = 0.1
    m_adjusted = m.adjust(f)

    x1_new = x1 * (1 - f)
    y1_new = y1 * (1 - f)
    d = {
        "type": "decagon",
        "x1": x1 * (1 - f),
        "x2": 0.5 - (0.5 - x1_new) * (0.5 - x2) / (0.5 - x1),
        "x3": 0.5 - (0.5 - x1_new) * (0.5 - x3) / (0.5 - x1),
        "y1": y1 * (1 - f),
        "y2": 0.5 - (0.5 - y1_new) * (0.5 - y2) / (0.5 - y1),
        "yl": yl * (1 - f),
        "yu": yu * (1 - f)
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
        "yl": 0.5 * (-f) + yl * (1 + f),
        "yu": 0.5 * (-f) + yu * (1 + f)
    }
    self.assertDictEqual(d, m_adjusted.to_dict())
