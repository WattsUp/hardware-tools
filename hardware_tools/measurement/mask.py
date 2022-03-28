"""2D mask to validate waveform shape or other 2D data
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union


class RealUnits(float):
  """Real world unit values, will be converted to UA/UI
  """
  pass


MixedUnits = Union[float, RealUnits]


class Mask(ABC):
  """Mask base class to hold a collection of paths to validate waveform shape
  or other 2D data

  A mask hit occurs when a segment in the test data intersects with one of the
  path segments. Checking for segments lying inside the closed region formed by
  the paths is not performed.

  Properties:
    paths: list of paths describing edges of mask. A path is a list of points
  """

  def __init__(self, paths: list = None) -> None:
    """Create a new Mask

    Args:
      paths: list of paths describing edges of mask. A path is a list of points
    """
    if paths is None:
      paths = []
    self._paths = paths
    self._converted = False

  @property
  def paths(self) -> list:
    if not self._converted:
      raise ValueError(
          "Mask.convert_mixed_units must be called before obtaining paths")
    return self._paths

  @staticmethod
  def _convert_ua(actual: MixedUnits, ua: float) -> float:
    """Convert mixed units to normalized amplitude

    Args:
      actual: Value to convert
      ua: Normalized amplitude

    Returns:
      (actual / ua) if actual is RealUnits type
      else returns actual
    """
    if isinstance(actual, RealUnits):
      return actual / ua
    return actual

  @staticmethod
  def _convert_ui(actual: MixedUnits, ui: float) -> float:
    """Convert mixed units to normalized interval

    Args:
      actual: Value to convert
      ua: Normalized interval

    Returns:
      (actual / ui) if actual is RealUnits type
      else returns actual
    """
    if isinstance(actual, RealUnits):
      return actual / ui
    return actual

  @abstractmethod
  def convert_mixed_units(self, ua: float, ui: float) -> Mask:
    """Convert path parameters with mixed units to all UI/UA

    Args:
      ua: Normalized amplitude
      ui: Normalized interval

    Returns:
      Converted Mask
    """
    pass  # pragma: no cover

  @abstractmethod
  def adjust(self, factor: float) -> Mask:
    """Adjust the size of the Mask and return a new Mask

    Args:
      factor: Adjustment factor -1=no mask, 0=unadjusted, 1=no gaps

    Returns:
      Newly adjusted mask
    """
    pass  # pragma: no cover

  def to_dict(self) -> dict:
    """Convert Mask to dictionary

    Compatible with default JSONEncoder

    Returns:
      dictionary of Mask definition
    """
    return {"paths": self.paths}


class MaskHexagon(Mask):
  """A Mask with a center hexagon (6-sides)

  zero_ref = True (default)
         ▐███████████████████████████████████████████████████████▌
  1 + yu ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆                       ┆       ┆       │
       1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆                       ┆       ┆       │
         │       ┆       ┆                       ┆       ┆       │
  1 - y1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄▄█████████████████████████▄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆ ▄▄▄███████████████████████████████▄▄▄ ┆       │
     0.5 ├┄┄┄┄┄┄┄█████████████████████████████████████████┄┄┄┄┄┄┄┤
         │       ┆ ▀▀▀███████████████████████████████▀▀▀ ┆       │
      y1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄▀█████████████████████████▀┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆                       ┆       ┆       │
         │       ┆       ┆                       ┆       ┆       │
       0 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆                       ┆       ┆       │
     -yl ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         ▐███████████████████████████████████████████████████████▌
         │       ┆       ┆                       ┆       ┆       │
         0      x1      x2                     1-x2    1-x1      1

  zero_ref = False
         ▐███████████████████████████████████████████████████████▌
  0.5+yu ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆                       ┆       ┆       │
       1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆                       ┆       ┆       │
         │       ┆       ┆                       ┆       ┆       │
  0.5+y1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄▄█████████████████████████▄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆ ▄▄▄███████████████████████████████▄▄▄ ┆       │
     0.5 ├┄┄┄┄┄┄┄█████████████████████████████████████████┄┄┄┄┄┄┄┤
         │       ┆ ▀▀▀███████████████████████████████▀▀▀ ┆       │
  0.5-y1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄▀█████████████████████████▀┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆                       ┆       ┆       │
         │       ┆       ┆                       ┆       ┆       │
       0 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆                       ┆       ┆       │
  0.5-yl ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         ▐███████████████████████████████████████████████████████▌
         │       ┆       ┆                       ┆       ┆       │
         0      x1      x2                     1-x2    1-x1      1
  """

  def __init__(self,
               x1: MixedUnits,
               x2: MixedUnits,
               y1: MixedUnits,
               yl: MixedUnits,
               yu: MixedUnits,
               zero_ref: bool = True) -> None:
    """Create a new Mask from systematic definition with a center hexagon

    All inputs are assumed to be positive (negative allowed) and normalized to
    bit duration / amplitude.

    See class docstring for ASCII art diagram of parameters

    Args:
      x1: X distance to 50% corner, UI
      x2: X distance to y2 corner, UI
      y1: Y distance to x2 edge, UI
      yl: Y distance to lower limit, UA
      yu: Y distance to upper limit, UA
      zero_ref: see ASCII art is easiest
    """
    self._x1 = x1
    self._x2 = x2
    self._y1 = y1
    self._yl = yl
    self._yu = yu
    self._zero_ref = zero_ref

    # yapf: disable
    path_center = [
      (x1, 0.5),
      (x2, 1 - y1),
      (1 - x2, 1 - y1),
      (1 - x1, 0.5),
      (1 - x2, y1),
      (x2, y1),
      (x1, 0.5)
    ]

    path_upper = [
      (0, 2),
      (0, 1 + yu),
      (1, 1 + yu),
      (1, 2)
    ]

    path_lower = [
      (0, -1),
      (0, -yl),
      (1, -yl),
      (1, -1)
    ]
    # yapf: enable

    super().__init__(paths=[path_center, path_upper, path_lower])

  def convert_mixed_units(self, ua: float, ui: float) -> MaskHexagon:
    x1 = self._convert_ui(self._x1, ui)
    x2 = self._convert_ui(self._x2, ui)
    y1 = self._convert_ua(self._y1, ua)
    yl = self._convert_ua(self._yl, ua)
    yu = self._convert_ua(self._yu, ua)
    if not self._zero_ref:
      # Flip parameters to proper shape
      y1 = 0.5 - y1
      yl = yl - 0.5
      yu = yu - 0.5
    m = MaskHexagon(x1, x2, y1, yl, yu, zero_ref=self._zero_ref)
    m._converted = True  # pylint: disable=protected-access
    return m

  def adjust(self, factor: float) -> MaskHexagon:
    if factor > 0:
      # x2, x3, and y2 remain proportional to preserve the shape
      x1 = (1 - factor) * self._x1
      x2 = 0.5 - (0.5 - x1) * (0.5 - self._x2) / (0.5 - self._x1)
      y1 = (1 - factor) * self._y1
      yl = (1 - factor) * self._yl
      yu = (1 - factor) * self._yu
    else:
      factor = -factor
      x1 = factor * 0.5 + (1 - factor) * self._x1
      x2 = factor * 0.5 + (1 - factor) * self._x2
      y1 = factor * 0.5 + (1 - factor) * self._y1
      yl = factor * 0.5 + (1 - factor) * self._yl
      yu = factor * 0.5 + (1 - factor) * self._yu
    m = MaskHexagon(x1, x2, y1, yl, yu, zero_ref=self._zero_ref)
    m._converted = self._converted  # pylint: disable=protected-access
    return m

  def to_dict(self) -> dict:
    return {
        "type": "hexagon",
        "x1": self._x1,
        "x2": self._x2,
        "y1": self._y1,
        "yl": self._yl,
        "yu": self._yu
    }


class MaskDecagon(Mask):
  """A Mask with a center decagon (10-sides)

  zero_ref = True (default)
         ▐███████████████████████████████████████████████████████▌
  1 + yu ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆       ┆       ┆       ┆       ┆       │
       1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆       ┆       ┆       ┆       ┆       │
  1 - y1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄▄▄▄▄█████████▄▄▄▄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
  1 - y2 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄▄█████████████████████████▄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆ ▄▄▄███████████████████████████████▄▄▄ ┆       │
     0.5 ├┄┄┄┄┄┄┄█████████████████████████████████████████┄┄┄┄┄┄┄┤
         │       ┆ ▀▀▀███████████████████████████████▀▀▀ ┆       │
      y2 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄▀█████████████████████████▀┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
      y1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄▀▀▀▀█████████▀▀▀▀┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆       ┆       ┆       ┆       ┆       │
       0 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆       ┆       ┆       ┆       ┆       │
     -yl ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         ▐███████████████████████████████████████████████████████▌
         │       ┆       ┆       ┆       ┆       ┆       ┆       │
         0      x1      x2      x3     1-x3    1-x2    1-x1      1

  zero_ref = False
         ▐███████████████████████████████████████████████████████▌
  0.5+yu ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆       ┆       ┆       ┆       ┆       │
       1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆       ┆       ┆       ┆       ┆       │
  0.5+y1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄▄▄▄▄█████████▄▄▄▄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
  0.5+y2 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄▄█████████████████████████▄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆ ▄▄▄███████████████████████████████▄▄▄ ┆       │
     0.5 ├┄┄┄┄┄┄┄█████████████████████████████████████████┄┄┄┄┄┄┄┤
         │       ┆ ▀▀▀███████████████████████████████▀▀▀ ┆       │
  0.5-y2 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄▀█████████████████████████▀┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
  0.5-y1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄▀▀▀▀█████████▀▀▀▀┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆       ┆       ┆       ┆       ┆       │
       0 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         │       ┆       ┆       ┆       ┆       ┆       ┆       │
  0.5-yl ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         ▐███████████████████████████████████████████████████████▌
         │       ┆       ┆       ┆       ┆       ┆       ┆       │
         0      x1      x2      x3     1-x3    1-x2    1-x1      1
  """

  def __init__(self,
               x1: MixedUnits,
               x2: MixedUnits,
               x3: MixedUnits,
               y1: MixedUnits,
               y2: MixedUnits,
               yl: MixedUnits,
               yu: MixedUnits,
               zero_ref: bool = True) -> None:
    """Create a new Mask from systematic definition with a center decagon

    All inputs are assumed to be positive (negative allowed) and normalized to
    bit duration / amplitude.

    See class docstring for ASCII art diagram of parameters

    Args:
      x1: X distance to 50% corner, UI
      x2: X distance to y2 corner, UI
      x3: X distance to y1 corner, UI
      y1: Y distance to x3 edge, UI
      y2: Y distance to x2 corner, UA
      yl: Y distance to lower limit, UA
      yu: Y distance to upper limit, UA
      zero_ref: see ASCII art is easiest
    """
    self._x1 = x1
    self._x2 = x2
    self._x3 = x3
    self._y1 = y1
    self._y2 = y2
    self._yl = yl
    self._yu = yu
    self._zero_ref = zero_ref

    # yapf: disable
    path_center = [
      (x1, 0.5),
      (x2, 1 - y2),
      (x3, 1 - y1),
      (1 - x3, 1 - y1),
      (1 - x2, 1 - y2),
      (1 - x1, 0.5),
      (1 - x2, y2),
      (1 - x3, y1),
      (x3, y1),
      (x2, y2),
      (x1, 0.5)
    ]

    path_upper = [
      (0, 2),
      (0, 1 + yu),
      (1, 1 + yu),
      (1, 2)
    ]

    path_lower = [
      (0, -1),
      (0, -yl),
      (1, -yl),
      (1, -1)
    ]
    # yapf: enable

    super().__init__(paths=[path_center, path_upper, path_lower])

  def convert_mixed_units(self, ua: float, ui: float) -> MaskDecagon:
    x1 = self._convert_ui(self._x1, ui)
    x2 = self._convert_ui(self._x2, ui)
    x3 = self._convert_ui(self._x3, ui)
    y1 = self._convert_ua(self._y1, ua)
    y2 = self._convert_ua(self._y2, ua)
    yl = self._convert_ua(self._yl, ua)
    yu = self._convert_ua(self._yu, ua)
    if not self._zero_ref:
      # Flip parameters to proper shape
      y1 = 0.5 - y1
      y2 = 0.5 - y2
      yl = yl - 0.5
      yu = yu - 0.5
    m = MaskDecagon(x1, x2, x3, y1, y2, yl, yu, zero_ref=self._zero_ref)
    m._converted = True  # pylint: disable=protected-access
    return m

  def adjust(self, factor: float) -> MaskDecagon:
    if factor > 0:
      # x2, x3, and y2 remain proportional to preserve the shape
      x1 = (1 - factor) * self._x1
      x2 = 0.5 - (0.5 - x1) * (0.5 - self._x2) / (0.5 - self._x1)
      x3 = 0.5 - (0.5 - x1) * (0.5 - self._x3) / (0.5 - self._x1)
      y1 = (1 - factor) * self._y1
      y2 = 0.5 - (0.5 - y1) * (0.5 - self._y2) / (0.5 - self._y1)
      yl = (1 - factor) * self._yl
      yu = (1 - factor) * self._yu
    else:
      factor = -factor
      x1 = factor * 0.5 + (1 - factor) * self._x1
      x2 = factor * 0.5 + (1 - factor) * self._x2
      x3 = factor * 0.5 + (1 - factor) * self._x3
      y1 = factor * 0.5 + (1 - factor) * self._y1
      y2 = factor * 0.5 + (1 - factor) * self._y2
      yl = factor * 0.5 + (1 - factor) * self._yl
      yu = factor * 0.5 + (1 - factor) * self._yu
    m = MaskDecagon(x1, x2, x3, y1, y2, yl, yu, zero_ref=self._zero_ref)
    m._converted = self._converted  # pylint: disable=protected-access
    return m

  def to_dict(self) -> dict:
    return {
        "type": "decagon",
        "x1": self._x1,
        "x2": self._x2,
        "x3": self._x3,
        "y1": self._y1,
        "y2": self._y2,
        "yl": self._yl,
        "yu": self._yu
    }
