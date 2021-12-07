"""2D mask to validate waveform shape or other 2D data
"""

from __future__ import annotations

from abc import ABC, abstractmethod


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
    self.paths = paths

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


class MaskDecagon(Mask):
  """A Mask with a center decagon (10-sides)
         ▐███████████████████████████████████████████████████████▌
  1 + y4 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
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
     -y3 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
         ▐███████████████████████████████████████████████████████▌
         │       ┆       ┆       ┆       ┆       ┆       ┆       │
         0      x1      x2      x3     1-x3    1-x2    1-x1      1
  """

  def __init__(self, x1: float, x2: float, x3: float, y1: float, y2: float,
               y3: float, y4: float) -> None:
    """Create a new Mask from systematic definition with a center decagon

    All inputs are assumed to be positive (negative allowed) and normalized to
    bit duration / amplitude.

    See class docstring for ASCII art diagram of parameters

    Args:
      x1: X distance to 50% corner (likely normalized time UI)
      x2: X distance to y2 corner (likely normalized time UI)
      x3: X distance to y1 corner (likely normalized time UI)
      y1: Y distance to x3 edge (likely normalized amplitude UA)
      y2: Y distance to x2 corner (likely normalized amplitude UA)
      y3: Y distance to lower limit (likely normalized amplitude UA)
      y4: Y distance to upper limit (likely normalized amplitude UA)
    """
    self._x1 = x1
    self._x2 = x2
    self._x3 = x3
    self._y1 = y1
    self._y2 = y2
    self._y3 = y3
    self._y4 = y4

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
      (0, 1e3),
      (0, 1 + y4),
      (1, 1 + y4),
      (1, 1e3)
    ]

    path_lower = [
      (0, -1e3),
      (0, -y3),
      (1, -y3),
      (1, -1e3)
    ]
    # yapf: enable

    super().__init__(paths=[path_center, path_upper, path_lower])

  def adjust(self, factor: float) -> MaskDecagon:
    if factor > 0:
      x1 = factor * 0.0 + (1 - factor) * self._x1
      x2 = factor * 0.0 + (1 - factor) * self._x2
      x3 = factor * 0.0 + (1 - factor) * self._x3
      y1 = factor * 0.0 + (1 - factor) * self._y1
      y2 = factor * 0.0 + (1 - factor) * self._y2
      y3 = factor * 0.0 + (1 - factor) * self._y3
      y4 = factor * 0.0 + (1 - factor) * self._y4
    else:
      factor = -factor
      x1 = factor * 0.5 + (1 - factor) * self._x1
      x2 = factor * 0.5 + (1 - factor) * self._x2
      x3 = factor * 0.5 + (1 - factor) * self._x3
      y1 = factor * 0.5 + (1 - factor) * self._y1
      y2 = factor * 0.5 + (1 - factor) * self._y2
      y3 = factor * 0.5 + (1 - factor) * self._y3
      y4 = factor * 0.5 + (1 - factor) * self._y4
    return MaskDecagon(x1, x2, x3, y1, y2, y3, y4)

  def to_dict(self) -> dict:
    return {
        "type": "decagon",
        "x1": self._x1,
        "x2": self._x2,
        "x3": self._x3,
        "y1": self._y1,
        "y2": self._y2,
        "y3": self._y3,
        "y4": self._y4
    }
