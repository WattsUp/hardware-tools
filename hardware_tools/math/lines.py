"""Line math operations

Intersections, edges, drawing, etc.
"""

from __future__ import annotations

from typing import List

import numpy as np

try:
  from hardware_tools.math._lines import *  # pylint: disable=wildcard-import
  # raise ImportError
except ImportError:
  print(f"The cython version of {__name__} is not available")
  from hardware_tools.math._lines_fb import *  # pylint: disable=wildcard-import, unused-wildcard-import


class Point2D:
  """2D Point class with an x-coordinate and y-coordinate

  Attributes:
    x: X-coordinate (horizontal)
    y: Y-coordinate (vertical)
  """

  def __init__(self, x: float, y: float) -> None:
    """Initialize a Point2D

    Args:
      x: X-coordinate (horizontal)
      y: Y-coordinate (vertical)
    """
    self.x = x
    self.y = y

  def __str__(self) -> str:
    return f"({self.x}, {self.y})"

  def in_rect(self, start: Point2D, end: Point2D) -> bool:
    """Check if point lies in rectangle formed by start and stop

    Args:
      start: One corner of rectangle
      end: Second corner of rectangle

    Returns:
      True if self is within all edges of rectangle or on an edge
    """
    if ((self.x <= max(start.x, end.x)) and (self.x >= min(start.x, end.x)) and
        (self.y <= max(start.y, end.y)) and (self.y >= min(start.y, end.y))):
      return True
    return False

  @staticmethod
  def orientation(p: Point2D, q: Point2D, r: Point2D) -> int:
    """Compute the orientation of three points from p->q->r

    Anticlockwise:
      r--q
        /
       /
      p

    Linear:
      p---q---r

    Clockwise:
         q-----r
        /
       /
      p

    Args:
      p: Point 1
      q: Point 2
      r: Point 3

    Returns:
      -1: anticlockwise
      0: colinear
      1: clockwise
    """
    val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y))
    norm = ((p.x**2 + p.y**2) * (q.x**2 + q.y**2) * (r.x**2 + r.y**2))**(1 / 3)
    val = val / norm
    if abs(val) < np.finfo(type(val)).eps * 10:
      return 0
    return np.sign(val)


class Line2D:
  """2D Line formed between two points

  Attributes:
    p: First point
    q: second point
  """

  def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
    """Initialize Line2D

    Args:
      x1: X-coordinate of first point
      y1: Y-coordinate of first point
      x2: X-coordinate of second point
      y2: Y-coordinate of second point
    """
    self.p = Point2D(x1, y1)
    self.q = Point2D(x2, y2)

  def __str__(self) -> str:
    return f"({self.p}, {self.q})"

  def intersecting(self, l: Line2D) -> bool:
    """Check if two line segments intersect

    Checks line segments, not lines of infinite length

    Args:
      l: Second line segment

    Returns:
      True if line segments intersect
    """
    return self.intersecting_points(l.p, l.q)

  def intersecting_points(self, p: Point2D, q: Point2D) -> bool:
    """Check if two line segments intersect

    Checks line segments, not lines of infinite length

    Args:
      p: First point of second line segment
      q: Second point of second line segment

    Returns:
      True if line segments intersect
    """
    # 4 Combinations of points
    o1 = Point2D.orientation(self.p, self.q, p)
    o2 = Point2D.orientation(self.p, self.q, q)
    o3 = Point2D.orientation(p, q, self.p)
    o4 = Point2D.orientation(p, q, self.q)

    # General case
    if (o1 != o2) and (o3 != o4):
      return True

    # self.p, self.q, p are colinear and p lies in self
    if (o1 == 0) and p.in_rect(self.p, self.q):
      return True

    # self.p, self.q, q are colinear and q lies in self
    if (o2 == 0) and q.in_rect(self.p, self.q):
      return True

    # p, q, self.p are colinear and self.p lies in pq
    if (o3 == 0) and self.p.in_rect(p, q):
      return True

    # p, q, self.q are colinear and self.q lies in pq
    if (o4 == 0) and self.q.in_rect(p, q):
      # Should always be false because no situation only satisfies only 1 of
      # these checks
      return True  # pragma: no cover

    return False

  def intersection(self, l: Line2D) -> Point2D:
    """Get the intersection of two lines [not line segments]

    Args:
      l: Second line

    Returns:
      Intersection point, None if lines to do not intersect
    """
    return self.intersection_points(l.p, l.q)

  def intersection_points(self, p: Point2D, q: Point2D) -> Point2D:
    """Get the intersection of two lines [not line segments]

    Args:
      p: First point of second line segment
      q: Second point of second line segment

    Returns:
      Intersection point, None if lines are parallel
    """

    def line_params(p: Point2D, q: Point2D) -> List[float]:
      """Convert pair of points to line parameters

      Args:
        p: First point
        q: Second point

      Returns:
        List[dy, dx, det([p, q])]
      """
      dy = p.y - q.y
      dx = q.x - p.x
      det = p.x * q.y - q.x * p.y
      return dy, dx, -det

    l1 = line_params(self.p, self.q)
    l2 = line_params(p, q)
    det = l1[0] * l2[1] - l1[1] * l2[0]
    if det == 0:
      return None
    det_x = l2[1] * l1[2] - l2[2] * l1[1]
    det_y = l1[0] * l2[2] - l1[2] * l2[0]
    return Point2D(det_x / det, det_y / det)
