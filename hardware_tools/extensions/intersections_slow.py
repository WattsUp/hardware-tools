"""Intersection finding algorithm
"""

import numpy as np


def get(p1: float,
        p2: float,
        q1: float,
        q2: float,
        r1: float,
        r2: float,
        s1: float,
        s2: float,
        segments: bool = True) -> tuple:
  """Get the intersection point between lines pq and rs

  Args:
    p1: Coordinate of first line segment start point
    p2: Coordinate of first line segment start point
    q1: Coordinate of first line segment end point
    q2: Coordinate of first line segment end point
    r1: Coordinate of second line segment start point
    r2: Coordinate of second line segment start point
    s1: Coordinate of second line segment end point
    s2: Coordinate of second line segment end point
    segments: True treats lines as line segments, False treats as lines of
      infinite length

  Returns:
    Intersection point coordinates, None if not intersecting or parallel
  """

  def line_params(a1: float, a2: float, b1: float, b2: float) -> list[float]:
    """Convert pair of points to line parameters

    Args:
      a1: Coordinate of first point
      a2: Coordinate of first point
      b1: Coordinate of second point
      b2: Coordinate of second point

    Returns:
      list[d2, d1, det([a, b])]
    """
    d2 = a2 - b2
    d1 = b1 - a1
    det = a1 * b2 - b1 * a2
    return d2, d1, -det

  l1 = line_params(p1, p2, q1, q2)
  l2 = line_params(r1, r2, s1, s2)
  det = l1[0] * l2[1] - l1[1] * l2[0]
  if det == 0:
    return None
  det1 = l2[1] * l1[2] - l2[2] * l1[1]
  det2 = l1[0] * l2[2] - l1[2] * l2[0]
  i1 = det1 / det
  i2 = det2 / det

  if not segments:
    return i1, i2

  # Check if point lies on the segments via bounds checking
  if p1 < q1:
    if (i1 < p1) or (i1 > q1):
      return None
  elif p1 > q1:
    if (i1 > p1) or (i1 < q1):
      return None
  if p2 < q2:
    if (i2 < p2) or (i2 > q2):
      return None
  elif p2 > q2:
    if (i2 > p2) or (i2 < q2):
      return None

  if r1 < s1:
    if (i1 < r1) or (i1 > s1):
      return None
  elif r1 > s1:
    if (i1 > r1) or (i1 < s1):
      return None
  if r2 < s2:
    if (i2 < r2) or (i2 > s2):
      return None
  elif r2 > s2:
    if (i2 > r2) or (i2 < s2):
      return None

  return i1, i2


def get_hits(t: list, y: list, paths: list[list[tuple]]) -> list[tuple]:
  """Get all intersections between waveform and paths (mask lines)

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    paths: list of (path: list of points defining open path (n-1 segments))

  Returns:
    List of intersection points (t, y)
  """
  intersections = []
  for path in paths:
    path_t = [p[0] for p in path]
    path_y = [p[1] for p in path]
    min_t = min(path_t)
    max_t = max(path_t)
    min_y = min(path_y)
    max_y = max(path_y)
    for i in range(1, len(t)):
      if (t[i] < min_t) or (t[i - 1] > max_t):
        continue
      if y[i] > y[i - 1]:
        if (y[i] < min_y) or (y[i - 1] > max_y):
          continue
      else:
        if (y[i] > max_y) or (y[i - 1] < min_y):
          continue

      for ii in range(1, len(path)):
        intersection = get(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                           path_y[ii], path_t[ii - 1], path_y[ii - 1])
        if intersection is not None:
          intersections.append(intersection)
  return intersections


def get_hits_np(t: np.ndarray, y: np.ndarray,
                paths: list[list[tuple]]) -> np.ndarray:
  """Get all intersections between waveform and paths (mask lines)

  Converts numpy array to list for faster processing then casts results into
  numpy array

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    paths: list of (path: list of points defining open path (n-1 segments))

  Returns:
    List of intersection points (t, y)
  """
  return np.array(get_hits(t.tolist(), y.tolist(), paths))


def is_hitting(t: list, y: list, paths: list[list[tuple]]) -> bool:
  """Check for any intersections between waveform and paths (mask lines)

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    paths: list of (path: list of points defining open path (n-1 segments))

  Returns:
    True if there exist at least one intersection, False otherwise
  """
  for path in paths:
    path_t = [p[0] for p in path]
    path_y = [p[1] for p in path]
    min_t = min(path_t)
    max_t = max(path_t)
    min_y = min(path_y)
    max_y = max(path_y)
    for i in range(1, len(t)):
      if (t[i] < min_t) or (t[i - 1] > max_t):
        continue
      if y[i] > y[i - 1]:
        if (y[i] < min_y) or (y[i - 1] > max_y):
          continue
      else:
        if (y[i] > max_y) or (y[i - 1] < min_y):
          continue

      for ii in range(1, len(path)):
        intersection = get(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                           path_y[ii], path_t[ii - 1], path_y[ii - 1])
        if intersection is not None:
          return True
  return False


def is_hitting_np(t: np.ndarray, y: np.ndarray,
                  paths: list[list[tuple]]) -> bool:
  """Check for any intersections between waveform and paths (mask lines)

  Converts numpy array to list for faster processing then casts results into
  numpy array

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    paths: list of (path: list of points defining open path (n-1 segments))

  Returns:
    True if there exist at least one intersection, False otherwise
  """
  return is_hitting(t.tolist(), y.tolist(), paths)
