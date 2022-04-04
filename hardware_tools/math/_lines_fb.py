"""Lines helper functions, see math.lines
Fallback version for no cython
"""

from typing import List, Tuple

import numpy as np


def _draw_segment(x1: int, y1: int, x2: int, y2: int, grid: np.ndarray) -> None:
  """Interpolate a line segment onto a grid

  The value of grid[x,y] is incremented for each x,y in the line from (x0,y0) up
  to but not including (x1, y1).

  Args:
    x1: X-coordinate of first point
    y1: Y-coordinate of first point
    x2: X-coordinate of second point
    y2: Y-coordinate of second point
    grid: Grid to interpolate onto
  """
  n_x, n_y = grid.shape

  dx = abs(x2 - x1)
  dy = abs(y2 - y1)

  sx = 0
  if x1 < x2:
    sx = 1
  else:
    sx = -1
  sy = 0
  if y1 < y2:
    sy = 1
  else:
    sy = -1

  err = dx - dy

  while True:
    # Note: this test is moved before setting
    # the value, so we don't set the last point.
    if x1 == x2 and y1 == y2:
      break

    if (0 <= x1 < n_x) and (0 <= y1 < n_y):
      grid[x1, y1] += 1

    e2 = 2 * err
    if e2 > -dy:
      err -= dy
      x1 += sx
    if e2 < dx:
      err += dx
      y1 += sy


def draw(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> None:
  """Linearly interpolate a series of points onto a grid

  Args:
    x: Series of x values
    y: Series of y values
    grid: Grid to interpolate onto
  """
  for i in range(x.size - 1):
    x1 = x[i]
    y1 = y[i]
    x2 = x[i + 1]
    y2 = y[i + 1]
    _draw_segment(x1, y1, x2, y2, grid)
  if (0 <= x2 < grid.shape[0]) and (0 <= y2 < grid.shape[1]):
    grid[x2, y2] += 1


def draw_points(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> None:
  """Plots a series of points onto a grid

  Args:
    x: Series of x values
    y: Series of y values
    grid: Grid to plot onto
  """
  for i in range(x.size):
    x1 = x[i]
    y1 = y[i]
    if (0 <= x1 < grid.shape[0]) and (0 <= y1 < grid.shape[1]):
      grid[x1, y1] += 1


def crossing(axis_return: list,
             axis_search: list,
             i: int,
             value: float,
             step_forward: bool = False,
             n: int = 0) -> Tuple[float, int]:
  """Get crossing value in the axis_return by searching and interpolating
  axis_search

  Example:
    axis_return = [0,   1,   2,  3,   4]
    axis_search = [10, 20, -10, 50, -50]
    value = 0
    if i = 1, return (np.nan, 0)
    if i = 2, return (1.66667, 2)
    if i = 3, return (2.16667, 3)
    if i = 4, return (3.5, 4)

  Args:
    axis_return: Data axis to return value for
    axis_search: Data axis to find and interpolate value
    i: Beginning index to look at, step until found
    value: Value to find and interpolate in axis_search
    step_forward: True will iterate forward until crossing, False will iterate
      backwards
    n: length of data axis, 0 will read len()

  Returns:
    (interpolated value, index)
  """
  # Back up
  n = len(axis_search) if n == 0 else n
  while 0 < i < n and (axis_search[i] > value) == (axis_search[i - 1] > value):
    i += 1 if step_forward else -1

  if i < 1 or i >= n:
    return (np.nan, i)

  x1 = axis_return[i - 1]
  x2 = axis_return[i]
  y1 = axis_search[i - 1]
  y2 = axis_search[i]
  v = (x2 - x1) / (y2 - y1) * (value - y1) + x1
  return (v, i)


def edges(t: list, y: list, y_rise: float, y_half: float,
          y_fall: float) -> Tuple[list, list]:
  """Get rising and falling edges of a waveform with hysteresis

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    y_rise: Rising threshold
    y_half: Interpolated edge value
    y_fall: Falling threshold

  Returns:
    (rising edges, falling edges) timestamps
  """
  edges_rise = []
  edges_fall = []

  state_low = y[0] < y_fall

  for i in range(1, len(y)):
    if state_low:
      if y[i] > y_rise:
        state_low = False
        # interpolate y_half crossing
        edges_rise.append(crossing(t, y, i, y_half)[0])
    else:
      if y[i] < y_fall:
        state_low = True
        # interpolate y_half crossing
        edges_fall.append(crossing(t, y, i, y_half)[0])

  return (edges_rise, edges_fall)


def edges_np(t: np.ndarray, y: np.ndarray, y_rise: float, y_half: float,
             y_fall: float) -> Tuple[np.ndarray, np.ndarray]:
  """Get rising and falling edges of a waveform with hysteresis

  Converts numpy array to list for faster processing then casts results into
  numpy array

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    y_rise: Rising threshold
    y_half: Interpolated edge value
    y_fall: Falling threshold

  Returns:
    (rising edges, falling edges) timestamp
  """
  n = len(t)
  edges_rise = np.zeros(n)
  edges_fall = np.zeros(n)
  i_rise = 0
  i_fall = 0

  t = t.tolist()
  y = y.tolist()

  state_low = y[0] < y_fall

  for i in range(1, n):
    if state_low:
      if y[i] > y_rise:
        state_low = False
        # interpolate y_half crossing
        edges_rise[i_rise] = crossing(t, y, i, y_half, False, n)[0]
        i_rise += 1
    else:
      if y[i] < y_fall:
        state_low = True
        # interpolate y_half crossing
        edges_fall[i_fall] = crossing(t, y, i, y_half, False, n)[0]
        i_fall += 1

  return (edges_rise[:i_rise], edges_fall[:i_fall])


def intersection(p1: float,
                 p2: float,
                 q1: float,
                 q2: float,
                 r1: float,
                 r2: float,
                 s1: float,
                 s2: float,
                 segments: bool = True) -> Tuple[float]:
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
  d = (p1 - q1) * (r2 - s2) - (p2 - q2) * (r1 - s1)
  if d == 0:
    return None
  t_n = (p1 - r1) * (r2 - s2) - (p2 - r2) * (r1 - s1)
  u_n = (p1 - r1) * (p2 - q2) - (p2 - r2) * (p1 - q1)
  if segments:
    if d > 0:
      if t_n < 0 or u_n < 0 or t_n > d or u_n > d:
        return None
    else:
      if t_n > 0 or u_n > 0 or t_n < d or u_n < d:
        return None
  t = t_n / d
  i1 = p1 + t * (q1 - p1)
  i2 = p2 + t * (q2 - p2)
  return i1, i2


def hits(t: list, y: list, paths: List[List[tuple]]) -> List[tuple]:
  """Get all intersections between waveform and paths (mask lines)

  Might double count corners due to floating point rounding

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
        point = intersection(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                             path_y[ii], path_t[ii - 1], path_y[ii - 1])
        if point is not None:
          intersections.append(point)
  return intersections


def hits_np(t: np.ndarray, y: np.ndarray,
            paths: List[List[tuple]]) -> np.ndarray:
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
  return np.array(hits(t.tolist(), y.tolist(), paths))


def is_hitting(t: list, y: list, paths: List[List[tuple]]) -> bool:
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
        point = intersection(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                             path_y[ii], path_t[ii - 1], path_y[ii - 1])
        if point is not None:
          return True
  return False


def is_hitting_np(t: np.ndarray, y: np.ndarray,
                  paths: List[List[tuple]]) -> bool:
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
