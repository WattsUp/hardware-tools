"""Lines helper functions, see math.lines
"""

from typing import List, Tuple

import numpy as np
cimport numpy as np
cimport cython

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _draw_segment_c(np.int32_t x1,
                        np.int32_t y1,
                        np.int32_t x2,
                        np.int32_t y2,
                        np.ndarray[np.int32_t, ndim=2] grid,
                        Py_ssize_t n_x,
                        Py_ssize_t n_y):
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
  cdef np.int32_t e2, sx, sy, err
  cdef np.int32_t dx, dy


  if x2 > x1:
    dx = x2 - x1
  else:
    dx = x1 - x2
  if y2 > y1:
    dy = y2 - y1
  else:
    dy = y1 - y2

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
    # Note: this test occurs before increment the
    # grid value, so we don't count the last point.
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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void draw_c(np.ndarray[np.int32_t, ndim=1] x,
         np.ndarray[np.int32_t, ndim=1] y,
         np.ndarray[np.int32_t, ndim=2] grid):
  """Linearly interpolate a series of points onto a grid

  Args:
    x: Series of x values
    y: Series of y values
    grid: Grid to interpolate onto
  """
  cdef Py_ssize_t n = x.shape[0]
  cdef Py_ssize_t n_x = grid.shape[0]
  cdef Py_ssize_t n_y = grid.shape[1]
  cdef Py_ssize_t i
  cdef np.int32_t x1, y1, x2, y2

  for i in range(n - 1):
    x1 = x[i]
    y1 = y[i]
    x2 = x[i + 1]
    y2 = y[i + 1]
    _draw_segment_c(x1, y1, x2, y2, grid, n_x, n_y)
  if (0 <= x2 < n_x) and (0 <= y2 < n_y):
    grid[x2, y2] += 1

def draw(np.ndarray x,
         np.ndarray y,
         np.ndarray grid) -> None:
  draw_c(x, y, grid)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void draw_points_c(np.ndarray[np.int32_t, ndim=1] x,
         np.ndarray[np.int32_t, ndim=1] y,
         np.ndarray[np.int32_t, ndim=2] grid):
  """Plots a series of points onto a grid

  Args:
    x: Series of x values
    y: Series of y values
    grid: Grid to plot onto
  """
  cdef Py_ssize_t n = x.shape[0]
  cdef Py_ssize_t n_x = grid.shape[0]
  cdef Py_ssize_t n_y = grid.shape[1]
  cdef Py_ssize_t i
  cdef np.int32_t x1, y1

  for i in range(n):
    x1 = x[i]
    y1 = y[i]
    if (0 <= x1 < n_x) and (0 <= y1 < n_y):
      grid[x1, y1] += 1

def draw_points(np.ndarray x,
         np.ndarray y,
         np.ndarray grid) -> None:
  draw_points_c(x, y, grid)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple crossing_c(list axis_return,
                        list axis_search,
                        Py_ssize_t i,
                        np.float64_t value,
                        bint step_forward,
                        Py_ssize_t n):
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
    n: length of data axis

  Returns:
    (interpolated value, index)
  """
  if i < 1 or i >= n:
    return (np.nan, i)

  # Back up
  if step_forward:
    while (axis_search[i] > value) == (axis_search[i - 1] > value):
      i += 1
      if i >= n:
        return (np.nan, i)
  else:
    while (axis_search[i] > value) == (axis_search[i - 1] > value):
      i += -1
      if i < 1:
        return (np.nan, i)

  cdef np.float64_t x1 = axis_return[i - 1]
  cdef np.float64_t x2 = axis_return[i]
  cdef np.float64_t y1 = axis_search[i - 1]
  cdef np.float64_t y2 = axis_search[i]
  cdef np.float64_t v = (x2 - x1) / (y2 - y1) * (value - y1) + x1
  return (v, i)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float64_t crossing_c_v(list axis_return,
                        list axis_search,
                        Py_ssize_t i,
                        np.float64_t value,
                        bint step_forward,
                        Py_ssize_t n):
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
    n: length of data axis

  Returns:
    interpolated value
  """
  if i < 1 or i >= n:
    return np.nan

  # Back up
  if step_forward:
    while (axis_search[i] > value) == (axis_search[i - 1] > value):
      i += 1
      if i >= n:
        return np.nan
  else:
    while (axis_search[i] > value) == (axis_search[i - 1] > value):
      i += -1
      if i < 1:
        return np.nan

  cdef np.float64_t x1 = axis_return[i - 1]
  cdef np.float64_t x2 = axis_return[i]
  cdef np.float64_t y1 = axis_search[i - 1]
  cdef np.float64_t y2 = axis_search[i]
  cdef np.float64_t v = (x2 - x1) / (y2 - y1) * (value - y1) + x1
  return v

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float64_t crossing_c_v_np(np.ndarray[np.float64_t, ndim=1] axis_return,
                        np.ndarray[np.float64_t, ndim=1] axis_search,
                        Py_ssize_t i,
                        np.float64_t value,
                        bint step_forward,
                        Py_ssize_t n):
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
    n: length of data axis

  Returns:
    interpolated value
  """
  if i < 1 or i >= n:
    return np.nan

  # Back up
  if step_forward:
    while (axis_search[i] > value) == (axis_search[i - 1] > value):
      i += 1
      if i >= n:
        return np.nan
  else:
    while (axis_search[i] > value) == (axis_search[i - 1] > value):
      i += -1
      if i < 1:
        return np.nan

  cdef np.float64_t x1 = axis_return[i - 1]
  cdef np.float64_t x2 = axis_return[i]
  cdef np.float64_t y1 = axis_search[i - 1]
  cdef np.float64_t y2 = axis_search[i]
  cdef np.float64_t v = (x2 - x1) / (y2 - y1) * (value - y1) + x1
  return v

def crossing(list axis_return,
                 list axis_search,
                 int i,
                 float value,
                 bint step_forward = False,
                 int n = 0) -> Tuple[float]:
  n = len(axis_search) if n == 0 else n
  return crossing_c(axis_return, axis_search, i, value, step_forward, n)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple edges_c(
      list t,
      list y,
      np.float64_t y_rise,
      np.float64_t y_half,
      np.float64_t y_fall):
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
  cdef Py_ssize_t n = len(t)
  cdef list edges_rise = [0.0] * n
  cdef list edges_fall = [0.0] * n

  cdef bint state_low = y[0] < y_fall

  cdef Py_ssize_t i
  cdef Py_ssize_t i_rise = 0
  cdef Py_ssize_t i_fall = 0

  for i in range(1, n):
    if state_low:
      if y[i] > y_rise:
        state_low = False
        # interpolate y_half crossing
        edges_rise[i_rise] = crossing_c_v(t, y, i, y_half, False, n)
        i_rise += 1
    else:
      if y[i] < y_fall:
        state_low = True
        # interpolate y_half crossing
        edges_fall[i_fall] = crossing_c_v(t, y, i, y_half, False, n)
        i_fall += 1

  return (edges_rise[:i_rise], edges_fall[:i_fall])

def edges(
      list t,
      list y,
      float y_rise,
      float y_half,
      float y_fall) -> Tuple[list, list]:
  return edges_c(t, y, y_rise, y_half, y_fall)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple edges_np_c(
      np.ndarray[np.float64_t, ndim=1] t,
      np.ndarray[np.float64_t, ndim=1] y,
      np.float64_t y_rise,
      np.float64_t y_half,
      np.float64_t y_fall):
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
  cdef Py_ssize_t n = len(t)
  cdef np.ndarray[np.float64_t, ndim=1] edges_rise = np.zeros(n)
  cdef np.ndarray[np.float64_t, ndim=1] edges_fall = np.zeros(n)

  cdef bint state_low = y[0] < y_fall

  cdef Py_ssize_t i
  cdef Py_ssize_t i_rise = 0
  cdef Py_ssize_t i_fall = 0

  for i in range(1, n):
    if state_low:
      if y[i] > y_rise:
        state_low = False
        # interpolate y_half crossing
        edges_rise[i_rise] = crossing_c_v_np(t, y, i, y_half, False, n)
        i_rise += 1
    else:
      if y[i] < y_fall:
        state_low = True
        # interpolate y_half crossing
        edges_fall[i_fall] = crossing_c_v_np(t, y, i, y_half, False, n)
        i_fall += 1

  return (edges_rise[:i_rise], edges_fall[:i_fall])

def edges_np(
      np.ndarray[np.float64_t, ndim=1] t,
      np.ndarray[np.float64_t, ndim=1] y,
      float y_rise,
      float y_half,
      float y_fall) -> Tuple[np.ndarray, np.ndarray]:
  return edges_np_c(t, y, y_rise, y_half, y_fall)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef LineParams line_params_c(np.float64_t a1, np.float64_t a2, np.float64_t b1, np.float64_t b2):
  """Convert pair of points to line parameters

  Args:
    a1: Coordinate of first point
    a2: Coordinate of first point
    b1: Coordinate of second point
    b2: Coordinate of second point

  Returns:
    List[d2, d1, det([a, b])]
  """
  cdef LineParams params = LineParams()
  params.d2 = a2 - b2
  params.d1 = b1 - a1
  params.det = b1 * a2 - a1 * b2 # Already negated
  return params

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Point intersection_c(np.float64_t p1,
                 np.float64_t p2,
                 np.float64_t q1,
                 np.float64_t q2,
                 np.float64_t r1,
                 np.float64_t r2,
                 np.float64_t s1,
                 np.float64_t s2,
                 bint segments):
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
  cdef np.float64_t d = (p1 - q1) * (r2 - s2) - (p2 - q2) * (r1 - s1)
  if d == 0:
    return None
  cdef np.float64_t t_n = (p1 - r1) * (r2 - s2) - (p2 - r2) * (r1 - s1)
  cdef np.float64_t u_n = (p1 - r1) * (p2 - q2) - (p2 - r2) * (p1 - q1)
  if segments:
    if d > 0:
      if t_n < 0 or u_n < 0 or t_n > d or u_n > d:
        return None
    else:
      if t_n > 0 or u_n > 0 or t_n < d  or u_n < d:
        return None
  cdef Point point = Point()
  t_n = t_n / d
  point.x = p1 + t_n * (q1 - p1)
  point.y = p2 + t_n * (q2 - p2)
  return point

def intersection(p1: float,
        p2: float,
        q1: float,
        q2: float,
        r1: float,
        r2: float,
        s1: float,
        s2: float,
        segments: bool = True) -> Tuple(float):
  point = intersection_c(p1, p2, q1, q2, r1, r2, s1, s2, segments)
  if point is None:
    return None
  return (point.x, point.y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list hits_c(list t, list y, list paths):
  """Get all intersections between waveform and paths (mask lines)

  Might double count corners due to floating point rounding

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    paths: list of (path: list of points defining open path (n-1 segments))

  Returns:
    List of intersection points (t, y)
  """
  cdef list intersections = []
  cdef list path_t
  cdef list path_y
  cdef np.float64_t min_t
  cdef np.float64_t max_t
  cdef np.float64_t min_y
  cdef np.float64_t max_y
  cdef np.float64_t point_t
  cdef np.float64_t point_y
  
  cdef Py_ssize_t n_p = len(paths)
  cdef Py_ssize_t n_t = len(t)
  cdef Py_ssize_t i_p = 0
  cdef Py_ssize_t i = 0
  cdef Py_ssize_t ii = 0
  cdef Py_ssize_t n_path = 0

  for i_p in range(n_p):
    n_path = len(paths[i_p])
    min_t = paths[i_p][0][0]
    max_t = min_t
    min_y = paths[i_p][0][1]
    max_y = min_y
    path_t = [min_t]
    path_y = [min_y]
    for ii in range(1, n_path):
      point_t = paths[i_p][ii][0]
      point_y = paths[i_p][ii][1]
      if point_t > max_t:
        max_t = point_t
      elif point_t < min_t:
        min_t = point_t
      if point_y > max_y:
        max_y = point_y
      elif point_y < min_y:
        min_y = point_y
      path_t.append(point_t)
      path_y.append(point_y)

    for i in range(1, n_t):
      if (t[i] < min_t) or (t[i - 1] > max_t):
        continue
      if y[i] > y[i - 1]:
        if (y[i] < min_y) or (y[i - 1] > max_y):
          continue
      else:
        if (y[i] > max_y) or (y[i - 1] < min_y):
          continue

      for ii in range(1, n_path):
        point = intersection_c(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                           path_y[ii], path_t[ii - 1], path_y[ii - 1], True)
        if point is not None:
          intersections.append((point.x, point.y))
  return intersections

def hits(t: list, y: list, paths: List[List[tuple]]) -> List[tuple]:
  return hits_c(t, y, paths)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=2] hits_np_c(np.ndarray[np.float64_t, ndim=1] t,
                        np.ndarray[np.float64_t, ndim=1] y,
                        list paths):
  """Get all intersections between waveform and paths (mask lines)

  Might double count corners due to floating point rounding

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    paths: list of (path: list of points defining open path (n-1 segments))

  Returns:
    List of intersection points (t, y)
  """
  cdef list path_t
  cdef list path_y
  cdef np.float64_t min_t
  cdef np.float64_t max_t
  cdef np.float64_t min_y
  cdef np.float64_t max_y
  cdef np.float64_t point_t
  cdef np.float64_t point_y
  
  cdef Py_ssize_t n_p = len(paths)
  cdef Py_ssize_t n_t = t.shape[0]
  cdef Py_ssize_t i_p = 0
  cdef Py_ssize_t i = 0
  cdef Py_ssize_t ii = 0
  cdef Py_ssize_t n_path = 0

  cdef Py_ssize_t n_intersections = n_t
  cdef np.ndarray[np.float64_t, ndim=2] intersections = np.zeros((n_intersections, 2))
  cdef Py_ssize_t i_intersection = 0

  for i_p in range(n_p):
    n_path = len(paths[i_p])
    min_t = paths[i_p][0][0]
    max_t = min_t
    min_y = paths[i_p][0][1]
    max_y = min_y
    path_t = [min_t]
    path_y = [min_y]
    for ii in range(1, n_path):
      point_t = paths[i_p][ii][0]
      point_y = paths[i_p][ii][1]
      if point_t > max_t:
        max_t = point_t
      elif point_t < min_t:
        min_t = point_t
      if point_y > max_y:
        max_y = point_y
      elif point_y < min_y:
        min_y = point_y
      path_t.append(point_t)
      path_y.append(point_y)

    for i in range(1, n_t):
      if (t[i] < min_t) or (t[i - 1] > max_t):
        continue
      if y[i] > y[i - 1]:
        if (y[i] < min_y) or (y[i - 1] > max_y):
          continue
      else:
        if (y[i] > max_y) or (y[i - 1] < min_y):
          continue

      for ii in range(1, n_path):
        point = intersection_c(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                           path_y[ii], path_t[ii - 1], path_y[ii - 1], True)
        if point is not None:
          if i_intersection >= n_intersections:
            intersections = np.concatenate((intersections, np.zeros((n_t, 2))), axis=0)
            n_intersections += n_t
          intersections[i_intersection, 0] = point.x
          intersections[i_intersection, 1] = point.y
          i_intersection += 1
  return intersections[:i_intersection]

def hits_np(np.ndarray[np.float64_t, ndim=1] t,
                np.ndarray[np.float64_t, ndim=1] y,
                list paths) -> np.ndarray:
  return hits_np_c(t, y, paths)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint is_hitting_c(list t, list y, list paths):
  """Check for any intersections between waveform and paths (mask lines)

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    paths: list of (path: list of points defining open path (n-1 segments))

  Returns:
    True if there exist at least one intersection, False otherwise
  """
  cdef list path_t
  cdef list path_y
  cdef np.float64_t min_t
  cdef np.float64_t max_t
  cdef np.float64_t min_y
  cdef np.float64_t max_y
  cdef np.float64_t point_t
  cdef np.float64_t point_y
  
  cdef Py_ssize_t n_p = len(paths)
  cdef Py_ssize_t n_t = len(t)
  cdef Py_ssize_t i_p = 0
  cdef Py_ssize_t i = 0
  cdef Py_ssize_t ii = 0
  cdef Py_ssize_t n_path = 0

  for i_p in range(n_p):
    n_path = len(paths[i_p])
    min_t = paths[i_p][0][0]
    max_t = min_t
    min_y = paths[i_p][0][1]
    max_y = min_y
    path_t = [min_t]
    path_y = [min_y]
    for ii in range(1, n_path):
      point_t = paths[i_p][ii][0]
      point_y = paths[i_p][ii][1]
      if point_t > max_t:
        max_t = point_t
      elif point_t < min_t:
        min_t = point_t
      if point_y > max_y:
        max_y = point_y
      elif point_y < min_y:
        min_y = point_y
      path_t.append(point_t)
      path_y.append(point_y)

    for i in range(1, n_t):
      if (t[i] < min_t) or (t[i - 1] > max_t):
        continue
      if y[i] > y[i - 1]:
        if (y[i] < min_y) or (y[i - 1] > max_y):
          continue
      else:
        if (y[i] > max_y) or (y[i - 1] < min_y):
          continue

      for ii in range(1, n_path):
        point = intersection_c(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                           path_y[ii], path_t[ii - 1], path_y[ii - 1], True)
        if point is not None:
          return True
  return False

def is_hitting(t: list, y: list, paths: List[List[tuple]]) -> bool:
  return is_hitting_c(t, y, paths)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint is_hitting_np_c(np.ndarray[np.float64_t, ndim=1] t,
                          np.ndarray[np.float64_t, ndim=1] y,
                          list paths):
  """Check for any intersections between waveform and paths (mask lines)

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    paths: list of (path: list of points defining open path (n-1 segments))

  Returns:
    True if there exist at least one intersection, False otherwise
  """
  cdef list path_t
  cdef list path_y
  cdef np.float64_t min_t
  cdef np.float64_t max_t
  cdef np.float64_t min_y
  cdef np.float64_t max_y
  cdef np.float64_t point_t
  cdef np.float64_t point_y
  
  cdef Py_ssize_t n_p = len(paths)
  cdef Py_ssize_t n_t = t.shape[0]
  cdef Py_ssize_t i_p = 0
  cdef Py_ssize_t i = 0
  cdef Py_ssize_t ii = 0
  cdef Py_ssize_t n_path = 0

  for i_p in range(n_p):
    n_path = len(paths[i_p])
    min_t = paths[i_p][0][0]
    max_t = min_t
    min_y = paths[i_p][0][1]
    max_y = min_y
    path_t = [min_t]
    path_y = [min_y]
    for ii in range(1, n_path):
      point_t = paths[i_p][ii][0]
      point_y = paths[i_p][ii][1]
      if point_t > max_t:
        max_t = point_t
      elif point_t < min_t:
        min_t = point_t
      if point_y > max_y:
        max_y = point_y
      elif point_y < min_y:
        min_y = point_y
      path_t.append(point_t)
      path_y.append(point_y)

    for i in range(1, n_t):
      if (t[i] < min_t) or (t[i - 1] > max_t):
        continue
      if y[i] > y[i - 1]:
        if (y[i] < min_y) or (y[i - 1] > max_y):
          continue
      else:
        if (y[i] > max_y) or (y[i - 1] < min_y):
          continue

      for ii in range(1, n_path):
        point = intersection_c(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                           path_y[ii], path_t[ii - 1], path_y[ii - 1], True)
        if point is not None:
          return True
  return False


@cython.boundscheck(False)
@cython.wraparound(False)
def is_hitting_np(np.ndarray[np.float64_t, ndim=1] t,
                  np.ndarray[np.float64_t, ndim=1] y,
                  list paths) -> bool:
  return is_hitting_np_c(t, y, paths)
