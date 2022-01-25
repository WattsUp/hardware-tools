"""Intersection finding algorithm
"""

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple line_params_c(np.float64_t a1, np.float64_t a2, np.float64_t b1, np.float64_t b2):
  """Convert pair of points to line parameters

  Args:
    a1: Coordinate of first point
    a2: Coordinate of first point
    b1: Coordinate of second point
    b2: Coordinate of second point

  Returns:
    list[d2, d1, det([a, b])]
  """
  cdef np.float64_t d2 = a2 - b2
  cdef np.float64_t d1 = b1 - a1
  cdef np.float64_t det = b1 * a2 - a1 * b2 # Already negated
  return d2, d1, det

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple get_c(np.float64_t p1,
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
  # Get intersection point of lines [not segments]
  cdef tuple l1 = line_params_c(p1, p2, q1, q2)
  cdef tuple l2 = line_params_c(r1, r2, s1, s2)
  cdef np.float64_t det = l1[0] * l2[1] - l1[1] * l2[0]
  if det == 0:
    return None
  cdef np.float64_t det1 = l2[1] * l1[2] - l2[2] * l1[1]
  cdef np.float64_t det2 = l1[0] * l2[2] - l1[2] * l2[0]
  cdef np.float64_t i1 = det1 / det
  cdef np.float64_t i2 = det2 / det

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
  return get_c(p1, p2, q1, q2, r1, r2, s1, s2, segments)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list get_hits_c(list t, list y, list paths):
  """Get all intersections between waveform and paths (mask lines)

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
        intersection = get_c(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                           path_y[ii], path_t[ii - 1], path_y[ii - 1], True)
        if intersection is not None:
          intersections.append(intersection)
  return intersections

def get_hits(t: list, y: list, paths: list[list[tuple]]) -> list[tuple]:
  """Get all intersections between waveform and paths (mask lines)

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    paths: list of (path: list of points defining open path (n-1 segments))

  Returns:
    List of intersection points (t, y)
  """
  return get_hits_c(t, y, paths)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=2] get_hits_np_c(np.ndarray[np.float64_t, ndim=1] t,
                        np.ndarray[np.float64_t, ndim=1] y,
                        list paths):
  """Get all intersections between waveform and paths (mask lines)

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
        intersection = get_c(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                           path_y[ii], path_t[ii - 1], path_y[ii - 1], True)
        if intersection is not None:
          if i_intersection >= n_intersections:
            intersections = np.concatenate((intersections, np.zeros((n_t, 2))), axis=0)
            n_intersections += n_t
          intersections[i_intersection] = intersection
          i_intersection += 1
  return intersections[:i_intersection]

def get_hits_np(np.ndarray[np.float64_t, ndim=1] t,
                np.ndarray[np.float64_t, ndim=1] y,
                list paths) -> np.ndarray:
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
  return get_hits_np_c(t, y, paths)

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
        intersection = get_c(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                           path_y[ii], path_t[ii - 1], path_y[ii - 1], True)
        if intersection is not None:
          return True
  return False

def is_hitting(t: list, y: list, paths: list[list[tuple]]) -> bool:
  """Check for any intersections between waveform and paths (mask lines)

  Args:
    t: Waveform time array [t0, t1, ..., tn]
    y: Waveform data array [y0, y1, ..., yn]
    paths: list of (path: list of points defining open path (n-1 segments))

  Returns:
    True if there exist at least one intersection, False otherwise
  """
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
        intersection = get_c(t[i], y[i], t[i - 1], y[i - 1], path_t[ii],
                           path_y[ii], path_t[ii - 1], path_y[ii - 1], True)
        if intersection is not None:
          return True
  return False


@cython.boundscheck(False)
@cython.wraparound(False)
def is_hitting_np(np.ndarray[np.float64_t, ndim=1] t,
                  np.ndarray[np.float64_t, ndim=1] y,
                  list paths) -> bool:
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
  return is_hitting_np_c(t, y, paths)
