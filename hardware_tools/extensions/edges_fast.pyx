"""Edge finding algorithm
"""

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple get_crossing_c(list axis_return,
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
cdef np.float64_t get_crossing_c_v(list axis_return,
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
cdef np.float64_t get_crossing_c_v_np(np.ndarray[np.float64_t, ndim=1] axis_return,
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

def get_crossing(list axis_return,
                 list axis_search,
                 int i,
                 float value,
                 bint step_forward = False,
                 int n = 0) -> tuple:
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
  n = len(axis_search) if n == 0 else n
  return get_crossing_c(axis_return, axis_search, i, value, step_forward, n)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple get_c(
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
        edges_rise[i_rise] = get_crossing_c_v(t, y, i, y_half, False, n)
        i_rise += 1
    else:
      if y[i] < y_fall:
        state_low = True
        # interpolate y_half crossing
        edges_fall[i_fall] = get_crossing_c_v(t, y, i, y_half, False, n)
        i_fall += 1

  return (edges_rise[:i_rise], edges_fall[:i_fall])

def get(
      list t,
      list y,
      float y_rise,
      float y_half,
      float y_fall) -> tuple[list, list]:
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
  return get_c(t, y, y_rise, y_half, y_fall)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple get_np_c(
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
        edges_rise[i_rise] = get_crossing_c_v_np(t, y, i, y_half, False, n)
        i_rise += 1
    else:
      if y[i] < y_fall:
        state_low = True
        # interpolate y_half crossing
        edges_fall[i_fall] = get_crossing_c_v_np(t, y, i, y_half, False, n)
        i_fall += 1

  return (edges_rise[:i_rise], edges_fall[:i_fall])

def get_np(
      np.ndarray[np.float64_t, ndim=1] t,
      np.ndarray[np.float64_t, ndim=1] y,
      float y_rise,
      float y_half,
      float y_fall) -> tuple[np.ndarray, np.ndarray]:
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
  return get_np_c(t, y, y_rise, y_half, y_fall)
