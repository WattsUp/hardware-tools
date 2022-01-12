"""Edge finding algorithm
"""

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
cdef tuple get_crossing_c(list axis_return,
                        list axis_search,
                        int i,
                        float value,
                        bint step_forward):
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

  Returns:
    (interpolated value, index)
  """

  # Back up
  cdef Py_ssize_t n = len(axis_search)
  while i > 0 and i < n and (axis_search[i] > value) == (axis_search[i - 1] >
                                                         value):
    i += 1 if step_forward else -1

  if i < 1 or i >= n:
    return (np.nan, 0)

  v = (axis_return[i] - axis_return[i - 1]) / (axis_search[i] - axis_search[
      i - 1]) * (value - axis_search[i - 1]) + axis_return[i - 1]
  return (v, i)

def get_crossing(list axis_return,
                 list axis_search,
                 int i,
                 float value,
                 bint step_forward = False) -> tuple:
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

  Returns:
    (interpolated value, index)
  """
  return get_crossing_c(axis_return, axis_search, i, value, step_forward)

@cython.boundscheck(False)
cdef tuple get_c(
      list t,
      list y,
      float y_rise,
      float y_half,
      float y_fall):
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
  cdef list edges_rise = []
  cdef list edges_fall = []

  cdef bint state_low = y[0] < y_fall

  for i in range(1, len(y)):
    if state_low:
      if y[i] > y_rise:
        state_low = False
        # interpolate y_half crossing
        edges_rise.append(get_crossing(t, y, i, y_half)[0])
    else:
      if y[i] < y_fall:
        state_low = True
        # interpolate y_half crossing
        edges_fall.append(get_crossing(t, y, i, y_half)[0])

  return (edges_rise, edges_fall)

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

def get_np(
      np.ndarray t,
      np.ndarray y,
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
  edges_rise, edges_fall = get_c(t.tolist(), y.tolist(), y_rise, y_half, y_fall)
  return (np.array(edges_rise), np.array(edges_fall))
