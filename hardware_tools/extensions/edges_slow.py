"""Edge finding algorithm
"""

import numpy as np


def get_crossing(axis_return: list,
                 axis_search: list,
                 i: int,
                 value: float,
                 step_forward: bool = False,
                 n: int = 0) -> tuple[float, int]:
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


def get(t: list, y: list, y_rise: float, y_half: float,
        y_fall: float) -> tuple[list, list]:
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
        edges_rise.append(get_crossing(t, y, i, y_half)[0])
    else:
      if y[i] < y_fall:
        state_low = True
        # interpolate y_half crossing
        edges_fall.append(get_crossing(t, y, i, y_half)[0])

  return (edges_rise, edges_fall)


def get_np(t: np.ndarray, y: np.ndarray, y_rise: float, y_half: float,
           y_fall: float) -> tuple[np.ndarray, np.ndarray]:
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
        edges_rise[i_rise] = get_crossing(t, y, i, y_half, False, n)[0]
        i_rise += 1
    else:
      if y[i] < y_fall:
        state_low = True
        # interpolate y_half crossing
        edges_fall[i_fall] = get_crossing(t, y, i, y_half, False, n)[0]
        i_fall += 1

  return (edges_rise[:i_rise], edges_fall[:i_fall])
