"""PAM2 helper functions, see measurement.eyediagram.pam2
"""

from typing import List

import numpy as np

cimport numpy as np
cimport cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dict sample_vertical_c(
    np.ndarray[np.float64_t, ndim=1] waveform_y,
    np.ndarray[np.float64_t, ndim=1] centers_t,
    np.ndarray[np.int32_t, ndim=1] centers_i,
    np.float64_t t_delta,
    np.float64_t t_sym,
    np.float64_t y_half,
    np.float64_t level_width,
    np.float64_t cross_width):
  """Measure vertical parameters

  Args:
    waveform_y: Waveform data array [y0, y1,..., yn]
    centers_t: List of symbol centers in time for sub t_delta alignment.
      Grid spans [-0.5*t_sym, 1.5*t_sym] + center_t
    centers_i: List of symbol centers indices
    t_delta: Time between samples
    t_sym: Duration of one symbol
    y_half: Decision threshold for a low or high symbol
    level_width: Width of y_0, y_1 windows, UI
    cross_width: Width of y_cross window, UI

  Returns:
    Dictionary of values:
      y_0: List of samples within the y_0 window, logical 0
      y_1: List of samples within the y_1 window, logical 1
      y_cross: List of samples within the y_cross window, edge
      y_0_cross: List of samples within the y_cross window, logical 0
      y_1_cross: List of samples within the y_cross window, logical 1
      y_avg: List of samples within [0, 1]UI
      transitions: Dictionary of collected transitions, see MeasuresPAM2
      edge_dir: List of edge directions, 1=rising, -1=falling, 0=none
  """
  cdef Py_ssize_t i_width = int((t_sym / t_delta) + 0.5) + 2
  cdef np.float64_t t_width_ui = (i_width * t_delta / t_sym)

  cdef Py_ssize_t n = i_width * 2 + 1
  cdef np.ndarray[np.float64_t, ndim=1] t0 = np.linspace(0.5 - t_width_ui,
      0.5 + t_width_ui, n)

  cdef np.float64_t abc_width = max(level_width, 1.01 * t_delta / t_sym)
  cdef np.float64_t t_a_min = -0.5 - abc_width / 2
  cdef np.float64_t t_a_max = -0.5 + abc_width / 2
  cdef np.float64_t t_b_min = 0.5 - abc_width / 2
  cdef np.float64_t t_b_max = 0.5 + abc_width / 2
  cdef np.float64_t t_c_min = 1.5 - abc_width / 2
  cdef np.float64_t t_c_max = 1.5 + abc_width / 2

  cdef np.float64_t t_sym_min = 0.5 - level_width / 2
  cdef np.float64_t t_sym_max = 0.5 + level_width / 2

  cdef np.float64_t t_cross_min = 0.0 - cross_width / 2
  cdef np.float64_t t_cross_max = 0.0 + cross_width / 2

  cdef np.float64_t t_sym_cross_min = 0.5 - level_width / 2
  cdef np.float64_t t_sym_cross_max = 0.5 + level_width / 2

  cdef dict values = {
      "y_0": [],
      "y_1": [],
      "y_cross": [],
      "y_0_cross": [],
      "y_1_cross": [],
      "y_avg": [],
      "transitions": {
          "000": 0,
          "001": 0,
          "010": 0,
          "011": 0,
          "100": 0,
          "101": 0,
          "110": 0,
          "111": 0,
      },
      "edge_dir": []
  }

  cdef Py_ssize_t i, ii, c_i
  cdef np.float64_t c_t, t, y, sym_a, sym_b, sym_c, n_a, n_b, n_c
  cdef str seq

  cdef list samples_sym = []
  cdef list samples_cross = []
  cdef list samples_sym_cross = []
  for i in range(centers_t.shape[0]):
    c_i = centers_i[i] - i_width
    c_t = centers_t[i] / t_sym

    sym_a = 0
    sym_b = 0
    sym_c = 0
    n_a = 0
    n_b = 0
    n_c = 0

    samples_sym = []
    samples_cross = []
    samples_sym_cross = []
    for ii in range(n):
      t = t0[ii] + c_t
      y = waveform_y[c_i + ii]

      if t_a_min <= t <= t_a_max:
        sym_a += y
        n_a += 1
      elif t_b_min <= t <= t_b_max:
        sym_b += y
        n_b += 1
      elif t_c_min <= t <= t_c_max:
        sym_c += y
        n_c += 1

      if t_sym_min <= t <= t_sym_max:
        samples_sym.append(y)

      if t_cross_min <= t <= t_cross_max:
        samples_cross.append(y)

      if t_sym_cross_min <= t <= t_sym_cross_max:
        samples_sym_cross.append(y)

      if 0 <= t <= 1:
        values["y_avg"].append(y)

    sym_a = sym_a > (y_half * n_a)
    sym_b = sym_b > (y_half * n_b)
    sym_c = sym_c > (y_half * n_c)

    if sym_a != sym_b:
      values["y_cross"].extend(samples_cross)
      values["edge_dir"].append(1 if sym_b else -1)
    else:
      values["edge_dir"].append(0)
      if sym_b:
        values["y_1_cross"].extend(samples_sym_cross)
      else:
        values["y_0_cross"].extend(samples_sym_cross)

    seq = "1" if sym_a else "0"

    if sym_b:
      values["y_1"].extend(samples_sym)
      seq += "1"
    else:
      values["y_0"].extend(samples_sym)
      seq += "0"

    seq += "1" if sym_c else "0"

    values["transitions"][seq] += 1

  return values

def sample_vertical(waveform_y: np.ndarray, centers_t: np.ndarray,
                    centers_i: np.ndarray, t_delta: float, t_sym: float,
                    y_half: float, level_width: float,
                    cross_width: float) -> dict:
  return sample_vertical_c(waveform_y, centers_t, centers_i, t_delta, t_sym,
      y_half, level_width, cross_width)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dict sample_horizontal_c(
    np.ndarray[np.float64_t, ndim=1] waveform_y,
    np.ndarray[np.float64_t, ndim=1] centers_t,
    np.ndarray[np.int32_t, ndim=1] centers_i,
    np.ndarray[np.int8_t, ndim=1] edge_dir,
    np.float64_t t_delta,
    np.float64_t t_sym,
    np.float64_t y_zero,
    np.float64_t y_ua,
    np.float64_t y_cross,
    np.float64_t hist_height,
    np.float64_t edge_lower,
    np.float64_t edge_upper):
  """Measure horizontal parameters

  Args:
    waveform_y: Waveform data array [y0, y1,..., yn]
    centers_t: List of symbol centers in time for sub t_delta alignment.
      Grid spans [-0.5*t_sym, 1.5*t_sym] + center_t
    centers_i: List of symbol centers indices
    edge_dir: List of edge directions, True=rising, False=falling, None=none
    t_delta: Time between samples
    t_sym: Duration of one symbol
    y_zero: Amplitude of a logical 0
    y_ua: Normalized amplitude
    y_cross: Amplitude of crossing point
    hist_height: Height of time windows, UA
    edge_lower: Threshold level for lower transition (rise start), UA
    edge_upper: Threshold level for upper transition (rise stop), UA

  Returns:
    Dictionary of values, all UI:
      t_rise_lower: List of samples within the lower edge window, rising
      t_rise_upper: List of samples within the upper edge window, rising
      t_rise_half: List of samples within the 50% window, rising
      t_fall_lower: List of samples within the lower edge window, falling
      t_fall_upper: List of samples within the upper edge window, falling
      t_fall_half: List of samples within the 50% window, falling
      t_cross_left: List of samples within the cross window, left edge
      t_cross_right: List of samples within the cross window, right edge
  """
  cdef Py_ssize_t i_width = int((t_sym / t_delta) + 0.5) + 2
  cdef np.float64_t t_width_ui = (i_width * t_delta / t_sym)

  cdef Py_ssize_t n = i_width * 2 + 1
  cdef np.ndarray[np.float64_t, ndim=1] t0 = np.linspace(0.5 - t_width_ui,
      0.5 + t_width_ui, n)

  waveform_y = (waveform_y - y_zero) / y_ua
  y_cross = (y_cross - y_zero) / y_ua

  cdef np.float64_t y_lower_min = edge_lower - hist_height / 2
  cdef np.float64_t y_lower_max = edge_lower + hist_height / 2
  cdef np.float64_t y_upper_min = edge_upper - hist_height / 2
  cdef np.float64_t y_upper_max = edge_upper + hist_height / 2

  cdef np.float64_t y_cross_min = y_cross - hist_height / 2
  cdef np.float64_t y_cross_max = y_cross + hist_height / 2

  cdef np.float64_t y_half_min = 0.5 - hist_height / 2
  cdef np.float64_t y_half_max = 0.5 + hist_height / 2

  cdef dict values = {
      "t_rise_lower": [],
      "t_rise_upper": [],
      "t_rise_half": [],
      "t_fall_lower": [],
      "t_fall_upper": [],
      "t_fall_half": [],
      "t_cross_left": [],
      "t_cross_right": []
  }

  cdef Py_ssize_t i, ii, c_i
  cdef np.float64_t c_t, t, y
  cdef np.int8_t e

  for i in range(centers_t.shape[0]):
    c_i = centers_i[i] - i_width
    c_t = centers_t[i] / t_sym
    e = edge_dir[i]

    for ii in range(n):
      t = t0[ii] + c_t
      y = waveform_y[c_i + ii]

      if y_cross_min <= y <= y_cross_max:
        if -0.5 <= t <= 0.5:
          values["t_cross_left"].append(t)
        elif 0.5 < t <= 1.5:
          values["t_cross_right"].append(t)
        else:
          pass  # pragma: no cover, just a glitch
      
      if e == 0 or t > 0.5:
        continue

      if y_lower_min <= y <= y_lower_max:
        if e == 1:
          values["t_rise_lower"].append(t)
        else:
          values["t_fall_lower"].append(t)

      if y_upper_min <= y <= y_upper_max:
        if e == 1:
          values["t_rise_upper"].append(t)
        else:
          values["t_fall_upper"].append(t)

      if y_half_min <= y <= y_half_max:
        if e == 1:
          values["t_rise_half"].append(t)
        else:
          values["t_fall_half"].append(t)

  return values

def sample_horizontal(waveform_y: np.ndarray, centers_t: np.ndarray,
                      centers_i: np.ndarray, edge_dir: np.ndarray,
                      t_delta: float, t_sym: float, y_zero: float, y_ua: float,
                      y_cross: float, hist_height: float, edge_lower: float,
                      edge_upper: float) -> dict:
  return sample_horizontal_c(waveform_y, centers_t, centers_i, edge_dir,
      t_delta, t_sym, y_zero, y_ua, y_cross, hist_height, edge_lower, edge_upper)
