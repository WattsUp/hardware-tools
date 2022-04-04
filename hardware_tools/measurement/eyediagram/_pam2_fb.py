"""PAM2 helper functions, see measurement.eyediagram.pam2
Fallback version for no cython
"""

from typing import List

import numpy as np


def sample_vertical(waveform_y: np.ndarray, centers_t: List[float],
                    centers_i: List[int], t_delta: float, t_sym: float,
                    y_half: float, level_width: float,
                    cross_width: float) -> dict:
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
      transitions: Dictionary of collected transitions, see MeasuresPAM2
      edge_dir: List of edge directions, True=rising, False=falling, None=none
  """
  i_width = int((t_sym / t_delta) + 0.5) + 2
  t_width_ui = (i_width * t_delta / t_sym)

  n = i_width * 2 + 1
  t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, n)

  abc_width = max(level_width, 1.01 * t_delta / t_sym)
  t_a_min = -0.5 - abc_width / 2
  t_a_max = -0.5 + abc_width / 2
  t_b_min = 0.5 - abc_width / 2
  t_b_max = 0.5 + abc_width / 2
  t_c_min = 1.5 - abc_width / 2
  t_c_max = 1.5 + abc_width / 2

  t_sym_min = 0.5 - level_width / 2
  t_sym_max = 0.5 + level_width / 2

  t_cross_min = 0.0 - cross_width / 2
  t_cross_max = 0.0 + cross_width / 2

  t_sym_cross_min = 0.5 - level_width / 2
  t_sym_cross_max = 0.5 + level_width / 2

  values = {
      "y_0": [],
      "y_1": [],
      "y_cross": [],
      "y_0_cross": [],
      "y_1_cross": [],
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

  for i in range(len(centers_t)):
    c_i = centers_i[i]
    c_t = centers_t[i] / t_sym

    samples_a = []
    samples_b = []
    samples_c = []

    samples_sym = []
    samples_cross = []
    samples_sym_cross = []
    for ii in range(n):
      t = t0[ii] + c_t
      y = waveform_y[c_i - i_width + ii]

      if t_a_min <= t <= t_a_max:
        samples_a.append(y)
      elif t_b_min <= t <= t_b_max:
        samples_b.append(y)
      elif t_c_min <= t <= t_c_max:
        samples_c.append(y)

      if t_sym_min <= t <= t_sym_max:
        samples_sym.append(y)

      if t_cross_min <= t <= t_cross_max:
        samples_cross.append(y)

      if t_sym_cross_min <= t <= t_sym_cross_max:
        samples_sym_cross.append(y)

    sym_a = np.mean(samples_a) > y_half
    sym_b = np.mean(samples_b) > y_half
    sym_c = np.mean(samples_c) > y_half

    if sym_a != sym_b:
      values["y_cross"].extend(samples_cross)
      values["edge_dir"].append(sym_b)
    else:
      values["edge_dir"].append(None)
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


def sample_horizontal(waveform_y: np.ndarray, centers_t: List[float],
                      centers_i: List[int], edge_dir: List[bool],
                      t_delta: float, t_sym: float, y_zero: float, y_ua: float,
                      y_cross: float, hist_height: float, edge_lower: float,
                      edge_upper: float) -> dict:
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
  i_width = int((t_sym / t_delta) + 0.5) + 2
  t_width_ui = (i_width * t_delta / t_sym)

  n = i_width * 2 + 1
  t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, n)
  center_offset = int(n * 1 / 4)

  waveform_y = (waveform_y - y_zero) / y_ua
  y_cross = (y_cross - y_zero) / y_ua

  y_lower_min = edge_lower - hist_height / 2
  y_lower_max = edge_lower + hist_height / 2
  y_upper_min = edge_upper - hist_height / 2
  y_upper_max = edge_upper + hist_height / 2

  y_cross_min = y_cross - hist_height / 2
  y_cross_max = y_cross + hist_height / 2

  y_half_min = 0.5 - hist_height / 2
  y_half_max = 0.5 + hist_height / 2

  values = {
      "t_rise_lower": [],
      "t_rise_upper": [],
      "t_rise_half": [],
      "t_fall_lower": [],
      "t_fall_upper": [],
      "t_fall_half": [],
      "t_cross_left": [],
      "t_cross_right": []
  }

  for i in range(len(centers_t)):
    c_i = centers_i[i]
    c_t = centers_t[i] / t_sym

    for ii in range(n):
      t = t0[ii] + c_t
      y = waveform_y[c_i - i_width + ii]

      if y_cross_min <= y <= y_cross_max:
        if -0.5 <= t <= 0.5:
          values["t_cross_left"].append(t)
        elif 0.5 < t <= 1.5:
          values["t_cross_right"].append(t)
        else:
          pass  # pragma: no cover, just a glitch

    if edge_dir[i] is None:
      continue
    y_front = waveform_y[c_i - i_width:c_i + 1]
    y_center = waveform_y[c_i - i_width + center_offset:c_i + center_offset + 1]

    if edge_dir[i]:
      # Rising edge starts at minimum on [-0.5, 0.5]
      ii = np.argmin(y_front)
      # Stop at maximum on [0.0, 1.0]
      ii_stop = center_offset + np.argmax(y_center)
    else:
      # Falling edge starts at maximum on [-0.5, 0.5]
      ii = np.argmax(y_front)
      # Stop at minimum on [0.0, 1.0]
      ii_stop = center_offset + np.argmin(y_center)
    while ii <= ii_stop:
      t = t0[ii] + c_t
      y = waveform_y[c_i - i_width + ii]

      if y_lower_min <= y <= y_lower_max:
        if edge_dir[i]:
          values["t_rise_lower"].append(t)
        else:
          values["t_fall_lower"].append(t)

      if y_upper_min <= y <= y_upper_max:
        if edge_dir[i]:
          values["t_rise_upper"].append(t)
        else:
          values["t_fall_upper"].append(t)

      if y_half_min <= y <= y_half_max:
        if edge_dir[i]:
          values["t_rise_half"].append(t)
        else:
          values["t_fall_half"].append(t)

      ii += 1

  return values
