"""PAM2 helper functions, see measurement.eyediagram.pam2
Fallback version for no cython
"""

from typing import List

import numpy as np

from hardware_tools.math import lines


def y_slice(waveform_y: np.ndarray, centers_t: List[float],
            centers_i: List[int], t_delta: float, t_sym: float, y_zero: float,
            y_ua: float, y_slices: List[float]) -> List[List[float]]:
  """Slice waveform at a level and record position of intersections

  Args:
    waveform_y: Waveform data array [y0, y1,..., yn]
    centers_t: List of symbol centers in time for sub t_delta alignment.
      Grid spans [-0.5*t_sym, 1.5*t_sym] + center_t
    centers_i: List of symbol centers indices
    t_delta: Time between samples
    t_sym: Duration of one symbol
    y_zero: Amplitude of a logical 0
    y_ua: Normalized amplitude
    y_slices: Amplitudes of slices for hits (normally threshold level)

  Returns:
    list per y_slice level containing
    list of intersections in time
  """
  i_width = int((t_sym / t_delta) + 0.5) + 2
  t_width_ui = (i_width * t_delta / t_sym)

  n = i_width * 2 + 1
  t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, n)

  waveform_y = (waveform_y - y_zero) / y_ua

  slices = []
  paths = []
  n_slices = len(y_slices)
  for y in y_slices:
    slices.append([])
    paths.append([(0.0, y), (1.0, y)])

  for i in range(len(centers_t)):
    c_i = centers_i[i]
    c_t = centers_t[i] / t_sym

    t = t0 + c_t
    y = waveform_y[c_i - i_width:c_i + i_width + 1]

    for ii in range(n_slices):
      hits = lines.hits_np(t, y, [paths[ii]])
      if hits.size > 0:
        slices[ii].extend(hits[:, 0].tolist())

  return slices


def stack(waveform_y: np.ndarray, centers_t: List[float], centers_i: List[int],
          t_delta: float, t_sym: float, min_y: float, max_y: float,
          resolution: int, grid: np.ndarray, point_cloud: bool) -> None:
  """Stack waveforms and counting overlaps in a heat map

  Args:
    waveform_y: Waveform data array [y0, y1,..., yn]
    centers_t: List of symbol centers in time for sub t_delta alignment.
      Grid spans [-0.5*t_sym, 1.5*t_sym] + center_t
    centers_i: List of symbol centers indices
    t_delta: Time between samples
    t_sym: Duration of one symbol
    min_y: Lower vertical value for bottom of grid
    max_y: Upper vertical value for top of grid.
      Grid units = (y - min_y) / (max_y - min_y)
    resolution: Resolution of square eye diagram image, 2UI x 2UA
    grid: Grid to stack onto
    point_cloud: True will not linearly interpolate, False will
  """
  i_width = int((t_sym / t_delta) + 0.5) + 2
  t_width_ui = (i_width * t_delta / t_sym)

  t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, i_width * 2 + 1)

  waveform_y = (waveform_y - min_y) / (max_y - min_y)

  for i in range(len(centers_t)):
    c_i = centers_i[i]
    c_t = centers_t[i] / t_sym

    t = (t0 + c_t)
    y = waveform_y[c_i - i_width:c_i + i_width + 1]

    td = (((t + 0.5) / 2) * resolution).astype(np.int32)
    yd = (y * resolution).astype(np.int32)
    if point_cloud:
      lines.draw_points(td, yd, grid)
    else:
      lines.draw(td, yd, grid)


def sample_mask(waveform_y: np.ndarray, centers_t: List[float],
                centers_i: List[int], t_delta: float, t_sym: float,
                y_zero: float, y_ua: float, mask_paths: list,
                mask_margins: list) -> dict:
  """Measure mask parameters

  Args:
    waveform_y: Waveform data array [y0, y1,..., yn]
    centers_t: List of symbol centers in time for sub t_delta alignment.
      Grid spans [-0.5*t_sym, 1.5*t_sym] + center_t
    centers_i: List of symbol centers indices
    t_delta: Time between samples
    t_sym: Duration of one symbol
    y_zero: Amplitude of a logical 0
    y_ua: Normalized amplitude
    mask_paths: list of mask.adjust(mask_margins).paths
    mask_margins: Margins used when computing mask_paths, starts at front.
      Assumes center index (n // 2) is margin=0

  Returns:
    Dictionary of values:
      offenders: List of symbol indices that hit the mask
      hits: List of mask collision coordinates [[UI, UA],...]
      margin: Largest mask margin with zero hits [-1.0, 1.0]
  """
  values = {"offenders": [], "hits": [], "margin": 1.0}
  i_width = int((t_sym / t_delta) + 0.5) + 2
  t_width_ui = (i_width * t_delta / t_sym)

  n = i_width * 2 + 1
  t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, n)

  waveform_y = (waveform_y - y_zero) / y_ua

  n_masks = len(mask_margins)
  i_zero = n_masks // 2

  i_margin = 0

  for i in range(len(centers_t)):
    c_i = centers_i[i]
    c_t = centers_t[i] / t_sym

    t = t0 + c_t
    y = waveform_y[c_i - i_width:c_i + i_width + 1]

    while i_margin < n_masks and lines.is_hitting_np(t, y,
                                                     mask_paths[i_margin]):
      i_margin += 1

    if i_margin < i_zero:
      continue  # There won't be hits until the margin is negative

    hits = lines.hits_np(t, y, mask_paths[i_zero])
    if hits.size > 0:
      values["offenders"].append(i)
      values["hits"].extend(hits.tolist())

  values["margin"] = mask_margins[i_margin]
  return values
