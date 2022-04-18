"""Eye Diagram helper functions, see measurement.eyediagram.eyediagram
"""

from typing import List

import numpy as np

cimport numpy as np
cimport cython

from hardware_tools.math cimport _lines

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dict sample_mask_c(
    np.ndarray[np.float64_t, ndim=1] waveform_y,
    list centers_t,
    list centers_i,
    np.float64_t t_delta,
    np.float64_t t_sym,
    np.float64_t y_zero,
    np.float64_t y_ua,
    list mask_paths,
    list mask_margins):
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
  cdef dict values = {"offenders": [], "hits": [], "margin": 1.0}

  cdef Py_ssize_t i_width = int((t_sym / t_delta) + 0.5) + 2
  cdef np.float64_t t_width_ui = (i_width * t_delta / t_sym)

  cdef Py_ssize_t n = i_width * 2 + 1
  cdef np.ndarray[np.float64_t, ndim=1] t0 = np.linspace(0.5 - t_width_ui,
      0.5 + t_width_ui, n)
  cdef np.ndarray[np.float64_t, ndim=1] t = np.zeros(n)
  cdef np.ndarray[np.float64_t, ndim=1] y = np.zeros(n)
  cdef np.ndarray[np.float64_t, ndim=2] hits

  waveform_y = (waveform_y - y_zero) / y_ua

  cdef Py_ssize_t n_masks = len(mask_margins)
  cdef Py_ssize_t i_zero = n_masks // 2

  cdef Py_ssize_t i_margin = 0

  cdef Py_ssize_t i, ii, c_i
  cdef np.float64_t c_t

  cdef list mask_zero = mask_paths[i_zero]
  cdef list mask_margin = mask_paths[i_margin]

  for i in range(len(centers_t)):
    c_i = centers_i[i] - i_width
    c_t = centers_t[i] / t_sym

    for ii in range(n):
      y[ii] = waveform_y[c_i + ii]
      t[ii] = t0[ii] + c_t

    # c_i = centers_i[i]
    # c_t = centers_t[i] / t_sym

    # t = t0 + c_t
    # y = waveform_y[c_i - i_width:c_i + i_width + 1]

    while i_margin < n_masks and _lines.is_hitting_np_c(t, y, mask_margin):
      i_margin += 1
      if i_margin < n_masks:
        mask_margin = mask_paths[i_margin]

    if i_margin < i_zero:
      continue  # There won't be hits until the margin is negative

    hits = _lines.hits_np_c(t, y, mask_zero)
    if hits.size > 0:
      values["offenders"].append(i)
      values["hits"].extend(hits.tolist())

  values["margin"] = mask_margins[min(n_masks - 1, i_margin)]
  return values

def sample_mask(waveform_y: np.ndarray, centers_t: List[float],
                centers_i: List[int], t_delta: float, t_sym: float,
                y_zero: float, y_ua: float, mask_paths: list,
                mask_margins: list) -> dict:
  return sample_mask_c(waveform_y, centers_t, centers_i, t_delta, t_sym, y_zero,
      y_ua, mask_paths, mask_margins)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list y_slice_c(
    np.ndarray[np.float64_t, ndim=1] waveform_y,
    list centers_t,
    list centers_i,
    np.float64_t t_delta,
    np.float64_t t_sym,
    np.float64_t y_zero,
    np.float64_t y_ua,
    list y_slices):
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
  cdef Py_ssize_t i_width = int((t_sym / t_delta) + 0.5) + 2
  cdef np.float64_t t_width_ui = (i_width * t_delta / t_sym)

  cdef Py_ssize_t n = i_width * 2 + 1
  cdef np.ndarray[np.float64_t, ndim=1] t0 = np.linspace(0.5 - t_width_ui,
      0.5 + t_width_ui, n)
  cdef np.ndarray[np.float64_t, ndim=1] t = np.zeros(n)
  cdef np.ndarray[np.float64_t, ndim=1] y = np.zeros(n)
  cdef np.ndarray[np.float64_t, ndim=2] hits

  waveform_y = (waveform_y - y_zero) / y_ua

  cdef list slices = []
  cdef list paths = []
  cdef Py_ssize_t n_slices = len(y_slices)
  cdef np.float64_t y_level
  for y_level in y_slices:
    slices.append([])
    paths.append([(0.0, y_level), (1.0, y_level)])

  cdef Py_ssize_t i, ii, c_i
  cdef np.float64_t c_t

  for i in range(len(centers_t)):
    c_i = centers_i[i] - i_width
    c_t = centers_t[i] / t_sym

    for ii in range(n):
      y[ii] = waveform_y[c_i + ii]
      t[ii] = t0[ii] + c_t

    # c_i = centers_i[i]
    # c_t = centers_t[i] / t_sym

    # t = t0 + c_t
    # y = waveform_y[c_i - i_width:c_i + i_width + 1]

    for ii in range(n_slices):
      hits = _lines.hits_np_c(t, y, [paths[ii]])
      if hits.size > 0:
        slices[ii].extend(hits[:, 0].tolist())
  
  return slices

def y_slice(waveform_y: np.ndarray, centers_t: List[float],
            centers_i: List[int], t_delta: float, t_sym: float, y_zero: float,
            y_ua: float, y_slices: List[float]) -> List[List[float]]:
  return y_slice_c(waveform_y, centers_t, centers_i, t_delta, t_sym, y_zero,
      y_ua, y_slices)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void stack_c(
    np.ndarray[np.float64_t, ndim=1] waveform_y,
    list centers_t,
    list centers_i,
    np.float64_t t_delta,
    np.float64_t t_sym,
    np.float64_t min_y,
    np.float64_t max_y,
    int resolution,
    np.ndarray[np.int32_t, ndim=2] grid,
    bint point_cloud):
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

  Returns:
    2D grid of heat map counts
  """
  cdef Py_ssize_t i_width = int((t_sym / t_delta) + 0.5) + 2
  cdef np.float64_t t_width_ui = (i_width * t_delta / t_sym)

  cdef Py_ssize_t n = i_width * 2 + 1
  cdef np.ndarray[np.float64_t, ndim=1] t0 = np.linspace(0.5 - t_width_ui,
      0.5 + t_width_ui, n)
  t0 = ((t0 + 0.5) / 2) * resolution

  cdef np.ndarray[np.int32_t, ndim=1] td = np.zeros(n, dtype=np.int32)
  cdef np.ndarray[np.int32_t, ndim=1] yd = np.zeros(n, dtype=np.int32)

  waveform_y = (waveform_y - min_y) / (max_y - min_y)
  cdef np.ndarray[np.int32_t, ndim=1] waveform_yd = (waveform_y * resolution).astype(np.int32)

  cdef Py_ssize_t i, ii, c_i
  cdef np.float64_t c_t

  for i in range(len(centers_t)):
    c_i = centers_i[i] - i_width
    c_t = (centers_t[i] / t_sym / 2) * resolution

    for ii in range(n):
      yd[ii] = waveform_yd[c_i + ii]
      td[ii] = int(t0[ii] + c_t)

    if point_cloud:
      _lines.draw_points_c(td, yd, grid)
    else:
      _lines.draw_c(td, yd, grid)

def stack(waveform_y: np.ndarray, centers_t: List[float], centers_i: List[int],
          t_delta: float, t_sym: float, min_y: float, max_y: float,
          resolution: int, grid: np.ndarray, point_cloud: bool) -> None:
  stack_c(waveform_y, centers_t, centers_i, t_delta, t_sym, min_y, max_y,
      resolution, grid, point_cloud)