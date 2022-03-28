"""Interpolation of waveform between sampled values
"""

import numpy as np


def sinc(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
  """Resample a time series using sinc interpolation

  Adds a sinc function at each point then resamples at x. x is expected to
  be regularly sampled (constant sinc width). x does not need to be
  regularly sampled.

  Args:
    x: The x-coordinates at which to evaluate the interpolated values
    xp: The x-coordinates of the data points, regularly sampled
    yp: The y-coordinates of the data points, same length as xp (>= 2)

  Returns:
    The interpolated values, same shape as x

  Raises:
    ValueError: If xp and yp are not the same length, have fewer than 2 points,
    are not 1D. Checking for regularly sampled xp is expensive and not checked.
  """
  if len(xp.shape) != 1:
    raise ValueError("Input must be 1D")
  if xp.shape != yp.shape:
    raise ValueError("Input must be same shape")
  if xp.shape[0] < 2:
    raise ValueError("Input must have at least 2 points")
  period = xp[1] - xp[0]
  x_tile = np.tile(x, (len(xp), 1)) - np.tile(xp[:, np.newaxis], (1, len(x)))
  y = np.dot(yp, np.sinc(x_tile / period))
  return y
