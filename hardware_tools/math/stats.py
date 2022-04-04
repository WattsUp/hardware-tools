"""Statistical functions

Binning, Uncertain Value, etc.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def bin_linear(y: np.ndarray,
               bin_count: int = 100,
               density: bool = None) -> Tuple[np.ndarray, np.ndarray]:
  """Bin values with equal width bins

  Args:
    y: Sample values
    bin_count: Number of equal width bins
    density: Passed to np.histogram, roughly True to get a PDF

  Returns:
    counts, edges
  """
  y_min = min(y)
  y_max = max(y)
  if y_min == y_max:
    if y_min == 0:
      y_min = -1
      y_max = 1
    else:
      center = y_min
      y_min = center - abs(center) * 0.05
      y_max = center + abs(center) * 0.05
  edges = np.linspace(y_min, y_max, bin_count + 1)
  return np.histogram(y, edges, density=density)


def bin_exact(y: Iterable) -> Tuple[list, list]:
  """Bin values with exact indices

  Args:
    y: Sample values

  Returns:
    counts, bins
  """
  counts = {}
  for e in y:
    if e in counts:
      counts[e] += 1
    else:
      counts[e] = 1
  bins = sorted(counts.keys())
  counts = [counts[b] for b in bins]
  return counts, bins


def bin_exact_np(y: Iterable) -> Tuple[np.ndarray, np.ndarray]:
  """Bin values with exact indices

  Args:
    y: Sample values

  Returns:
    counts, bins as np arrays
  """
  counts = {}
  for e in y:
    if e in counts:
      counts[e] += 1
    else:
      counts[e] = 1
  bins = sorted(counts.keys())
  counts = [counts[b] for b in bins]
  return np.array(counts), np.array(bins)


def bin_exponential(y: Iterable,
                    bin_count: int = 100,
                    density: bool = None) -> Tuple[np.array, np.array]:
  """Bin values with equal exponential width bins

  Args:
    y: Sample values
    bin_count: Number of equal exponential width bins
    density: Passed to np.histogram, roughly True to get a PDF

  Returns:
    counts, edges
  """
  with np.errstate(divide="ignore"):
    y_exp = np.log10(y)
  y_min = min(y_exp)
  y_max = max(y_exp)
  if np.isneginf(y_min):
    if np.isneginf(y_max):
      y_max = 0
    else:
      y_max = int(np.ceil(y_max))
    edges = [np.NINF]
    bins = np.arange(y_max - (bin_count - 1), y_max + 1, 1, dtype=np.float64)
    edges.extend(bins)
    edges = np.array(edges)
    counts, _ = np.histogram(y_exp, edges, density=False)
    if density:
      counts = counts / 1 / counts.sum()
    return counts, edges
  else:
    edges = np.linspace(y_min, y_max, bin_count + 1)
    return np.histogram(y_exp, edges, density=density)


def downsample(y: np.ndarray, n_max: int = 50e3, bin_count=500) -> np.ndarray:
  """Reduce the number of samples to at most n_max, preserving sample
  frequency

  Bins the values, scales the counts to total of n_max, then undoes the
  binning. Does use floor for sample count, be wary of loss of low frequency
  samples.

  Args:
    y: Sample values
    n_max: Maximum number of samples in returned dataset
    bin_count: Number of equal width bins, None for exact binning

  Returns:
    Downsampled dataset samples
  """
  scale = n_max / len(y)
  if scale >= 1:
    return y
  if bin_count is not None:
    counts, edges = bin_linear(y, bin_count=bin_count, density=False)
    bins = (edges[:-1] + edges[1:]) / 2
  else:
    counts, bins = bin_exact(y)
  y_down = []
  for i in range(len(counts)):
    count_down = int(np.floor(counts[i] * scale))
    y_down.extend([bins[i]] * count_down)
  return np.array(y_down)


class UncertainValue:
  """Value with uncertainty

  Properities:
    value: Value of statistic
    stddev: Standard deviation of statistic
  """

  def __init__(self, value: float, stddev: float) -> None:
    self.value = value
    self.stddev = stddev

  @staticmethod
  def samples(x: np.ndarray) -> UncertainValue:
    """Create an UncertainValue from samples

    Args:
      x: List of samples to compute mean and population standard deviation

    Returns:
      UncertainValue with the mean and standard deviation of the samples
    """
    if x.size < 1:
      return UncertainValue(np.nan, np.nan)
    m = x.mean()
    c = x - m
    return UncertainValue(m, np.sqrt(np.dot(c, c) / x.size))

  def __format__(self, format_spec: str) -> str:
    if len(format_spec) == 0:
      return str(self)
    understood = [str(i) for i in range(10)]
    understood.extend(["e", "E", "g", "G", "f", "F", "n", "%"])
    if format_spec[-1] not in understood:
      raise TypeError(
          f"Unsupported format spec passed to UncertainValue '{format_spec}'")
    return f"(µ={self.value:{format_spec}},σ={self.stddev:{format_spec}})"

  def __repr__(self) -> str:
    return f"(µ={self.value},σ={self.stddev})"

  def __add__(self, b) -> UncertainValue:
    if isinstance(b, UncertainValue):
      v = self.value + b.value
      s = np.sqrt(self.stddev**2 + b.stddev**2)
    else:
      v = self.value + b
      s = self.stddev
    return UncertainValue(v, s)

  def __sub__(self, b) -> UncertainValue:
    if isinstance(b, UncertainValue):
      v = self.value - b.value
      s = np.sqrt(self.stddev**2 + b.stddev**2)
    else:
      v = self.value - b
      s = self.stddev
    return UncertainValue(v, s)

  def __mul__(self, b) -> UncertainValue:
    if isinstance(b, UncertainValue):
      v = self.value * b.value
      s = abs(v) * np.sqrt((self.stddev / self.value)**2 +
                           (b.stddev / b.value)**2)
    else:
      v = self.value * b
      s = self.stddev * abs(b)
    return UncertainValue(v, s)

  def __truediv__(self, b) -> UncertainValue:
    if isinstance(b, UncertainValue):
      v = self.value / b.value
      s = abs(v) * np.sqrt((self.stddev / self.value)**2 +
                           (b.stddev / b.value)**2)
    else:
      v = self.value / b
      s = self.stddev / abs(b)
    return UncertainValue(v, s)

  def __lt__(self, b) -> bool:
    if isinstance(b, UncertainValue):
      return self.value < b.value
    else:
      return self.value < b

  def __le__(self, b) -> bool:
    if isinstance(b, UncertainValue):
      return self.value <= b.value
    else:
      return self.value <= b

  def __eq__(self, b) -> bool:
    if isinstance(b, UncertainValue):
      return self.value == b.value
    else:
      return self.value == b

  def __ne__(self, b) -> bool:
    if isinstance(b, UncertainValue):
      return self.value != b.value
    else:
      return self.value != b

  def __ge__(self, b) -> bool:
    if isinstance(b, UncertainValue):
      return self.value >= b.value
    else:
      return self.value >= b

  def __gt__(self, b) -> bool:
    if isinstance(b, UncertainValue):
      return self.value > b.value
    else:
      return self.value > b

  def log(self) -> UncertainValue:
    if np.isnan(self.value) or self.value <= 0.0:
      return UncertainValue(np.nan, np.nan)
    v = np.log(self.value)
    s = np.abs(self.stddev / self.value)
    return UncertainValue(v, s)

  def log10(self) -> UncertainValue:
    if np.isnan(self.value) or self.value <= 0.0:
      return UncertainValue(np.nan, np.nan)
    v = np.log10(self.value)
    s = np.abs(self.stddev / (self.value * np.log(10)))
    return UncertainValue(v, s)

  def isnan(self) -> bool:
    return np.isnan(self.value) or np.isnan(self.stddev)
