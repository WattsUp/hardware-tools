"""Clock Data Recovery worker
"""

import numpy as np
from scipy import stats

from hardware_tools import math


class CDR:
  """Clock Data Recovery worker

  Base generates a constant clock with minimum mean squared error
  """

  def __init__(self, t_sym: float = None) -> None:
    """Initialize CDR

    Args:
      t_sym: Initial PLL period for a single symbol, None will best guess
    """
    self._t_sym_initial = t_sym
    self._avg_sym_min = 0.9
    self._avg_sym_max = 5

  def run(self, data_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run CDR to generate clock edges from data edges

    The number of clock edges may be less than expected due to delay in locking
    on to the recovered clock. The number of TIEs will match the number of data
    edges.

    Args:
      data_edges: List of data edges in time domain

    Returns:
      List of clock edges in time domain.
      List of Time Interval Errors (TIEs).
    """
    t_sym = self._t_sym_initial
    n = len(data_edges)
    data_periods = np.diff(data_edges)
    # Step 0: Find most probable duration of 1 symbol
    # Shortest time interval ignoring any glitches
    if t_sym is None:
      fit = math.GaussianMix.fit_samples(data_periods)
      scores = []
      for c in fit.components:
        accuracy = 0
        avg_sym = 0
        for c2 in fit.components:
          integerness = abs((c2.mean / c.mean + 0.5) % 1.0 - 0.5)
          accuracy += c2.amplitude * (1 - integerness)**2
          avg_sym += c2.amplitude * (c2.mean // c.mean)
        if (avg_sym < self._avg_sym_min) or (avg_sym > self._avg_sym_max):
          continue
        scores.append((c.mean, accuracy))
      scores = sorted(scores, key=lambda s: -s[1])
      if len(scores) == 0:
        t_sym = fit.components[0].mean
      else:
        t_sym = scores[0][0]

    # Step 1: Find the average period by estimating the number of symbols
    # between edges
    num_sym = np.floor(data_periods / t_sym + 0.5)
    # Check it isn't wildly off
    avg_sym = num_sym.mean()
    if (avg_sym < self._avg_sym_min) or (avg_sym > self._avg_sym_max):
      raise ArithmeticError(
          "Average number of symbol between edges outside of allowable bounds"
          f" {self._avg_sym_min} < {avg_sym} < {self._avg_sym_max}")
    t_sym = (data_periods / np.maximum(1, num_sym)).mean()
    t_start = data_edges[0]

    # Step 2: Measure time interval error and minimize
    bits, ties = np.divmod(data_edges - t_start + t_sym / 2, t_sym)
    ties = ties - t_sym / 2
    # Fix disjoints, assume jumps over 1/2 are going the other way
    ties_diff = np.diff(ties)
    disjoints = np.where(np.abs(ties_diff) > t_sym / 2)[0]
    for disjoint in disjoints:
      offset = np.append([0.0] * (disjoint + 1), [t_sym] * (n - disjoint - 1))
      if ties_diff[disjoint] > 0:
        offset = -offset
      ties += offset
    trend = stats.linregress(bits, ties)
    t_sym = t_sym + trend.slope

    # Step 3: Adjust t_start for zero mean ties
    ties = (data_edges - t_start + t_sym / 2) % t_sym - t_sym / 2
    t_start = t_start + ties.mean()
    ties = ties - ties.mean()

    # Step 3: Generate clean clock
    n = int(np.ceil((data_edges[-1] - data_edges[0]) / t_sym))
    clk_edges = np.linspace(0, t_sym * (n - 1), n) + t_start

    return clk_edges, ties
