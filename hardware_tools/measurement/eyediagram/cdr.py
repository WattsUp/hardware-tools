"""Clock Data Recovery worker
"""

from typing import Tuple

import numpy as np

try:
  from hardware_tools.measurement.eyediagram import _cdr
except ImportError:
  print(f"The cython version of {__name__} is not available")
  from hardware_tools.measurement.eyediagram import _cdr_fb as _cdr


class CDR:
  """Clock Data Recovery worker

  Base generates a constant clock with minimum mean squared error
  """

  def __init__(self, t_sym: float, fixed_period: bool = False) -> None:
    """Initialize CDR

    Args:
      t_sym: Initial PLL period for a single symbol
      fixed_period: True will only let CDR adjust phase, False allows period
        and phase adjustment
    """
    self._t_sym_initial = t_sym
    self._t_sym_initial_error = 0.01
    self._avg_sym_min = 0.9
    self._avg_sym_max = 5
    self._max_correctable_disjoints = 100
    self._fixed_period = fixed_period

  def run(self, data_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run CDR to generate clock edges from data edges

    The number of clock edges may be less than expected due to delay in locking
    on to the recovered clock. The number of TIEs will match the number of data
    edges.

    Args:
      data_edges: List of data edges in time domain

    Returns:
      List of clock edges in the time domain.
      List of Time Interval Errors (TIEs).

    Raises:
      ValueError if there are fewer than 2 edges provided
      ArithmeticError if CDR fails to recover clock
    """
    t_sym = self._t_sym_initial
    t_start = data_edges[0]
    n = len(data_edges)
    if n < 2:
      raise ValueError("Need at least 2 edges to compute clock period")
    if t_sym is None:
      raise ValueError("Need at t_sym_initial within 10% of real value")
    data_periods = np.diff(data_edges)

    # Step 1: Validate the rough t_sym by estimating the number of symbols
    # between edges
    num_sym = np.floor(data_periods / t_sym + 0.5)
    avg_sym = num_sym.mean()
    if (avg_sym < self._avg_sym_min) or (avg_sym > self._avg_sym_max):
      raise ArithmeticError(
          "Average number of symbol between edges outside of allowable bounds"
          f" {self._avg_sym_min} < {avg_sym} < {self._avg_sym_max}")

    if not self._fixed_period:
      # Step 2: Minimize number of TIE disjoints
      t_sym = _cdr.minimize_tie_disjoints(
          data_edges,
          t_min=t_sym * (1 - self._t_sym_initial_error),
          t_max=t_sym * (1 + self._t_sym_initial_error),
          tol=self._max_correctable_disjoints)

      # Step 3: Remove linear drift from TIEs
      t_sym = _cdr.detrend_ties(data_edges, t_sym)

    # Step 4: Adjust t_start for zero mean ties
    ties = (data_edges - t_start + t_sym / 2) % t_sym - t_sym / 2
    t_start = t_start + ties.mean()
    ties = ties - ties.mean()

    ties_diff = np.diff(ties)
    disjoints = np.where(np.abs(ties_diff) > t_sym / 2)[0]
    if len(disjoints) > 0:
      raise ArithmeticError("Recovered clock has disjoints")

    # Step 5: Generate clean clock
    n = int(np.ceil((data_edges[-1] - data_edges[0]) / t_sym))
    clk_edges = np.linspace(0, t_sym * (n - 1), n) + t_start + t_sym / 2

    return clk_edges, ties
