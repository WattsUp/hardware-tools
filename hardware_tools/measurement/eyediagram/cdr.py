"""Clock Data Recovery worker
"""

import traceback
from typing import Callable, Tuple

import numpy as np
import scipy.interpolate

try:
  from hardware_tools.measurement.eyediagram import _cdr
except ImportError:
  traceback.print_exc()
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


class CDRWithFFTFilter(CDR):
  """CDR with a FFT based filter for selectively removing jitter
  """

  def __init__(self, t_sym: float, ojtf: Callable[[np.ndarray],
                                                  np.ndarray]) -> None:
    """Initialize CDR

    Phase is removed from OJTF such that the filter's group delay is zero

    Args:
      t_sym: Initial PLL period for a single symbol
      ojtf: Observed jitter transfer function, see high_pass_butter
    """
    super().__init__(t_sym, fixed_period=False)
    self._ojtf = ojtf

  @staticmethod
  def high_pass_butter(freqs: np.ndarray, bandwidth: float,
                       order: int) -> np.ndarray:
    """High pass Butterworth OJTF, jitter below bandwidth will be absorbed by
    the CDR and thus not observed on the output TIEs

    Args:
      freqs: Frequencies the FFT was taken at, in Hz
      bandwidth: -3dB cutoff frequency, in Hz
      order: filter order

    Returns:
      Gain of OJTF at each frequency
    """
    g = 1 / np.sqrt(1 +
                    np.power(freqs / bandwidth, -2 * order, where=(freqs != 0)))
    g[0] = 0  # Fix DC
    return g

  def run(self, data_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out_constant, ties_constant = super().run(data_edges)
    t_sym_recovered = ((out_constant[-1] - out_constant[0]) /
                       (len(out_constant) - 1))

    # Trim to n=even for rfft
    n_fft = (len(out_constant) // 2) * 2
    out_constant = out_constant[:n_fft]

    # Resample TIEs at every out, to be fixed frequency
    # Ties are (edges, ties) want (out, ties_interpolated)
    # When missing an edge, use previous TIE
    func = scipy.interpolate.interp1d(data_edges,
                                      ties_constant,
                                      kind="previous",
                                      fill_value="extrapolate")
    ties_interpolated = func(out_constant)

    # Use FFT to change amplitude without adding an group delay
    fft_ties = np.fft.rfft(ties_interpolated, norm="forward")
    fft_freq = np.fft.rfftfreq(n_fft, d=t_sym_recovered)

    # Apply (1 - filter_gain) and irfft to get the edge adjustment
    g = self._ojtf(fft_freq)
    edge_adj = np.fft.irfft(fft_ties * (1 - np.abs(g)), norm="forward")

    ties_filtered = ties_interpolated - edge_adj
    out_filtered = out_constant + edge_adj
    return out_filtered, ties_filtered
