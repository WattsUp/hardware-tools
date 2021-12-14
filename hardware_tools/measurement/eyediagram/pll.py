"""Phase Locked Loop for clock recovery
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats


class PLL(ABC):
  """Phase Locked Loop for clock recovery

  Must be derrived to support proper signal encoding
  """

  def __init__(self, t_sym: float) -> None:
    """Initialize PLL

    Args:
      t_sym: Initial PLL period for a single symbol
    """
    self.t_sym_initial = t_sym

  @abstractmethod
  def run(
      self,
      data_edges: list[float]) -> tuple[list[float], list[float], list[float]]:
    """Run PLL to generate clock edges from data edges

    Args:
      data_edges: List of data edges in time domain

    Returns:
      List of clock edges in time domain.
      List of clock periods.
      List of Time Interval Errors (TIEs).
    """
    pass  # pragma: no cover


class PLLSingleLowPass(PLL):
  """PLL with a single pole low pass filter on feedback
  """

  def __init__(self, t_sym: float, bandwidth: float) -> None:
    """Initialize PLL with a single pole low pass filter on feedback

    Args:
      t_sym: Initial PLL period for a single symbol
      bandwidth: Cutoff frequency for low pass filter
    """
    super().__init__(t_sym)
    self._bandwidth = bandwidth

  def run(
      self,
      data_edges: list[float]) -> tuple[list[float], list[float], list[float]]:
    clock = []
    periods = []
    ties = []

    # Initial PLL settings
    t = data_edges[0]
    delay = self.t_sym_initial
    phase = 0
    t_min = self.t_sym_initial / 2

    # Error variables
    t_ideal = t
    error_phase = 0
    error_delay = 0
    error_phase_last = 0

    # Debug information
    phases = []
    delays = []
    errors_phase = []
    errors_delay = []

    # Run PLL
    # Adjust delay and phase on even edges
    # Adjust offset on odd edges
    for edge in data_edges:
      n = 1
      error_edge = t - edge
      while error_edge < self.t_sym_initial / 2:
        # Record clock edge
        clock.append(t + delay / 2)

        # Adjust delay
        error_phase = 0
        error_delay = 0
        if error_edge > -self.t_sym_initial / 2:
          tie = (edge - t_ideal)
          ties.append(tie)
          error_phase = error_edge
          error_delay = (error_phase - error_phase_last) / n
        w = 2 * np.pi * delay * self._bandwidth
        alpha = w / (w + 1)
        # alpha = 1
        delay = (1 - alpha) * delay + alpha * (delay - error_delay)
        phase = (1 - alpha) * phase + alpha * (-error_phase)

        period = delay + phase
        if period < t_min:
          raise ArithmeticError("Clock recovery has failed with minimum period")

        # Record statistical information
        periods.append(period)
        phases.append(phase)
        delays.append(delay)
        errors_phase.append(error_phase)
        errors_delay.append(error_delay)

        # Step to next bit
        t += period
        t_ideal += self.t_sym_initial
        error_edge = t - edge
        n += 1

    # Compensate TIE for wrong tBit
    # Removes linear drift
    ties = np.array(ties)
    data_edges = np.array(data_edges)
    trend = stats.linregress(data_edges, ties)
    ties = (ties - (data_edges * trend.slope + trend.intercept)).tolist()

    self._errors_phase = errors_phase
    self._errors_delay = errors_delay
    return clock, periods, ties
