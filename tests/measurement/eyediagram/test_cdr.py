"""Test module hardware_tools.measurement.cdr
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from hardware_tools.measurement.eyediagram import cdr
from hardware_tools.signal import clock

from tests import base


class TestCDR(base.TestBase):
  """Base CDR testing class
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._cdr = None
    self._t_sym = 0.5e-9  # 1GHz clock
    self._n_sym = int(50e3)
    self._n_data_edges = self._n_sym

  def _plot(self, out, ties, ties_fit=None):
    from matplotlib import pyplot  # pylint: disable=import-outside-toplevel
    _, subplots = pyplot.subplots(3, 1)
    periods = np.diff(out)
    subplots[0].plot(np.array(periods) - self._t_sym)
    subplots[0].set_title("Period error")
    subplots[1].plot(ties)
    subplots[1].set_title("Time interval error")
    subplots[2].hist(ties, 50, density=True, alpha=0.5)
    subplots[2].set_title("Time interval error")
    if ties_fit is not None:
      subplots[2].plot(ties_fit[0], ties_fit[1])
    pyplot.show()

  def _validate_quality(self, out, ties):
    # Lock on within 1000 clocks
    self.assertGreaterEqual(len(out), self._n_sym - 1000)
    self.assertEqual(len(ties), self._n_data_edges)

    periods = np.diff(out)

    # Within 10ppm accuracy
    self.assertEqualWithinError(self._t_sym, periods.mean(), 10e-6)

    # Lower than 100ppm rms jitter and 1000ppm peak-to-peak jitter
    self.assertEqualWithinError(0, periods.std(), 100e-6)
    self.assertEqualWithinError(0, periods.ptp(), 1000e-6)

    # Average TIE should be zero due to de-trending
    self.assertEqualWithinError(0, ties.mean(), 100e-6)

  def _adjust_edges(self, edges):
    # Offset the edges to a random start
    t_start = -self._t_sym * self._RNG.uniform(-20, 20)

    # Remove edges every so often
    every = self._RNG.integers(3, 7)
    edges_a = edges[::every]
    edges_b = edges[::every + 7]
    edges = np.unique(np.append(edges_a, edges_b)) + t_start
    self._n_data_edges = len(edges)
    return edges

  def setUp(self):
    super().setUp()
    # Override for derrived classes
    self._cdr = cdr.CDR(t_sym=self._t_sym)

  def test_ideal(self):
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

  def test_fewer_edges(self):
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

  def test_no_initial(self):
    self._cdr._t_sym_initial = None  # pylint: disable=protected-access

    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

  def test_glitch(self):
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym)
    edges = self._adjust_edges(edges)
    glitches = edges[::1000] + 0.2 * self._t_sym
    edges = np.unique(np.append(edges, glitches))
    self._n_data_edges = len(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

    self._cdr._t_sym_initial = None  # pylint: disable=protected-access

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

  def test_no_lock(self):
    # Wrong initial conditions
    edges = clock.edges(t_sym=self._t_sym / 10, n=self._n_sym)
    self.assertRaises(ArithmeticError, self._cdr.run, edges)

    self._cdr._t_sym_initial = None  # pylint: disable=protected-access

    edges = np.array([0, 0.1, 10])
    self.assertRaises(ArithmeticError, self._cdr.run, edges)

  def test_slow(self):
    factor = 1 + self._RNG.uniform(1000e-6, 5000e-6)
    self._cdr._t_sym_initial = self._t_sym * factor  # pylint: disable=protected-access

    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

  def test_random_jitter(self):
    t_rj = 0.002 * self._t_sym
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym, t_rj=t_rj)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

    # Check TIEs is gaussian of proper width
    self.assertEqualWithinSampleError(t_rj, ties.std(), len(ties))

  def test_uniform_jitter(self):
    t_uj = 0.002 * self._t_sym
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym, t_uj=t_uj)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

    # Check TIEs is uniform of proper width
    self.assertEqualWithinSampleError(t_uj, ties.ptp(), len(ties))
    adj_stddev_tie = np.sqrt(ties.std()**2 * 12)
    self.assertEqualWithinSampleError(t_uj, adj_stddev_tie, len(ties))

  def test_sinusoidal_jitter(self):
    t_sj = 0.002 * self._t_sym
    f_sj = 0.01 / self._t_sym
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym, t_sj=t_sj, f_sj=f_sj)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

    # Check TIEs is sinusoid of proper frequency and amplitude
    self.assertEqualWithinSampleError(t_sj * 2, ties.ptp(), len(ties))

    # FFT does not work due to uneven sample spacing
    # Use Lomb-Scargle periodogram instead
    duration = edges.ptp()
    freqs = np.linspace(1 / duration, self._n_data_edges / duration, 1000)
    periodogram = signal.lombscargle(edges, ties, freqs)
    freq = freqs[periodogram.argmax()] / (2 * np.pi)
    self.assertEqualWithinError(f_sj, freq, 0.01)

  def test_dcd(self):
    dcd = 0.1
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym, dcd=dcd)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

    # Check TIEs is bimodal of proper width
    self.assertEqualWithinSampleError(dcd * self._t_sym, ties.ptp(), len(ties))
    lower = ties.min() + 0.1 * dcd * self._t_sym
    upper = ties.max() - 0.1 * dcd * self._t_sym
    central_count = ((ties >= lower) & (ties <= upper)).mean()
    self.assertLessEqual(central_count, 0.01)

  def test_complex_jitter(self):
    factor = 1 - self._RNG.uniform(1000e-6, 5000e-6)
    self._cdr._t_sym_initial = self._t_sym * factor  # pylint: disable=protected-access

    t_rj = 0.002 * self._t_sym
    t_uj = 0.002 * self._t_sym
    t_sj = 0.002 * self._t_sym
    f_sj = 0.01 / self._t_sym
    dcd = 0.05
    edges = clock.edges(t_sym=self._t_sym,
                        n=self._n_sym,
                        t_rj=t_rj,
                        t_uj=t_uj,
                        t_sj=t_sj,
                        f_sj=f_sj,
                        dcd=dcd)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)
