"""Test module hardware_tools.measurement.cdr
"""

from __future__ import annotations

import functools
import time

import numpy as np
from scipy import signal

from hardware_tools.math import lines
from hardware_tools.measurement.eyediagram import cdr, _cdr, _cdr_fb
from hardware_tools.signal import clock

from tests import base


class TestCDRExt(base.TestBase):
  """Test _CDR methods
  """

  def _generate_edges(self, t_sym, n):
    t_rj = 0.002 * t_sym
    t_uj = 0.002 * t_sym
    t_sj = 0.002 * t_sym
    f_sj = 0.01 / t_sym
    dcd = 0.05
    edges = clock.edges(t_sym=t_sym,
                        n=n,
                        t_rj=t_rj,
                        t_uj=t_uj,
                        t_sj=t_sj,
                        f_sj=f_sj,
                        dcd=dcd)

    # Offset the edges to a random start
    t_start = -t_sym * self._RNG.uniform(-20, 20)

    # Remove edges every so often
    every = self._RNG.integers(3, 7)
    edges_a = edges[::every]
    edges_b = edges[::every + 7]
    edges = np.unique(np.append(edges_a, edges_b)) + t_start

    # Add glitches
    glitches = edges[::1000] + 0.2 * t_sym
    edges = np.unique(np.append(edges, glitches))
    return edges

  def _test_minimize_tie_disjoints(self, module: _cdr_fb):
    t_sym = 0.5e-9
    n = 10e3
    edges = self._generate_edges(t_sym, n)

    tol = 10
    result = module.minimize_tie_disjoints(edges, t_sym=t_sym, tol=tol)
    self.assertEqualWithinError(t_sym, result, 0.01)
    ties = np.mod(edges - edges[0] + result / 2, result)
    disjoints = (np.abs(np.diff(ties)) > result / 2).sum()
    self.assertLessEqual(disjoints, tol)

    self.assertRaises(ValueError, module.minimize_tie_disjoints, edges)

    self.assertRaises(ArithmeticError,
                      module.minimize_tie_disjoints,
                      edges,
                      t_min=t_sym * 1.311,
                      t_max=t_sym * 1.312,
                      max_iter=3)

    t_sj = 0.6 * t_sym
    f_sj = 2 / (n * t_sym)
    edges = clock.edges(t_sym=t_sym, n=n, t_sj=t_sj, f_sj=f_sj)
    self.assertRaises(ArithmeticError,
                      module.minimize_tie_disjoints,
                      edges,
                      t_sym=t_sym,
                      max_iter=0)

    # Fixed data to force successful finer step route
    n = 100e3
    edges = clock.edges(t_sym=t_sym, n=n)
    result = module.minimize_tie_disjoints(edges, t_sym=t_sym, tol=tol)
    self.assertEqualWithinError(t_sym, result, 0.01)
    ties = np.mod(edges - edges[0] + result / 2, result)
    disjoints = (np.abs(np.diff(ties)) > result / 2).sum()
    self.assertLessEqual(disjoints, tol)

  def test_minimize_tie_disjoints(self):
    self._test_minimize_tie_disjoints(_cdr_fb)
    self._test_minimize_tie_disjoints(_cdr)

    # Validate fast is actually faster
    t_sym = 0.5e-9
    n = 10e3
    edges = self._generate_edges(t_sym, n)

    tol = 10

    start = time.perf_counter()
    result_slow = _cdr_fb.minimize_tie_disjoints(edges, t_sym=t_sym, tol=tol)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = _cdr.minimize_tie_disjoints(edges, t_sym=t_sym, tol=tol)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertEqualWithinError(result_slow, result_fast, 1e-15)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_detrend_ties(self, module: _cdr_fb):
    t_sym = 0.5e-9
    n = 10e3
    edges = self._generate_edges(t_sym, n)

    period_tol = 1e-6
    result = module.detrend_ties(edges, t_sym=t_sym * (1 + 100e-6))
    self.assertEqualWithinError(t_sym, result, period_tol)

  def test_detrend_ties(self):
    self._test_detrend_ties(_cdr_fb)
    self._test_detrend_ties(_cdr)

    # Validate fast is actually faster
    t_sym = 0.5e-9
    n = 10e3
    edges = self._generate_edges(t_sym, n)

    start = time.perf_counter()
    result_slow = _cdr_fb.detrend_ties(edges, t_sym=t_sym * (1 - 100e-6))
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = _cdr.detrend_ties(edges, t_sym=t_sym * (1 - 100e-6))
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertEqualWithinError(result_slow, result_fast, 1e-15)
    self.assertLess(elapsed_fast, elapsed_slow)


class TestCDR(base.TestBase):
  """Base CDR testing class
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._cdr = None

  def _plot(self, out, ties, edges, ties_fit=None, resample_ties: bool = True):
    from matplotlib import pyplot  # pylint: disable=import-outside-toplevel
    _, subplots = pyplot.subplots(4, 1)
    periods = np.diff(out)
    subplots[0].plot(np.array(periods) / self._t_sym - 1)
    subplots[0].set_ylabel("UI")
    subplots[0].set_title(f"{self._testMethodName}: Period error")
    if resample_ties:
      subplots[1].plot(out, ties / self._t_sym)
    else:
      subplots[1].plot(edges, ties / self._t_sym)
    subplots[1].set_title("Time interval error")
    subplots[1].set_ylabel("UI")
    subplots[2].hist(ties / self._t_sym, 50, density=True, alpha=0.5)
    subplots[2].set_title("Time interval error")
    subplots[2].set_xlabel("UI")
    if ties_fit is not None:
      subplots[2].plot(ties_fit[0], ties_fit[1])

    t_sym_recovered = ((out[-1] - out[0]) / (len(out) - 1))
    fft = np.abs(np.fft.rfft(ties, norm="forward")) * 2
    freqs = np.fft.rfftfreq(len(ties), d=t_sym_recovered)
    subplots[3].plot(freqs, fft)
    pyplot.tight_layout()
    pyplot.show()

  def _validate_quality(self, out, ties, resample_ties: bool = True):
    # Lock on within 1000 clocks
    self.assertGreaterEqual(len(out), self._n_sym - 1000)
    if resample_ties:
      self.assertEqual(len(ties), len(out))
    else:
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
    every = 4
    t_start = -1.12312e-10
    edges_a = edges[::every]
    edges_b = edges[::every + 7]
    edges = np.unique(np.append(edges_a, edges_b)) + t_start
    self._n_data_edges = len(edges)
    return edges

  def setUp(self):
    super().setUp()
    self._t_sym = 0.5e-9  # 1GHz clock
    self._n_sym = int(10e3)
    self._n_data_edges = self._n_sym

    # Override for derived classes
    self._cdr = cdr.CDR(t_sym=self._t_sym)
    self._skip_real = False

  def test_ideal(self):
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

    # Clock center of data
    # Search for edge since CDR might trim some at start/end
    clock_edge = out[10] - self._t_sym / 2
    valid_min = clock_edge - self._t_sym / 2
    valid_max = clock_edge + self._t_sym / 2
    for data_edge in edges:
      if valid_min < data_edge < valid_max:
        self.assertEqualWithinError(data_edge, clock_edge, 0.01)
        return
    self.fail(msg="No matching data edge for clock edge")

  def test_fewer_edges(self):
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

  def test_no_initial(self):
    self._cdr._t_sym_initial = None  # pylint: disable=protected-access

    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym)
    edges = self._adjust_edges(edges)

    self.assertRaises(ValueError, self._cdr.run, edges)

  def test_glitch(self):
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym)
    edges = self._adjust_edges(edges)
    glitches = edges[::1000] + 0.2 * self._t_sym
    edges = np.unique(np.append(edges, glitches))
    self._n_data_edges = len(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

  def test_bad_initial(self):
    # Wrong initial conditions
    edges = clock.edges(t_sym=self._t_sym / 10, n=self._n_sym)
    self.assertRaises(ArithmeticError, self._cdr.run, edges)

    self.assertRaises(ValueError, self._cdr.run, edges[:1])

  def test_slow(self):
    factor = 1 + self._RNG.uniform(1000e-6, 5000e-6)
    self._cdr._t_sym_initial = self._t_sym * factor  # pylint: disable=protected-access

    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges)
    self._validate_quality(out, ties)

  def test_fixed_period(self):
    factor = 1 + self._RNG.uniform(10e-6, 50e-6)
    self._cdr._t_sym_initial = self._t_sym * factor  # pylint: disable=protected-access
    self._cdr._fixed_period = True  # pylint: disable=protected-access

    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges)
    # self._plot(out, ties, edges)
    periods = np.diff(out)

    # Within 100ppm accuracy
    self.assertEqualWithinError(self._t_sym * factor, periods.mean(), 1e-6)

    # Lower than 100ppm rms jitter and 1000ppm peak-to-peak jitter
    self.assertEqualWithinError(0, periods.std(), 100e-6)
    self.assertEqualWithinError(0, periods.ptp(), 1000e-6)

    # Average TIE should be zero due to de-trending
    self.assertEqualWithinError(0, ties.mean(), 100e-6)

  def test_random_jitter(self):
    t_rj = 0.002 * self._t_sym
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym, t_rj=t_rj)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges, resample_ties=False)
    self._validate_quality(out, ties, resample_ties=False)

    # Check TIEs is gaussian of proper width
    self.assertEqualWithinSampleError(t_rj, ties.std(), len(ties))

  def test_uniform_jitter(self):
    t_uj = 0.002 * self._t_sym
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym, t_uj=t_uj)
    edges = self._adjust_edges(edges)

    out, ties = self._cdr.run(edges, resample_ties=False)
    self._validate_quality(out, ties, resample_ties=False)

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
    # duration = edges.ptp()
    # freqs = np.linspace(1 / duration, self._n_data_edges / duration, 1000)
    # periodogram = signal.lombscargle(edges, ties, freqs)
    # freq = freqs[periodogram.argmax()] / (2 * np.pi)

    # Resampled TIEs allows FFT
    t_sym_recovered = ((out[-1] - out[0]) / (len(out) - 1))
    fft = np.abs(np.fft.rfft(ties, norm="forward")) * 2
    freqs = np.fft.rfftfreq(len(ties), d=t_sym_recovered)

    argmax = fft.argmax()
    freq = freqs[argmax]
    self.assertEqualWithinError(f_sj, freq, 0.01)
    # Sum the power nearby in case the peak is between buckets
    power = np.sum(fft[argmax - 1:argmax + 2])
    self.assertEqualWithinError(t_sj, power, 0.05)

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

  def test_too_much_jitter(self):
    t_sj = 0.6 * self._t_sym
    f_sj = 2 / (self._n_sym * self._t_sym)
    edges = clock.edges(t_sym=self._t_sym, n=self._n_sym, t_sj=t_sj, f_sj=f_sj)
    edges = self._adjust_edges(edges)

    self.assertRaises(ArithmeticError, self._cdr.run, edges)

  def test_real_optical_1e8(self):
    if self._skip_real:
      self.skipTest("Real waveform not being tested")
    data_path = str(self._DATA_ROOT.joinpath("pam2-optical-1e8.npz"))
    with np.load(data_path) as file_zip:
      waveforms = file_zip[file_zip.files[0]]
    waveforms = waveforms[0]

    # 4th order bessel filter since O/E converter's is too high bandwidth
    fs = (waveforms.shape[1] - 1) / (waveforms[0, -1] - waveforms[0, 0])
    f_lp = 0.75 / 8e-9
    sos = signal.bessel(4, f_lp, fs=fs, output="sos")
    zi = signal.sosfilt_zi(sos) * waveforms[1, 0]
    waveforms[1], _ = signal.sosfilt(sos, waveforms[1], zi=zi)

    edges = lines.edges_np(waveforms[0], waveforms[1], 7.39e-6, 6.82e-6,
                           6.26e-6)
    edges = np.sort(np.concatenate(edges))

    self._t_sym = 8.000084e-9
    self._cdr._t_sym_initial = 8e-9  # pylint: disable=protected-access
    out, ties = self._cdr.run(edges)
    # self._plot(out, ties, edges)
    periods = np.diff(out)

    # Within 100ppm accuracy
    self.assertEqualWithinError(self._t_sym, periods.mean(), 100e-6)

    # Lower than 100ppm rms jitter and 1000ppm peak-to-peak jitter
    self.assertEqualWithinError(0, periods.std(), 100e-6)
    self.assertEqualWithinError(0, periods.ptp(), 1000e-6)

    # Average TIE should be zero due to de-trending
    self.assertEqualWithinError(0, ties.mean(), 100e-6)

  def test_real_optical_1e9(self):
    if self._skip_real:
      self.skipTest("Real waveform not being tested")
    data_path = str(self._DATA_ROOT.joinpath("pam2-optical-1e9.npz"))
    with np.load(data_path) as file_zip:
      waveforms = file_zip[file_zip.files[0]]
    edges = lines.edges_np(waveforms[0], waveforms[1], 2.5e-04, 2.3e-4, 2.1e-4)
    edges = np.sort(np.concatenate(edges))

    self._t_sym = 7.999799e-10
    self._cdr._t_sym_initial = 8e-10  # pylint: disable=protected-access
    out, ties = self._cdr.run(edges)
    # self._plot(out, ties, edges)
    periods = np.diff(out)

    # Within 100ppm accuracy
    self.assertEqualWithinError(self._t_sym, periods.mean(), 100e-6)

    # Lower than 100ppm rms jitter and 1000ppm peak-to-peak jitter
    self.assertEqualWithinError(0, periods.std(), 100e-6)
    self.assertEqualWithinError(0, periods.ptp(), 1000e-6)

    # Average TIE should be zero due to de-trending
    self.assertEqualWithinError(0, ties.mean(), 100e-6)


class TestCDRWithFFTFilter(TestCDR):
  """CDRWithFFTFilter testing class
  """

  def setUp(self):
    super().setUp()
    self._bw = 0.01 / self._t_sym
    self._order = 1
    self._ojtf = functools.partial(cdr.CDRWithFFTFilter.high_pass_butter,
                                   bandwidth=self._bw,
                                   order=self._order)
    self._cdr = cdr.CDRWithFFTFilter(t_sym=self._t_sym, ojtf=self._ojtf)
    self._skip_real = False

  def test_fixed_period(self):
    self.skipTest("Filtered CDR will never have a fixed period")

  def test_random_jitter(self):
    # Remove filter since the filter will change the std.dev
    self._cdr._ojtf = lambda _: 1  # pylint: disable=protected-access
    return super().test_random_jitter()

  def test_uniform_jitter(self):
    # Remove filter since the filter will change the std.dev
    self._cdr._ojtf = lambda _: 1  # pylint: disable=protected-access
    return super().test_uniform_jitter()

  def test_sinusoidal_jitter(self):
    # Remove filter since the filter will change the std.dev
    self._cdr._ojtf = lambda _: 1  # pylint: disable=protected-access
    super().test_sinusoidal_jitter()

    # Revert filter
    self._cdr._ojtf = self._ojtf  # pylint: disable=protected-access

    # Test sj is attenuated proper amount
    for f_sj in [0.5 * self._bw, self._bw, 2 * self._bw]:
      t_sj = 0.002 * self._t_sym
      edges = clock.edges(t_sym=self._t_sym,
                          n=self._n_sym,
                          t_sj=t_sj,
                          f_sj=f_sj)
      edges = self._adjust_edges(edges)

      out, ties = self._cdr.run(edges)
      self._validate_quality(out, ties)

      g = self._ojtf(np.array([f_sj]))[0]
      self.assertEqualWithinSampleError(t_sj * 2 * g, ties.ptp(), len(ties))
