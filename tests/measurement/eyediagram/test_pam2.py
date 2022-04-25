"""Test module hardware_tools.measurement.pam2
"""

from __future__ import annotations

import io
import os
import time
from unittest import mock

import numpy as np
from scipy import signal

from hardware_tools.math import image, stats
from hardware_tools.measurement import mask
from hardware_tools.measurement.eyediagram import pam2, _pam2, _pam2_fb
from hardware_tools.measurement.eyediagram import cdr, eyediagram

from tests import base

# TODO (WattsUp) Replace these generators with signal module
_rng = np.random.default_rng()
f_scope = 5e9
n_scope = int(1e5)
t_scope = n_scope / f_scope
n_bits = int(1e3)
t_bit = t_scope / n_bits
f_bit = 1 / t_bit
bits = signal.max_len_seq(5, length=n_bits)[0]
t = np.linspace(0, (n_scope - 1) / f_scope, n_scope)
v_signal = 3.3
v_error = v_signal * 0.05
clock = (signal.square(2 * np.pi * (f_bit * t + 0.5)) + 1) / 2 * v_signal
y = (np.repeat(bits, n_scope / n_bits) +
     _rng.normal(0, 0.05, size=n_scope)) * v_signal
_f_lp = 0.75 / t_bit
_sos = signal.bessel(4, _f_lp, fs=f_scope, output="sos", norm="mag")

_zi = signal.sosfilt_zi(_sos) * y[0]
y_filtered, _ = signal.sosfilt(_sos, y, zi=_zi)

t_delta = 1 / f_scope

i_width = int((t_bit / t_delta) + 0.5) + 2
max_i = n_scope

centers_i = []
centers_t = []
t_zero = t[0]
clock_edges = np.linspace(t_zero, t_zero + t_bit * (n_bits - 1),
                          n_bits) - 0.06 * t_bit  # Delay for filter
for b in clock_edges:
  center_t = b % t_delta
  center_i = int(((b - t_zero - center_t) / t_delta) + 0.5)

  if (center_i - i_width) < 0 or (center_i + i_width) >= max_i:
    continue
  centers_t.append(-center_t)
  centers_i.append(center_i)
centers_i = np.fromiter(centers_i, np.int32)
centers_t = np.fromiter(centers_t, np.float64)
edge_dir = _pam2_fb.sample_vertical(y_filtered, centers_t, centers_i, t_delta,
                                    t_bit, v_signal / 2, 0.2, 0.1)["edge_dir"]
edge_dir = np.fromiter(edge_dir, np.int8)


class TestPAM2Ext(base.TestBase):
  """Test PAM2 methods
  """

  def _test_sample_vertical(self, module: _pam2_fb):
    result = module.sample_vertical(y_filtered, centers_t, centers_i, t_delta,
                                    t_bit, v_signal / 2, 0.2, 0.1)

    target = {
        "y_0": 0,
        "y_1": v_signal,
        "y_cross": v_signal / 2,
        "y_0_cross": 0,
        "y_1_cross": v_signal
    }
    for k, v in result.items():
      if k in target:
        v_mean = np.mean(v)
        self.assertEqualWithinError(target[k], v_mean, 0.05)
    self.assertEqualWithinError(y_filtered.mean(), np.mean(result["y_avg"]),
                                0.01)

    for k, v in result["transitions"].items():
      # Generated bit pattern lacks even number of 000
      if k == "000":
        self.assertEqualWithinError(n_bits / 8 * 25 / 32, v, 0.1)
      else:
        self.assertEqualWithinError(n_bits / 8 * 33 / 32, v, 0.1)

  def test_sample_vertical(self):
    self._test_sample_vertical(_pam2_fb)
    self._test_sample_vertical(_pam2)

    # Validate fast is actually faster

    start = time.perf_counter()
    result_slow = _pam2_fb.sample_vertical(y_filtered, centers_t, centers_i,
                                           t_delta, t_bit, v_signal / 2, 0.2,
                                           0.1)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = _pam2.sample_vertical(y_filtered, centers_t, centers_i,
                                        t_delta, t_bit, v_signal / 2, 0.2, 0.1)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertListEqual(result_slow["edge_dir"], result_fast["edge_dir"])
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_sample_horizontal(self, module: _pam2_fb):
    result = module.sample_horizontal(y_filtered, centers_t, centers_i,
                                      edge_dir, t_delta, t_bit, 0.0, v_signal,
                                      v_signal / 2, 0.05, 0.2, 0.8)

    target = {
        "t_rise_lower": -0.148,
        "t_rise_upper": 0.168,
        "t_rise_half": 0,
        "t_fall_lower": 0.168,
        "t_fall_upper": -0.148,
        "t_fall_half": 0,
        "t_cross_left": 0,
        "t_cross_right": 1
    }
    for k, v in result.items():
      if k in target:
        v_mean = np.mean(v)
        self.assertEqualWithinError(target[k], v_mean, 0.05)

  def test_sample_horizontal(self):
    self._test_sample_horizontal(_pam2_fb)
    self._test_sample_horizontal(_pam2)

    # Validate fast is actually faster

    start = time.perf_counter()
    result_slow = _pam2_fb.sample_horizontal(y_filtered, centers_t, centers_i,
                                             edge_dir, t_delta, t_bit, 0.0,
                                             v_signal, v_signal / 2, 0.05, 0.2,
                                             0.8)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = _pam2.sample_horizontal(y_filtered, centers_t, centers_i,
                                          edge_dir, t_delta, t_bit, 0.0,
                                          v_signal, v_signal / 2, 0.05, 0.2,
                                          0.8)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    for k, v in result_slow.items():
      v_mean_slow = np.mean(v)
      v_mean_fast = np.mean(result_fast[k])
      self.assertEqualWithinError(v_mean_slow, v_mean_fast, 0.01)
    self.assertLess(elapsed_fast, elapsed_slow)


class TestPAM2(base.TestBase):
  """Test PAM2
  """

  def test_init(self):
    waveforms = np.array([t, y])
    clocks = np.array([t, clock])
    pam2.PAM2(waveforms)
    pam2.PAM2(waveforms, clocks=clocks)

    self.assertRaises(ValueError,
                      pam2.PAM2,
                      waveforms,
                      config=eyediagram.Config())

  def test_step1(self):
    n = n_scope // 10  # Don't need that many samples to test step 1
    path = str(self._TEST_ROOT.joinpath("eyediagram_pam2"))
    waveforms = np.array([t[:n], y[:n]])

    noise = np.array([t[:n], self._RNG.uniform(-v_signal, 2 * v_signal, n)])

    v_half = v_signal * 0.3
    v_rising = v_signal * 0.35
    v_falling = v_signal * 0.25
    c = pam2.PAM2Config(y_0=0, y_1=v_signal, y_th=v_half, levels_n_max=None)
    eye = pam2.PAM2(noise, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, debug_plots=path)  # pylint: disable=protected-access
    self.assertTrue(os.path.exists(path + ".step1.png"))
    fake_stdout = fake_stdout.getvalue()
    self.assertNotIn("Completed PAM2 levels", fake_stdout)
    self.assertIn("Saved image to", fake_stdout)
    self.assertEqualWithinError(0, eye._y_zero / v_signal, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_signal, eye._y_ua, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_half, eye._y_half, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_rising, eye._y_rising, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_falling, eye._y_falling, 0.01)  # pylint: disable=protected-access
    self.assertTrue(eye._low_snr)  # pylint: disable=protected-access

    noise = np.array([t[:n], self._RNG.uniform(-v_signal, 2 * v_signal, n)])
    c = pam2.PAM2Config(y_0=0, y_1=v_signal, levels_n_max=n // 9)
    eye = pam2.PAM2(noise, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, debug_plots=path)  # pylint: disable=protected-access
    self.assertTrue(os.path.exists(path + ".step1.png"))
    fake_stdout = fake_stdout.getvalue()
    self.assertNotIn("Completed PAM2 levels", fake_stdout)
    self.assertIn("Saved image to", fake_stdout)
    self.assertEqualWithinError(0, eye._y_zero / v_signal, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_signal, eye._y_ua, 0.01)  # pylint: disable=protected-access
    v_half = v_signal * 0.5
    v_rising = v_signal * 0.55
    v_falling = v_signal * 0.45
    self.assertEqualWithinError(v_half, eye._y_half, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_rising, eye._y_rising, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_falling, eye._y_falling, 0.01)  # pylint: disable=protected-access
    self.assertTrue(eye._low_snr)  # pylint: disable=protected-access

    hysteresis = 0.5
    c = pam2.PAM2Config(y_0=0, y_1=v_signal, hysteresis=hysteresis)
    eye = pam2.PAM2(waveforms, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
    self.assertEqual("", fake_stdout.getvalue())
    self.assertEqualWithinError(0, eye._y_zero / v_signal, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_signal, eye._y_ua, 0.01)  # pylint: disable=protected-access
    v_half = v_signal * 0.5
    v_rising = v_half + hysteresis / 2
    v_falling = v_half - hysteresis / 2
    self.assertEqualWithinError(v_half, eye._y_half, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_rising, eye._y_rising, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_falling, eye._y_falling, 0.01)  # pylint: disable=protected-access
    self.assertFalse(eye._low_snr)  # pylint: disable=protected-access

  def test_step2(self):
    path = str(self._TEST_ROOT.joinpath("eyediagram_pam2"))
    waveforms = np.array([t, y])
    clocks = np.array([t, clock])

    c = pam2.PAM2Config(y_0=0,
                        y_1=v_signal,
                        cdr=cdr.CDR(t_bit),
                        clock_polarity=pam2.EdgePolarity.RISING)
    eye = pam2.PAM2(waveforms, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, debug_plots=path)  # pylint: disable=protected-access
    self.assertTrue(os.path.exists(path + ".step2.png"))
    fake_stdout = fake_stdout.getvalue()
    self.assertNotIn("Calculating symbol period", fake_stdout)
    self.assertIn("Saved image to", fake_stdout)

    c = pam2.PAM2Config(y_0=0,
                        y_1=v_signal,
                        fallback_period=t_bit,
                        clock_polarity=pam2.EdgePolarity.RISING)
    eye = pam2.PAM2(waveforms, clocks=clocks, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, debug_plots=path)  # pylint: disable=protected-access
    self.assertTrue(os.path.exists(path + ".step2.png"))
    fake_stdout = fake_stdout.getvalue()
    self.assertNotIn("Calculating symbol period", fake_stdout)
    self.assertIn("Saved image to", fake_stdout)
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits), 1)  # pylint: disable=protected-access

    c = pam2.PAM2Config(y_0=0,
                        y_1=v_signal,
                        fallback_period=t_bit,
                        clock_polarity=pam2.EdgePolarity.FALLING)
    eye = pam2.PAM2(waveforms, clocks=clocks, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
    self.assertEqual("", fake_stdout.getvalue())
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits), 1)  # pylint: disable=protected-access

    c = pam2.PAM2Config(y_0=0,
                        y_1=v_signal,
                        fallback_period=t_bit,
                        clock_polarity=pam2.EdgePolarity.BOTH)
    eye = pam2.PAM2(waveforms, clocks=clocks, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
    self.assertEqual("", fake_stdout.getvalue())
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits * 2), 1)  # pylint: disable=protected-access

    c = pam2.PAM2Config(y_0=0, y_1=v_signal, fallback_period=t_bit)
    eye = pam2.PAM2(waveforms, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._low_snr = True  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
    self.assertEqual("", fake_stdout.getvalue())
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits), 1)  # pylint: disable=protected-access

  def test_step4(self):
    path = str(self._TEST_ROOT.joinpath("eyediagram_pam2"))
    waveforms = np.array([t, y])
    clocks = np.array([t, clock])

    c = pam2.PAM2Config(y_0=0, y_1=v_signal, y_th=v_signal / 2)
    eye = pam2.PAM2(waveforms, clocks=clocks, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._clock_edges = [[]]  # pylint: disable=protected-access
      eye._step3_sample(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._step4_measure(print_progress=False, debug_plots=path)  # pylint: disable=protected-access
    self.assertTrue(os.path.exists(path + ".step4.png"))
    fake_stdout = fake_stdout.getvalue()
    self.assertNotIn("Completed PAM2 measuring", fake_stdout)
    self.assertIn("Saved image to", fake_stdout)

    target = {
        "n_sym": 0,
        "n_samples": 0,
        "transition_dist": {
            "000": 0,
            "001": 0,
            "010": 0,
            "011": 0,
            "100": 0,
            "101": 0,
            "110": 0,
            "111": 0,
        }
    }
    m = eye._measures  # pylint: disable=protected-access
    for k, v in m.to_dict().items():
      if k in target:
        if isinstance(v, dict):
          self.assertDictEqual(target[k], v)
        else:
          self.assertEqual(target[k], v)
      elif not k.startswith("image"):
        if isinstance(v, stats.UncertainValue):
          self.assertTrue(v.isnan())
        elif not isinstance(v, dict):
          self.assertTrue(np.isnan(v))

    c = pam2.PAM2Config(y_0=0,
                        y_1=v_signal,
                        y_th=v_signal / 2,
                        skip_measures=True)
    eye = pam2.PAM2(waveforms, clocks=clocks, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._clock_edges = [[]]  # pylint: disable=protected-access
      eye._step3_sample(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._step4_measure(print_progress=False, debug_plots=path)  # pylint: disable=protected-access
    m = eye._measures  # pylint: disable=protected-access
    for k, v in m.to_dict().items():
      if k in target:
        self.assertEqual(target[k], v)
      elif not k.startswith("image"):
        self.assertIsNone(v)

  def test_optical_1e8(self):
    path = str(self._TEST_ROOT.joinpath("eyediagram_pam2-optical-1e8"))
    data_path = str(self._DATA_ROOT.joinpath("pam2-optical-1e8.npz"))
    with np.load(data_path) as file_zip:
      waveforms = file_zip[file_zip.files[0]]
    waveforms_unfiltered = np.copy(waveforms)

    # 4th order bessel filter since O/E converter's is too high bandwidth
    fs = (waveforms.shape[2] - 1) / (waveforms[0, 0, -1] - waveforms[0, 0, 0])
    t_sym = 8e-9
    f_lp = 0.75 / t_sym
    filter_delay = t_sym / 0.75 / 2.11391767490422 / np.sqrt(2)
    sos = signal.bessel(4, f_lp, fs=fs, output="sos", norm="mag")

    zi = signal.sosfilt_zi(sos) * waveforms[0, 1, 0]
    waveforms[0, 1], _ = signal.sosfilt(sos, waveforms[0, 1], zi=zi)
    zi = signal.sosfilt_zi(sos) * waveforms[1, 1, 0]
    waveforms[1, 1], _ = signal.sosfilt(sos, waveforms[1, 1], zi=zi)

    m = mask.MaskDecagon(0.01, 0.29, 0.35, 0.35, 0.38, 0.4, 0.5)
    c = pam2.PAM2Config(fallback_period=t_sym,
                        clock_polarity=pam2.EdgePolarity.BOTH,
                        noise_floor=stats.UncertainValue(
                            9.80964764e-08, 2.50442542e-07))

    eye = pam2.PAM2(waveforms[:1], resolution=1000, mask=m, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye.calculate(print_progress=True, debug_plots=None)
    fake_stdout = fake_stdout.getvalue()
    self.assertIn("Starting eye diagram calculation", fake_stdout)
    self.assertIn("Step 1: Finding threshold levels", fake_stdout)
    self.assertIn("Starting PAM2 levels", fake_stdout)
    self.assertIn("Computing thresholds", fake_stdout)
    self.assertIn("Completed PAM2 levels", fake_stdout)
    self.assertIn("Step 2: Determining receiver clock", fake_stdout)
    self.assertIn("Running clock data recovery", fake_stdout)
    self.assertIn("Calculating symbol period", fake_stdout)
    self.assertIn("Step 3: Aligning symbol sampling points", fake_stdout)
    self.assertIn("Starting sampling", fake_stdout)
    self.assertIn("Completed sampling", fake_stdout)
    self.assertIn("Step 4: Measuring waveform", fake_stdout)
    self.assertIn("Measuring waveform vertically", fake_stdout)
    self.assertIn("Measuring waveform horizontally", fake_stdout)
    self.assertIn("Measuring waveform mask", fake_stdout)
    self.assertIn("Completed PAM2 measuring", fake_stdout)
    self.assertIn("Step 5: Stacking waveforms", fake_stdout)
    self.assertIn("Starting stacking", fake_stdout)
    self.assertIn("Completed stacking", fake_stdout)
    self.assertIn("Completed eye diagram calculation", fake_stdout)

    m = eye.get_measures()
    for k, v in m.to_dict().items():
      self.assertIsNotNone(v, msg=f"Key={k}")

    m.save_images(path)

    path = str(
        self._TEST_ROOT.joinpath("eyediagram_pam2-optical-1e8-unfiltered"))
    c = pam2.PAM2Config(noise_floor=stats.UncertainValue(
        9.80964764e-08, 2.50442542e-07),
                        y_0=m.y_0.value,
                        y_1=m.y_1.value)
    m = mask.MaskDecagon(0.01, 0.29, 0.35, 0.35, 0.38, 0.4, 0.5)

    clk_edges = eye.get_clock_edges()
    for edges in clk_edges:
      for ii in range(len(edges)):
        edges[ii] -= filter_delay

    eye = pam2.PAM2(waveforms_unfiltered[:1],
                    clock_edges=clk_edges,
                    resolution=1000,
                    mask=m,
                    config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye.calculate(print_progress=True, debug_plots=None)

    m = eye.get_measures()
    for k, v in m.to_dict().items():
      self.assertIsNotNone(v, msg=f"Key={k}")
    m.save_images(path)

  def test_get_bitstream(self):
    waveforms = np.array([t, y])
    clocks = np.array([t, clock])

    c = pam2.PAM2Config(y_0=0, y_1=v_signal, y_th=v_signal / 2)
    eye = pam2.PAM2(waveforms, clocks=clocks, config=c)
    self.assertRaises(RuntimeError, eye.get_bit_stream)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._step3_sample(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._step4_measure(print_progress=False, debug_plots=None)  # pylint: disable=protected-access
      eye._calculated = True  # pylint: disable=protected-access

    result = eye.get_bit_stream()
    self.assertIsInstance(result, list)
    self.assertEqual(len(result), 1)
    result = result[0]
    self.assertIsInstance(result, list)
    # Approx since eye will trim a few on either side
    self.assertGreaterEqual(len(result), n_bits - 5)
    self.assertLessEqual(len(result), n_bits)
    bits_str = "".join([f"{i}" for i in bits])
    result_str = "".join([f"{i}" for i in result])
    self.assertIn(result_str, bits_str)

    bits_nrzm = []
    state = bits[0]
    for bit in bits:
      if bit == state:
        bits_nrzm.append(0)
      else:
        bits_nrzm.append(1)
        state = bit
    result = eye.get_bit_stream(nrzm=True)
    self.assertIsInstance(result, list)
    self.assertEqual(len(result), 1)
    result = result[0]
    self.assertIsInstance(result, list)
    # Approx since eye will trim a few on either side
    self.assertGreaterEqual(len(result), n_bits - 5)
    self.assertLessEqual(len(result), n_bits)
    bits_str = "".join([f"{i}" for i in bits_nrzm])
    result_str = "".join([f"{i}" for i in result])
    self.assertIn(result_str, bits_str)

    eye._edge_dir = [[]]  # pylint: disable=protected-access
    result = eye.get_bit_stream()
    self.assertIsInstance(result, list)
    self.assertEqual(len(result), 1)
    result = result[0]
    self.assertIsInstance(result, list)
    self.assertEqual(len(result), 0)


class TestEyeDiagramMeasuresPAM2(base.TestBase):
  """Test MeasuresPAM2
  """

  def test_to_dict(self):
    n_sym = self._RNG.integers(1000, 10000)
    n_sym_bad = self._RNG.integers(0, n_sym)
    n_samples = n_sym * self._RNG.integers(5, 50)
    mask_margin = self._RNG.uniform(-1, 1)
    transition_dist = {
        "000": self._RNG.integers(0, n_sym),
        "001": self._RNG.integers(0, n_sym),
        "010": self._RNG.integers(0, n_sym),
        "011": self._RNG.integers(0, n_sym),
        "100": self._RNG.integers(0, n_sym),
        "101": self._RNG.integers(0, n_sym),
        "110": self._RNG.integers(0, n_sym),
        "111": self._RNG.integers(0, n_sym),
    }

    resolution = 10
    shape = (resolution, resolution, 4)

    np_clean = self._RNG.uniform(0.0, 1.0, size=shape)
    np_grid = self._RNG.uniform(0.0, 1.0, size=shape)
    np_mask = self._RNG.uniform(0.0, 1.0, size=shape)
    np_hits = self._RNG.uniform(0.0, 1.0, size=shape)
    np_margin = self._RNG.uniform(0.0, 1.0, size=shape)

    bathtub_curves = {"Eye 0": self._RNG.uniform(0.0, 1.0, size=(2, n_sym))}

    y_0 = self._RNG.uniform(-1, 1)
    y_1 = self._RNG.uniform(-1, 1)
    y_cross = self._RNG.uniform(-1, 1)
    y_cross_r = self._RNG.uniform(-1, 1)
    y_avg = self._RNG.uniform(-1, 1)

    y_0_cross = self._RNG.uniform(-1, 1)
    y_1_cross = self._RNG.uniform(-1, 1)

    amp = self._RNG.uniform(-1, 1)
    height = self._RNG.uniform(-1, 1)
    height_r = self._RNG.uniform(-1, 1)
    snr = self._RNG.uniform(-1, 1)

    t_sym = self._RNG.uniform(-1, 1)
    t_0 = self._RNG.uniform(-1, 1)
    t_1 = self._RNG.uniform(-1, 1)
    t_rise = self._RNG.uniform(-1, 1)
    t_fall = self._RNG.uniform(-1, 1)
    t_rise_start = self._RNG.uniform(-1, 1)
    t_fall_start = self._RNG.uniform(-1, 1)
    t_cross = self._RNG.uniform(-1, 1)

    f_sym = self._RNG.uniform(-1, 1)

    width = self._RNG.uniform(-1, 1)
    width_r = self._RNG.uniform(-1, 1)
    dcd = self._RNG.uniform(-1, 1)
    dcd_r = self._RNG.uniform(-1, 1)
    jitter_pp = self._RNG.uniform(-1, 1)
    jitter_rms = self._RNG.uniform(-1, 1)

    extinction_ratio = self._RNG.uniform(-1, 1)
    oma_cross = self._RNG.uniform(-1, 1)
    vecp = self._RNG.uniform(-1, 1)

    m = pam2.MeasuresPAM2()
    m.set_images(np_clean, np_grid, np_mask, np_hits, np_margin)
    m.n_samples = n_samples
    m.n_sym = n_sym
    m.n_sym_bad = n_sym_bad
    m.transition_dist = transition_dist
    m.mask_margin = mask_margin
    m.bathtub_curves = bathtub_curves

    m.y_0 = y_0
    m.y_1 = y_1
    m.y_cross = y_cross
    m.y_cross_r = y_cross_r
    m.y_avg = y_avg

    m.y_0_cross = y_0_cross
    m.y_1_cross = y_1_cross

    m.amp = amp
    m.height = height
    m.height_r = height_r
    m.snr = snr

    m.t_sym = t_sym
    m.t_0 = t_0
    m.t_1 = t_1
    m.t_rise = t_rise
    m.t_fall = t_fall
    m.t_rise_start = t_rise_start
    m.t_fall_start = t_fall_start
    m.t_cross = t_cross

    m.f_sym = f_sym

    m.width = width
    m.width_r = width_r
    m.dcd = dcd
    m.dcd_r = dcd_r
    m.jitter_pp = jitter_pp
    m.jitter_rms = jitter_rms

    m.extinction_ratio = extinction_ratio
    m.oma_cross = oma_cross
    m.vecp = vecp

    d = {
        "n_samples": n_samples,
        "n_sym": n_sym,
        "n_sym_bad": n_sym_bad,
        "mask_margin": mask_margin,
        "transition_dist": transition_dist,
        "bathtub_curves": bathtub_curves,
        "y_0": y_0,
        "y_1": y_1,
        "y_cross": y_cross,
        "y_cross_r": y_cross_r,
        "y_avg": y_avg,
        "y_0_cross": y_0_cross,
        "y_1_cross": y_1_cross,
        "amp": amp,
        "height": height,
        "height_r": height_r,
        "snr": snr,
        "t_sym": t_sym,
        "t_0": t_0,
        "t_1": t_1,
        "t_rise": t_rise,
        "t_fall": t_fall,
        "t_rise_start": t_rise_start,
        "t_fall_start": t_fall_start,
        "t_cross": t_cross,
        "f_sym": f_sym,
        "width": width,
        "width_r": width_r,
        "dcd": dcd,
        "dcd_r": dcd_r,
        "jitter_pp": jitter_pp,
        "jitter_rms": jitter_rms,
        "extinction_ratio": extinction_ratio,
        "oma_cross": oma_cross,
        "vecp": vecp,
        "image_clean": image.np_to_base64(np_clean),
        "image_grid": image.np_to_base64(np_grid),
        "image_mask": image.np_to_base64(np_mask),
        "image_hits": image.np_to_base64(np_hits),
        "image_margin": image.np_to_base64(np_margin)
    }
    m_dict = m.to_dict()
    self.assertListEqual(sorted(d.keys()), sorted(m_dict.keys()))
    self.assertDictEqual(d, m_dict)
