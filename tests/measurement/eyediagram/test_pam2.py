"""Test module hardware_tools.measurement.pam2
"""

from __future__ import annotations

import io
import os
from unittest import mock

import numpy as np
from scipy import signal

from hardware_tools import math
from hardware_tools.measurement import mask
from hardware_tools.measurement.eyediagram import cdr, pam2, eyediagram

from tests import base

_rng = np.random.default_rng()
f_scope = 5e9
n_scope = int(1e5)
t_scope = n_scope / f_scope
n_bits = int(10e3)
t_bit = t_scope / n_bits
f_bit = 1 / t_bit
bits = signal.max_len_seq(5, length=n_bits)[0]
t = np.linspace(0, t_scope, n_scope)
v_signal = 3.3
v_error = v_signal * 0.05
clock = (signal.square(2 * np.pi * (f_bit * t + 0.5)) + 1) / 2 * v_signal
y = (np.repeat(bits, n_scope / n_bits) +
     _rng.normal(0, 0.05, size=n_scope)) * v_signal


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

    c = pam2.PAM2Config(y_0=0, y_1=v_signal, levels_n_max=None)
    eye = pam2.PAM2(noise, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
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

    noise = np.array([t[:n], self._RNG.uniform(-v_signal, 2 * v_signal, n)])
    c = pam2.PAM2Config(y_0=0, y_1=v_signal, levels_n_max=n // 9)
    eye = pam2.PAM2(noise, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
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
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
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
                        clock_polarity=eyediagram.ClockPolarity.RISING)
    eye = pam2.PAM2(waveforms, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    self.assertTrue(os.path.exists(path + ".step2.png"))
    fake_stdout = fake_stdout.getvalue()
    self.assertNotIn("Calculating symbol period", fake_stdout)
    self.assertIn("Saved image to", fake_stdout)

    c = pam2.PAM2Config(y_0=0,
                        y_1=v_signal,
                        fallback_period=t_bit,
                        clock_polarity=eyediagram.ClockPolarity.RISING)
    eye = pam2.PAM2(waveforms, clocks=clocks, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    self.assertTrue(os.path.exists(path + ".step2.png"))
    fake_stdout = fake_stdout.getvalue()
    self.assertNotIn("Calculating symbol period", fake_stdout)
    self.assertIn("Saved image to", fake_stdout)
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits), 1)  # pylint: disable=protected-access

    c = pam2.PAM2Config(y_0=0,
                        y_1=v_signal,
                        fallback_period=t_bit,
                        clock_polarity=eyediagram.ClockPolarity.FALLING)
    eye = pam2.PAM2(waveforms, clocks=clocks, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
    self.assertEqual("", fake_stdout.getvalue())
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits), 1)  # pylint: disable=protected-access

    c = pam2.PAM2Config(y_0=0,
                        y_1=v_signal,
                        fallback_period=t_bit,
                        clock_polarity=eyediagram.ClockPolarity.BOTH)
    eye = pam2.PAM2(waveforms, clocks=clocks, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
    self.assertEqual("", fake_stdout.getvalue())
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits * 2), 1)  # pylint: disable=protected-access

    c = pam2.PAM2Config(y_0=0, y_1=v_signal, fallback_period=t_bit)
    eye = pam2.PAM2(waveforms, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._low_snr = True  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
    self.assertEqual("", fake_stdout.getvalue())
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits), 1)  # pylint: disable=protected-access

  def test_step4(self):
    path = str(self._TEST_ROOT.joinpath("eyediagram_pam2"))
    waveforms = np.array([t, y])
    clocks = np.array([t, clock])

    c = pam2.PAM2Config(y_0=0, y_1=v_signal)
    eye = pam2.PAM2(waveforms, clocks=clocks, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._clock_edges = [[]]  # pylint: disable=protected-access
      eye._step3_sample(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step4_measure(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    self.assertTrue(os.path.exists(path + ".step4.png"))
    fake_stdout = fake_stdout.getvalue()
    self.assertNotIn("Completed PAM2 measuring", fake_stdout)
    self.assertIn("Saved image to", fake_stdout)
    self.assertIn("y_0 does not", fake_stdout)
    self.assertIn("y_1 does not", fake_stdout)
    self.assertIn("y_cross does not", fake_stdout)
    self.assertIn("y_0_cross does not", fake_stdout)
    self.assertIn("y_1_cross does not", fake_stdout)
    self.assertIn("t_rise_lower does not", fake_stdout)
    self.assertIn("t_rise_upper does not", fake_stdout)
    self.assertIn("t_rise_half does not", fake_stdout)
    self.assertIn("t_fall_lower does not", fake_stdout)
    self.assertIn("t_fall_upper does not", fake_stdout)
    self.assertIn("t_fall_half does not", fake_stdout)
    self.assertIn("t_cross_left does not", fake_stdout)
    self.assertIn("t_cross_right does not", fake_stdout)

    m = eye._measures  # pylint: disable=protected-access
    self.assertTrue(np.isnan(m.amp.value))
    self.assertTrue(np.isnan(m.dcd.value))
    self.assertTrue(np.isnan(m.extinction_ratio.value))
    self.assertTrue(np.isnan(m.f_sym.value))
    self.assertTrue(np.isnan(m.height.value))
    self.assertTrue(np.isnan(m.height_r.value))
    self.assertTrue(np.isnan(m.jitter_pp))
    self.assertTrue(np.isnan(m.jitter_rms))
    self.assertTrue(np.isnan(m.mask_margin))
    self.assertEqual(0, m.n_samples)
    self.assertEqual(0, m.n_sym)
    self.assertTrue(np.isnan(m.n_sym_bad))
    self.assertTrue(np.isnan(m.oma_cross.value))
    self.assertTrue(np.isnan(m.snr.value))
    self.assertTrue(np.isnan(m.t_0.value))
    self.assertTrue(np.isnan(m.t_1.value))
    self.assertTrue(np.isnan(m.t_cross.value))
    self.assertTrue(np.isnan(m.t_fall.value))
    self.assertTrue(np.isnan(m.t_rise.value))
    self.assertTrue(np.isnan(m.t_sym.value))
    self.assertDictEqual(
        m.transition_dist, {
            "000": 0,
            "001": 0,
            "010": 0,
            "011": 0,
            "100": 0,
            "101": 0,
            "110": 0,
            "111": 0,
        })
    self.assertTrue(np.isnan(m.vecp.value))
    self.assertTrue(np.isnan(m.width.value))
    self.assertTrue(np.isnan(m.width_r.value))
    self.assertTrue(np.isnan(m.y_0.value))
    self.assertTrue(np.isnan(m.y_0_cross.value))
    self.assertTrue(np.isnan(m.y_1.value))
    self.assertTrue(np.isnan(m.y_1_cross.value))
    self.assertTrue(np.isnan(m.y_cross.value))
    self.assertTrue(np.isnan(m.y_cross_r.value))

  def test_optical_1e8(self):
    data_path = str(self._DATA_ROOT.joinpath("pam2-optical-1e8.npz"))
    with np.load(data_path) as file_zip:
      waveforms = file_zip[file_zip.files[0]]

    # 4th order bessel filter since O/E converter's is too high bandwidth
    fs = (waveforms.shape[2] - 1) / (waveforms[0, 0, -1] - waveforms[0, 0, 0])
    f_lp = 0.75 / 8e-9
    sos = signal.bessel(4, f_lp, fs=fs, output="sos", norm="mag")

    zi = signal.sosfilt_zi(sos) * waveforms[0, 1, 0]
    waveforms[0, 1], _ = signal.sosfilt(sos, waveforms[0, 1], zi=zi)
    zi = signal.sosfilt_zi(sos) * waveforms[1, 1, 0]
    waveforms[1, 1], _ = signal.sosfilt(sos, waveforms[1, 1], zi=zi)

    m = mask.MaskDecagon(0.01, 0.29, 0.35, 0.35, 0.38, 0.4, 0.5)
    c = pam2.PAM2Config(fallback_period=8e-9,
                        clock_polarity=eyediagram.ClockPolarity.BOTH,
                        noise_floor=math.UncertainValue(9.80964764e-08,
                                                        2.50442542e-07))

    eye = pam2.PAM2(waveforms, resolution=1000, mask=m, config=c)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye.calculate(print_progress=True, n_threads=1, debug_plots=None)
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
    self.assertNotIn("does not have any samples", fake_stdout)
    self.assertIn("Step 5: Stacking waveforms", fake_stdout)
    self.assertIn("Starting stacking", fake_stdout)
    self.assertIn("Completed stacking", fake_stdout)
    self.assertIn("Completed eye diagram calculation", fake_stdout)

    m = eye.get_measures()
    for k, v in m.to_dict().items():
      self.assertIsNotNone(v, msg=f"Key={k}")


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

    y_0 = self._RNG.uniform(-1, 1)
    y_1 = self._RNG.uniform(-1, 1)
    y_cross = self._RNG.uniform(-1, 1)
    y_cross_r = self._RNG.uniform(-1, 1)

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
    t_cross = self._RNG.uniform(-1, 1)

    f_sym = self._RNG.uniform(-1, 1)

    width = self._RNG.uniform(-1, 1)
    width_r = self._RNG.uniform(-1, 1)
    dcd = self._RNG.uniform(-1, 1)
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

    m.y_0 = y_0
    m.y_1 = y_1
    m.y_cross = y_cross
    m.y_cross_r = y_cross_r

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
    m.t_cross = t_cross

    m.f_sym = f_sym

    m.width = width
    m.width_r = width_r
    m.dcd = dcd
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
        "y_0": y_0,
        "y_1": y_1,
        "y_cross": y_cross,
        "y_cross_r": y_cross_r,
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
        "t_cross": t_cross,
        "f_sym": f_sym,
        "width": width,
        "width_r": width_r,
        "dcd": dcd,
        "jitter_pp": jitter_pp,
        "jitter_rms": jitter_rms,
        "extinction_ratio": extinction_ratio,
        "oma_cross": oma_cross,
        "vecp": vecp,
        "image_clean": math.Image.np_to_base64(np_clean),
        "image_grid": math.Image.np_to_base64(np_grid),
        "image_mask": math.Image.np_to_base64(np_mask),
        "image_hits": math.Image.np_to_base64(np_hits),
        "image_margin": math.Image.np_to_base64(np_margin)
    }
    m_dict = m.to_dict()
    self.assertListEqual(sorted(d.keys()), sorted(m_dict.keys()))
    self.assertDictEqual(d, m_dict)
