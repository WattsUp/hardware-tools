"""Test module hardware_tools.measurement.pam2
"""

from __future__ import annotations

import io
import os
import pathlib
from unittest import mock

import numpy as np
from scipy import signal

from hardware_tools import math
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

    self.assertRaises(KeyError,
                      pam2.PAM2,
                      waveforms,
                      this_keyword_does_not_exist=None)

  def test_step1(self):
    n = n_scope // 10  # Don't need that many samples to test step 1
    path = str(self._TEST_ROOT.joinpath("eyediagram_pam2"))
    waveforms = np.array([t[:n], y[:n]])

    eye = pam2.PAM2(waveforms, hist_n_max=None)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
    self.assertEqualWithinError(0, eye._y_zero / v_signal, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_signal, eye._y_ua, 0.01)  # pylint: disable=protected-access
    v_half = v_signal * 0.5
    v_rising = v_signal * 0.55
    v_falling = v_signal * 0.45
    self.assertEqualWithinError(v_half, eye._y_half, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_rising, eye._y_rising, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_falling, eye._y_falling, 0.01)  # pylint: disable=protected-access
    self.assertFalse(eye._low_snr)  # pylint: disable=protected-access

    eye = pam2.PAM2(waveforms, y_0=0, y_1=v_signal)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
    self.assertEqualWithinError(0, eye._y_zero / v_signal, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_signal, eye._y_ua, 0.01)  # pylint: disable=protected-access
    v_half = v_signal * 0.5
    v_rising = v_signal * 0.55
    v_falling = v_signal * 0.45
    self.assertEqualWithinError(v_half, eye._y_half, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_rising, eye._y_rising, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_falling, eye._y_falling, 0.01)  # pylint: disable=protected-access
    self.assertFalse(eye._low_snr)  # pylint: disable=protected-access

    hysteresis = 0.5
    y_new = y.copy()
    y_new[:-1] += y_new[1:] * 0.1
    waveforms = np.array([[t[:n], y[:n]], [t[:n], y_new[:n]]])
    eye = pam2.PAM2(waveforms, hysteresis=hysteresis)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=True, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    self.assertIn("Completed PAM2 levels", fake_stdout.getvalue())
    self.assertEqualWithinError(0, eye._y_zero / v_signal, 0.01)  # pylint: disable=protected-access
    self.assertEqualWithinError(v_signal * 1.05, eye._y_ua, 0.01)  # pylint: disable=protected-access
    v_half = v_signal * 0.525
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

    eye = pam2.PAM2(waveforms,
                    y_0=0,
                    y_1=v_signal,
                    cdr=cdr.CDR(t_bit),
                    clock_polarity=eyediagram.ClockPolarity.RISING)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access

    eye = pam2.PAM2(waveforms,
                    y_0=0,
                    y_1=v_signal,
                    fallback_period=t_bit,
                    clock_polarity=eyediagram.ClockPolarity.FALLING)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=True, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    self.assertIn("Calculating symbol period", fake_stdout.getvalue())

    eye = pam2.PAM2(waveforms,
                    y_0=0,
                    y_1=v_signal,
                    fallback_period=t_bit,
                    clock_polarity=eyediagram.ClockPolarity.BOTH)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access

    eye = pam2.PAM2(waveforms,
                    clocks=clocks,
                    y_0=0,
                    y_1=v_signal,
                    fallback_period=t_bit,
                    clock_polarity=eyediagram.ClockPolarity.RISING)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits), 1)  # pylint: disable=protected-access

    eye = pam2.PAM2(waveforms,
                    clocks=clocks,
                    y_0=0,
                    y_1=v_signal,
                    fallback_period=t_bit,
                    clock_polarity=eyediagram.ClockPolarity.FALLING)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits), 1)  # pylint: disable=protected-access

    eye = pam2.PAM2(waveforms,
                    clocks=clocks,
                    y_0=0,
                    y_1=v_signal,
                    fallback_period=t_bit,
                    clock_polarity=eyediagram.ClockPolarity.BOTH)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits * 2), 1)  # pylint: disable=protected-access

    eye = pam2.PAM2(waveforms, y_0=0, y_1=v_signal, fallback_period=t_bit)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._low_snr = True  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
    self.assertLessEqual(np.abs(len(eye._clock_edges[0]) - n_bits), 1)  # pylint: disable=protected-access


class TestEyeDiagramMeasuresPAM2(base.TestBase):
  """Test MeasuresPAM2
  """

  _TEST_ROOT = pathlib.Path(".test")

  def __clean_test_root(self):
    if self._TEST_ROOT.exists():
      for f in os.listdir(self._TEST_ROOT):
        os.remove(self._TEST_ROOT.joinpath(f))

  def setUp(self):
    self.__clean_test_root()
    self._TEST_ROOT.mkdir(parents=True, exist_ok=True)

  def tearDown(self):
    self.__clean_test_root()

  def test_to_dict(self):
    n_sym = self._RNG.integers(1000, 10000)
    n_sym_bad = self._RNG.integers(0, n_sym)
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

    resolution = 200
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

    amp = self._RNG.uniform(-1, 1)
    height = self._RNG.uniform(-1, 1)
    height_r = self._RNG.uniform(-1, 1)
    snr = self._RNG.uniform(-1, 1)

    t_sym = self._RNG.uniform(-1, 1)
    t_0 = self._RNG.uniform(-1, 1)
    t_1 = self._RNG.uniform(-1, 1)
    t_rise = self._RNG.uniform(-1, 1)
    t_fall = self._RNG.uniform(-1, 1)

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
    m.n_sym = n_sym
    m.n_sym_bad = n_sym_bad
    m.transition_dist = transition_dist
    m.mask_margin = mask_margin

    m.y_0 = y_0
    m.y_1 = y_1
    m.y_cross = y_cross
    m.y_cross_r = y_cross_r

    m.amp = amp
    m.height = height
    m.height_r = height_r
    m.snr = snr

    m.t_sym = t_sym
    m.t_0 = t_0
    m.t_1 = t_1
    m.t_rise = t_rise
    m.t_fall = t_fall

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
        "n_sym": n_sym,
        "n_sym_bad": n_sym_bad,
        "mask_margin": mask_margin,
        "transition_dist": transition_dist,
        "y_0": y_0,
        "y_1": y_1,
        "y_cross": y_cross,
        "y_cross_r": y_cross_r,
        "amp": amp,
        "height": height,
        "height_r": height_r,
        "snr": snr,
        "t_sym": t_sym,
        "t_0": t_0,
        "t_1": t_1,
        "t_rise": t_rise,
        "t_fall": t_fall,
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
    self.assertDictEqual(d, m.to_dict())
