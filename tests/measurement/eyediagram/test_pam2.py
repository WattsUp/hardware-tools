"""Test module hardware_tools.measurement.pam2
"""

from __future__ import annotations

import io
import os
import pathlib
import unittest
from unittest import mock

import numpy as np
from scipy import signal

from hardware_tools import math
from hardware_tools.measurement.eyediagram import pam2

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
clock = (signal.square(2 * np.pi * f_bit * t) + 1) / 2 * v_signal
y = (np.repeat(bits, n_scope / n_bits) +
     np.random.normal(0, 0.05, size=n_scope)) * v_signal


class TestPAM2(unittest.TestCase):
  """Test PAM2
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
    path = str(self._TEST_ROOT.joinpath("eyediagram_pam2"))
    waveforms = np.array([t, y])

    eye = pam2.PAM2(waveforms, hist_n_max=None)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
    self.assertGreaterEqual(eye._y_zero, 0 - v_error)  # pylint: disable=protected-access
    self.assertLessEqual(eye._y_zero, 0 + v_error)  # pylint: disable=protected-access
    self.assertGreaterEqual(eye._y_ua, v_signal - v_error)  # pylint: disable=protected-access
    self.assertLessEqual(eye._y_ua, v_signal + v_error)  # pylint: disable=protected-access
    v_half = v_signal * 0.5
    v_rising = v_signal * 0.55
    v_falling = v_signal * 0.45
    self.assertGreaterEqual(eye._y_half, v_half - v_error)  # pylint: disable=protected-access
    self.assertLessEqual(eye._y_half, v_half + v_error)  # pylint: disable=protected-access
    self.assertGreaterEqual(eye._y_rising, v_rising - v_error)  # pylint: disable=protected-access
    self.assertLessEqual(eye._y_rising, v_rising + v_error)  # pylint: disable=protected-access
    self.assertGreaterEqual(eye._y_falling, v_falling - v_error)  # pylint: disable=protected-access
    self.assertLessEqual(eye._y_falling, v_falling + v_error)  # pylint: disable=protected-access
    self.assertFalse(eye._low_snr)  # pylint: disable=protected-access

    # eye = pam2.PAM2(waveforms)
    # with mock.patch("sys.stdout", new=io.StringIO()) as _:
    #   eye._step1_levels(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    # self.assertGreaterEqual(eye._y_zero, 0 - v_error)  # pylint: disable=protected-access
    # self.assertLessEqual(eye._y_zero, 0 + v_error)  # pylint: disable=protected-access
    # self.assertGreaterEqual(eye._y_ua, v_signal - v_error)  # pylint: disable=protected-access
    # self.assertLessEqual(eye._y_ua, v_signal + v_error)  # pylint: disable=protected-access
    # v_half = v_signal * 0.5
    # v_rising = v_signal * 0.55
    # v_falling = v_signal * 0.45
    # self.assertGreaterEqual(eye._y_half, v_half - v_error)  # pylint: disable=protected-access
    # self.assertLessEqual(eye._y_half, v_half + v_error)  # pylint: disable=protected-access
    # self.assertGreaterEqual(eye._y_rising, v_rising - v_error)  # pylint: disable=protected-access
    # self.assertLessEqual(eye._y_rising, v_rising + v_error)  # pylint: disable=protected-access
    # self.assertGreaterEqual(eye._y_falling, v_falling - v_error)  # pylint: disable=protected-access
    # self.assertLessEqual(eye._y_falling, v_falling + v_error)  # pylint: disable=protected-access
    # self.assertFalse(eye._low_snr)  # pylint: disable=protected-access

    eye = pam2.PAM2(waveforms, y_0=0, y_1=v_signal)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
    self.assertAlmostEqual(eye._y_zero, 0)  # pylint: disable=protected-access
    self.assertAlmostEqual(eye._y_ua, v_signal)  # pylint: disable=protected-access
    v_half = v_signal * 0.5
    v_rising = v_signal * 0.55
    v_falling = v_signal * 0.45
    self.assertAlmostEqual(eye._y_half, v_half)  # pylint: disable=protected-access
    self.assertAlmostEqual(eye._y_rising, v_rising)  # pylint: disable=protected-access
    self.assertAlmostEqual(eye._y_falling, v_falling)  # pylint: disable=protected-access

    hysteresis = 0.5
    y_new = y.copy()
    y_new[:-1] += y_new[1:] * 0.1
    waveforms = np.array([[t, y], [t, y_new]])
    eye = pam2.PAM2(waveforms, hysteresis=hysteresis)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=True, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    self.assertIn("Completed PAM2 levels", fake_stdout.getvalue())
    v_half = v_signal * 0.5
    v_rising = v_half + hysteresis / 2
    v_falling = v_half - hysteresis / 2
    self.assertGreaterEqual(eye._y_rising, v_rising - v_error)  # pylint: disable=protected-access
    self.assertLessEqual(eye._y_rising, v_rising + v_error)  # pylint: disable=protected-access
    self.assertGreaterEqual(eye._y_falling, v_falling - v_error)  # pylint: disable=protected-access
    self.assertLessEqual(eye._y_falling, v_falling + v_error)  # pylint: disable=protected-access
    self.assertFalse(eye._low_snr)  # pylint: disable=protected-access


class TestEyeDiagramMeasuresPAM2(unittest.TestCase):
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
    n_sym = np.random.randint(1000, 10000)
    n_sym_bad = np.random.randint(0, n_sym)
    mask_margin = np.random.uniform(-1, 1)
    transition_dist = {
        "000": np.random.randint(0, n_sym),
        "001": np.random.randint(0, n_sym),
        "010": np.random.randint(0, n_sym),
        "011": np.random.randint(0, n_sym),
        "100": np.random.randint(0, n_sym),
        "101": np.random.randint(0, n_sym),
        "110": np.random.randint(0, n_sym),
        "111": np.random.randint(0, n_sym),
    }

    resolution = 200
    shape = (resolution, resolution, 4)

    np_clean = np.random.uniform(0.0, 1.0, size=shape)
    np_grid = np.random.uniform(0.0, 1.0, size=shape)
    np_mask = np.random.uniform(0.0, 1.0, size=shape)
    np_hits = np.random.uniform(0.0, 1.0, size=shape)
    np_margin = np.random.uniform(0.0, 1.0, size=shape)

    y_0 = np.random.uniform(-1, 1)
    y_1 = np.random.uniform(-1, 1)
    y_cross = np.random.uniform(-1, 1)
    y_cross_r = np.random.uniform(-1, 1)

    amp = np.random.uniform(-1, 1)
    height = np.random.uniform(-1, 1)
    height_r = np.random.uniform(-1, 1)
    snr = np.random.uniform(-1, 1)

    t_sym = np.random.uniform(-1, 1)
    t_0 = np.random.uniform(-1, 1)
    t_1 = np.random.uniform(-1, 1)
    t_rise = np.random.uniform(-1, 1)
    t_fall = np.random.uniform(-1, 1)

    f_sym = np.random.uniform(-1, 1)

    width = np.random.uniform(-1, 1)
    width_r = np.random.uniform(-1, 1)
    dcd = np.random.uniform(-1, 1)
    jitter_pp = np.random.uniform(-1, 1)
    jitter_rms = np.random.uniform(-1, 1)

    extinction_ratio = np.random.uniform(-1, 1)
    oma_cross = np.random.uniform(-1, 1)
    vecp = np.random.uniform(-1, 1)

    m = pam2.MeasuresPAM2(np_clean, np_grid, np_mask, np_hits, np_margin)
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
