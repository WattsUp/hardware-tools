"""Test module hardware_tools.measurement.eyediagram
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
from hardware_tools.measurement.eyediagram import eyediagram

f_scope = 5e9
n_scope = int(1e5)
t_scope = n_scope / f_scope
n_bits = int(10e3)
t_bit = t_scope / n_bits
f_bit = 1 / t_bit
bits = signal.max_len_seq(5, length=n_bits)[0]
t = np.linspace(0, t_scope, n_scope)
v_signal = 3.3
clock = (signal.square(2 * np.pi * f_bit * t) + 1) / 2 * v_signal
y = (np.repeat(bits, n_scope / n_bits) +
     np.random.normal(0, 0.05, size=n_scope)) * v_signal


class Derrived(eyediagram.EyeDiagram):

  def _step1_levels(self,
                    n_threads: int = 1,
                    print_progress: bool = True,
                    indent: int = 0,
                    debug_plots: str = None) -> None:
    self._y_zero = 0
    self._y_ua = v_signal

  def _step2_clock(self,
                   n_threads: int = 1,
                   print_progress: bool = True,
                   indent: int = 0,
                   debug_plots: str = None) -> None:
    self._t_sym = t_bit

  def _step3_sample(self,
                    n_threads: int = 1,
                    print_progress: bool = True,
                    indent: int = 0,
                    debug_plots: str = None) -> None:
    i_sym_width = int(n_scope / n_bits)
    center_i = list(range(i_sym_width // 2, n_scope, i_sym_width))
    t_off = ((i_sym_width + 1) % 2) * (1 / f_scope) / 2
    self._centers_i = []
    self._centers_t = []
    for _ in range(self._waveforms.shape[0]):
      self._centers_i.append(center_i)
      self._centers_t.append(
          np.random.normal(t_off, 1 / f_scope / 10, n_bits).tolist())

  def _step5_measure(self,
                     n_threads: int = 1,
                     print_progress: bool = True,
                     indent: int = 0,
                     debug_plots: str = None) -> None:
    self._measures = eyediagram.Measures(None, None, None, None, None)


class TestEyeDiagram(unittest.TestCase):
  """Test EyeDiagram
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
    Derrived(waveforms)
    Derrived(waveforms, clocks=clocks)

    waveforms = np.array([[t, y], [t, y]])
    clocks = np.array([[t, clock], [t, clock]])
    Derrived(waveforms, clocks=clocks)

    self.assertRaises(ValueError, Derrived, y)

    waveforms = np.array([y])
    self.assertRaises(ValueError, Derrived, waveforms)

    waveforms = np.array([[t, y]]).T
    self.assertRaises(ValueError, Derrived, waveforms)

    waveforms = np.array([[t, y], [t, y]])
    clocks = np.array([t, clock])
    self.assertRaises(ValueError, Derrived, waveforms, clocks=clocks)
    self.assertRaises(ValueError, Derrived, waveforms, clocks=clock)

  def test_get_raw_heatmap(self):
    waveforms = np.array([t, y])
    clocks = np.array([t, clock])
    eye = Derrived(waveforms, clocks=clocks, resolution=200)

    self.assertRaises(RuntimeError, eye.get_raw_heatmap)

    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye.calculate(print_progress=True, n_threads=0)

    heatmap = eye.get_raw_heatmap()
    self.assertIsInstance(heatmap, np.ndarray)

    heatmap = eye.get_raw_heatmap(True)
    self.assertIsInstance(heatmap, bytes)

  def test_get_measures(self):
    waveforms = np.array([t, y])
    clocks = np.array([t, clock])
    eye = Derrived(waveforms, clocks=clocks, resolution=200)

    self.assertRaises(RuntimeError, eye.get_measures)

    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye.calculate(print_progress=False, n_threads=1)

    m = eye.get_measures()
    self.assertIsInstance(m, eyediagram.Measures)

  def test_stack(self):
    waveforms = np.array([[t, y], [t, y]])
    clocks = np.array([[t, clock], [t, clock]])

    eye = Derrived(waveforms, clocks=clocks, resolution=1000, resample=0)
    path = str(self._TEST_ROOT.joinpath("eyediagram"))
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye.calculate(print_progress=False, n_threads=0, debug_plots=path)

    eye = Derrived(waveforms, clocks=clocks, resolution=1000, resample=0)
    path = str(self._TEST_ROOT.joinpath("eyediagram"))
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye.calculate(print_progress=False, n_threads=1, debug_plots=path)

    eye = Derrived(waveforms, clocks=clocks, resolution=1000, resample=50)
    path = str(self._TEST_ROOT.joinpath("eyediagram_sinc"))
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye.calculate(print_progress=True, n_threads=1, debug_plots=path)
    self.assertIn("Stacked", fake_stdout.getvalue())


class TestEyeDiagramMeasures(unittest.TestCase):
  """Test Measures
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

    resolution = 200
    shape = (resolution, resolution, 4)

    np_clean = np.random.uniform(0.0, 1.0, size=shape)
    np_grid = np.random.uniform(0.0, 1.0, size=shape)
    np_mask = np.random.uniform(0.0, 1.0, size=shape)
    np_hits = np.random.uniform(0.0, 1.0, size=shape)
    np_margin = np.random.uniform(0.0, 1.0, size=shape)

    m = eyediagram.Measures(np_clean, np_grid, np_mask, np_hits, np_margin)
    m.n_sym = n_sym
    m.n_sym_bad = n_sym_bad
    m.mask_margin = mask_margin

    d = {
        "n_sym": n_sym,
        "n_sym_bad": n_sym_bad,
        "mask_margin": mask_margin,
        "image_clean": math.Image.np_to_base64(np_clean),
        "image_grid": math.Image.np_to_base64(np_grid),
        "image_mask": math.Image.np_to_base64(np_mask),
        "image_hits": math.Image.np_to_base64(np_hits),
        "image_margin": math.Image.np_to_base64(np_margin)
    }
    self.assertDictEqual(d, m.to_dict())

  def test_save_images(self):
    n_sym = np.random.randint(1000, 10000)
    n_sym_bad = np.random.randint(0, n_sym)
    mask_margin = np.random.uniform(-1, 1)

    resolution = 200
    shape = (resolution, resolution, 4)

    np_clean = np.random.uniform(0.0, 1.0, size=shape)
    np_grid = np.random.uniform(0.0, 1.0, size=shape)
    np_mask = np.random.uniform(0.0, 1.0, size=shape)
    np_hits = np.random.uniform(0.0, 1.0, size=shape)
    np_margin = np.random.uniform(0.0, 1.0, size=shape)

    m = eyediagram.Measures(np_clean, np_grid, np_mask, np_hits, np_margin)
    m.n_sym = n_sym
    m.n_sym_bad = n_sym_bad
    m.mask_margin = mask_margin

    basename = str(self._TEST_ROOT.joinpath("eyediagram"))
    self._TEST_ROOT.mkdir(parents=True, exist_ok=True)
    m.save_images(basename)

    path = self._TEST_ROOT.joinpath("eyediagram.clean.png")
    self.assertTrue(path.exists())

    path = self._TEST_ROOT.joinpath("eyediagram.grid.png")
    self.assertTrue(path.exists())

    path = self._TEST_ROOT.joinpath("eyediagram.mask.png")
    self.assertTrue(path.exists())

    path = self._TEST_ROOT.joinpath("eyediagram.hits.png")
    self.assertTrue(path.exists())

    path = self._TEST_ROOT.joinpath("eyediagram.margin.png")
    self.assertTrue(path.exists())
