"""Test module hardware_tools.measurement.eyediagram
"""

from __future__ import annotations

import io
from unittest import mock

import numpy as np
from scipy import signal

from hardware_tools import math
from hardware_tools.measurement.eyediagram import eyediagram

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
clock = (signal.square(2 * np.pi * (f_bit * t + 0.5)) + 1) / 2 * v_signal
y = (np.repeat(bits, n_scope / n_bits) +
     _rng.normal(0, 0.05, size=n_scope)) * v_signal


class Derrived(eyediagram.EyeDiagram):
  """Derrived EyeDiagram to test abstract class without errors
  """

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
    self._t_sym = math.UncertainValue(t_bit, 0)
    self._clock_edges = []
    for i in range(self._waveforms.shape[0]):
      t_start = self._waveforms[i][0][0] - 1 / f_scope / 2 - t_bit / 2
      e = np.arange(t_start, self._waveforms[i][0][-1], t_bit)
      self._clock_edges.append(e.tolist())
    self._clock_edges[-1] = self._clock_edges[-1][-5:]

  def _step4_measure(self,
                     n_threads: int = 1,
                     print_progress: bool = True,
                     indent: int = 0,
                     debug_plots: str = None) -> None:
    self._measures = eyediagram.Measures()


class TestEyeDiagram(base.TestBase):
  """Test EyeDiagram
  """

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

  def test_step3(self):
    path = str(self._TEST_ROOT.joinpath("eyediagram_step3"))
    waveforms = np.array([[t, y], [t, y]])
    clocks = np.array([[t, clock], [t, clock]])

    eye = Derrived(waveforms, clocks=clocks, resolution=1000)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step3_sample(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access

    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step3_sample(print_progress=True, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    self.assertIn("Completed sampling", fake_stdout.getvalue())

  def test_step5(self):
    path = str(self._TEST_ROOT.joinpath("eyediagram_step5"))
    waveforms = np.array([[t, y], [t, y]])
    clocks = np.array([[t, clock], [t, clock]])

    eye = Derrived(waveforms, clocks=clocks, resolution=200)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step3_sample(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step4_measure(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step5_stack(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access

    eye = Derrived(waveforms, clocks=clocks, resolution=200)
    with mock.patch("sys.stdout", new=io.StringIO()) as _:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step3_sample(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step4_measure(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step5_stack(print_progress=False, n_threads=2, debug_plots=None)  # pylint: disable=protected-access

    eye = Derrived(waveforms, clocks=clocks, resolution=200)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step3_sample(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step4_measure(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step5_stack(print_progress=True, n_threads=2, debug_plots=None)  # pylint: disable=protected-access
    self.assertIn("Ran waveform #0", fake_stdout.getvalue())


class TestEyeDiagramMeasures(base.TestBase):
  """Test Measures
  """

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

    m = eyediagram.Measures()
    m.set_images(np_clean, np_grid, np_mask, np_hits, np_margin)
    m.n_sym = n_sym
    m.n_sym_bad = n_sym_bad
    m.transition_dist = transition_dist
    m.mask_margin = mask_margin

    d = {
        "n_sym": n_sym,
        "n_sym_bad": n_sym_bad,
        "mask_margin": mask_margin,
        "transition_dist": transition_dist,
        "image_clean": math.Image.np_to_base64(np_clean),
        "image_grid": math.Image.np_to_base64(np_grid),
        "image_mask": math.Image.np_to_base64(np_mask),
        "image_hits": math.Image.np_to_base64(np_hits),
        "image_margin": math.Image.np_to_base64(np_margin)
    }
    self.assertDictEqual(d, m.to_dict())

  def test_save_images(self):
    n_sym = self._RNG.integers(1000, 10000)
    n_sym_bad = self._RNG.integers(0, n_sym)
    mask_margin = self._RNG.uniform(-1, 1)

    resolution = 200
    shape = (resolution, resolution, 4)

    np_clean = self._RNG.uniform(0.0, 1.0, size=shape)
    np_grid = self._RNG.uniform(0.0, 1.0, size=shape)
    np_mask = self._RNG.uniform(0.0, 1.0, size=shape)
    np_hits = self._RNG.uniform(0.0, 1.0, size=shape)
    np_margin = self._RNG.uniform(0.0, 1.0, size=shape)

    m = eyediagram.Measures()
    m.set_images(np_clean, np_grid, np_mask, np_hits, np_margin)
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
