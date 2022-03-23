"""Test module hardware_tools.measurement.eyediagram
"""

from __future__ import annotations

import io
import os
from unittest import mock

import numpy as np
from scipy import signal

from hardware_tools.math import image
from hardware_tools.measurement import mask
from hardware_tools.measurement.eyediagram import eyediagram

from tests import base

from tests.measurement.eyediagram.addition_runner import runner_addition

_rng = np.random.default_rng()
f_scope = 5e9
n_scope = int(1e5)
t_scope = n_scope / f_scope
n_bits = int(10e3)
t_bit = t_scope / n_bits
f_bit = 1 / t_bit
bits = signal.max_len_seq(5, length=n_bits)[0]
t = np.linspace(0, (n_scope - 1) / f_scope, n_scope)
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
    self._low_snr = False

  def _step2_clock(self,
                   n_threads: int = 1,
                   print_progress: bool = True,
                   indent: int = 0,
                   debug_plots: str = None) -> None:
    if not self._low_snr:
      self._clock_edges = []
      self._ties = []
      for i in range(self._waveforms.shape[0]):
        t_start = self._waveforms[i][0][0] - 1 / f_scope / 2 - t_bit / 2
        e = np.arange(t_start, self._waveforms[i][0][-1], t_bit)
        self._clock_edges.append(e.tolist())
        self._ties.append(_rng.normal(0, 0.1 * t_bit, size=n_bits).tolist())
    super()._step2_clock(n_threads=n_threads,
                         print_progress=print_progress,
                         indent=indent,
                         debug_plots=debug_plots)

  def _step4_measure(self,
                     n_threads: int = 1,
                     print_progress: bool = True,
                     indent: int = 0,
                     debug_plots: str = None) -> None:
    m = eyediagram.Measures()
    if self._mask is not None:
      results = eyediagram._runner_sample_mask(  # pylint: disable=protected-access
          self._waveforms[0][1], self._centers_t[0], self._centers_i[0],
          self._t_delta, self._t_sym, self._y_zero, self._y_ua, self._mask)
      m.mask_margin = results["margin"]
      self._offenders = [results["offenders"]]
      self._hits = results["hits"]
    m.bathtub_curves = self._generate_bathtub_curves(
        {"Eye 0": 0.5},
        n_threads=n_threads,
        print_progress=print_progress,
        indent=indent)
    self._measures = m

  def _draw_grid(self, image_grid: np.ndarray) -> None:
    super()._draw_grid(image_grid)
    image_grid[self._uia_to_image(-10), ::10, 3] = 1.0


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

    waveforms = np.array([[t, y], [t, y]])
    clock_edges = [[], []]
    Derrived(waveforms, clock_edges=clock_edges)

    clock_edges = np.array([[], []])
    self.assertRaises(ValueError, Derrived, waveforms, clock_edges=clock_edges)
    clock_edges = [None, None]
    self.assertRaises(ValueError, Derrived, waveforms, clock_edges=clock_edges)
    clock_edges = [[]]
    self.assertRaises(ValueError, Derrived, waveforms, clock_edges=clock_edges)

    self.assertRaises(KeyError,
                      eyediagram.Config,
                      this_keyword_does_not_exist=None)

  def test_get_raw_heatmap(self):
    waveforms = np.array([t, y])
    clocks = np.array([t, clock])
    eye = Derrived(waveforms, clocks=clocks, resolution=200)

    self.assertRaises(RuntimeError, eye.get_raw_heatmap)

    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye.calculate(print_progress=True, n_threads=0)
    fake_stdout = fake_stdout.getvalue()
    self.assertIn("Starting eye", fake_stdout)
    self.assertIn("Step 1", fake_stdout)
    self.assertIn("Step 2", fake_stdout)
    self.assertIn("Step 3", fake_stdout)
    self.assertIn("Step 4", fake_stdout)
    self.assertIn("Step 5", fake_stdout)
    self.assertIn("Completed eye", fake_stdout)

    heatmap = eye.get_raw_heatmap()
    self.assertIsInstance(heatmap, np.ndarray)

    heatmap = eye.get_raw_heatmap(True)
    self.assertIsInstance(heatmap, bytes)

  def test_get_measures(self):
    waveforms = np.array([t, y])
    clocks = np.array([t, clock])
    eye = Derrived(waveforms, clocks=clocks, resolution=200)

    self.assertRaises(RuntimeError, eye.get_measures)

    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye.calculate(print_progress=False, n_threads=1)
    self.assertEqual("", fake_stdout.getvalue())

    m = eye.get_measures()
    self.assertIsInstance(m, eyediagram.Measures)

  def test_collect_runners(self):

    args_list = [[1, 2], [10, 20]]

    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      output = eyediagram.EyeDiagram._collect_runners(  # pylint: disable=protected-access
          runner_addition,
          args_list,
          n_threads=1,
          print_progress=True)
    self.assertIn("Ran waveform #1", fake_stdout.getvalue())
    self.assertEqual(output[0], args_list[0][0] + args_list[0][1])
    self.assertEqual(output[1], args_list[1][0] + args_list[1][1])

    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      output = eyediagram.EyeDiagram._collect_runners(  # pylint: disable=protected-access
          runner_addition,
          args_list,
          n_threads=1,
          print_progress=False)
    self.assertEqual("", fake_stdout.getvalue())
    self.assertEqual(output[0], args_list[0][0] + args_list[0][1])
    self.assertEqual(output[1], args_list[1][0] + args_list[1][1])

    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      output = eyediagram.EyeDiagram._collect_runners(  # pylint: disable=protected-access
          runner_addition,
          args_list,
          n_threads=2,
          print_progress=True)
    self.assertIn("Ran waveform #1", fake_stdout.getvalue())
    self.assertEqual(output[0], args_list[0][0] + args_list[0][1])
    self.assertEqual(output[1], args_list[1][0] + args_list[1][1])

    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      output = eyediagram.EyeDiagram._collect_runners(  # pylint: disable=protected-access
          runner_addition,
          args_list,
          n_threads=2,
          print_progress=False)
    self.assertEqual("", fake_stdout.getvalue())
    self.assertEqual(output[0], args_list[0][0] + args_list[0][1])
    self.assertEqual(output[1], args_list[1][0] + args_list[1][1])

  def test_step2(self):
    path = str(self._TEST_ROOT.joinpath("eyediagram_step2"))
    waveforms = np.array([[t, y]])

    eye = Derrived(waveforms, resolution=1000)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    self.assertIn("Saved image to", fake_stdout.getvalue())
    self.assertTrue(os.path.exists(path + ".step2.png"))

    eye = Derrived(waveforms, resolution=1000)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._low_snr = True  # pylint: disable=protected-access
      eye._step2_clock(print_progress=True, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    self.assertIn("Calculating symbol period", fake_stdout.getvalue())
    self.assertIn("Saved image to", fake_stdout.getvalue())
    self.assertTrue(os.path.exists(path + ".step2.png"))

  def test_step3(self):
    path = str(self._TEST_ROOT.joinpath("eyediagram_step3"))
    waveforms = np.array([[t, y]])
    clocks = np.array([[t, clock]])

    eye = Derrived(waveforms, clocks=clocks, resolution=1000)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step3_sample(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
    self.assertIn("Saved image to", fake_stdout.getvalue())
    self.assertTrue(os.path.exists(path + ".step3.png"))

  def test_step5(self):
    path = str(self._TEST_ROOT.joinpath("eyediagram_step5"))
    waveforms = np.array([[t, y]])
    clocks = np.array([[t, clock]])

    m = mask.MaskDecagon(0.18, 0.29, 0.35, 0.35, 0.38, 0.4, 0.5)
    eye = Derrived(waveforms, clocks=clocks, resolution=1000, mask=m)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step3_sample(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step4_measure(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step5_stack(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
      eye._calculated = True  # pylint: disable=protected-access
    self.assertIn("Saved image to", fake_stdout.getvalue())
    self.assertTrue(os.path.exists(path + ".step5.png"))

    m = eye.get_measures().to_dict()
    self.assertIsNotNone(m["image_clean"])
    self.assertIsNotNone(m["image_grid"])
    self.assertIsNotNone(m["image_mask"])
    self.assertIsNotNone(m["image_hits"])
    self.assertIsNotNone(m["image_margin"])

    eye.get_measures().save_images(path)

    path = str(self._TEST_ROOT.joinpath("eyediagram_step5_points"))
    c = eyediagram.Config(point_cloud=True,)
    m = mask.MaskDecagon(0.01, 0.29, 0.35, 0.35, 0.38, 0.4, 0.5)

    f_lp = 0.75 / t_bit
    sos = signal.bessel(4, f_lp, fs=f_scope, output="sos", norm="mag")
    zi = signal.sosfilt_zi(sos) * y[0]
    y_filtered, _ = signal.sosfilt(sos, y, zi=zi)
    waveforms = np.array([t, y_filtered])
    eye = Derrived(waveforms,
                   clock_edges=eye.get_clock_edges(),
                   resolution=1000,
                   mask=m,
                   config=c)
    self.assertRaises(RuntimeError, eye.get_clock_edges)
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
      eye._step1_levels(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step2_clock(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step3_sample(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step4_measure(print_progress=False, n_threads=1, debug_plots=None)  # pylint: disable=protected-access
      eye._step5_stack(print_progress=False, n_threads=1, debug_plots=path)  # pylint: disable=protected-access
      eye._calculated = True  # pylint: disable=protected-access
    self.assertIn("Saved image to", fake_stdout.getvalue())
    self.assertTrue(os.path.exists(path + ".step5.png"))

    m = eye.get_measures().to_dict()
    self.assertIsNotNone(m["image_clean"])
    self.assertIsNotNone(m["image_grid"])
    self.assertIsNotNone(m["image_mask"])
    self.assertIsNotNone(m["image_hits"])
    self.assertIsNotNone(m["image_margin"])

    eye.get_measures().save_images(path)


class TestEyeDiagramMeasures(base.TestBase):
  """Test Measures
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

    m = eyediagram.Measures()
    m.n_samples = n_samples
    m.n_sym = n_sym
    m.n_sym_bad = n_sym_bad
    m.transition_dist = transition_dist
    m.mask_margin = mask_margin
    m.bathtub_curves = bathtub_curves

    d = {
        "n_samples": n_samples,
        "n_sym": n_sym,
        "n_sym_bad": n_sym_bad,
        "mask_margin": mask_margin,
        "transition_dist": transition_dist,
        "bathtub_curves": bathtub_curves,
        "image_clean": None,
        "image_grid": None,
        "image_mask": None,
        "image_hits": None,
        "image_margin": None
    }
    m_dict = m.to_dict()
    self.assertListEqual(sorted(d.keys()), sorted(m_dict.keys()))
    self.assertDictEqual(d, m_dict)

    m.set_images(np_clean, np_grid, np_mask, np_hits, np_margin)

    d = {
        "n_samples": n_samples,
        "n_sym": n_sym,
        "n_sym_bad": n_sym_bad,
        "mask_margin": mask_margin,
        "transition_dist": transition_dist,
        "bathtub_curves": bathtub_curves,
        "image_clean": image.np_to_base64(np_clean),
        "image_grid": image.np_to_base64(np_grid),
        "image_mask": image.np_to_base64(np_mask),
        "image_hits": image.np_to_base64(np_hits),
        "image_margin": image.np_to_base64(np_margin)
    }
    m_dict = m.to_dict()
    self.assertListEqual(sorted(d.keys()), sorted(m_dict.keys()))
    self.assertDictEqual(d, m_dict)

  def test_save_images(self):
    n_sym = self._RNG.integers(1000, 10000)
    n_sym_bad = self._RNG.integers(0, n_sym)
    n_samples = n_sym * self._RNG.integers(5, 50)
    mask_margin = self._RNG.uniform(-1, 1)

    resolution = 10
    shape = (resolution, resolution, 4)

    np_clean = self._RNG.uniform(0.0, 1.0, size=shape)
    np_grid = self._RNG.uniform(0.0, 1.0, size=shape)
    np_mask = self._RNG.uniform(0.0, 1.0, size=shape)
    np_hits = self._RNG.uniform(0.0, 1.0, size=shape)
    np_margin = self._RNG.uniform(0.0, 1.0, size=shape)

    m = eyediagram.Measures()
    m.set_images(np_clean, np_grid, np_mask, np_hits, np_margin)
    m.n_samples = n_samples
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
