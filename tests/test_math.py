"""Test module hardware_tools.math
"""

import base64
import io
import os
import pathlib
import unittest

import numpy as np
import PIL
import PIL.Image

from hardware_tools import math


class TestMath(unittest.TestCase):
  """Test math methods
  """

  _TEST_ROOT = pathlib.Path(".test")

  def __clean_test_root(self):
    if self._TEST_ROOT.exists():
      for f in os.listdir(self._TEST_ROOT):
        os.remove(self._TEST_ROOT.joinpath(f))
      os.rmdir(self._TEST_ROOT)

  def test_interpolate_sinc(self):
    x_min = np.random.randint(-1000, 500)
    x_max = x_min + np.random.randint(1, 500)
    n_in = np.random.randint(500, 1000)
    n_down = np.random.randint(10, n_in - 10)
    n_up = np.random.randint(1000, 5000)

    x_in = np.linspace(x_min, x_max, n_in)
    y_in = np.random.uniform(-1, 1, n_in)
    self.assertEqual(x_in.shape, y_in.shape)

    x_down = np.linspace(x_min, x_max, n_down)
    y_down = math.interpolate_sinc(x_down, x_in, y_in)

    self.assertEqual(y_down.shape, x_down.shape)

    x_up = np.linspace(x_min, x_max, n_up)
    y_up = math.interpolate_sinc(x_up, x_in, y_in)

    self.assertEqual(y_up.shape, x_up.shape)

    # Sinc interpolation should have a rectangle frequency output
    # Compare relative power in the max input frequency band
    fft = np.abs(np.power(np.fft.rfft(y_up), 2))
    total_power = fft.sum()
    fft = fft / total_power
    f_in = n_in // 2 + 1

    input_band_power = fft[:f_in].sum()

    try:
      self.assertGreaterEqual(input_band_power, 0.99)
    except AssertionError as e:
      # TODO remove
      import matplotlib.pyplot as pyplot
      pyplot.plot(fft)
      pyplot.axvline(x=n_in // 2)
      pyplot.axhline(y=0)
      pyplot.show()
      raise e

    # 2D input
    x_in = np.random.uniform(x_min, x_max, (n_in, n_in))
    self.assertRaises(ValueError, math.interpolate_sinc, x_up, x_in, y_in)

    # Mismatched input
    x_in = np.linspace(x_min, x_max, n_in + 1)
    self.assertRaises(ValueError, math.interpolate_sinc, x_up, x_in, y_in)

    # Single point input
    x_in = np.linspace(x_min, x_max, 1)
    y_in = np.random.uniform(-1, 1, 1)
    self.assertRaises(ValueError, math.interpolate_sinc, x_up, x_in, y_in)

  def test_point2d(self):
    x0 = np.random.uniform(-1000, 1000)
    y0 = np.random.uniform(-1000, 1000)
    x1 = np.random.uniform(-1000, 1000)
    y1 = np.random.uniform(-1000, 1000)

    if x0 == x1 and y0 == y1:
      x0 = x1 + 1

    p0 = math.Point2D(x0, y0)
    p1 = math.Point2D(x1, y1)
    self.assertEqual(p0.x, x0)
    self.assertEqual(p0.y, y0)
    self.assertEqual(str(p0), f"({x0}, {y0})")

    p_center = math.Point2D((x0 + x1) / 2, (y0 + y1) / 2)
    self.assertTrue(p_center.in_rect(p0, p1))
    self.assertTrue(p_center.in_rect(p1, p0))
    self.assertTrue(p0.in_rect(p0, p1))
    self.assertTrue(p1.in_rect(p0, p1))
    self.assertFalse(p1.in_rect(p0, p_center))
    self.assertFalse(p1.in_rect(p_center, p0))
    self.assertFalse(p0.in_rect(p1, p_center))
    self.assertFalse(p0.in_rect(p_center, p1))

    self.assertEqual(0, math.Point2D.orientation(p0, p_center, p1))
    self.assertEqual(0, math.Point2D.orientation(p0, p1, p_center))
    self.assertEqual(0, math.Point2D.orientation(p_center, p0, p1))

    p_knee = math.Point2D(p_center.x, 2000)
    if x0 < x1:
      self.assertEqual(1, math.Point2D.orientation(p0, p_knee, p1))
      self.assertEqual(-1, math.Point2D.orientation(p1, p_knee, p0))
    elif x0 > x1:
      self.assertEqual(-1, math.Point2D.orientation(p0, p_knee, p1))
      self.assertEqual(1, math.Point2D.orientation(p1, p_knee, p0))
    else:
      p_knee = math.Point2D(2000, p_center.y)
      if y0 < y1:
        self.assertEqual(-1, math.Point2D.orientation(p0, p_knee, p1))
        self.assertEqual(1, math.Point2D.orientation(p1, p_knee, p0))
      else:
        self.assertEqual(1, math.Point2D.orientation(p0, p_knee, p1))
        self.assertEqual(-1, math.Point2D.orientation(p1, p_knee, p0))

  def test_line2d(self):
    x0 = np.random.uniform(-1000, 1000)
    y0 = np.random.uniform(-1000, 1000)
    x1 = np.random.uniform(-1000, 1000)
    y1 = np.random.uniform(-1000, 1000)

    if x0 == x1 and y0 == y1:
      x0 = x1 + 1

    l0 = math.Line2D(x0, y0, x1, y1)
    self.assertEqual(l0.p.x, x0)
    self.assertEqual(l0.p.y, y0)
    self.assertEqual(l0.q.x, x1)
    self.assertEqual(l0.q.y, y1)
    self.assertEqual(str(l0), f"(({x0}, {y0}), ({x1}, {y1}))")

    l1 = math.Line2D(x0, y1, x1, y0)
    self.assertTrue(l0.intersecting(l1))
    l1 = math.Line2D(x0 + 1, y0 + 1, x1 + 1, y1 + 1)
    self.assertFalse(l0.intersecting(l1))

    p0 = math.Point2D((x0 + x1) / 2, (y0 + y1) / 2)
    p1 = math.Point2D(x0 + (x1 - x0) * 2, y0 + (y1 - y0) * 2)
    self.assertTrue(l0.intersecting_points(p0, p1))
    self.assertTrue(l0.intersecting_points(p1, p0))

    p0 = math.Point2D(x0 - (x1 - x0) / 2, y0 - (y1 - y0) / 2)
    p1 = math.Point2D(x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2)
    l1 = math.Line2D(x0, y0, x0, y0)
    self.assertTrue(l1.intersecting_points(p0, p1))

    l1 = math.Line2D(x0, y1, x1, y0)
    p_intersect = l0.intersection(l1)
    self.assertIsNotNone(p_intersect)
    self.assertAlmostEqual(p_intersect.x, (x0 + x1) / 2)
    self.assertAlmostEqual(p_intersect.y, (y0 + y1) / 2)

    self.assertIsNone(l0.intersection(l0))
    l1 = math.Line2D(x0 + 1, y0 + 1, x1 + 1, y1 + 1)
    self.assertIsNone(l0.intersection(l1))

  def test_gaussian(self):
    x0 = np.random.uniform(-1000, 1000)
    x1 = np.random.uniform(-1000, 1000)
    amplitude = np.random.uniform(-1000, 1000)
    mean = np.random.uniform(-1000, 1000)
    stddev = np.random.uniform(-1000, 1000)

    g = math.Gaussian(amplitude, mean, stddev)
    self.assertEqual(str(g), f"{{A={amplitude}, mu={mean}, stddev={stddev}}}")
    y0 = amplitude * np.exp(-(
        (x0 - mean) / stddev)**2 / 2) / (np.sqrt(2 * np.pi * stddev**2))
    y1 = amplitude * np.exp(-(
        (x1 - mean) / stddev)**2 / 2) / (np.sqrt(2 * np.pi * stddev**2))
    self.assertAlmostEqual(y0, g.compute(x0))
    self.assertAlmostEqual(y1, g.compute(x1))

    x = np.array([x0, x1])
    y = g.compute(x)
    self.assertAlmostEqual(y0, y[0])
    self.assertAlmostEqual(y1, y[1])

  def test_gaussian_fit(self):
    amplitude = np.random.uniform(-1000, 1000)
    mean = np.random.uniform(-1000, 1000)
    stddev = np.random.uniform(-1000, 1000)
    n = np.random.randint(500, 1000)

    x = np.random.uniform(mean - stddev * 5, mean + stddev * 5, n)
    g = math.Gaussian(amplitude, mean, stddev)
    y = g.compute(x)

    g_fit = math.Gaussian.fit_pdf(x, y)
    self.assertAlmostEqual(amplitude, g_fit.amplitude)
    self.assertAlmostEqual(mean, g_fit.mean)
    self.assertAlmostEqual(abs(stddev), abs(g_fit.stddev))

  def test_gaussian_mix(self):
    n_components = np.random.randint(2, 5)
    amplitude = np.random.uniform(1, 1000, n_components)
    mean = np.random.uniform(-1000, 1000, n_components)
    stddev = np.random.uniform(1, 1000, n_components)
    n = np.random.randint(500, 1000)

    components = [
        math.Gaussian(a, m, s) for a, m, s in zip(amplitude, mean, stddev)
    ]
    g = math.GaussianMix(components[:1])
    self.assertEqual(mean[0], g.center())

    g = math.GaussianMix(components)
    self.assertEqual(str(g), str(components))
    x = np.random.uniform(
        min(mean) - max(abs(stddev)) * 5,
        max(mean) + max(abs(stddev)) * 5, n)
    y = np.zeros(n)
    for c in components:
      y += c.compute(x)

    y_compute = g.compute(x)
    for i in range(n):
      self.assertAlmostEqual(y[i], y_compute[i])

    components = components[:1]
    components.append(math.Gaussian(0, 0, 1))
    g = math.GaussianMix(components)
    center_compute = g.center()
    error = center_compute / components[0].mean - 1
    self.assertAlmostEqual(0, error, places=3)

  def test_gaussian_mix_fit(self):
    n_components = np.random.randint(2, 5)
    amplitude = np.random.uniform(1, 5, n_components)
    mean = np.random.uniform(-1000, 1000, n_components)
    stddev = np.random.uniform(1, 100, n_components)
    n = np.random.randint(5000, 10000)

    amplitude = amplitude / amplitude.sum()

    components = [
        math.Gaussian(a, m, s) for a, m, s in zip(amplitude, mean, stddev)
    ]
    components = sorted(components, key=lambda c: -c.amplitude)

    y = np.array([amplitude[0]] * n)
    g_fit = math.GaussianMix.fit_samples(y, n_max=n_components)
    self.assertEqual(g_fit, [[1, amplitude[0], 0]])

    g = math.GaussianMix(components)
    y = []
    for c in components:
      y.extend(
          np.random.normal(c.mean, c.stddev, int(n * c.amplitude)).tolist())
    y = np.array(y)
    x = np.linspace(min(y), max(y), 1000)
    g_fit = math.GaussianMix.fit_samples(y, n_max=n_components)

    pdf = g.compute(x)
    pdf_fit = g_fit.compute(x)
    errors = (pdf_fit - pdf) / max(pdf)

    error = np.sqrt(np.mean(errors**2))

    self.assertLess(error, 0.15)

  def test_bin_linear(self):
    bin_count = np.random.randint(5, 100)
    y_min = np.random.uniform(-10, 0)
    y_max = np.random.uniform(1, 10)
    edges = np.linspace(y_min, y_max, bin_count + 1)
    counts = np.random.randint(100, 1000, bin_count)
    y = []
    for i in range(bin_count):
      y.extend(
          np.random.uniform(edges[i], edges[i + 1], counts[i] - 2).tolist())
      y.append(edges[i])
      y.append(edges[i + 1])
    y = np.array(y)

    counts_fit, edges_fit = math.Bin.linear(y,
                                            bin_count=bin_count,
                                            density=True)

    for e, e_fit in zip(edges, edges_fit):
      self.assertAlmostEqual(e, e_fit)

    counts_norm = counts / np.diff(edges) / counts.sum()

    for c, c_fit in zip(counts_norm, counts_fit):
      self.assertAlmostEqual(c, c_fit, 2)

    y = np.repeat(y[0], counts.sum())
    counts_fit, edges_fit = math.Bin.linear(y,
                                            bin_count=bin_count,
                                            density=False)
    self.assertEqual(counts_fit.sum(), counts.sum())

    y = np.repeat(0, counts.sum())
    counts_fit, edges_fit = math.Bin.linear(y,
                                            bin_count=bin_count,
                                            density=False)
    self.assertEqual(counts_fit.sum(), counts.sum())

  def test_bin_exact(self):
    n = np.random.randint(100, 1000)
    bins = list(range(-10, 10))
    y = np.random.randint(-10, 10, n)
    for i in range(len(bins)):
      y[i] = bins[i]

    counts_fit, bins_fit = math.Bin.exact(y)
    self.assertEqual(n, sum(counts_fit))
    self.assertListEqual(bins, bins_fit)

    counts_fit, bins_fit = math.Bin.exact_np(y)
    self.assertIsInstance(counts_fit, np.ndarray)
    self.assertIsInstance(bins_fit, np.ndarray)

  def test_bin_exponential(self):
    bins = list(range(-10, 10))
    y = []
    counts = np.random.randint(100, 1000, len(bins))
    for i in range(len(bins)):
      y.extend(10**(bins[i] + np.random.uniform(0, 1, counts[i] - 1)))
      y.append(10**bins[i])
    y[-1] = 10**(bins[-1] + 1)
    y = np.array(y)
    n = sum(counts)

    counts_fit, edges_fit = math.Bin.exponential(y,
                                                 bin_count=len(counts),
                                                 density=False)
    self.assertEqual(n, sum(counts_fit))
    self.assertListEqual(bins, edges_fit.tolist()[:-1])
    self.assertListEqual(counts.tolist(), counts_fit.tolist())

    y[0] = 0
    counts_fit, edges_fit = math.Bin.exponential(y,
                                                 bin_count=len(counts),
                                                 density=False)
    self.assertEqual(n, sum(counts_fit))
    self.assertTrue(np.isneginf(edges_fit[0]))
    self.assertListEqual(bins[1:], edges_fit.tolist()[1:-1])
    self.assertListEqual(counts.tolist(), counts_fit.tolist())

    y = np.zeros(n)
    counts_fit, edges_fit = math.Bin.exponential(y,
                                                 bin_count=len(counts),
                                                 density=False)
    self.assertEqual(n, sum(counts_fit))
    self.assertTrue(np.isneginf(edges_fit[0]))
    self.assertEqual(n, counts_fit[0])
    self.assertEqual(0, edges_fit[-1])

    counts_fit, edges_fit = math.Bin.exponential(y,
                                                 bin_count=len(counts),
                                                 density=True)
    self.assertEqual(1, sum(counts_fit))
    self.assertTrue(np.isneginf(edges_fit[0]))
    self.assertEqual(1, counts_fit[0])
    self.assertEqual(0, edges_fit[-1])

  def test_bin_downsample(self):
    bin_count = np.random.randint(5, 100)
    y_min = np.random.uniform(-10, 0)
    y_max = np.random.uniform(1, 10)
    edges = np.linspace(y_min, y_max, bin_count + 1)
    counts = np.random.randint(100, 1000, bin_count)
    y = []
    for i in range(bin_count):
      y.extend(
          np.random.uniform(edges[i], edges[i + 1], counts[i] - 2).tolist())
      y.append(edges[i])
      y.append(edges[i + 1])
    y = np.array(y)
    n = sum(counts)

    counts_fit, edges_fit = math.Bin.linear(y,
                                            bin_count=bin_count,
                                            density=True)

    for e, e_fit in zip(edges, edges_fit):
      self.assertAlmostEqual(e, e_fit)

    counts_norm = counts / np.diff(edges) / n

    for c, c_fit in zip(counts_norm, counts_fit):
      self.assertAlmostEqual(c, c_fit, 2)

    y_down = math.Bin.downsample(y, n_max=n * 10)
    self.assertListEqual(y.tolist(), y_down.tolist())

    y_up = []
    for s in y:
      y_up.extend([s] * 5)
    y_up = np.array(y_up)

    y_down = math.Bin.downsample(y_up, n_max=n, bin_count=bin_count)

    counts_fit, _ = np.histogram(y_down, bins=edges_fit, density=True)

    for c, c_fit in zip(counts_norm, counts_fit):
      self.assertAlmostEqual(c, c_fit, 2)

    y_up = np.round(y_up, 2)

    y_down = math.Bin.downsample(y_up, n_max=n, bin_count=None)
    self.assertLessEqual(len(y_down), n)

  def test_image_layer(self):
    shape = np.random.randint(10, 100, 3)

    above = np.zeros(shape=shape, dtype=np.float32)
    below = np.zeros(shape=shape[:2], dtype=np.float32)

    self.assertNotEqual(above.shape, below.shape)
    self.assertRaises(ValueError, math.Image.layer_rgba, below, above)

    above = np.zeros(shape=shape, dtype=np.float32)
    below = np.zeros(shape=shape, dtype=np.float32)

    self.assertEqual(above.shape, below.shape)
    self.assertRaises(ValueError, math.Image.layer_rgba, below, above)

    shape[2] = 4
    above = np.zeros(shape=shape, dtype=np.float32)
    below = np.zeros(shape=shape, dtype=np.float32)

    above_rgba = np.random.uniform(0, 1, 4)
    above[:, :] = above_rgba
    below_rgba = np.random.uniform(0, 1, 4)
    below[:, :] = below_rgba

    out = math.Image.layer_rgba(below, above)
    out_rgba = out[0][0]
    out_a = above_rgba[3] + below_rgba[3] * (1 - above_rgba[3])
    self.assertAlmostEqual(out_a, out_rgba[3], 3)

    out_r = (above_rgba[0] * above_rgba[3] + (below_rgba[0] * below_rgba[3]) *
             (1 - above_rgba[3])) / out_a
    self.assertAlmostEqual(out_r, out_rgba[0], 3)

    out_g = (above_rgba[1] * above_rgba[3] + (below_rgba[1] * below_rgba[3]) *
             (1 - above_rgba[3])) / out_a
    self.assertAlmostEqual(out_g, out_rgba[1], 3)

    out_b = (above_rgba[2] * above_rgba[3] + (below_rgba[2] * below_rgba[3]) *
             (1 - above_rgba[3])) / out_a
    self.assertAlmostEqual(out_b, out_rgba[2], 3)

  def test_image_base64(self):
    shape = np.random.randint(10, 100, 3)
    shape[2] = 4

    image = np.random.uniform(0.0, 1.0, size=shape)

    out = math.Image.np_to_base64(image)
    with io.BytesIO(base64.b64decode(out)) as buf:
      image_pil = PIL.Image.open(buf, formats=["PNG"])
      image_pil.load()

    out = b"0123456789ABCDEF" + out
    with io.BytesIO(base64.b64decode(out)) as buf:
      self.assertRaises(PIL.UnidentifiedImageError,
                        PIL.Image.open,
                        buf,
                        formats=["PNG"])

  def test_image_file(self):
    self.__clean_test_root()

    shape = np.random.randint(10, 100, 3)
    shape[2] = 4

    image = np.random.uniform(0.0, 1.0, size=shape)

    path = str(self._TEST_ROOT.joinpath("image.png"))
    self._TEST_ROOT.mkdir(parents=True, exist_ok=True)

    math.Image.np_to_file(image, path)

    image_pil = PIL.Image.open(path, formats=["PNG"])
    image_pil.load()

    self.__clean_test_root()
