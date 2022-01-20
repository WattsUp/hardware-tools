"""Test module hardware_tools.math
"""

import base64
import io

import numpy as np
import PIL
import PIL.Image

from hardware_tools import math

from tests import base


class TestMath(base.TestBase):
  """Test math methods
  """

  def test_interpolate_sinc(self):
    x_min = self._RNG.integers(-1000, 500)
    x_max = x_min + self._RNG.integers(1, 500)
    n_in = self._RNG.integers(500, 1000)
    n_down = self._RNG.integers(10, n_in - 10)
    n_up = self._RNG.integers(1000, 5000)

    x_in = np.linspace(x_min, x_max, n_in)
    y_in = self._RNG.uniform(-1, 1, n_in)
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

    self.assertGreaterEqual(input_band_power, 0.99)

    # 2D input
    x_in = self._RNG.uniform(x_min, x_max, (n_in, n_in))
    self.assertRaises(ValueError, math.interpolate_sinc, x_up, x_in, y_in)

    # Mismatched input
    x_in = np.linspace(x_min, x_max, n_in + 1)
    self.assertRaises(ValueError, math.interpolate_sinc, x_up, x_in, y_in)

    # Single point input
    x_in = np.linspace(x_min, x_max, 1)
    y_in = self._RNG.uniform(-1, 1, 1)
    self.assertRaises(ValueError, math.interpolate_sinc, x_up, x_in, y_in)

  def test_point2d(self):
    x0 = self._RNG.uniform(-1000, 1000)
    y0 = self._RNG.uniform(-1000, 1000)
    x1 = self._RNG.uniform(-1000, 1000)
    y1 = self._RNG.uniform(-1000, 1000)

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
    x0 = self._RNG.uniform(-1000, 1000)
    y0 = self._RNG.uniform(-1000, 1000)
    x1 = self._RNG.uniform(-1000, 1000)
    y1 = self._RNG.uniform(-1000, 1000)

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
    x0 = self._RNG.uniform(-1000, 1000)
    x1 = self._RNG.uniform(-1000, 1000)
    amplitude = self._RNG.uniform(-1000, 1000)
    mean = self._RNG.uniform(-1000, 1000)
    stddev = self._RNG.uniform(1, 1000)

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

    g = math.Gaussian(amplitude, mean, 0)
    self.assertEqual(0, g.compute(mean + 1))
    self.assertTrue(np.isposinf(g.compute(mean)))
    y = g.compute(x)
    self.assertTrue(np.isposinf(y.max()))
    y = np.nan_to_num(y, posinf=1)
    self.assertEqual(1, y.sum())

  def test_gaussian_comparison(self):
    mean = self._RNG.uniform(-1000, 1000)
    stddev = self._RNG.uniform(1, 1000)

    n = 100
    n_samples = 10000
    errors = []
    for _ in range(n_samples):
      x = self._RNG.normal(mean, stddev, n)
      error = np.abs(x.std() / stddev - 1)
      errors.append(error)
    errors = np.array(errors)

    p_fail = 0.1
    threshold = math.Gaussian.sample_error_inv(n, p_fail)
    self.assertAlmostEqual(p_fail, math.Gaussian.sample_error(n, threshold))
    p_fail_real = np.mean(errors > threshold)
    self.assertEqualWithinError(p_fail, p_fail_real, 0.1)

    n = 1000
    n_samples = 10000
    errors = []
    for _ in range(n_samples):
      x = self._RNG.normal(mean, stddev, n)
      error = np.abs(x.std() / stddev - 1)
      errors.append(error)
    errors = np.array(errors)

    p_fail = 0.1
    threshold = math.Gaussian.sample_error_inv(n, p_fail)
    self.assertAlmostEqual(p_fail, math.Gaussian.sample_error(n, threshold))
    p_fail_real = np.mean(errors > threshold)
    self.assertEqualWithinError(p_fail, p_fail_real, 0.1)

  def test_gaussian_fit(self):
    amplitude = self._RNG.uniform(-1000, 1000)
    mean = self._RNG.uniform(-1000, 1000)
    stddev = self._RNG.uniform(1, 1000)
    n = self._RNG.integers(500, 1000)

    x = self._RNG.uniform(mean - stddev * 5, mean + stddev * 5, n)
    g = math.Gaussian(amplitude, mean, stddev)
    y = g.compute(x)

    g_fit = math.Gaussian.fit_pdf(x, y)
    self.assertAlmostEqual(amplitude, g_fit.amplitude)
    self.assertAlmostEqual(mean, g_fit.mean)
    self.assertAlmostEqual(abs(stddev), abs(g_fit.stddev))

  def test_gaussian_mix(self):
    n_components = self._RNG.integers(2, 5)
    amplitude = self._RNG.uniform(1, 1000, n_components)
    mean = self._RNG.uniform(-1000, 1000, n_components)
    stddev = self._RNG.uniform(1, 1000, n_components)
    n = self._RNG.integers(500, 1000)

    components = [
        math.Gaussian(a, m, s) for a, m, s in zip(amplitude, mean, stddev)
    ]
    g = math.GaussianMix(components[:1])
    self.assertEqual(mean[0], g.center())

    g = math.GaussianMix(components)
    self.assertEqual(str(g), str(components))
    x = self._RNG.uniform(
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
    self.assertEqualWithinError(components[0].mean, g.center(), 0.01)

  def test_gaussian_mix_fit(self):
    n_components = self._RNG.integers(2, 5)
    amplitude = self._RNG.uniform(1, 5, n_components)
    mean = self._RNG.uniform(-1000, 1000, n_components)
    stddev = self._RNG.uniform(1, 100, n_components)
    n = self._RNG.integers(5000, 10000)

    amplitude = amplitude / amplitude.sum()

    components = [
        math.Gaussian(a, m, s) for a, m, s in zip(amplitude, mean, stddev)
    ]
    components = sorted(components, key=lambda c: -c.amplitude)

    y = np.array([amplitude[0]] * n)
    g_fit = math.GaussianMix.fit_samples(y, n_max=n_components)
    self.assertEqual(len(g_fit.components), 1)
    self.assertEqualWithinError(1, g_fit.components[0].amplitude, 0.01)
    self.assertEqualWithinError(amplitude[0], g_fit.components[0].mean, 0.01)
    self.assertEqualWithinError(0, g_fit.components[0].stddev, 0.01)

    g = math.GaussianMix(components)
    y = []
    for c in components:
      y.extend(
          self._RNG.normal(c.mean, c.stddev, int(n * c.amplitude)).tolist())
    y = np.array(y)
    x = np.linspace(min(y), max(y), 1000)
    g_fit = math.GaussianMix.fit_samples(y, n_max=n_components)

    pdf = g.compute(x)
    pdf_fit = g_fit.compute(x)
    errors = (pdf_fit - pdf) / max(pdf)

    error = np.sqrt(np.mean(errors**2))

    self.assertLess(error, 0.15)

  def test_bin_linear(self):
    bin_count = self._RNG.integers(5, 100)
    y_min = self._RNG.uniform(-10, 0)
    y_max = self._RNG.uniform(1, 10)
    edges = np.linspace(y_min, y_max, bin_count + 1)
    counts = self._RNG.integers(100, 1000, bin_count)
    y = []
    for i in range(bin_count):
      y.extend(
          self._RNG.uniform(edges[i], edges[i + 1], counts[i] - 2).tolist())
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
      self.assertEqualWithinError(c, c_fit, 0.01)

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
    n = self._RNG.integers(100, 1000)
    bins = list(range(-10, 10))
    y = self._RNG.integers(-10, 10, n)
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
    counts = self._RNG.integers(100, 1000, len(bins))
    for i in range(len(bins)):
      y.extend(10**(bins[i] + self._RNG.uniform(0, 1, counts[i] - 1)))
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
    bin_count = self._RNG.integers(5, 100)
    y_min = self._RNG.uniform(-10, 0)
    y_max = self._RNG.uniform(1, 10)
    edges = np.linspace(y_min, y_max, bin_count + 1)
    counts = self._RNG.integers(100, 1000, bin_count)
    y = []
    for i in range(bin_count):
      y.extend(
          self._RNG.uniform(edges[i], edges[i + 1], counts[i] - 2).tolist())
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
      self.assertEqualWithinError(c, c_fit, 0.01)

    y_down = math.Bin.downsample(y, n_max=n * 10)
    self.assertListEqual(y.tolist(), y_down.tolist())

    y_up = []
    for s in y:
      y_up.extend([s] * 5)
    y_up = np.array(y_up)

    y_down = math.Bin.downsample(y_up, n_max=n, bin_count=bin_count)

    counts_fit, _ = np.histogram(y_down, bins=edges_fit, density=True)

    for c, c_fit in zip(counts_norm, counts_fit):
      self.assertEqualWithinError(c, c_fit, 0.01)

    y_up = np.round(y_up, 2)

    y_down = math.Bin.downsample(y_up, n_max=n, bin_count=None)
    self.assertLessEqual(len(y_down), n)

  def test_image_layer(self):
    shape = self._RNG.integers(10, 100, 3)

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

    above_rgba = self._RNG.uniform(0, 1, 4)
    above[:, :] = above_rgba
    below_rgba = self._RNG.uniform(0, 1, 4)
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
    shape = self._RNG.integers(10, 100, 3)
    shape[2] = 4

    image = self._RNG.uniform(0.0, 1.0, size=shape)

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
    shape = self._RNG.integers(10, 100, 3)
    shape[2] = 4

    image = self._RNG.uniform(0.0, 1.0, size=shape)

    path = str(self._TEST_ROOT.joinpath("image.png"))

    math.Image.np_to_file(image, path)

    image_pil = PIL.Image.open(path, formats=["PNG"])
    image_pil.load()

  def test_uncertain_value(self):
    n = int(1e6)
    samples_a = self._RNG.uniform(0.0, 1.0, n)
    samples_b = self._RNG.normal(1.0, 2.0, n)
    c = self._RNG.uniform(2.0, 100.0)

    avg_a = samples_a.mean()
    avg_b = samples_b.mean()

    std_a = samples_a.std()
    std_b = samples_b.std()

    a = math.UncertainValue(avg_a, std_a)
    b = math.UncertainValue(avg_b, std_b)

    samples_r = samples_a + samples_b
    r = a + b
    self.assertEqualWithinSampleError(samples_r.mean(), r.value, n)
    self.assertEqualWithinSampleError(samples_r.std(), r.stddev, n)

    samples_r = samples_a + c
    r = a + c
    self.assertEqualWithinSampleError(samples_r.mean(), r.value, n)
    self.assertEqualWithinSampleError(samples_r.std(), r.stddev, n)

    samples_r = samples_a - samples_b
    r = a - b
    self.assertEqualWithinSampleError(samples_r.mean(), r.value, n)
    self.assertEqualWithinSampleError(samples_r.std(), r.stddev, n)

    samples_r = samples_a - c
    r = a - c
    self.assertEqualWithinSampleError(samples_r.mean(), r.value, n)
    self.assertEqualWithinSampleError(samples_r.std(), r.stddev, n)

    # This is an approximation, check approximation
    r = a * b
    r_stddev = (np.sqrt((std_a / avg_a)**2 +
                        (std_b / avg_b)**2)) * (avg_a * avg_b)
    self.assertEqualWithinSampleError(avg_a * avg_b, r.value, n)
    self.assertEqualWithinSampleError(r_stddev, r.stddev, n)

    samples_r = samples_a * c
    r = a * c
    self.assertEqualWithinSampleError(samples_r.mean(), r.value, n)
    self.assertEqualWithinSampleError(samples_r.std(), r.stddev, n)

    # This is an approximation, check approximation
    r = a / b
    r_stddev = (np.sqrt((std_a / avg_a)**2 +
                        (std_b / avg_b)**2)) * (avg_a / avg_b)
    self.assertEqualWithinSampleError(avg_a / avg_b, r.value, n)
    self.assertEqualWithinSampleError(r_stddev, r.stddev, n)

    samples_r = samples_a / c
    r = a / c
    self.assertEqualWithinSampleError(samples_r.mean(), r.value, n)
    self.assertEqualWithinSampleError(samples_r.std(), r.stddev, n)

    self.assertTrue((a < b) == (avg_a < avg_b))
    self.assertTrue((a < c) == (avg_a < c))
    self.assertFalse(a < avg_a)

    self.assertTrue((a <= b) == (avg_a <= avg_b))
    self.assertTrue((a <= c) == (avg_a <= c))
    self.assertLessEqual(a, avg_a)

    self.assertTrue((a == b) == (avg_a == avg_b))
    self.assertTrue((a == c) == (avg_a == c))
    self.assertEqual(a, avg_a)

    self.assertTrue((a != b) == (avg_a != avg_b))
    self.assertTrue((a != c) == (avg_a != c))
    self.assertNotEqual(a, c)

    self.assertTrue((a >= b) == (avg_a >= avg_b))
    self.assertTrue((a >= c) == (avg_a >= c))
    self.assertGreaterEqual(a, avg_a)

    self.assertTrue((a > b) == (avg_a > avg_b))
    self.assertTrue((a > c) == (avg_a > c))
    self.assertFalse(a > avg_a)
