"""Test module hardware_tools.math.stats
"""

import numpy as np

from hardware_tools.math import stats

from tests import base


class TestStats(base.TestBase):
  """Test stats methods
  """

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

    counts_fit, edges_fit = stats.bin_linear(y,
                                             bin_count=bin_count,
                                             density=True)

    for e, e_fit in zip(edges, edges_fit):
      self.assertAlmostEqual(e, e_fit)

    counts_norm = counts / np.diff(edges) / counts.sum()

    for c, c_fit in zip(counts_norm, counts_fit):
      self.assertEqualWithinError(c, c_fit, 0.01)

    y = np.repeat(y[0], counts.sum())
    counts_fit, edges_fit = stats.bin_linear(y,
                                             bin_count=bin_count,
                                             density=False)
    self.assertEqual(counts_fit.sum(), counts.sum())

    y = np.repeat(0, counts.sum())
    counts_fit, edges_fit = stats.bin_linear(y,
                                             bin_count=bin_count,
                                             density=False)
    self.assertEqual(counts_fit.sum(), counts.sum())

  def test_bin_exact(self):
    n = self._RNG.integers(100, 1000)
    bins = list(range(-10, 10))
    y = self._RNG.integers(-10, 10, n)
    for i in range(len(bins)):
      y[i] = bins[i]

    counts_fit, bins_fit = stats.bin_exact(y)
    self.assertEqual(n, sum(counts_fit))
    self.assertListEqual(bins, bins_fit)

    counts_fit, bins_fit = stats.bin_exact_np(y)
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

    counts_fit, edges_fit = stats.bin_exponential(y,
                                                  bin_count=len(counts),
                                                  density=False)
    self.assertEqual(n, sum(counts_fit))
    self.assertListEqual(bins, edges_fit.tolist()[:-1])
    self.assertListEqual(counts.tolist(), counts_fit.tolist())

    y[0] = 0
    counts_fit, edges_fit = stats.bin_exponential(y,
                                                  bin_count=len(counts),
                                                  density=False)
    self.assertEqual(n, sum(counts_fit))
    self.assertTrue(np.isneginf(edges_fit[0]))
    self.assertListEqual(bins[1:], edges_fit.tolist()[1:-1])
    self.assertListEqual(counts.tolist(), counts_fit.tolist())

    y = np.zeros(n)
    counts_fit, edges_fit = stats.bin_exponential(y,
                                                  bin_count=len(counts),
                                                  density=False)
    self.assertEqual(n, sum(counts_fit))
    self.assertTrue(np.isneginf(edges_fit[0]))
    self.assertEqual(n, counts_fit[0])
    self.assertEqual(0, edges_fit[-1])

    counts_fit, edges_fit = stats.bin_exponential(y,
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

    counts_fit, edges_fit = stats.bin_linear(y,
                                             bin_count=bin_count,
                                             density=True)

    for e, e_fit in zip(edges, edges_fit):
      self.assertAlmostEqual(e, e_fit)

    counts_norm = counts / np.diff(edges) / n

    for c, c_fit in zip(counts_norm, counts_fit):
      self.assertEqualWithinError(c, c_fit, 0.01)

    y_down = stats.downsample(y, n_max=n * 10)
    self.assertListEqual(y.tolist(), y_down.tolist())

    y_up = []
    for s in y:
      y_up.extend([s] * 5)
    y_up = np.array(y_up)

    y_down = stats.downsample(y_up, n_max=n, bin_count=bin_count)

    counts_fit, _ = np.histogram(y_down, bins=edges_fit, density=True)

    for c, c_fit in zip(counts_norm, counts_fit):
      self.assertEqualWithinError(c, c_fit, 0.01)

    y_up = np.round(y_up, 2)

    y_down = stats.downsample(y_up, n_max=n, bin_count=None)
    self.assertLessEqual(len(y_down), n)

  def test_uncertain_value(self):
    n = int(1e6)
    samples_a = self._RNG.uniform(0.0, 1.0, n)
    samples_b = self._RNG.normal(1.0, 2.0, n)
    c = self._RNG.uniform(2.0, 100.0)

    avg_a = samples_a.mean()
    avg_b = samples_b.mean()

    std_a = samples_a.std()
    std_b = samples_b.std()

    a = stats.UncertainValue.samples(samples_a)
    b = stats.UncertainValue.samples(samples_b)
    self.assertEqualWithinSampleError(avg_a, a.value, n)
    self.assertEqualWithinSampleError(std_a, a.stddev, n)
    self.assertEqualWithinSampleError(avg_b, b.value, n)
    self.assertEqualWithinSampleError(std_b, b.stddev, n)

    r = stats.UncertainValue.samples(np.array([]))
    self.assertTrue(np.isnan(r.value))
    self.assertTrue(np.isnan(r.stddev))

    r = str(a)
    self.assertEqual(r, f"(µ={a.value},σ={a.stddev})")
    r = f"{a}"
    self.assertEqual(r, f"(µ={a.value},σ={a.stddev})")
    r = f"{a:6.4e}"
    self.assertEqual(r, f"(µ={a.value:6.4e},σ={a.stddev:6.4e})")
    self.assertRaises(TypeError, str.format, "{0:5d}", a)

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

    # This is an approximation, check approximation
    samples_a = self._RNG.uniform(9.0, 10.0, n)
    a = stats.UncertainValue.samples(samples_a)
    r = np.log(a)
    r_stddev = a.stddev / a.value
    self.assertEqualWithinSampleError(np.log(a.value), r.value, n)
    self.assertEqualWithinSampleError(r_stddev, r.stddev, n)

    # This is an approximation, check approximation
    r = np.log10(a)
    r_stddev = a.stddev / (a.value * np.log(10))
    self.assertEqualWithinSampleError(np.log10(a.value), r.value, n)
    self.assertEqualWithinSampleError(r_stddev, r.stddev, n)

    self.assertFalse(a.isnan())
    a = stats.UncertainValue(0.0, np.nan)
    self.assertTrue(a.isnan())
    a = stats.UncertainValue(np.nan, 0.0)
    self.assertTrue(a.isnan())

    r = np.log(a)
    self.assertTrue(r.isnan())

    r = np.log10(a)
    self.assertTrue(r.isnan())

    a = stats.UncertainValue.samples(-samples_a)

    r = np.log(a)
    self.assertTrue(r.isnan())

    r = np.log10(a)
    self.assertTrue(r.isnan())
