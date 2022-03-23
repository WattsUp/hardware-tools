"""Test module hardware_tools.math.gaussian
"""

import numpy as np
from hardware_tools.math import gaussian

from tests import base


class TestGaussian(base.TestBase):
  """Test gaussian methods
  """

  def test_class(self):
    x0 = self._RNG.uniform(-1000, 1000)
    x1 = self._RNG.uniform(-1000, 1000)
    amplitude = self._RNG.uniform(-1000, 1000)
    mean = self._RNG.uniform(-1000, 1000)
    stddev = self._RNG.uniform(1, 1000)

    g = gaussian.Gaussian(amplitude, mean, stddev)
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

    g = gaussian.Gaussian(amplitude, mean, 0)
    self.assertEqual(0, g.compute(mean + 1))
    self.assertTrue(np.isposinf(g.compute(mean)))
    y = g.compute(x)
    self.assertTrue(np.isposinf(y.max()))
    y = np.nan_to_num(y, posinf=1)
    self.assertEqual(1, y.sum())

  def test_comparison(self):
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
    threshold = gaussian.sample_error_inv(n, p_fail)
    self.assertAlmostEqual(p_fail, gaussian.sample_error(n, threshold))
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
    threshold = gaussian.sample_error_inv(n, p_fail)
    self.assertAlmostEqual(p_fail, gaussian.sample_error(n, threshold))
    p_fail_real = np.mean(errors > threshold)
    self.assertEqualWithinError(p_fail, p_fail_real, 0.1)

  def test_fit(self):
    amplitude = self._RNG.uniform(-1000, 1000)
    mean = self._RNG.uniform(-1000, 1000)
    stddev = self._RNG.uniform(1, 1000)
    n = self._RNG.integers(500, 1000)

    x = self._RNG.uniform(mean - stddev * 5, mean + stddev * 5, n)
    g = gaussian.Gaussian(amplitude, mean, stddev)
    y = g.compute(x)

    g_fit = gaussian.fit_pdf(x, y)
    self.assertAlmostEqual(amplitude, g_fit.amplitude)
    self.assertAlmostEqual(mean, g_fit.mean)
    self.assertAlmostEqual(abs(stddev), abs(g_fit.stddev))

  def test_mix(self):
    n_components = self._RNG.integers(2, 5)
    amplitude = self._RNG.uniform(1, 1000, n_components)
    mean = self._RNG.uniform(-1000, 1000, n_components)
    stddev = self._RNG.uniform(1, 1000, n_components)
    n = self._RNG.integers(500, 1000)

    components = [
        gaussian.Gaussian(a, m, s) for a, m, s in zip(amplitude, mean, stddev)
    ]
    g = gaussian.GaussianMix(components[:1])
    self.assertEqual(mean[0], g.center())

    g = gaussian.GaussianMix(components)
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
    components.append(gaussian.Gaussian(0, 0, 1))
    g = gaussian.GaussianMix(components)
    self.assertEqualWithinError(components[0].mean, g.center(), 0.01)

  def test_mix_fit(self):
    n_components = self._RNG.integers(2, 5)
    amplitude = self._RNG.uniform(1, 5, n_components)
    mean = self._RNG.uniform(-1000, 1000, n_components)
    stddev = self._RNG.uniform(1, 100, n_components)
    n = self._RNG.integers(5000, 10000)

    amplitude = amplitude / amplitude.sum()

    components = [
        gaussian.Gaussian(a, m, s) for a, m, s in zip(amplitude, mean, stddev)
    ]
    components = sorted(components, key=lambda c: -c.amplitude)

    y = np.array([amplitude[0]] * n)
    g_fit = gaussian.fit_mix_samples(y, n_max=n_components)
    self.assertEqual(len(g_fit.components), 1)
    self.assertEqualWithinError(1, g_fit.components[0].amplitude, 0.01)
    self.assertEqualWithinError(amplitude[0], g_fit.components[0].mean, 0.01)
    self.assertEqualWithinError(0, g_fit.components[0].stddev, 0.01)

    g = gaussian.GaussianMix(components)
    y = []
    for c in components:
      y.extend(
          self._RNG.normal(c.mean, c.stddev, int(n * c.amplitude)).tolist())
    y = np.array(y)
    x = np.linspace(min(y), max(y), 1000)
    g_fit = gaussian.fit_mix_samples(y, n_max=n_components)

    pdf = g.compute(x)
    pdf_fit = g_fit.compute(x)
    errors = (pdf_fit - pdf) / max(pdf)

    error = np.sqrt(np.mean(errors**2))

    self.assertLess(error, 0.15)
