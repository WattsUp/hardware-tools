"""All things Gaussian
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from scipy import optimize, special
import sklearn.exceptions
import sklearn.mixture
import sklearn.utils._testing


class Gaussian:
  """Gaussian function

  y = A / (sqrt(2pi*stddev^2)) * exp(-((x - mu) / stddev)^2 / 2)

  Attributes:
    amplitude: A
    mean: mu
    stddev: sigma
  """

  def __init__(self, amplitude: float, mean: float, stddev: float) -> None:
    """Initialize Gaussian

    y = A / (sqrt(2pi*stddev^2)) * exp(-((x - mu) / stddev)^2 / 2)

    Args:
      amplitude: A
      mean: mu
      stddev: sigma
    """
    self.amplitude = amplitude
    self.mean = mean
    self.stddev = stddev

  def __repr__(self) -> str:
    return f"{{A={self.amplitude}, mu={self.mean}, stddev={self.stddev}}}"

  def compute(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute gaussian function at values of x

    Args:
      x: Single or multiple values of x

    Returns:
      Single or multiple values of y
    """
    return f(x, self.amplitude, self.mean, self.stddev)


class GaussianMix:
  """Gaussian mix function, sum of multiple gaussian functions

  y = sum(A / (sqrt(2pi*stddev^2)) * exp(-((x - mu) / stddev)^2 / 2))

  Attributes:
    components: list of gaussian functions
  """

  def __init__(self, components: List[Gaussian]) -> None:
    """Initialize Gaussian

    y = sum(A / (sqrt(2pi*stddev^2)) * exp(-((x - mu) / stddev)^2 / 2))

    Args:
      components: list of gaussian functions
    """
    self.components = components

  def __repr__(self) -> str:
    return str(self.components)

  def compute(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute gaussian mix function at values of x

    Args:
      x: Single or multiple values of x

    Returns:
      Single or multiple values of y
    """
    y = np.array([c.compute(x) for c in self.components])
    return np.dot(y.T, [1] * len(self.components))

  def center(self) -> float:
    """Compute the center of gaussian mix distribution

    Returns:
      Center of gaussian where peak occurs. Exact result for n=1, approximate
      for n>1
    """
    if len(self.components) == 1:
      return self.components[0].mean

    x_min = min([c.mean for c in self.components])
    x_max = max([c.mean for c in self.components])
    x = np.linspace(x_min, x_max, 1000)
    y = self.compute(x)
    x = x[y.argmax()]
    x = np.linspace(x - (x_max - x_min) / 1000, x + (x_max - x_min) / 1000,
                    1000)
    y = self.compute(x)
    return x[y.argmax()]


def f(x: Union[float, np.ndarray], amplitude: float, mean: float,
      stddev: float) -> Union[float, np.ndarray]:
  """Compute gaussian function at values of x

  If stddev if zero, the return value is positive infinity (np.PINF)

  Args:
    x: Single or multiple values of x
    amplitude: A
    mean: mu
    stddev: sigma

  Returns:
    Single or multiple values of y
  """
  if stddev == 0:
    if isinstance(x, float):
      return np.PINF if x == mean else 0.0
    # Return an impulse at the location closest to mean
    y = np.zeros(x.shape)
    y[np.abs(x - mean).argmin()] = np.PINF
    return y
  return amplitude * np.exp(-(
      (x - mean) / stddev)**2 / 2) / (np.sqrt(2 * np.pi * stddev**2))


def fit_pdf(x: np.ndarray, frequency: np.ndarray) -> Gaussian:
  """Fit a gaussian function to PDF data

  Args:
    x: sample values
    frequency: frequency of values

  Returns:
    Gaussian function
  """
  # Normalize frequency
  total = frequency.sum()
  frequency = frequency / total

  guess_mean = np.average(x, weights=frequency)
  guess_stddev = np.sqrt(np.average((x - guess_mean)**2, weights=frequency))
  opt = optimize.curve_fit(f,
                           x,
                           frequency,
                           p0=[frequency.max(), guess_mean, guess_stddev])[0]

  # Undo normalization
  opt[0] = opt[0] * total
  return Gaussian(opt[0], opt[1], opt[2])


def fit_mix_samples(y: np.ndarray,
                    n_max: int = 10,
                    tol: float = 1e-3) -> GaussianMix:
  """Fit a Gaussian mixture to sampled data

  Args:
    y: Sample values
    n_max: Maximum number of components
    tol: Tolerance of fit (normalized units)

  Returns:
    GaussianMix with best fit
  """
  y_span = y.ptp()
  if y_span == 0:
    return GaussianMix([Gaussian(1, y[0], 0)])
  y_avg = y.mean()
  y_norm = y.copy().reshape(-1, 1)
  y_norm = (y_norm - y_avg) / y_span

  models = []
  # Limit number of components to number of unique elements
  n_max = min(n_max, len(np.unique(y_norm.round(decimals=6))))
  for i in range(n_max):
    models.append(sklearn.mixture.GaussianMixture(i + 1, tol=tol).fit(y_norm))

  aic = [m.aic(y_norm) for m in models]
  m_best = models[np.argmin(aic)]

  components = []
  for i in range(m_best.n_components):
    amplitude = m_best.weights_[i]
    mean = m_best.means_[i][0] * y_span + y_avg
    stddev = abs(np.sqrt(m_best.covariances_[i][0][0]) * y_span)
    components.append(Gaussian(amplitude, mean, stddev))
  components = sorted(components, key=lambda c: -c.amplitude)
  return GaussianMix(components)


def sample_error(n: int, threshold: float) -> float:
  """Compute the probability of error when comparing gaussian random sample

  Given a sample X=np.random.norm(mu, stddev, n), its standard deviation will
  equal stddev±threshold with a probability of failure=p_fail.

  Args:
    n: Size of sample
    threshold: Maximum difference between sample standard deviation and
      population standard deviation. np.abs(np.std(X) / stddev - 1)

  Returns:
    The probability of failure that the sample standard deviation will not be
    close enough to the population standard deviation.
  """
  # Stddev of errors described above = 1 / np.sqrt(2 * n)
  # p_fail = q(threshold_sigmas) * 2
  # p_fail = erfc(threshold_sigmas / sqrt(2)) / 2 * 2
  # p_fail = erfc(threshold / stddev / sqrt(2))
  return special.erfc(threshold * np.sqrt(n))


def sample_error_inv(n: int, p_fail: float) -> float:
  """Compute the threshold when comparing gaussian random sample

  Given a sample X=np.random.norm(mu, stddev, n), its standard deviation will
  equal stddev±threshold with a probability of failure=p_fail.

  Args:
    n: Size of sample
    p_fail: The probability of failure that the sample standard deviation will
      not be close enough to the population standard deviation.

  Returns:
    Maximum difference between sample standard deviation and population
    standard deviation. np.abs(np.std(X) / stddev - 1)
  """
  # Stddev of errors described above = 1 / np.sqrt(2 * n)
  # threshold_sigmas = q_inv(p_fail / 2)
  # threshold_sigmas = sqrt(2) * erfc_inv(2 * p_fail / 2)
  # threshold = np.sqrt(2) * special.erfcinv(p_fail) * stddev
  return special.erfcinv(p_fail) / np.sqrt(n)
