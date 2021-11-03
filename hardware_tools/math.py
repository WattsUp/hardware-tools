"""Collection of math functions
"""

from __future__ import annotations

import base64
import io
from typing import Iterable, Union

import numpy as np
import PIL.Image
from scipy import optimize
import sklearn.exceptions
import sklearn.mixture
import sklearn.utils._testing


def interpolate_sinc(x: np.ndarray, xp: np.ndarray,
                     yp: np.ndarray) -> np.ndarray:
  """Resample a time series using sinc interpolation

  Adds a sinc function at each point then resamples at x. x is expected to
  be regularly sampled (constant sinc width). x does not need to be
  regularly sampled.

  Args:
    x: The x-coordinates at which to evaluate the interpolated values
    xp: The x-coordinates of the data points, regularly sampled
    yp: The y-coordinates of the data points, same length as xp (>= 2)

  Returns:
    The interpolated values, same shape as x

  Raises:
    ValueError: If xp and yp are not the same length, have fewer than 2 points,
    are not 1D. Checking for regularly sampled xp is expensive and not checked.
  """
  if len(xp.shape) != 1:
    raise ValueError("Input must be 1D")
  if xp.shape != yp.shape:
    raise ValueError("Input must be same shape")
  if xp.shape[0] < 2:
    raise ValueError("Input must have at least 2 points")
  period = xp[1] - xp[0]
  sinc = np.tile(x, (len(xp), 1)) - np.tile(xp[:, np.newaxis], (1, len(x)))
  y = np.dot(yp, np.sinc(sinc / period))
  return y


class Point2D:
  """2D Point class with an x-coordinate and y-coordinate

  Attributes:
    x: X-coordinate (horizontal)
    y: Y-coordinate (vertical)
  """

  def __init__(self, x: float, y: float) -> None:
    """Initialize a Point2D

    Args:
      x: X-coordinate (horizontal)
      y: Y-coordinate (vertical)
    """
    self.x = x
    self.y = y

  def __str__(self) -> str:
    return f"({self.x}, {self.y})"

  def in_rect(self, start: Point2D, end: Point2D) -> bool:
    """Check if point lies in rectangle formed by start and stop

    Args:
      start: One corner of rectangle
      end: Second corner of rectangle

    Returns:
      True if self is within all edges of rectangle or on an edge
    """
    if ((self.x <= max(start.x, end.x)) and (self.x >= min(start.x, end.x)) and
        (self.y <= max(start.y, end.y)) and (self.y >= min(start.y, end.y))):
      return True
    return False

  @staticmethod
  def orientation(p: Point2D, q: Point2D, r: Point2D) -> int:
    """Compute the orientation of three points from p->q->r

    Anticlockwise:
      r--q
        /
       /
      p

    Linear:
      p---q---r

    Clockwise:
         q-----r
        /
       /
      p

    Args:
      p: Point 1
      q: Point 2
      r: Point 3

    Returns:
      -1: anticlockwise
      0: colinear
      1: clockwise
    """
    val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y))
    norm = ((p.x**2 + p.y**2) * (q.x**2 + q.y**2) * (r.x**2 + r.y**2))**(1 / 3)
    val = val / norm
    if abs(val) < np.finfo(type(val)).eps * 10:
      return 0
    return np.sign(val)


class Line2D:
  """2D Line formed between two points

  Attributes:
    p: First point
    q: second point
  """

  def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
    """Initialize Line2D

    Args:
      x1: X-coordinate of first point
      y1: Y-coordinate of first point
      x2: X-coordinate of second point
      y2: Y-coordinate of second point
    """
    self.p = Point2D(x1, y1)
    self.q = Point2D(x2, y2)

  def __str__(self) -> str:
    return f"({self.p}, {self.q})"

  def intersecting(self, l: Line2D) -> bool:
    """Check if two line segments intersect

    Checks line segments, not lines of infinite length

    Args:
      l: Second line segment

    Returns:
      True if line segments intersect
    """
    return self.intersecting_points(l.p, l.q)

  def intersecting_points(self, p: Point2D, q: Point2D) -> bool:
    """Check if two line segments intersect

    Checks line segments, not lines of infinite length

    Args:
      p: First point of second line segment
      q: Second point of second line segment

    Returns:
      True if line segments intersect
    """
    # 4 Combinations of points
    o1 = Point2D.orientation(self.p, self.q, p)
    o2 = Point2D.orientation(self.p, self.q, q)
    o3 = Point2D.orientation(p, q, self.p)
    o4 = Point2D.orientation(p, q, self.q)

    # General case
    if (o1 != o2) and (o3 != o4):
      return True

    # self.p, self.q, p are colinear and p lies in self
    if (o1 == 0) and p.in_rect(self.p, self.q):
      return True

    # self.p, self.q, q are colinear and q lies in self
    if (o2 == 0) and q.in_rect(self.p, self.q):
      return True

    # p, q, self.p are colinear and self.p lies in pq
    if (o3 == 0) and self.p.in_rect(p, q):
      return True

    # p, q, self.q are colinear and self.q lies in pq
    if (o4 == 0) and self.q.in_rect(p, q):
      # Should always be false because no situation only satisfies only 1 of
      # these checks
      return True  # pragma: no cover

    return False

  def intersection(self, l: Line2D) -> Point2D:
    """Get the intersection of two lines [not line segments]

    Args:
      l: Second line

    Returns:
      Intersection point, None if lines to do not intersect
    """
    return self.intersection_points(l.p, l.q)

  def intersection_points(self, p: Point2D, q: Point2D) -> Point2D:
    """Get the intersection of two lines [not line segments]

    Args:
      p: First point of second line segment
      q: Second point of second line segment

    Returns:
      Intersection point, None if lines are parallel
    """

    def line_params(p: Point2D, q: Point2D) -> list[float]:
      """Convert pair of points to line parameters

      Args:
        p: First point
        q: Second point

      Returns:
        list[dy, dx, det([p, q])]
      """
      dy = p.y - q.y
      dx = q.x - p.x
      det = p.x * q.y - q.x * p.y
      return dy, dx, -det

    l1 = line_params(self.p, self.q)
    l2 = line_params(p, q)
    det = l1[0] * l2[1] - l1[1] * l2[0]
    if det == 0:
      return None
    det_x = l2[1] * l1[2] - l2[2] * l1[1]
    det_y = l1[0] * l2[2] - l1[2] * l2[0]
    return Point2D(det_x / det, det_y / det)


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

  @staticmethod
  def y(x: Union[float, np.ndarray], amplitude: float, mean: float,
        stddev: float) -> Union[float, np.ndarray]:
    """Compute gaussian function at values of x

    Args:
      x: Single or multiple values of x
      amplitude: A
      mean: mu
      stddev: sigma

    Returns:
      Single or multiple values of y
    """
    return amplitude * np.exp(-(
        (x - mean) / stddev)**2 / 2) / (np.sqrt(2 * np.pi * stddev**2))

  def compute(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute gaussian function at values of x

    Args:
      x: Single or multiple values of x

    Returns:
      Single or multiple values of y
    """
    return Gaussian.y(x, self.amplitude, self.mean, self.stddev)

  @staticmethod
  def fit_pdf(x: np.ndarray, frequency: np.ndarray) -> Gaussian:
    """Fit a gaussian function to PDF data

    Args:
      x: sample values
      frequency: frequency of values

    Returns:
      Gaussian function
    """
    # Normalize frequency
    total = np.sum(frequency)
    frequency = frequency / total

    guess_mean = np.average(x, weights=frequency)
    guess_stddev = np.sqrt(np.average((x - guess_mean)**2, weights=frequency))
    opt = optimize.curve_fit(Gaussian.y,
                             x,
                             frequency,
                             p0=[np.max(frequency), guess_mean,
                                 guess_stddev])[0]

    # Undo normalization
    opt[0] = opt[0] * total
    return Gaussian(opt[0], opt[1], opt[2])


class GaussianMix:
  """Gaussian mix function, sum of multiple gaussian functions

  y = sum(A / (sqrt(2pi*stddev^2)) * exp(-((x - mu) / stddev)^2 / 2))

  Attributes:
    components: list of gaussian functions
  """

  def __init__(self, components: list[Gaussian]) -> None:
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
    x = x[np.argmax(y)]
    x = np.linspace(x - (x_max - x_min) / 1000, x + (x_max - x_min) / 1000,
                    1000)
    y = self.compute(x)
    return x[np.argmax(y)]

  @staticmethod
  # @sklearn.utils._testing.ignore_warnings(
  #     category=sklearn.exceptions.ConvergenceWarning)
  def fit_samples(y: np.ndarray,
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
    y_span = np.amax(y) - np.amin(y)
    if y_span == 0:
      return [[1, y[0], 0]]
    y_avg = np.average(y)
    y_norm = y.copy().reshape(-1, 1)
    y_norm = (y_norm - y_avg) / y_span

    models = []
    for i in range(min(len(y_norm), n_max)):
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


class Bin:
  """Collection of binning and other histogram functions
  """

  @staticmethod
  def linear(y: np.array,
             bin_count: int = 100,
             density: bool = None) -> tuple[np.array, np.array]:
    """Bin values with equal width bins

    Args:
      y: Sample values
      bin_count: Number of equal width bins
      density: Passed to np.histogram, roughly True to get a PDF

    Returns:
      counts, edges
    """
    y_min = min(y)
    y_max = max(y)
    if y_min == y_max:
      if y_min == 0:
        y_min = -1
        y_max = 1
      else:
        center = y_min
        y_min = center - abs(center) * 0.05
        y_max = center + abs(center) * 0.05
    edges = np.linspace(y_min, y_max, bin_count + 1)
    return np.histogram(y, edges, density=density)

  @staticmethod
  def exact(y: Iterable) -> tuple[list, list]:
    """Bin values with exact indices

    Args:
      y: Sample values

    Returns:
      counts, bins
    """
    counts = {}
    for e in y:
      if e in counts:
        counts[e] += 1
      else:
        counts[e] = 1
    bins = sorted(counts.keys())
    counts = [counts[b] for b in bins]
    return counts, bins

  @staticmethod
  def exact_np(y: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """Bin values with exact indices

    Args:
      y: Sample values

    Returns:
      counts, bins as np arrays
    """
    counts = {}
    for e in y:
      if e in counts:
        counts[e] += 1
      else:
        counts[e] = 1
    bins = sorted(counts.keys())
    counts = [counts[b] for b in bins]
    return np.array(counts), np.array(bins)

  @staticmethod
  def exponential(y: Iterable,
                  bin_count: int = 100,
                  density: bool = None) -> tuple[np.array, np.array]:
    """Bin values with equal exponential width bins

    Args:
      y: Sample values
      bin_count: Number of equal exponential width bins
      density: Passed to np.histogram, roughly True to get a PDF

    Returns:
      counts, edges
    """
    with np.errstate(divide="ignore"):
      y_exp = np.log10(y)
    y_min = min(y_exp)
    y_max = max(y_exp)
    if np.isneginf(y_min):
      if np.isneginf(y_max):
        y_max = 0
      else:
        y_max = int(np.ceil(y_max))
      edges = [np.NINF]
      bins = np.arange(y_max - (bin_count - 1), y_max + 1, 1, dtype=np.float64)
      edges.extend(bins)
      edges = np.array(edges)
      counts, _ = np.histogram(y_exp, edges, density=False)
      if density:
        counts = counts / 1 / counts.sum()
      return counts, edges
    else:
      edges = np.linspace(y_min, y_max, bin_count + 1)
      return np.histogram(y_exp, edges, density=density)

  @staticmethod
  def downsample(y: np.ndarray, n_max: int = 50e3, bin_count=500) -> np.ndarray:
    """Reduce the number of samples to at most n_max, preserving sample
    frequency

    Bins the values, scales the counts to total of n_max, then undoes the
    binning. Does use floor for sample count, be wary of loss of low frequency
    samples.

    Args:
      y: Sample values
      n_max: Maximum number of samples in returned dataset
      bin_count: Number of equal width bins, None for exact binning

    Returns:
      Downsampled dataset samples
    """
    scale = n_max / len(y)
    if scale >= 1:
      return y
    if bin_count is not None:
      counts, edges = Bin.linear(y, bin_count=bin_count, density=False)
      bins = (edges[:-1] + edges[1:]) / 2
    else:
      counts, bins = Bin.exact(y)
    y_down = []
    for i in range(len(counts)):
      count_down = int(np.floor(counts[i] * scale))
      y_down.extend([bins[i]] * count_down)
    return np.array(y_down)


class Image:
  """Collection of image processing functions
  """

  @staticmethod
  def layer_rgba(below: np.ndarray, above: np.ndarray) -> np.ndarray:
    """Layer a RGBA image on top of another using alpha compositing

    Images are np.arrays [row, column, channel=4]

    Args:
      below: Image on bottom of stack
      above: Image on top of stack

    Returns:
      Combined image

    Raises:
      ValueError if image shapes (resolution) don't match or they are not 4
      channels
    """
    if below.shape != above.shape:
      raise ValueError(
          f"Images must be same shape {below.shape} vs. {above.shape}")
    if below.shape[2] != 4:
      raise ValueError("Image is not RGBA")

    alpha_a = above[:, :, 3]
    alpha_b = below[:, :, 3]
    alpha_out = alpha_a + np.multiply(alpha_b, 1 - alpha_a)
    out = np.zeros(below.shape, dtype=below.dtype)
    out[:, :, 0] = np.divide(
        np.multiply(above[:, :, 0], alpha_a) +
        np.multiply(np.multiply(below[:, :, 0], alpha_b), 1 - alpha_a),
        alpha_out,
        where=alpha_out != 0)
    out[:, :, 1] = np.divide(
        np.multiply(above[:, :, 1], alpha_a) +
        np.multiply(np.multiply(below[:, :, 1], alpha_b), 1 - alpha_a),
        alpha_out,
        where=alpha_out != 0)
    out[:, :, 2] = np.divide(
        np.multiply(above[:, :, 2], alpha_a) +
        np.multiply(np.multiply(below[:, :, 2], alpha_b), 1 - alpha_a),
        alpha_out,
        where=alpha_out != 0)
    out[:, :, 3] = alpha_out
    return out

  @staticmethod
  def np_to_base64(image: np.ndarray) -> bytes:
    """Convert a numpy image to base64 encoded PNG

    Args:
      image: Image to convert, shape=[row, column, channels] from 0.0 to 1.0

    Returns:
      base64 encoded PNG image
    """
    image = PIL.Image.fromarray((255 * image).astype("uint8"))
    with io.BytesIO() as buf:
      image.save(buf, "PNG")
      return base64.b64encode(buf.getvalue())

  @staticmethod
  def np_to_file(image: np.ndarray, path: str):
    """Save a numpy image to file

    Args:
      image: Image to save, shape=[row, column, channels] from 0.0 to 1.0
      path: Path to output file
    """
    image = PIL.Image.fromarray((255 * image).astype("uint8"))
    image.save(path)
