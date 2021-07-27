from __future__ import annotations
import datetime
import numpy as np
from scipy import optimize
from sklearn.mixture import GaussianMixture
from typing import Iterable, Union

def interpolateSinc(x, y, xNew) -> np.ndarray:
  '''!@brief Resample a time series using sinc interpolation

  @param x Input sample points
  @param y Input sample values
  @param xNew Output sample points
  @return np.ndarray Output sample values
  '''
  if len(x) != len(y):
    raise Exception(f'Cannot interpolate arrays of different lengths')
  if len(x) < 2:
    raise Exception(f'Cannot interpolate arrays with fewer than 2 elements')
  T = x[1] - x[0]
  sincM = np.tile(xNew, (len(x), 1)) - \
      np.tile(x[:, np.newaxis], (1, len(xNew)))
  yNew = np.dot(y, np.sinc(sincM / T))
  return yNew

def metricPrefix(value: float, unit: str = '') -> str:
  '''!@brief Format a value using metric prefixes to constrain string length

  @param value Value to format
  @param unit Unit string to append to number and metric prefix
  @return str \'±xxx.x PU\' where P is metric prefix and U is unit
  '''
  metricPrefixes = {
    'T': 1e12,
    'G': 1e9,
    'M': 1e6,
    'k': 1e3,
    ' ': 1e0,
    'm': 1e-3,
    'µ': 1e-6,
    'n': 1e-9,
    'p': 1e-12
  }
  for p, f in metricPrefixes.items():
    if abs(value) > (2 * f):
      return f'{value / f:6.1f} {p}{unit}'
  p, f = list(metricPrefixes.items())[-1]
  return f'{value / f:6.3f} {p}{unit}'

def elapsedStr(start: datetime.datetime,
               end: datetime.datetime = None) -> str:
  '''!@brief Calculate elapsed time since start and format as MM:SS.ss

  @param start Start timestamp
  @param end End timestamp (None will default to now)
  @return str 'MM:SS.ss'
  '''
  if end is None:
    end = datetime.datetime.now()
  d = end - start
  s = d.total_seconds()
  minutes, seconds = divmod(s, 60)
  return f"{int(minutes):02}:{seconds:05.2f}"

class Point:
  def __init__(self, x, y) -> None:
    '''!@brief Create a new Point

    @param x X coordinate
    @param y Y coordinate
    '''
    self.x = x
    self.y = y

  def __str__(self) -> str:
    '''!@brief Get a string representation of the Point

    @return str
    '''
    return f"({self.x}, {self.y})"

  def inRectangle(self, start: Point, end: Point):
    '''!@brief Check if point lies in rectangle formed by start and stop

    @param start Point of line segment
    @param end Point of line segment
    @return bool True if self is within all edges of rectangle
    '''
    if ((self.x <= max(start.x, end.x)) and (self.x >= min(start.x, end.x)) and
            (self.y <= max(start.y, end.y)) and (self.y >= min(start.y, end.y))):
      return True
    return False

  @staticmethod
  def orientation(p: Point, q: Point, r: Point) -> int:
    '''!@brief Compute the orientation of three points

    @param p Point 1
    @param q Point 2
    @param r Point 3
    @return int -1=anticlockwise, 0=colinear, 1=clockwise
    '''
    val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y))
    return np.sign(val)

class Line:
  def __init__(self, x1, y1, x2, y2) -> None:
    '''!@brief Create a new Point

    @param x1 X coordinate of first point
    @param y1 Y coordinate of first point
    @param x2 X coordinate of second point
    @param y2 Y coordinate of second point
    '''
    self.p = Point(x1, y1)
    self.q = Point(x2, y2)

  def __str__(self) -> str:
    '''!@brief Get a string representation of the Line

    @return str
    '''
    return f"({self.p}, {self.q})"

  def intersecting(self, l: Line) -> bool:
    '''!@brief Check if two line segments intersect

    @param l Second line segment
    @return bool True if line segments intersect, false otherwise
    '''
    return self.intersectingPQ(l.p, l.q)

  def intersectingPQ(self, p: Point, q: Point) -> bool:
    '''!@brief Check if two line segments intersect

    @param p First Point of second line segment
    @param q Second Point of second line segment
    @return bool True if line segments intersect, false otherwise
    '''
    # 4 Combinations of points
    o1 = Point.orientation(self.p, self.q, p)
    o2 = Point.orientation(self.p, self.q, q)
    o3 = Point.orientation(p, q, self.p)
    o4 = Point.orientation(p, q, self.q)

    # General case
    if (o1 != o2) and (o3 != o4):
      return True

    # self.p, self.q, p are colinear and p2 lies in self
    if (o1 == 0) and p.inRectangle(self.p, self.q):
      return True

    # self.p, self.q, q are colinear and q2 lies in self
    if (o2 == 0) and q.inRectangle(self.p, self.q):
      return True

    # p, q, self.p are colinear and self.p lies in pq
    if (o3 == 0) and self.p.inRectangle(p, q):
      return True

    # p, q, self.q are colinear and self.q lies in pq
    if (o4 == 0) and self.q.inRectangle(p, q):
      return True

    return False

  def intersection(self, l: Line) -> Point:
    '''!@brief Get the intersection of two lines [not line segments]

    @param l Second line
    @return Point Intersection point, None if lines do not intersect aka parallel
    '''
    return self.intersectionPQ(l.p, l.q)

  def intersectionPQ(self, p: Point, q: Point) -> Point:
    '''!@brief Get the intersection of two lines [not line segments]

    @param p First Point of second line
    @param q Second Point of second line
    @return Point Intersection point, None if lines do not intersect aka parallel
    '''
    def lineParams(p: Point, q: Point):
      A = p.y - q.y
      B = q.x - p.x
      C = p.x * q.y - q.x * p.y
      return A, B, -C
    l1 = lineParams(self.p, self.q)
    l2 = lineParams(p, q)
    det = l1[0] * l2[1] - l1[1] * l2[0]
    detX = l2[1] * l1[2] - l2[2] * l1[1]
    detY = l1[0] * l2[2] - l1[2] * l2[0]
    if det == 0:
      return None
    return Point(detX / det, detY / det)


def gaussian(x: np.ndarray, amplitude: float, mean: float,
             stddev: float) -> np.ndarray:
  '''!@brief Gaussian function

  @param x
  @param amplitude peak value
  @param mean
  @param stddev
  @return np.ndarray gaussian calculated at x values
  '''
  return amplitude * np.exp(-((x - mean) / stddev) **
                            2 / 2) / (stddev * np.sqrt(2 * np.pi))

def gaussianMix(x: np.ndarray, components: list[float]) -> np.ndarray:
  '''!@brief Gaussian function

  @param x
  @return list of components [amplitude: float, mean: float, stddev: float]
  @return np.ndarray gaussian calculated at x values
  '''
  y = np.array([gaussian(x, *components[i]) for i in range(len(components))])
  return np.dot(y.T, [1] * len(components))

def gaussianMixCenter(components: list[float]) -> float:
  '''!@brief Compute the center of a gaussian mix distribution

  Returns exact results for n=1. Approximate for n>1 cause math is hard

  @param list of components [amplitude: float, mean: float, stddev: float]
  @return float Center of gaussian where peak occurs
  '''
  if len(components) == 1:
    return components[0][1]
  xMin = min([c[1] for c in components])
  xMax = max([c[1] for c in components])
  x = np.linspace(xMin, xMax, 1000)
  y = gaussianMix(x, components)
  x = x[np.argmax(y)]
  x = np.linspace(x - (xMax - xMin) / 1000, x + (xMax - xMin) / 1000, 1000)
  y = gaussianMix(x, components)
  return x[np.argmax(y)]


def fitGaussian(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
  '''!@brief Fit a gaussian function to data

  @param x X values
  @param y Y values
  @return amplitude: float, mean: float, stddev: float
  '''
  sumY = np.max(y)
  y = y / sumY

  guessMean = np.average(x, weights=y)
  guessStdDev = (x[-1] - x[0]) / 10
  opt, _ = optimize.curve_fit(
      gaussian, x, y, p0=[
          np.max(y), guessMean, guessStdDev])
  opt[0] = opt[0] * sumY
  return opt

def binLinear(values: Iterable, binCount: int = 100) -> tuple[list, list]:
  '''!@brief Bin values with equal width bins

  @param values Values iterable over
  @param binCount Number of equal width bins
  @return tuple[list, list] (bins: list, counts: list)
  '''
  minValue = min(values)
  maxValue = max(values)
  edges = np.linspace(minValue, maxValue, binCount + 1)
  counts, _ = np.histogram(values, edges)
  bins = edges[:-1] + (edges[1] - edges[0]) / 2
  return bins, counts

def binExact(values: Iterable) -> tuple[list, list]:
  '''!@brief Bin values with exact indices

  @param values Values iterable over
  @return tuple[list, list] (bins: list, counts: list)
  '''
  counts = {}
  for e in values:
    if e in counts:
      counts[e] += 1
    else:
      counts[e] = 1
  bins = sorted(counts.keys())
  counts = [counts[b] for b in bins]
  return bins, counts

def fitGaussianMix(
  x: list, nMax: int = 10, tol: float = 1e-3) -> list[tuple[float, float, float]]:
  '''!@brief Fit a mixture of gaussian curves, returning the best combination

  @param x Samples
  @param nMax Maximum number of components
  @param tol Tolerance of curve fitting (normalized units)
  @return list of components sorted by amplitude [amplitude: float, mean: float, stddev: float]
  '''
  xMax = np.average(x)
  x = np.array(x).reshape(-1, 1) / xMax

  models = []
  for i in range(nMax):
    models.append(GaussianMixture(i + 1, tol=tol).fit(x))

  aic = [m.aic(x) for m in models]
  mBest = models[np.argmin(aic)]

  # xRange = np.linspace(np.amin(x), np.amax(x), 1000)
  # logprob = mBest.score_samples(xRange.reshape(-1, 1))
  # responsibilities = mBest.predict_proba(xRange.reshape(-1, 1))
  # pdf = np.exp(logprob)
  # pdf_individual = responsibilities * pdf[:, np.newaxis]

  # pyplot.hist(x * xMax, 30, density=True, histtype='stepfilled', alpha=0.4)
  # pyplot.plot(xRange * xMax, pdf / xMax, '-k')
  # pyplot.plot(xRange * xMax, pdf_individual / xMax, '--k')
  # pyplot.show()

  components = []
  for i in range(mBest.n_components):
    amplitude = mBest.weights_[i]
    mean = mBest.means_[i][0] * xMax
    stddev = abs(np.sqrt(mBest.covariances_[i][0][0]) * xMax)
    components.append([amplitude, mean, stddev])
  return sorted(components, key=lambda c: -c[0])

def layerNumpyImageRGBA(below: np.ndarray, above: np.ndarray) -> np.ndarray:
  '''!@brief Layer a RGBA image on top of another using alpha compositing

  @param below Image on bottom of stack
  @param above Image on top of stack
  @return np.ndarray Alpha composited image
  '''
  if below.shape != above.shape:
    raise Exception(
      f'Cannot layer images of different shapes {below.shape} vs. {above.shape}')
  if below.shape[2] != 4 or above.shape[2] != 4:
    raise Exception(f'Image is not RGBA')

  alphaA = above[:, :, 3]
  alphaB = below[:, :, 3]
  alphaOut = alphaA + np.multiply(alphaB, 1 - alphaA)
  out = np.zeros(below.shape, dtype=below.dtype)
  out[:, :, 0] = np.divide(np.multiply(above[:, :, 0], alphaA) +
                           np.multiply(np.multiply(below[:, :, 0], alphaB), 1 -
                                       alphaA), alphaOut, where=alphaOut != 0)
  out[:, :, 1] = np.divide(np.multiply(above[:, :, 1], alphaA) +
                           np.multiply(np.multiply(below[:, :, 1], alphaB), 1 -
                                       alphaA), alphaOut, where=alphaOut != 0)
  out[:, :, 2] = np.divide(np.multiply(above[:, :, 2], alphaA) +
                           np.multiply(np.multiply(below[:, :, 2], alphaB), 1 -
                                       alphaA), alphaOut, where=alphaOut != 0)
  out[:, :, 3] = alphaOut
  return out
