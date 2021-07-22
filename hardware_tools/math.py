import datetime
import numpy as np
from scipy import optimize
from typing import Iterable, Union

def interpolate(x, y, xNew) -> np.ndarray:
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

def gaussian(x: np.ndarray, amplitude: float, mean: float,
             stddev: float) -> np.ndarray:
  '''!@brief Gaussian function

  @param x
  @param amplitude peak value, not normalized with stddev
  @param mean
  @param stddev
  @return np.ndarray gaussian calculated at x values
  '''
  return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

def fitGaussian(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
  '''!@brief Fit a gaussian function to data

  @param x X values
  @param y Y values
  @return amplitude: float, mean: float, stddev: float
  '''
  sumY = np.sum(y)
  y = y / sumY

  guessMean = np.average(x, weights=y)
  guessStdDev = (x[-1] - x[0]) / 10
  opt, _ = optimize.curve_fit(
      gaussian, x, y, p0=[
          np.max(y), guessMean, guessStdDev])
  opt[0] = opt[0] * sumY
  return opt


def binExact(values: Iterable,
             split: bool = True) -> Union[tuple[list, list], dict]:
  '''!@brief Bin values with exact indices

  @param values Values iterable over
  @param split True will sort bins and return bins[list], counts[list]. False will return dictionary with value indices keys
  @return tuple[list, list] or dict See split
  '''
  counts = {}
  for e in values:
    if e in counts:
      counts[e] += 1
    else:
      counts[e] = 1
  if not split:
    return counts
  bins = sorted(counts.keys())
  counts = [counts[b] for b in bins]
  return bins, counts
