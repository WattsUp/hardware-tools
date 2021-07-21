
import numpy as np

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