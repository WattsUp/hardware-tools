from .extensionSlow import getEdgesSlow, getCrossingSlow, getHitsSlow
getEdges = getEdgesSlow
getCrossing = getCrossingSlow
getHits = getHitsSlow

import numpy as np

try:
  from .extension import getEdgesFast, getCrossingFast, getHitsFast
  getEdges = getEdgesFast
  getCrossing = getCrossingFast
  getHits = getHitsFast
except ImportError:
  print('The cython version of the measurement extension is not available')

def getEdgesNumpy(w: np.ndarray, yRise: float,
                  yHalf: float, yFall: float) -> tuple[list, list]:
  '''!@brief Collect rising and falling edges

  @param w Waveform time/data array [[t0, t1,..., tn], [y0, y1,..., yn]]
  @param yRise Rising threshold
  @param yHalf Interpolated edge value
  @param yFall Falling threshold
  @return tuple(list, list) tuple of rising edges, falling edges
  '''
  return getEdges(w[0].tolist(), w[1].tolist(), yRise, yHalf, yFall)
