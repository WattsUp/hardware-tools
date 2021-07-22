from .extensionSlow import getEdgesSlow, getCrossingSlow
getEdges = getEdgesSlow
getCrossing = getCrossingSlow

try:
  from .extension import getEdgesFast, getCrossingFast
  getEdges = getEdgesFast
  getCrossing = getCrossingFast
except ImportError:
  print('The cython version of the measurement extension is not available')
