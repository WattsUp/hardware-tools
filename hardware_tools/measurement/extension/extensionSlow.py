import numpy as np

def getCrossingSlow(returnAxis: list, searchAxis: list,
                    i: int, value: float, silent: bool = False) -> float:
  '''!@brief Get crossing value in the returnAxis by searching and interpolating searchAxis

  Example:
    returnAxis = [0,   1,   2,  3,   4]
    searchAxis = [10, 20, -10, 50, -50]
    value = 0
    if i = 1, return None or raise Excpetion
    if i = 2, return 1.66667
    if i = 3, return 2.16667
    if i = 4, return 3.5

  @param returnAxis Data axis to return value for
  @param searchAxis Data axis to find and interpolate value
  @param i Begining index to look at, back up until found
  @param value Value to find and interpolate in searchAxis
  @param silent True will return None if not found, False will raise Exception if not found
  '''

  # Back up
  while (searchAxis[i] > value) == (searchAxis[i - 1] > value) and i > 0:
    i -= 1

  if i < 1:
    if silent:
      return None
    errorStr = f'Cannot get crossing of index 0'
    errorStr += f'  returnAxis: {returnAxis}'
    errorStr += f'  searchAxis: {searchAxis}'
    errorStr += f'  value:      {value}'
    raise Exception(errorStr)

  return (returnAxis[i] - returnAxis[i - 1]) / (searchAxis[i] - searchAxis[i - 1]) * \
      (value - searchAxis[i - 1]) + returnAxis[i - 1]

def getEdgesSlow(t: list, y: list, yRise: float,
                 yHalf: float, yFall: float) -> tuple[list, list]:
  '''!@brief Collect rising and falling edges

  @param t Waveform time array [t0, t1,..., tn]
  @param y Waveform data array [y0, y1,..., yn]
  @param yRise Rising threshold
  @param yHalf Interpolated edge value
  @param yFall Falling threshold
  @return tuple(list, list) tuple of rising edges, falling edges
  '''
  edgesRising = []
  edgesFalling = []
  stateLow = y[0] < yFall

  for i in range(1, len(y)):
    if stateLow:
      if y[i] > yRise:
        stateLow = False
        # interpolate 50% crossing
        edgesRising.append(getCrossingSlow(t, y, i, yHalf))
    else:
      if y[i] < yFall:
        stateLow = True
        # interpolate 50% crossing
        edgesFalling.append(getCrossingSlow(t, y, i, yHalf))

  return (edgesRising, edgesFalling)
