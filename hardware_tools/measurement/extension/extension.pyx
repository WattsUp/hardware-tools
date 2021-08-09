import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
cdef tuple get_crossing_c(list returnAxis,
                        list searchAxis,
                        int i,
                        float value,
                        bint stepForward):
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
  @param stepForward True will iterate forward until crossing, False will iterate backwards
  @return tuple (interpolated value, index)
  '''

  # Back up
  while (searchAxis[i] > value) == (searchAxis[i - 1] > value) and i > 0:
    i += 1 if stepForward else -1

  if i < 1:
    return (np.nan, 0)

  v = (returnAxis[i] - returnAxis[i - 1]) / (searchAxis[i] - searchAxis[i - 1]) * \
        (value - searchAxis[i - 1]) + returnAxis[i - 1]
  return (v, i)

def getCrossingFast(list returnAxis,
                    list searchAxis,
                    int i,
                    float value,
                    bint stepForward=False) -> tuple:
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
  @param stepForward True will iterate forward until crossing, False will iterate backwards
  @return tuple (interpolated value, index)
  '''
  return get_crossing_c(returnAxis, searchAxis, i, value, stepForward)

@cython.boundscheck(False)
cdef tuple get_edges_c(
      list t,
      list y, 
      float yRise,
      float yHalf,
      float yFall):
  '''!@brief Collect rising and falling edges

  @param t Waveform time array [t0, t1,..., tn]
  @param y Waveform data array [y0, y1,..., yn]
  @param yRise Rising threshold
  @param yHalf Interpolated edge value
  @param yFall Falling threshold
  @return tuple(list, list) tuple of rising edges, falling edges
  '''
  cdef list edgesRise = []
  cdef list edgesFall = []
  cdef bint stateLow = y[0] < yFall

  for i in range(1, len(y)):
    if stateLow:
      if y[i] > yRise:
        stateLow = False
        # interpolate 50% crossing
        edgesRise.append(get_crossing_c(t, y, i, yHalf, False)[0])
    else:
      if y[i] < yFall:
        stateLow = True
        # interpolate 50% crossing
        edgesFall.append(get_crossing_c(t, y, i, yHalf, False)[0])
  return (edgesRise, edgesFall)

def getEdgesFast(
      list t,
      list y, 
      float yRise,
      float yHalf,
      float yFall) -> tuple[float, int]:
  '''!@brief Collect rising and falling edges

  @param t Waveform time array [t0, t1,..., tn]
  @param y Waveform data array [y0, y1,..., yn]
  @param yRise Rising threshold
  @param yHalf Interpolated edge value
  @param yFall Falling threshold
  @return tuple(list, list) tuple of rising edges, falling edges
  '''
  return get_edges_c(t, y, yRise, yHalf, yFall)

@cython.boundscheck(False)
cdef tuple line_params_c(float a1, float a2, float b1, float b2):
  cdef float A = a2 - b2
  cdef float B = b1 - a1
  cdef float C = a1 * b2 - b1 * a2
  return A, B, -C

@cython.boundscheck(False)
cdef tuple get_intersection_c(float p1, float p2, float q1, float q2, float r1, float r2, float s1, float s2):
  '''!brief Get the intersection point between line segments pq and rs

  @param p1 Coordinate of first line segment start point
  @param p2 Coordinate of first line segment start point
  @param q1 Coordinate of first line segment end point
  @param q2 Coordinate of first line segment end point
  @param r1 Coordinate of second line segment start point
  @param r2 Coordinate of second line segment start point
  @param s1 Coordinate of second line segment end point
  @param s2 Coordinate of second line segment end point
  '''

  # Get intersection point of lines [not segments]
  cdef tuple l1 = line_params_c(p1, p2, q1, q2)
  cdef tuple l2 = line_params_c(r1, r2, s1, s2)
  cdef float det = l1[0] * l2[1] - l1[1] * l2[0]
  if det == 0:
    return None
  cdef float det1 = l2[1] * l1[2] - l2[2] * l1[1]
  cdef float det2 = l1[0] * l2[2] - l1[2] * l2[0]
  cdef float i1 = det1 / det
  cdef float i2 = det2 / det

  # Check if point lies on the segments via bounds checking
  if p1 < q1:
    if (i1 < p1) or (i1 > q1):
      return None
  elif p1 > q1:
    if (i1 > p1) or (i1 < q1):
      return None
  if p2 < q2:
    if (i2 < p2) or (i2 > q2):
      return None
  elif p2 > q2:
    if (i2 > p2) or (i2 < q2):
      return None

  if r1 < s1:
    if (i1 < r1) or (i1 > s1):
      return None
  elif r1 > s1:
    if (i1 > r1) or (i1 < s1):
      return None
  if r2 < s2:
    if (i2 < r2) or (i2 > s2):
      return None
  elif r2 > s2:
    if (i2 > r2) or (i2 < s2):
      return None

  return i1, i2

def getIntersectionFast(p1: float, p2: float, q1: float, q2: float, r1: float, r2: float, s1: float, s2: float) -> tuple:
  '''!brief Get the intersection point between line segments pq and rs

  @param p1 Coordinate of first line segment start point
  @param p2 Coordinate of first line segment start point
  @param q1 Coordinate of first line segment end point
  @param q2 Coordinate of first line segment end point
  @param r1 Coordinate of second line segment start point
  @param r2 Coordinate of second line segment start point
  @param s1 Coordinate of second line segment end point
  @param s2 Coordinate of second line segment end point
  '''

  return get_intersection_c(p1, p2, q1, q2, r1, r2, s1, s2)

@cython.boundscheck(False)
cdef list get_hits_c(list t, list y, list paths):
  '''!@brief Get instersections between waveform and mask lines

  @param t Waveform time array [t0, t1,..., tn]
  @param y Waveform data array [y0, y1,..., yn]
  @param lineSets list of path: list of points defining open path (n-1 segments)
  @return list[tuple] List of intersections points (t, y)
  '''
  cdef list intersections = []
  cdef list pathT
  cdef list pathY
  cdef float minT 
  cdef float maxT 
  cdef float minY
  cdef float maxY
  for path in paths:
    pathT = [p[0] for p in path]
    pathY = [p[1] for p in path]
    minT = min(pathT)
    maxT = max(pathT)
    minY = min(pathY)
    maxY = max(pathY)
    for i in range(1, len(t)):
      if (t[i] < minT) or (t[i - 1] > maxT):
        continue
      if y[i] > y[i - 1]:
        if (y[i] < minY) or (y[i - 1] > maxY):
          continue
      else:
        if (y[i] > maxY) or (y[i - 1] < minY):
          continue

      for ii in range(1, len(path)):
        intersection = get_intersection_c(t[i],
                                          y[i],
                                          t[i - 1],
                                          y[i - 1],
                                          pathT[ii],
                                          pathY[ii],
                                          pathT[ii - 1],
                                          pathY[ii - 1])
        if intersection is not None:
          intersections.append(intersection)
  return intersections

def getHitsFast(t: list, y: list, paths: list[list[tuple]]) -> list[tuple]:
  '''!@brief Get instersections between waveform and mask lines

  @param t Waveform time array [t0, t1,..., tn]
  @param y Waveform data array [y0, y1,..., yn]
  @param lineSets list of path: list of points defining open path (n-1 segments)
  @return list[tuple] List of intersections points (t, y)
  '''
  return get_hits_c(t, y, paths)

@cython.boundscheck(False)
cdef list is_hitting_c(list t, list y, list paths):
  '''!@brief Check for instersections between waveform and mask lines

  @param t Waveform time array [t0, t1,..., tn]
  @param y Waveform data array [y0, y1,..., yn]
  @param lineSets list of path: list of points defining open path (n-1 segments)
  @return bool True if any intersections occur, False otherwise
  '''
  cdef list pathT
  cdef list pathY
  cdef float minT 
  cdef float maxT 
  cdef float minY
  cdef float maxY
  for path in paths:
    pathT = [p[0] for p in path]
    pathY = [p[1] for p in path]
    minT = min(pathT)
    maxT = max(pathT)
    minY = min(pathY)
    maxY = max(pathY)
    for i in range(1, len(t)):
      if (t[i] < minT) or (t[i - 1] > maxT):
        continue
      if y[i] > y[i - 1]:
        if (y[i] < minY) or (y[i - 1] > maxY):
          continue
      else:
        if (y[i] > maxY) or (y[i - 1] < minY):
          continue

      for ii in range(1, len(path)):
        intersection = get_intersection_c(t[i],
                                          y[i],
                                          t[i - 1],
                                          y[i - 1],
                                          pathT[ii],
                                          pathY[ii],
                                          pathT[ii - 1],
                                          pathY[ii - 1])
        if intersection is not None:
          return True
  return False

def isHittingFast(t: list, y: list, paths: list[list[tuple]]) -> bool:
  '''!@brief Check for instersections between waveform and mask lines

  @param t Waveform time array [t0, t1,..., tn]
  @param y Waveform data array [y0, y1,..., yn]
  @param lineSets list of path: list of points defining open path (n-1 segments)
  @return bool True if any intersections occur, False otherwise
  '''
  return is_hitting_c(t, y, paths)