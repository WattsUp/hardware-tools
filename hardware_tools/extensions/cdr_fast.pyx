"""Clock data recovery helper functions, see measurement.eyediagram.cdr
"""

import numpy as np

cimport numpy as np
cimport cython

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=2] _calculate_tie_scores_c(
    np.ndarray[np.float64_t, ndim=1] data_edges,
    np.float64_t t_min,
    np.float64_t t_max,
    int n):
  """Calculate the success of a range of t_sym on TIEs

  Args:
    data_edges: List of data edges in time domain
    t_min: Minimum t_sym to test
    t_max: Maximum t_sym to test
    n: Number of values to test

  Returns:
    Sorted scores, best first. [[t_sym0, score0],[t_sym1, score1],...]
  """
  cdef np.ndarray[np.float64_t, ndim=2] scores = np.zeros((n, 2))
  cdef Py_ssize_t i, i_score
  cdef np.float64_t t = t_min
  cdef np.float64_t t_step = (t_max - t_min) / (n - 1)
  cdef np.float64_t t2, score, ssxm, ssxym, prev_edge, edge
  cdef np.ndarray[np.float64_t, ndim=1] ties

  for i_score in range(n):
    t = t_min + i_score * t_step
    t2 = t / 2
    score = 0
    prev_edge = (data_edges[0] + t2) % t
    for i in range(1, data_edges.shape[0]):
      edge = (data_edges[i] + t2) % t
      prev_edge = edge - prev_edge
      if prev_edge > t2 or prev_edge < -t2:
        score += 1
      prev_edge = edge


    # ties = (data_edges + t2) % t
    # score = (np.abs(np.diff(ties)) > t2).sum()
    if score < 1:
      bits, ties = np.divmod(data_edges + t2, t)
      ssxm, ssxym, _, _ = np.cov(bits, ties, bias=1).flat
      score = -(1 - np.abs(ssxym / ssxm))
    scores[i_score, 0] = t
    scores[i_score, 1] = score
  return scores[scores[:, 1].argsort()]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float64_t minimize_tie_disjoints_c(np.ndarray[np.float64_t, ndim=1] data_edges,
                                  np.float64_t t_min,
                                  np.float64_t t_max,
                                  int tol,
                                  int max_iter):
  """Determine most likely clock period from minimizing time interval errors

  Iterates until there are tol or fewer disjoints in the TIEs

  Args:
    data_edges: List of data edges in time domain
    t_min: Minimum clock period
    t_max: Maximum clock period
    tol: Maximum number of disjoints willing to fix
    max_iter: Maximum iterations to run

  Returns:
    Most probable clock period

  Raises:
    ArithmeticError if a symbol period cannot be determined
  """
  cdef np.ndarray[np.float64_t, ndim=2] scores
  cdef np.ndarray[np.float64_t, ndim=1] edges = data_edges - data_edges[0]

  cdef int n_comp = int(1e6)

  # Hope first round works
  cdef int n = max(100000, len(edges))
  scores = _calculate_tie_scores_c(edges, t_min, t_max, (n_comp // n))
  if scores[0][1] < tol:
    return scores[0][0]

  if max_iter < 1:
    return np.nan

  # Do a finer step, fewer edges to catch a zero
  n = 10000
  cdef np.ndarray[np.float64_t, ndim=1] edges_short = edges[:n]
  scores = _calculate_tie_scores_c(edges_short, t_min, t_max, (n_comp // n)).T
  cdef np.ndarray[np.float64_t, ndim=2] lower = scores[:, scores[1] < 1]

  if lower.shape[1] == 0:
    # Do a finer step, fewer edges to catch a zero
    n = 1000
    edges_short = edges[:n]
    scores = _calculate_tie_scores_c(edges_short, t_min, t_max, (n_comp // n)).T
    lower = scores[:, scores[1] < 1]

  if lower.shape[1] == 0:
    # Do a finer step, fewer edges to catch a zero
    n = 100
    edges_short = edges[:n]
    scores = _calculate_tie_scores_c(edges_short, t_min, t_max, (n_comp // n)).T
    lower = scores[:, scores[1] < 1]

  if lower.shape[1] == 0:
    return np.nan

  cdef np.float64_t t_step = (t_max - t_min) / ((n_comp // n) - 1)
  t_min = max(t_min, lower[0].min() - t_step)
  t_max = min(t_max, lower[0].max() + t_step)

  return minimize_tie_disjoints_c(data_edges, t_min, t_max, tol, max_iter - 1)


def minimize_tie_disjoints(data_edges: np.ndarray,
                           t_sym: float = None,
                           t_min: float = None,
                           t_max: float = None,
                           tol: int = 1,
                           max_iter: int = 10) -> float:
  """Determine most likely clock period from minimizing time interval errors

  Iterates until there are tol or fewer disjoints in the TIEs

  Args:
    data_edges: List of data edges in time domain
    t_sym: Rough clock period, t_min = t_sym * 0.9, t_max = t_sym * 1.1
    t_min: Minimum clock period, None requires t_sym instead
    t_max: Maximum clock period, None requires t_sym instead
    tol: Maximum number of disjoints willing to fix
    max_iter: Maximum iterations to run

  Returns:
    Most probable clock period

  Raises:
    ArithmeticError if a symbol period cannot be determined
  """
  if t_sym is not None:
    t_min = t_sym * 0.9
    t_max = t_sym * 1.1
  elif t_min is None or t_max is None:
    raise ValueError("minimize_tie_disjoints requires t_sym or t_min & t_max")
  t_sym = minimize_tie_disjoints_c(data_edges, t_min, t_max, tol, max_iter)
  if np.isnan(t_sym):
    raise ArithmeticError("Failed to find any t_sym with few disjoints")
  return t_sym


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float64_t detrend_ties_c(np.ndarray[np.float64_t, ndim=1] data_edges,
                          np.float64_t t_sym):
  """Remove linear drift from TIEs (wrong clock period)

  Iterates the clock drifts less than tol over the duration

  Args:
    data_edges: List of data edges in time domain
    t_sym: Initial clock period

  Returns:
    Most probable clock period
  """
  cdef Py_ssize_t n = data_edges.shape[0]
  cdef Py_ssize_t i
  cdef np.float64_t t_start = data_edges[0]
  cdef np.float64_t t2 = t_sym / 2
  cdef np.float64_t bit_offset = 0
  cdef np.float64_t tie_offset = 0
  cdef np.float64_t edge = 0
  cdef np.float64_t diff = 0
  cdef np.float64_t prev_tie = 0

  cdef np.ndarray[np.float64_t, ndim=1] bits = np.zeros(n)
  cdef np.ndarray[np.float64_t, ndim=1] ties = np.zeros(n)

  for i in range(1, n):
    edge = data_edges[i] - t_start
    bits[i] = (edge + t2) // t_sym
    ties[i] = edge - bits[i] * t_sym
    diff = ties[i] - prev_tie
    prev_tie = ties[i]
    
    if diff < -t2:
      tie_offset += t_sym
      bit_offset += -1
    elif diff > t2:
      tie_offset += -t_sym
      bit_offset += 1

    bits[i] += bit_offset
    ties[i] += tie_offset

  ssxm, ssxym, _, _ = np.cov(bits, ties, bias=1).flat
  slope = ssxym / ssxm
  t_sym = t_sym + slope

  return t_sym


def detrend_ties(data_edges: np.ndarray,
                 t_sym: float) -> float:
  """Remove linear drift from TIEs (wrong clock period)

  Iterates the clock drifts less than tol over the duration

  Args:
    data_edges: List of data edges in time domain
    t_sym: Initial clock period

  Returns:
    Most probable clock period
  """
  return detrend_ties_c(data_edges, t_sym)
