"""Clock data recovery helper functions, see measurement.eyediagram.cdr
Fallback version for no cython
"""

import numpy as np


def _calculate_tie_scores(data_edges: np.ndarray,
                          t_span: np.ndarray) -> np.ndarray:
  """Calculate the success of a range of t_sym on TIEs

  Args:
    data_edges: List of data edges in time domain
    t_span: Range of t_sym to test

  Returns:
    Sorted scores, best first. [[t_sym0, score0],[t_sym1, score1],...]
  """
  scores = []
  for t in t_span:
    ties = np.mod(data_edges + t / 2, t)
    score = (np.abs(np.diff(ties)) > t / 2).sum()
    if score < 1:
      bits = np.floor_divide(data_edges + t / 2, t)
      ssxm, ssxym, _, _ = np.cov(bits, ties, bias=1).flat
      slope = ssxym / ssxm
      score = -(1 - np.abs(slope))
    scores.append((t, score))
  return np.array(sorted(scores, key=lambda s: s[1]))


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
    ValueError is t_sym is not given or t_min & t_max is not given
    ArithmeticError if a symbol period cannot be determined
  """
  if t_sym is not None:
    t_min = t_sym * 0.9
    t_max = t_sym * 1.1
  elif t_min is None or t_max is None:
    raise ValueError("minimize_tie_disjoints requires t_sym or t_min & t_max")
  edges = data_edges - data_edges[0]

  n_comp = int(1e6)

  # Hope first round works
  t_span = np.linspace(t_min, t_max, max(20, n_comp // len(edges)))
  scores = _calculate_tie_scores(edges, t_span)
  if scores[0][1] < tol:
    return scores[0][0]

  # from matplotlib import pyplot
  # x = scores[:, 0]
  # y = scores[:, 1]
  # pyplot.scatter(x, y)
  # pyplot.axvline(x=t_min, color="g")
  # pyplot.axvline(x=t_max, color="g")
  # pyplot.show()
  # return

  if max_iter < 1:
    raise ArithmeticError("Failed to find any t_sym with few disjoints")

  # Do a finer step, fewer edges to catch a zero
  for n in [100000, 10000, 1000, 100]:
    edges_short = edges[:n]
    t_span = np.linspace(t_min, t_max, max(20, n_comp // n))
    scores = _calculate_tie_scores(edges_short, t_span).T
    lower = scores[:, scores[1] < 1]
    if lower.shape[1] != 0:
      break

  if lower.shape[1] == 0:
    raise ArithmeticError("Failed to find any t_sym with few disjoints")

  t_step = t_span[1] - t_span[0]
  t_min = max(t_min, lower[0].min() - t_step)
  t_max = min(t_max, lower[0].max() + t_step)

  # from matplotlib import pyplot
  # x = scores[0]
  # y = scores[1]
  # pyplot.scatter(x, y)
  # x = lower[0]
  # y = lower[1]
  # pyplot.scatter(x, y)
  # # pyplot.axvline(x=t_sym, color="r")
  # pyplot.axvline(x=t_min, color="g")
  # pyplot.axvline(x=t_max, color="g")
  # pyplot.title(f"n {n}")
  # pyplot.show()

  return minimize_tie_disjoints(data_edges,
                                t_min=t_min,
                                t_max=t_max,
                                tol=tol,
                                max_iter=max_iter - 1)


def detrend_ties(data_edges: np.ndarray, t_sym: float) -> float:
  """Remove linear drift from TIEs (wrong clock period)

  Iterates the clock drifts less than tol over the duration

  Args:
    data_edges: List of data edges in time domain
    t_sym: Initial clock period

  Returns:
    Most probable clock period
  """
  n = len(data_edges)

  bits, ties = np.divmod(data_edges - data_edges[0] + t_sym / 2, t_sym)
  ties = ties - t_sym / 2

  # Fix disjoints, assume jumps over 1/2 are going the other way
  ties_diff = np.diff(ties)
  disjoints = np.where(np.abs(ties_diff) > t_sym / 2)[0]
  for disjoint in disjoints:
    offset = np.append([0] * (disjoint + 1), [1] * (n - disjoint - 1))
    if ties_diff[disjoint] > 0:
      offset = -offset
    ties += offset * t_sym
    bits += -offset

  # from matplotlib import pyplot
  # pyplot.plot(bits, ties)
  # pyplot.show()

  ssxm, ssxym, _, _ = np.cov(bits, ties, bias=1).flat
  slope = ssxym / ssxm
  t_sym = t_sym + slope
  return t_sym
