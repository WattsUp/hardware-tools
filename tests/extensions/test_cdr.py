"""Test module hardware_tools.extension.cdr
"""

import time

import numpy as np

from hardware_tools.extensions import cdr_slow, cdr_fast
from hardware_tools.signal import clock

from tests import base


class TestCDR(base.TestBase):
  """Test CDR methods
  """

  def _generate_edges(self, t_sym, n):
    t_rj = 0.002 * t_sym
    t_uj = 0.002 * t_sym
    t_sj = 0.002 * t_sym
    f_sj = 0.01 / t_sym
    dcd = 0.05
    edges = clock.edges(t_sym=t_sym,
                        n=n,
                        t_rj=t_rj,
                        t_uj=t_uj,
                        t_sj=t_sj,
                        f_sj=f_sj,
                        dcd=dcd)

    # Offset the edges to a random start
    t_start = -t_sym * self._RNG.uniform(-20, 20)

    # Remove edges every so often
    every = self._RNG.integers(3, 7)
    edges_a = edges[::every]
    edges_b = edges[::every + 7]
    edges = np.unique(np.append(edges_a, edges_b)) + t_start

    # Add glitches
    glitches = edges[::1000] + 0.2 * t_sym
    edges = np.unique(np.append(edges, glitches))
    return edges

  def _test_minimize_tie_disjoints(self, module):
    t_sym = 0.5e-9
    n = 10e3
    edges = self._generate_edges(t_sym, n)

    tol = 10
    result = module.minimize_tie_disjoints(edges, t_sym=t_sym, tol=tol)
    self.assertEqualWithinError(t_sym, result, 0.01)
    ties = np.mod(edges - edges[0] + result / 2, result)
    disjoints = (np.abs(np.diff(ties)) > result / 2).sum()
    self.assertLessEqual(disjoints, tol)

    self.assertRaises(ValueError, module.minimize_tie_disjoints, edges)

    self.assertRaises(ArithmeticError,
                      module.minimize_tie_disjoints,
                      edges,
                      t_min=t_sym * 1.311,
                      t_max=t_sym * 1.312,
                      max_iter=3)

    t_sj = 0.6 * t_sym
    f_sj = 2 / (n * t_sym)
    edges = clock.edges(t_sym=t_sym, n=n, t_sj=t_sj, f_sj=f_sj)
    self.assertRaises(ArithmeticError,
                      module.minimize_tie_disjoints,
                      edges,
                      t_sym=t_sym,
                      max_iter=0)

  def test_minimize_tie_disjoints(self):
    self._test_minimize_tie_disjoints(cdr_slow)
    self._test_minimize_tie_disjoints(cdr_fast)

    # Validate fast is actually faster
    t_sym = 0.5e-9
    n = 10e3
    edges = self._generate_edges(t_sym, n)

    tol = 10

    start = time.perf_counter()
    result_slow = cdr_slow.minimize_tie_disjoints(edges, t_sym=t_sym, tol=tol)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = cdr_fast.minimize_tie_disjoints(edges, t_sym=t_sym, tol=tol)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertEqualWithinError(result_slow, result_fast, 1e-15)
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_detrend_ties(self, module):
    t_sym = 0.5e-9
    n = 10e3
    edges = self._generate_edges(t_sym, n)

    period_tol = 1e-6
    result = module.detrend_ties(edges, t_sym=t_sym * (1 + 100e-6))
    self.assertEqualWithinError(t_sym, result, period_tol)

  def test_detrend_ties(self):
    self._test_detrend_ties(cdr_slow)
    self._test_detrend_ties(cdr_fast)

    # Validate fast is actually faster
    t_sym = 0.5e-9
    n = 10e3
    edges = self._generate_edges(t_sym, n)

    start = time.perf_counter()
    result_slow = cdr_slow.detrend_ties(edges, t_sym=t_sym * (1 - 100e-6))
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = cdr_fast.detrend_ties(edges, t_sym=t_sym * (1 - 100e-6))
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertEqualWithinError(result_slow, result_fast, 1e-15)
    self.assertLess(elapsed_fast, elapsed_slow)
