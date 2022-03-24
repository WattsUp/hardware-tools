"""Clock data recovery helper functions definitions, see measurement.eyediagram.cdr
"""

cimport numpy as np

cdef np.ndarray[np.float64_t, ndim=2] _calculate_tie_scores_c(
    np.ndarray[np.float64_t, ndim=1] data_edges,
    np.float64_t t_min,
    np.float64_t t_max,
    int n)

cdef np.float64_t minimize_tie_disjoints_c(np.ndarray[np.float64_t, ndim=1] data_edges,
                                  np.float64_t t_min,
                                  np.float64_t t_max,
                                  int tol,
                                  int max_iter)

cdef np.float64_t detrend_ties_c(np.ndarray[np.float64_t, ndim=1] data_edges,
                          np.float64_t t_sym)
