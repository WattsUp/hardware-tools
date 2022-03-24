"""Lines helper functions definitions, see math.lines
"""

cimport numpy as np

cdef class Point:
  cdef np.float64_t x
  cdef np.float64_t y

cdef class LineParams:
  cdef np.float64_t d2
  cdef np.float64_t d1
  cdef np.float64_t det

cdef void _draw_segment_c(np.int32_t x1,
                        np.int32_t y1,
                        np.int32_t x2,
                        np.int32_t y2,
                        np.ndarray[np.int32_t, ndim=2] grid,
                        Py_ssize_t n_x,
                        Py_ssize_t n_y)

cdef void draw_c(np.ndarray[np.int32_t, ndim=1] x,
         np.ndarray[np.int32_t, ndim=1] y,
         np.ndarray[np.int32_t, ndim=2] grid)

cdef void draw_points_c(np.ndarray[np.int32_t, ndim=1] x,
         np.ndarray[np.int32_t, ndim=1] y,
         np.ndarray[np.int32_t, ndim=2] grid)

cdef tuple crossing_c(list axis_return,
                        list axis_search,
                        Py_ssize_t i,
                        np.float64_t value,
                        bint step_forward,
                        Py_ssize_t n)

cdef np.float64_t crossing_c_v(list axis_return,
                        list axis_search,
                        Py_ssize_t i,
                        np.float64_t value,
                        bint step_forward,
                        Py_ssize_t n)

cdef np.float64_t crossing_c_v_np(np.ndarray[np.float64_t, ndim=1] axis_return,
                        np.ndarray[np.float64_t, ndim=1] axis_search,
                        Py_ssize_t i,
                        np.float64_t value,
                        bint step_forward,
                        Py_ssize_t n)

cdef tuple edges_c(
      list t,
      list y,
      np.float64_t y_rise,
      np.float64_t y_half,
      np.float64_t y_fall)

cdef tuple edges_np_c(
      np.ndarray[np.float64_t, ndim=1] t,
      np.ndarray[np.float64_t, ndim=1] y,
      np.float64_t y_rise,
      np.float64_t y_half,
      np.float64_t y_fall)

cdef LineParams line_params_c(np.float64_t a1, np.float64_t a2, np.float64_t b1, np.float64_t b2)

cdef Point intersection_c(np.float64_t p1,
                 np.float64_t p2,
                 np.float64_t q1,
                 np.float64_t q2,
                 np.float64_t r1,
                 np.float64_t r2,
                 np.float64_t s1,
                 np.float64_t s2,
                 bint segments)

cdef list hits_c(list t, list y, list paths)

cdef np.ndarray[np.float64_t, ndim=2] hits_np_c(np.ndarray[np.float64_t, ndim=1] t,
                        np.ndarray[np.float64_t, ndim=1] y,
                        list paths)

cdef bint is_hitting_c(list t, list y, list paths)

cdef bint is_hitting_np_c(np.ndarray[np.float64_t, ndim=1] t,
                          np.ndarray[np.float64_t, ndim=1] y,
                          list paths)
