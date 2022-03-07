"""Line interpolation using Bresenham's algorithm

See http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
"""

cimport numpy as np
cimport cython

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void draw_segment_c(np.int32_t x1,
                        np.int32_t y1,
                        np.int32_t x2,
                        np.int32_t y2,
                        np.ndarray[np.int32_t, ndim=2] grid,
                        Py_ssize_t n_x,
                        Py_ssize_t n_y):
  """Interpolate a line segment onto a grid

  The value of grid[x,y] is incremented for each x,y in the line from (x0,y0) up
  to but not including (x1, y1).

  Args:
    x1: X-coordinate of first point
    y1: Y-coordinate of first point
    x2: X-coordinate of second point
    y2: Y-coordinate of second point
    grid: Grid to interpolate onto
  """
  cdef np.int32_t e2, sx, sy, err
  cdef np.int32_t dx, dy


  if x2 > x1:
    dx = x2 - x1
  else:
    dx = x1 - x2
  if y2 > y1:
    dy = y2 - y1
  else:
    dy = y1 - y2

  sx = 0
  if x1 < x2:
    sx = 1
  else:
    sx = -1
  sy = 0
  if y1 < y2:
    sy = 1
  else:
    sy = -1

  err = dx - dy

  while True:
    # Note: this test occurs before increment the
    # grid value, so we don't count the last point.
    if x1 == x2 and y1 == y2:
      break

    if (0 <= x1 < n_x) and (0 <= y1 < n_y):
      grid[x1, y1] += 1

    e2 = 2 * err
    if e2 > -dy:
      err -= dy
      x1 += sx
    if e2 < dx:
      err += dx
      y1 += sy


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void draw_c(np.ndarray[np.int32_t, ndim=1] x,
         np.ndarray[np.int32_t, ndim=1] y,
         np.ndarray[np.int32_t, ndim=2] grid):
  """Linearly interpolate a series of points onto a grid

  Args:
    x: Series of x values
    y: Series of y values
    grid: Grid to interpolate onto
  """
  cdef Py_ssize_t n = x.shape[0]
  cdef Py_ssize_t n_x = grid.shape[0]
  cdef Py_ssize_t n_y = grid.shape[1]
  cdef Py_ssize_t i
  cdef np.int32_t x1, y1, x2, y2

  for i in range(n - 1):
    x1 = x[i]
    y1 = y[i]
    x2 = x[i + 1]
    y2 = y[i + 1]
    draw_segment_c(x1, y1, x2, y2, grid, n_x, n_y)
  if (0 <= x2 < n_x) and (0 <= y2 < n_y):
    grid[x2, y2] += 1

def draw(np.ndarray x,
         np.ndarray y,
         np.ndarray grid) -> None:
  """Linearly interpolate a series of points onto a grid

  Args:
    x: Series of x values
    y: Series of y values
    grid: Grid to interpolate onto
  """
  draw_c(x, y, grid)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void draw_points_c(np.ndarray[np.int32_t, ndim=1] x,
         np.ndarray[np.int32_t, ndim=1] y,
         np.ndarray[np.int32_t, ndim=2] grid):
  """Plots a series of points onto a grid

  Args:
    x: Series of x values
    y: Series of y values
    grid: Grid to plot onto
  """
  cdef Py_ssize_t n = x.shape[0]
  cdef Py_ssize_t n_x = grid.shape[0]
  cdef Py_ssize_t n_y = grid.shape[1]
  cdef Py_ssize_t i
  cdef np.int32_t x1, y1

  for i in range(n):
    x1 = x[i]
    y1 = y[i]
    if (0 <= x1 < n_x) and (0 <= y1 < n_y):
      grid[x1, y1] += 1

def draw_points(np.ndarray x,
         np.ndarray y,
         np.ndarray grid) -> None:
  """Plots a series of points onto a grid

  Args:
    x: Series of x values
    y: Series of y values
    grid: Grid to plot onto
  """
  draw_points_c(x, y, grid)
