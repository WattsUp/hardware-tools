"""Line interpolation using Bresenham's algorithm

See http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
"""

cimport numpy as np
cimport cython


@cython.boundscheck(False)
cdef void draw_segment(int x1, int y1,
                      int x2, int y2,
                      np.ndarray[np.int32_t, ndim=2] grid):
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

  cdef unsigned n_x, n_y
  cdef int e2, sx, sy, err
  cdef int dx, dy

  n_x = grid.shape[0]
  n_y = grid.shape[1]

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

    if (x1 < n_x) and (y1 < n_y) and (x1 >= 0) and (y1 >= 0):
      grid[x1, y1] += 1

    e2 = 2 * err
    if e2 > -dy:
      err -= dy
      x1 += sx
    if e2 < dx:
      err += dx
      y1 += sy


def draw(np.ndarray[np.int32_t, ndim=1] x,
         np.ndarray[np.int32_t, ndim=1] y,
         np.ndarray[np.int32_t, ndim=2] grid):
  cdef unsigned i
  cdef int x1, y1, x2, y2

  for i in range(len(x)-1):
    x1 = x[i]
    y1 = y[i]
    x2 = x[i + 1]
    y2 = y[i + 1]
    draw_segment(x1, y1, x2, y2, grid)
  if (x2 < grid.shape[0]) and (y2 < grid.shape[1]) and (x2 >= 0) and (y2 >= 0):
    grid[x2, y2] += 1
