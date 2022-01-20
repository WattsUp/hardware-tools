"""Line interpolation using Bresenham's algorithm

See http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
"""

import numpy as np


def draw_segment(x1: int, y1: int, x2: int, y2: int, grid: np.ndarray) -> None:
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
  n_x, n_y = grid.shape

  dx = abs(x2 - x1)
  dy = abs(y2 - y1)

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
    # Note: this test is moved before setting
    # the value, so we don't set the last point.
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


def draw(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> None:
  """Linearly interpolate a series of points onto a grid

  Args:
    x: Series of x values
    y: Series of y values
    grid: Grid to interpolate onto
  """
  for i in range(x.size - 1):
    x1 = x[i]
    y1 = y[i]
    x2 = x[i + 1]
    y2 = y[i + 1]
    draw_segment(x1, y1, x2, y2, grid)
  if (0 <= x2 < grid.shape[0]) and (0 <= y2 < grid.shape[1]):
    grid[x2, y2] += 1
