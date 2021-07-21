def bres_segment_count_slow(x0, y0, x1, y1, grid):
  '''Bresenham's algorithm.

  The value of grid[x,y] is incremented for each x,y
  in the line from (x0,y0) up to but not including (x1, y1).
  '''

  nrows, ncols = grid.shape

  dx = abs(x1 - x0)
  dy = abs(y1 - y0)

  sx = 0
  if x0 < x1:
    sx = 1
  else:
    sx = -1
  sy = 0
  if y0 < y1:
    sy = 1
  else:
    sy = -1

  err = dx - dy

  while True:
    # Note: this test is moved before setting
    # the value, so we don't set the last point.
    if x0 == x1 and y0 == y1:
      break

    if 0 <= x0 < nrows and 0 <= y0 < ncols:
      grid[x0, y0] += 1

    e2 = 2 * err
    if e2 > -dy:
      err -= dy
      x0 += sx
    if e2 < dx:
      err += dx
      y0 += sy

def bres_curve_count_slow(x, y, grid):
  for k in range(x.size - 1):
    x0 = x[k]
    y0 = y[k]
    x1 = x[k + 1]
    y1 = y[k + 1]
    bres_segment_count_slow(x0, y0, x1, y1, grid)
