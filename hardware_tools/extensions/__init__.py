"""Extensions written in cython to increase speed
"""

try:
  from hardware_tools.extensions import bresenham_fast as bresenham
except ImportError:
  print("The cython version of bresenham is not available")
  from hardware_tools.extensions import bresenham_slow as bresenham

try:
  from hardware_tools.extensions import cdr_fast as cdr
except ImportError:
  print("The cython version of CDR is not available")
  from hardware_tools.extensions import cdr_slow as cdr

try:
  from hardware_tools.extensions import edges_fast as edges
except ImportError:
  print("The cython version of edges is not available")
  from hardware_tools.extensions import edges_slow as edges

try:
  from hardware_tools.extensions import intersections_fast as intersections
except ImportError:
  print("The cython version of intersections is not available")
  from hardware_tools.extensions import intersections_slow as intersections
