"""Extensions written in cython to increase speed
"""

try:
  from hardware_tools.extensions import bresenham_fast as bresenham
except ImportError:
  print("The cython version of bresenham is not available")
  from hardware_tools.extensions import bresenham_slow as bresenham
