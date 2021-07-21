# Source: https://scipy.github.io/old-wiki/pages/Cookbook/EyeDiagram.html

from .brescountSlow import bres_curve_count_slow
counter = bres_curve_count_slow

try:
  from .brescount import bres_curve_count
  counter = bres_curve_count
except ImportError:
  print('The cython version of the curve counter is not available')
  useFastBresCount = False
