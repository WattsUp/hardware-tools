from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext = Extension('extension', ['extension.pyx'],
                include_dirs=[numpy.get_include()])

if __name__ == '__main__':
  setup(ext_modules=[ext],
        cmdclass={'build_ext': build_ext})
