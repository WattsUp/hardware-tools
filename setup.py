import numpy
import os
import setuptools

from setuptools import setup, find_packages
import setuptools.command.develop
import setuptools.command.build_py

from tools import gitsemver

with open('README.md') as f:
  longDescription = f.read()

with open('requirements.txt') as f:
  required = f.read().splitlines()

version = gitsemver.getVersion()
with open('hardware_tools/version.py', 'w') as file:
  file.write(f'version = \'{version}\'\n')
  file.write(f'versionFull = \'{version.fullStr()}\'\n')

cwd = os.path.dirname(os.path.abspath(__file__))

try:
  from Cython.Build import cythonize
except ImportError:
  def cythonize(*args, **kwargs):
    from Cython.Build import cythonize
    return cythonize(*args, **kwargs)

def findPyx(path='.'):
  pyxFiles = []
  for root, _, filenames in os.walk(path):
    for file in filenames:
      if file.endswith('.pyx'):
        pyxFiles.append(os.path.join(root, file))
  return pyxFiles

def findCythonExtensions(path='.'):
  extensions = cythonize(findPyx(path), language_level=3)
  for ext in extensions:
    ext.include_dirs = [numpy.get_include()]
  return extensions

class BuildPy(setuptools.command.build_py.build_py):
  def run(self):
    setuptools.command.build_py.build_py.run(self)

class Develop(setuptools.command.develop.develop):
  def run(self):
    setuptools.command.develop.develop.run(self)


setup(
    name='hardware-tools',
    version=version,
    description='A library for automating hardware development and testing',
    long_description=longDescription,
    long_description_content_type='text/markdown',
    license='MIT',
    ext_modules=findCythonExtensions(),
    packages=find_packages(),
    package_data={'hardware_tools': []},
    install_requires=required,
    tests_require=['json'],
    test_suite='tests',
    scripts=[],
    author='Bradley Davis',
    author_email='me@bradleydavis.tech',
    url='https://github.com/WattsUp/hardware-tools',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.6',
    include_package_data=True,
    cmdclass={
        'build_py': BuildPy,
        'develop': Develop,
    },
    zip_safe=False,
)
