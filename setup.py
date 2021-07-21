import numpy
import os
import setuptools
import subprocess

from setuptools import setup, find_packages
import setuptools.command.develop
import setuptools.command.build_py

with open('README.md') as f:
  longDescription = f.read()

with open('requirements.txt') as f:
  required = f.read().splitlines()

version = '0.0.1'
cwd = os.path.dirname(os.path.abspath(__file__))
try:
  sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                cwd=cwd).decode('ascii').strip()
  version += '+' + sha[:7]
except subprocess.CalledProcessError:
  pass
except IOError:  # FileNotFoundError for python 3
  pass

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
    self.createVersionFile()
    setuptools.command.build_py.build_py.run(self)

  @staticmethod
  def createVersionFile():
    print('-- Building veresion' + version)
    versionPath = os.path.join(cwd, 'version.py')
    with open(versionPath, 'w') as file:
      file.write(f'__version__ = \'{version}\'\n')

class Develop(setuptools.command.develop.develop):
  def run(self):
    BuildPy.createVersionFile()
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
