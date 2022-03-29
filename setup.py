"""Setup and install hardware-tools

Typical usage:
  python setup.py develop
  python setup.py install
  python setup.py test
"""

import os
import setuptools
import setuptools.command.build_py
import setuptools.command.develop

module_name = "hardware-tools"
module_folder = "hardware_tools"

with open("README.md", encoding="utf-8") as file:
  longDescription = file.read()

required = [
    "numpy", "pyvisa", "cython", "colorama", "matplotlib", "scipy", "sklearn",
    "Pillow", "scikit-image"
]

try:
  from tools import gitsemver
  version = gitsemver.get_version()
  with open(f"{module_folder}/version.py", "w", encoding="utf-8") as file:
    file.write('"""Module version information\n"""\n\n')
    file.write(f'version = "{version}"\n')
    file.write(f'version_full = "{version.full_str()}"\n')
    file.write(f'tag = "{version.raw}"\n')
except ImportError:
  import re
  with open(f"{module_folder}/version.py", "r", encoding="utf-8") as file:
    version = re.search(r'version = "(.*)"', file.read())[1]

cwd = os.path.dirname(os.path.abspath(__file__))


def find_pyx(path="."):
  pyx_files = []
  for root, _, filenames in os.walk(path):
    for f in filenames:
      if f.endswith(".pyx"):
        pyx_files.append(os.path.join(root, f))
  return pyx_files


def find_cython_extensions(path="."):
  pyx_files = find_pyx(path)
  if len(pyx_files) == 0:
    return []
  import Cython.Build  # pylint: disable=import-outside-toplevel
  extensions = Cython.Build.cythonize(pyx_files, language_level=3)
  if "numpy" in required:
    import numpy  # pylint: disable=import-outside-toplevel

    for ext in extensions:
      ext.include_dirs = [numpy.get_include()]
  return extensions


class BuildPy(setuptools.command.build_py.build_py):

  def run(self):
    setuptools.command.build_py.build_py.run(self)


class Develop(setuptools.command.develop.develop):

  def run(self):
    setuptools.command.develop.develop.run(self)


setuptools.setup(
    name=module_name,
    version=str(version),
    description="A library for automating hardware development and testing",
    long_description=longDescription,
    long_description_content_type="text/markdown",
    license="MIT",
    ext_modules=find_cython_extensions(),
    packages=setuptools.find_packages(),
    package_data={module_folder: []},
    install_requires=required,
    extras_require={"test": ["time-machine", "AutoDict", "coverage", "pylint"]},
    test_suite="tests",
    scripts=[],
    author="Bradley Davis",
    author_email="me@bradleydavis.tech",
    url="https://github.com/WattsUp/hardware-tools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    cmdclass={
        "build_py": BuildPy,
        "develop": Develop,
    },
    zip_safe=False,
)
