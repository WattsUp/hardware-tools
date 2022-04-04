"""Setup and install hardware-tools

Typical usage:
  python setup.py develop
  python setup.py install
  python setup.py test
"""

from typing import List, Tuple

import os
import setuptools
import setuptools.command.build_py
import setuptools.command.develop

module_name = "hardware-tools"
module_folder = "hardware_tools"

with open("README.md", encoding="utf-8") as readme:
  long_description = readme.read()

required = [
    "numpy", "pyvisa", "colorama", "matplotlib", "scipy", "sklearn", "Pillow",
    "scikit-image"
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


def is_package_dir(dir_path: str) -> bool:
  """Check if folder is a python package

  Args:
    dir_path: Folder name to process

  Returns:
    True if folder contains __init__.(py|pyc|pyx|pxd)
  """
  for filename in ("__init__.py", "__init__.pyc", "__init__.pyx",
                   "__init__.pxd"):
    path = os.path.join(dir_path, filename)
    if os.path.exists(path):
      return True
  return False


def package(filename: str) -> Tuple[str]:
  """Get package of file

  Args:
    filename: Name of file to process

  Returns:
    Tuple of modules and submodules
  """
  folder = os.path.dirname(os.path.abspath(str(filename)))
  if folder != filename and is_package_dir(folder):
    return package(folder) + (os.path.basename(folder),)
  else:
    return ()


def fully_qualified_name(filename: str) -> str:
  """Get name of module from filename

  Args:
    filename: Name of file to process

  Returns:
    Fully qualified name of module: module.submodule.name
  """
  module = os.path.splitext(os.path.basename(filename))[0]
  return ".".join(package(filename) + (module,))


def find_extensions(path: str = ".") -> List[setuptools.Extension]:
  """Find extension in folder, Cython or C

  Returns:
    list of Extensions to build
  """
  pyx_files = []
  c_files = []
  for root, _, filenames in os.walk(path):
    for f in filenames:
      if f.endswith(".pyx"):
        pyx_files.append(os.path.join(root, f))
      elif f.endswith(".c"):
        c_files.append(os.path.join(root, f))
  for pyx_file in pyx_files:
    c_file = pyx_file[:-4] + ".c"
    if c_file in c_files:
      c_files.remove(c_file)

  extensions = [
      setuptools.Extension(name=fully_qualified_name(c_file), sources=[c_file])
      for c_file in c_files
  ]
  if len(pyx_files) > 0:
    import Cython.Build  # pylint: disable=import-outside-toplevel
    extensions.extend(Cython.Build.cythonize(pyx_files, language_level=3))
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    ext_modules=find_extensions(),
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    package_data={module_folder: ["**/*.pxd"]},
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
