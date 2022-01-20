# hardware-tools
A library for automating hardware development and testing

----
## Environment
List of dependencies for package to run.
### Required
* python modules, installed via `pip install hardware_tools`
  * numpy
  * pyvisa
  * cython
  * colorama
  * matplotlib
  * scipy
  * sklean
  * Pillow
  * scikit-image

### Optional
* Test extensions, installed via `pip install hardware_tools[test]`
  * time-machine
  * AutoDict
----
## Installation / Build / Deployment
```bash
# To install latest stable version on PyPi, execute:
python -m pip install hardware_tools

# To install from source, execute:
git clone https://github.com/WattsUp/hardware-tools
cd hardware-tools
python -m pip install .

# For development, install as a link to repository such that code changes are used. And include testing packages
git clone https://github.com/WattsUp/hardware-tools
cd hardware-tools
python -m pip install -e .[test]
```

----
## Usage
Explain how to use your project.
```Python
# TODO
```
----
## Running Tests
Make sure to install package with [testing extension](#optional)
```bash
# To run the automated tests, execute:
python tests discover -s tests -t . --locals

# To save the results to file, execute:
python tests discover -s tests -t . --locals &> testing.log

## The following is a synopsis of unittest main arguments ##
# To run a singular test file, execute:
python tests $path_to_test_file
python tests tests.measurement.test_mask

# To run a singular test class, execute:
python tests $path_to_test_file.$class
python tests tests.measurement.test_mask.TestMaskDecagon

# To run a singular test method, execute:
python tests $path_to_test_file.$class.$method
python tests tests.measurement.test_mask.TestMaskDecagon.test_init

# Multiple can be strung together
python tests tests.measurement.test_mask tests.test_math
```
```bash
# To run coverage and print the report with missing lines, execute:
python -m coverage run && python -m coverage report -m

# To run profiler, execute:
python -m cProfile -s tottime -m tests discover -s tests -t . --locals > profile.log

# To run linting, execute:
python -m pylint hardware_tools tests tools setup.py
```
----
## Development
Code development of this project adheres to [Google Python Guide](https://google.github.io/styleguide/pyguide.html)

### Styling
Use `yapf` to format files, based on Google's guide with the exception of indents being 2 spaces.
```bash
# To format all files, execute:
yapf -ir .
```

---
## Versioning
Versioning of this projects adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and is implemented using git tags.
