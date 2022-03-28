"""hardware-tools

A library for automating hardware development and testing
"""

from hardware_tools import version

__version__ = version.version_full

__all__ = ["equipment", "measurement", "math"]

from hardware_tools import equipment
# from hardware_tools import measurement

from hardware_tools import math
