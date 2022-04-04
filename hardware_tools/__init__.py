"""hardware-tools

A library for automating hardware development and testing
"""

from hardware_tools import version

__version__ = version.version_full

__all__ = ["strformat", "equipment", "measurement", "math", "signal"]

from hardware_tools import strformat
