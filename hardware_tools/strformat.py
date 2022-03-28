"""Collection of string formatting functions
"""

from __future__ import annotations

import datetime
import time
from typing import Union


def metric_prefix(value: float,
                  unit: str = "",
                  specifier: str = "8.2f",
                  specifier_small: str = "8.3f",
                  threshold: float = 2,
                  prefixes: dict = None) -> str:
  """Format a value using metric prefixes to constrain string length

  Args:
    value: Number to format
    unit: Unit string to append to number and metric prefix
    specifier: Format specifier to convert value at
    specifier_small: Format specifier when under range
    threshold: Decision multiplier to determine order of magnitude.
      2 => 1999.9 vs 1 => 2.0
    prefixes: Dictionary of metric prefixes and their magnitude. None will
      default to T(1e12) to p(1e-12). Must be sorted largest to smallest

  Returns:
    "±xxx.x PU" where P is metric prefix and U is unit. Precision (xxx.x) is
    determined by specifier.
  """
  if prefixes is None:
    prefixes = {
        "T": 1e12,
        "G": 1e9,
        "M": 1e6,
        "k": 1e3,
        " ": 1e0,
        "m": 1e-3,
        "µ": 1e-6,
        "n": 1e-9,
        "p": 1e-12
    }
  for p, f in prefixes.items():
    if abs(value) >= (threshold * f):
      return f"{value / f:{specifier}} {p}{unit}"
  p, f = list(prefixes.items())[-1]
  return f"{value / f:{specifier_small}} {p}{unit}"


def time_str(duration: float, sub: bool = True, hours: bool = True) -> str:
  """Format time as HH:MM:SS.ss

  Args:
    duration: Time in seconds
    sub: True will include time less than a second ".ss"
    hours: True will include hours "HH:"

  Returns:
    Time string "[HH:]MM:SS[.ss]"
  """
  minutes, seconds = divmod(duration, 60)
  if hours:
    hours, minutes = divmod(minutes, 60)
    buf = f"{int(hours):02}:"
  else:
    buf = ""
  buf += f"{int(minutes):02}:"
  if sub:
    buf += f"{seconds:05.2f}"
  else:
    buf += f"{int(seconds):02}"
  return buf


def elapsed_str(start: Union[float, datetime.datetime],
                end: datetime.datetime = None,
                sub: bool = True,
                hours: bool = True) -> str:
  """Calculate elapsed time since start and format using time_str

  Params:
    start: Start timestamp, either from datetime.now() or time.perf_counter()
    end: End timestamp, None will fetch now()
    sub: True will include time less than a second ".ss"
    hours: True will include hours "HH:"

  Returns:
    Duration between start and end as string using time_str
  """
  if isinstance(start, datetime.datetime):
    if end is None:
      end = datetime.datetime.now(start.tzinfo)
    duration = (end - start).total_seconds()
  elif isinstance(start, float):
    if end is None:
      end = time.perf_counter()
    duration = end - start
  else:
    raise TypeError(f"Unknown start type: {type(start)}")
  return time_str(duration, sub=sub, hours=hours)
