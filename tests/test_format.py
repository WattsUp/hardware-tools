"""Test module hardware_tools.format
"""

import unittest

import datetime
import random
import string

import numpy as np
import time_machine

from hardware_tools import strformat


class TestStringFormat(unittest.TestCase):
  """Test math methods
  """

  def test_metric_prefix(self):
    x = np.random.uniform(2, 1000)
    unit = random.choice(string.ascii_letters)

    specifier = "6.1f"
    specifier_small = "6.3f"

    s = strformat.metric_prefix(x, unit=unit, specifier=specifier)
    self.assertEqual(s, f"{x:{specifier}}  {unit}")

    s = strformat.metric_prefix(x * 1e6, unit=unit, specifier=specifier)
    self.assertEqual(s, f"{x:{specifier}} M{unit}")

    s = strformat.metric_prefix(-x * 1e-6, unit=unit, specifier=specifier)
    self.assertEqual(s, f"{-x:{specifier}} µ{unit}")

    s = strformat.metric_prefix(x * 1e-15,
                                unit=unit,
                                specifier_small=specifier_small)
    self.assertEqual(s, f"{x / 1000:{specifier_small}} p{unit}")

    prefixes = {
        "Ti": 1024**4,
        "Gi": 1024**3,
        "Mi": 1024**2,
        "ki": 1024**1,
        "  ": 1,
    }

    s = strformat.metric_prefix(x * 1024**2,
                                unit="B",
                                specifier=specifier,
                                threshold=2,
                                prefixes=prefixes)
    self.assertEqual(s, f"{x:{specifier}} MiB")

    x = 1.999
    s = strformat.metric_prefix(x, unit=unit, specifier=specifier, threshold=2)
    self.assertEqual(s, f"{x * 1000:{specifier}} m{unit}")

    x = 2
    s = strformat.metric_prefix(x, unit=unit, specifier=specifier, threshold=2)
    self.assertEqual(s, f"{x:{specifier}}  {unit}")

  def test_time_str(self):
    hours = np.random.randint(0, 100)
    minutes = np.random.randint(0, 60)
    seconds = np.random.uniform(0, 60)
    duration = hours * 3600 + minutes * 60 + seconds

    s = strformat.time_str(duration, sub=True, hours=True)
    self.assertEqual(s, f"{hours:02}:{minutes:02}:{seconds:05.2f}")

    s = strformat.time_str(duration, sub=False, hours=True)
    self.assertEqual(s, f"{hours:02}:{minutes:02}:{int(seconds):02}")

    s = strformat.time_str(duration, sub=False, hours=False)
    self.assertEqual(s, f"{hours * 60 + minutes:02}:{int(seconds):02}")

  def test_elapsed_str(self):
    hours = np.random.randint(0, 100)
    minutes = np.random.randint(0, 60)
    seconds = np.random.uniform(0, 60)
    duration = hours * 3600 + minutes * 60 + seconds
    start = datetime.datetime.now(datetime.timezone.utc)
    end = start + datetime.timedelta(seconds=duration)

    s = strformat.elapsed_str(start, end=end, sub=True, hours=True)
    self.assertEqual(s, f"{hours:02}:{minutes:02}:{seconds:05.2f}")

    with time_machine.travel(end, tick=False):
      s = strformat.elapsed_str(start, sub=True, hours=True)
    self.assertEqual(s, f"{hours:02}:{minutes:02}:{seconds:05.2f}")
