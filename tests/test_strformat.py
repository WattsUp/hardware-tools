"""Test module hardware_tools.strformat
"""

import datetime
import random
import string
import time

import time_machine

from hardware_tools import strformat

from tests import base


class TestStringFormat(base.TestBase):
  """Test math methods
  """

  def test_metric_prefix(self):
    x = self._RNG.uniform(2, 1000)
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
    hours = self._RNG.integers(0, 100)
    minutes = self._RNG.integers(0, 60)
    seconds = self._RNG.uniform(0, 60)
    duration = hours * 3600 + minutes * 60 + seconds

    s = strformat.time_str(duration, sub=True, hours=True)
    self.assertEqual(s, f"{hours:02}:{minutes:02}:{seconds:05.2f}")

    s = strformat.time_str(duration, sub=False, hours=True)
    self.assertEqual(s, f"{hours:02}:{minutes:02}:{int(seconds):02}")

    s = strformat.time_str(duration, sub=False, hours=False)
    self.assertEqual(s, f"{hours * 60 + minutes:02}:{int(seconds):02}")

  def test_elapsed_str(self):
    hours = self._RNG.integers(0, 100)
    minutes = self._RNG.integers(0, 60)
    seconds = self._RNG.uniform(0, 60)
    duration = hours * 3600 + minutes * 60 + seconds
    start = datetime.datetime.now(datetime.timezone.utc)
    end = start + datetime.timedelta(seconds=duration)

    s = strformat.elapsed_str(start, end=end, sub=True, hours=True)
    self.assertEqual(s, f"{hours:02}:{minutes:02}:{seconds:05.2f}")

    with time_machine.travel(end, tick=False):
      s = strformat.elapsed_str(start, sub=True, hours=True)
    self.assertEqual(s, f"{hours:02}:{minutes:02}:{seconds:05.2f}")

    end_perf = time.perf_counter()
    start_perf = end_perf - duration

    # time.perf_counter() is not adjustable so need to run this test within 0.5s
    s = strformat.elapsed_str(start_perf, sub=False, hours=True)
    self.assertEqual(s, f"{hours:02}:{minutes:02}:{int(seconds):02}")

    s = strformat.elapsed_str(start_perf, end=end_perf, sub=True, hours=True)
    self.assertEqual(s, f"{hours:02}:{minutes:02}:{seconds:05.2f}")

    self.assertRaises(TypeError, strformat.elapsed_str, None)
