"""Test base class
"""

import os
import pathlib
import time
import unittest

import autodict
import numpy as np

from hardware_tools.math import gaussian

from tests import TEST_LOG


class TestBase(unittest.TestCase):
  """Test base class
  """

  _TEST_ROOT = pathlib.Path(".test")
  _DATA_ROOT = pathlib.Path(__file__).parent.joinpath("data")
  _P_FAIL = 1e-4
  _RNG = np.random.default_rng()

  def __clean_test_root(self):
    if self._TEST_ROOT.exists():
      for f in os.listdir(self._TEST_ROOT):
        os.remove(self._TEST_ROOT.joinpath(f))

  def assertEqualWithinSampleError(self, target, real, n):
    if target == 0.0:
      error = np.abs(real - target)
    else:
      error = np.abs(real / target - 1)
    threshold = gaussian.sample_error_inv(n, self._P_FAIL)
    self.assertLessEqual(error, threshold)

  def assertEqualWithinError(self, target, real, threshold):
    if target == 0.0:
      error = np.abs(real - target)
    else:
      error = np.abs(real / target - 1)
    self.assertLessEqual(error, threshold)

  def setUp(self):
    self.__clean_test_root()
    self._TEST_ROOT.mkdir(parents=True, exist_ok=True)
    self._test_start = time.perf_counter()

    # Remove sleeping by default, mainly in read hardware interaction
    self._original_sleep = time.sleep
    time.sleep = lambda *args: None

  def tearDown(self):
    duration = time.perf_counter() - self._test_start
    with autodict.JSONAutoDict(TEST_LOG) as d:
      d["methods"][self.id()] = duration
    self.__clean_test_root()

    # Restore sleeping
    time.sleep = self._original_sleep

  def log_speed(self, slow_duration, fast_duration):
    with autodict.JSONAutoDict(TEST_LOG) as d:
      d["speed"][self.id()] = {
          "slow": slow_duration,
          "fast": fast_duration,
          "increase": slow_duration / fast_duration
      }

  @classmethod
  def setUpClass(cls):
    print(f"{cls.__module__}.{cls.__qualname__}[", end="", flush=True)
    cls._CLASS_START = time.perf_counter()

  @classmethod
  def tearDownClass(cls):
    print("]done", flush=True)
    # time.sleep(10)
    duration = time.perf_counter() - cls._CLASS_START
    with autodict.JSONAutoDict(TEST_LOG) as d:
      d["classes"][f"{cls.__module__}.{cls.__qualname__}"] = duration
