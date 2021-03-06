"""Test main
"""

import os
import unittest

import autodict

from tests import TEST_LOG


def pre_tests():
  if TEST_LOG.exists():
    os.remove(TEST_LOG)
  with autodict.JSONAutoDict(TEST_LOG) as d:
    d["classes"] = {}
    d["methods"] = {}


def post_tests():
  n_slowest = 10
  with autodict.JSONAutoDict(TEST_LOG) as d:
    classes = sorted(d["classes"].items(),
                     key=lambda item: -item[1])[:n_slowest]
    methods = sorted(d["methods"].items(),
                     key=lambda item: -item[1])[:n_slowest]

  print(f"{n_slowest} slowest classes")
  if len(classes) != 0:
    n_pad = max(len(k) for k, _ in classes) + 1
    for cls, duration in classes:
      print(f"  {cls:{n_pad}}: {duration:6.2f}s")

  print(f"{n_slowest} slowest tests")
  if len(methods) != 0:
    n_pad = max(len(k) for k, _ in methods) + 1
    for method, duration in methods:
      print(f"  {method:{n_pad}}: {duration:6.2f}s")


pre_tests()
unittest.main(module=None, exit=False)
post_tests()
