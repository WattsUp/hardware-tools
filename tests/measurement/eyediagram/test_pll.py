"""Test module hardware_tools.measurement.pll
"""

from __future__ import annotations

import unittest

import numpy as np

from hardware_tools.measurement.eyediagram import pll


class TestPLLSingleLowPass(unittest.TestCase):
  """Test PLLSingleLowPass
  """

  def test_pll(self):
    t_sym = 10e-9
    n_sym = int(50e3)
    bandwidth = 100e3

    p = pll.PLLSingleLowPass(t_sym, bandwidth)
    # import matplotlib.pyplot as pyplot
    # _, subplots = pyplot.subplots(3, 1)
    # subplots[0].plot(periods)
    # subplots[1].plot(ties)
    # subplots[2].hist(ties, 50)
    # pyplot.show()

    # Ideal clock
    edges = np.linspace(0, t_sym * (n_sym - 1), n_sym)
    out, periods, ties = p.run(edges)
    self.assertEqual(len(out), n_sym)
    self.assertEqual(len(periods), len(edges))
    self.assertEqual(len(ties), len(edges))
    average_period = np.average(periods)
    self.assertLessEqual(average_period, 1.01 * t_sym)
    self.assertGreaterEqual(average_period, 0.99 * t_sym)
    average_tie = np.average(ties)
    self.assertLessEqual(average_tie, 0.01 * t_sym)
    self.assertGreaterEqual(average_tie, -0.01 * t_sym)

    # Wrong initial conditions
    edges = np.linspace(0, t_sym / 10 * (n_sym - 1), n_sym)
    self.assertRaises(ArithmeticError, p.run, edges)

    # Ideal data stream (drop 3/4 clock edges)
    edges = np.linspace(0, t_sym * (n_sym - 1), n_sym)
    edges = edges[::4]
    out, periods, ties = p.run(edges)
    self.assertEqual(len(ties), len(edges))
    average_period = np.average(periods)
    self.assertLessEqual(average_period, 1.01 * t_sym)
    self.assertGreaterEqual(average_period, 0.99 * t_sym)
    average_tie = np.average(ties)
    self.assertLessEqual(average_tie, 0.01 * t_sym)
    self.assertGreaterEqual(average_tie, -0.01 * t_sym)

    # Periodic jitter
    f_jitter = bandwidth * 0.1
    a_periodic = 0.01
    edges = np.linspace(0, t_sym * (n_sym - 1), n_sym)
    edges = edges[::4]
    edges = edges + np.cos(2 * np.pi * f_jitter * edges) * t_sym * a_periodic
    out, periods, ties = p.run(edges)
    average_period = np.average(periods)
    self.assertLessEqual(average_period, 1.01 * t_sym)
    self.assertGreaterEqual(average_period, 0.99 * t_sym)
    average_tie = np.average(ties)
    self.assertLessEqual(average_tie, 0.01 * t_sym)
    self.assertGreaterEqual(average_tie, -0.01 * t_sym)
    range_tie = np.max(ties) - np.min(ties)
    self.assertLessEqual(range_tie, 1.01 * 2 * t_sym * a_periodic)
    self.assertGreaterEqual(range_tie, 0.99 * 2 * t_sym * a_periodic)

    # Random jitter
    a_random = 0.002
    edges = np.linspace(0, t_sym * (n_sym - 1), n_sym)
    edges = edges[::4]
    edges = edges + np.random.normal(0, a_random * t_sym, edges.size)
    out, periods, ties = p.run(edges)
    average_period = np.average(periods)
    self.assertLessEqual(average_period, 1.01 * t_sym)
    self.assertGreaterEqual(average_period, 0.99 * t_sym)
    average_tie = np.average(ties)
    self.assertLessEqual(average_tie, 0.01 * t_sym)
    self.assertGreaterEqual(average_tie, -0.01 * t_sym)
    stddev_tie = np.std(ties)
    self.assertLessEqual(stddev_tie, 1.1 * t_sym * a_random)
    self.assertGreaterEqual(stddev_tie, 0.9 * t_sym * a_random)

    # Periodic jitter + random jitter
    edges = np.linspace(0, t_sym * (n_sym - 1), n_sym)
    edges = edges[::4]
    edges = edges + np.cos(2 * np.pi * f_jitter * edges) * t_sym * a_periodic
    edges = edges + np.random.normal(0, a_random * t_sym, edges.size)
    out, periods, ties = p.run(edges)
    average_period = np.average(periods)
    self.assertLessEqual(average_period, 1.01 * t_sym)
    self.assertGreaterEqual(average_period, 0.99 * t_sym)
    average_tie = np.average(ties)
    self.assertLessEqual(average_tie, 0.01 * t_sym)
    self.assertGreaterEqual(average_tie, -0.01 * t_sym)

    # Slow clock
    stretch = 1.01
    edges = np.linspace(0, t_sym * stretch * (n_sym - 1), n_sym)
    edges = edges[::4]
    out, periods, ties = p.run(edges)
    average_period = np.average(periods)
    self.assertLessEqual(average_period, 1.01 * t_sym * stretch)
    self.assertGreaterEqual(average_period, 0.99 * t_sym * stretch)
    average_tie = np.average(ties)
    self.assertLessEqual(average_tie, 0.01 * t_sym * stretch)
    self.assertGreaterEqual(average_tie, -0.01 * t_sym * stretch)
