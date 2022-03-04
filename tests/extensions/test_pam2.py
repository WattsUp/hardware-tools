"""Test module hardware_tools.extension.pam2
"""

import time

import numpy as np
from scipy import signal

from hardware_tools.extensions import pam2_slow, pam2_fast

from tests import base

_rng = np.random.default_rng()
f_scope = 5e9
n_scope = int(1e5)
t_scope = n_scope / f_scope
n_bits = int(1e3)
t_bit = t_scope / n_bits
bits = signal.max_len_seq(5, length=n_bits)[0]
t = np.linspace(0, t_scope, n_scope)
v_signal = 3.3
y = (np.repeat(bits, n_scope / n_bits) +
     _rng.normal(0, 0.05, size=n_scope)) * v_signal
f_lp = 0.75 / t_bit
sos = signal.bessel(4, f_lp, fs=f_scope, output="sos", norm="mag")

zi = signal.sosfilt_zi(sos) * y[0]
y, _ = signal.sosfilt(sos, y, zi=zi)

t_delta = 1 / f_scope

i_width = int((t_bit / t_delta) + 0.5) + 2
max_i = n_scope

centers_i = []
centers_t = []
t_zero = t[0]
clock_edges = np.linspace(t_zero, t_zero + t_bit * (n_bits - 1), n_bits)
for b in clock_edges:
  center_t = b % t_delta
  center_i = int(((b - t_zero - center_t) / t_delta) + 0.5)

  if (center_i - i_width) < 0 or (center_i + i_width) >= max_i:
    continue
  centers_t.append(-center_t)
  centers_i.append(center_i)
edge_dir = pam2_slow.sample_vertical(y, centers_t, centers_i, t_delta, t_bit,
                                     v_signal / 2, 0.2, 0.1)["edge_dir"]


class TestPAM2(base.TestBase):
  """Test PAM2 methods
  """

  def _test_sample_vertical(self, module):
    result = module.sample_vertical(y, centers_t, centers_i, t_delta, t_bit,
                                    v_signal / 2, 0.2, 0.1)

    target = {
        "y_0": 0,
        "y_1": v_signal,
        "y_cross": v_signal / 2,
        "y_0_cross": 0,
        "y_1_cross": v_signal
    }
    for k, v in result.items():
      if k in target:
        v_mean = np.mean(v)
        self.assertEqualWithinError(target[k], v_mean, 0.05)

    for k, v in result["transitions"].items():
      # Generated bit pattern lacks even number of 000
      if k == "000":
        self.assertEqualWithinError(n_bits / 8 * 25 / 32, v, 0.1)
      else:
        self.assertEqualWithinError(n_bits / 8 * 33 / 32, v, 0.1)

  def test_sample_vertical(self):
    self._test_sample_vertical(pam2_slow)
    self._test_sample_vertical(pam2_fast)

    # Validate fast is actually faster

    start = time.perf_counter()
    result_slow = pam2_slow.sample_vertical(y, centers_t, centers_i, t_delta,
                                            t_bit, v_signal / 2, 0.2, 0.1)
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = pam2_fast.sample_vertical(y, centers_t, centers_i, t_delta,
                                            t_bit, v_signal / 2, 0.2, 0.1)
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    self.assertListEqual(result_slow["edge_dir"], result_fast["edge_dir"])
    self.assertLess(elapsed_fast, elapsed_slow)

  def _test_sample_horizontal(self, module):
    result = module.sample_horizontal(y, centers_t, centers_i, edge_dir,
                                      t_delta, t_bit, 0.0, v_signal,
                                      v_signal / 2, 0.05, [0.2, 0.8])

    center = -0.066  # Due to filter delay
    target = {
        "t_rise_lower": center - 0.145,
        "t_rise_upper": center + 0.165,
        "t_rise_half": center,
        "t_fall_lower": center + 0.165,
        "t_fall_upper": center - 0.145,
        "t_fall_half": center,
        "t_cross_left": center,
        "t_cross_right": center + 1
    }
    for k, v in result.items():
      if k in target:
        v_mean = np.mean(v)
        self.assertEqualWithinError(target[k], v_mean, 0.05)

  def test_sample_horizontal(self):
    self._test_sample_horizontal(pam2_slow)
    self._test_sample_horizontal(pam2_fast)

    # Validate fast is actually faster

    start = time.perf_counter()
    result_slow = pam2_slow.sample_horizontal(y, centers_t, centers_i, edge_dir,
                                              t_delta, t_bit, 0.0, v_signal,
                                              v_signal / 2, 0.05, [0.2, 0.8])
    elapsed_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = pam2_fast.sample_horizontal(y, centers_t, centers_i, edge_dir,
                                              t_delta, t_bit, 0.0, v_signal,
                                              v_signal / 2, 0.05, [0.2, 0.8])
    elapsed_fast = time.perf_counter() - start

    self.log_speed(elapsed_slow, elapsed_fast)
    for k, v in result_slow.items():
      v_mean_slow = np.mean(v)
      v_mean_fast = np.mean(result_fast[k])
      self.assertEqualWithinError(v_mean_slow, v_mean_fast, 0.01)
    self.assertLess(elapsed_fast, elapsed_slow)
