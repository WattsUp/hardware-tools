"""Test module hardware_tools.math.interpolation
"""

import numpy as np

from hardware_tools.math import interpolation

from tests import base


class TestInterpolation(base.TestBase):
  """Test interpolation methods
  """

  def test_sinc(self):
    x_min = self._RNG.integers(-1000, 500)
    x_max = x_min + self._RNG.integers(1, 500)
    n_in = self._RNG.integers(500, 1000)
    n_down = self._RNG.integers(10, n_in - 10)
    n_up = self._RNG.integers(1000, 5000)

    x_in = np.linspace(x_min, x_max, n_in)
    y_in = self._RNG.uniform(-1, 1, n_in)
    self.assertEqual(x_in.shape, y_in.shape)

    x_down = np.linspace(x_min, x_max, n_down)
    y_down = interpolation.sinc(x_down, x_in, y_in)

    self.assertEqual(y_down.shape, x_down.shape)

    x_up = np.linspace(x_min, x_max, n_up)
    y_up = interpolation.sinc(x_up, x_in, y_in)

    self.assertEqual(y_up.shape, x_up.shape)

    # Sinc interpolation should have a rectangle frequency output
    # Compare relative power in the max input frequency band
    fft = np.abs(np.power(np.fft.rfft(y_up), 2))
    total_power = fft.sum()
    fft = fft / total_power
    f_in = n_in // 2 + 1

    input_band_power = fft[:f_in].sum()

    self.assertGreaterEqual(input_band_power, 0.99)

    # 2D input
    x_in = self._RNG.uniform(x_min, x_max, (n_in, n_in))
    self.assertRaises(ValueError, interpolation.sinc, x_up, x_in, y_in)

    # Mismatched input
    x_in = np.linspace(x_min, x_max, n_in + 1)
    self.assertRaises(ValueError, interpolation.sinc, x_up, x_in, y_in)

    # Single point input
    x_in = np.linspace(x_min, x_max, 1)
    y_in = self._RNG.uniform(-1, 1, 1)
    self.assertRaises(ValueError, interpolation.sinc, x_up, x_in, y_in)
