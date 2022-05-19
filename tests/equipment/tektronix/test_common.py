"""Test module hardware_tools.equipment.tektronix.common
"""

import numpy as np

from hardware_tools.equipment.tektronix import common

from tests import base


class TestCommon(base.TestBase):
  """Test Equipment Tektronix Common
  """

  def test_parse_wfm(self):
    data = b""
    self.assertRaises(ValueError, common.parse_wfm, data)
    data = (b":WFMOUTPRE:NR_PT 1;XINCR 1;XZERO 0;XUNIT s;WFID empty;BYT_OR MSB;"
            b"BN_FMT RI;BYT_NR 4;:CURVE #11\0\0\0\0")
    self.assertRaises(ValueError, common.parse_wfm, data)

    with open(self._DATA_ROOT.joinpath("tektronix", "waveform.0.isf"),
              "rb") as file:
      data = file.read()
      samples, info = common.parse_wfm(data, add_noise=True)
      self.assertIsInstance(samples, np.ndarray)
      self.assertEqual(samples.shape, (2, 20000000))
      target = {
          "clipping_bottom": False,
          "clipping_top": False,
          "config_string": "Ch1, DC coupling, 10.00uW/div, 400.0us/div, "
                           "20000000 points, Sample mode",
          "x_incr": 2e-10,
          "x_unit": "s",
          "y_incr": 1.5625e-9,
          "y_unit": "W",
          "y_clip_min": -1.73984375e-05,
          "y_clip_max": 8.49984375e-05
      }
      self.assertDictEqual(target, info)

    with open(self._DATA_ROOT.joinpath("tektronix", "waveform.1.isf"),
              "rb") as file:
      data = file.read()
      samples, info = common.parse_wfm(data, raw=True)
      self.assertIsInstance(samples, np.ndarray)
      self.assertEqual(samples.shape, (2, 10000))
      target = {
          "clipping_bottom": False,
          "clipping_top": False,
          "config_string": "Ch2, DC coupling, 50.00mV/div, 1.000ms/div, "
                           "10000 points, Sample mode",
          "x_incr": 1e-6,
          "x_unit": "s",
          "y_incr": 1,
          "y_unit": "ADC Counts",
          "y_clip_min": -127,
          "y_clip_max": 127
      }
      self.assertDictEqual(target, info)

  def test_parse_comparison(self):
    v = "OUT"
    self.assertEqual(common.parse_comparison(v), common.Comparison.OUTSIDE)

    v = "OUTRANGE"
    self.assertEqual(common.parse_comparison(v), common.Comparison.OUTSIDE)

    v = "OU"
    self.assertEqual(common.parse_comparison(v), v)

  def test_parse_polarity(self):
    v = "EIT"
    self.assertEqual(common.parse_polarity(v), common.EdgePolarity.BOTH)

    v = "EITHER"
    self.assertEqual(common.parse_polarity(v), common.EdgePolarity.BOTH)

    v = "EI"
    self.assertEqual(common.parse_polarity(v), v)

  def test_parse_threshold(self):
    v = "ECL"
    self.assertEqual(common.parse_threshold(v), -1.3)

    v = "TTL"
    self.assertEqual(common.parse_threshold(v), 1.4)

    v = 3.14
    self.assertEqual(common.parse_threshold(v), v)
