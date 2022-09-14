"""Test module hardware_tools.equipment.tektronix.common
"""

import datetime

import numpy as np

from hardware_tools.equipment.tektronix import common

from tests import base


class TestCommon(base.TestBase):
  """Test Equipment Tektronix Common
  """

  def test_parse_waveform_query(self):
    data = b""
    self.assertRaises(ValueError, common.parse_waveform_query, data)
    data = (b":WFMOUTPRE:NR_PT 1;XINCR 1;XZERO 0;XUNIT s;WFID empty;"
            b"BYT_OR MSB;BN_FMT RI;BYT_NR 4;:CURVE #11\0\0\0\0")
    self.assertRaises(ValueError, common.parse_waveform_query, data)
    data = (b":WFMOUTPRE:NR_PT 1;XINCR 1;XZERO 0;XUNIT s;WFID \xc3\x28;"
            b"BYT_OR MSB;BN_FMT RI;BYT_NR 2;:CURVE #11\0\0")
    self.assertRaises(ValueError, common.parse_waveform_query, data)

    # waveform.0 is 16b and has a CSV copy to compare against
    with open(self._DATA_ROOT.joinpath("tektronix", "waveform.0.isf"),
              "rb") as file:
      data = file.read()
      samples_noisy, info = common.parse_waveform_query(data, add_noise=True)
      self.assertIsInstance(samples_noisy, np.ndarray)
      self.assertEqual(samples_noisy.shape, (2, 10000))
      target = {
          "clipping_bottom": False,
          "clipping_top": False,
          "config_string": "Ch1, DC coupling, 500.0mV/div, 400.0us/div, "
                           "10000 points, Sample mode",
          "x_incr": 4e-7,
          "x_unit": "s",
          "y_incr": 7.8125e-05,
          "y_unit": "V",
          "y_clip_min": -0.559921875,
          "y_clip_max": 4.559921875
      }
      self.assertDictEqual(target, info)

      samples, info = common.parse_waveform_query(data, add_noise=False)
      self.assertIsInstance(samples, np.ndarray)
      self.assertEqual(samples.shape, (2, 10000))
      target = {
          "clipping_bottom": False,
          "clipping_top": False,
          "config_string": "Ch1, DC coupling, 500.0mV/div, 400.0us/div, "
                           "10000 points, Sample mode",
          "x_incr": 4e-7,
          "x_unit": "s",
          "y_incr": 7.8125e-05,
          "y_unit": "V",
          "y_clip_min": -0.559921875,
          "y_clip_max": 4.559921875
      }
      self.assertDictEqual(target, info)

      # Validate a uniform distribution
      noise: np.ndarray = samples_noisy[1] - samples[1]
      self.assertEqualWithinSampleError(0.0, noise.mean(), noise.size)
      noise_width = noise.max() - noise.min()
      self.assertEqualWithinSampleError(noise_width,
                                        np.sqrt(noise.std()**2 * 12),
                                        noise.size)

      # Compare against CSV the scope also saved
      path = self._DATA_ROOT.joinpath("tektronix", "waveform.0.csv")
      csv_samples = np.genfromtxt(path, delimiter=",").T
      for i in range(samples.shape[1]):
        self.assertEqualWithinError(csv_samples[0, i], samples[0, i], 1e-6)
        self.assertEqualWithinError(csv_samples[1, i], samples[1, i], 1e-6)

    # waveform.1 is 8b
    with open(self._DATA_ROOT.joinpath("tektronix", "waveform.1.isf"),
              "rb") as file:
      data = file.read()
      samples, info = common.parse_waveform_query(data, raw=True)
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

  def test_parse_wfm_file(self):
    data = b""
    self.assertRaises(ValueError, common.parse_wfm_file, data)
    data = b"12" * 0x346
    self.assertRaises(ValueError, common.parse_wfm_file, data)

    epoch = datetime.datetime.utcfromtimestamp(0).replace(
        tzinfo=datetime.timezone.utc)
    with open(self._DATA_ROOT.joinpath("tektronix", "waveform.2.wfm"),
              "rb") as file:
      data = file.read()
      samples, info = common.parse_wfm_file(data, add_noise=True)
      self.assertIsInstance(samples, np.ndarray)
      self.assertEqual(samples.shape, (2, 1000))
      target = {
          "clipping_bottom": False,
          "clipping_top": False,
          "config_string": "Label: , 1000 points, 1 frame",
          "x_incr": 2e-11,
          "x_unit": "s",
          "y_incr": 7.8125e-6,
          "y_unit": "V",
          "y_clip_min": 1.1420078125000002,
          "y_clip_max": 1.6539921875,
          "timestamp": epoch
      }
      self.assertDictEqual(target, info)

      samples, info = common.parse_wfm_file(data,
                                            add_noise=False,
                                            include_prepost=True)
      self.assertIsInstance(samples, np.ndarray)
      self.assertEqual(samples.shape, (2, 1064))
      target = {
          "clipping_bottom": False,
          "clipping_top": False,
          "config_string": "Label: , 1064 points, 1 frame",
          "x_incr": 2e-11,
          "x_unit": "s",
          "y_incr": 7.8125e-6,
          "y_unit": "V",
          "y_clip_min": 1.1420078125000002,
          "y_clip_max": 1.6539921875,
          "timestamp": epoch
      }
      self.assertDictEqual(target, info)

    # waveform.3 is just the header and checksum
    with open(self._DATA_ROOT.joinpath("tektronix", "waveform.3.wfm"),
              "rb") as file:
      data = file.read()
      samples, info = common.parse_wfm_file(data, add_noise=True, raw=True)
      self.assertIsInstance(samples, np.ndarray)
      self.assertEqual(samples.shape, (2, 0))
      target = {
          "clipping_bottom": False,
          "clipping_top": False,
          "config_string": "Label: , 0 points, 1 frame",
          "x_incr": 1.0,
          "x_unit": "",
          "y_incr": 1,
          "y_unit": "ADC Counts",
          "y_clip_min": -127,
          "y_clip_max": 127,
          "timestamp": epoch
      }
      self.assertDictEqual(target, info)

      # Short on the checksum
      self.assertRaises(ValueError, common.parse_wfm_file, data[:-1])

      # Short on the header
      self.assertRaises(ValueError, common.parse_wfm_file, data[:-8])

      # Checksum error
      data_list = [int(i) for i in data]
      data_list[-1] = 0xFF
      self.assertRaises(ValueError, common.parse_wfm_file, bytes(data_list))

      # Bad bytes_per_point
      data_list[0x00F] = 0xFF
      self.assertRaises(ValueError, common.parse_wfm_file, bytes(data_list))

      # Bad exp_dim_format
      data_list[0x0F0] = 0xFF
      self.assertRaises(ValueError, common.parse_wfm_file, bytes(data_list))

      # Unsupported exp_dim_storage_type
      data_list[0x0F4] = 0x01
      self.assertRaises(ValueError, common.parse_wfm_file, bytes(data_list))

      # Unsupported data_type
      data_list[0x07A] = 0x06
      self.assertRaises(ValueError, common.parse_wfm_file, bytes(data_list))

      # Unsupported version
      data_list[0x009] = ord("2")
      self.assertRaises(ValueError, common.parse_wfm_file, bytes(data_list))

    # waveform.4 has fast frames
    with open(self._DATA_ROOT.joinpath("tektronix", "waveform.4.wfm"),
              "rb") as file:
      data = file.read()
      samples, info = common.parse_wfm_file(data, add_noise=True, raw=True)
      self.assertIsInstance(samples, np.ndarray)
      self.assertEqual(samples.shape, (5, 2, 1000))
      kwargs = {
          "year": 2018,
          "month": 9,
          "day": 19,
          "hour": 19,
          "minute": 10,
          "second": 45,
          "tzinfo": datetime.timezone.utc
      }
      timestamps = [
          datetime.datetime(**kwargs, microsecond=461392),
          datetime.datetime(**kwargs, microsecond=461393),
          datetime.datetime(**kwargs, microsecond=461393),
          datetime.datetime(**kwargs, microsecond=461394),
          datetime.datetime(**kwargs, microsecond=461394)
      ]
      target = {
          "clipping_bottom": False,
          "clipping_top": False,
          "config_string": "Label: , 1000 points, 5 frames",
          "x_incr": 3.2e-10,
          "x_unit": "s",
          "y_incr": 1,
          "y_unit": "ADC Counts",
          "y_clip_min": -32767,
          "y_clip_max": 32767,
          "timestamp": timestamps
      }
      self.assertDictEqual(target, info)

      samples, info = common.parse_wfm_file(data,
                                            raw=True,
                                            include_prepost=True)
      self.assertIsInstance(samples, np.ndarray)
      self.assertEqual(samples.shape, (5, 2, 1064))
      target = {
          "clipping_bottom": False,
          "clipping_top": False,
          "config_string": "Label: , 1064 points, 5 frames",
          "x_incr": 3.2e-10,
          "x_unit": "s",
          "y_incr": 1,
          "y_unit": "ADC Counts",
          "y_clip_min": -32767,
          "y_clip_max": 32767,
          "timestamp": timestamps
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

  def test_parse_sample_mode(self):
    v = "ENV"
    self.assertEqual(common.parse_sample_mode(v), common.SampleMode.ENVELOPE)

    v = "ENVELOPE"
    self.assertEqual(common.parse_sample_mode(v), common.SampleMode.ENVELOPE)

    v = "EN"
    self.assertEqual(common.parse_sample_mode(v), v)
