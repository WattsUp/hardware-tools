"""Test module hardware_tools.equipment.tektronix.family_mso4000
"""

import re
import unittest

import numpy as np
import pyvisa

from hardware_tools.equipment import equipment, utility
from hardware_tools.equipment import tektronix

from .. import mock_pyvisa


class MockMDO3054(mock_pyvisa.Resource):
  """Mock Tektronix MDO3054
  """

  record_length = 1000
  horizontal_scale = 1e-9
  sample_frequency = min(2.5e9, record_length / (10 * horizontal_scale))
  trigger_a_mode = "AUTO"
  trigger_a_source = "CH1"
  trigger_a_coupling = "DC"
  trigger_a_slope = "FALL"
  acquire_mode = "SAMPLE"

  channels = {
      c: {
          "scale": 1,
          "position": 0,
          "offset": 0,
          "label": c,
          "bandwidth": 500e6,
          "termination": 1e6,
          "invert": False,
          "gain": 1,
          "coupling": "DC",
          "trigger": 0
      } for c in ["CH1", "CH2", "CH3", "CH4"]
  }

  def write(self, command: str) -> None:
    self.queue_tx.append(command)
    command = re.split(":| ", command.upper())
    if command[0] in ["HEADER", "VERBOSE"]:
      return
    if command[0] == "HORIZONTAL":
      if command[1] == "RECORDLENGTH":
        value = int(command[2])
        choices = [1e3, 10e3, 100e3, 1e6, 5e6, 10e6]
        choices = sorted(choices, key=lambda c: (value - c)**2)
        self.record_length = int(choices[0])
        self.sample_frequency = min(
            2.5e9, self.record_length / (10 * self.horizontal_scale))
        return
      if command[1] == "SCALE":
        value = float(command[2])
        choices = []
        for i in range(-9, 2):
          if 1 * 10**i == 0.1e-9 * self.record_length:
            choices.append(0.8 * 10**i)
          else:
            choices.append(1 * 10**i)
          choices.append(2 * 10**i)
          choices.append(4 * 10**i)
        if self.record_length >= 10e3:
          choices.extend([100, 200, 400])
        if self.record_length >= 100e3:
          choices.append(1000)
        choices = sorted(choices, key=lambda c: (value - c)**2)
        self.horizontal_scale = choices[0]
        self.sample_frequency = min(
            2.5e9, self.record_length / (10 * self.horizontal_scale))
        return
      if command[1] == "DELAY":
        if command[2] == "MODE":
          return
        if command[2] == "TIME":
          value = float(command[3])
          sample_t = 1 / self.sample_frequency
          value = round(value / sample_t) * sample_t
          min_delay = -sample_t * self.record_length
          self.horizontal_delay = min(5e3, max(min_delay, value))
          return
    if command[0] == "TRIGGER":
      if command[1] == "A":
        if command[2] == "TYPE":
          return
        if command[2] == "MODE":
          value = command[3]
          if value in ["AUTO", "NORM", "NORMAL"]:
            self.trigger_a_mode = value
          return
        if command[2] == "EDGE":
          if command[3] == "SOURCE":
            value = command[4]
            choices = [
                "AUX", "CH1", "CH2", "CH3", "CH4", "D0", "D1", "D2", "D3", "D4",
                "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14",
                "D15", "LINE", "RF"
            ]
            if value in choices:
              self.trigger_a_source = value
            return
          if command[3] == "COUPLING":
            value = command[4]
            choices = [
                "AC", "DC", "HFR", "HFREJ", "LFR", "LFREJ", "NOISE", "NOISEREJ"
            ]
            if value in choices:
              self.trigger_a_coupling = value
            return
          if command[3] == "SLOPE":
            value = command[4]
            choices = ["RIS", "RISE", "FALL", "EITH", "EITHER"]
            if value in choices:
              self.trigger_a_slope = value
            return
        if command[2] == "LEVEL":
          if command[3] in ["CH1", "CH2", "CH3", "CH4"]:
            if command[4] == "ECL":
              value = -1.3
            elif command[4] == "TTL":
              value = 1.4
            else:
              value = float(command[4])
            self.channels[command[3]]["trigger"] = value
            return
    if command[0] == "ACQUIRE":
      if command[1] == "MODE":
        value = command[2]
        choices = [
            "SAM", "SAMPLE", "PEAK", "PEAKDETECT", "HIR", "HIRES", "AVE",
            "AVERAGE", "ENV", "ENVELOPE"
        ]
        if value in choices:
          self.acquire_mode = value
        return
    if command[0] in ["CH1", "CH2", "CH3", "CH4"]:
      if command[1] == "SCALE":
        value = float(command[2])
        self.channels[command[0]]["scale"] = value
        return
      if command[1] == "POSITION":
        value = float(command[2])
        value = max(-5, min(5, value))
        self.channels[command[0]]["position"] = value
        return
      if command[1] == "OFFSET":
        value = float(command[2])
        self.channels[command[0]]["offset"] = value
        return
      if command[1] == "LABEL":
        value = " ".join(command[2:])
        if value[0] in ["'", '"']:
          value = value[1:]
        if value[-1] in ["'", '"']:
          value = value[:-1]
        self.channels[command[0]]["label"] = value
        return
      if command[1] == "BANDWIDTH":
        choices = [20e6, 200e6, 500e6]
        if command[2] == "FULL":
          self.channels[command[0]]["bandwidth"] = choices[-1]
        else:
          value = float(command[2])
          choices = sorted(choices, key=lambda c: (value - c)**2)
          self.channels[command[0]]["label"] = choices[0]
        return
      if command[1] == "TERMINATION":
        choices = [50, 75, 1e6]
        if command[2] == "MEG":
          self.channels[command[0]]["termination"] = 1e6
        elif command[2] in ["FIFTY", "FIF"]:
          self.channels[command[0]]["termination"] = 50
        else:
          value = float(command[2])
          choices = sorted(choices, key=lambda c: (value - c)**2)
          self.channels[command[0]]["termination"] = choices[0]
        return
      if command[1] == "INVERT":
        if command[2] in ["1", "ON"]:
          self.channels[command[0]]["invert"] = True
          return
        if command[2] in ["0", "OFF"]:
          self.channels[command[0]]["invert"] = False
          return
      if command[1] == "COUPLING":
        if command[2] in ["AC", "DC", "DCREJ", "DCREJECT"]:
          self.channels[command[0]]["coupling"] = command[2]
          return
      if command[1] == "PROBE":
        if command[2] == "GAIN":
          value = float(command[3])
          attenuations = [
              0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10,
              20, 50, 100, 200, 500, 1000, 2000, 2500, 5000, 10e3
          ]
          choices = [1 / a for a in attenuations]
          choices = sorted(choices, key=lambda c: (value - c)**2)
          self.channels[command[0]]["gain"] = choices[0]
          return
    if command[0] == "SELECT":
      if command[1] in ["CH1", "CH2", "CH3", "CH4"]:
        if command[2] in ["1", "ON"]:
          self.channels[command[1]]["active"] = True
          return
        if command[2] in ["0", "OFF"]:
          self.channels[command[1]]["active"] = False
          return

    raise KeyError(f"Unknown command {command}")

  def query(self, command: str) -> str:
    self.queue_tx.append(command)

    command = re.split(":| ", command.upper())
    if command[0] == "*IDN?":
      return "TEKTRONIX,MDO3054"
    if command[0] == "HORIZONTAL":
      if command[1] == "RECORDLENGTH?":
        return f"{self.record_length}"
      if command[1] == "SCALE?":
        return f"{self.horizontal_scale:.6E}"
      if command[1] == "SAMPLERATE?":
        return f"{self.sample_frequency:.6E}"
      if command[1] == "DELAY":
        if command[2] == "TIME?":
          return f"{self.horizontal_delay:.6E}"
    if command[0] == "TRIGGER":
      if command[1] == "A":
        if command[2] == "MODE?":
          return f"{self.trigger_a_mode}"
        if command[2] == "EDGE":
          if command[3] == "SOURCE?":
            return f"{self.trigger_a_source}"
          if command[3] == "COUPLING?":
            return f"{self.trigger_a_coupling}"
          if command[3] == "SLOPE?":
            return f"{self.trigger_a_slope}"
        if command[2] == "LEVEL":
          if command[3][:-1] in ["CH1", "CH2", "CH3", "CH4"]:
            return f"{self.channels[command[3][:-1]]['trigger']}"
    if command[0] == "ACQUIRE":
      if command[1] == "MODE?":
        return f"{self.acquire_mode}"
    if command[0] in ["CH1", "CH2", "CH3", "CH4"]:
      if command[1] == "SCALE?":
        return f"{self.channels[command[0]]['scale']:.6E}"
      if command[1] == "POSITION?":
        return f"{self.channels[command[0]]['position']:.6E}"
      if command[1] == "OFFSET?":
        return f"{self.channels[command[0]]['offset']:.6E}"
      if command[1] == "LABEL?":
        return f'"{self.channels[command[0]]["label"]}"'
      if command[1] == "BANDWIDTH?":
        return f"{self.channels[command[0]]['bandwidth']:.6E}"
      if command[1] == "TERMINATION?":
        return f"{self.channels[command[0]]['termination']:.6E}"
      if command[1] == "INVERT?":
        return f"{int(self.channels[command[0]]['invert'])}"
      if command[1] == "COUPLING?":
        return f"{self.channels[command[0]]['coupling']}"
      if command[1] == "PROBE":
        if command[2] == "GAIN?":
          return f"{self.channels[command[0]]['gain']:.6E}"
    if command[0] == "SELECT":
      if command[1][:-1] in ["CH1", "CH2", "CH3", "CH4"]:
        return f"{int(self.channels[command[1][:-1]]['active'])}"

    raise KeyError(f"Unknown query {command}")


class TestEquipmentTektronixMSO4000(unittest.TestCase):
  """Test Equipment Tektronix MSO4000
  """

  _TRY_REAL_SCOPE = False

  def setUp(self) -> None:
    super().setUp()

    mock_pyvisa.resources = {}
    mock_pyvisa.available = []
    equipment.pyvisa = mock_pyvisa

  def tearDown(self) -> None:
    super().tearDown()
    mock_pyvisa.resources = {}
    mock_pyvisa.available = []
    mock_pyvisa.no_pop = False

  def test_init(self):
    address = "USB::0x0000::0x0000:C000000::INSTR"

    mock_pyvisa.no_pop = True
    instrument = mock_pyvisa.Resource(address)

    instrument.query_map["*IDN?"] = "FAKE"
    self.assertRaises(ValueError, tektronix.MSO4000, address)
    self.assertRaises(ValueError, tektronix.MDO4000, address)
    self.assertRaises(ValueError, tektronix.MDO3000, address)
    e = tektronix.MSO4000(address, check_identity=False)
    e = tektronix.MDO4000(address, check_identity=False)
    e = tektronix.MDO3000(address, check_identity=False)

    instrument.query_map["*IDN?"] = "TEKTRONIX,MSO4104"
    e = tektronix.MSO4000(address)
    self.assertListEqual(e.channels, ["CH1", "CH2", "CH3", "CH4"])
    self.assertListEqual(instrument.queue_tx[-2:], ["HEADER OFF", "VERBOSE ON"])

    instrument.query_map["*IDN?"] = "TEKTRONIX,MSO4032"
    e = tektronix.MSO4000(address)
    self.assertListEqual(e.channels, ["CH1", "CH2"])
    self.assertListEqual(instrument.queue_tx[-2:], ["HEADER OFF", "VERBOSE ON"])

  def test_configure(self):
    # return # TODO remove
    e = None
    if self._TRY_REAL_SCOPE:
      utility.pyvisa = pyvisa
      equipment.pyvisa = pyvisa
      available = utility.get_available()
      for a in available:
        if a.startswith("USB::0x0699::0x0408::"):
          e = tektronix.MDO3000(a)
          break

    if e is None:
      utility.pyvisa = mock_pyvisa
      equipment.pyvisa = mock_pyvisa
      address = "USB::0x0000::0x0000:C000000::INSTR"

      mock_pyvisa.no_pop = True
      _ = MockMDO3054(address)

      e = tektronix.MDO3000(address)

    self.assertRaises(KeyError, e.configure, "FAKE", None)

    value = 2.5e9
    reply = e.configure("SAMPLE_RATE", (value, 1e6))
    self.assertEqual(value, reply)
    self.assertRaises(ValueError, e.configure, "SAMPLE_RATE", value)

    value = 1e-3
    reply = e.configure("TIME_OFFSET", value)
    self.assertEqual(value, reply)

    value = "AUTO"
    reply = e.configure("TRIGGER_MODE", value)
    self.assertEqual(value, reply)
    self.assertRaises(ValueError, e.configure, "TRIGGER_MODE", "FAKE")

    value = "CH2"
    reply = e.configure("TRIGGER_SOURCE", value)
    self.assertEqual(value, reply)
    self.assertRaises(ValueError, e.configure, "TRIGGER_SOURCE", "FAKE")

    value = "AC"
    reply = e.configure("TRIGGER_COUPLING", value)
    self.assertEqual(value, reply)
    self.assertRaises(ValueError, e.configure, "TRIGGER_COUPLING", "FAKE")

    value = "RISE"
    reply = e.configure("TRIGGER_POLARITY", value)
    self.assertEqual(value, reply)
    self.assertRaises(ValueError, e.configure, "TRIGGER_POLARITY", "FAKE")

    value = "AVERAGE"
    reply = e.configure("ACQUIRE_MODE", value)
    self.assertEqual(value, reply)
    self.assertRaises(ValueError, e.configure, "ACQUIRE_MODE", "FAKE")

  def test_configure_channel(self):
    e = None
    if self._TRY_REAL_SCOPE:
      utility.pyvisa = pyvisa
      equipment.pyvisa = pyvisa
      available = utility.get_available()
      for a in available:
        if a.startswith("USB::0x0699::0x0408::"):
          e = tektronix.MDO3000(a)
          break

    if e is None:
      utility.pyvisa = mock_pyvisa
      equipment.pyvisa = mock_pyvisa
      address = "USB::0x0000::0x0000:C000000::INSTR"

      mock_pyvisa.no_pop = True
      _ = MockMDO3054(address)

      e = tektronix.MDO3000(address)

    self.assertRaises(KeyError, e.configure_channel, "CHFake", "FAKE", None)

    value = 2
    reply = e.configure_channel("CH4", "SCALE", value)
    self.assertEqual(value, reply)
    self.assertRaises(KeyError, e.configure_channel, "CHFake", "SCALE", value)

    value = 2
    reply = e.configure_channel("CH4", "POSITION", value)
    self.assertEqual(value, reply)
    self.assertRaises(KeyError, e.configure_channel, "CHFake", "POSITION",
                      value)

    value = 0.1
    reply = e.configure_channel("CH4", "OFFSET", value)
    self.assertEqual(value, reply)
    self.assertRaises(KeyError, e.configure_channel, "CHFake", "OFFSET", value)

    value = "I2C CLOCK"
    reply = e.configure_channel("CH4", "LABEL", value)
    self.assertEqual(f'"{value}"', reply)
    self.assertRaises(KeyError, e.configure_channel, "CHFake", "LABEL", value)

    value = 500e6
    reply = e.configure_channel("CH4", "BANDWIDTH", value)
    self.assertEqual(value, reply)
    reply = e.configure_channel("CH4", "BANDWIDTH", "FULL")
    self.assertEqual(value, reply)
    self.assertRaises(KeyError, e.configure_channel, "CHFake", "BANDWIDTH",
                      value)

    value = 50
    reply = e.configure_channel("CH4", "TERMINATION", value)
    self.assertEqual(value, reply)
    reply = e.configure_channel("CH4", "TERMINATION", "FIFTY")
    self.assertEqual(value, reply)
    self.assertRaises(KeyError, e.configure_channel, "CHFake", "TERMINATION",
                      value)
    value = 1e6
    reply = e.configure_channel("CH4", "TERMINATION", value)
    self.assertEqual(value, reply)

    value = True
    reply = e.configure_channel("CH4", "INVERT", value)
    self.assertEqual(value, reply)
    self.assertRaises(KeyError, e.configure_channel, "CHFake", "INVERT", value)

    value = 10
    reply = e.configure_channel("CH4", "PROBE_GAIN", value)
    self.assertEqual(value, reply)
    self.assertRaises(KeyError, e.configure_channel, "CHFake", "PROBE_GAIN",
                      value)

    value = 10
    reply = e.configure_channel("CH4", "PROBE_ATTENUATION", value)
    self.assertEqual(value, reply)
    self.assertRaises(KeyError, e.configure_channel, "CHFake",
                      "PROBE_ATTENUATION", value)

    value = "AC"
    reply = e.configure_channel("CH4", "COUPLING", value)
    self.assertEqual(value, reply)
    self.assertRaises(KeyError, e.configure_channel, "CHFake", "COUPLING",
                      value)
    self.assertRaises(ValueError, e.configure_channel, "CH4", "COUPLING",
                      "FAKE")

    value = True
    reply = e.configure_channel("CH4", "ACTIVE", value)
    self.assertEqual(value, reply)

    value = 1.4
    reply = e.configure_channel("CH4", "TRIGGER_LEVEL", value)
    self.assertEqual(value, reply)
    reply = e.configure_channel("CH4", "TRIGGER_LEVEL", "TTL")
    self.assertEqual(value, reply)
