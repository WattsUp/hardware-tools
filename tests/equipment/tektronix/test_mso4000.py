"""Test module hardware_tools.equipment.tektronix.family_mso4000
"""

import io
import re
import time
from typing import Any, Callable, List
from unittest import mock

import numpy as np
import pyvisa

import sys

from hardware_tools.equipment import equipment, utility, scope
from hardware_tools.equipment import tektronix

from tests import base
from tests.equipment import mock_pyvisa


class MockMSO4104B(mock_pyvisa.Resource):
  """Mock Tektronix MSO4104B
  """

  def __init__(self, resource_manager: pyvisa.ResourceManager,
               address: str) -> None:
    super().__init__(resource_manager, address)

    self.reset()

  def reset(self) -> None:
    # Taken from Appendix C: Factory Defaults in Programming Manual
    # Tuples are (type [or callable to convert str to value], value)
    self._running = True
    self.query_map = {
        "*IDN": "TEKTRONIX,MSO4104",
        "ACQUIRE": {
            "MODE": (tektronix.parse_sample_mode, tektronix.SampleMode.SAMPLE),
            "NUMAVG": (int, 16),
            "NUMENV": (lambda x: x if x == "INFINITE" else int(x), "INFINITE"),
            "NUMACQ": 0,
            "STATE": (self.acquire_state, self.acquire_state),
            "STOPAFTER": (str, "RUNSTOP")
        },
        "CONFIGURATION": {
            "ANALOG": {
                "NUMCHANNELS": 1,
                "BANDWIDTH": 1e9
            },
            "DIGITAL": {
                "NUMCHANNELS": 1
            },
            "AUXIN": 0
        },
        "HEADER": (lambda x: x in ["ON", "1"], False),
        "VERBOSE": (lambda x: x in ["ON", "1"], True),
        "HORIZONTAL": {
            "RECORDLENGTH": (int, 10000),
            "SAMPLERATE":
                lambda: (self.query_map["HORIZONTAL"]["RECORDLENGTH"][1] /
                         (10 * self.query_map["HORIZONTAL"]["SCALE"][1])),
            "SCALE": (float, 4e-6),
            "DELAY": {
                "MODE": (lambda x: x in ["ON", "1"], True),
                "TIME": (float, 0)
            }
        },
        "TRIGGER": {
            "A": {
                "EDGE": {
                    "SOURCE": (str, "CH1"),
                    "SLOPE": (tektronix.parse_polarity,
                              tektronix.EdgePolarity.RISING),
                    "COUPLING": (str, "DC")
                },
                "HOLDOFF": {
                    "TIME": (float, 20e-9),
                },
                "LEVEL": {
                    "CH1": (float, 0.0)
                },
                "MODE": (str, "AUTO"),
                "PULSE": {
                    "CLASS": (str, "WIDTH")
                },
                "PULSEWIDTH": {
                    "SOURCE": (str, "CH1"),
                    "WIDTH": (float, 8e-9),
                    "HIGHLIMIT": (float, 12e-9),
                    "LOWLIMIT": (float, 8e-9),
                    "POLARITY": (tektronix.parse_polarity,
                                 tektronix.EdgePolarity.RISING),
                    "WHEN":
                        (tektronix.parse_comparison, tektronix.Comparison.LESS)
                },
                "TIMEOUT": {
                    "SOURCE": (str, "CH1"),
                    "TIME": (float, 8e-9),
                    "POLARITY": (tektronix.parse_polarity,
                                 tektronix.EdgePolarity.RISING)
                },
                "TYPE": (str, "EDGE")
            },
            "STATE": self.trigger_state
        }
    }

  def write(self, command: str) -> None:
    if command == "TRIGGER FORCE":
      self.trigger_state(triggered=True)
      return
    return super().write(command)

  def query_str(self, keys: List[str], value: Any) -> str:
    s = None
    if isinstance(value, tektronix.SampleMode):
      if value == tektronix.SampleMode.SAMPLE:
        s = "SAMPLE"
      elif value == tektronix.SampleMode.AVERAGE:
        s = "AVERAGE"
      elif value == tektronix.SampleMode.ENVELOPE:
        s = "ENVELOPE"
    elif isinstance(value, tektronix.EdgePolarity):
      if keys[-2:] == ["EDGE", "SLOPE"]:
        if value == tektronix.EdgePolarity.RISING:
          s = "RISE"
        elif value == tektronix.EdgePolarity.FALLING:
          s = "FALL"
        elif value == tektronix.EdgePolarity.BOTH:
          s = "EITHER"
      elif keys[-2:] == ["TIMEOUT", "POLARITY"]:
        if value == tektronix.EdgePolarity.RISING:
          s = "STAYSHIGH"
        elif value == tektronix.EdgePolarity.FALLING:
          s = "STAYSLOW"
        elif value == tektronix.EdgePolarity.BOTH:
          s = "EITHER"
      elif keys[-2:] == ["PULSEWIDTH", "POLARITY"]:
        if value == tektronix.EdgePolarity.RISING:
          s = "POSITIVE"
        elif value == tektronix.EdgePolarity.FALLING:
          s = "NEGATIVE"
    elif isinstance(value, tektronix.Comparison):
      if value in [tektronix.Comparison.LESS, tektronix.Comparison.LESSEQUAL]:
        s = "LESSTHAN"
      elif value in [tektronix.Comparison.MORE, tektronix.Comparison.MOREEQUAL]:
        s = "MORETHAN"
      elif value == tektronix.Comparison.EQUAL:
        s = "EQUAL"
      elif value == tektronix.Comparison.UNEQUAL:
        s = "UNEQUAL"
      elif (value
            in [tektronix.Comparison.WITHIN, tektronix.Comparison.WITHININC]):
        s = "WITHIN"
      elif (value
            in [tektronix.Comparison.OUTSIDE, tektronix.Comparison.OUTSIDEINC]):
        s = "OUTSIDE"
    else:
      s = super().query_str(keys, value)

    if self.query_map["HEADER"][1]:
      return ":".join(keys) + " " + s
    else:
      return s

  def trigger_state(self, triggered=False) -> str:
    if self._running:
      # Acquisition engine is running
      t_type = self.query_map["TRIGGER"]["A"]["TYPE"][1]
      if t_type == "EDGE":
        src = self.query_map["TRIGGER"]["A"]["EDGE"]["SOURCE"][1]
        level = self.query_map["TRIGGER"]["A"]["LEVEL"][src][1]
        if 0 <= level <= 2.5:
          triggered = True
      if triggered:
        self.query_map["ACQUIRE"]["NUMACQ"] += 1

      if self.query_map["ACQUIRE"]["STOPAFTER"][1] == "SEQUENCE":
        # Single capture will stop acquisition if triggered
        if triggered:
          self._running = False
          return "SAVE"
        else:
          return "READY"
      elif self.query_map["TRIGGER"]["A"]["MODE"][1] == "AUTO":
        # Auto mode is always AUTO
        return "AUTO"
      else:
        # Normal mode is always READY
        # (technically ARMED->TRIGGER->READY->ARMED loop)
        if triggered:
          return "TRIGGER"
        else:
          return "READY"
    else:
      return "SAVE"

  def acquire_state(self, value=None) -> Callable:
    if value is None:
      # Call trigger_state to update state machine
      self.trigger_state()
      return self._running
    else:
      self.query_map["ACQUIRE"]["NUMACQ"] = 0
      self._running = value in ["ON", "1", "RUN"]

    # Since writes assign the result to the value, need to return this
    # function so it can be called in the future
    return self.acquire_state


#     self.record_length = 1000
#     self.horizontal_scale = 1e-9
#     self.horizontal_delay = 0
#     self.sample_frequency = min(
#         2.5e9, self.record_length / (10 * self.horizontal_scale))
#     self.trigger_a_mode = "AUTO"
#     self.trigger_a_source = "CH1"
#     self.trigger_a_coupling = "DC"
#     self.trigger_a_slope = "FALL"
#     self.trigger_state = "AUTO"
#     self.acquire_mode = "SAMPLE"
#     self.acquire_single = False
#     self.acquire_state_running = True
#     self.acquire_num = 0
#     self.data_source = "CH1"
#     self.data_start = 1
#     self.data_stop = self.record_length

#     self.waveform_amp = 1.25
#     self.waveform_offset = 1.25

#     self.channels = {
#         c: {
#             "scale": 0.1,
#             "position": 0,
#             "offset": 0,
#             "label": c,
#             "bandwidth": 500e6,
#             "termination": 1e6,
#             "invert": False,
#             "gain": 1,
#             "coupling": "DC",
#             "trigger": 0
#         } for c in ["CH1", "CH2", "CH3", "CH4"]
#     }

#   def write(self, command: str) -> None:
#     self.queue_tx.append(command)
#     command = re.split(":| ", command.upper())
#     if command[0] in ["HEADER", "VERBOSE", "CLEARMENU"]:
#       return
#     if command[0] == "HORIZONTAL":
#       if command[1] == "RECORDLENGTH":
#         value = int(command[2])
#         choices = [1e3, 10e3, 100e3, 1e6, 5e6, 10e6]
#         choices = sorted(choices, key=lambda c: (value - c)**2)
#         self.record_length = int(choices[0])
#         self.sample_frequency = min(
#             2.5e9, self.record_length / (10 * self.horizontal_scale))
#         return
#       if command[1] == "SCALE":
#         value = float(command[2])
#         choices = []
#         for i in range(-9, 2):
#           if 1 * 10**i == 0.1e-9 * self.record_length:
#             choices.append(0.8 * 10**i)
#           else:
#             choices.append(1 * 10**i)
#           choices.append(2 * 10**i)
#           choices.append(4 * 10**i)
#         if self.record_length >= 10e3:
#           choices.extend([100, 200, 400])
#         if self.record_length >= 100e3:
#           choices.append(1000)
#         choices = sorted(choices, key=lambda c: (value - c)**2)
#         self.horizontal_scale = choices[0]
#         self.sample_frequency = min(
#             2.5e9, self.record_length / (10 * self.horizontal_scale))
#         return
#       if command[1] == "DELAY":
#         if command[2] == "MODE":
#           return
#         if command[2] == "TIME":
#           value = float(command[3])
#           sample_t = 1 / self.sample_frequency
#           value = round(value / sample_t) * sample_t
#           min_delay = -sample_t * self.record_length
#           self.horizontal_delay = min(5e3, max(min_delay, value))
#           return
#     if command[0] == "TRIGGER":
#       if command[1] == "FORCE":
#         return
#       if command[1] == "A":
#         if command[2] == "TYPE":
#           return
#         if command[2] == "MODE":
#           value = command[3]
#           if value in ["AUTO", "NORM", "NORMAL"]:
#             self.trigger_a_mode = value
#           return
#         if command[2] == "EDGE":
#           if command[3] == "SOURCE":
#             value = command[4]
#             choices = [
#                 "AUX", "CH1", "CH2", "CH3", "CH4", "D0", "D1", "D2", "D3", "D4",
#                 "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14",
#                 "D15", "LINE", "RF"
#             ]
#             if value in choices:
#               self.trigger_a_source = value
#               return
#           if command[3] == "COUPLING":
#             value = command[4]
#             choices = [
#                 "AC", "DC", "HFR", "HFREJ", "LFR", "LFREJ", "NOISE", "NOISEREJ"
#             ]
#             if value in choices:
#               self.trigger_a_coupling = value
#             return
#           if command[3] == "SLOPE":
#             value = command[4]
#             choices = ["RIS", "RISE", "FALL", "EITH", "EITHER"]
#             if value in choices:
#               self.trigger_a_slope = value
#             return
#         if command[2] == "LEVEL":
#           if command[3] in ["CH1", "CH2", "CH3", "CH4"]:
#             if command[4] == "ECL":
#               value = -1.3
#             elif command[4] == "TTL":
#               value = 1.4
#             else:
#               value = float(command[4])
#             self.channels[command[3]]["trigger"] = value
#             return
#     if command[0] == "ACQUIRE":
#       if command[1] == "MODE":
#         value = command[2]
#         choices = [
#             "SAM", "SAMPLE", "PEAK", "PEAKDETECT", "HIR", "HIRES", "AVE",
#             "AVERAGE", "ENV", "ENVELOPE"
#         ]
#         if value in choices:
#           self.acquire_mode = value
#           return
#       if command[1] == "STATE":
#         if command[2] in ["1", "ON", "RUN"]:
#           self.acquire_num = 1
#           if self.acquire_single:
#             self.trigger_state = "SAVE"
#             self.acquire_state_running = False
#           else:
#             self.trigger_state = "TRIGGER"
#             self.acquire_state_running = True
#           return
#         if command[2] in ["0", "OFF", "STOP"]:
#           self.acquire_state_running = False
#           self.trigger_state = "SAVE"
#           return
#       if command[1] == "STOPAFTER":
#         if command[2] == "SEQUENCE":
#           self.acquire_single = True
#           return
#         if command[2] == "RUNSTOP":
#           self.acquire_single = False
#           return
#     if command[0] in ["CH1", "CH2", "CH3", "CH4"]:
#       if command[1] == "SCALE":
#         value = float(command[2])
#         self.channels[command[0]]["scale"] = value
#         return
#       if command[1] == "POSITION":
#         value = float(command[2])
#         value = max(-5, min(5, value))
#         self.channels[command[0]]["position"] = value
#         return
#       if command[1] == "OFFSET":
#         value = float(command[2])
#         self.channels[command[0]]["offset"] = value
#         return
#       if command[1] == "LABEL":
#         value = " ".join(command[2:])
#         if value[0] in ["'", '"']:
#           value = value[1:]
#         if value[-1] in ["'", '"']:
#           value = value[:-1]
#         self.channels[command[0]]["label"] = value
#         return
#       if command[1] == "BANDWIDTH":
#         choices = [20e6, 200e6, 500e6]
#         if command[2] == "FULL":
#           self.channels[command[0]]["bandwidth"] = choices[-1]
#         else:
#           value = float(command[2])
#           choices = sorted(choices, key=lambda c: (value - c)**2)
#           self.channels[command[0]]["label"] = choices[0]
#         return
#       if command[1] == "TERMINATION":
#         choices = [50, 75, 1e6]
#         if command[2] == "MEG":
#           self.channels[command[0]]["termination"] = 1e6
#         elif command[2] in ["FIFTY", "FIF"]:
#           self.channels[command[0]]["termination"] = 50
#         else:
#           value = float(command[2])
#           choices = sorted(choices, key=lambda c: (value - c)**2)
#           self.channels[command[0]]["termination"] = choices[0]
#         return
#       if command[1] == "INVERT":
#         if command[2] in ["1", "ON"]:
#           self.channels[command[0]]["invert"] = True
#           return
#         if command[2] in ["0", "OFF"]:
#           self.channels[command[0]]["invert"] = False
#           return
#       if command[1] == "COUPLING":
#         if command[2] in ["AC", "DC", "DCREJ", "DCREJECT"]:
#           self.channels[command[0]]["coupling"] = command[2]
#           return
#       if command[1] == "PROBE":
#         if command[2] == "GAIN":
#           value = float(command[3])
#           attenuations = [
#               0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10,
#               20, 50, 100, 200, 500, 1000, 2000, 2500, 5000, 10e3
#           ]
#           choices = [1 / a for a in attenuations]
#           choices = sorted(choices, key=lambda c: (value - c)**2)
#           self.channels[command[0]]["gain"] = choices[0]
#           return
#     if command[0] == "SELECT":
#       if command[1] in ["CH1", "CH2", "CH3", "CH4"]:
#         if command[2] in ["1", "ON"]:
#           self.channels[command[1]]["active"] = True
#           return
#         if command[2] in ["0", "OFF"]:
#           self.channels[command[1]]["active"] = False
#           return
#     if command[0] == "DATA":
#       if command[1] == "SOURCE":
#         value = command[2]
#         choices = [
#             "CH1", "CH2", "CH3", "CH4", "MATH", "REF1", "REF2", "REF3", "REF4",
#             "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
#             "D11", "D12", "D13", "D14", "D15", "DIGITAL", "RF_AMPLITUDE",
#             "RF_FREQUENCY", "RF_PHASE", "RF_NORMAL", "RF_AVERAGE", "RF_MAXHOLD",
#             "RF_MINHOLD"
#         ]
#         if value in choices:
#           self.data_source = value
#           return
#       if command[1] == "START":
#         self.data_start = max(1, int(float(command[2])))
#         return
#       if command[1] == "STOP":
#         self.data_stop = max(1, int(float(command[2])))
#         return
#       if command[1] == "WIDTH":
#         if int(command[2]) != 1:
#           raise ValueError("Mock MDO3054 only knows DATA:WIDTH 1")
#         return
#       if command[1] == "ENCDG":
#         if command[2] != "FASTEST":
#           raise ValueError("Mock MDO3054 only knows DATA:ENCDG FASTEST")
#         return
#     if command[0] == "WAVFRM?":
#       points = int(min(self.record_length,
#                        self.data_stop - self.data_start + 1))
#       x_incr = 1 / self.sample_frequency
#       x_zero = -x_incr * points / 2 + self.horizontal_delay
#       x_unit = "s"
#       y_mult = 10 * self.channels[self.data_source]["scale"] / 250
#       y_off = self.channels[self.data_source]["position"] / 10 * 250
#       y_zero = self.channels[self.data_source]["offset"]
#       y_unit = "V"
#       wf_id = (f"{self.data_source}, "
#                f"{self.channels[self.data_source]['coupling']} coupling, "
#                f"{self.channels[self.data_source]['scale']}V/div, "
#                f"{self.horizontal_scale}s/div, "
#                f"{self.record_length} points")

#       x = x_zero + x_incr * np.arange(points).astype(np.float32)
#       y_real = self.waveform_amp * np.sin(
#           x * 2 * np.pi * 1e3) + self.waveform_offset
#       y = np.clip(np.floor((y_real - y_zero) / y_mult + y_off + 0.5), -127, 127)

#       waveform = y.astype(">b").tobytes()
#       n_bytes = f"{len(waveform)}"

#       real_data = (f':WFMOUTPRE:WFID "{wf_id}";'
#                    f"NR_PT {points};"
#                    f'XUNIT "{x_unit}";XINCR {x_incr:.4E};XZERO {x_zero:.4E};'
#                    f'YUNIT "{y_unit}";YMULT {y_mult:.4E};'
#                    f"YOFF {y_off:.4E};YZERO {y_zero:.4E};"
#                    "BYT_OR MSB;BYT_NR 1;BN_FMT RI;"
#                    f":CURVE #{len(n_bytes)}{n_bytes}")
#       real_data = real_data.encode(encoding="ascii") + waveform + b"\n"

#       self.queue_rx.append(real_data)
#       return

#     raise KeyError(f"Unknown command {command}")

#   def query(self, command: str) -> str:
#     self.queue_tx.append(command)

#     command = re.split(":| ", command.upper())
#     if command[0] == "*IDN?":
#       return "TEKTRONIX,MDO3054"
#     if command[0] == "HORIZONTAL":
#       if command[1] == "RECORDLENGTH?":
#         return f"{self.record_length}"
#       if command[1] == "SCALE?":
#         return f"{self.horizontal_scale:.6E}"
#       if command[1] == "SAMPLERATE?":
#         return f"{self.sample_frequency:.6E}"
#       if command[1] == "DELAY":
#         if command[2] == "TIME?":
#           return f"{self.horizontal_delay:.6E}"
#     if command[0] == "TRIGGER":
#       if command[1] == "STATE?":
#         return self.trigger_state
#       if command[1] == "A":
#         if command[2] == "MODE?":
#           return f"{self.trigger_a_mode}"
#         if command[2] == "EDGE":
#           if command[3] == "SOURCE?":
#             return f"{self.trigger_a_source}"
#           if command[3] == "COUPLING?":
#             return f"{self.trigger_a_coupling}"
#           if command[3] == "SLOPE?":
#             return f"{self.trigger_a_slope}"
#         if command[2] == "LEVEL":
#           if command[3][:-1] in ["CH1", "CH2", "CH3", "CH4"]:
#             return f"{self.channels[command[3][:-1]]['trigger']}"
#     if command[0] == "ACQUIRE":
#       if command[1] == "STATE?":
#         return "1" if self.acquire_state_running else "0"
#       if command[1] == "NUMACQ?":
#         if not self.acquire_single:
#           self.acquire_num += 1
#         return f"{self.acquire_num}"
#       if command[1] == "MODE?":
#         return f"{self.acquire_mode}"
#     if command[0] in ["CH1", "CH2", "CH3", "CH4"]:
#       if command[1] == "SCALE?":
#         return f"{self.channels[command[0]]['scale']:.6E}"
#       if command[1] == "POSITION?":
#         return f"{self.channels[command[0]]['position']:.6E}"
#       if command[1] == "OFFSET?":
#         return f"{self.channels[command[0]]['offset']:.6E}"
#       if command[1] == "LABEL?":
#         return f'"{self.channels[command[0]]["label"]}"'
#       if command[1] == "BANDWIDTH?":
#         return f"{self.channels[command[0]]['bandwidth']:.6E}"
#       if command[1] == "TERMINATION?":
#         return f"{self.channels[command[0]]['termination']:.6E}"
#       if command[1] == "INVERT?":
#         return f"{int(self.channels[command[0]]['invert'])}"
#       if command[1] == "COUPLING?":
#         return f"{self.channels[command[0]]['coupling']}"
#       if command[1] == "PROBE":
#         if command[2] == "GAIN?":
#           return f"{self.channels[command[0]]['gain']:.6E}"
#     if command[0] == "SELECT":
#       if command[1][:-1] in ["CH1", "CH2", "CH3", "CH4"]:
#         return f"{int(self.channels[command[1][:-1]]['active'])}"

#     raise KeyError(f"Unknown query {command}")


class TestMSO4000(base.TestBase):
  """Test Equipment Tektronix MSO4000
  """

  # When true, connect CH1 to probe compensation, and all others open
  _TRY_REAL_SCOPE = False

  def setUp(self) -> None:
    super().setUp()

    mock_pyvisa.resources = {}
    mock_pyvisa.available = []

    self._scope: tektronix.MSO4000Family = None

  def tearDown(self) -> None:
    super().tearDown()

    mock_pyvisa.resources = {}
    mock_pyvisa.available = []
    mock_pyvisa.no_pop = False

    if self._scope is not None:
      self._scope.close()

  def connect(self) -> tektronix.MSO4000Family:
    e = None
    if self._TRY_REAL_SCOPE:
      available = utility.get_available()
      for a in available:
        if a.startswith("USB::0x0699::"):
          e = tektronix.MSO4000Family(a)

    if e is None:
      address = "USB::0x0000::0x0000:C000000::INSTR"

      mock_pyvisa.no_pop = True
      rm = mock_pyvisa.ResourceManager()
      _ = MockMSO4104B(rm, address)

      e = tektronix.MSO4000Family(address, rm=rm)
    else:
      time.sleep = self._original_sleep
      e.reset()
    self._scope = e
    return e

  def test_init(self):
    address = "USB::0x0000::0x0000:C000000::INSTR"

    mock_pyvisa.no_pop = True
    rm = mock_pyvisa.ResourceManager()
    instrument = MockMSO4104B(rm, address)

    instrument.query_map["*IDN"] = "FAKE"
    self.assertRaises(ValueError, tektronix.MSO4000Family, address, rm=rm)
    instrument.query_map["*IDN"] = "TEKTRONIX,MSO2010"
    self.assertRaises(ValueError, tektronix.MSO4000Family, address, rm=rm)

    instrument.query_map["*IDN"] = "TEKTRONIX,FAKE012"
    self.assertRaises(ValueError, tektronix.MSO4000Family, address, rm=rm)
    instrument.query_map["*IDN"] = "TEKTRONIX,MSO4104"
    e = tektronix.MSO4000Family(address, rm=rm, name="Emulated MSO4104B")
    self.assertEqual(e.max_bandwidth, 1000e6)
    self.assertListEqual(list(e._channels.keys()), [1])  # pylint: disable=protected-access
    self.assertListEqual(list(e._digitals.keys()), [0])  # pylint: disable=protected-access

  def test_configure_generic(self):
    e = self.connect()

    value = 1e6
    e.sample_rate = value
    self.assertEqual(value, e.sample_rate)

    value = tektronix.SampleMode.AVERAGE
    e.sample_mode = value
    self.assertEqual(value, e.sample_mode)
    value = 16
    e.sample_mode_n = value
    self.assertEqual(value, e.sample_mode_n)
    value = tektronix.SampleMode.ENVELOPE
    e.sample_mode = value
    self.assertEqual(value, e.sample_mode)
    value = 16
    e.sample_mode_n = value
    self.assertEqual(value, e.sample_mode_n)
    value = None
    self.assertRaises(ValueError, setattr, e, "sample_mode", value)
    e.send("ACQUIRE:MODE HIRES")
    self.assertEqual(None, e.sample_mode)
    value = tektronix.SampleMode.SAMPLE
    e.sample_mode = value
    self.assertEqual(value, e.sample_mode)
    self.assertEqual(1, e.sample_mode_n)

    value = 2e-3
    e.time_scale = value
    self.assertEqual(value, e.time_scale)

    value = 0.2e-3
    e.time_offset = value
    self.assertEqual(value, e.time_offset)

    value = 1000
    e.time_points = value
    self.assertEqual(value, e.time_points)

  def test_configure_trigger(self):
    e = self.connect()

    t = tektronix.TriggerEdge("CH1", 0.5)
    e.trigger = t
    result = e.trigger
    self.assertEqual(t.src, result.src)
    self.assertEqual(t.level, result.level)
    self.assertEqual(t.slope, result.slope)
    self.assertEqual(t.dc_coupling, result.dc_coupling)
    self.assertEqual(t.holdoff, result.holdoff)

    t = tektronix.TriggerEdge("CH1", 0.5, dc_coupling=False)
    e._aux = False  # pylint: disable=protected-access
    e.trigger = t
    result = e.trigger
    self.assertEqual(t.src, result.src)
    self.assertEqual(t.level, result.level)
    self.assertEqual(t.slope, result.slope)
    self.assertEqual(t.dc_coupling, result.dc_coupling)
    self.assertEqual(t.holdoff, result.holdoff)

    t = tektronix.TriggerEdge("FAKE", 0.5)
    self.assertRaises(ValueError, setattr, e, "trigger", t)

    t = tektronix.TriggerEdgeTimeout("CH1", 0.5, 1e-3)
    e._aux = True  # pylint: disable=protected-access
    e.trigger = t
    result = e.trigger
    self.assertEqual(t.src, result.src)
    self.assertEqual(t.level, result.level)
    self.assertEqual(t.slope, result.slope)
    self.assertEqual(t.timeout, result.timeout)
    self.assertEqual(t.holdoff, result.holdoff)

    t = tektronix.TriggerEdgeTimeout("FAKE", 0.5, 1e-3)
    e._aux = False  # pylint: disable=protected-access
    self.assertRaises(ValueError, setattr, e, "trigger", t)

    t = tektronix.TriggerPulseWidth("CH1", 0.5, 0.5e-3,
                                    tektronix.Comparison.EQUAL)
    e._aux = True  # pylint: disable=protected-access
    e.trigger = t
    result = e.trigger
    self.assertEqual(t.src, result.src)
    self.assertEqual(t.level, result.level)
    self.assertEqual(t.width, result.width)
    self.assertEqual(t.comparison, result.comparison)
    self.assertEqual(t.positive, result.positive)
    self.assertEqual(t.holdoff, result.holdoff)

    t = tektronix.TriggerPulseWidth("CH1",
                                    0.5, (0.4e-3, 0.6e-3),
                                    tektronix.Comparison.WITHIN,
                                    positive=False)
    e.trigger = t
    result = e.trigger
    self.assertEqual(t.src, result.src)
    self.assertEqual(t.level, result.level)
    self.assertEqual(t.width, result.width)
    self.assertEqual(t.comparison, result.comparison)
    self.assertEqual(t.positive, result.positive)
    self.assertEqual(t.holdoff, result.holdoff)

    t = tektronix.TriggerPulseWidth("FAKE", 0.5, 1e-3,
                                    tektronix.Comparison.EQUAL)
    e._aux = False  # pylint: disable=protected-access
    self.assertRaises(ValueError, setattr, e, "trigger", t)

    t = tektronix.TriggerPulseWidth("CH1", 0.5, 1e-3,
                                    tektronix.Comparison.OUTSIDE)
    e._aux = False  # pylint: disable=protected-access
    self.assertRaises(ValueError, setattr, e, "trigger", t)

    e.send("TRIGGER:A:TYPE PULSE")
    e.send("TRIGGER:A:PULSE:CLASS RUNT")
    self.assertEqual(None, e.trigger)

    e.send("TRIGGER:A:TYPE BUS")
    self.assertEqual(None, e.trigger)

    self.assertRaises(ValueError, setattr, e, "trigger", None)

  def test_run_stop(self):
    e = self.connect()

    # Setup for probe composition signal
    e.trigger = tektronix.TriggerEdge("CH1", 1.25)

    e.run(normal=False)
    result = e.ask("TRIGGER:STATE?")
    self.assertIn(result, ["ARMED", "AUTO", "TRIGGER", "READY"])

    e.stop()
    result = e.ask("TRIGGER:STATE?")
    self.assertEqual(result, "SAVE")

    e.trigger = tektronix.TriggerEdge("CH1", -1.25)
    e.run(normal=True)
    result = e.ask("TRIGGER:STATE?")
    # Real scope usually is still arming without any delays
    self.assertIn(result, ["ARMED", "READY"])

  def test_single(self):
    e = self.connect()

    # Setup for probe composition signal
    e.trigger = tektronix.TriggerEdge("CH1", 1.25)

    e.single()
    result = e.ask("TRIGGER:STATE?")
    self.assertEqual(result, "SAVE")

    e.trigger = tektronix.TriggerEdge("CH1", -1.25)
    self.assertRaises(TimeoutError, e.single)

    e.single(force=True)
    result = e.ask("TRIGGER:STATE?")
    self.assertEqual(result, "SAVE")

    def force_and_wait():
      e.ask_and_wait("TRIGGER:STATE?", ["READY"])
      e.force()

    e.single(trigger_cmd=force_and_wait)
    result = e.ask("TRIGGER:STATE?")
    self.assertEqual(result, "SAVE")

  def test_channel(self):
    e = self.connect()

    # Setup for probe composition signal
    e.trigger = tektronix.TriggerEdge("CH1", 1.25)

    c = e.ch(1)

    e.single(force=True)
    data, info = c.read_waveform()
    self.assertFalse(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])
    self.assertEqual(data.shape[1], e.time_points)

    value = 3
    c.position = value
    self.assertEqual(value, c.position)
    e.single(force=True)
    _, info = c.read_waveform()
    self.assertTrue(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])

    value = "CLOCK"
    c.label = value
    self.assertEqual(value, c.label)
    value = "CLOCK and a bunch of other stuff to be longer"
    c.label = value
    self.assertEqual(value[:30], c.label)

    value = False
    c.active = value
    self.assertEqual(value, c.active)

    value = 5
    c.scale = value
    self.assertEqual(value, c.scale)
    e.single(force=True)
    _, info = c.read_waveform()
    self.assertFalse(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])

    value = 20e6
    c.bandwidth = value
    self.assertEqual(value, c.bandwidth)

    value = False
    c.dc_coupling = value
    self.assertEqual(value, c.dc_coupling)
    value = True
    c.dc_coupling = value
    self.assertEqual(value, c.dc_coupling)

    value = 10e-9
    c.deskew = value
    self.assertEqual(value, c.deskew)

    value = True
    c.inverted = value
    self.assertEqual(value, c.inverted)

    value = 1
    c.offset = value
    self.assertEqual(value, c.offset)

    value = 1e6
    c.termination = value
    self.assertEqual(value, c.termination)

    value = 0.1
    c.probe_gain = value
    self.assertEqual(value, c.probe_gain)

    d = e.d(0)
    value = 0.5
    d.threshold = value
    self.assertEqual(value, d.threshold)
