"""Test module hardware_tools.equipment.tektronix.family_mso4000
"""

import io
import re
import time
from unittest import mock

import numpy as np
import pyvisa

import sys
# sys.modules["pyvisa"] = __import__("tests.equipment.mock_pyvisa", fromlist=[None])
# # sys.modules["pyvisa.resources"] = __import__("pyvisa.resources")
# import pyvisa as mock_pyvisa

from hardware_tools.equipment import equipment, utility, scope
from hardware_tools.equipment import tektronix

from tests import base
from tests.equipment import mock_pyvisa

# class MockMDO3054(mock_pyvisa.Resource):
#   """Mock Tektronix MDO3054
#   """

#   def __init__(self, address: str) -> None:
#     super().__init__(address)
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
  _TRY_REAL_SCOPE = True

  def setUp(self) -> None:
    super().setUp()

    mock_pyvisa.resources = {}
    mock_pyvisa.available = []

  def tearDown(self) -> None:
    super().tearDown()

    mock_pyvisa.resources = {}
    mock_pyvisa.available = []
    mock_pyvisa.no_pop = False

  def test_init(self):
    address = "USB::0x0000::0x0000:C000000::INSTR"

    mock_pyvisa.no_pop = True
    rm = mock_pyvisa.ResourceManager()
    instrument = mock_pyvisa.Resource(rm, address)

    instrument.query_map["*IDN?"] = "FAKE"
    self.assertRaises(ValueError, tektronix.MSO4000Family, address, rm=rm)
    instrument.query_map["*IDN?"] = "TEKTRONIX,MSO2010"
    self.assertRaises(ValueError, tektronix.MSO4000Family, address, rm=rm)

    instrument.query_map["CONFIGURATION:ANALOG:NUMCHANNELS?"] = "1"
    instrument.query_map["CONFIGURATION:ANALOG:BANDWIDTH?"] = "1.0000E+9"
    instrument.query_map["CONFIGURATION:DIGITAL:NUMCHANNELS?"] = "1"
    instrument.query_map["CONFIGURATION:AUXIN?"] = "0"

    instrument.query_map["*IDN?"] = "TEKTRONIX,FAKE012"
    self.assertRaises(ValueError, tektronix.MSO4000Family, address, rm=rm)
    instrument.query_map["*IDN?"] = "TEKTRONIX,MSO4104"
    e = tektronix.MSO4000Family(address, rm=rm, name="Emulated MSO4104B")
    self.assertEqual(e.max_bandwidth, 1000e6)
    self.assertListEqual(list(e._channels.keys()), [1])  # pylint: disable=protected-access
    self.assertListEqual(list(e._digitals.keys()), [0])  # pylint: disable=protected-access

  def test_configure_generic(self):
    try:
      e = None
      if self._TRY_REAL_SCOPE:
        available = utility.get_available()
        for a in available:
          if a.startswith("USB::0x0699::"):
            e = tektronix.MSO4000Family(a)

      if e is None:
        address = "USB::0x0000::0x0000:C000000::INSTR"

        mock_pyvisa.no_pop = True
        # _ = MockMDO3054(address) # TODO (WattsUp) Fix MockMDO3054

        rm = mock_pyvisa.ResourceManager()
        e = tektronix.MSO4000Family(address, rm=rm)
      else:
        time.sleep = self._original_sleep
        e.reset()

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

    finally:
      if e is not None:
        e.close()

  def test_configure_trigger(self):
    try:
      e = None
      if self._TRY_REAL_SCOPE:
        available = utility.get_available()
        for a in available:
          if a.startswith("USB::0x0699::"):
            e = tektronix.MSO4000Family(a)

      if e is None:
        address = "USB::0x0000::0x0000:C000000::INSTR"

        mock_pyvisa.no_pop = True
        # _ = MockMDO3054(address) # TODO (WattsUp) Fix MockMDO3054

        rm = mock_pyvisa.ResourceManager()
        e = tektronix.MSO4000Family(address, rm=rm)
      else:
        time.sleep = self._original_sleep
        e.reset()

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

    finally:
      if e is not None:
        e.close()

  def test_run_stop(self):
    try:
      e = None
      if self._TRY_REAL_SCOPE:
        available = utility.get_available()
        for a in available:
          if a.startswith("USB::0x0699::"):
            e = tektronix.MSO4000Family(a)

      if e is None:
        address = "USB::0x0000::0x0000:C000000::INSTR"

        mock_pyvisa.no_pop = True
        # _ = MockMDO3054(address) # TODO (WattsUp) Fix MockMDO3054

        rm = mock_pyvisa.ResourceManager()
        e = tektronix.MSO4000Family(address, rm=rm)
      else:
        time.sleep = self._original_sleep
        e.reset()

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
      self.assertEqual(result, "ARMED")

    finally:
      if e is not None:
        e.close()

  def test_single(self):
    try:
      e = None
      if self._TRY_REAL_SCOPE:
        available = utility.get_available()
        for a in available:
          if a.startswith("USB::0x0699::"):
            e = tektronix.MSO4000Family(a)

      if e is None:
        address = "USB::0x0000::0x0000:C000000::INSTR"

        mock_pyvisa.no_pop = True
        # _ = MockMDO3054(address) # TODO (WattsUp) Fix MockMDO3054

        rm = mock_pyvisa.ResourceManager()
        e = tektronix.MSO4000Family(address, rm=rm)
      else:
        time.sleep = self._original_sleep
        e.reset()

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

    finally:
      if e is not None:
        e.close()

  # def test_configure_channel(self):
  #   e = None
  #   if self._TRY_REAL_SCOPE:
  #     utility.pyvisa = pyvisa
  #     equipment.pyvisa = pyvisa
  #     available = utility.get_available()
  #     for a in available:
  #       if a.startswith("USB::0x0699::"):
  #         model = a[15:19]
  #         if model in ["0409", "040A", "040B", "041A", "041B", "041C", "0428"]:
  #           e = tektronix.MSO4000(a)
  #           break
  #         elif model in ["040C", "040D", "040E", "040F", "042A", "042E"]:
  #           e = tektronix.MDO4000(a)
  #           break
  #         elif model in ["0408"]:
  #           e = tektronix.MDO3000(a)
  #           break

  #   if e is None:
  #     utility.pyvisa = mock_pyvisa
  #     equipment.pyvisa = mock_pyvisa
  #     address = "USB::0x0000::0x0000:C000000::INSTR"

  #     mock_pyvisa.no_pop = True
  #     _ = MockMDO3054(address)

  #     e = tektronix.MDO3000(address)
  #   else:
  #     time.sleep = self._original_sleep
  #     e.send("*RST")

  #   self.assertRaises(KeyError, e.configure_channel, "CHFake", "FAKE", None)

  #   value = 2
  #   reply = e.configure_channel("CH2", "SCALE", value)
  #   self.assertEqual(value, reply)
  #   self.assertRaises(KeyError, e.configure_channel, "CHFake", "SCALE", value)

  #   value = 2
  #   reply = e.configure_channel("CH2", "POSITION", value)
  #   self.assertEqual(value, reply)
  #   self.assertRaises(KeyError, e.configure_channel, "CHFake", "POSITION",
  #                     value)

  #   value = 0.1
  #   reply = e.configure_channel("CH2", "OFFSET", value)
  #   self.assertEqual(value, reply)
  #   self.assertRaises(KeyError, e.configure_channel, "CHFake", "OFFSET", value)

  #   value = "I2C CLOCK"
  #   reply = e.configure_channel("CH2", "LABEL", value)
  #   self.assertEqual(f'"{value}"', reply)
  #   self.assertRaises(KeyError, e.configure_channel, "CHFake", "LABEL", value)

  #   value = e.max_bandwidth
  #   reply = e.configure_channel("CH2", "BANDWIDTH", value)
  #   self.assertEqual(value, reply)
  #   reply = e.configure_channel("CH2", "BANDWIDTH", "FULL")
  #   self.assertEqual(value, reply)
  #   self.assertRaises(KeyError, e.configure_channel, "CHFake", "BANDWIDTH",
  #                     value)

  #   value = 50
  #   reply = e.configure_channel("CH2", "TERMINATION", value)
  #   self.assertEqual(value, reply)
  #   reply = e.configure_channel("CH2", "TERMINATION", "FIFTY")
  #   self.assertEqual(value, reply)
  #   self.assertRaises(KeyError, e.configure_channel, "CHFake", "TERMINATION",
  #                     value)
  #   value = 1e6
  #   reply = e.configure_channel("CH2", "TERMINATION", value)
  #   self.assertEqual(value, reply)

  #   value = True
  #   reply = e.configure_channel("CH2", "INVERT", value)
  #   self.assertEqual(value, reply)
  #   self.assertRaises(KeyError, e.configure_channel, "CHFake", "INVERT", value)

  #   value = 10
  #   reply = e.configure_channel("CH2", "PROBE_GAIN", value)
  #   self.assertEqual(value, reply)
  #   self.assertRaises(KeyError, e.configure_channel, "CHFake", "PROBE_GAIN",
  #                     value)

  #   value = 10
  #   reply = e.configure_channel("CH2", "PROBE_ATTENUATION", value)
  #   self.assertEqual(value, reply)
  #   self.assertRaises(KeyError, e.configure_channel, "CHFake",
  #                     "PROBE_ATTENUATION", value)

  #   value = "AC"
  #   reply = e.configure_channel("CH2", "COUPLING", value)
  #   self.assertEqual(value, reply)
  #   self.assertRaises(KeyError, e.configure_channel, "CHFake", "COUPLING",
  #                     value)
  #   self.assertRaises(ValueError, e.configure_channel, "CH2", "COUPLING",
  #                     "FAKE")

  #   value = True
  #   reply = e.configure_channel("CH2", "ACTIVE", value)
  #   self.assertEqual(value, reply)

  #   value = 1.4
  #   reply = e.configure_channel("CH2", "TRIGGER_LEVEL", value)
  #   self.assertEqual(value, reply)
  #   reply = e.configure_channel("CH2", "TRIGGER_LEVEL", "TTL")
  #   self.assertEqual(value, reply)

  # def test_command(self):
  #   e = None
  #   if self._TRY_REAL_SCOPE:
  #     utility.pyvisa = pyvisa
  #     equipment.pyvisa = pyvisa
  #     available = utility.get_available()
  #     for a in available:
  #       if a.startswith("USB::0x0699::"):
  #         model = a[15:19]
  #         if model in ["0409", "040A", "040B", "041A", "041B", "041C", "0428"]:
  #           e = tektronix.MSO4000(a)
  #           break
  #         elif model in ["040C", "040D", "040E", "040F", "042A", "042E"]:
  #           e = tektronix.MDO4000(a)
  #           break
  #         elif model in ["0408"]:
  #           e = tektronix.MDO3000(a)
  #           break

  #   if e is None:
  #     utility.pyvisa = mock_pyvisa
  #     equipment.pyvisa = mock_pyvisa
  #     address = "USB::0x0000::0x0000:C000000::INSTR"

  #     mock_pyvisa.no_pop = True
  #     _ = MockMDO3054(address)

  #     e = tektronix.MDO3000(address)
  #   else:
  #     time.sleep = self._original_sleep
  #     e.send("*RST")

  #   self.assertRaises(KeyError, e.command, "Fake")

  #   e.configure("TIME_SCALE", 1e-3)
  #   e.configure("TRIGGER_SOURCE", "LINE")
  #   e.configure("TIME_POINTS", 10e6)

  #   e.command("STOP")

  #   e.command("RUN")

  #   e.command("FORCE_TRIGGER")

  #   e.command("SINGLE")

  #   e.command("SINGLE_FORCE")

  #   e.command("CLEAR_MENU")

  #   channel = "CH1"
  #   e.configure_channel(channel, "SCALE", 0.1)
  #   e.configure_channel(channel, "POSITION", 0)
  #   self.assertRaises(ValueError, e.command, "AUTOSCALE", channel=None)
  #   self.assertRaises(ValueError, e.command, "AUTOSCALE", channel="CHFake")
  #   e.command("AUTOSCALE", channel=channel, silent=True)
  #   position = float(e.ask(f"{channel}:POSITION?"))
  #   scale = float(e.ask(f"{channel}:SCALE?"))
  #   self.assertEqualWithinError(2.5 / 8, scale, 0.1)
  #   self.assertEqualWithinError(-4, position, 0.1)

  # def test_autoscale(self):
  #   address = "USB::0x0000::0x0000:C000000::INSTR"
  #   instrument = MockMDO3054(address)

  #   e = tektronix.MDO3000(address)

  #   channel = "CH1"
  #   self.assertRaises(ValueError, e.command, "AUTOSCALE", channel=None)
  #   self.assertRaises(ValueError, e.command, "AUTOSCALE", channel="CHFake")

  #   e.configure("TIME_SCALE", 1e-3)
  #   e.configure_channel(channel, "SCALE", 0.1)
  #   e.configure_channel(channel, "POSITION", 0)
  #   with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
  #     e.command("AUTOSCALE", channel=channel, silent=False)

  #   position = float(e.ask(f"{channel}:POSITION?"))
  #   scale = float(e.ask(f"{channel}:SCALE?"))
  #   self.assertEqualWithinError(2.5 / 8, scale, 0.1)
  #   self.assertEqualWithinError(-4, position, 0.1)
  #   self.assertTrue(
  #       fake_stdout.getvalue().startswith(f"Autoscaling channel '{channel}'"))

  #   # Too low, too small
  #   e.configure_channel(channel, "SCALE", 5)
  #   e.configure_channel(channel, "POSITION", -4.9)
  #   with mock.patch("sys.stdout", new=io.StringIO()) as _:
  #     e.command("AUTOSCALE", channel=channel, silent=True)
  #   position = float(e.ask(f"{channel}:POSITION?"))
  #   scale = float(e.ask(f"{channel}:SCALE?"))
  #   self.assertEqualWithinError(2.5 / 8, scale, 0.1)
  #   self.assertEqualWithinError(-4, position, 0.1)

  #   # Too low, too large
  #   e.configure_channel(channel, "SCALE", 0.3)
  #   e.configure_channel(channel, "POSITION", -4.9)
  #   with mock.patch("sys.stdout", new=io.StringIO()) as _:
  #     e.command("AUTOSCALE", channel=channel, silent=False)
  #   position = float(e.ask(f"{channel}:POSITION?"))
  #   scale = float(e.ask(f"{channel}:SCALE?"))
  #   self.assertEqualWithinError(2.5 / 8, scale, 0.1)
  #   self.assertEqualWithinError(-4, position, 0.1)

  #   # Too high, too small
  #   e.configure_channel(channel, "SCALE", 50)
  #   e.configure_channel(channel, "POSITION", 5)
  #   with mock.patch("sys.stdout", new=io.StringIO()) as _:
  #     e.command("AUTOSCALE", channel=channel, silent=True)
  #   position = float(e.ask(f"{channel}:POSITION?"))
  #   scale = float(e.ask(f"{channel}:SCALE?"))
  #   self.assertEqualWithinError(2.5 / 8, scale, 0.1)
  #   self.assertEqualWithinError(-4, position, 0.1)

  #   instrument.waveform_amp = 0
  #   instrument.waveform_offset = 0
  #   e.configure_channel(channel, "SCALE", 0.1)
  #   e.configure_channel(channel, "POSITION", 0)
  #   with mock.patch("sys.stdout", new=io.StringIO()) as fake_stdout:
  #     self.assertRaises(TimeoutError,
  #                       e.command,
  #                       "AUTOSCALE",
  #                       channel=channel,
  #                       silent=False)

  # def test_read_waveform(self):
  #   e = None
  #   if self._TRY_REAL_SCOPE:
  #     utility.pyvisa = pyvisa
  #     equipment.pyvisa = pyvisa
  #     available = utility.get_available()
  #     for a in available:
  #       if a.startswith("USB::0x0699::"):
  #         model = a[15:19]
  #         if model in ["0409", "040A", "040B", "041A", "041B", "041C", "0428"]:
  #           e = tektronix.MSO4000(a)
  #           break
  #         elif model in ["040C", "040D", "040E", "040F", "042A", "042E"]:
  #           e = tektronix.MDO4000(a)
  #           break
  #         elif model in ["0408"]:
  #           e = tektronix.MDO3000(a)
  #           break

  #   if e is None:
  #     utility.pyvisa = mock_pyvisa
  #     equipment.pyvisa = mock_pyvisa
  #     address = "USB::0x0000::0x0000:C000000::INSTR"

  #     mock_pyvisa.no_pop = True
  #     _ = MockMDO3054(address)

  #     e = tektronix.MDO3000(address)
  #   else:
  #     time.sleep = self._original_sleep
  #     e.send("*RST")

  #   self.assertRaises(KeyError, e.read_waveform, "CHFake")

  #   num_points = 10e3
  #   e.configure("TIME_POINTS", num_points)
  #   e.configure("TIME_SCALE", 1e-3)
  #   e.configure("TIME_OFFSET", -1e-3)
  #   e.configure_channel("CH2", "SCALE", 0.05)
  #   e.configure_channel("CH2", "OFFSET", -0.01)
  #   e.configure_channel("CH2", "POSITION", 1)
  #   e.command("SINGLE_FORCE")

  #   samples, info = e.read_waveform("CH2")

  #   self.assertEqual(samples.shape[0], 2)
  #   self.assertEqual(samples.shape[1], num_points)

  #   self.assertEqual(info["x_unit"], "s")

  #   samples, info = e.read_waveform("CH2", raw=True, add_noise=False)

  #   self.assertEqual(samples.shape[0], 2)
  #   self.assertEqual(samples.shape[1], num_points)

  #   self.assertEqual(info["x_unit"], "s")
  #   self.assertEqual(info["y_incr"], 1)

  #   for i in range(samples.shape[1]):
  #     self.assertAlmostEqual(0, samples[1, i] % 1)

  #   samples, info = e.read_waveform("CH2", raw=True, add_noise=True)

  #   self.assertEqual(samples.shape[0], 2)
  #   self.assertEqual(samples.shape[1], num_points)

  #   self.assertEqual(info["x_unit"], "s")
  #   self.assertEqual(info["y_incr"], 1)

  #   # At least 1 instance of decimal precision
  #   sub_detected = False
  #   for i in range(samples.shape[1]):
  #     if round(samples[1, i] % 1, 7) != 0:
  #       sub_detected = True
  #   self.assertTrue(sub_detected)
