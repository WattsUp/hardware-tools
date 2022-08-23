"""Test module hardware_tools.equipment.tektronix.family_mso64
"""

import time
from typing import Any, Callable, List

import numpy as np
import pyvisa

from hardware_tools.equipment import utility
from hardware_tools.equipment import tektronix

from tests import base
from tests.equipment import mock_pyvisa

_rng = np.random.default_rng()


class MockMSO64(mock_pyvisa.Resource):
  """Mock Tektronix MSO64
  """

  def __init__(self, resource_manager: pyvisa.ResourceManager,
               address: str) -> None:
    super().__init__(resource_manager, address)

    self._cache_x_incr: float = None
    self._cache_n: int = None
    self._cache_random: np.ndarray = None
    self._cache_y: np.ndarray = None

    self.reset()

  def reset(self) -> None:
    # Taken from Appendix C: Factory Defaults in Programming Manual
    # Tuples are (type [or callable to convert str to value], value)
    self._running = True
    self.query_map = {
        "*IDN": "TEKTRONIX,MSO64",
        #         "ACQUIRE": {
        #             "MODE": (tektronix.parse_sample_mode, tektronix.SampleMode.SAMPLE),
        #             "NUMAVG": (lambda x: int(float(x)), 16),
        #             "NUMENV": (lambda x: x if x == "INFINITE" else int(x), "INFINITE"),
        #             "NUMACQ": 0,
        #             "STATE": (self.acquire_state, self.acquire_state),
        #             "STOPAFTER": (str, "RUNSTOP")
        #         },
        "CONFIGURATION": {
            "ANALOG": {
                "BANDWIDTH": 4e9
            }
        },
        #         "CH1": {
        #             "BANDWIDTH": (float, 1e9),
        #             "DESKEW": (float, 0.0),
        #             "COUPLING": (str, "DC"),
        #             "SCALE": (float, 1.0),  # Technically 0.1 but 10x probe
        #             "OFFSET": (float, 0.0),
        #             "POSITION": (lambda x: min(5.0, max(-5.0, float(x))), 0.0),
        #             "LABEL": (lambda x: str.strip(x, "'")[:30], ""),
        #             "INVERT": (lambda x: x in ["ON", "1"], False),
        #             "TERMINATION": (float, 1e6),  # Fixed 1MΩ termination
        #             "PROBE": {
        #                 "GAIN": (float, 10.0)  # Technically 0.1 but 10x probe
        #             }
        #         },
        #         "D0": {
        #             "THRESHOLD": (float, 1.4)
        #         },
        "DISPLAY": {
            "GLOBAL": {
                "CH1": {
                    "STATE": (lambda x: x in ["ON", "1"], True)
                }
            }
        },
        #         "DATA": {
        #             "ENCDG": (str, "RIBINARY"),
        #             "SOURCE": (str, "CH1"),
        #             "START": (lambda x: int(float(x)), 1),
        #             "STOP": (lambda x: int(float(x)), 10000),
        #             "WIDTH": (lambda x: int(float(x)), 1)
        #         },
        "HEADER": (lambda x: x in ["ON", "1"], False),
        "VERBOSE": (lambda x: x in ["ON", "1"], True),
        #         "HORIZONTAL": {
        #             "RECORDLENGTH": (lambda x: int(float(x)), 10000),
        #             "SAMPLERATE":
        #                 lambda: (self.query_map["HORIZONTAL"]["RECORDLENGTH"][1] /
        #                          (10 * self.query_map["HORIZONTAL"]["SCALE"][1])),
        #             "SCALE": (float, 4e-6),
        #             "DELAY": {
        #                 "MODE": (lambda x: x in ["ON", "1"], True),
        #                 "TIME": (float, 0)
        #             }
        #         },
        #         "SELECT": {
        #             "CH1": (lambda x: x in ["ON", "1"], True)
        #         },
        #         "TRIGGER": {
        #             "A": {
        #                 "EDGE": {
        #                     "SOURCE": (str, "CH1"),
        #                     "SLOPE": (tektronix.parse_polarity,
        #                               tektronix.EdgePolarity.RISING),
        #                     "COUPLING": (str, "DC")
        #                 },
        #                 "HOLDOFF": {
        #                     "TIME": (float, 20e-9),
        #                 },
        #                 "LEVEL": {
        #                     "CH1": (float, 0.0)
        #                 },
        #                 "MODE": (str, "AUTO"),
        #                 "PULSE": {
        #                     "CLASS": (str, "WIDTH")
        #                 },
        #                 "PULSEWIDTH": {
        #                     "SOURCE": (str, "CH1"),
        #                     "WIDTH": (float, 8e-9),
        #                     "HIGHLIMIT": (float, 12e-9),
        #                     "LOWLIMIT": (float, 8e-9),
        #                     "POLARITY": (tektronix.parse_polarity,
        #                                  tektronix.EdgePolarity.RISING),
        #                     "WHEN":
        #                         (tektronix.parse_comparison, tektronix.Comparison.LESS)
        #                 },
        #                 "TIMEOUT": {
        #                     "SOURCE": (str, "CH1"),
        #                     "TIME": (float, 8e-9),
        #                     "POLARITY": (tektronix.parse_polarity,
        #                                  tektronix.EdgePolarity.RISING)
        #                 },
        #                 "TYPE": (str, "EDGE")
        #             },
        #             "STATE": self.trigger_state
        #         }
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
    elif len(keys) == 1 and keys[0] == "DISPLAY":
      s = ""
      for ch, v in self.query_map["DISPLAY"]["GLOBAL"].items():
        v: dict
        for kk, vv in v.items():
          s += (f":DISPLAY:GLOBAL:{ch}:{kk} "
                f"{super().query_str(keys + [ch, kk], vv)};")
      return s
    else:
      s = super().query_str(keys, value)

    if self.query_map["HEADER"][1]:
      return ":".join(keys) + " " + s
    else:
      return s


#   def trigger_state(self, triggered=False) -> str:
#     if self._running:
#       # Acquisition engine is running
#       t_type = self.query_map["TRIGGER"]["A"]["TYPE"][1]
#       if t_type == "EDGE":
#         src = self.query_map["TRIGGER"]["A"]["EDGE"]["SOURCE"][1]
#         level = self.query_map["TRIGGER"]["A"]["LEVEL"][src][1]
#         if 0 <= level <= 2.5:
#           triggered = True
#       if triggered:
#         self.query_map["ACQUIRE"]["NUMACQ"] += 1

#       if self.query_map["ACQUIRE"]["STOPAFTER"][1] == "SEQUENCE":
#         # Single capture will stop acquisition if triggered
#         if triggered:
#           self._running = False
#           return "SAVE"
#         else:
#           return "READY"
#       elif self.query_map["TRIGGER"]["A"]["MODE"][1] == "AUTO":
#         # Auto mode is always AUTO
#         return "AUTO"
#       else:
#         # Normal mode is always READY
#         # (technically ARMED->TRIGGER->READY->ARMED loop)
#         if triggered:
#           return "TRIGGER"
#         else:
#           return "READY"
#     else:
#       return "SAVE"

#   def acquire_state(self, value=None) -> Callable:
#     if value is None:
#       # Call trigger_state to update state machine
#       self.trigger_state()
#       return self._running
#     else:
#       self.query_map["ACQUIRE"]["NUMACQ"] = 0
#       self._running = value in ["ON", "1", "RUN"]

#     # Since writes assign the result to the value, need to return this
#     # function so it can be called in the future
#     return self.acquire_state

#   def read_raw(self) -> bytes:
#     if len(self.queue_rx) > 0 and self.queue_rx[-1] == "WAVFRM?":
#       points = self.query_map["HORIZONTAL"]["RECORDLENGTH"][1]
#       h_scale = self.query_map["HORIZONTAL"]["SCALE"][1]
#       h_delay = self.query_map["HORIZONTAL"]["DELAY"]["TIME"][1]
#       fs = points / (10 * h_scale)

#       src = self.query_map["DATA"]["SOURCE"][1]
#       coupling = self.query_map[src]["COUPLING"][1]
#       scale = self.query_map[src]["SCALE"][1]
#       position = self.query_map[src]["POSITION"][1]
#       offset = self.query_map[src]["OFFSET"][1]
#       wf_id = (f"{src}, "
#                f"{coupling} coupling, "
#                f"{scale}V/div, "
#                f"{h_scale}s/div, "
#                f"{points} points")

#       x_incr = 1 / fs
#       if (self._cache_x_incr != x_incr or self._cache_n != points):
#         self._cache_x_incr = x_incr
#         self._cache_n = points
#         t = np.linspace(0, x_incr * (points - 1), points) - x_incr * points / 2
#         self._cache_y = (np.mod(t * 1e3, 1) > 0.5) * 2.5
#       y: np.ndarray = self._cache_y.copy()
#       if coupling != "DC":
#         y -= 1.25

#       # Random takes a long time, doesn't really matter
#       if (self._cache_random is None or self._cache_random.shape[0] != points):
#         self._cache_random = _rng.normal(0, 0.01, points)
#       self._cache_random = np.roll(self._cache_random, 10)
#       y += self._cache_random

#       # Transform real world units to ADC counts
#       x_zero = -x_incr * points / 2 + h_delay
#       y_mult = 10 * scale / 254
#       y_off = position / 10 * 254
#       y_zero = offset
#       y = np.floor((y - y_zero) / y_mult + y_off + 0.5)
#       y = np.clip(y, -127, 127)

#       waveform = y.astype(">b").tobytes()
#       n_bytes = f"{len(waveform)}"

#       header = (f':WFMOUTPRE:WFID "{wf_id}";'
#                 f"NR_PT {points};"
#                 f'XUNIT "s";XINCR {x_incr:.4E};XZERO {x_zero:.4E};'
#                 f'YUNIT "V";YMULT {y_mult:.4E};'
#                 f"YOFF {y_off:.4E};YZERO {y_zero:.4E};"
#                 "BYT_OR MSB;BYT_NR 1;BN_FMT RI;"
#                 f":CURVE #{len(n_bytes)}{n_bytes}")
#       return header.encode(encoding="ascii") + waveform + b"\n"
#     return super().read_raw()


class TestMSO64(base.TestBase):
  """Test Equipment Tektronix MSO64
  """

  # When true, connect CH1 to probe compensation with fixed 10x probe,
  # and all others open
  _TRY_REAL_SCOPE = True

  def setUp(self) -> None:
    super().setUp()

    mock_pyvisa.resources = {}
    mock_pyvisa.available = []

    self._scope: tektronix.MSO456Family = None

  def tearDown(self) -> None:
    super().tearDown()

    mock_pyvisa.resources = {}
    mock_pyvisa.available = []
    mock_pyvisa.no_pop = False

    if self._scope is not None:
      self._scope.close()

  def connect(self) -> tektronix.MSO456Family:
    e = None
    if self._TRY_REAL_SCOPE:
      available = utility.get_available()
      for a in available:
        if a.startswith("USB::0x0699::"):
          e = tektronix.MSO456Family(a)

    if e is None:
      address = "USB::0x0000::0x0000:C000000::INSTR"

      mock_pyvisa.no_pop = True
      rm = mock_pyvisa.ResourceManager()
      _ = MockMSO64(rm, address)

      e = tektronix.MSO456Family(address, rm=rm)
    else:
      time.sleep = self._original_sleep
      e.reset()
    self._scope = e
    return e

  def test_init(self):
    address = "USB::0x0000::0x0000:C000000::INSTR"

    mock_pyvisa.no_pop = True
    rm = mock_pyvisa.ResourceManager()
    instrument = MockMSO64(rm, address)

    instrument.query_map["*IDN"] = "FAKE"
    self.assertRaises(ValueError, tektronix.MSO456Family, address, rm=rm)
    instrument.query_map["*IDN"] = "TEKTRONIX,MSO65,serial number"
    self.assertRaises(ValueError, tektronix.MSO456Family, address, rm=rm)

    instrument.query_map["*IDN"] = "TEKTRONIX,MSO64A,serial number"
    self.assertRaises(ValueError, tektronix.MSO456Family, address, rm=rm)
    instrument.query_map["*IDN"] = "TEKTRONIX,MSO64,serial number"
    e = tektronix.MSO456Family(address, rm=rm, name="Emulated MSO64")
    self.assertEqual(e.max_bandwidth, 4000e6)
    self.assertListEqual(list(e._channels.keys()), [1])  # pylint: disable=protected-access
    self.assertListEqual(list(e._digitals.keys()), [])  # pylint: disable=protected-access

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
    value = 1
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

    value = 12500
    e.time_points = value
    self.assertEqual(value, e.time_points)

    value = 2e-3
    e.time_scale = value
    self.assertEqual(value, e.time_scale)

    value = 0.2e-3
    e.time_offset = value
    self.assertEqual(value, e.time_offset)

  def test_configure_trigger(self):
    e = self.connect()

    t = tektronix.TriggerEdge("AUX", 0.5)
    e.trigger = t
    result = e.trigger
    self.assertEqual(t.src, result.src)
    self.assertEqual(t.level, result.level)
    self.assertEqual(t.slope, result.slope)
    self.assertEqual(t.dc_coupling, result.dc_coupling)
    self.assertEqual(t.holdoff, result.holdoff)

    t = tektronix.TriggerEdge("LINE", 0.5)
    e.trigger = t
    result = e.trigger
    self.assertEqual(t.src, result.src)
    self.assertEqual(0.0, result.level)  # Line has no level
    self.assertEqual(t.slope, result.slope)
    self.assertEqual(t.dc_coupling, result.dc_coupling)
    self.assertEqual(t.holdoff, result.holdoff)

    t = tektronix.TriggerEdge("CH1", 0.5, dc_coupling=False)
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
    e.trigger = t
    result = e.trigger
    self.assertEqual(t.src, result.src)
    self.assertEqual(t.level, result.level)
    self.assertEqual(t.slope, result.slope)
    self.assertEqual(t.timeout, result.timeout)
    self.assertEqual(t.holdoff, result.holdoff)

    t = tektronix.TriggerEdgeTimeout("FAKE", 0.5, 1e-3)
    self.assertRaises(ValueError, setattr, e, "trigger", t)

    t = tektronix.TriggerPulseWidth("CH1", 0.5, 0.5e-3,
                                    tektronix.Comparison.EQUAL)
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
    self.assertRaises(ValueError, setattr, e, "trigger", t)

    t = tektronix.TriggerPulseWidth("CH1", 0.5, 1e-3,
                                    tektronix.Comparison.OUTSIDE)
    self.assertRaises(ValueError, setattr, e, "trigger", t)

    e.send("TRIGGER:A:TYPE RUNT")
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
    self.assertTrue(info["clipping_top"])
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

    value = 0.1
    c.probe_gain = value
    self.assertEqual(value, c.probe_gain)

    c = e.ch(2)

    value = 50
    c.termination = value
    self.assertEqual(value, c.termination)

    value = 1e6
    c.termination = value
    self.assertEqual(value, c.termination)
