"""Test module hardware_tools.equipment.scope
"""

from typing import Tuple

import numpy as np

from hardware_tools.equipment import scope

from tests import base
from tests.equipment import mock_pyvisa

_rng = np.random.default_rng()


class Channel(scope.Channel):
  """Channel for testing
  """

  def __init__(self, alias: str, parent: scope.Scope) -> None:
    super().__init__(alias, parent)

    self._position = 0

  @property
  def position(self) -> float:
    return self._position

  @position.setter
  def position(self, value: float) -> None:
    self._position = min(max(round(value, 2), -self._parent.n_div_vert / 2),
                         self._parent.n_div_vert / 2)

  @property
  def label(self) -> str:
    return ""

  @property
  def active(self) -> bool:
    return False

  def read_waveform(self,
                    raw: bool = False,
                    add_noise: bool = False) -> Tuple[np.ndarray, dict]:
    return None, None


class AnalogChannel(scope.AnalogChannel, Channel):
  """AnalogChannel for testing
  """

  def __init__(self, alias: str, parent: scope.Scope) -> None:
    super().__init__(alias, parent)

    self._scale = 1
    self._offset = 0
    self.input_grounded = False

    self._cache_x_incr: float = None
    self._cache_n: int = None
    self._cache_random: np.ndarray = None
    self._cache_t: np.ndarray = None
    self._cache_y: np.ndarray = None

  @property
  def scale(self) -> float:
    return self._scale

  @scale.setter
  def scale(self, value: float) -> None:
    self._scale = float(f"{value:.3g}")

  @property
  def bandwidth(self) -> float:
    return 0

  @property
  def dc_coupling(self) -> bool:
    return True

  @property
  def deskew(self) -> float:
    return 0

  @property
  def inverted(self) -> bool:
    return False

  @property
  def offset(self) -> float:
    return self._offset

  @offset.setter
  def offset(self, value: float) -> None:
    self._offset = value

  @property
  def termination(self) -> float:
    return 50

  @property
  def probe_gain(self) -> float:
    return 0.1

  def read_waveform(self,
                    raw: bool = False,
                    add_noise: bool = False) -> Tuple[np.ndarray, dict]:
    # AnalogChannel always sources 0-1V square wave @ 1kHz
    # Assume 8b ADC for testing
    self._parent.time_points = int(self._parent.time_points)
    fs = self._parent.time_points / (self._parent.time_scale *
                                     self._parent.n_div_horz)
    x_incr = 1 / fs
    if (self._cache_x_incr != x_incr or
        self._cache_n != self._parent.time_points):
      self._cache_x_incr = x_incr
      self._cache_n = self._parent.time_points
      self._cache_t = np.linspace(0, x_incr * (self._parent.time_points - 1),
                                  self._parent.time_points)
      self._cache_y = (np.mod(self._cache_t * 1e3, 1) > 0.5) * 1.0
    t = self._cache_t
    y = self._cache_y.copy()

    # Random takes a long time, doesn't really matter
    if (self._cache_random is None or
        self._cache_random.shape[0] != self._parent.time_points):
      self._cache_random = _rng.normal(0, 0.01, self._parent.time_points)
    self._cache_random = np.roll(self._cache_random, 10)
    y += self._cache_random

    if self.input_grounded:
      y *= 0.0

    y_min = (-self._parent.n_div_vert / 2 -
             self.position) * self.scale + self.offset
    y_max = (self._parent.n_div_vert / 2 -
             self.position) * self.scale + self.offset

    clipping_top = y > y_max
    clipping_bottom = y < y_min
    y[clipping_top] = y_max
    y[clipping_bottom] = y_min

    clipping_top = np.any(clipping_top)
    clipping_bottom = np.any(clipping_bottom)

    # from matplotlib import pyplot
    # pyplot.plot(t, y)
    # pyplot.axhline(y=y_min)
    # pyplot.axhline(y=y_max)
    # pyplot.show()

    info = {
        "config_str": "Fake 1kHz source",
        "x_unit": "s",
        "y_unit": "V",
        "x_incr": x_incr,
        "y_incr": 0,
        "y_clip_min": y_min,
        "y_clip_max": y_max,
        "clipping_top": clipping_top,
        "clipping_bottom": clipping_bottom
    }

    if raw:
      y = np.round((y - y_min) / (y_max - y_min) * 256 - 128)
      if add_noise:
        y += _rng.uniform(-0.5, 0.5, self._parent.time_points)
      y = np.clip(y, -128, 127)
      info["y_unit"] = "ADC Counts"
      info["y_clip_min"] = -128
      info["y_clip_max"] = 127

    return np.array([t, y]), info


class DigitalChannel(scope.DigitalChannel, Channel):
  """DigitalChannel for testing
  """

  @property
  def threshold(self) -> float:
    return 0.5


class Scope(scope.Scope):
  """Scope for testing
  """

  def _init_channels(self) -> None:
    self._channels[1] = AnalogChannel("CH1", self)
    self._digitals[0] = DigitalChannel("D0", self)

  @property
  def sample_rate(self) -> float:
    return 1

  @property
  def sample_mode(self) -> scope.SampleMode:
    return scope.SampleMode.SAMPLE

  @property
  def sample_mode_n(self) -> int:
    return 1

  @property
  def time_scale(self) -> float:
    return 1

  @property
  def time_offset(self) -> float:
    return 0

  @property
  def time_points(self) -> int:
    return 1

  @property
  def trigger(self) -> scope.Trigger:
    return scope.TriggerEdge("CH1", 0.5)

  def stop(self, timeout: float = 1) -> None:
    pass

  def run(self, normal: bool = True, timeout: float = 1) -> None:
    pass

  def single(self,
             trigger_cmd: callable = None,
             force: bool = False,
             timeout: float = 1) -> None:
    pass

  def force(self) -> None:
    pass


class TestTrigger(base.TestBase):
  """Test Scope Trigger
  """

  def test_base(self):
    holdoff = _rng.uniform(-1.0, 1.0)

    t = scope.Trigger(holdoff=holdoff)

    self.assertEqual(t.holdoff, holdoff)

  def test_edge(self):
    src = "Trigger src"
    level = _rng.uniform(-1.0, 1.0)
    slope = _rng.choice(list(scope.EdgePolarity))
    dc_coupling = bool(_rng.integers(0, 1))
    holdoff = _rng.uniform(-1.0, 1.0)

    t = scope.TriggerEdge(src,
                          level,
                          slope=slope,
                          dc_coupling=dc_coupling,
                          holdoff=holdoff)

    self.assertEqual(t.src, src)
    self.assertEqual(t.level, level)
    self.assertEqual(t.slope, slope)
    self.assertEqual(t.dc_coupling, dc_coupling)
    self.assertEqual(t.holdoff, holdoff)

  def test_edge_timeout(self):
    src = "Trigger src"
    level = _rng.uniform(-1.0, 1.0)
    timeout = _rng.uniform(-1.0, 1.0)
    slope = _rng.choice(list(scope.EdgePolarity))
    holdoff = _rng.uniform(-1.0, 1.0)

    t = scope.TriggerEdgeTimeout(src,
                                 level,
                                 timeout,
                                 slope=slope,
                                 holdoff=holdoff)

    self.assertEqual(t.src, src)
    self.assertEqual(t.level, level)
    self.assertEqual(t.timeout, timeout)
    self.assertEqual(t.slope, slope)
    self.assertEqual(t.holdoff, holdoff)

  def test_pulse_width(self):
    src = "Trigger src"
    level = _rng.uniform(-1.0, 1.0)
    width = _rng.uniform(-1.0, 1.0)
    comparison = _rng.choice(list(scope.Comparison))
    positive = bool(_rng.integers(0, 1))
    holdoff = _rng.uniform(-1.0, 1.0)

    t = scope.TriggerPulseWidth(src,
                                level,
                                width,
                                comparison,
                                positive=positive,
                                holdoff=holdoff)

    self.assertEqual(t.src, src)
    self.assertEqual(t.level, level)
    self.assertEqual(t.width, width)
    self.assertEqual(t.comparison, comparison)
    self.assertEqual(t.positive, positive)
    self.assertEqual(t.holdoff, holdoff)


class TestChannel(base.TestBase):
  """Test Scope Channel
  """

  def test_base(self):
    alias = "CH20"
    parent = _rng.uniform(-1.0, 1.0)

    c = Channel(alias=alias, parent=parent)

    self.assertEqual(c._alias, alias)  # pylint: disable=protected-access
    self.assertEqual(c._parent, parent)  # pylint: disable=protected-access

  def test_analog(self):
    alias = "CH20"
    parent = _rng.uniform(-1.0, 1.0)

    c = AnalogChannel(alias=alias, parent=parent)

    self.assertEqual(c._alias, alias)  # pylint: disable=protected-access
    self.assertEqual(c._parent, parent)  # pylint: disable=protected-access

  def test_analog_autoscale(self):
    alias = "CH20"
    time_points = 1000

    class FakeScope:

      def __init__(self) -> None:
        self.time_points = time_points
        self.n_div_horz = 10
        self.n_div_vert = 10
        self.time_scale = 1e-3

      def single(self, *args, **kwargs) -> None:
        pass

    s = FakeScope()
    self.assertEqual(s.time_points, time_points)

    c = AnalogChannel(alias=alias, parent=s)
    # AnalogChannel always sources 0-1V square wave @ 1kHz
    # Ideal scale = 0.1 / 0.8, position = 0.0
    c.offset = 0.5
    ideal_scale = 0.1 / 0.8
    ideal_position = 0.0

    # Right scale, too high
    c.scale = ideal_scale
    c.position = 4.0
    _, info = c.read_waveform()
    self.assertTrue(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])

    c.autoscale()

    _, info = c.read_waveform()
    self.assertFalse(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])
    self.assertEqual(s.time_points, time_points)  # Should revert any changes
    self.assertEqualWithinError(ideal_scale, c.scale, 0.2)
    self.assertEqualWithinError(ideal_position, c.position, 0.2)

    # Right scale, too low
    c.scale = ideal_scale
    c.position = -4.0
    _, info = c.read_waveform()
    self.assertFalse(info["clipping_top"])
    self.assertTrue(info["clipping_bottom"])

    c.autoscale()

    _, info = c.read_waveform()
    self.assertFalse(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])
    self.assertEqual(s.time_points, time_points)  # Should revert any changes
    self.assertEqualWithinError(ideal_scale, c.scale, 0.2)
    self.assertEqualWithinError(ideal_position, c.position, 0.2)

    # Clipping both sides
    c.scale = 0.05
    c.position = 0.0
    c.offset = 0.5
    _, info = c.read_waveform()
    self.assertTrue(info["clipping_top"])
    self.assertTrue(info["clipping_bottom"])

    c.autoscale()

    _, info = c.read_waveform()
    self.assertFalse(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])
    self.assertEqual(s.time_points, time_points)  # Should revert any changes
    self.assertEqualWithinError(ideal_scale, c.scale, 0.2)
    self.assertEqualWithinError(ideal_position, c.position, 0.2)

    # Small signal and too high
    c.scale = 1.0
    c.position = 5.0
    c.offset = 0.5
    _, info = c.read_waveform()
    self.assertTrue(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])

    c.autoscale()

    _, info = c.read_waveform()
    self.assertFalse(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])
    self.assertEqual(s.time_points, time_points)  # Should revert any changes
    self.assertEqualWithinError(ideal_scale, c.scale, 0.2)
    self.assertEqualWithinError(ideal_position, c.position, 0.2)

    # Small signal and too low
    c.scale = 1.0
    c.position = -5.0
    c.offset = 0.5
    _, info = c.read_waveform()
    self.assertFalse(info["clipping_top"])
    self.assertTrue(info["clipping_bottom"])

    c.autoscale()

    _, info = c.read_waveform()
    self.assertFalse(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])
    self.assertEqual(s.time_points, time_points)  # Should revert any changes
    self.assertEqualWithinError(ideal_scale, c.scale, 0.2)
    self.assertEqualWithinError(ideal_position, c.position, 0.2)

    # DC too small
    c.input_grounded = True
    c.scale = 0.1
    c.position = 0.0
    c.offset = 0.0

    self.assertRaises(TimeoutError, c.autoscale)
    self.assertEqual(s.time_points, time_points)  # Should revert any changes

  def test_digital(self):
    alias = "CH20"
    parent = _rng.uniform(-1.0, 1.0)

    c = DigitalChannel(alias=alias, parent=parent)

    self.assertEqual(c._alias, alias)  # pylint: disable=protected-access
    self.assertEqual(c._parent, parent)  # pylint: disable=protected-access


class TestScope(base.TestBase):
  """Test Scope
  """

  def setUp(self) -> None:
    super().setUp()

    mock_pyvisa.resources = {}
    mock_pyvisa.available = []
    scope.equipment.pyvisa = mock_pyvisa

  def tearDown(self) -> None:
    super().tearDown()
    mock_pyvisa.resources = {}
    mock_pyvisa.available = []
    mock_pyvisa.no_pop = False

  def test_init(self):
    name = "Mock Equipment"
    address = "USB::0x0000::0x0000:C000000::INSTR"
    s = Scope(address, name=name)
    self.assertIn(address, mock_pyvisa.resources)
    self.assertEqual(s._instrument, mock_pyvisa.resources[address])  # pylint: disable=protected-access

    self.assertIn(1, s._channels)  # pylint: disable=protected-access
    self.assertIn(0, s._digitals)  # pylint: disable=protected-access

  def test_ch(self):
    name = "Mock Equipment"
    address = "USB::0x0000::0x0000:C000000::INSTR"
    s = Scope(address, name=name)

    self.assertEqual(s.ch(1), s._channels[1])  # pylint: disable=protected-access

  def test_d(self):
    name = "Mock Equipment"
    address = "USB::0x0000::0x0000:C000000::INSTR"
    s = Scope(address, name=name)

    self.assertEqual(s.d(0), s._digitals[0])  # pylint: disable=protected-access
