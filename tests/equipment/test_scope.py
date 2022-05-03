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

  @property
  def position(self) -> float:
    return 0

  @property
  def label(self) -> str:
    return ""

  @property
  def active(self) -> bool:
    return False

  def read_waveform(self,
                    raw: bool = False,
                    add_noise: bool = False) -> Tuple[np.ndarray, dict]:
    return np.array([[], []]), {}


class AnalogChannel(scope.AnalogChannel, Channel):
  """AnalogChannel for testing
  """

  @property
  def scale(self) -> float:
    return 1

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
    return 0

  @property
  def termination(self) -> float:
    return 50

  @property
  def probe_gain(self) -> float:
    return 0.1


class DigitalChannel(scope.DigitalChannel, Channel):
  """DigitalChannel for testing
  """

  @property
  def threshold(self) -> float:
    return 0.5


class Scope(scope.Scope):

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
        self.n_div_vert = 10
        self.sample_rate = 1e6

    s = FakeScope()
    self.assertEqual(s.time_points, time_points)

    c = AnalogChannel(alias=alias, parent=s)
    # AnalogChannel always sources 0-1V square wave @ 1kHz
    # Ideal scale = 0.1, position = -4

    c.scale = 0.1
    c.position = 0
    _, info = c.read_waveform()
    self.assertTrue(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])

    c.autoscale()  # TODO (WattsUp) Implement in test AnalogChannel

    _, info = c.read_waveform()
    self.assertFalse(info["clipping_top"])
    self.assertFalse(info["clipping_bottom"])
    self.assertEqual(s.time_points, time_points)  # Should revert any changes
    self.assertEqualWithinError(0.1, c.scale, 0.05)
    self.assertEqualWithinError(-4, c.position, 0.05)

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
