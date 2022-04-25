"""Scope base class to interface to oscilloscope
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import enum
from typing import Tuple, Union

import numpy as np

from hardware_tools.equipment import equipment
from hardware_tools.math.stats import Comparison
from hardware_tools.math.lines import EdgePolarity


class SampleMode(enum.Enum):
  """Sample Mode enumeration describing how each point's value is calculated
  from one or more acquisitions

  SAMPLE: Each point is a sampled value from the ADC
  AVERAGE: Each point is an average of samples across acquisitions
  ENVELOPE: Each point is the max and min of samples across acquisitions
  """
  SAMPLE = 1
  AVERAGE = 2
  ENVELOPE = 3


class Trigger(ABC):
  """Base Scope Trigger class
  """

  def __init__(self, holdoff: float = 20e-9) -> None:
    """Create a trigger

    Args:
      holdoff: Time after trigger than a new trigger cannot be generated
    """
    super().__init__()
    self.holdoff = holdoff


class TriggerEdge(Trigger):
  """Trigger on edge
  """

  def __init__(self,
               src: str,
               level: float,
               slope: EdgePolarity = EdgePolarity.RISING,
               dc_coupling: bool = True,
               holdoff: float = 20e-9) -> None:
    """Create an edge coupling trigger

    Args:
      src: Source of trigger, CH1, CH2,...
      level: Decision level for edge
      slope: Direction of edge, BOTH will trigger on either polarity
      dc_coupling: True will DC couple the trigger source, False will AC couple
      holdoff: Time after trigger than a new trigger cannot be generated
    """
    super().__init__(holdoff=holdoff)
    self.src = src
    self.level = level
    self.slope = slope
    self.dc_coupling = dc_coupling


class TriggerPulse(Trigger):
  """Trigger on pulse
  """

  def __init__(self,
               src: str,
               level: float,
               width: Union[float, Tuple[float, float]],
               comparision: Comparison,
               positive: bool = True,
               dc_coupling: bool = True,
               holdoff: float = 20e-9) -> None:
    """Create an edge coupling trigger

    Args:
      src: Source of trigger, CH1, CH2,...
      level: Decision level for edge
      width: Pulse width, or (lower, upper) limits for WITHIN/OUTSIDE
      comparision: Comparision operator to trigger off of, EQUAL/UNEQUAL has a
        tolerance, usually ±5%. Use WITHIN/OUTSIDE for finer control
      positive: True will trigger of of positive polarity pulses. False will use
        negative polarity pulses
      dc_coupling: True will DC couple the trigger source, False will AC couple
      holdoff: Time after trigger than a new trigger cannot be generated
    """
    super().__init__(holdoff=holdoff)
    self.src = src
    self.level = level
    self.width = width
    self.comparision = comparision
    self.positive = positive
    self.dc_coupling = dc_coupling


class Channel(ABC):
  """Base Scope Channel

  Properties:
    label: str = string label, only seen on Scope screen
    position: float = units of vertical divisions
    active: bool = True when channel is on, False is off
  """

  def __init__(self, alias: str, parent: Scope) -> None:
    """Create scope channel

    Args:
      alias: String name for channel, likely matching how to talk to the scope
      parent: Owning scope used for ask and send functions
    """
    super().__init__()

    self._alias = alias
    self._parent = parent

  @property
  @abstractmethod
  def position(self) -> float:
    """Units of vertical divisions.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def label(self) -> str:
    """String label, only seen on Scope screen
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def active(self) -> bool:
    """True when channel is on, False is off
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @abstractmethod
  def read_waveform(self,
                    raw: bool = False,
                    add_noise: bool = False) -> Tuple[np.ndarray, dict]:
    """Read waveform from the Scope Channel

    Stop the scope before reading multiple channels to ensure concurrent
    sampling.

    Args:
      raw: True will return raw ADC values, False will transform into
        real-world units
      add_noise: True will add uniform noise to the LSB for antialiasing

    Returns:
      Samples: [[x0, x1,..., xn], [y0, y1,..., yn]]
      Dictionary of sampling information:
        config_string: str = Human readable string describing configuration
        x_unit: str = Unit string for horizontal axis
        y_unit: str = Unit string for vertical axis
        x_incr: float = Step between horizontal axis values
        y_incr: float = Step between vertical axis values (LSB)
        resolution: int = number of bits of conversion
    """
    pass  # pragma: no cover


class AnalogChannel(Channel):
  """Analog waveform channel

  Properties:
    label: str = string label, only seen on Scope screen
    position: float = units of vertical divisions
    active: bool = True when channel is on, False is off
    scale: float = units of real units per vertical division
    bandwidth: float = units of Hz
    dc_coupling: bool = True when DC coupled, False when AC coupled
    deskew: float = units of seconds
    inverted: bool = True when inverted, False when non-inverted
    offset: float = Units of real units. (Often to remove DC offset without AC
      coupling)
    termination: float = Units of ohms
    probe_gain: float = Unitless. (A 10x probe has a gain of 0.1)
  """

  @property
  @abstractmethod
  def scale(self) -> float:
    """Units of real units per vertical division.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def bandwidth(self) -> float:
    """Units of Hz.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def dc_coupling(self) -> bool:
    """True when DC coupled, False when AC coupled.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def deskew(self) -> float:
    """Units of seconds.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def inverted(self) -> bool:
    """True when inverted, False when non-inverted.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def offset(self) -> float:
    """Units of real units. (Often to remove DC offset without AC coupling)
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def termination(self) -> float:
    """Units of ohms.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def probe_gain(self) -> float:
    """Unitless. (A 10x probe has a gain of 0.1)
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  def autoscale(self) -> None:
    """Autoscale channel
    """
    pass  # TODO (WattsUp) move autoscale code here


class DigitalChannel(Channel):
  """Digital waveform channel

  Properties:
    label: str = string label, only seen on Scope screen
    position: float = units of vertical divisions
    active: bool = True when channel is on, False is off
    threshold: float = Unit of real units to decide high or low
  """

  @property
  @abstractmethod
  def threshold(self) -> float:
    """Unit of real units to decide high or low.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover


class Scope(equipment.Equipment):
  """Scope base class to interface to oscilloscope

  Properties:
    n_div_horz: int = number of horizontal divisions
    n_div_vert: int = number of vertical divisions
    sample_rate: float = samples per second
    sample_mode: SampleMode enumeration
    sample_mode_n: int = number of acquisitions for sample modes AVERAGE and
      ENVELOPE
    time_scale: float = seconds per division
    time_offset: float = seconds
    time_points: int = samples
  """

  n_div_horz: int = 1
  n_div_vert: int = 1

  def __init__(self, address: str, name: str = "") -> None:
    super().__init__(address, name)
    self._channels = {}
    self._digitals = {}

  def ch(self, index: int) -> AnalogChannel:
    """Get analog channel of scope

    Args:
      index: 1-indexed channels (CH1 = Scope.ch(1))

    Returns:
      Proper AnalogChannel

    Raises:
      KeyError if channel is not found
    """
    return self._channels[index]

  def d(self, index: int) -> DigitalChannel:
    """Get digital channel of scope

    Args:
      index: 0-indexed channels (D0 = Scope.d(0))

    Returns:
      Proper DigitalChannel

    Raises:
      KeyError if channel is not found
    """
    return self._digitals[index]

  @property
  @abstractmethod
  def sample_rate(self) -> float:
    """Units of samples per second.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def sample_mode(self) -> SampleMode:
    """Enumeration, see SampleMode.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def sample_mode_n(self) -> int:
    """Number of acquisitions for sample modes AVERAGE and ENVELOPE.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def time_scale(self) -> float:
    """Units of seconds per division.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def time_offset(self) -> float:
    """Units of seconds.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def time_points(self) -> int:
    """Units of samples.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @property
  @abstractmethod
  def trigger(self) -> Trigger:
    """Trigger configuration, see derrived classes.
    Changes will immediately configure the scope.
    """
    pass  # pragma: no cover

  @abstractmethod
  def stop(self, timeout: float = 1) -> None:
    """Stop acquiring waveforms, equivalent to pressing STOP button

    Args:
      timeout: Timeout for state confirmation

    Raises:
      TimeoutError if timeout was succeeded before the scope entered the
      desired state
    """
    pass  # pragma: no cover

  @abstractmethod
  def run(self, normal: bool = True, timeout: float = 1) -> None:
    """Start acquiring waveforms, equivalent to pressing RUN button

    Args:
      normal: True will wait until the trigger occurs,
        False will generate a trigger after a time out (auto)
      timeout: Timeout for state confirmation

    Raises:
      TimeoutError if timeout was succeeded before the scope entered the
      desired state
    """
    pass  # pragma: no cover

  @abstractmethod
  def single(self,
             trigger_cmd: callable = None,
             force: bool = False,
             timeout: float = 1) -> None:
    """Acquire a single waveform, equivalent to pressing SINGLE button

    Args:
      trigger_cmd: Function to call to cause trigger (such as start toggling),
        None will wait 100ms
      force: False will wait until the trigger occurs,
        True will immediately generate a trigger regardless of actual trigger
      timeout: Timeout for state confirmation

    Raises:
      TimeoutError if timeout was succeeded before the scope entered the
      desired state
    """
    pass  # pragma: no cover

  @abstractmethod
  def force(self) -> None:
    """Immediately generate a trigger, equivalent to pressing FORCE button
    """
    pass  # pragma: no cover
