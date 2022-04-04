"""Scope base class to interface to oscilloscope
"""

from abc import abstractmethod
from typing import Any, Tuple

import numpy as np

from hardware_tools.equipment import equipment


class Scope(equipment.Equipment):
  """Scope base class to interface to oscilloscope

  Properties:
    settings: List of settings recognized to change
    commands: List of commands recognized to execute
    channels: List of channels recognized to read
    channel_settings: List of channel specific settings recognized to change
  """

  settings = [
      "SAMPLE_RATE", "TIME_SCALE", "TIME_OFFSET", "TIME_POINTS", "TRIGGER_MODE",
      "TRIGGER_SOURCE", "TRIGGER_COUPLING", "TRIGGER_POLARITY", "ACQUIRE_MODE"
  ]

  channels = ["CH1"]

  channel_settings = [
      "SCALE", "POSITION", "OFFSET", "LABEL", "BANDWIDTH", "ACTIVE",
      "TERMINATION", "INVERT", "PROBE_ATTENUATION", "PROBE_GAIN", "COUPLING",
      "TRIGGER_LEVEL"
  ]

  commands = ["STOP", "RUN", "FORCE_TRIGGER", "SINGLE", "SINGLE_FORCE"]

  @abstractmethod
  def configure_channel(self, channel: str, setting: str, value: Any) -> Any:
    """Configure a channel setting to a new value

    Args:
      channel: The channel to configure (see self.channels)
      setting: The setting to change (see self.channel_settings)
      value: The value to change to

    Returns:
      Setting change validation

    Raises:
      KeyError if setting is improper

      ValueError if value is improper
    """
    pass  # pragma: no cover

  @abstractmethod
  def command(self,
              command: str,
              timeout: float = 1,
              silent: bool = True,
              channel: str = None) -> None:
    """Perform a command sequence

    Args:
      command: The command to perform (see self.commands)
      timeout: Time in seconds to wait until giving up
      silent: True will not print anythin except errors
      channel: The channel to perform on if applicable (see self.channels)

    Raises:
      KeyError if command is improper

      ValueError if channel is improper

      TimeoutError if timeout was exceeded
    """
    pass  # pragma: no cover

  @abstractmethod
  def read_waveform(self,
                    channel: str,
                    raw: bool = False,
                    add_noise: bool = False) -> Tuple[np.ndarray, dict]:
    """Read waveform from the Scope

    Stop the scope before reading multiple channels to ensure concurrent
    sampling.

    Args:
      channel: The channel to read (see self.channels)
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

    Raises:
      ValueError if channel is improper
    """
    pass  # pragma: no cover
