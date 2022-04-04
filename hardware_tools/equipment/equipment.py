"""Equipment base class to interface to physical testing hardware
"""

from abc import ABC, abstractmethod
import time
from typing import Any, Iterable

import pyvisa

# TODO (WattsUp) [Future] add more scopes and other instrument types


class Equipment(ABC):
  """Equipment base class to interface to physical testing hardware

  Properties:
    settings: List of settings recognized to change
    commands: List of commands recognized to execute
  """

  settings = []
  commands = []

  def __init__(self, address: str, name: str = "") -> None:
    """Initialize Equipment by connecting to it

    Args:
      address: Address to the Equipment (VISA resource string)
      name: Name of the Equipment
    """
    self._address = address
    self._name = name

    rm = pyvisa.ResourceManager()
    self._instrument = rm.open_resource(address)

  def __del__(self) -> None:
    try:
      self._instrument.close()
    except AttributeError:  # pragma: no cover
      pass

  def __repr__(self) -> str:
    return f"{self._name} @ {self._address}"

  def send(self, command: str) -> None:
    """Send a command to the Equipment

    Args:
      command: Command string to write
    """
    self._instrument.write(command)

  def ask(self, command: str) -> str:
    """Send a command to the Equipment and receive a reply

    Args:
      command: Command string to write

    Returns:
      Reply string
    """
    return self._instrument.query(command).strip()

  def ask_and_wait(self,
                   command: str,
                   states: Iterable[str],
                   timeout: float = 1,
                   additional_command: str = None) -> str:
    """Send a command to the Equipment and wait until reply is desired

    Args:
      command: Command string to write (passed to self.ask)
      states: Desired states. Returns if reply matches any element
      timeout: Time in seconds to wait until giving up
      additional_command: Additional command string to send before each ask

    Returns:
      Last reply received

    Raises:
      TimeoutError if timeout was exceeded
    """
    interval = 0.05
    timeout = int(timeout / interval)

    seen = []
    if additional_command is not None:
      self.send(additional_command)
    state = self.ask(command)
    seen.append(state)
    while (state not in states and timeout >= 0):
      time.sleep(interval)
      if additional_command is not None:
        self.send(additional_command)
      state = self.ask(command)
      seen.append(state)
      timeout -= 1
    if timeout < 0:
      raise TimeoutError(
          f"{self} failed to wait for '{command}' = '{states}' = '{seen}'")
    return state

  def receive(self) -> bytes:
    """Receive raw data from the Equipment

    Returns:
      Reply as bytes
    """
    return self._instrument.read_raw()

  @abstractmethod
  def configure(self, setting: str, value: Any) -> Any:
    """Configure a setting to a new value

    Args:
      setting: The setting to change (see self.settings)
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
              silent: bool = True) -> None:
    """Perform a command sequence

    Args:
      command: The command to perform (see self.commands)
      timeout: Time in seconds to wait until giving up
      silent: True will not print anything except errors

    Raises:
      TimeoutError if timeout was exceeded
    """
    pass  # pragma: no cover
