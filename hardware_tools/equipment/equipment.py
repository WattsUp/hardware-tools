"""Equipment base class to interface to physical testing hardware
"""

from __future__ import annotations

from abc import ABC
import time
from typing import Iterable

import pyvisa
from pyvisa import resources

# TODO (WattsUp) [Future] add more scopes and other instrument types


class Equipment(ABC):
  """Equipment base class to interface to physical testing hardware

  Properties:
    settings: List of settings recognized to change
    commands: List of commands recognized to execute
  """

  settings = []
  commands = []

  def __init__(self,
               address: str,
               rm: pyvisa.ResourceManager = None,
               name: str = "") -> None:
    """Initialize Equipment by connecting to it

    Args:
      address: Address to the Equipment (VISA resource string)
      rm: pyvisa ResourceManager to connect via, None for default
      name: Name of the Equipment
    """
    self._instrument = None
    self._address = address
    self._name = name

    if rm is None:
      rm = pyvisa.ResourceManager()
    resource = rm.open_resource(address)

    if not isinstance(resource, resources.MessageBasedResource):
      raise NotImplementedError("Only know MessageBasedResource")
    self._instrument = resource

  def __del__(self) -> None:
    self.close()

  def close(self) -> None:
    if self._instrument is not None:
      self._instrument.close()
      self._instrument = None

  def __enter__(self) -> Equipment:
    return self

  def __exit__(self, *args) -> None:
    self.close()

  def __repr__(self) -> str:
    return f"{self._name} @ {self._address}"

  def send(self, command: str) -> None:
    """Send a command to the Equipment

    Args:
      command: Command string to write
    """
    self._instrument.write(command)

  def reset(self) -> None:
    """Send reset command
    """
    self.send("*RST")
    self.send("*WAI")  # Wait until complete

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
