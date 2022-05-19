"""Mock pyvisa module to allow hardwareless testing
"""

from __future__ import annotations

from typing import Any, List

import pyvisa
from pyvisa import resources as pyvisa_resources

resources = {}
available = []

no_pop = False


class Resource(pyvisa_resources.MessageBasedResource):
  """Mock pyvisa.Resource
  """

  def __init__(self, resource_manager: pyvisa.ResourceManager,
               address: str) -> None:
    self.address = address
    super().__init__(resource_manager, address)
    resources[address] = self

    self.queue_tx = []
    self.queue_rx = []

    # Tuples are (type [or callable to convert str to value], value)
    self.query_map = {}

  def __enter__(self) -> Resource:
    return self

  def __exit__(self, *args) -> None:
    self.close()

  def __del__(self) -> None:
    self.close()

  def close(self) -> None:
    if not no_pop and self.address in resources:
      resources.pop(self.address)
      self.address = None

  def write(self, command: str) -> None:
    self.queue_rx.append(command)
    if " " not in command:
      return
    command_raw, value = command.split(" ", maxsplit=1)
    command = command_raw.split(":")
    d = self.query_map
    while len(command) > 1:
      if command[0] in d:
        d = d[command[0]]
        command = command[1:]
      else:
        d = None
        break
    k = command[0]
    if d is not None and k in d:
      if isinstance(d[k], tuple):
        d[k] = (d[k][0], d[k][0](value))
      else:
        e = TypeError("Cannot convert read only register: "
                      f"{command_raw} {d[k]}->{value}")
        raise pyvisa.VisaIOError(-1073807339) from e
    else:
      e = KeyError(f"Register not found: {command_raw}")
      raise pyvisa.VisaIOError(-1073807339) from e

  def query_str(self, keys: List[str], value: Any) -> str:
    if isinstance(value, tuple):
      return self.query_str(keys, value[1])
    elif callable(value):
      return self.query_str(keys, value())
    elif isinstance(value, bool):
      return "1" if value else "0"
    else:
      return str(value)

  def query(self, command: str) -> str:
    if not command.endswith("?"):
      # Not a query
      raise pyvisa.VisaIOError(-1073807339)

    self.queue_rx.append(command)
    command = command.removesuffix("?").split(":")
    keys = command
    d = self.query_map
    while len(command) > 1:
      if command[0] in d:
        d = d[command[0]]
        command = command[1:]
      else:
        d = None
        break
    k = command[0]
    if d is not None and k in d:
      return self.query_str(keys, d[k])
    if len(self.queue_tx) == 0:
      raise pyvisa.VisaIOError(-1073807339)
    return str(self.queue_tx.pop(0))

  def read_raw(self) -> bytes:
    if len(self.queue_tx) == 0:
      raise pyvisa.VisaIOError(-1073807339)
    return self.queue_tx.pop(0)


class VisaLib:

  def __init__(self) -> None:
    self.library_path = None


class ResourceManager:
  """Mock pyvisa.ResourceManager
  """

  def __init__(self) -> None:
    self.session = None
    self.visalib = VisaLib()

  def open_resource(self, address: str) -> Resource:
    if address in resources:
      return resources[address]
    return Resource(self, address)

  def list_resources(self) -> List[str]:
    return available
