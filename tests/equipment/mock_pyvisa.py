"""Mock pyvisa module to allow hardwareless testing
"""

from __future__ import annotations

from typing import List

import pyvisa

resources = {}
available = []

no_pop = False


class Resource:
  """Mock pyvisa.Resource
  """

  def __init__(self, address: str) -> None:
    self.address = address
    resources[address] = self

    self.queue_tx = []
    self.queue_rx = []
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

  def write(self, command: str) -> None:
    self.queue_tx.append(command)

  def query(self, command: str) -> str:
    self.queue_tx.append(command)
    if command in self.query_map:
      return self.query_map[command]
    if len(self.queue_rx) == 0:
      raise pyvisa.VisaIOError(-1073807339)
    return self.queue_rx.pop(0)

  def read_raw(self) -> bytes:
    if len(self.queue_rx) == 0:
      raise pyvisa.VisaIOError(-1073807339)
    return self.queue_rx.pop(0)


class ResourceManager:
  """Mock pyvisa.ResourceManager
  """

  def open_resource(self, address: str) -> Resource:
    if address in resources:
      return resources[address]
    return Resource(address)

  def list_resources(self) -> List[str]:
    return available
