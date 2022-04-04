"""Utility functions to ease equipment connection
"""

from typing import List

import pyvisa

from hardware_tools.equipment.equipment import Equipment
from hardware_tools.equipment import tektronix


def get_available() -> List[str]:
  """Get a list of available equipment addresses

  Returns:
    List of VISA address strings
  """
  rm = pyvisa.ResourceManager()
  return rm.list_resources()


def connect(address: str) -> Equipment:
  """Open an address to an Equipment and return the appropriate derrived object

  Queries the identity and switches based on reply.

  Args:
    address: Address to the Equipment (VISA resource string)

  Returns:
    Derrived class of Equipment as appropriate

  Raises:
    LookupError if equipment ID is not recognized
  """
  rm = pyvisa.ResourceManager()
  with rm.open_resource(address) as instrument:
    identity = instrument.query("*IDN?").strip()
  if identity.startswith("TEKTRONIX,MSO4"):
    return tektronix.MSO4000(address)
  if identity.startswith("TEKTRONIX,MDO4"):
    return tektronix.MDO4000(address)
  if identity.startswith("TEKTRONIX,MDO3"):
    return tektronix.MDO3000(address)

  raise LookupError(f"Unknown equipment identity '{identity}'")
