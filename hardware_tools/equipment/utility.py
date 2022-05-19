"""Utility functions to ease equipment connection
"""

import re
from typing import Callable, List, Tuple

import pyvisa
from pyvisa import resources

from hardware_tools.equipment.equipment import Equipment
from hardware_tools.equipment import tektronix


def get_available(rm: pyvisa.ResourceManager = None) -> List[str]:
  """Get a list of available equipment addresses

  Args:
    rm: pyvisa ResourceManager to connect via, None for default

  Returns:
    List of VISA address strings
  """
  if rm is None:
    rm = pyvisa.ResourceManager()
  return list(rm.list_resources())


def connect(address: str, rm: pyvisa.ResourceManager = None) -> Equipment:
  """Open an address to an Equipment and return the appropriate derrived object

  Queries the identity and switches based on reply.

  Args:
    address: Address to the Equipment (VISA resource string)
    rm: pyvisa ResourceManager to connect via, None for default

  Returns:
    Derrived class of Equipment as appropriate

  Raises:
    LookupError if equipment ID is not recognized
  """
  if rm is None:
    rm = pyvisa.ResourceManager()
  with rm.open_resource(address) as instrument:
    if isinstance(instrument, resources.MessageBasedResource):
      identity = instrument.query("*IDN?").strip()
    else:
      raise NotImplementedError("Only know MessageBasedResource")
  classes = {
      "TEKTRONIX,MDO4": tektronix.MSO4000Family,
      "TEKTRONIX,MSO4": tektronix.MSO4000Family,
      "TEKTRONIX,DPO4": tektronix.MSO4000Family,
      "TEKTRONIX,MDO3": tektronix.MSO4000Family
  }
  for name, c in classes.items():
    if identity.startswith(name):
      return c(address, rm=rm)

  raise LookupError(f"Unknown equipment identity '{identity}'")


def parse_scpi(data: str, flat: bool = False, types: dict = None) -> dict:
  """Parse stream of scpi values

  Args:
    data: Stream to parse such as
      ":TRIGGER:A:MODE AUTO;TYPE EDGE;LEVEL 20.0000E-3;LEVEL:CH1..."
    flat: True will create a single dict with subkeys prefixed with parents
      False will create a nest of dicts with the subkeys
    types: Dictionary of known keys and subkeys with types (or callables) to
      convert the raw string to a int, float, etc. Keys are converted to upper
      before returning, any lower case letters need not match (non-verbose short
      version).

  Returns:
    Dict of values. If flat, subkeys are prefixed with parents. If not flat,
    parent keys will have a dict of subkeys.
  """
  data = data.removesuffix(":").strip(";").split(";")

  def key_match(k: str, d: dict) -> Tuple[str, str]:
    """Match a key in a dict with SCPI short hand rules

    Keys will only match all caps portions, lower case letters are optional.
    Matched keys will be converted to upper when returned.
    Use pipes | to multi-match.
    Use <number-number> to specify range such as "CH<1-4>"

    k = "wfmi" and d = {"WFMOpre|WFMInpre|WFMPre": a} will return (WFMINPRE, a)
    k = "wfm"  and d = {"WFMOpre|WFMInpre|WFMPre": a} will return (WFM, None)

    Args:
      k: Key to match
      d: Dict to search in

    Returns:
      Key (expanded if matched), dict element (None if no match)
    """
    if k is None or d is None:
      return k, None
    k = k.upper()
    for kd, vd in d.items():
      kd: str
      kd_list = []
      for kk in kd.split("|"):
        match = re.match(r"(\w+)<(\d+)\-(\d+)>", kk)
        if match:
          for i in range(int(match.group(2)), int(match.group(3)) + 1):
            kd_list.append(f"{match.group(1)}{i}")
        else:
          kd_list.append(kk)
      for kk in kd_list:
        kk_short = re.sub(r"([a-z]+)", lambda m: f"({m.group(1).upper()})?", kk)
        match = re.match(f"^{kk_short}$", k)
        if match:
          return kk.upper(), vd
    return k, None

  values = {}
  parent = ":"
  parent_d = values
  types_d = types
  for d in data:
    k, v = d.split(" ", maxsplit=1)
    k = k.upper().split(":")
    if k[0] == "":
      parent = ":"
      parent_d = values
      types_d = types
      k.pop(0)
    while len(k) > 1:
      k[0], types_d = key_match(k[0], types_d)
      parent += k[0] + ":"
      if not flat:
        if k[0] not in parent_d or not isinstance(parent_d[k[0]], dict):
          parent_d[k[0]] = {}
        parent_d = parent_d[k[0]]
      k.pop(0)
    k, type_callable = key_match(k[0], types_d)
    if isinstance(type_callable, Callable):
      v = type_callable(v)
    if flat:
      values[parent + k] = v
    else:
      parent_d[k] = v

  return values
