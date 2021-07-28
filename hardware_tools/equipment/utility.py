from hardware_tools.equipment.scope import Scope
from typing import Union
import pyvisa

from .equipment import Equipment
from . import tektronik

def getAvailableEquipment() -> list[str]:
  '''!@brief Get a list of available equipment addresses

  @return list[str] List of VISA address strings
  '''
  rm = pyvisa.ResourceManager()
  return rm.list_resources()

def getEquipmentObject(addr: str) -> Union[Scope]:
  '''!@brief Ask equipment for identity and create appropriate Equipment object

  @param addr The address of the equipment
  @return Equipment Appropriate Equipment object
  '''
  e = Equipment('', addr)
  identity = e.ask('*IDN?')
  if identity.startswith('TEKTRONIX,MSO4'):
    return tektronik.MSO4000(addr)
  if identity.startswith('TEKTRONIX,MDO4'):
    return tektronik.MDO4000(addr)
  if identity.startswith('TEKTRONIX,MDO3'):
    return tektronik.MDO3000(addr)

  raise Exception(f'Unknown equipment identity \'{identity}\'')
