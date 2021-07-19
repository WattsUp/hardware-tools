from unittest import TestCase

from hardware_tools.equipment import *

def test() -> None:
  for r in utility.getAvailableEquipment():
    if r.startswith("USB"):
      s = tektronik.MDO3000(r)
      break
  
  print(s)
  print(s.capture())