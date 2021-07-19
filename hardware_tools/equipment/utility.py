import pyvisa

def getAvailableEquipment() -> list[str]:
  rm = pyvisa.ResourceManager()
  return rm.list_resources()
