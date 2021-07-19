import numpy as np
import pyvisa

class Scope:

  def __init__(self, name: str, addr: str) -> None:
    self.name = name
    self.addr = addr

    rm = pyvisa.ResourceManager()
    self.instrument = rm.open_resource(addr)

  def __str__(self):
    return f'{self.name} @ {self.addr}'

  def capture(self, n: int = 1) -> np.array:
    data = [] * n
    return np.array(data)
