from ..scope import Scope

import colorama
import numpy as np

colorama.init(autoreset=True)

class MDO4000(Scope):

  def __init__(self, addr: str) -> None:
    super().__init__(name="TEKTRONIX-MDO4000", addr=addr)
    query = self.instrument.query("*IDN?")
    if query.startswith(
      "TEKTRONIX,MDO4") or query.startswith("TEKTRONIX,MDO3"):
      self.name = '-'.join(query.split(',')[:2])
    else:
      e = f'{colorama.Fore.RED}{addr} did not connect to a Tektronik MDO4000/MDO3000 scope\n'
      e += f'  "*IDN?" returned {colorama.Fore.YELLOW}{query}'
      raise BaseException(e)

  def captureRaw(self) -> np.array:
    print("Capturing raw from MDO4000")
