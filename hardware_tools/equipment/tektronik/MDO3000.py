from .MDO4000 import MDO4000

class MDO3000(MDO4000):

  def __init__(self, addr: str) -> None:
    super().__init__(addr)
