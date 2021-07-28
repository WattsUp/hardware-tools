from .MSO4000 import MSO4000

class MDO3000(MSO4000):

  def __init__(self, addr: str, checkIdentity: bool = True) -> None:
    super().__init__(addr, checkIdentity=False)
    if checkIdentity:
      query = self.ask('*IDN?')
      if query.startswith('TEKTRONIX,MDO3'):
        self.name = '-'.join(query.split(',')[:2])
      else:
        e = f'{addr} did not connect to a Tektronik MDO3000 scope\n'
        e += f'  \'*IDN?\' returned {query}'
        raise Exception(e)
    # TODO add other channel operations: RF
