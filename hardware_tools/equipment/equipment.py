import pyvisa
import time

# TODO [Future] add more scopes and other instrument types

class Equipment:

  settings = []

  commands = []

  def __init__(self, name: str, addr: str) -> None:
    '''!@brief Create a new abstract Equipment object

    @param name The name of the Equipment
    @param addr The address of the Equipment (VISA resource string)
    '''
    self.name = name
    self.addr = addr

    rm = pyvisa.ResourceManager()
    self.instrument = rm.open_resource(addr)
    # TODO add TCP socket connection type

  def __str__(self) -> str:
    '''!@brief Get a string representation of the Equipment

    @return str
    '''
    return f'{self.name} @ {self.addr}'

  def send(self, cmd: str) -> None:
    '''!@brief Send a command to the Equipment

    @param cmd Command string to write
    '''
    self.instrument.write(cmd)

  def ask(self, cmd: str) -> str:
    '''!@brief Send a command to the Equipment and receive a reply

    @param cmd Command string to write
    @return str Reply
    '''
    return self.instrument.query(cmd).strip()

  def receive(self) -> bytes:
    '''!@brief Receive raw data from the Equipment

    @return bytes Reply
    '''
    return self.instrument.read_raw()

  def configure(self, setting: str, value) -> str:
    '''!@brief Configure a setting to a new value

    @param setting The setting to change (see self.settings)
    @param value The value to change to
    @return str Setting change validation
    '''
    raise Exception('configure called on base Equipment')

  def command(self, command: str, timeout: float = 1,
              silent: bool = True) -> None:
    '''!@brief Perform a command sequence

    @param command The command to perform (see self.commands)
    @param timeout Time in seconds to wait until giving up
    @param silent True will not print anything except errors
    '''
    raise Exception('command called on base Equipment')

  def waitForReply(
    self, cmd: str, states: list[str], timeout: float = 1, repeatSend: str = None) -> str:
    '''!@brief Send a command to the Equipment and wait repeat until reply is desired

    @param cmd Command string to self.ask
    @param states Desired states. Returns if reply matches any element
    @param timeout Time in seconds to wait until giving up
    @param repeatSend Additional command string to send before each ask
    @return str Last reply
    '''
    interval = 0.05
    timeout = int(timeout / interval)

    seenStates = []
    if repeatSend:
      self.send(repeatSend)
    state = self.ask(cmd)
    seenStates.append(state)
    while (state not in states and timeout >= 0):
      time.sleep(interval)
      if repeatSend:
        self.send(repeatSend)
      state = self.ask(cmd)
      seenStates.append(state)
      timeout -= 1
    # print(f'Waited {len(seenStates) * interval:.2f}s')
    if timeout < 0:
      raise Exception(
        f'{self.name}@{self.addr} failed to wait for \'{cmd}\' = \'{states}\' = \'{seenStates}\'')
    return state
