import numpy as np
import pyvisa
import time

# TODO [Future] add more scopes and other instrument types

class Scope:

  settings = [
    'TIME_SCALE',
    'TIME_OFFSET',
    'TIME_POINTS',
    'TRIGGER_MODE',
    'TRIGGER_SOURCE',
    'TRIGGER_COUPLING',
    'TRIGGER_POLARITY',
    'ACQUIRE_MODE'
  ]

  channels = [
    'CH1'
  ]

  channelSettings = [
    'SCALE',
    'POSITION',
    'OFFSET',
    'LABEL',
    'BANDWIDTH',
    'ACTIVE',
    'TERMINATION',
    'INVERT',
    'PROBE_ATTENUATION',
    'PROBE_GAIN',
    'COUPLING',
    'TRIGGER_LEVEL'
  ]

  commands = [
    'STOP',
    'RUN',
    'FORCE_TRIGGER',
    'SINGLE',
    'SINGLE_FORCE'
  ]

  def __init__(self, name: str, addr: str) -> None:
    '''!@brief Create a new abstract Scope object

    @param name The name of the scope
    @param addr The address of the scope (VISA resource string)
    '''
    self.name = name
    self.addr = addr

    rm = pyvisa.ResourceManager()
    self.instrument = rm.open_resource(addr)
    # TODO add TCP socket connection type

  def __str__(self) -> str:
    '''!@brief Get a string representation of the Scope

    @return str
    '''
    return f'{self.name} @ {self.addr}'

  def send(self, cmd: str) -> None:
    '''!@brief Send a command to the Scope

    @param cmd Command string to write
    '''
    self.instrument.write(cmd)

  def ask(self, cmd: str) -> str:
    '''!@brief Send a command to the Scope and receive a reply

    @param cmd Command string to write
    @return str Reply
    '''
    return self.instrument.query(cmd).strip()

  def receive(self) -> bytes:
    '''!@brief Receive raw data from the Scope

    @return bytes Reply
    '''
    return self.instrument.read_raw()

  def configure(self, setting: str, value) -> str:
    '''!@brief Configure a setting to a new value

    @param setting The setting to change (see self.settings)
    @param value The value to change to
    @return str Setting change validation
    '''
    raise Exception('configure called on base Scope')

  def configureChannel(self, channel: str, setting: str, value) -> str:
    '''!@brief Configure a channel setting to a new value

    @param channel The channel to configure (see self.channels)
    @param setting The setting to change (see self.channelSettings)
    @param value The value to change to
    @return str Setting change validation
    '''
    raise Exception('configureChannel called on base Scope')

  def command(self, command: str, channel: str = None,
              timeout: float = 1, silent: bool = True) -> None:
    '''!@brief Perform a command sequence

    @param command The command to perform (see self.commands)
    @param channel The channel to perform on if applicable (see self.channels)
    @param timeout Time in seconds to wait until giving up
    @param silent True will not print anything except errors
    '''
    raise Exception('command called on base Scope')

  def readWaveform(self, channel: str, interpolate: float = 1,
                   raw: bool = False) -> tuple[np.ndarray, dict]:
    '''!@brief Read waveform from the Scope and interpolate as necessary (sinc interpolation)

    Stop the scope before reading multiple channels to ensure same sampling time

    @param channel The channel to read
    @param interpolate The interpolation factor. 1 will return original, >1 will upsample, <1 will downsample
    @param raw True will return raw ADC values, False (default) will transform into real-world units
    @return tuple[np.array(dtype=float), dict]
        Samples are columns [[t0, t1,..., tn], [y0, y1,..., yn]]
        Diction are units ['tUnit', 'yUnit']:str and scales ['tIncr', 'yIncr']:float
    '''
    raise Exception('readWaveform called on base Scope')

  def waitForReply(
    self, cmd: str, states: list[str], timeout: float = 1) -> str:
    '''!@brief Send a command to the Scope and wait repeat until reply is desired

    @param cmd Command string to self.ask
    @param states Desired states. Returns if reply matches any element
    @param timeout Time in seconds to wait until giving up
    @return str Last reply
    '''
    interval = 0.05
    timeout = int(timeout / interval)

    seenStates = []
    state = self.ask(cmd)
    seenStates.append(state)
    while (state not in states and timeout >= 0):
      time.sleep(interval)
      state = self.ask(cmd)
      seenStates.append(state)
      timeout -= 1
    # print(f'Waited {len(seenStates) * interval:.2f}s')
    if timeout < 0:
      raise Exception(
        f'{self.name}@{self.addr} failed to wait for \'{cmd}\' = \'{states}\' = \'{seenStates}\'')
    return state
