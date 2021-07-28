import numpy as np

from .equipment import Equipment

# TODO [Future] add more scopes and other instrument types

class Scope(Equipment):

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

  def configureChannel(self, channel: str, setting: str, value) -> str:
    '''!@brief Configure a channel setting to a new value

    @param channel The channel to configure (see self.channels)
    @param setting The setting to change (see self.channelSettings)
    @param value The value to change to
    @return str Setting change validation
    '''
    raise Exception('configureChannel called on base Scope')

  def command(self, command: str, timeout: float = 1,
              silent: bool = True, channel: str = None) -> None:
    '''!@brief Perform a command sequence

    @param command The command to perform (see self.commands)
    @param timeout Time in seconds to wait until giving up
    @param silent True will not print anything except errors
    @param channel The channel to perform on if applicable (see self.channels)
    '''
    raise Exception('command called on base Scope')

  def readWaveform(self, channel: str, raw: bool = False,
                   addNoise: bool = False) -> tuple[np.ndarray, dict]:
    '''!@brief Read waveform from the Scope and interpolate as necessary (sinc interpolation)

    Stop the scope before reading multiple channels to ensure same sampling time

    @param channel The channel to read
    @param raw True will return raw ADC values, False (default) will transform into real-world units
    @param addNoise True will add uniform noise to the LSB for antialiasing, False will not
    @return tuple[np.array(dtype=float), dict]
        Samples are columns [[t0, t1,..., tn], [y0, y1,..., yn]]
        Dictionary are units ['tUnit', 'yUnit']:str, scales ['tIncr', 'yIncr']:float, and clippingTop/Bottom:bool
    '''
    raise Exception('readWaveform called on base Scope')
