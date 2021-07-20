from ..scope import Scope
from ... import math

import numpy as np
from struct import unpack
import time

class MDO4000(Scope):

  def __init__(self, addr: str) -> None:
    '''!@brief Create a new Tektronix MDO4000 Scope object

    @param addr The address of the scope (VISA resource string)
    '''
    super().__init__(name='TEKTRONIX-MDO4000', addr=addr)
    query = self.ask('*IDN?')
    if query.startswith(
      'TEKTRONIX,MDO4') or query.startswith('TEKTRONIX,MDO3'):
      self.name = '-'.join(query.split(',')[:2])
    else:
      e = f'{addr} did not connect to a Tektronik MDO4000/MDO3000 scope\n'
      e += f'  \'*IDN?\' returned {query}'
      raise Exception(e)

    self.send('HEADER OFF')
    self.send('VERBOSE ON')

    # Add specific settings and commands
    self.settings.extend([])
    self.channels = [
      'CH1',
      'CH2',
      'CH3',
      'CH4',
      'MATH',
      'REF1',
      'REF2',
      'REF3',
      'REF4',
    ]
    self.channelSettings.extend([])
    self.commands.extend([
      'AUTOSCALE',
      'CLEARMENU'
    ])

  def configure(self, setting: str, value) -> str:
    setting = setting.upper()
    if setting not in self.settings:
      raise Exception(
        f'{self.name}@{self.addr} cannot configure setting \'{setting}\'')

    if setting == 'TIME_SCALE':
      self.send(f'HORIZONTAL:SCALE {float(value):.6E}')
      value = self.ask(f'HORIZONTAL:SCALE?')
      return f'{setting}={value}s'
    elif setting == 'TIME_OFFSET':
      self.send(f'HORIZONTAL:DELAY:MODE ON')
      self.send(f'HORIZONTAL:DELAY:TIME {float(value):.6E}')
      value = self.ask(f'HORIZONTAL:DELAY:TIME?')
      return f'{setting}={value}s'
    elif setting == 'TIME_POINTS':
      self.send(f'HORIZONTAL:RECORDLENGTH {int(value)}')
      value = self.ask(f'HORIZONTAL:RECORDLENGTH?')
      return f'{setting}={value}pts'
    elif setting == 'TRIGGER_MODE':
      value = value.upper()
      if value not in ['AUTO', 'NORM', 'NORMAL']:
        raise Exception(
          f'{self.name}@{self.addr} cannot set trigger mode \'{value}\'')
      self.send(f'TRIGGER:A:MODE {value}')
      value = self.ask(f'TRIGGER:A:MODE?')
      return f'{setting}={value}'
    elif setting == 'TRIGGER_SOURCE':
      value = value.upper()
      if value not in self.channels:
        raise Exception(
          f'{self.name}@{self.addr} cannot set trigger off of chanel \'{value}\'')
      self.send(f'TRIGGER:A:TYPE EDGE')
      self.send(f'TRIGGER:A:EDGE:SOURCE {value}')
      value = self.ask(f'TRIGGER:A:EDGE:SOURCE?')
      return f'{setting}={value}'
    elif setting == 'TRIGGER_COUPLING':
      value = value.upper()
      if value not in ['AC', 'DC', 'HFR', 'HFREJ',
                       'LFR', 'LFREJ', 'NOISE', 'NOISEREJ']:
        raise Exception(
          f'{self.name}@{self.addr} cannot set trigger coupling to \'{value}\'')
      self.send(f'TRIGGER:A:TYPE EDGE')
      self.send(f'TRIGGER:A:EDGE:COUPling {value}')
      value = self.ask(f'TRIGGER:A:EDGE:COUPling?')
      return f'{setting}={value}'
    elif setting == 'TRIGGER_POLARITY':
      value = value.upper()
      if value not in ['RIS', 'RISE', 'FALL', 'EITH', 'EITHER']:
        raise Exception(
          f'{self.name}@{self.addr} cannot set trigger polarity to \'{value}\'')
      self.send(f'TRIGGER:A:TYPE EDGE')
      self.send(f'TRIGGER:A:EDGE:SLOPE {value}')
      value = self.ask(f'TRIGGER:A:EDGE:SLOPE?')
      return f'{setting}={value}'
    elif setting == 'ACQUIRE_MODE':
      value = value.upper()
      if value not in ['SAM', 'SAMPLE', 'PEAK', 'PEAKDETECT',
                       'HIR', 'HIRES', 'AVE', 'AVERAGE', 'ENV', 'ENVELOPE']:
        raise Exception(
          f'{self.name}@{self.addr} cannot set acquire mode to \'{value}\'')
      self.send(f'ACQUIRE:MODE {value}')
      value = self.ask(f'ACQUIRE:MODE?')
      return f'{setting}={value}'

    raise Exception(
      f'{self.name}@{self.addr} cannot configure setting \'{setting}\'')

  def configureChannel(self, channel: str, setting: str, value) -> str:
    setting = setting.upper()
    if setting not in self.channelSettings:
      raise Exception(
        f'{self.name}@{self.addr} cannot configure chanel setting \'{setting}\'')
    channel = channel.upper()
    if channel not in self.channels:
      raise Exception(
        f'{self.name}@{self.addr} cannot configure chanel \'{channel}\'')

    if setting == 'SCALE':
      if channel in ['CH1', 'CH2', 'CH3', 'CH4']:
        self.send(f'{channel}:SCALE {float(value):.6E}')
        value = self.ask(f'{channel}:SCALE?')
        return f'{channel}.{setting}={value}'
    elif setting == 'POSITION':
      if channel in ['CH1', 'CH2', 'CH3', 'CH4']:
        self.send(f'{channel}:POSITION {float(value):.6E}')
        value = self.ask(f'{channel}:POSITION?')
        return f'{channel}.{setting}={value}'
    elif setting == 'OFFSET':
      if channel in ['CH1', 'CH2', 'CH3', 'CH4']:
        self.send(f'{channel}:OFFSET {float(value):.6E}')
        value = self.ask(f'{channel}:OFFSET?')
        return f'{channel}.{setting}={value}'
    elif setting == 'LABEL':
      if channel in ['CH1', 'CH2', 'CH3', 'CH4']:
        value = value.encode('ascii', errors='ignore').decode()
        self.send(f'{channel}:LABEL \'{value[:30]}\'')
        value = self.ask(f'{channel}:LABEL?')
        return f'{channel}.{setting}={value}'
    elif setting == 'BANDWIDTH':
      if channel in ['CH1', 'CH2', 'CH3', 'CH4']:
        if isinstance(value, str):
          self.send(f'{channel}:BANDWIDTH {value}')
        else:
          self.send(f'{channel}:BANDWIDTH {float(value):.6E}')
        value = self.ask(f'{channel}:BANDWIDTH?')
        return f'{channel}.{setting}={value}'
    elif setting == 'ACTIVE':
      self.send(f'SELECT:{channel} {value}')
      value = self.ask(f'SELECT:{channel}?')
      return f'{channel}.{setting}={value}'
    elif setting == 'TERMINATION':
      if channel in ['CH1', 'CH2', 'CH3', 'CH4']:
        self.send(f'{channel}:TERMINATION {value}')
        value = self.ask(f'{channel}:TERMINATION?')
        return f'{channel}.{setting}={value}'
    elif setting == 'INVERT':
      if channel in ['CH1', 'CH2', 'CH3', 'CH4']:
        self.send(f'{channel}:INVERT {value}')
        value = self.ask(f'{channel}:INVERT?')
        return f'{channel}.{setting}={value}'
    elif setting == 'PROBE_ATTENUATION':
      if channel in ['CH1', 'CH2', 'CH3', 'CH4']:
        self.send(f'{channel}:PROBE:GAIN {1 / float(value):.6E}')
        value = self.ask(f'{channel}:PROBE:GAIN?')
        return f'{channel}.PROBE_GAIN={value}'
    elif setting == 'PROBE_GAIN':
      if channel in ['CH1', 'CH2', 'CH3', 'CH4']:
        self.send(f'{channel}:PROBE:GAIN {float(value):.6E}')
        value = self.ask(f'{channel}:PROBE:GAIN?')
        return f'{channel}.{setting}={value}'
    elif setting == 'COUPLING':
      if channel in ['CH1', 'CH2', 'CH3', 'CH4']:
        value = value.upper()
        if value not in ['AC', 'DC', 'HFR', 'HFREJ',
                         'LFR', 'LFREJ', 'NOISE', 'NOISEREJ']:
          raise Exception(
            f'{self.name}@{self.addr} cannot set chanel \'{channel}\' coupling to \'{value}\'')
        self.send(f'{channel}:COUPLING {value}')
        value = self.ask(f'{channel}:COUPLING?')
        return f'{channel}.{setting}={value}'
    elif setting == 'TRIGGER_LEVEL':
      self.send(f'TRIGGER:A:LEVEL:{channel} {float(value):.6E}')
      value = self.ask(f'TRIGGER:A:LEVEL:{channel}?')
      return f'{channel}.{setting}={value}'

    raise Exception(
      f'{self.name}@{self.addr} cannot configure chanel \'{channel}\' setting \'{setting}\'')

  def command(self, command: str, channel: str = None,
              timeout: float = 1, silent: bool = True) -> None:
    command = command.upper()
    if command not in self.commands:
      raise Exception(
        f'{self.name}@{self.addr} cannot perform command \'{command}\'')

    if command == 'STOP':
      self.send('ACQUIRE:STATE STOP')
      self.waitForReply('TRIGGER:STATE?', ['SAVE'], timeout=timeout)
      return
    elif command == 'RUN':
      self.send('ACQUIRE:STATE STOP')
      self.send('ACQUIRE:STOPAFTER RUNSTOP')
      self.send('ACQUIRE:STATE RUN')
      self.waitForReply(
          'TRIGGER:STATE?', [
              'ARMED', 'AUTO', 'TRIGGER', 'READY'], timeout=timeout)
      return
    elif command == 'FORCE_TRIGGER':
      self.waitForReply(
          'TRIGGER:STATE?', ['READY', 'AUTO', 'SAVE', 'TRIGGER'], timeout=timeout)
      self.send('TRIGGER FORCE')
      return
    elif command == 'SINGLE':
      self.send('ACQUIRE:STATE STOP')
      self.waitForReply('TRIGGER:STATE?', ['SAVE'], timeout=timeout)
      self.send('ACQUIRE:STOPAFTER SEQUENCE')
      self.send('ACQUIRE:STATE RUN')
      time.sleep(0.5)

      if self.waitForReply(
        'TRIGGER:STATE?', ['ARMED', 'AUTO', 'SAVE'], timeout=timeout) != 'SAVE':
        self.waitForReply('TRIGGER:STATE?', ['SAVE'], timeout=timeout)
      self.waitForReply('ACQUIRE:NUMACQ?', ['1'])
      return
    elif command == 'SINGLE_FORCE':
      self.send('ACQUIRE:STATE STOP')
      self.waitForReply('TRIGGER:STATE?', ['SAVE'], timeout=timeout)
      self.send('ACQUIRE:STOPAFTER SEQUENCE')
      self.send('ACQUIRE:STATE RUN')
      time.sleep(0.5)

      if self.waitForReply(
        'TRIGGER:STATE?', ['ARMED', 'AUTO', 'SAVE'], timeout=timeout) == 'SAVE':
        self.waitForReply('ACQUIRE:NUMACQ?', ['1'])
        return

      reply = self.waitForReply(
          'TRIGGER:STATE?', ['TRIGGER', 'READY', 'SAVE'], timeout=timeout)
      if reply == 'SAVE':
        self.waitForReply('ACQUIRE:NUMACQ?', ['1'])
        return
      if reply == 'READY':
        self.send('TRIGGER FORCE')
        self.waitForReply(
            'TRIGGER:STATE?', [
                'TRIGGER', 'SAVE'], timeout=timeout)
      self.waitForReply('TRIGGER:STATE?', ['SAVE'], timeout=timeout)
      self.waitForReply('ACQUIRE:NUMACQ?', ['1'])
      return
    elif command == 'AUTOSCALE':
      if channel is None:
        raise Exception(
          f'{self.name}@{self.addr} cannot autoscale chanel \'None\'')
      channel = channel.upper()
      if channel not in self.channels:
        raise Exception(
          f'{self.name}@{self.addr} cannot autoscale chanel \'{channel}\'')

      if not silent:
        print(f'Autoscaling channel \'{channel}\'')

      attempts = 10
      while attempts > 0:
        if attempts != 10 and not silent:
          print(f' Remaining attempts: {attempts}')
        self.command('SINGLE_FORCE')
        data = self.readWaveform(channel=channel, raw=True)[0][1]
        attempts -= 1

        position = float(self.ask(f'{channel}:POSITION?'))
        scale = float(self.ask(f'{channel}:SCALE?'))
        newScale = scale
        newPosition = position

        dataMin = np.min(data) / 256
        dataMax = np.max(data) / 256
        dataMid = (dataMin + dataMax) / 2
        range = (dataMax - dataMin)
        # print(f'{dataMin:.2f}, {dataMid:.2f}, {dataMax:.2f}, {range:.2f}, {position:.2f}, {scale}')

        if dataMax > 0.95:
          if not silent:
            print('    Too high')
          if range > 0.6:
            newScale = scale * 4
          if range < 0.1:
            newScale = scale / 4
          newPosition = (position + 10 * (0.5 - dataMid)) * scale / newScale
        elif dataMin < 0.05:
          if not silent:
            print('    Too low')
          if range > 0.6:
            newScale = scale * 4
          if range < 0.1:
            newScale = scale / 4
          newPosition = (position + 10 * (0.5 - dataMid)) * scale / newScale
        elif range < 0.05:
          if not silent:
            print('    Too small')
          newScale = scale / 2
          newPosition = (position + 10 * (0.5 - dataMid)) * scale / newScale
        elif range > 0.9:
          if not silent:
            print('    Too large')
          newScale = scale * 2
          newPosition = (position + 10 * (0.5 - dataMid)) * scale / newScale
        else:
          if range < 0.7 or range > 0.9:
            if not silent:
              print('    Adjusting scale')
            newScale = scale / (0.8 / range)

          if dataMid > 0.6 or dataMid < 0.4 or newScale != scale:
            if not silent:
              print('    Adjusting position')
            newPosition = (position + 10 * (0.5 - dataMid)) * scale / newScale

        if newPosition != position or newScale != scale:
          if not silent:
            print(f'  Scale: {scale:.6g}=>{newScale:.6g}')
            print(f'  Position: {position:.2f}=>{newPosition:.2f}')
          self.configureChannel(channel, 'SCALE', newScale)
          self.configureChannel(channel, 'POSITION', newPosition)
          continue

        break

      if attempts == 0:
        raise Exception(
          f'{self.name}@{self.addr} failed to autoscale channel \'{channel}\'')

      return
    elif command == 'CLEARMENU':
      self.send('CLEARMENU')
      return

    raise Exception(
      f'{self.name}@{self.addr} cannot perform command \'{command}\'')

  def readWaveform(self, channel: str,
                   interpolate: float = 1, raw: bool = False) -> tuple[np.array, dict[str: str]]:
    channel = channel.upper()
    if channel not in self.channels:
      raise Exception(
        f'{self.name}@{self.addr} cannot read chanel \'{channel}\'')

    self.send(f'DATA:SOURCE {channel}')
    self.send('DATA:START 1')
    self.send('DATA:STOP 1E9')
    self.send('DATA:WIDTH 1')
    self.send('DATA:ENC RPB')

    xIncr = float(self.ask('WFMOUTPRE:XINCR?'))
    xZero = float(self.ask('WFMOUTPRE:XZERO?'))
    xUnit = self.ask('WFMOUTPRE:XUNIT?').replace('"', '')

    infoDict = {
      'tUnit': xUnit,
      'yUnit': 'ADC Counts',
      'tIncr': xIncr,
      'yIncr': 1
    }

    self.instrument.write('CURVE?')
    data = self.instrument.read_raw()
    headerLen = 2 + int(chr(data[1]), 16)
    wave = data[headerLen:-1]
    wave = np.array(unpack('%sB' % len(wave), wave))
    x = np.arange(xZero, xIncr * len(wave) + xZero, xIncr)

    if raw:
      return (np.array([x, wave]), infoDict)

    yMult = float(self.ask('WFMOUTPRE:YMULT?'))
    yZero = float(self.ask('WFMOUTPRE:YZERO?'))
    yOff = float(self.ask('WFMOUTPRE:YOFF?'))
    yUnit = self.ask('WFMOUTPRE:YUNIT?').replace('"', '')
    y = (wave - yOff) * yMult + yZero

    infoDict['yUnit'] = yUnit
    infoDict['yIncr'] = yMult

    if interpolate == 1:
      return (np.array([x, y]), infoDict)

    infoDict['tIncr'] = xIncr * interpolate
    infoDict['yIncr'] = None

    xNew = np.arange(xZero, xIncr * len(wave) + xZero, xIncr / interpolate)
    yNew = math.interpolate(x, y, xNew)
    return (np.array([xNew, yNew]), infoDict)
