from hardware_tools.equipment import utility
from unittest import TestCase

import json
# from scipy.fft import fft, fftfreq

import time
import numpy as np

def test() -> None:
  for r in utility.getAvailableEquipment():
    if r.startswith('USB'):
      s = utility.getEquipmentObject(r)
      break

  print(s)
  print(s.configure('TIME_SCALE', 10e-9))
  print(s.configure('TIME_OFFSET', 0e-9))
  print(s.configure('TIME_POINTS', 1))
  print(s.configure('TRIGGER_MODE', 'NORMAL'))
  print(s.configure('TRIGGER_SOURCE', 'CH1'))
  print(s.configure('TRIGGER_COUPLING', 'DC'))
  print(s.configure('TRIGGER_POLARITY', 'EITH'))
  print(s.configure('ACQUIRE_MODE', 'SAMPLE'))

  print(s.configureChannel('CH1', 'SCALE', 0.001e-6))
  print(s.configureChannel('CH1', 'OFFSET', 0))
  print(s.configureChannel('CH1', 'POSITION', 3))
  print(s.configureChannel('CH1', 'LABEL', 'Clock'))
  print(s.configureChannel('CH1', 'BANDWIDTH', 'FULL'))
  print(s.configureChannel('CH3', 'ACTIVE', 1))
  print(s.configureChannel('CH3', 'TERMINATION', 50e6))
  print(s.configureChannel('CH1', 'INVERT', 'OFF'))
  print(s.configureChannel('CH4', 'PROBE_ATTENUATION', 1000))
  print(s.configureChannel('CH4', 'PROBE_GAIN', 1000))
  print(s.configureChannel('CH2', 'COUPLING', 'AC'))
  print(s.configureChannel('CH1', 'TRIGGER_LEVEL', 10e-6))

  # s.command('STOP')
  # s.command('RUN')
  # time.sleep(0.5)
  # s.command('STOP')
  # print('Run/Stop', s.ask('ACQuire:NUMACq?'))

  # s.command('SINGLE')
  # print('Single', s.ask('ACQuire:NUMACq?'))

  # s.command('SINGLE_FORCE')
  # print('Single Force', s.ask('ACQuire:NUMACq?'))

  # print(s.configureChannel('CH1', 'TRIGGER_LEVEL', -1000))
  # s.command('RUN')
  # time.sleep(0.1)
  # s.command('FORCE_TRIGGER')
  # time.sleep(0.1)
  # s.command('FORCE_TRIGGER')
  # time.sleep(0.1)
  # s.command('FORCE_TRIGGER')
  # time.sleep(0.1)
  # s.command('FORCE_TRIGGER')
  # time.sleep(0.1)
  # s.command('STOP')
  # print('Run/Stop force trigger', s.ask('ACQuire:NUMACq?'))

  # s.command('SINGLE_FORCE')
  # print('Single Force no trigger', s.ask('ACQuire:NUMACq?'))

  # print(s.configure('TRIGGER_MODE', 'AUTO'))
  # s.command('RUN')
  # time.sleep(1.0)
  # s.command('STOP')
  # print('Run/Stop auto trigger', s.ask('ACQuire:NUMACq?'))

  # s.command('CLEARMENU')
  s.command('AUTOSCALE', channel='CH1')

  s.command('SINGLE_FORCE')
  dataPair = s.readWaveform('CH1', addNoise=True)

def collectTestData():
  for r in utility.getAvailableEquipment():
    if r.startswith('USB'):
      s = utility.getEquipmentObject(r)
      break

  s.configure('TIME_SCALE', 0.1e-9)
  s.configure('TIME_OFFSET', 0e-9)
  s.configure('TIME_POINTS', 1e5)
  s.configure('TRIGGER_MODE', 'NORMAL')
  s.configure('TRIGGER_SOURCE', 'CH1')
  s.configure('TRIGGER_COUPLING', 'DC')
  s.configure('TRIGGER_POLARITY', 'RISE')
  s.configure('ACQUIRE_MODE', 'SAMPLE')

  s.configureChannel('CH1', 'SCALE', 10e-6)
  s.configureChannel('CH1', 'OFFSET', 0)
  s.configureChannel('CH1', 'POSITION', 0)
  s.configureChannel('CH1', 'LABEL', '')
  s.configureChannel('CH1', 'BANDWIDTH', 'FULL')
  s.configureChannel('CH1', 'INVERT', 'OFF')
  # s.configureChannel('CH1', 'TRIGGER_LEVEL', 10e-6)
  s.configureChannel('CH1', 'TRIGGER_LEVEL', -1000)
  s.configureChannel('CH1', 'ACTIVE', 1)
  s.configureChannel('CH2', 'ACTIVE', 0)
  s.configureChannel('CH3', 'ACTIVE', 0)
  s.configureChannel('CH4', 'ACTIVE', 0)

  s.command('AUTOSCALE', channel='CH1')

  waveforms = []

  s.command('SINGLE_FORCE')
  data, info = s.readWaveform('CH1', addNoise=True)
  info['clippingTop'] = int(info['clippingTop'])
  info['clippingBottom'] = int(info['clippingBottom'])

  waveforms.append(data)

  for _ in range(1, 10):
    s.command('SINGLE_FORCE')
    data, _ = s.readWaveform('CH1', addNoise=True)
    waveforms.append(data)

  waveforms = np.array(waveforms)
  np.save('data/waveforms.npy', waveforms)
  with open('data/waveformInfo.json', 'w') as file:
    json.dump(info, file)
