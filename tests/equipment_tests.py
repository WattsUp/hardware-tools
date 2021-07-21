from unittest import TestCase

from hardware_tools.equipment import *

import matplotlib.pyplot as pyplot
# from scipy.fft import fft, fftfreq

def test() -> None:
  for r in utility.getAvailableEquipment():
    if r.startswith('USB'):
      s = tektronik.MDO3000(r)
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

  print(s.configureChannel('CH1', 'SCALE', 10e-6))
  print(s.configureChannel('CH1', 'OFFSET', 0))
  print(s.configureChannel('CH1', 'POSITION', 0))
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

  s.command('CLEARMENU')
  s.command('AUTOSCALE', 'CH1')

  s.command('SINGLE_FORCE')
  data1Pair = s.readWaveform('CH1', interpolate=1)
  data1 = data1Pair[0]
  print(f'{1/data1Pair[1]["tIncr"]:.3g}')
  data10Pair = s.readWaveform('CH1', interpolate=10)
  data10 = data10Pair[0]
  print(f'{1/data10Pair[1]["tIncr"]:.3g}')

  # n = len(data10[0])
  # yf = fft(data10[1])
  # xf = fftfreq(n, data10[0][1] - data10[0][0])[:n//2]
  # pyplot.plot(xf, 2.0/n * np.abs(yf[0:n//2]))

  # n = len(data1[0])
  # yf = fft(data1[1])
  # xf = fftfreq(n, data1[0][1] - data1[0][0])[:n//2]
  # pyplot.plot(xf, 2.0/n * np.abs(yf[0:n//2]))
  # pyplot.xlim([0, 1/(data1[0][1] - data1[0][0])])

  pyplot.plot(data1[0], data1[1])
  pyplot.plot(data10[0], data10[1])
  pyplot.xlim([0, 30e-9])
  pyplot.show()
