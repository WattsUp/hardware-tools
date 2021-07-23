from hardware_tools.measurement.eyediagram import EyeDiagram, MaskDecagon
from hardware_tools.measurement import *

import json
import numpy as np
import os

def test() -> None:
  # waveforms = np.load('data/waveforms.npy')
  waveforms = np.load('data/waveforms.npy')[:, :, :1000000]
  # waveforms = np.load('data/waveforms.npy')[:1]
  with open('data/waveformInfo.json', 'r') as file:
    waveformInfo = json.load(file)

  m = MaskDecagon(0.18, 0.29, 0.35, 0.35, 0.38, 0.4, 0.55)
  e = EyeDiagram(waveforms, waveformInfo, mask=m)
  try:
    # e.calculate(nThreads=1)
    e.calculate()

    # import cProfile
    # cProfile.runctx('e.calculate()', {'e': e, 'calculate': e.calculate}, {})
    # import cProfile
    # import pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # e.calculate()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats()

  except KeyboardInterrupt:
    os.kill(0, 9)
