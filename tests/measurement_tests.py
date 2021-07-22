from hardware_tools.measurement.eyediagram import EyeDiagram
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

  e = EyeDiagram(waveforms, waveformInfo)
  try:
    # e.calculate(nThreads=1)
    e.calculate()
  except KeyboardInterrupt:
    os.kill(0, 9)
