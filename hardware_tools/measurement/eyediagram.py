from __future__ import annotations
from hardware_tools.measurement.extension.extensionSlow import getCrossingSlow
import colorama
from colorama import Fore
import datetime
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as pyplot
from matplotlib.ticker import FuncFormatter
from multiprocessing import Pool, cpu_count
import numpy as np
import os
from PIL import Image
import skimage.draw

from ..math import *
from .extension import getEdgesNumpy, getCrossing, getHits, isHitting
from . import brescount

colorama.init(autoreset=True)

class Mask:
  def __init__(self) -> None:
    '''!@brief Create a new Mask
    Derive mask shapes from this class such as a MaskOctagon, MaskDecagon, MaskPulse, etc.
    '''
    # TODO add support for AC coupled masks
    self.paths = []

  def adjust(self, factor: float) -> Mask:
    '''!@brief Adjust the size of the Mask and return a new Mask

    @param factor Adjustment factor -1=no mask, 0=unadjusted, 1=no gaps
    @return Mask Newly adjusted mask
    '''
    raise Exception('Adjust called on base Mask')

class MaskDecagon(Mask):
  def __init__(self, x1: float, x2: float, x3: float, y1: float,
               y2: float, y3: float, y4: float = None) -> None:
    '''!@brief Create a new Mask from systematic definition with a center decagon (10-sides)

    All inputs are assumed to be positive (negative allowed) and normalized to bit duration / amplitude.

           ▐███████████████████████████████████████████████████████▌
    1 + y4 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
           │       ┆       ┆       ┆       ┆       ┆       ┆       │
         1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
           │       ┆       ┆       ┆       ┆       ┆       ┆       │
    1 - y1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄▄▄▄▄█████████▄▄▄▄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
    1 - y2 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄▄█████████████████████████▄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
           │       ┆ ▄▄▄███████████████████████████████▄▄▄ ┆       │
       0.5 ├┄┄┄┄┄┄┄█████████████████████████████████████████┄┄┄┄┄┄┄┤
           │       ┆ ▀▀▀███████████████████████████████▀▀▀ ┆       │
        y2 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄▀█████████████████████████▀┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
        y1 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄▀▀▀▀█████████▀▀▀▀┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
           │       ┆       ┆       ┆       ┆       ┆       ┆       │
         0 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
           │       ┆       ┆       ┆       ┆       ┆       ┆       │
       -y3 ├┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┤
           ▐███████████████████████████████████████████████████████▌
           │       ┆       ┆       ┆       ┆       ┆       ┆       │
           0      x1      x2      x3     1-x3    1-x2    1-x1      1


    @param x1 Normalized time [UI] to 50% corner
    @param x2 Normalized time [UI] to y2 corner
    @param x3 Normalized time [UI] to y1 corner
    @param y1 Normalized amplitude [UA] to x3 edge
    @param y2 Normalized amplitude [UA] to x2 corner
    @param y3 Normalized amplitude [UA] to lower threshold
    @param y4 Normalized amplitude [UA] to upper threshold (None will take y3's value)
    '''
    super().__init__()

    self.x1 = x1
    self.x2 = x2
    self.x3 = x3
    self.y1 = y1
    self.y2 = y2
    self.y3 = y3
    if y4 is None:
      y4 = y3
    self.y4 = y4

    pathCenter = [
      (x1, 0.5),
      (x2, 1 - y2),
      (x3, 1 - y1),
      (1 - x3, 1 - y1),
      (1 - x2, 1 - y2),
      (1 - x1, 0.5),
      (1 - x2, y2),
      (1 - x3, y1),
      (x3, y1),
      (x2, y2),
      (x1, 0.5)
    ]

    pathUpper = [
      (0, 10),
      (0, 1 + y4),
      (1, 1 + y4),
      (1, 10)
    ]

    pathLower = [
      (0, -10),
      (0, -y3),
      (1, -y3),
      (1, -10)
    ]

    self.paths.append(pathCenter)
    self.paths.append(pathUpper)
    self.paths.append(pathLower)

  def adjust(self, factor: float) -> MaskDecagon:
    if factor > 1:
      x1 = factor * 0.0 + (1 - factor) * self.x1
      x2 = factor * 0.0 + (1 - factor) * self.x2
      x3 = factor * 0.0 + (1 - factor) * self.x3
      y1 = factor * 0.0 + (1 - factor) * self.y1
      y2 = factor * 0.0 + (1 - factor) * self.y2
      y3 = factor * 0.0 + (1 - factor) * self.y3
      y4 = factor * 0.0 + (1 - factor) * self.y4
    else:
      factor = -factor
      x1 = factor * 0.5 + (1 - factor) * self.x1
      x2 = factor * 0.5 + (1 - factor) * self.x2
      x3 = factor * 0.5 + (1 - factor) * self.x3
      y1 = factor * 0.5 + (1 - factor) * self.y1
      y2 = factor * 0.5 + (1 - factor) * self.y2
      y3 = factor * 0.5 + (1 - factor) * self.y3
      y4 = factor * 0.5 + (1 - factor) * self.y4
    return MaskDecagon(x1, x2, x3, y1, y2, y3, y4)

  def toDict(self) -> dict:
    '''!@brief Get a dictionary representation of the Mask

    @return dict
    '''
    return {
      'type': 'decagon',
      'x1': self.x1,
      'x2': self.x2,
      'x3': self.x3,
      'y1': self.y1,
      'y2': self.y2,
      'y3': self.y3,
      'y4': self.y4
    }

class EyeDiagram:
  def __init__(self, waveforms: np.ndarray, waveformInfo: dict, mask: Mask = None, resolution: int = 2000,
               nBitsMax: int = 5, method: str = 'average', resample: int = 50, yLevels: list = None, hysteresis: float = 0.1, tBit: float = None, pllBandwidth: float = 100e3) -> None:
    '''!@brief Create a new EyeDiagram from a collection of waveforms

    Assumes signal has two levels

    Waveforms must have same dimensions
    Waveforms must have same sampling period and units

    @param waveforms 3d array of waveforms [waveform0([[t0, t1,..., tn], [y0, y1,..., yn]]), waveform1,...]
    @param waveformInfo Dictionary of units ['tUnit', 'yUnit']:str and scales ['tIncr']:float
    @param mask Mask object for hit detection, None will not check a mask
    @param resolution Resolution of square eye diagram image
    @param nBitsMax Maximum number of consecutive bits of the same state allowed before an exception is raised
    @param method Method for deciding on values 'average' runs a simple average, 'peak' will find the peak of a gaussian curve fit
    @param resample n=0 will not resample, n>0 will use sinc interpolation to resample a single bit to at least n segments (tDelta = tBit / n)
    @param yLevels Manually specify fixed levels for state identification, None will auto calculate levels
    @param hysteresis Difference between rising and falling thresholds (units of normalized amplitude)
    @param pllBandwidth -3dB cutoff frequency of pll feedback (1st order low pass)
    '''
    if len(waveforms.shape) != 3:
      raise Exception(
        f'waveforms must contain 3 axis: nWaveform, time domain, vertical domain not {waveforms.shape}')
    self.waveforms = waveforms
    self.tUnit = waveformInfo['tUnit']
    self.yUnit = waveformInfo['yUnit']
    self.tDelta = waveformInfo['tIncr']
    self.resolution = resolution
    self.mask = mask
    self.nBitsMax = nBitsMax
    self.nBitsMin = 0.2
    self.method = method
    self.resample = resample
    self.manualYLevels = yLevels
    self.manualTBit = tBit
    self.calculated = False
    self.hysteresis = hysteresis
    self.histNMax = 100e3
    self.imageMin = -0.5
    self.imageMax = 1.5
    self.pllBandwidth = pllBandwidth

  def calculate(self, printProgress: bool = True, indent: int = 0, plot: bool = False,
                nThreads: int = 0) -> dict:
    '''!@brief Calculate and prepare eye diagram

    @param printProgress True or int will print progress statements (int specifies indent). False will not
    @param indent Base indentation for progress statements
    @param plot True will create plots during calculation (histograms and such). False will not
    @param multithreaded Specify number of threads to use, 0=all, 1=single, or n
    @return dict Report dictionary of measures and images, see getMeasures() and getImage()
    '''
    start = datetime.datetime.now()
    if nThreads == 0:
      nThreads = cpu_count()
    else:
      nThreads = min(nThreads, cpu_count())

    if printProgress:
      print(
        f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}Starting eye diagram calculation with {nThreads} threads')
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Finding threshold levels')
    self.__calculateLevels(plot=plot, nThreads=nThreads)

    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Determining receiver clock')
    self.__calculateClock(plot=plot, nThreads=nThreads)

    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Extracting bit centers')
    self.__extractBitCenters(plot=plot, nThreads=nThreads)

    self.calculated = True
    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Measuring waveform')
    self.__measureWaveform(
        plot=plot,
        printProgress=printProgress,
        indent=2 + indent,
        nThreads=nThreads)
    self.calculated = False

    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Generating images')
    self.__generateImages(
        printProgress=printProgress,
        indent=2 + indent,
        nThreads=nThreads)

    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}Complete eye diagram')

    self.calculated = True
    return self.measures

  def __calculateLevels(self, plot: bool = True, nThreads: int = 1) -> None:
    '''!@brief Calculate high/low levels and crossing thresholds

    Assumes signal has two levels

    @param plot True will create plots during calculation (histograms and such). False will not
    @param nThreads Specify number of threads to use
    '''

    if self.manualYLevels is None or plot:
      if nThreads <= 1:
        output = [
            _runnerCalculateLevels(
                self.waveforms[i][1],
                self.histNMax) for i in range(
                self.waveforms.shape[0])]
      else:
        with Pool(nThreads) as p:
          results = [p.apply_async(
            _runnerCalculateLevels,
            args=[self.waveforms[i][1], self.histNMax]) for i in range(self.waveforms.shape[0])]
          output = [p.get() for p in results]

      yMin = min([o['yMin'] for o in output])
      yMax = max([o['yMax'] for o in output])
      yZeroStdDev = np.sqrt(np.average([o['yZeroStdDev']**2 for o in output]))
      yOneStdDev = np.sqrt(np.average([o['yOneStdDev']**2 for o in output]))
    else:
      yZeroStdDev = 0
      yOneStdDev = 0

    if self.manualYLevels:
      self.yZero = self.manualYLevels[0]
      self.yOne = self.manualYLevels[1]
    else:
      # Use min/max to get worst case
      self.yZero = np.amax([o['yZero'] for o in output])
      self.yOne = np.amin([o['yOne'] for o in output])

    hys = 0.5 - self.hysteresis / 2
    self.thresholdRise = self.yOne * (1 - hys) + self.yZero * hys
    self.thresholdFall = self.yOne * hys + self.yZero * (1 - hys)
    self.thresholdHalf = (self.yOne + self.yZero) / 2

    # Check high level is not close to low level
    snr = (self.yOne - self.yZero) / (yZeroStdDev + yOneStdDev)
    self.lowSNR = snr < 2

    if plot:
      yRange = np.linspace(yMin, yMax, 1000)
      yZeroCurve = np.zeros(1000)
      yOneCurve = np.zeros(1000)
      for o in output:
        yZeroCurve += gaussianMix(yRange, o['yZeroComponents'])
        yOneCurve += gaussianMix(yRange, o['yOneComponents'])
      pyplot.plot(yRange, yZeroCurve / len(output))
      pyplot.plot(yRange, yOneCurve / len(output))
      pyplot.axvline(x=self.yZero, color='g')
      pyplot.axvline(x=self.yOne, color='g')
      pyplot.axvline(x=self.thresholdRise, color='r')
      pyplot.axvline(x=self.thresholdHalf, color='k')
      pyplot.axvline(x=self.thresholdFall, color='r')
      pyplot.axvline(x=(self.yZero + yZeroStdDev), color='y')
      pyplot.axvline(x=(self.yZero - yZeroStdDev), color='y')
      pyplot.axvline(x=(self.yOne + yOneStdDev), color='y')
      pyplot.axvline(x=(self.yOne - yOneStdDev), color='y')

      ax = pyplot.gca()

      def tickFormatterY(y, _):
        return metricPrefix(y, self.yUnit)
      formatterY = FuncFormatter(tickFormatterY)
      ax.xaxis.set_major_formatter(formatterY)

      pyplot.xlabel('Vertical scale')
      pyplot.ylabel('Density')
      pyplot.title('Vertical levels')

      pyplot.show()

  def __calculateClock(self, plot: bool = True, nThreads: int = 1) -> None:
    '''!@brief Calculate clock for each waveform using a pll

    Assumes shortest period is a single bit

    @param plot True will create plots during calculation (histograms and such). False will not
    @param nThreads Specify number of threads to use
    '''
    if self.lowSNR:
      if self.manualTBit is None:
        self.tBit = 10e-9
      else:
        self.tBit = self.manualTBit
      self.tBitStdDev = 0

      self.clockEdges = []
      for i in range(self.waveforms.shape[0]):
        t = self.waveforms[i][0][0] + self.tBit
        clockEdges = np.arange(t, self.waveforms[i][0][-1], self.tBit)
        self.clockEdges.append(clockEdges.tolist())

      self.bitDistribution = ([0], [1])
      return

    # Get waveform edges using hysteresis and 50% interpolation
    if nThreads <= 1:
      self.waveformEdges = [
          getEdgesNumpy(self.waveforms[i],
                        self.thresholdRise,
                        self.thresholdHalf,
                        self.thresholdFall) for i in range(
              self.waveforms.shape[0])]
    else:
      with Pool(nThreads) as p:
        results = [p.apply_async(
            getEdgesNumpy,
            args=[self.waveforms[i],
                  self.thresholdRise,
                  self.thresholdHalf,
                  self.thresholdFall]) for i in range(
            self.waveforms.shape[0])]
        self.waveformEdges = [p.get() for p in results]

    amplitudeThreshold = 0.1
    if self.manualTBit is None:
      # Need to get an estimate for tBit to initialize pll
      periods = []
      for edgesRise, edgesFall in self.waveformEdges:
        for i in range(1, len(edgesRise)):
          duration = edgesRise[i] - edgesRise[i - 1]
          periods.append(duration)

        for i in range(1, len(edgesFall)):
          duration = edgesFall[i] - edgesFall[i - 1]
          periods.append(duration)
      periods = histogramDownsample(periods, self.histNMax)

      # Step 1 Find rough period from just the smallest pulses
      components = fitGaussianMix(periods, nMax=self.nBitsMax)
      components = [c for c in components if c[0] > amplitudeThreshold]
      components = sorted(components, key=lambda c: c[1])
      if len(components) > 1:
        bitPeriodsShort = []
        weights = []
        for c in components:
          if c[1] < (2.5 * components[0][1] / 2):
            bitPeriodsShort.append(c[1])
            weights.append(c[0])
        tBit = np.average(bitPeriodsShort, weights=weights) / 2
      else:
        tBit = components[0][1] / 2
    else:
      tBit = self.manualTBit

    # Step 1.5 Remove glitches
    if nThreads <= 1:
      self.waveformEdges = [
          _runnerCleanEdges(
              self.waveformEdges[i],
              tBit,
              self.nBitsMin) for i in range(
              self.waveforms.shape[0])]
    else:
      with Pool(nThreads) as p:
        results = [
            p.apply_async(
                _runnerCleanEdges,
                args=[
                    self.waveformEdges[i],
                    tBit,
                    self.nBitsMin]) for i in range(
                self.waveforms.shape[0])]
        self.waveformEdges = [p.get() for p in results]

    # Step 2 do clock recovery, do both rising and falling edges leading
    if nThreads <= 1:
      outputFalse = [
          _runnerClockRecovery(self.waveformEdges[i],
                               self.tDelta,
                               tBit,
                               self.pllBandwidth,
                               False, plot) for i in range(
              self.waveforms.shape[0])]
      outputTrue = [
          _runnerClockRecovery(self.waveformEdges[i],
                               self.tDelta,
                               tBit,
                               self.pllBandwidth,
                               True, plot) for i in range(
              self.waveforms.shape[0])]
    else:
      with Pool(nThreads) as p:
        resultsFalse = [p.apply_async(
            _runnerClockRecovery,
            args=[self.waveformEdges[i],
                  self.tDelta,
                  tBit,
                  self.pllBandwidth,
                  False, plot]) for i in range(
            self.waveforms.shape[0])]
        resultsTrue = [p.apply_async(
            _runnerClockRecovery,
            args=[self.waveformEdges[i],
                  self.tDelta,
                  tBit,
                  self.pllBandwidth,
                  True, plot]) for i in range(
            self.waveforms.shape[0])]
        outputFalse = [p.get() for p in resultsFalse]
        outputTrue = [p.get() for p in resultsTrue]

    periodsFalse = []
    for o in outputFalse:
      periodsFalse.extend(o[1])
    periodsTrue = []
    for o in outputTrue:
      periodsTrue.extend(o[1])
    # Choose whichever one is more consistent
    if np.std(periodsFalse) < np.std(periodsTrue):
      output = outputFalse
    else:
      output = outputTrue

    self.clockEdges = []
    periods = []
    tie = []
    nBits = []
    phaseErrors = []
    offsetErrors = []
    delayErrors = []
    phases = []
    offsets = []
    delays = []
    for o in output:
      self.clockEdges.append(o[0])
      periods.extend(o[1])
      tie.extend(o[2])
      nBits.extend(o[3])
      if plot:
        phases.extend(o[4])
        phaseErrors.extend(o[5])
        offsets.extend(o[6])
        offsetErrors.extend(o[7])
        delays.extend(o[8])
        delayErrors.extend(o[9])

    self.bitDistribution = binExact(nBits)

    self.tBit = np.average(periods)
    self.tBitStdDev = np.std(periods)

    if plot:
      def tickFormatterX(x, _):
        return metricPrefix(x, self.tUnit)
      formatterX = FuncFormatter(tickFormatterX)

      periods = histogramDownsample(periods, self.histNMax)
      components = fitGaussianMix(periods, nMax=self.nBitsMax)

      yRange = np.linspace(min(periods), max(periods), 1000)
      _, subplots = pyplot.subplots(6, 1, sharex=False)
      subplots[0].set_title('Bit period')
      subplots[0].hist(periods, 50, density=True, color='b', alpha=0.5)
      subplots[0].plot(yRange, gaussianMix(yRange, components), color='r')
      subplots[0].axvline(x=self.tBit, color='g')
      subplots[0].axvline(x=(self.tBit + self.tBitStdDev), color='y')
      subplots[0].axvline(x=(self.tBit - self.tBitStdDev), color='y')

      subplots[1].set_title('Time Interval Error')
      subplots[1].hist(tie, 100, density=True, color='b', alpha=0.5)

      subplots[2].set_title('PLL Errors')
      subplots[2].plot(phaseErrors, color='r')
      subplots[2].plot(offsetErrors, color='g')
      subplots[2].plot(delayErrors, color='b')

      subplots[3].set_title('PLL Phases')
      subplots[3].plot(phases, color='r')

      subplots[4].set_title('PLL Offsets')
      subplots[4].plot(offsets, color='g')

      subplots[3].set_title('PLL Delays')
      subplots[5].plot(delays, color='b')

      subplots[0].xaxis.set_major_formatter(formatterX)
      subplots[1].xaxis.set_major_formatter(formatterX)

      subplots[-1].set_xlabel('Time')
      subplots[0].set_ylabel('Density')
      subplots[1].set_ylabel('Density')

      pyplot.show()

  def __extractBitCenters(self, plot: bool = True, nThreads: int = 1) -> None:
    '''!@brief Extract center positions of bits

    @param plot True will create plots during calculation (histograms and such). False will not
    @param nThreads Specify number of threads to use
    '''
    if nThreads <= 1:
      output = [
          _runnerBitExtract(
              self.clockEdges[i],
              self.waveforms[i][0][0],
              self.tDelta) for i in range(self.waveforms.shape[0])]
    else:
      with Pool(nThreads) as p:
        results = [p.apply_async(
          _runnerBitExtract,
          args=[
              self.clockEdges[i],
              self.waveforms[i][0][0],
              self.tDelta]) for i in range(self.waveforms.shape[0])]
        output = [p.get() for p in results]

    self.bitCentersT = [o[0] for o in output]
    self.bitCentersY = [o[1] for o in output]

    self.nBits = 0
    for o in output:
      self.nBits += len(o[0])

    if plot:
      def tickFormatterX(x, _):
        return metricPrefix(x, self.tUnit)
      formatterX = FuncFormatter(tickFormatterX)

      def tickFormatterY(y, _):
        return metricPrefix(y, self.yUnit)
      formatterY = FuncFormatter(tickFormatterY)

      hw = int((self.tBit / self.tDelta) + 0.5)
      n = hw * 2 + 1

      waveformIndex = 0

      for i in range(min(20, len(self.bitCentersT[waveformIndex]))):
        cT = self.bitCentersT[waveformIndex][i]
        cY = self.bitCentersY[waveformIndex][i]
        plotX = np.linspace(cT - hw * self.tDelta, cT + hw * self.tDelta, n)
        plotY = self.waveforms[waveformIndex][1][cY - hw: cY + hw + 1]
        pyplot.plot(plotX, plotY)

      pyplot.axvline(x=(-self.tBit / 2), color='g')
      pyplot.axvline(x=0, color='r')
      pyplot.axvline(x=(self.tBit / 2), color='r')
      pyplot.axhline(y=self.yZero, color='g')
      pyplot.axhline(y=self.thresholdFall, color='r')
      pyplot.axhline(y=self.thresholdHalf, color='k')
      pyplot.axhline(y=self.thresholdRise, color='r')
      pyplot.axhline(y=self.yOne, color='g')

      ax = pyplot.gca()
      ax.xaxis.set_major_formatter(formatterX)
      ax.yaxis.set_major_formatter(formatterY)

      pyplot.xlabel('Time')
      pyplot.ylabel('Vertical')
      pyplot.title('Select Bit Sequence')

      pyplot.show()

  def __measureWaveform(self, plot: bool = True,
                        printProgress: bool = True, indent: int = 0, nThreads: int = 1) -> None:
    '''!@brief Extract center positions of bits

    @param plot True will create plots during calculation (histograms and such). False will not
    @param printProgress True will print progress statements. False will not
    @param indent Base indentation for progress statements
    @param nThreads Specify number of threads to use
    '''
    self.measures = {}
    m = self.measures

    start = datetime.datetime.now()
    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}Starting measurement')

    nBits = 0
    for b in self.bitCentersT:
      nBits += len(b)
    m['nBits'] = nBits

    m['maxStateLength'] = max(self.bitDistribution[0])
    m['averageStateLength'] = 0
    m['longStateP'] = 0
    nStates = 0
    for length, count in zip(*self.bitDistribution):
      m['averageStateLength'] += length * count
      if length != 1:
        m['longStateP'] += count
      nStates += count
    m['averageStateLength'] = m['averageStateLength'] / nStates
    m['longStateP'] = 100 * m['longStateP'] / nStates

    if printProgress:
      print(
        f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Measuring \'0\', \'0.5\', \'1\' values')

    if nThreads <= 1:
      output = [
          _runnerCollectValuesY(
              self.waveforms[i][1],
              self.bitCentersT[i],
              self.bitCentersY[i],
              self.tDelta,
              self.tBit,
              self.thresholdHalf,
              self.resample) for i in range(self.waveforms.shape[0])]
    else:
      with Pool(nThreads) as p:
        results = [p.apply_async(
          _runnerCollectValuesY,
          args=[
              self.waveforms[i][1],
              self.bitCentersT[i],
              self.bitCentersY[i],
              self.tDelta,
              self.tBit,
              self.thresholdHalf,
              self.resample]) for i in range(self.waveforms.shape[0])]
        output = [p.get() for p in results]

    valuesY = {
      'zero': [],
      'cross': [],
      'one': []
    }

    for o in output:
      for k, v in o.items():
        valuesY[k].extend(v)
    output = None

    if printProgress:
      print(
        f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Computing \'0\', \'0.5\', \'1\' statistics')

    if self.method == 'average':
      m['yZero'] = np.average(valuesY['zero'])
      m['yCross'] = np.average(valuesY['cross'])
      m['yOne'] = np.average(valuesY['one'])
    m['yZeroStdDev'] = np.std(valuesY['zero'])
    m['yCrossStdDev'] = np.std(valuesY['cross'])
    m['yOneStdDev'] = np.std(valuesY['one'])

    if self.method == 'peak' or plot:
      valuesY['zero'] = histogramDownsample(valuesY['zero'], self.histNMax)
      componentsZero = fitGaussianMix(valuesY['zero'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}yZero has {len(componentsZero)} modes')

      valuesY['cross'] = histogramDownsample(valuesY['cross'], self.histNMax)
      componentsCross = fitGaussianMix(valuesY['cross'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}yCross has {len(componentsCross)} modes')

      valuesY['one'] = histogramDownsample(valuesY['one'], self.histNMax)
      componentsOne = fitGaussianMix(valuesY['one'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}yOne has {len(componentsOne)} modes')

    if self.method == 'peak':
      m['yZero'] = gaussianMixCenter(componentsZero)
      m['yCross'] = gaussianMixCenter(componentsCross)
      m['yOne'] = gaussianMixCenter(componentsOne)
    elif self.method == 'average':
      # Already did before histogram downsampling
      pass
    else:
      raise Exception(f'Unrecognized measure method: {self.method}')

    m['eyeAmplitude'] = m['yOne'] - m['yZero']
    m['eyeAmplitudeStdDev'] = np.sqrt(m['yOneStdDev']**2 + m['yZeroStdDev']**2)

    m['snr'] = (m['eyeAmplitude']) / (m['yOneStdDev'] + m['yZeroStdDev'])

    m['eyeHeight'] = (m['yOne'] - 3 * m['yOneStdDev']) - \
        (m['yZero'] + 3 * m['yZeroStdDev'])
    m['eyeHeightP'] = 100 * m['eyeHeight'] / (m['eyeAmplitude'])

    m['yCrossP'] = 100 * (m['yCross'] - m['yZero']
                          ) / (m['eyeAmplitude'])
    m['yCrossPStdDev'] = 100 * m['yCrossStdDev'] / (m['eyeAmplitude'])

    if plot:
      def tickFormatterY(y, _):
        return metricPrefix(y, self.yUnit)
      formatterY = FuncFormatter(tickFormatterY)

      yRange = np.linspace(
          np.min(
              valuesY['zero']), np.max(
              valuesY['one']), 1000)
      _, subplots = pyplot.subplots(3, 1, sharex=False)
      subplots[0].set_title('Zero Level')
      subplots[0].hist(valuesY['zero'], 50, density=True, color='b', alpha=0.5)
      subplots[0].plot(yRange, gaussianMix(yRange, componentsZero), color='r')
      subplots[0].axvline(x=m["yZero"], color='g')
      subplots[0].axvline(x=(m["yZero"] + m["yZeroStdDev"]), color='y')
      subplots[0].axvline(x=(m["yZero"] - m["yZeroStdDev"]), color='y')

      subplots[1].set_title('Crossing Level')
      subplots[1].hist(
          valuesY['cross'],
          50,
          density=True,
          color='b',
          alpha=0.5)
      subplots[1].plot(yRange, gaussianMix(yRange, componentsCross), color='r')
      subplots[1].axvline(x=m["yCross"], color='g')
      subplots[1].axvline(x=(m["yCross"] - m["yCrossStdDev"]), color='y')
      subplots[1].axvline(x=(m["yCross"] + m["yCrossStdDev"]), color='y')

      subplots[2].set_title('One Level')
      subplots[2].hist(valuesY['one'], 50, density=True, color='b', alpha=0.5)
      subplots[2].plot(yRange, gaussianMix(yRange, componentsOne), color='r')
      subplots[2].axvline(x=m["yOne"], color='g')
      subplots[2].axvline(x=(m["yOne"] + m["yOneStdDev"]), color='y')
      subplots[2].axvline(x=(m["yOne"] - m["yOneStdDev"]), color='y')

      subplots[0].xaxis.set_major_formatter(formatterY)
      subplots[1].xaxis.set_major_formatter(formatterY)
      subplots[2].xaxis.set_major_formatter(formatterY)

      subplots[-1].set_xlabel('Vertical scale')
      subplots[0].set_ylabel('Density')
      subplots[1].set_ylabel('Density')
      subplots[2].set_ylabel('Density')

      pyplot.show()
    valuesY = None

    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Measuring edges values')

    # Horizontal domain
    m['tBit'] = self.tBit
    m['tBitStdDev'] = self.tBitStdDev

    m['fBit'] = 1 / self.tBit
    m['fBitStdDev'] = m['tBitStdDev'] * m['fBit'] / m['tBit']

    if self.manualYLevels is None:
      self.yZero = m['yZero']
      self.yOne = m['yOne']

      hys = 0.5 - self.hysteresis
      self.thresholdRise = self.yOne * (1 - hys) + self.yZero * hys
      self.thresholdFall = self.yOne * hys + self.yZero * (1 - hys)
      self.thresholdHalf = (self.yOne + self.yZero) / 2

    try:
      if nThreads <= 1:
        output = [
            _runnerCollectValuesT(
                self.waveforms[i][1],
                self.bitCentersT[i],
                self.bitCentersY[i],
                self.tDelta,
                self.tBit,
                self.yZero,
                m['yCross'],
                self.yOne,
                self.resample,
                i) for i in range(self.waveforms.shape[0])]
      else:
        with Pool(nThreads) as p:
          results = [p.apply_async(
            _runnerCollectValuesT,
            args=[
              self.waveforms[i][1],
                self.bitCentersT[i],
                self.bitCentersY[i],
                self.tDelta,
                self.tBit,
                self.yZero,
                m['yCross'],
                self.yOne,
                self.resample,
                i]) for i in range(self.waveforms.shape[0])]
          output = [p.get() for p in results]
    except WaveformException as e:
      if not plot:
        raise e
      print(e)
      hw = int((self.tBit / self.tDelta) + 0.5)
      i = self.bitCentersY[e.waveformIndex][e.bitIndex]
      t = self.waveforms[e.waveformIndex, 0, i - hw: i + hw + 1]
      y = self.waveforms[e.waveformIndex, 1, i - hw: i + hw + 1]
      y = (y - self.yZero) / (self.yOne - self.yZero)
      pyplot.plot(t, y)
      pyplot.axhline(y=0, color='r')
      pyplot.axhline(y=((m['yCross'] - self.yZero) /
                     (self.yOne - self.yZero)), color='b')
      pyplot.axhline(y=1, color='r')
      pyplot.axhline(y=0.2, color='g')
      pyplot.axhline(y=0.8, color='g')
      pyplot.show()
      import sys
      sys.exit(1)

    valuesT = {
        'rise20': [],
        'rise50': [],
        'rise80': [],
        'rise': [],
        'fall20': [],
        'fall50': [],
        'fall80': [],
        'fall': [],
        'cross': []
    }

    for o in output:
      for k, v in o.items():
        valuesT[k].extend(v)
    output = None

    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Computing edges statistics')

    crossNan = np.count_nonzero(np.isnan(valuesT['cross']))
    if crossNan > 0:
      print(
        f'{"":{indent}}{Fore.RED}{crossNan}/{len(valuesT["cross"])}={crossNan/len(valuesT["cross"]) *100:.2f}% bits did not reach yCross')
      valuesT['cross'] = np.array(valuesT['cross'])
      valuesT['cross'] = valuesT['cross'][~np.isnan(valuesT['cross'])].tolist()

    edgeNan = np.count_nonzero(np.isnan(valuesT['rise']))
    edgeNan += np.count_nonzero(np.isnan(valuesT['fall']))
    if edgeNan > 0:
      edges = len(valuesT['rise'])
      edges += len(valuesT['fall'])
      print(
        f'{"":{indent}}{Fore.RED}{edgeNan}/{edges}={edgeNan/edges *100:.2f}% bits did not reach 0.2UA and/or 0.8UA')
      valuesT['rise'] = np.array(valuesT['rise'])
      valuesT['rise'] = valuesT['rise'][~np.isnan(valuesT['rise'])].tolist()
      valuesT['fall'] = np.array(valuesT['fall'])
      valuesT['fall'] = valuesT['fall'][~np.isnan(valuesT['fall'])].tolist()
    valuesT['rise20'] = np.array(valuesT['rise20'])
    valuesT['rise20'] = valuesT['rise20'][~np.isnan(
      valuesT['rise20'])].tolist()
    valuesT['rise50'] = np.array(valuesT['rise50'])
    valuesT['rise50'] = valuesT['rise50'][~np.isnan(
      valuesT['rise50'])].tolist()
    valuesT['rise80'] = np.array(valuesT['rise80'])
    valuesT['rise80'] = valuesT['rise80'][~np.isnan(
      valuesT['rise80'])].tolist()
    valuesT['fall20'] = np.array(valuesT['fall20'])
    valuesT['fall20'] = valuesT['fall20'][~np.isnan(
      valuesT['fall20'])].tolist()
    valuesT['fall50'] = np.array(valuesT['fall50'])
    valuesT['fall50'] = valuesT['fall50'][~np.isnan(
      valuesT['fall50'])].tolist()
    valuesT['fall80'] = np.array(valuesT['fall80'])
    valuesT['fall80'] = valuesT['fall80'][~np.isnan(
      valuesT['fall80'])].tolist()

    if self.method == 'average':
      m['tRise20'] = np.average(valuesT['rise20'])
      m['tRise50'] = np.average(valuesT['rise50'])
      m['tRise80'] = np.average(valuesT['rise80'])
      m['tRise'] = np.average(valuesT['rise'])
      m['tFall20'] = np.average(valuesT['fall20'])
      m['tFall50'] = np.average(valuesT['fall50'])
      m['tFall80'] = np.average(valuesT['fall80'])
      m['tFall'] = np.average(valuesT['fall'])
      m['tCross'] = np.average(valuesT['cross'])
    m['tRise20StdDev'] = np.std(valuesT['rise20'])
    m['tRise50StdDev'] = np.std(valuesT['rise50'])
    m['tRise80StdDev'] = np.std(valuesT['rise80'])
    m['tRiseStdDev'] = np.std(valuesT['rise'])
    m['tFall20StdDev'] = np.std(valuesT['fall20'])
    m['tFall50StdDev'] = np.std(valuesT['fall50'])
    m['tFall80StdDev'] = np.std(valuesT['fall80'])
    m['tFallStdDev'] = np.std(valuesT['fall'])
    m['tCrossStdDev'] = np.std(valuesT['cross'])

    spanThreshold = 0.25
    span = (np.amax(valuesT['rise20']) -
            np.amin(valuesT['rise20'])) / self.tBit
    if span > spanThreshold:
      print(f'{"":{indent}}{Fore.RED}tRise20 spans {span:.2f}b > {spanThreshold:.2f}b')
    span = (np.amax(valuesT['rise50']) -
            np.amin(valuesT['rise50'])) / self.tBit
    if span > spanThreshold:
      print(f'{"":{indent}}{Fore.RED}tRise50 spans {span:.2f}b > {spanThreshold:.2f}b')
    span = (np.amax(valuesT['rise80']) -
            np.amin(valuesT['rise80'])) / self.tBit
    if span > spanThreshold:
      print(f'{"":{indent}}{Fore.RED}tRise80 spans {span:.2f}b > {spanThreshold:.2f}b')
    span = (np.amax(valuesT['rise']) - np.amin(valuesT['rise'])) / self.tBit
    if span > spanThreshold:
      print(f'{"":{indent}}{Fore.RED}tRise spans {span:.2f}b > {spanThreshold:.2f}b')
    span = (np.amax(valuesT['fall20']) -
            np.amin(valuesT['fall20'])) / self.tBit
    if span > spanThreshold:
      print(f'{"":{indent}}{Fore.RED}tFall20 spans {span:.2f}b > {spanThreshold:.2f}b')
    span = (np.amax(valuesT['fall50']) -
            np.amin(valuesT['fall50'])) / self.tBit
    if span > spanThreshold:
      print(f'{"":{indent}}{Fore.RED}tFall50 spans {span:.2f}b > {spanThreshold:.2f}b')
    span = (np.amax(valuesT['fall80']) -
            np.amin(valuesT['fall80'])) / self.tBit
    if span > spanThreshold:
      print(f'{"":{indent}}{Fore.RED}tFall80 spans {span:.2f}b > {spanThreshold:.2f}b')
    span = (np.amax(valuesT['fall']) - np.amin(valuesT['fall'])) / self.tBit
    if span > spanThreshold:
      print(f'{"":{indent}}{Fore.RED}tFall spans {span:.2f}b > {spanThreshold:.2f}b')
    span = (np.amax(valuesT['cross']) - np.amin(valuesT['cross'])) / self.tBit
    if span > spanThreshold and (m['yCrossP'] > 20 and m['yCrossP'] < 80):
      print(f'{"":{indent}}{Fore.RED}tCross spans {span:.2f}b > {spanThreshold:.2f}b')

    if self.method == 'peak' or plot:
      valuesT['rise20'] = histogramDownsample(valuesT['rise20'], self.histNMax)
      componentsRise20 = fitGaussianMix(valuesT['rise20'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}tRise20 has {len(componentsRise20)} modes')

      valuesT['rise50'] = histogramDownsample(valuesT['rise50'], self.histNMax)
      componentsRise50 = fitGaussianMix(valuesT['rise50'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}tRise50 has {len(componentsRise50)} modes')

      valuesT['rise80'] = histogramDownsample(valuesT['rise80'], self.histNMax)
      componentsRise80 = fitGaussianMix(valuesT['rise80'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}tRise80 has {len(componentsRise80)} modes')

      valuesT['rise'] = histogramDownsample(valuesT['rise'], self.histNMax)
      componentsRise = fitGaussianMix(valuesT['rise'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}tRise has {len(componentsRise)} modes')

      valuesT['fall20'] = histogramDownsample(valuesT['fall20'], self.histNMax)
      componentsFall20 = fitGaussianMix(valuesT['fall20'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}tFall20 has {len(componentsFall20)} modes')

      valuesT['fall50'] = histogramDownsample(valuesT['fall50'], self.histNMax)
      componentsFall50 = fitGaussianMix(valuesT['fall50'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}tFall50 has {len(componentsFall50)} modes')

      valuesT['fall80'] = histogramDownsample(valuesT['fall80'], self.histNMax)
      componentsFall80 = fitGaussianMix(valuesT['fall80'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}tFall80 has {len(componentsFall80)} modes')

      valuesT['fall'] = histogramDownsample(valuesT['fall'], self.histNMax)
      componentsFall = fitGaussianMix(valuesT['fall'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}tFall has {len(componentsFall)} modes')

      valuesT['cross'] = histogramDownsample(valuesT['cross'], self.histNMax)
      componentsTCross = fitGaussianMix(valuesT['cross'], nMax=3)
      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}tCross has {len(componentsTCross)} modes')

    if self.method == 'peak':
      m["tRise20"] = gaussianMixCenter(componentsRise20)
      m["tRise50"] = gaussianMixCenter(componentsRise50)
      m["tRise80"] = gaussianMixCenter(componentsRise80)
      m["tRise"] = gaussianMixCenter(componentsRise)
      m["tFall20"] = gaussianMixCenter(componentsFall20)
      m["tFall50"] = gaussianMixCenter(componentsFall50)
      m["tFall80"] = gaussianMixCenter(componentsFall80)
      m["tFall"] = gaussianMixCenter(componentsFall)
      m["tCross"] = gaussianMixCenter(componentsTCross)
    elif self.method == 'average':
      # Already did before histogram downsampling
      pass
    else:
      raise Exception(f'Unrecognized measure method: {self.method}')

    m["tLow"] = (m["tRise50"] + m["tBit"]) - m["tFall50"]
    m["tLowStdDev"] = np.sqrt(
        m["tRise50StdDev"]**2 +
        m["tBitStdDev"]**2 +
        m["tFall50StdDev"]**2)
    m["tHigh"] = (m["tFall50"] + m["tBit"]) - m["tRise50"]
    m["tHighStdDev"] = np.sqrt(
        m["tFall50StdDev"]**2 +
        m["tBitStdDev"]**2 +
        m["tRise50StdDev"]**2)

    m["tJitterPP"] = np.max(valuesT['cross']) - np.min(valuesT['cross'])
    m["tJitterRMS"] = m["tCrossStdDev"]

    m["eyeWidth"] = (m["tBit"] + m["tCross"] - 3 *
                     m["tCrossStdDev"]) - (m["tCross"] + 3 * m["tCrossStdDev"])
    m["eyeWidthP"] = 100 * m["eyeWidth"] / m["tBit"]

    edgeDifference = (m['tHigh'] - m['tLow']) * 0.5
    edgeDifferenceStdDev = np.sqrt(
        m['tHighStdDev']**2 + m['tLowStdDev']**2) * 0.5
    m["dcd"] = 100 * edgeDifference / m["tBit"]
    m["dcdStdDev"] = abs(m["dcd"]) * np.sqrt((edgeDifferenceStdDev /
                                              edgeDifference) ** 2 + (m["tBitStdDev"] / m["tBit"]) ** 2)

    if plot:
      def tickFormatterX(x, _):
        return metricPrefix(x, self.tUnit)
      formatterX = FuncFormatter(tickFormatterX)

      def tickFormatterY(y, _):
        return metricPrefix(y, self.yUnit)
      formatterY = FuncFormatter(tickFormatterY)

      xRange = np.linspace(-m["tBit"] / 2, m["tBit"] / 2, 1000)
      _, subplots = pyplot.subplots(4, 1, sharex=True)
      subplots[0].set_title('Rising Edges')
      subplots[0].hist(
          valuesT['rise20'],
          50,
          density=True,
          color='k',
          alpha=0.5)
      subplots[0].hist(
          valuesT['rise50'],
          50,
          density=True,
          color='b',
          alpha=0.5)
      subplots[0].hist(
          valuesT['rise80'],
          50,
          density=True,
          color='m',
          alpha=0.5)
      subplots[0].plot(
          xRange,
          gaussianMix(
              xRange,
              componentsRise20),
          color='k')
      subplots[0].plot(
          xRange,
          gaussianMix(
              xRange,
              componentsRise50),
          color='b')
      subplots[0].plot(
          xRange,
          gaussianMix(
              xRange,
              componentsRise80),
          color='m')
      subplots[0].axvline(x=m["tRise20"], color='g')
      subplots[0].axvline(x=m["tRise50"], color='g')
      subplots[0].axvline(x=m["tRise80"], color='g')
      subplots[0].axvline(x=(m["tRise20"] + m["tRise20StdDev"]), color='y')
      subplots[0].axvline(x=(m["tRise20"] - m["tRise20StdDev"]), color='y')
      subplots[0].axvline(x=(m["tRise50"] + m["tRise50StdDev"]), color='y')
      subplots[0].axvline(x=(m["tRise50"] - m["tRise50StdDev"]), color='y')
      subplots[0].axvline(x=(m["tRise80"] + m["tRise80StdDev"]), color='y')
      subplots[0].axvline(x=(m["tRise80"] - m["tRise80StdDev"]), color='y')

      subplots[1].set_title('Falling Edges')
      subplots[1].hist(
          valuesT['fall20'],
          50,
          density=True,
          color='k',
          alpha=0.5)
      subplots[1].hist(
          valuesT['fall50'],
          50,
          density=True,
          color='b',
          alpha=0.5)
      subplots[1].hist(
          valuesT['fall80'],
          50,
          density=True,
          color='m',
          alpha=0.5)
      subplots[1].plot(
          xRange,
          gaussianMix(
              xRange,
              componentsFall20),
          color='k')
      subplots[1].plot(
          xRange,
          gaussianMix(
              xRange,
              componentsFall50),
          color='b')
      subplots[1].plot(
          xRange,
          gaussianMix(
              xRange,
              componentsFall80),
          color='m')
      subplots[1].axvline(x=m["tFall20"], color='g')
      subplots[1].axvline(x=m["tFall50"], color='g')
      subplots[1].axvline(x=m["tFall80"], color='g')
      subplots[1].axvline(x=(m["tFall20"] + m["tFall20StdDev"]), color='y')
      subplots[1].axvline(x=(m["tFall20"] - m["tFall20StdDev"]), color='y')
      subplots[1].axvline(x=(m["tFall50"] + m["tFall50StdDev"]), color='y')
      subplots[1].axvline(x=(m["tFall50"] - m["tFall50StdDev"]), color='y')
      subplots[1].axvline(x=(m["tFall80"] + m["tFall80StdDev"]), color='y')
      subplots[1].axvline(x=(m["tFall80"] - m["tFall80StdDev"]), color='y')

      subplots[2].set_title('Interpolated Waveforms')
      x01 = list(np.linspace(-self.tBit / 2, 2 *
                 m["tRise20"] - m["tRise50"], 10))
      x01.extend([m["tRise20"], m["tRise50"], m["tRise80"]])
      x01.extend(list(
          np.linspace(2 * m["tRise80"] - m["tRise50"], self.tBit / 2, 10)))
      y01 = [m["yZero"]] * 10
      y01.extend([m["yZero"] + 0.2 * m["eyeAmplitude"],
                  m["yZero"] + 0.5 * m["eyeAmplitude"],
                  m["yZero"] + 0.8 * m["eyeAmplitude"]])
      y01.extend([m["yOne"]] * 10)
      # f01 = interp1d(x01, y01, kind='cubic')
      subplots[2].plot(x01, y01, color='b')
      # subplots[2].plot(xRange, f01(xRange), color='b', linestyle=':')
      x10 = list(np.linspace(-self.tBit / 2, 2 *
                 m["tFall80"] - m["tFall50"], 10))
      x10.extend([m["tFall80"], m["tFall50"], m["tFall20"]])
      x10.extend(list(
          np.linspace(2 * m["tFall20"] - m["tFall50"], self.tBit / 2, 10)))
      y10 = [m["yOne"]] * 10
      y10.extend([m["yZero"] + 0.8 * m["eyeAmplitude"],
                  m["yZero"] + 0.5 * m["eyeAmplitude"],
                  m["yZero"] + 0.2 * m["eyeAmplitude"]])
      y10.extend([m["yZero"]] * 10)
      # f10 = interp1d(x10, y10, kind='cubic')
      subplots[2].plot(x10, y10, color='g')
      # subplots[2].plot(xRange, f10(xRange), color='g', linestyle=':')

      subplots[3].set_title('Crossing Point')
      subplots[3].hist(
          valuesT['cross'],
          50,
          density=True,
          color='k',
          alpha=0.5)
      subplots[3].plot(
          xRange,
          gaussianMix(
              xRange,
              componentsTCross),
          color='k')
      subplots[3].axvline(x=m["tCross"], color='g')
      subplots[3].axvline(x=(m["tCross"] + m["tCrossStdDev"]), color='y')
      subplots[3].axvline(x=(m["tCross"] - m["tCrossStdDev"]), color='y')

      subplots[0].xaxis.set_major_formatter(formatterX)
      subplots[1].xaxis.set_major_formatter(formatterX)
      subplots[2].xaxis.set_major_formatter(formatterX)
      subplots[2].yaxis.set_major_formatter(formatterY)
      subplots[3].xaxis.set_major_formatter(formatterX)

      subplots[-1].set_xlabel('Time')
      subplots[0].set_ylabel('Counts')
      subplots[1].set_ylabel('Counts')
      subplots[2].set_ylabel('Vertical Scale')
      subplots[3].set_ylabel('Counts')

      pyplot.show()
    valuesT = None

    if self.mask:
      if printProgress:
        print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Checking mask for hits')

      if nThreads <= 1:
        output = [
            _runnerCollectMaskHits(
                self.waveforms[i][1],
                self.bitCentersT[i],
                self.bitCentersY[i],
                self.tDelta,
                self.tBit,
                self.yZero,
                self.yOne,
                self.mask,
                self.resample) for i in range(self.waveforms.shape[0])]
      else:
        with Pool(nThreads) as p:
          results = [p.apply_async(
            _runnerCollectMaskHits,
            args=[
              self.waveforms[i][1],
              self.bitCentersT[i],
              self.bitCentersY[i],
              self.tDelta,
              self.tBit,
              self.yZero,
              self.yOne,
              self.mask,
              self.resample]) for i in range(self.waveforms.shape[0])]
          output = [p.get() for p in results]

      self.offenders = []
      self.hits = []
      offenderCount = 0
      for o in output:
        self.offenders.append(o[0])
        offenderCount += len(o[0])
        self.hits.extend(o[1])
      output = None

      m['offenderCount'] = offenderCount
      m['ber'] = m['offenderCount'] / m['nBits']

      if printProgress:
        print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Finding mask margin')

      if nThreads <= 1:
        output = [
            _runnerAdjustMaskMargin(
                self.waveforms[i][1],
                self.bitCentersT[i],
                self.bitCentersY[i],
                self.tDelta,
                self.tBit,
                self.yZero,
                self.yOne,
                self.mask,
                self.resample,
                self.offenders[i]) for i in range(self.waveforms.shape[0])]
      else:
        with Pool(nThreads) as p:
          results = [p.apply_async(
            _runnerAdjustMaskMargin,
            args=[
              self.waveforms[i][1],
              self.bitCentersT[i],
              self.bitCentersY[i],
              self.tDelta,
              self.tBit,
              self.yZero,
              self.yOne,
              self.mask,
              self.resample,
              self.offenders[i]]) for i in range(self.waveforms.shape[0])]
          output = [p.get() for p in results]

      output = [(output[i], i) for i in range(len(output))]
      output = sorted(output, key=lambda o: o[0])
      self.maskMarginPair = output[0]
      output = None

      m['maskMargin'] = 100 * self.maskMarginPair[0][0]

      if plot:
        # Plot mask
        for path in self.mask.paths:
          x = [p[0] for p in path]
          y = [p[1] for p in path]

          pyplot.plot(x, y, linestyle='-', color='m')

        # Plot margin mask
        for path in self.mask.adjust(self.maskMarginPair[0][0]).paths:
          x = [p[0] for p in path]
          y = [p[1] for p in path]

          pyplot.plot(x, y, linestyle=':', color='m')

        # Plot subset of mask hits
        x = [p[0] for p in self.hits][:1000]
        y = [p[1] for p in self.hits][:1000]

        pyplot.scatter(x, y, marker='.', color='b')

        iBitWidth = int((self.tBit / self.tDelta) + 0.5)
        tBitWidthUI = (iBitWidth * self.tDelta / self.tBit)

        t0 = np.linspace(
            0.5 - tBitWidthUI,
            0.5 + tBitWidthUI,
            iBitWidth * 2 + 1)
        factor = int(np.ceil(self.resample / iBitWidth))
        if factor > 1:
          # Expand to 3 bits for better sinc interpolation at the edges
          iBitWidth = int((self.tBit / self.tDelta * 1.5) + 0.5)
          tBitWidthUI = (iBitWidth * self.tDelta / self.tBit)
          t0 = np.linspace(
            0.5 - tBitWidthUI,
            0.5 + tBitWidthUI,
            iBitWidth * 2 + 1)
          tNew = np.linspace(
              0.5 - tBitWidthUI,
              0.5 + tBitWidthUI,
              iBitWidth * 2 * factor + 1)

          T = t0[1] - t0[0]
          sincM = np.tile(tNew, (len(t0), 1)) - \
              np.tile(t0[:, np.newaxis], (1, len(tNew)))
          referenceSinc = np.sinc(sincM / T)

        # Plot subset of offenders
        for i in range(len(self.offenders)):
          waveformY = (self.waveforms[i][1] -
                       self.yZero) / (self.yOne - self.yZero)
          waveformY = waveformY.tolist()
          for o in self.offenders[i][:1]:
            cY = self.bitCentersY[i][o]
            cT = self.bitCentersT[i][o] / self.tBit

            if factor > 1:
              t = (tNew + cT).tolist()
              yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
              y = np.dot(yOriginal, referenceSinc).tolist()
            else:
              t = (t0 + cT).tolist()
              y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]

            pyplot.plot(t, y, linestyle='-', color='b')

        # Plot worst offender
        waveformIndex = self.maskMarginPair[1]
        i = self.maskMarginPair[0][1]
        waveformY = (self.waveforms[waveformIndex]
                     [1] - self.yZero) / (self.yOne - self.yZero)
        waveformY = waveformY.tolist()

        cY = self.bitCentersY[waveformIndex][i]
        cT = self.bitCentersT[waveformIndex][i] / self.tBit

        if factor > 1:
          t = (tNew + cT).tolist()
          yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
          y = np.dot(yOriginal, referenceSinc).tolist()
        else:
          t = (t0 + cT).tolist()
          y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]

        pyplot.plot(t, y, linestyle=':', color='k')

        # Make sure worst offender has hits plotted
        hits = getHits(t, y, self.mask.paths)
        x = [p[0] for p in hits]
        y = [p[1] for p in hits]
        pyplot.scatter(x, y, marker='.', color='b')

        pyplot.xlim(-0.5, 1.5)
        pyplot.ylim(-1, 2)
        pyplot.show()

    else:
      m['offenderCount'] = None
      m['ber'] = None
      m['maskMargin'] = None

    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}Complete measurement')

  def __generateImages(self, printProgress: bool = True, indent: int = 0,
                       nThreads: int = 1) -> None:
    '''!@brief Extract center positions of bits

    @param printProgress True will print progress statements. False will not
    @param indent Base indentation for progress statements
    @param nThreads Specify number of threads to use
    '''
    start = datetime.datetime.now()
    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}Starting image generation')

    heatMap = np.zeros((self.resolution, self.resolution), dtype=np.int32)

    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Layering bit waveforms')

    if nThreads <= 1:
      output = [
          _runnerGenerateHeatMap(
              self.waveforms[i][1],
              self.bitCentersT[i],
              self.bitCentersY[i],
              self.tDelta,
              self.tBit,
              self.yZero,
              self.yOne,
              self.resample,
              self.resolution) for i in range(self.waveforms.shape[0])]
    else:
      with Pool(nThreads) as p:
        results = [p.apply_async(
          _runnerGenerateHeatMap,
          args=[
              self.waveforms[i][1],
              self.bitCentersT[i],
              self.bitCentersY[i],
              self.tDelta,
              self.tBit,
              self.yZero,
              self.yOne,
              self.resample,
              self.resolution]) for i in range(self.waveforms.shape[0])]
        output = [p.get() for p in results]

    for o in output:
      heatMap += o

    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Transforming into heatmap')

    heatMap = heatMap.T[::-1, :]
    heatMap = heatMap.astype(np.float32)

    # Replace 0s with nan to be colored transparent
    heatMap[heatMap == 0] = np.nan

    # Normalize heatmap to 0 to 1
    heatMapMax = np.nanmax(heatMap)
    heatMap = heatMap / heatMapMax

    npImageClean = pyplot.cm.jet(heatMap)
    self.imageClean = Image.fromarray((255 * npImageClean).astype('uint8'))

    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Drawing grid')

    zero = int(
      ((0 - self.imageMin) / (self.imageMax - self.imageMin)) * self.resolution)
    half = int(((0.5 - self.imageMin) /
               (self.imageMax - self.imageMin)) * self.resolution)
    one = int(((1 - self.imageMin) / (self.imageMax - self.imageMin))
              * self.resolution)

    # Draw a grid for reference levels
    npImageGrid = np.zeros(npImageClean.shape, dtype=npImageClean.dtype)
    npImageGrid[:, :, 0: 2] = 1.0
    npImageGrid[zero, :, 3] = 1.0
    npImageGrid[one, :, 3] = 1.0
    for v in [0.2, 0.5, 0.8]:
      pos = int(((v - self.imageMin) / (self.imageMax - self.imageMin))
                * self.resolution)
      npImageGrid[pos, ::4, 3] = 1.0
    for v in [self.thresholdRise, self.thresholdFall]:
      v = (v - self.yZero) / (self.yOne - self.yZero)
      pos = int(((v - self.imageMin) / (self.imageMax - self.imageMin))
                * self.resolution)
      npImageGrid[pos, ::6, 3] = 1.0

    npImageGrid[:, zero, 3] = 1.0
    npImageGrid[::4, half, 3] = 1.0
    npImageGrid[:, one, 3] = 1.0
    for i in range(1, 8, 2):
      pos = int(i / 8 * self.resolution)
      npImageGrid[pos, ::10, 3] = 1.0
      npImageGrid[::10, pos, 3] = 1.0
    npImageGrid = np.flip(npImageGrid, axis=0)

    self.imageGrid = Image.fromarray((255 * npImageGrid).astype('uint8'))

    if self.mask:

      if printProgress:
        print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Drawing mask')

      # Draw mask
      npImageMask = np.zeros(npImageClean.shape, dtype=npImageClean.dtype)
      for path in self.mask.paths:
        x = [((p[0] - self.imageMin) / (self.imageMax - self.imageMin))
             * self.resolution for p in path]
        y = [((p[1] - self.imageMin) / (self.imageMax - self.imageMin))
             * self.resolution for p in path]
        rr, cc = skimage.draw.polygon(y, x)
        rr, cc = trimImage(rr, cc, self.resolution)
        npImageMask[rr, cc] = [1.0, 0.0, 1.0, 0.5]
      npImageMask = np.flip(npImageMask, axis=0)

      self.imageMask = Image.fromarray((255 * npImageMask).astype('uint8'))

      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Drawing subset of hits and offending bit waveforms')

      # Draw hits and offending bit waveforms
      npImageHits = np.zeros(npImageClean.shape, dtype=npImageClean.dtype)
      npImageHits[:, :, 0] = 1.0
      radius = int(max(3, self.resolution / 500))
      for h in self.hits[:10000]:
        x = int(((h[0] - self.imageMin) /
                (self.imageMax - self.imageMin)) * self.resolution)
        y = int(((h[1] - self.imageMin) /
                (self.imageMax - self.imageMin)) * self.resolution)
        rr, cc = skimage.draw.circle_perimeter(y, x, radius)
        rr, cc = trimImage(rr, cc, self.resolution)
        npImageHits[rr, cc, 3] = 1

      # Plot subset of offenders
      for i in range(len(self.offenders)):
        self.__drawBits(npImageHits, i, self.offenders[i][:1])

      npImageHits = np.flip(npImageHits, axis=0)

      self.imageHits = Image.fromarray((255 * npImageHits).astype('uint8'))

      if printProgress:
        print(f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Drawing mask at margin')

      # Draw margin mask
      npImageMask = np.zeros(npImageClean.shape, dtype=npImageClean.dtype)
      for path in self.mask.adjust(self.maskMarginPair[0][0]).paths:
        x = [((p[0] - self.imageMin) / (self.imageMax - self.imageMin))
             * self.resolution for p in path]
        y = [((p[1] - self.imageMin) / (self.imageMax - self.imageMin))
             * self.resolution for p in path]
        rr, cc = skimage.draw.polygon(y, x)
        rr, cc = trimImage(rr, cc, self.resolution)
        npImageMask[rr, cc] = [1.0, 0.0, 1.0, 0.5]
      npImageMask = np.flip(npImageMask, axis=0)

      if printProgress:
        print(
          f'{"":{indent}}{elapsedStr(start)} {Fore.YELLOW}Drawing worst offending bit waveform')

      # Draw hits and offending bit waveforms
      npImageHits = np.zeros(npImageClean.shape, dtype=npImageClean.dtype)
      npImageHits[:, :, 0] = 1.0
      for h in self.maskMarginPair[0][2][:100000]:
        x = int(((h[0] - self.imageMin) /
                (self.imageMax - self.imageMin)) * self.resolution)
        y = int(((h[1] - self.imageMin) /
                (self.imageMax - self.imageMin)) * self.resolution)
        rr, cc = skimage.draw.circle_perimeter(y, x, radius)
        rr, cc = trimImage(rr, cc, self.resolution)
        npImageHits[rr, cc, 3] = 1

      # Plot subset worst offender
      self.__drawBits(
          npImageHits, self.maskMarginPair[1], [
              self.maskMarginPair[0][1]])

      npImageHits = np.flip(npImageHits, axis=0)

      npImageMargin = layerNumpyImageRGBA(npImageMask, npImageHits)
      self.imageMargin = Image.fromarray((255 * npImageMargin).astype('uint8'))

    else:
      self.imageMask = None
      self.imageHits = None
      self.imageMargin = None

    if printProgress:
      print(f'{"":{indent}}{elapsedStr(start)} {Fore.CYAN}Complete image generation')

  def __drawBits(self, image: np.ndarray, waveformIndex: int,
                 bitIndices: list[int]) -> None:
    '''!@brief Draw bit waveforms on image

    @param image RGBA image to draw on
    @param waveformIndex Index of self.waveforms
    @param bitIndices Index of self.bitCentersT
    '''
    iBitWidth = int((self.tBit / self.tDelta) + 0.5)
    tBitWidthUI = (iBitWidth * self.tDelta / self.tBit)

    t0 = np.linspace(
        0.5 - tBitWidthUI,
        0.5 + tBitWidthUI,
        iBitWidth * 2 + 1)
    factor = int(np.ceil(self.resample / iBitWidth))
    if factor > 1:
      # Expand to 3 bits for better sinc interpolation at the edges
      iBitWidth = int((self.tBit / self.tDelta * 1.5) + 0.5)
      tBitWidthUI = (iBitWidth * self.tDelta / self.tBit)
      t0 = np.linspace(
        0.5 - tBitWidthUI,
        0.5 + tBitWidthUI,
        iBitWidth * 2 + 1)
      tNew = np.linspace(
          0.5 - tBitWidthUI,
          0.5 + tBitWidthUI,
          iBitWidth * 2 * factor + 1)

      T = t0[1] - t0[0]
      sincM = np.tile(tNew, (len(t0), 1)) - \
          np.tile(t0[:, np.newaxis], (1, len(tNew)))
      referenceSinc = np.sinc(sincM / T)

    waveformY = (self.waveforms[waveformIndex][1] -
                 self.yZero) / (self.yOne - self.yZero)
    waveformY = waveformY
    imageMin = -0.5
    imageMax = 1.5
    for i in bitIndices:
      cY = self.bitCentersY[waveformIndex][i]
      cT = self.bitCentersT[waveformIndex][i] / self.tBit

      if factor > 1:
        t = (tNew + cT)
        yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
        y = np.dot(yOriginal, referenceSinc)
      else:
        t = (t0 + cT)
        y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]

      t = (((t - imageMin) / (imageMax - imageMin))
           * self.resolution).astype(np.int32)
      y = (((y - imageMin) / (imageMax - imageMin))
           * self.resolution).astype(np.int32)

      for ii in range(1, len(t)):
        if t[ii] < 0 or t[ii - 1] > (self.resolution):
          continue
        rr, cc, val = skimage.draw.line_aa(y[ii], t[ii], y[ii - 1], t[ii - 1])
        rr, cc, val = trimImage(rr, cc, self.resolution, val=val)
        image[rr, cc, 3] = val + (1 - val) * image[rr, cc, 3]

  def printMeasures(self, indent: int = 0) -> None:
    '''!@brief Print measures to console

    @param indent Base indentation for statements
    '''
    if not self.calculated:
      raise Exception(
        'Eye diagram must be calculated before printing measures')
    print(
      f'{"":{indent}}yZero:        {Fore.CYAN}{metricPrefix(self.measures["yZero"], self.yUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["yZeroStdDev"], self.yUnit)}')
    print(
      f'{"":{indent}}yCross:     {Fore.CYAN}{self.measures["yCrossP"]:6.2f} %   {Fore.BLUE}σ= {self.measures["yCrossPStdDev"]:6.2f} %')
    print(
      f'{"":{indent}}yOne:         {Fore.CYAN}{metricPrefix(self.measures["yOne"], self.yUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["yOneStdDev"], self.yUnit)}')
    print(
      f'{"":{indent}}SNR:           {Fore.CYAN}{self.measures["snr"]:6.2f}')
    print(
      f'{"":{indent}}eyeAmplitude: {Fore.CYAN}{metricPrefix(self.measures["eyeAmplitude"], self.yUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["eyeAmplitudeStdDev"], self.yUnit)}')
    print(
      f'{"":{indent}}eyeHeight:    {Fore.CYAN}{metricPrefix(self.measures["eyeHeight"], self.yUnit)}      {Fore.BLUE}{self.measures["eyeHeightP"]:6.2f} % ')
    print(
      f'{"":{indent}}tBit:         {Fore.CYAN}{metricPrefix(self.measures["tBit"], self.tUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["tBitStdDev"], self.tUnit)}')
    print(
      f'{"":{indent}}fBit:         {Fore.CYAN}{metricPrefix(self.measures["fBit"], self.tUnit)}⁻¹ {Fore.BLUE}σ={metricPrefix(self.measures["fBitStdDev"], self.tUnit)}⁻¹')
    print(
      f'{"":{indent}}tLow:         {Fore.CYAN}{metricPrefix(self.measures["tLow"], self.tUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["tLowStdDev"], self.tUnit)}')
    print(
      f'{"":{indent}}tHigh:        {Fore.CYAN}{metricPrefix(self.measures["tHigh"], self.tUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["tHighStdDev"], self.tUnit)}')
    print(
      f'{"":{indent}}tRise:        {Fore.CYAN}{metricPrefix(self.measures["tRise"], self.tUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["tRiseStdDev"], self.tUnit)}')
    print(
      f'{"":{indent}}tFall:        {Fore.CYAN}{metricPrefix(self.measures["tFall"], self.tUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["tFallStdDev"], self.tUnit)}')
    print(
      f'{"":{indent}}tJitter:      {Fore.CYAN}{metricPrefix(self.measures["tJitterPP"], self.tUnit)}.pp  {Fore.BLUE}{metricPrefix(self.measures["tJitterRMS"], self.tUnit)}.rms')
    print(
      f'{"":{indent}}eyeWidth:     {Fore.CYAN}{metricPrefix(self.measures["eyeWidth"], self.tUnit)}      {Fore.BLUE}{self.measures["eyeWidthP"]:6.2f} %')
    print(
      f'{"":{indent}}dutyCycleDist: {Fore.CYAN}{self.measures["dcd"]:6.2f} %   {Fore.BLUE}σ= {self.measures["dcdStdDev"]:6.2f} %')
    print(
      f'{"":{indent}}nBits:        {Fore.CYAN}{metricPrefix(self.measures["nBits"], "b")}')
    if self.mask:
      print(
        f'{"":{indent}}nBadBits:     {Fore.CYAN}{metricPrefix(self.measures["offenderCount"], "b")}       {Fore.BLUE}{self.measures["ber"]:9.2e}')
      print(
        f'{"":{indent}}maskMargin:    {Fore.CYAN}{self.measures["maskMargin"]:5.1f}%')

  def getMeasures(self) -> dict:
    '''!@brief Get measures

    yZero,        yZeroStdDev:        mean at t=[0.4,0.6]UI,    state=low,        units=yUnit
    yCross,    yCrossStdDev:    mean at t=[-0.05,0.05]UI, state=change,     units=yUnit
    yCrossP,   yCrossPStdDev:   100 * (yCross - yZero) / eyeAmplitude, state=change,     units=percent
    yOne,         yOneStdDev:         mean at t=[0.4,0.6]UI,    state=high,       units=yUnit
    snr:                              eyeAmplitude / (yZeroStdDev + yOneStdDev),  units=unitless
    eyeAmplitude, eyeAmplitudeStdDev: yOne - yZero,  units=yUnit
    eyeHeight:                        (yOne - 3 * yOneStdDev) - (yZero + 3 * yZeroStdDev),  units=yUnit
    eyeHeightP:                       100 * eyeHeight / eyeAmplitude, units=percent
    tBit,         tBitStdDev:         time between bits,                      units=tUnit
    fBit,         fBitStdDev:         1/tBit,                      units=tUnit⁻¹
    tRise20,      tRise20StdDev:      t @ y=20%,  edge=rising-, units=tUnit
    tRise50,      tRise50StdDev:      t @ y=50%,  edge=rising-, units=tUnit
    tRise80,      tRise80StdDev:      t @ y=80%,  edge=rising-, units=tUnit
    tFall20,      tFall20StdDev:      t @ y=20%,  edge=falling-, units=tUnit
    tRise50,      tFall50StdDev:      t @ y=50%,  edge=falling-,                              units=tUnit
    tRise80,      tFall80StdDev:      t @ y=80%,  edge=falling-,                              units=tUnit
    tRise,        tRiseStdDev:        tRise80 - tRise20, edge=rising-,                        units=tUnit
    tFall,        tFallStdDev:        tFall20 - tRise80, edge=falling-,                       units=tUnit
    tLow,         tLowStdDev:         tRise50+ - tFall50-, state=low, edges @ y=50%,  units=tUnit
    tHigh,        tHighStdDev:        tFall50+ - tRise50-, state=high, edges @ y=50%, units=tUnit
    tJitterPP:                        full width of time histogram at yCross,          state=change, units=tUnit
    tJitterRMS:                       stddev of time histogram at yCross,          state=change, units=tUnit
    eyeWidth:                         (tEdge+ - tEdge+StdDev) - (tEdge- + tEdge-StdDev), units=tUnit
    eyeWidthP:                        100 * eyeWidth / tBit, units=percent
    dcd, dcdStdDev: 100 * (tHigh - tLow) / (2 * tBit), edges @ y=50%, units=tUnit
    nBits:                            number of bits,  units=unitless
    offenderCount:                    number of bits that hit the mask,  units=unitless
    ber:                              offenderCount / nBits,  units=unitless
    maskMargin:                       100 * largest mask adjust without hits,  units=percent
    maxStateLength:                   largest number of consecutive bits with same state
    averageStateLength:               average number of consecutive bits with same state
    longStateP:                       percent of consecutive bits with same state longer than 1

    @return dict Dictionary of measured values
    '''
    if not self.calculated:
      raise Exception(
        'Eye diagram must be calculated before printing measures')
    return self.measures

  def getConfig(self) -> dict:
    '''!@brief Get the configuration used to generate the eye diagram

    @return dict Dictionary of parameters
    '''
    if not self.calculated:
      raise Exception(
        'Eye diagram must be calculated before printing measures')
    config = {
      'tUnit': self.tUnit,
      'tDelta': self.tDelta,
      'yUnit': self.yUnit,
      'nWaveforms': self.waveforms.shape[0],
      'nSamples': self.waveforms.size / 2,
      'resolution': self.resolution,
      'mask': self.mask,
      'nBitsMax': self.nBitsMax,
      'nBitsMin': self.nBitsMin,
      'method': self.method,
      'resample': self.resample,
      'yZero': self.yZero,
      'yOne': self.yOne,
      'hysteresis': self.hysteresis,
      'yRise': self.thresholdRise,
      'yHalf': self.thresholdHalf,
      'yFall': self.thresholdFall,
      'histNMax': self.histNMax,
      'imageMin': self.imageMin,
      'imageMax': self.imageMax,
      'manualTBit': self.manualTBit,
      'pllBandwidth': self.pllBandwidth
    }
    return config

  def saveImages(self, dir: str = '.') -> None:
    '''!@brief Save generated images to a directory

    self.imageClean => dir/eye-diagram-clean.png
    self.imageGrid => dir/eye-diagram-grid.png
    self.imageMask => dir/eye-diagram-mask-clean.png
    self.imageHits => dir/eye-diagram-mask-hits.png
    self.imageMargin => dir/eye-diagram-mask-margin.png

    @param dir Directory to save to
    '''
    if not self.calculated:
      raise Exception('Eye diagram must be calculated before saving images')
    os.makedirs(dir, exist_ok=True)
    self.imageClean.save(os.path.join(dir, 'eye-diagram-clean.png'))
    self.imageGrid.save(os.path.join(dir, 'eye-diagram-grid.png'))
    if self.mask:
      self.imageMask.save(os.path.join(dir, 'eye-diagram-mask-clean.png'))
      self.imageHits.save(os.path.join(dir, 'eye-diagram-mask-hits.png'))
      self.imageMargin.save(os.path.join(dir, 'eye-diagram-mask-margin.png'))

  def getImages(self, asString: bool = False) -> dict:
    '''!@brief Get generated images, optionally as a base64 encoded string (PNG)

    @param asString True will return base64 encoded string for each image as PNG
    @return dict {name: image} image is PIL.Image or str
    '''
    if not self.calculated:
      raise Exception('Eye diagram must be calculated before getting images')
    output = {}
    if asString:
      output['clean'] = imageToBase64Image(self.imageClean)
      output['grid'] = imageToBase64Image(self.imageGrid)
      if self.mask:
        output['mask'] = imageToBase64Image(self.imageMask)
        output['hits'] = imageToBase64Image(self.imageHits)
        output['margin'] = imageToBase64Image(self.imageMargin)
    else:
      output['clean'] = self.imageClean
      output['grid'] = self.imageGrid
      if self.mask:
        output['mask'] = self.imageMask
        output['hits'] = self.imageHits
        output['margin'] = self.imageMargin
    return output

class WaveformException(Exception):

  def __init__(self, str: str, waveformIndex: str = None,
               bitIndex: int = None) -> None:
    '''!@brief Create a waveform exception containing information to plot exception

    @param str Exception string
    @param waveformIndex Index of exceptional waveform
    @param bitIndex Index of exceptional bit
    '''
    super().__init__(str)
    self.waveformIndex = waveformIndex
    self.bitIndex = bitIndex

def _runnerCalculateLevels(
  waveformY: np.ndarray, nMax: int = 50e3) -> tuple[float, float]:
  '''!@brief Calculate the high and low levels of the waveform

  @param waveformY Waveform data array [y0, y1,..., yn]
  @param nMax Limit the number of points to calculate on (histogram downsampling), None for no limit
  @return dict Dictionary of yMin, yMax, yZero, yZeroStdDev, yZeroComponent, yOne, yOneStdDev, yOneComponents
  '''
  if nMax:
    waveformY = histogramDownsample(waveformY, nMax)

  yMin = np.amin(waveformY)
  yMax = np.amax(waveformY)
  yMid = (yMin + yMax) / 2

  lowerValues = waveformY[np.where(waveformY < yMid)]
  yZeroStdDev = np.std(lowerValues)

  upperValues = waveformY[np.where(waveformY > yMid)]
  yOneStdDev = np.std(upperValues)

  lowerComponents = fitGaussianMix(lowerValues, nMax=3)
  upperComponents = fitGaussianMix(upperValues, nMax=3)

  if len(lowerComponents) == 1:
    yZero = lowerComponents[0][1]
  elif lowerComponents[0][0] > 2 * lowerComponents[1][0]:
    # yZero prominent
    yZero = lowerComponents[0][1]
  else:
    # yZero not prominent find closest to average
    averageValue = np.average(lowerValues)
    lowerComponents = sorted(
        lowerComponents, key=lambda c: abs(
            c[1] - averageValue))
    yZero = lowerComponents[0][1]

  if len(upperComponents) == 1:
    yOne = upperComponents[0][1]
  elif upperComponents[0][0] > 2 * upperComponents[1][0]:
    # yOne prominent
    yOne = upperComponents[0][1]
  else:
    # yOne not prominent find closest to average
    averageValue = np.average(upperValues)
    upperComponents = sorted(
        upperComponents, key=lambda c: abs(
            c[1] - averageValue))
    yOne = upperComponents[0][1]

  results = {
    'yMin': yMin,
    'yMax': yMax,
    'yZero': yZero,
    'yZeroStdDev': yZeroStdDev,
    'yZeroComponents': lowerComponents,
    'yOne': yOne,
    'yOneStdDev': yOneStdDev,
    'yOneComponents': upperComponents,
  }
  return results

def _runnerCleanEdges(
  edges: tuple[list, list], tBit: float, nBitsMin: float = 0.1) -> tuple[list, list]:
  '''!@brief Runner to extract bit centers from list of rising and falling edges

  @param edges tuple of rising edges, falling edges
  @param tBit Time for a single bit
  @param nBitsMin Duration (specified in units of tBit) threshold to consider a state a glitch
  @return tuple[list, list] tuple of rising edges, falling edges
  '''
  edgesRiseDirty, edgesFallDirty = edges
  edgesRise = []
  edgesFall = []
  # Remove excessive edges using tHoldoff
  # If an edge is removed another from the other direction needs to be
  # removed too to maintain state balance
  tHoldOff = nBitsMin * tBit
  if edgesRiseDirty[0] < edgesFallDirty[0]:
    prevEdge = edgesFallDirty[0]
    skip = False
    for i in range(1, min(len(edgesRiseDirty) - 1, len(edgesFallDirty))):
      if skip:
        skip = False
      elif edgesRiseDirty[i] > (prevEdge + tHoldOff):
        edgesRise.append(edgesRiseDirty[i])
        prevEdge = edgesRiseDirty[i]
      else:
        skip = True

      if skip:
        skip = False
      elif edgesFallDirty[i] > (prevEdge + tHoldOff):
        edgesFall.append(edgesFallDirty[i])
        prevEdge = edgesFallDirty[i]
      else:
        skip = True
  else:
    prevEdge = edgesRiseDirty[0]
    skip = False
    for i in range(1, min(len(edgesRiseDirty), len(edgesFallDirty) - 1)):
      if skip:
        skip = False
      elif edgesFallDirty[i] > (prevEdge + tHoldOff):
        edgesFall.append(edgesFallDirty[i])
        prevEdge = edgesFallDirty[i]
      else:
        skip = True

      if skip:
        skip = False
      elif edgesRiseDirty[i] > (prevEdge + tHoldOff):
        edgesRise.append(edgesRiseDirty[i])
        prevEdge = edgesRiseDirty[i]
      else:
        skip = True
  return edgesRise, edgesFall

def _runnerClockRecovery(edges: tuple[list, list], tDelta: float,
                         tBit: float, pllBandwidth: float = 100e3, risingEdgesLead: bool = None, debug: bool = False) -> tuple[list, list, list]:
  '''!@brief Runner to recover the clock from edge transitions

  @param edges tuple of rising edges, falling edges
  @param tDelta Time between samples
  @param tBit Time for a single bit (initial pll settings)
  @param pllBandwidth -3dB cutoff frequency of pll feedback (1st order low pass)
  @param risingEdgesLead True will ensure even edges are rising, False will ensure even edges are falling, None will not check
  @param debug True will return more data for debugging purposes, see return, False will not
  @return tuple[list[float]...] (clockEdges, periods, tie, nBits, [phases], [phaseErrors], [offsets], [offsetErrors], [delays], [delayErrors])
      clockEdges is a list of clock edges, at sampling point
      periods is a list of period for each edge
      tie is a list of Time Interval Errors for each data edge (not clock edge)
      nBits is a list of bits for each data edge (not clock edge) aka number of same state periods
      if debug is True:
      phases, offsets, delays are lists of the PLL parameters for each clockEdge
      phaseErrors, offsetErrors, delayErrors are lists of the PLL error parameters for each clockEdge
  '''
  # Interleave edges
  edgesRise, edgesFall = edges
  edges = []
  edges.extend(edgesRise)
  edges.extend(edgesFall)
  edges = sorted(edges)
  if risingEdgesLead is not None:
    if (edges[0] == edgesRise[0]) != risingEdgesLead:
      # Remove first edge according to if rising or falling should be even
      edges = edges[1:]

  clockEdges = []
  ties = []
  periods = []
  nBits = []

  t = edges[0]
  delay = tBit
  phase = 0

  duration = (edges[1] - edges[0])
  offset = -((duration + tBit / 2) % tBit - tBit / 2)

  idealEdge = t
  phaseError = 0
  delayError = 0
  offsetError = t - edges[0]
  prevPhaseError = 0
  nBitsEven = 1

  # Debug information
  phases = []
  offsets = []
  delays = []
  phaseErrors = []
  offsetErrors = []
  delayErrors = []

  # Run PLL
  # Adjust delay and phase on even edges
  # Adjust offset on odd edges
  evenEdge = True
  for edge in edges:
    nBitsSame = 0
    if evenEdge:
      nBitsEven = 2
    else:
      edge += offset
    edgeError = t - edge
    while edgeError < tBit / 2:
      # Record clock edge
      clockEdges.append(t + delay / 2 - offset / 2)

      # Adjust delay
      phaseError = 0
      delayError = 0
      offsetError = 0
      if edgeError > -tBit / 2:
        if evenEdge:
          tie = (edge - idealEdge)
        else:
          tie = (edge - idealEdge - offset)
        ties.append((edge, tie))
        if evenEdge:
          # consider even edges a phase and delay error
          phaseError = edgeError
          delayError = (phaseError - prevPhaseError) / nBitsEven
        else:
          # consider odd edges an offset error
          offsetError = -edgeError
      w = 2 * np.pi * delay * pllBandwidth
      alpha = w / (w + 1)
      # alpha = 1
      delay = (1 - alpha) * delay + alpha * (delay - delayError)
      phase = (1 - alpha) * phase + alpha * (-phaseError)
      offset = (1 - alpha) * offset + alpha * (offset - offsetError)
      if edgeError > -tBit / 2:
        # Don't include phase offset in previous phase error
        if evenEdge:
          prevPhaseError = phaseError + phase

      period = delay + phase
      if period < tDelta:
        raise Exception(
          f'Clock recovery has failed with too small period {metricPrefix(period)}')

      # Record statistical information
      periods.append(period)
      phases.append(phase)
      offsets.append(offset)
      delays.append(delay)
      phaseErrors.append(phaseError)
      offsetErrors.append(offsetError)
      delayErrors.append(delayError)

      # Step to next bit
      t += period
      idealEdge += tBit
      edgeError = t - edge
      nBitsEven += 1
      nBitsSame += 1
    evenEdge = not evenEdge
    nBits.append(nBitsSame)

  # Compensate TIE for wrong tBit
  ties = np.array(ties)
  slope = (ties[-1, 1] - ties[0, 1]) / (ties[-1, 0] - ties[0, 0])
  ties[:, 1] = ties[:, 1] - slope * (ties[:, 0] - ties[0, 0])

  # High pass TIEs to remove spread spectrum clocking artifacts
  tiesFiltered = [ties[0, 1]]
  for i in range(1, len(ties)):
    alpha = 1 / (2 * np.pi * pllBandwidth * (ties[i, 0] - ties[i - 1, 0]) + 1)
    tiesFiltered.append(
      alpha * tiesFiltered[i - 1] + alpha * (ties[i, 1] - ties[i - 1, 1]))

  if debug:
    return clockEdges, periods, tiesFiltered, nBits, phases, phaseErrors, offsets, offsetErrors, delays, delayErrors
  return clockEdges, periods, tiesFiltered, nBits


def _runnerBitExtract(clockEdges: list, tZero: float,
                      tDelta: float) -> tuple[list, list]:
  '''!@brief Runner to extract bit centers from list of rising and falling edges

  @param edges tuple of rising edges, falling edges
  @param tZero Time for the first sample
  @param tDelta Time between samples
  @return tuple[list[float], list[int]] (bitCentersT, bitCentersY)
      bitCentersT is a list of centerTs. Use np.linspace(center - sliceHalfWidth * tDelta, c + shw * td, shw * 2 + 1)
      bitCentersY is a list of center indices. Use y[sliceCenter - sliceHalfWidth : c + shw + 1]
  '''
  bitCentersT = []
  bitCentersY = []

  for b in clockEdges[5:-5]:
    bitCenterAdjusted = b % tDelta
    # xMin = -bitCenterAdjusted - sliceHalfWidth * tDelta
    # xMax = -bitCenterAdjusted + sliceHalfWidth * tDelta
    bitCentersT.append(-bitCenterAdjusted)

    sliceCenter = int(((b - tZero - bitCenterAdjusted) / tDelta) + 0.5)
    # sliceMin = sliceCenter - sliceHalfWidth
    # sliceMax = sliceCenter + sliceHalfWidth
    bitCentersY.append(sliceCenter)

  return bitCentersT, bitCentersY

def _runnerCollectValuesY(waveformY: np.ndarray, bitCentersT: list[float], bitCentersY: list[int],
                          tDelta: float, tBit: float, yHalf: float, resample: int = 50) -> dict:
  '''!@brief Collect values from waveform: zero/cross/one vertical levels

  @param waveformY Waveform data array [y0, y1,..., yn]
  @param bitCentersT list of centerTs. Using np.linspace(center - sliceHalfWidth * tDelta, c + shw * td, shw * 2 + 1)
  @param bitCentersY list of center indices. Use y[sliceCenter - sliceHalfWidth : c + shw + 1]
  @param tDelta Time between samples
  @param tBit Time for a single bit
  @param yHalf Threshold to decide '1' or '0'
  @param resample n=0 will not resample, n>0 will use sinc interpolation to resample a single bit to at least n segments (tDelta = tBit / n)
  @return dict Dictionary of collected values: zero, cross, one
  '''
  iBitWidthOriginal = int((tBit / tDelta) + 0.5)
  iBitWidth = iBitWidthOriginal
  tBitWidthUI = (iBitWidth * tDelta / tBit)

  n = iBitWidth * 2 + 1
  t0 = np.linspace(0.5 - tBitWidthUI, 0.5 + tBitWidthUI, n)
  tList = t0.tolist()

  waveformY = waveformY.tolist()

  factor = int(np.ceil(resample / iBitWidth))
  if factor > 1:
    # Expand to 3 bits for better sinc interpolation at the edges
    iBitWidth = int((tBit / tDelta * 1.5) + 0.5)
    tBitWidthUI = (iBitWidth * tDelta / tBit)
    t0 = np.linspace(
      0.5 - tBitWidthUI,
      0.5 + tBitWidthUI,
      iBitWidth * 2 + 1)
    n = iBitWidth * 2 * factor + 1
    tNew = np.linspace(0.5 - tBitWidthUI, 0.5 + tBitWidthUI, n)

    T = t0[1] - t0[0]
    sincM = np.tile(tNew, (len(t0), 1)) - \
        np.tile(t0[:, np.newaxis], (1, len(tNew)))
    referenceSinc = np.sinc(sincM / T)
    tList = tNew.tolist()

  # One level and zero level
  # Historgram mean 40% to 60%
  valuesZero = []
  valuesOne = []
  # Historgram mean -5% to 5% for bit transitions
  valuesCross = []
  for i in range(len(bitCentersT)):
    cT = bitCentersT[i] / tBit
    cY = bitCentersY[i]

    # Only look at transitions at t=0 for yCross
    transition = (waveformY[cY - iBitWidthOriginal]
                  > yHalf) != (waveformY[cY] > yHalf)

    if factor > 1:
      yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
      y = np.dot(yOriginal, referenceSinc).tolist()
    else:
      y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]

    bitValues = []

    for ii in range(n):
      t = tList[ii] + cT
      if t >= 0.4 and t <= 0.6:
        bitValues.append(y[ii])
      if transition and t >= -0.05 and t <= 0.05:
        valuesCross.append(y[ii])

    if np.average(bitValues) > yHalf:
      valuesOne.extend(bitValues)
    else:
      valuesZero.extend(bitValues)

  values = {
    'zero': valuesZero,
    'cross': valuesCross,
    'one': valuesOne,
  }
  return values

def _runnerCollectValuesT(waveformY: np.ndarray, bitCentersT: list[float], bitCentersY: list[int], tDelta: float,
                          tBit: float, yZero: float, yCross: float, yOne: float, resample: int = 50, waveformI: int = 0) -> dict:
  '''!@brief Collect values from waveform: rise/fall times @ 20%, 50%, 80%, crossing

  @param waveformY Waveform data array [y0, y1,..., yn]
  @param bitCentersT list of centerTs. Using np.linspace(center - sliceHalfWidth * tDelta, c + shw * td, shw * 2 + 1)
  @param bitCentersY list of center indices. Use y[sliceCenter - sliceHalfWidth : c + shw + 1]
  @param tDelta Time between samples
  @param tBit Time for a single bit
  @param yZero Vertical value for a '0'
  @param yCross Vertical value for bit transition
  @param yOne Vertical value for a '1'
  @param resample n=0 will not resample, n>0 will use sinc interpolation to resample a single bit to at least n segments (tDelta = tBit / n)
  @param waveformI Add to an WaveformException
  @return dict Dictionary of collected values: rise20, rise50, rise80, fall20, fall50, fall80
  '''
  iBitWidthOriginal = int((tBit / tDelta) + 0.5)
  iBitWidth = iBitWidthOriginal
  tBitWidth = (iBitWidth * tDelta)

  n = iBitWidth * 2 + 1
  t0 = np.linspace(-tBitWidth + tBit / 2, tBitWidth + tBit / 2, n)
  sliceCenterStart = int(n * 1 / 4)  # T = 0b
  sliceCenterStop = int(n * 3 / 4) + 1  # T = 1b
  tList = t0.tolist()

  factor = int(np.ceil(resample / iBitWidth))
  if factor > 1:
    # Expand to 3 bits for better sinc interpolation at the edges
    iBitWidth = int((tBit / tDelta * 1.5) + 0.5)
    tBitWidth = (iBitWidth * tDelta)
    t0 = np.linspace(-tBitWidth + tBit / 2, tBitWidth + tBit / 2,
                     iBitWidth * 2 + 1)
    n = iBitWidth * 2 * factor + 1
    tNew = np.linspace(-tBitWidth + tBit / 2, tBitWidth + tBit / 2, n)

    T = t0[1] - t0[0]
    sincM = np.tile(tNew, (len(t0), 1)) - \
        np.tile(t0[:, np.newaxis], (1, len(tNew)))
    referenceSinc = np.sinc(sincM / T)
    sliceCenterStart = int(n * 1 / 3)  # T = 0b
    sliceCenterStop = int(n * 2 / 3) + 1  # T = 1b
    tList = tNew.tolist()

  waveformY = (waveformY - yZero) / (yOne - yZero)
  waveformY = waveformY.tolist()
  yCross = (yCross - yZero) / (yOne - yZero)

  # Collect time for edges at 20%, 50%, and 80% values
  valuesRise20 = []
  valuesRise50 = []
  valuesRise80 = []
  valuesRise = []
  valuesFall20 = []
  valuesFall50 = []
  valuesFall80 = []
  valuesFall = []
  valuesCross = []
  for i in range(len(bitCentersT)):
    cT = bitCentersT[i]
    cY = bitCentersY[i]
    # Only look at transitions at t=0
    if (waveformY[cY - iBitWidthOriginal] > 0.5) == (waveformY[cY] > 0.5):
      continue

    if factor > 1:
      yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
      y = np.dot(yOriginal, referenceSinc).tolist()
    else:
      y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
    yCenter = y[sliceCenterStart:sliceCenterStop]

    if (waveformY[cY] > 0.5):
      iMax = sliceCenterStart + np.argmax(yCenter)
      if y[iMax] < 0.8:
        # errorStr = 'Waveform does not reach 80%\n'
        # errorStr += f'  Waveform Index: {waveformI}\n'
        # errorStr += f'  i: {i}\n'
        # errorStr += f'  yMax: {metricPrefix(y[iMax])}\n'
        # raise WaveformException(errorStr, waveformI, i)
        t = tList[iMax]
        i80 = iMax
      else:
        t, i80 = getCrossing(tList, y, iMax, 0.8)
      rise80 = t + cT
      valuesRise80.append(rise80)
      t, i50 = getCrossing(tList, y, i80, 0.5)
      valuesRise50.append(t + cT)
      t, i20 = getCrossing(tList, y, i50, 0.2)
      # if np.isnan(t):
      #   errorStr = 'Waveform does not reach 20%\n'
      #   errorStr += f'  Waveform Index: {waveformI}\n'
      #   errorStr += f'  i: {i}\n'
      #   errorStr += f'  yMax: {metricPrefix(y[iMin])}\n'
      #   raise WaveformException(errorStr, waveformI, i)
      rise20 = t + cT
      valuesRise20.append(rise20)
      valuesRise.append(rise80 - rise20)
      if yCross > 0.5:
        crossStart = i50 - 1
        crossForward = True
      else:
        crossStart = i50 + 1
        crossForward = False
      t, _ = getCrossing(tList, y, crossStart, yCross, crossForward)
      t += cT
      if (t < -tBit / 2) or (t > tBit / 2):
        valuesCross.append(np.nan)
      else:
        valuesCross.append(t)
    else:
      iMin = sliceCenterStart + np.argmin(yCenter)
      if y[iMin] > 0.2:
        # errorStr = 'Waveform does not reach 20%\n'
        # errorStr += f'  Waveform Index: {waveformI}\n'
        # errorStr += f'  i: {i}\n'
        # errorStr += f'  yMax: {metricPrefix(y[iMin])}\n'
        # raise WaveformException(errorStr, waveformI, i)
        t = tList[iMin]
        i20 = iMin
      else:
        t, i20 = getCrossing(tList, y, iMin, 0.2)
      fall20 = t + cT
      valuesFall20.append(fall20)
      t, i50 = getCrossing(tList, y, i20, 0.5)
      valuesFall50.append(t + cT)
      t, i80 = getCrossing(tList, y, i50, 0.8)
      # if np.isnan(t):
      #   errorStr = 'Waveform does not reach 80%\n'
      #   errorStr += f'  Waveform Index: {waveformI}\n'
      #   errorStr += f'  i: {i}\n'
      #   errorStr += f'  yMax: {metricPrefix(y[iMax])}\n'
      #   raise WaveformException(errorStr, waveformI, i)
      fall80 = t + cT
      valuesFall80.append(fall80)
      valuesFall.append(fall20 - fall80)
      if yCross > 0.5:
        crossStart = i50 + 1
        crossForward = False
      else:
        crossStart = i50 - 1
        crossForward = True
      t, _ = getCrossing(tList, y, crossStart, yCross, crossForward)
      t += cT
      if (t < -tBit / 2) or (t > tBit / 2):
        valuesCross.append(np.nan)
      else:
        valuesCross.append(t)

  values = {
    'rise20': valuesRise20,
    'rise50': valuesRise50,
    'rise80': valuesRise80,
    'rise': valuesRise,
    'fall20': valuesFall20,
    'fall50': valuesFall50,
    'fall80': valuesFall80,
    'fall': valuesFall,
    'cross': valuesCross
  }
  return values

def _runnerCollectMaskHits(waveformY: np.ndarray, bitCentersT: list[float], bitCentersY: list[int],
                           tDelta: float, tBit: float, yZero: float, yOne: float, mask: Mask, resample: int = 50) -> tuple[list[int], list[tuple]]:
  '''!@brief Collect mask hits between waveform and mask

  @param waveformY Waveform data array [y0, y1,..., yn]
  @param bitCentersT list of centerTs. Using np.linspace(center - sliceHalfWidth * tDelta, c + shw * td, shw * 2 + 1)
  @param bitCentersY list of center indices. Use y[sliceCenter - sliceHalfWidth : c + shw + 1]
  @param tDelta Time between samples
  @param tBit Time for a single bit
  @param yZero Vertical value for a '0'
  @param yOne Vertical value for a '1'
  @param mask Mask to check against
  @param resample n=0 will not resample, n>0 will use sinc interpolation to resample a single bit to at least n segments (tDelta = tBit / n)
  @return tuple[list[int], list[tuple]] Tuple of (list of offending bit indices:int, list of mask hits:tuple)
  '''
  iBitWidth = int((tBit / tDelta) + 0.5)
  tBitWidthUI = (iBitWidth * tDelta / tBit)

  t0 = np.linspace(0.5 - tBitWidthUI, 0.5 + tBitWidthUI, iBitWidth * 2 + 1)

  waveformY = (waveformY - yZero) / (yOne - yZero)
  waveformY = waveformY.tolist()

  offenders = []
  hits = []
  factor = int(np.ceil(resample / iBitWidth))
  if factor > 1:
    # Expand to 3 bits for better sinc interpolation at the edges
    iBitWidth = int((tBit / tDelta * 1.5) + 0.5)
    tBitWidthUI = (iBitWidth * tDelta / tBit)
    t0 = np.linspace(
      0.5 - tBitWidthUI,
      0.5 + tBitWidthUI,
      iBitWidth * 2 + 1)
    tNew = np.linspace(
        0.5 - tBitWidthUI,
        0.5 + tBitWidthUI,
        iBitWidth * 2 * factor + 1)

    T = t0[1] - t0[0]
    sincM = np.tile(tNew, (len(t0), 1)) - \
        np.tile(t0[:, np.newaxis], (1, len(tNew)))
    referenceSinc = np.sinc(sincM / T)

  for i in range(len(bitCentersT)):
    cY = bitCentersY[i]
    cT = bitCentersT[i] / tBit

    if factor > 1:
      t = (tNew + cT).tolist()
      yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
      y = np.dot(yOriginal, referenceSinc).tolist()
    else:
      t = (t0 + cT).tolist()
      y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
    waveformHits = getHits(t, y, mask.paths)

    if len(waveformHits) != 0:
      offenders.append(i)
      hits.extend(waveformHits)

  return offenders, hits

def _runnerAdjustMaskMargin(waveformY: np.ndarray, bitCentersT: list[float], bitCentersY: list[int], tDelta: float, tBit: float,
                            yZero: float, yOne: float, mask: Mask, resample: int = 50, initialOffenders: list = []) -> tuple[float, int, list[tuple]]:
  '''!@brief Collect mask hits between waveform and mask

  @param waveformY Waveform data array [y0, y1,..., yn]
  @param bitCentersT list of centerTs. Using np.linspace(center - sliceHalfWidth * tDelta, c + shw * td, shw * 2 + 1)
  @param bitCentersY list of center indices. Use y[sliceCenter - sliceHalfWidth : c + shw + 1]
  @param tDelta Time between samples
  @param tBit Time for a single bit
  @param yZero Vertical value for a '0'
  @param yOne Vertical value for a '1'
  @param mask Mask to check against
  @param resample n=0 will not resample, n>0 will use sinc interpolation to resample a single bit to at least n segments (tDelta = tBit / n)
  @param initialOffenders Empty list will start at margin=+1 and check all bits. Non-empty list will start at margin=0 and check the bits in the list
  @return tuple[float, int, list[tuple]] Tuple of (mask margin:float, worst offending bit indice:int, worst offending bit mask hits:tuple)
  '''
  iBitWidth = int((tBit / tDelta) + 0.5)
  tBitWidthUI = (iBitWidth * tDelta / tBit)

  t0 = np.linspace(0.5 - tBitWidthUI, 0.5 + tBitWidthUI, iBitWidth * 2 + 1)

  waveformY = (waveformY - yZero) / (yOne - yZero)
  waveformY = waveformY.tolist()

  offender = None
  factor = int(np.ceil(resample / iBitWidth))
  if factor > 1:
    # Expand to 3 bits for better sinc interpolation at the edges
    iBitWidth = int((tBit / tDelta * 1.5) + 0.5)
    tBitWidthUI = (iBitWidth * tDelta / tBit)
    t0 = np.linspace(
      0.5 - tBitWidthUI,
      0.5 + tBitWidthUI,
      iBitWidth * 2 + 1)
    tNew = np.linspace(
        0.5 - tBitWidthUI,
        0.5 + tBitWidthUI,
        iBitWidth * 2 * factor + 1)

    T = t0[1] - t0[0]
    sincM = np.tile(tNew, (len(t0), 1)) - \
        np.tile(t0[:, np.newaxis], (1, len(tNew)))
    referenceSinc = np.sinc(sincM / T)

  maskMargin = 0
  if len(initialOffenders) == 0:
    maskMargin = 1
    initialOffenders = list(range(len(bitCentersT)))

  maskAdjusted = mask.adjust(maskMargin)
  # Roughly on first offender
  cY = bitCentersY[initialOffenders[0]]
  cT = bitCentersT[initialOffenders[0]] / tBit

  if factor > 1:
    t = (tNew + cT).tolist()
    yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
    y = np.dot(yOriginal, referenceSinc).tolist()
  else:
    t = (t0 + cT).tolist()
    y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]

  while isHitting(t, y, maskAdjusted.paths) and maskMargin >= -1:
    maskMargin -= 0.01
    maskAdjusted = mask.adjust(maskMargin)
    offender = initialOffenders[0]
  maskMargin += 0.01

  # Finely decrement mask until each bit passes
  for i in initialOffenders:
    cY = bitCentersY[i]
    cT = bitCentersT[i] / tBit

    if factor > 1:
      t = (tNew + cT).tolist()
      yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
      y = np.dot(yOriginal, referenceSinc).tolist()
    else:
      t = (t0 + cT).tolist()
      y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]

    while isHitting(t, y, maskAdjusted.paths) and maskMargin >= -1:
      maskMargin -= 0.001
      maskAdjusted = mask.adjust(maskMargin)
      offender = i

  # Collect mask hits for the worst offender
  cY = bitCentersY[offender]
  cT = bitCentersT[offender] / tBit

  if factor > 1:
    t = (tNew + cT).tolist()
    yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
    y = np.dot(yOriginal, referenceSinc).tolist()
  else:
    t = (t0 + cT).tolist()
    y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
  maskAdjusted = mask.adjust(maskMargin + 0.001)
  hits = getHits(t, y, maskAdjusted.paths)

  return maskMargin, offender, hits

def _runnerGenerateHeatMap(waveformY: np.ndarray, bitCentersT: list[float], bitCentersY: list[int], tDelta: float,
                           tBit: float, yZero: float, yOne: float, resample: int = 50, resolution: int = 500) -> np.ndarray:
  '''!@brief Generate heat map by layering all waveforms and counting overlaps

  @param waveformY Waveform data array [y0, y1,..., yn]
  @param bitCentersT list of centerTs. Using np.linspace(center - sliceHalfWidth * tDelta, c + shw * td, shw * 2 + 1)
  @param bitCentersY list of center indices. Use y[sliceCenter - sliceHalfWidth : c + shw + 1]
  @param tDelta Time between samples
  @param tBit Time for a single bit
  @param yZero Vertical value for a '0'
  @param yOne Vertical value for a '1'
  @param resample n=0 will not resample, n>0 will use sinc interpolation to resample a single bit to at least n segments (tDelta = tBit / n)
    @param resolution Resolution of square eye diagram image
  @return np.ndarray Grid of heat map counts
  '''
  iBitWidth = int((tBit / tDelta) + 0.5)
  tBitWidthUI = (iBitWidth * tDelta / tBit)

  t0 = np.linspace(0.5 - tBitWidthUI, 0.5 + tBitWidthUI, iBitWidth * 2 + 1)

  imageMin = -0.5
  imageMax = 1.5
  waveformY = (waveformY - yZero) / (yOne - yZero)

  factor = int(np.ceil(resample / iBitWidth))
  if factor > 1:
    # Expand to 3 bits for better sinc interpolation at the edges
    iBitWidth = int((tBit / tDelta * 1.5) + 0.5)
    tBitWidthUI = (iBitWidth * tDelta / tBit)
    t0 = np.linspace(
      0.5 - tBitWidthUI,
      0.5 + tBitWidthUI,
      iBitWidth * 2 + 1)
    tNew = np.linspace(
        0.5 - tBitWidthUI,
        0.5 + tBitWidthUI,
        iBitWidth * 2 * factor + 1)

    T = t0[1] - t0[0]
    sincM = np.tile(tNew, (len(t0), 1)) - \
        np.tile(t0[:, np.newaxis], (1, len(tNew)))
    referenceSinc = np.sinc(sincM / T)

  grid = np.zeros((resolution, resolution), dtype=np.int32)

  for i in range(len(bitCentersT)):
    cY = bitCentersY[i]
    cT = bitCentersT[i] / tBit

    if factor > 1:
      t = (tNew + cT)
      yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
      y = np.dot(yOriginal, referenceSinc)
    else:
      t = (t0 + cT)
      y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]

    td = (((t - imageMin) / (imageMax - imageMin))
          * resolution).astype(np.int32)
    yd = (((y - imageMin) / (imageMax - imageMin))
          * resolution).astype(np.int32)
    brescount.counter(td, yd, grid)

  return grid
