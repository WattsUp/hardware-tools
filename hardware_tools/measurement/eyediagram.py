from __future__ import annotations
import colorama
from colorama import Fore
import datetime
import matplotlib.pyplot as pyplot
from matplotlib.ticker import FuncFormatter
from multiprocessing import Pool, cpu_count
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.signal import argrelextrema

from ..math import Line, elapsedStr, metricPrefix, fitGaussian, gaussian, binExact
from .extension import getEdges

colorama.init(autoreset=True)

class Mask:
  def __init__(self) -> None:
    '''!@brief Create a new Mask
    Derrive mask shapes from this class such as a MaskOctagon, MaskDecagon, MaskPulse, etc.
    '''
    self.lines = []
    self.linesUpper = []
    self.linesLower = []

    self.bounds = {}
    self.boundsUpper = {}
    self.boundLower = {}

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
    self.x1 = x1
    self.x2 = x2
    self.x3 = x3
    self.y1 = y1
    self.y2 = y2
    self.y3 = y3
    if y4 is None:
      y4 = y3
    self.y4 = y4

    self.lines.append(Line(x1, 0.5, x2, 1 - y2))
    self.lines.append(Line(x1, 0.5, x2, y2))

    self.lines.append(Line(x2, 1 - y2, x3, 1 - y1))
    self.lines.append(Line(x2, y2, x3, y1))

    self.lines.append(Line(x3, 1 - y1, 1 - x3, 1 - y1))
    self.lines.append(Line(x3, y1, 1 - x3, y1))

    self.lines.append(Line(1 - x3, 1 - y1, 1 - x2, 1 - y2))
    self.lines.append(Line(1 - x3, y1, 1 - x2, y2))

    self.lines.append(Line(1 - x2, 1 - y2, 1 - x1, 0.5))
    self.lines.append(Line(1 - x2, y2, 1 - x1, 0.5))

    self.bounds = {
      'minX': x1,
      'maxX': 1 - x1,
      'minY': y1,
      'maxY': 1 - y1
    }

    self.linesUpper.append(Line(0, 1 + y4, 0, 10))
    self.linesUpper.append(Line(1, 1 + y4, 1, 10))
    self.linesUpper.append(Line(0, 1 + y4, 1, 1 + y4))
    self.boundsUpper = {
      'minX': 0,
      'maxX': 1,
      'minY': 1 + y4,
      'maxY': 10
    }

    self.linesLower = []
    self.linesLower.append(Line(0, -y3, 0, 10))
    self.linesLower.append(Line(1, -y3, 1, 10))
    self.linesLower.append(Line(0, -y3, 1, -y3))
    self.boundLower = {
      'minX': 0,
      'maxX': 1,
      'minY': -y3,
      'maxY': 10
    }

  def adjust(self, factor: float) -> MaskDecagon:
    if factor > 0:
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

class EyeDiagram:
  def __init__(self, waveforms: np.ndarray, waveformInfo,
               mask=None, resolution=2000) -> None:
    '''!@brief Create a new EyeDiagram from a collection of waveforms

    Assumes signal has two levels

    Waveforms must have same dimensions
    Waveforms must have same sampling period and units

    @param waveforms 3d array of waveforms [waveform0([[t0, t1,..., tn], [y0, y1,..., yn]]), waveform1,...]
    @param waveformInfo Dictionary of units ['tUnit', 'yUnit']:str and scales ['tIncr']:float
    @param mask Mask object for hit detection, None will not check a mask
    @param resolution Resolution of square eye diagram image
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
    self.calculated = False

  def calculate(self, verbose: bool = True, plot: bool = False,
                nThreads: int = 0) -> dict:
    '''!@brief Calculate and prepare eye diagram

    @param verbose True will print progress statements. False will not
    @param plot True will create plots during calculation (histograms and such). False will not
    @param multithreaded Specify number of threads to use, 0=all, 1=single, or n
    @return dict Report dictionary of measures and images, see getMeasures() and getImage()
    '''
    start = datetime.datetime.now()
    if nThreads == 0:
      nThreads = cpu_count()
    else:
      nThreads = min(nThreads, cpu_count())

    if verbose:
      print(
        f'{elapsedStr(start)} {Fore.GREEN}Starting calculation with {nThreads} threads')
      print(f'{elapsedStr(start)} {Fore.YELLOW}Finding threshold levels')
    self.__calculateLevels(plot=plot, nThreads=nThreads)
    if verbose:
      print(
        f'  Low:          {Fore.CYAN}{metricPrefix(self.yZero, self.yUnit)}')
      print(
        f'  Crossing:     {Fore.CYAN}{metricPrefix(self.thresholdHalf, self.yUnit)}')
      print(
        f'  High:         {Fore.CYAN}{metricPrefix(self.yOne, self.yUnit)}')

    if verbose:
      print(f'{elapsedStr(start)} {Fore.YELLOW}Determining bit period')
    self.__calculatePeriod(plot=plot, nThreads=nThreads)
    if verbose:
      print(
        f'  Bit period:   {Fore.CYAN}{metricPrefix(self.tBit, self.tUnit)}')
      print(
        f'  Duty cycle:   {Fore.CYAN}{self.tHigh / (2 * self.tBit) * 100:6.2f}%')
      print(
        f'  Frequency:    {Fore.CYAN}{metricPrefix(1/self.tBit, self.tUnit)}⁻¹')

    if verbose:
      print(f'{elapsedStr(start)} {Fore.GREEN}Complete')

  def __calculateLevels(self, plot: bool = True, nThreads: int = 0) -> None:
    '''!@brief Calculate high/low levels and crossing thresholds

    Assumes signal has two levels

    @param plot True will create plots during calculation (histograms and such). False will not
    @param multithreaded Specify number of threads to use, 0=all, 1=single, or n
    '''
    # Histogram vertical values for all waveforms
    minValue = np.amin(np.amin(self.waveforms, axis=2).T[1])
    maxValue = np.amax(np.amax(self.waveforms, axis=2).T[1])
    edges = np.linspace(minValue, maxValue, 101)

    with Pool(nThreads) as p:
      results = []
      for i in range(self.waveforms.shape[0]):
        results.append(p.apply_async(
            np.histogram,
            args=[self.waveforms[i][1], edges]))
      output = [p.get()[0] for p in results]

    counts = output[0]
    for i in range(1, len(output)):
      counts += output[i]
    bins = edges[:-1] + (edges[1] - edges[0]) / 2

    c = poly.polyfit(bins, counts, 4)
    fit = poly.polyval(bins, c)
    middleIndex = max(5, min(len(bins) - 5, argrelextrema(fit, np.less)[0][0]))

    binsZero = bins[:middleIndex]
    countsZero = counts[:middleIndex]
    binsOnes = bins[middleIndex:]
    countsOnes = counts[middleIndex:]

    optZero = fitGaussian(binsZero, countsZero)
    self.yZero = optZero[1]
    yZeroStdDev = abs(optZero[2])

    optOne = fitGaussian(binsOnes, countsOnes)
    self.yOne = optOne[1]
    yOneStdDev = abs(optOne[2])

    hys = 0.4
    self.thresholdRising = self.yOne * (1 - hys) + self.yZero * hys
    self.thresholdFalling = self.yOne * hys + self.yZero * (1 - hys)
    self.thresholdHalf = (self.yOne + self.yZero) / 2

    # Check high level is not close to low level
    errorStr = None
    if self.thresholdFalling < (
      self.yZero + yZeroStdDev) or self.thresholdRising > (self.yOne - yOneStdDev):
      errorStr = f'{Fore.RED}Too low signal to noise ratio{Fore.RESET}\n'
      errorStr += f'  Low:  {Fore.BLUE}{metricPrefix(self.yZero, self.yUnit)} σ={metricPrefix(yZeroStdDev, self.yUnit)}{Fore.RESET}\n'
      errorStr += f'  High: {Fore.BLUE}{metricPrefix(self.yOne, self.yUnit)} σ={metricPrefix(yOneStdDev, self.yUnit)}{Fore.RESET}'

    if plot:
      yRange = np.linspace(bins[0], bins[-1], 1000)
      pyplot.plot(bins, counts)
      pyplot.plot(yRange, gaussian(yRange, *optZero))
      pyplot.plot(yRange, gaussian(yRange, *optOne))
      pyplot.axvline(x=self.yZero, color='g')
      pyplot.axvline(x=self.yOne, color='g')
      pyplot.axvline(x=self.thresholdRising, color='r')
      pyplot.axvline(x=self.thresholdHalf, color='k')
      pyplot.axvline(x=self.thresholdFalling, color='r')
      pyplot.axvline(x=(self.yZero + yZeroStdDev), color='y')
      pyplot.axvline(x=(self.yZero - yZeroStdDev), color='y')
      pyplot.axvline(x=(self.yOne + yOneStdDev), color='y')
      pyplot.axvline(x=(self.yOne - yOneStdDev), color='y')

      ax = pyplot.gca()

      def tickFormatterY(y, pos):
        return metricPrefix(y, self.yUnit)
      formatterY = FuncFormatter(tickFormatterY)
      ax.xaxis.set_major_formatter(formatterY)

      pyplot.xlabel('Vertical scale')
      pyplot.ylabel('Counts')
      pyplot.title('Vertical levels')

      pyplot.show()

    if errorStr:
      raise Exception(errorStr)

  def __calculatePeriod(self, plot: bool = True, nThreads: int = 0) -> None:
    '''!@brief Calculate period for a single bit

    Assumes shortest period is a single bit

    @param plot True will create plots during calculation (histograms and such). False will not
    @param multithreaded Specify number of threads to use, 0=all, 1=single, or n
    '''
    # Histogram horizontal periods for all waveforms
    with Pool(nThreads) as p:
      results = []
      for i in range(self.waveforms.shape[0]):
        x = self.waveforms[i][0].tolist()
        y = self.waveforms[i][1].tolist()
        results.append(p.apply_async(
            getEdges,
            args=[x,
                  y,
                  self.thresholdRising,
                  self.thresholdHalf,
                  self.thresholdFalling]))
      waveformEdges = [p.get() for p in results]

    periods = []
    for edgesRising, edgesFalling in waveformEdges:
      for i in range(1, len(edgesRising)):
        duration = edgesRising[i] - edgesRising[i - 1]
        periods.append(duration)

      for i in range(1, len(edgesFalling)):
        duration = edgesFalling[i] - edgesFalling[i - 1]
        periods.append(duration)

    # Step 1 find rough period from just the smallest pulses
    bitDuration = min(periods) / 2
    durationThreshold = 2.5 * bitDuration

    periodsSmallest = []
    for p in periods:
      if p > durationThreshold:
        continue
      periodsSmallest.append(p / 2)

    bitDuration = np.average(periodsSmallest)
    durationThreshold = 2.5 * bitDuration

    # Step 2 include bits of all lengths
    periodsAdjusted = []
    for p in periods:
      if p > durationThreshold:
        nBits = int(round(p / bitDuration))
        p = p - bitDuration * (nBits - 2)
      periodsAdjusted.append(p / 2)

    minValue = min(periodsAdjusted)
    maxValue = max(periodsAdjusted)
    edges = np.linspace(minValue, maxValue, 101)
    countsPeriod, _ = np.histogram(periodsAdjusted, edges)
    binsPeriod = edges[:-1] + (edges[1] - edges[0]) / 2

    optPeriod = fitGaussian(binsPeriod, countsPeriod)
    self.tBit = optPeriod[1]
    self.tBitStdDev = abs(optPeriod[2])

    # Get high and low timing offsets
    durationsHigh = []
    durationsLow = []
    for edgesRising, edgesFalling in waveformEdges:
      for i in range(1, min(len(edgesRising), len(edgesFalling))):
        duration = edgesRising[i] - edgesFalling[i]
        if duration > 0:
          durationsLow.append(duration)

          duration = edgesFalling[i] - edgesRising[i - 1]
          durationsHigh.append(duration)
        else:
          duration = -duration
          durationsHigh.append(duration)

          duration = edgesRising[i] - edgesFalling[i - 1]
          durationsLow.append(duration)

    binsLow, countsLow = binExact(durationsLow)
    binsHigh, countsHigh = binExact(durationsHigh)

    nBits = 1
    durationsLow = []
    for i in range(len(binsLow)):
      if i > 0 and (binsLow[i] - binsLow[i - 1]) > (0.5 * self.tBit):
        nBits += 1
      duration = binsLow[i] - self.tBit * (nBits - 1)
      durationsLow.extend([duration] * countsLow[i])

    nBits = 1
    durationsHigh = []
    for i in range(len(binsHigh)):
      if i > 0 and (binsHigh[i] - binsHigh[i - 1]) > (0.5 * self.tBit):
        nBits += 1
      duration = binsHigh[i] - self.tBit * (nBits - 1)
      durationsHigh.extend([duration] * countsHigh[i])

    minValue = min(durationsLow)
    maxValue = max(durationsLow)
    edges = np.linspace(minValue, maxValue, 101)
    countsLow, _ = np.histogram(durationsLow, edges)
    binsLow = edges[:-1] + (edges[1] - edges[0]) / 2

    minValue = min(durationsHigh)
    maxValue = max(durationsHigh)
    edges = np.linspace(minValue, maxValue, 101)
    countsHigh, _ = np.histogram(durationsHigh, edges)
    binsHigh = edges[:-1] + (edges[1] - edges[0]) / 2

    optLow = fitGaussian(binsLow, countsLow)
    optHigh = fitGaussian(binsHigh, countsHigh)
    self.tLow = optLow[1]
    tLowStdDev = abs(optLow[2])
    self.tHigh = optHigh[1]
    tHighStdDev = abs(optHigh[2])

    if plot:
      def tickFormatterX(x, _):
        return metricPrefix(x, self.tUnit)
      formatterX = FuncFormatter(tickFormatterX)

      yRange = np.linspace(min(binsLow[0], binsHigh[0]), max(
        binsLow[-1], binsHigh[-1]), 1000)
      _, subplots = pyplot.subplots(3, 1, sharex=True)
      subplots[0].set_title('Bit period')
      subplots[0].plot(binsPeriod, countsPeriod)
      subplots[0].plot(yRange, gaussian(yRange, *optPeriod))
      subplots[0].axvline(x=self.tBit, color='g')
      subplots[0].axvline(x=(self.tBit + self.tBitStdDev), color='y')
      subplots[0].axvline(x=(self.tBit - self.tBitStdDev), color='y')

      subplots[1].set_title('Low period')
      subplots[1].plot(binsLow, countsLow)
      subplots[1].plot(yRange, gaussian(yRange, *optLow))
      subplots[1].axvline(x=self.tLow, color='g')
      subplots[1].axvline(x=(self.tLow + tLowStdDev), color='y')
      subplots[1].axvline(x=(self.tLow - tLowStdDev), color='y')
      subplots[1].axvline(x=self.tBit, color='r')

      subplots[2].set_title('High period')
      subplots[2].plot(binsHigh, countsHigh)
      subplots[2].plot(yRange, gaussian(yRange, *optHigh))
      subplots[2].axvline(x=self.tHigh, color='g')
      subplots[2].axvline(x=(self.tHigh - tHighStdDev), color='y')
      subplots[2].axvline(x=(self.tHigh + tHighStdDev), color='y')
      subplots[2].axvline(x=self.tBit, color='r')

      subplots[0].xaxis.set_major_formatter(formatterX)
      subplots[1].xaxis.set_major_formatter(formatterX)
      subplots[2].xaxis.set_major_formatter(formatterX)

      subplots[-1].set_xlabel('Time')
      subplots[0].set_ylabel('Counts')
      subplots[1].set_ylabel('Counts')
      subplots[2].set_ylabel('Counts')

      pyplot.show()
