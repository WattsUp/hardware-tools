from __future__ import annotations
import colorama
from colorama import Fore
import datetime
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as pyplot
from matplotlib.ticker import FuncFormatter
from multiprocessing import Pool, cpu_count
import numpy as np
import numpy.polynomial.polynomial as poly
import os
from PIL import Image
from scipy.signal import argrelextrema
import skimage.draw

import sys

from ..math import *
from .extension import getEdgesNumpy, getCrossing, getHits, isHitting
from . import brescount

colorama.init(autoreset=True)

class Mask:
  def __init__(self) -> None:
    '''!@brief Create a new Mask
    Derrive mask shapes from this class such as a MaskOctagon, MaskDecagon, MaskPulse, etc.
    '''
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
  def __init__(self, waveforms: np.ndarray, waveformInfo: dict, mask: Mask = None, resolution: int = 2000,
               nBitsMax: int = 5, method: str = 'average', resample: int = 50, yLevels: list = None) -> None:
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
    self.method = method
    self.resample = resample
    self.manualYLevels = yLevels
    self.calculated = False
    self.hysteresis = 0.1

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
    # if verbose:
    #   print(
    #     f'  Low:          {Fore.CYAN}{metricPrefix(self.yZero, self.yUnit)}')
    #   print(
    #     f'  Crossing:     {Fore.CYAN}{metricPrefix(self.thresholdHalf, self.yUnit)}')
    #   print(
    #     f'  High:         {Fore.CYAN}{metricPrefix(self.yOne, self.yUnit)}')

    if verbose:
      print(f'{elapsedStr(start)} {Fore.YELLOW}Determining bit period')
    self.__calculatePeriod(plot=plot, nThreads=nThreads)
    # if verbose:
    #   print(
    #     f'  Bit period:   {Fore.CYAN}{metricPrefix(self.tBit, self.tUnit)}')
    #   print(
    #     f'  Duty cycle:   {Fore.CYAN}{self.tHigh / (2 * self.tBit) * 100:6.2f}%')
    #   print(
    # f'  Frequency:    {Fore.CYAN}{metricPrefix(1/self.tBit, self.tUnit)}⁻¹')

    if verbose:
      print(f'{elapsedStr(start)} {Fore.YELLOW}Extracting bit centers')
    self.__extractBitCenters(plot=plot, nThreads=nThreads)
    # if verbose:
    #   print(f'  Number of bits: {Fore.CYAN}{self.nBits}')

    self.calculated = True
    if verbose:
      print(f'{elapsedStr(start)} {Fore.YELLOW}Measuring waveform')
    self.__measureWaveform(plot=plot, printProgress=verbose, nThreads=nThreads)
    if verbose:
      self.printMeasures()
    self.calculated = False

    if verbose:
      print(f'{elapsedStr(start)} {Fore.YELLOW}Generating images')
    self.__generateImages(printProgress=verbose, nThreads=nThreads)

    if verbose:
      print(f'{elapsedStr(start)} {Fore.GREEN}Complete')

    self.calculated = True

  def __calculateLevels(self, plot: bool = True, nThreads: int = 1) -> None:
    '''!@brief Calculate high/low levels and crossing thresholds

    Assumes signal has two levels

    @param plot True will create plots during calculation (histograms and such). False will not
    @param nThreads Specify number of threads to use
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
    if self.manualYLevels:
      self.yZero = self.manualYLevels[0]
      yZeroStdDev = 0
    else:
      self.yZero = optZero[1]
      yZeroStdDev = abs(optZero[2])

    optOne = fitGaussian(binsOnes, countsOnes)
    if self.manualYLevels:
      self.yOne = self.manualYLevels[1]
      yOneStdDev = 0
    else:
      self.yOne = optOne[1]
      yOneStdDev = abs(optOne[2])

    hys = 0.5 - self.hysteresis
    self.thresholdRise = self.yOne * (1 - hys) + self.yZero * hys
    self.thresholdFall = self.yOne * hys + self.yZero * (1 - hys)
    self.thresholdHalf = (self.yOne + self.yZero) / 2

    # Check high level is not close to low level
    errorStr = None
    if self.thresholdFall < (
      self.yZero + yZeroStdDev) or self.thresholdRise > (self.yOne - yOneStdDev):
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
      pyplot.ylabel('Counts')
      pyplot.title('Vertical levels')

      pyplot.show()

    if errorStr:
      raise Exception(errorStr)

  def __calculatePeriod(self, plot: bool = True, nThreads: int = 1) -> None:
    '''!@brief Calculate period for a single bit

    Assumes shortest period is a single bit

    @param plot True will create plots during calculation (histograms and such). False will not
    @param nThreads Specify number of threads to use
    '''
    # Histogram horizontal periods for all waveforms
    with Pool(nThreads) as p:
      results = []
      for i in range(self.waveforms.shape[0]):
        results.append(p.apply_async(
            getEdgesNumpy,
            args=[self.waveforms[i],
                  self.thresholdRise,
                  self.thresholdHalf,
                  self.thresholdFall]))
      self.waveformEdges = [p.get() for p in results]

    periods = []
    for edgesRise, edgesFall in self.waveformEdges:
      for i in range(1, len(edgesRise)):
        duration = edgesRise[i] - edgesRise[i - 1]
        periods.append(duration)

      for i in range(1, len(edgesFall)):
        duration = edgesFall[i] - edgesFall[i - 1]
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
        nBits = int((p / bitDuration) + 0.5)
        p = p - bitDuration * (nBits - 2)
      periodsAdjusted.append(p / 2)

    binsPeriod, countsPeriod = binLinear(periodsAdjusted, 100)
    optPeriod = fitGaussian(binsPeriod, countsPeriod)
    self.tBit = optPeriod[1]
    self.tBitStdDev = abs(optPeriod[2])

    # Get high and low timing offsets
    durationsHigh = []
    durationsLow = []
    for edgesRise, edgesFall in self.waveformEdges:
      for i in range(1, min(len(edgesRise), len(edgesFall))):
        duration = edgesRise[i] - edgesFall[i]
        if duration > 0:
          durationsLow.append(duration)

          duration = edgesFall[i] - edgesRise[i - 1]
          durationsHigh.append(duration)
        else:
          duration = -duration
          durationsHigh.append(duration)

          duration = edgesRise[i] - edgesFall[i - 1]
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

    binsLow, countsLow = binLinear(durationsLow, 100)
    binsHigh, countsHigh = binLinear(durationsHigh, 100)

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

  def __extractBitCenters(self, plot: bool = True, nThreads: int = 1) -> None:
    '''!@brief Extract center positions of bits

    @param plot True will create plots during calculation (histograms and such). False will not
    @param nThreads Specify number of threads to use
    '''
    # Padding to artificially add to pulses to meet 50% duty
    tHighOffset = self.tBit - self.tHigh
    tLowOffset = self.tBit - self.tLow

    # Histogram horizontal periods for all waveforms
    with Pool(nThreads) as p:
      results = [p.apply_async(
        _runnerBitExtract,
        args=[
            self.waveformEdges[i],
            self.waveforms[i][0][0],
            self.tDelta,
            self.tBit,
            tHighOffset,
            tLowOffset,
            self.nBitsMax]) for i in range(self.waveforms.shape[0])]
      output = [p.get() for p in results]

    # output = [
    #     _runnerBitExtract(
    #         self.waveformEdges[i],
    #         self.waveforms[i][0][0],
    #         self.tDelta,
    #         self.tBit,
    #         tHighOffset,
    #         tLowOffset,
    #         self.nBitsMax) for i in range(self.waveforms.shape[0])]

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
                        printProgress: bool = True, nThreads: int = 1) -> None:
    '''!@brief Extract center positions of bits

    @param plot True will create plots during calculation (histograms and such). False will not
    @param printProgress True will print progress statements. False will not
    @param nThreads Specify number of threads to use
    '''
    self.measures = {}
    m = self.measures

    start = datetime.datetime.now()
    if printProgress:
      print(f'  {elapsedStr(start)} {Fore.GREEN}Starting measurement')

    nBits = 0
    for b in self.bitCentersT:
      nBits += len(b)
    m['nBits'] = nBits

    if printProgress:
      print(
        f'  {elapsedStr(start)} {Fore.YELLOW}Measuring \'0\', \'0.5\', \'1\' values')

    with Pool(nThreads) as p:
      results = [p.apply_async(
        _runnerCollectValuesY,
        args=[
            self.waveforms[i][1],
            self.bitCentersT[i],
            self.bitCentersY[i],
            self.tDelta,
            self.tBit,
            self.thresholdHalf]) for i in range(self.waveforms.shape[0])]
      output = [p.get() for p in results]

    # output = [
    #     _runnerCollectValuesY(
    #         self.waveforms[i][1],
    #         self.bitCentersT[i],
    #         self.bitCentersY[i],
    #         self.tDelta,
    #         self.tBit,
    #         self.thresholdHalf) for i in range(self.waveforms.shape[0])]

    valuesY = {
      'zero': [],
      'cross': [],
      'one': []
    }

    for o in output:
      for k, v in o.items():
        valuesY[k].extend(v)

    if printProgress:
      print(
        f'  {elapsedStr(start)} {Fore.YELLOW}Computing \'0\', \'0.5\', \'1\' statistics')

    if self.method == 'peak' or plot:
      componentsZero = fitGaussianMix(valuesY['zero'], nMax=3)
      if printProgress:
        print(
          f'  {elapsedStr(start)} {Fore.CYAN}yZero has {len(componentsZero)} modes')

      componentsCross = fitGaussianMix(valuesY['cross'], nMax=3)
      if printProgress:
        print(
          f'  {elapsedStr(start)} {Fore.CYAN}yCross has {len(componentsCross)} modes')

      componentsOne = fitGaussianMix(valuesY['one'], nMax=3)
      if printProgress:
        print(
          f'  {elapsedStr(start)} {Fore.CYAN}yOne has {len(componentsOne)} modes')

    if self.method == 'peak':
      m['yZero'] = gaussianMixCenter(componentsZero)
      m['yCrossing'] = gaussianMixCenter(componentsCross)
      m['yOne'] = gaussianMixCenter(componentsOne)
    elif self.method == 'average':
      m['yZero'] = np.average(valuesY['zero'])
      m['yCrossing'] = np.average(valuesY['cross'])
      m['yOne'] = np.average(valuesY['one'])
    else:
      raise Exception(f'Unrecognized measure method: {self.method}')
    m['yZeroStdDev'] = np.std(valuesY['zero'])
    m['yCrossingStdDev'] = np.std(valuesY['cross'])
    m['yOneStdDev'] = np.std(valuesY['one'])

    m['eyeAmplitude'] = m['yOne'] - m['yZero']
    m['eyeAmplitudeStdDev'] = np.sqrt(m['yOneStdDev']**2 + m['yZeroStdDev']**2)

    m['snr'] = (m['eyeAmplitude']) / (m['yOneStdDev'] + m['yZeroStdDev'])

    m['eyeHeight'] = (m['yOne'] - 3 * m['yOneStdDev']) - \
        (m['yZero'] + 3 * m['yZeroStdDev'])
    m['eyeHeightP'] = 100 * m['eyeHeight'] / (m['eyeAmplitude'])

    m['yCrossingP'] = 100 * (m['yCrossing'] - m['yZero']
                             ) / (m['eyeAmplitude'])
    m['yCrossingPStdDev'] = 100 * \
        (m['yCrossingStdDev'] - m['yZero']) / (m['eyeAmplitude'])

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
      subplots[1].axvline(x=m["yCrossing"], color='g')
      subplots[1].axvline(x=(m["yCrossing"] - m["yCrossingStdDev"]), color='y')
      subplots[1].axvline(x=(m["yCrossing"] + m["yCrossingStdDev"]), color='y')

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

    if printProgress:
      print(f'  {elapsedStr(start)} {Fore.YELLOW}Measuring edges values')

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
            self.yOne]) for i in range(self.waveforms.shape[0])]
      output = [p.get() for p in results]

    # output = [
    #     _runnerCollectValuesT(
    #         self.waveforms[i][1],
    #         self.bitCentersT[i],
    #         self.bitCentersY[i],
    #         self.tDelta,
    #         self.tBit,
    #         m['yZero'],
    #         m['eyeAmplitude']) for i in range(self.waveforms.shape[0])]

    valuesT = {
        'rise20': [],
        'rise50': [],
        'rise80': [],
        'fall20': [],
        'fall50': [],
        'fall80': []
    }

    for o in output:
      for k, v in o.items():
        valuesT[k].extend(v)

    if printProgress:
      print(f'  {elapsedStr(start)} {Fore.YELLOW}Computing edges statistics')

    if self.method == 'peak' or plot:
      componentsRise20 = fitGaussianMix(valuesT['rise20'], nMax=3)
      if printProgress:
        print(
          f'  {elapsedStr(start)} {Fore.CYAN}tRise20 has {len(componentsRise20)} modes')

      componentsRise50 = fitGaussianMix(valuesT['rise50'], nMax=3)
      if printProgress:
        print(
          f'  {elapsedStr(start)} {Fore.CYAN}tRise50 has {len(componentsRise50)} modes')

      componentsRise80 = fitGaussianMix(valuesT['rise80'], nMax=3)
      if printProgress:
        print(
          f'  {elapsedStr(start)} {Fore.CYAN}tRise80 has {len(componentsRise80)} modes')

      componentsFall20 = fitGaussianMix(valuesT['fall20'], nMax=3)
      if printProgress:
        print(
          f'  {elapsedStr(start)} {Fore.CYAN}tFall20 has {len(componentsFall20)} modes')

      componentsFall50 = fitGaussianMix(valuesT['fall50'], nMax=3)
      if printProgress:
        print(
          f'  {elapsedStr(start)} {Fore.CYAN}tFall50 has {len(componentsFall50)} modes')

      componentsFall80 = fitGaussianMix(valuesT['fall80'], nMax=3)
      if printProgress:
        print(
          f'  {elapsedStr(start)} {Fore.CYAN}tFall80 has {len(componentsFall80)} modes')

    if self.method == 'peak':
      m["tRise20"] = gaussianMixCenter(componentsRise20)
      m["tRise50"] = gaussianMixCenter(componentsRise50)
      m["tRise80"] = gaussianMixCenter(componentsRise80)
      m["tFall20"] = gaussianMixCenter(componentsFall20)
      m["tFall50"] = gaussianMixCenter(componentsFall50)
      m["tFall80"] = gaussianMixCenter(componentsFall80)
    elif self.method == 'average':
      m["tRise20"] = np.average(valuesT['rise20'])
      m["tRise50"] = np.average(valuesT['rise50'])
      m["tRise80"] = np.average(valuesT['rise80'])
      m["tFall20"] = np.average(valuesT['fall20'])
      m["tFall50"] = np.average(valuesT['fall50'])
      m["tFall80"] = np.average(valuesT['fall80'])
    else:
      raise Exception(f'Unrecognized measure method: {self.method}')
    m["tRise20StdDev"] = np.std(valuesT['rise20'])
    m["tRise50StdDev"] = np.std(valuesT['rise50'])
    m["tRise80StdDev"] = np.std(valuesT['rise80'])
    m["tFall20StdDev"] = np.std(valuesT['fall20'])
    m["tFall50StdDev"] = np.std(valuesT['fall50'])
    m["tFall80StdDev"] = np.std(valuesT['fall80'])

    m["tRise"] = m["tRise80"] - m["tRise20"]
    m["tRiseStdDev"] = np.sqrt(m["tRise80StdDev"]**2 + m["tRise20StdDev"]**2)
    m["tFall"] = m["tFall20"] - m["tFall80"]
    m["tFallStdDev"] = np.sqrt(m["tFall20StdDev"]**2 + m["tFall80StdDev"]**2)

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

    m["tJitterPP"] = max(np.max(valuesT['rise50']) - np.min(valuesT['fall50']),
                         np.max(valuesT['fall50']) - np.min(valuesT['rise50']))
    m["tJitterRMS"] = np.sqrt(m["tFall50StdDev"]**2 + m["tFall50StdDev"]**2)

    eyeWidthLow = (m["tRise50"] + m["tBit"] - 3 * m["tRise50StdDev"]
                   ) - (m["tFall50"] + 3 * m["tFall50StdDev"])
    eyeWidthHigh = (m["tFall50"] + m["tBit"] - 3 * m["tFall50StdDev"]
                    ) - (m["tRise50"] + 3 * m["tRise50StdDev"])
    m["eyeWidth"] = min(eyeWidthLow, eyeWidthHigh)
    m["eyeWidthP"] = 100 * m["eyeWidth"] / m["tBit"]

    edgeDifference = (m["tRise20"] + m["tRise80"]) / 2 - \
        (m["tFall20"] + m["tFall80"]) / 2
    edgeDifferenceStdDev = np.sqrt(
        0.25 *
        m["tRise20StdDev"]**2 +
        0.25 *
        m["tRise80StdDev"]**2 +
        0.25 *
        m["tFall20StdDev"]**2 +
        0.25 *
        m["tFall80StdDev"]**2)
    m["dutyCycleDistortion"] = 100 * edgeDifference / m["tBit"]
    m["dutyCycleDistortionStdDev"] = abs(m["dutyCycleDistortion"]) * np.sqrt(
      (edgeDifferenceStdDev / edgeDifference) ** 2 + (m["tBitStdDev"] / m["tBit"]) ** 2)

    if plot:
      def tickFormatterX(x, _):
        return metricPrefix(x, self.tUnit)
      formatterX = FuncFormatter(tickFormatterX)

      def tickFormatterY(y, _):
        return metricPrefix(y, self.yUnit)
      formatterY = FuncFormatter(tickFormatterY)

      xRange = np.linspace(-m["tBit"], 0, 1000)
      _, subplots = pyplot.subplots(3, 1, sharex=True)
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
      x01 = list(np.linspace(-self.tBit, 2 * m["tRise20"] - m["tRise50"], 10))
      x01.extend([m["tRise20"], m["tRise50"], m["tRise80"]])
      x01.extend(list(np.linspace(2 * m["tRise80"] - m["tRise50"], 0, 10)))
      y01 = [m["yZero"]] * 10
      y01.extend([m["yZero"] + 0.2 * m["eyeAmplitude"],
                  m["yZero"] + 0.5 * m["eyeAmplitude"],
                  m["yZero"] + 0.8 * m["eyeAmplitude"]])
      y01.extend([m["yOne"]] * 10)
      # f01 = interp1d(x01, y01, kind='cubic')
      subplots[2].plot(x01, y01, color='b')
      # subplots[2].plot(xRange, f01(xRange), color='b', linestyle=':')
      x10 = list(np.linspace(-self.tBit, 2 * m["tFall80"] - m["tFall50"], 10))
      x10.extend([m["tFall80"], m["tFall50"], m["tFall20"]])
      x10.extend(list(np.linspace(2 * m["tFall20"] - m["tFall50"], 0, 10)))
      y10 = [m["yOne"]] * 10
      y10.extend([m["yZero"] + 0.8 * m["eyeAmplitude"],
                  m["yZero"] + 0.5 * m["eyeAmplitude"],
                  m["yZero"] + 0.2 * m["eyeAmplitude"]])
      y10.extend([m["yZero"]] * 10)
      # f10 = interp1d(x10, y10, kind='cubic')
      subplots[2].plot(x10, y10, color='g')
      # subplots[2].plot(xRange, f10(xRange), color='g', linestyle=':')

      subplots[0].xaxis.set_major_formatter(formatterX)
      subplots[1].xaxis.set_major_formatter(formatterX)
      subplots[2].xaxis.set_major_formatter(formatterX)
      subplots[2].yaxis.set_major_formatter(formatterY)

      subplots[-1].set_xlabel('Time')
      subplots[0].set_ylabel('Counts')
      subplots[1].set_ylabel('Counts')
      subplots[2].set_ylabel('Vertical Scale')

      pyplot.show()

    if self.mask:
      if printProgress:
        print(f'  {elapsedStr(start)} {Fore.YELLOW}Checking mask for hits')

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

      # output = [
      #     _runnerCollectMaskHits(
      #         self.waveforms[i][1],
      #         self.bitCentersT[i],
      #         self.bitCentersY[i],
      #         self.tDelta,
      #         self.tBit,
      #         m['yZero'],
      #         m['eyeAmplitude'],
      #         self.mask,
      #         self.resample) for i in range(self.waveforms.shape[0])]

      self.offenders = []
      self.hits = []
      offenderCount = 0
      for o in output:
        self.offenders.append(o[0])
        offenderCount += len(o[0])
        self.hits.extend(o[1])

      m['offenderCount'] = offenderCount
      m['ber'] = m['offenderCount'] / m['nBits']

      if printProgress:
        print(f'  {elapsedStr(start)} {Fore.YELLOW}Finding mask margin')

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

      # output = [
      #     _runnerAdjustMaskMargin(
      #         self.waveforms[i][1],
      #         self.bitCentersT[i],
      #         self.bitCentersY[i],
      #         self.tDelta,
      #         self.tBit,
      #         m['yZero'],
      #         m['eyeAmplitude'],
      #         self.mask,
      #         self.resample,
      #         self.offenders[i]) for i in range(self.waveforms.shape[0])]

      output = [(output[i], i) for i in range(len(output))]
      output = sorted(output, key=lambda o: o[0])
      self.maskMarginPair = output[0]
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

            if factor > 0:
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

        if factor > 0:
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
      print(f'  {elapsedStr(start)} {Fore.GREEN}Complete measurement')

  def __generateImages(self, printProgress: bool = True,
                       nThreads: int = 1) -> None:
    '''!@brief Extract center positions of bits

    @param printProgress True will print progress statements. False will not
    @param nThreads Specify number of threads to use
    '''
    start = datetime.datetime.now()
    if printProgress:
      print(f'  {elapsedStr(start)} {Fore.GREEN}Starting image generation')

    heatMap = np.zeros((self.resolution, self.resolution), dtype=np.int32)

    if printProgress:
      print(f'  {elapsedStr(start)} {Fore.YELLOW}Layering bit waveforms')

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

    # output = [
    #     _runnerGenerateHeatMap(
    #         self.waveforms[i][1],
    #         self.bitCentersT[i],
    #         self.bitCentersY[i],
    #         self.tDelta,
    #         self.tBit,
    #         self.yZero,
    #         self.yOne,
    #         self.resample,
    #         self.resolution) for i in range(self.waveforms.shape[0])]

    for o in output:
      heatMap += o

    if printProgress:
      print(f'  {elapsedStr(start)} {Fore.YELLOW}Transforming into heatmap')

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
      print(f'  {elapsedStr(start)} {Fore.YELLOW}Drawing grid')

    imageMin = -0.5
    imageMax = 1.5
    zero = int(((0 - imageMin) / (imageMax - imageMin)) * self.resolution)
    half = int(((0.5 - imageMin) / (imageMax - imageMin)) * self.resolution)
    one = int(((1 - imageMin) / (imageMax - imageMin)) * self.resolution)

    # Draw a grid for reference levels
    npImageGrid = np.zeros(npImageClean.shape, dtype=npImageClean.dtype)
    npImageGrid[:, :, 0: 2] = 1.0
    npImageGrid[zero, :, 3] = 1.0
    npImageGrid[half, ::4, 3] = 1.0
    npImageGrid[one, :, 3] = 1.0
    npImageGrid[:, zero, 3] = 1.0
    npImageGrid[::4, half, 3] = 1.0
    npImageGrid[:, one, 3] = 1.0
    for i in range(1, 8, 2):
      pos = int(i / 8 * self.resolution)
      npImageGrid[pos, ::10, 3] = 1.0
      npImageGrid[::10, pos, 3] = 1.0
    npImageGrid = np.flip(npImageGrid, axis=0)

    npImage = layerNumpyImageRGBA(npImageClean, npImageGrid)
    self.imageGrid = Image.fromarray((255 * npImage).astype('uint8'))

    if self.mask:
      npImageMargin = npImage.copy()

      if printProgress:
        print(f'  {elapsedStr(start)} {Fore.YELLOW}Drawing mask')

      # Draw mask
      npImageMask = np.zeros(npImageClean.shape, dtype=npImageClean.dtype)
      for path in self.mask.paths:
        x = [((p[0] - imageMin) / (imageMax - imageMin))
             * self.resolution for p in path]
        y = [((p[1] - imageMin) / (imageMax - imageMin))
             * self.resolution for p in path]
        x = [max(0, min(self.resolution - 1, v)) for v in x]
        y = [max(0, min(self.resolution - 1, v)) for v in y]
        polygon = skimage.draw.polygon(y, x)
        npImageMask[polygon] = [1.0, 0.0, 1.0, 0.5]
      npImageMask = np.flip(npImageMask, axis=0)

      npImage = layerNumpyImageRGBA(npImage, npImageMask)
      self.imageMask = Image.fromarray((255 * npImage).astype('uint8'))

      if printProgress:
        print(
          f'  {elapsedStr(start)} {Fore.YELLOW}Drawing hits and subset of offending bit waveforms')

      # Draw hits and offending bit waveforms
      npImageHits = np.zeros(npImageClean.shape, dtype=npImageClean.dtype)
      npImageHits[:, :, 0] = 1.0
      for h in self.hits:
        x = int(((h[0] - imageMin) / (imageMax - imageMin)) * self.resolution)
        y = int(((h[1] - imageMin) / (imageMax - imageMin)) * self.resolution)
        x = max(0, min(self.resolution - 1, x))
        y = max(0, min(self.resolution - 1, y))
        radius = int(max(3, self.resolution / 500))
        rr, cc = skimage.draw.circle_perimeter(y, x, radius)
        npImageHits[rr, cc, 3] = 1

      # Plot subset of offenders
      for i in range(len(self.offenders)):
        self.__drawBits(npImageHits, i, self.offenders[i][:1])

      npImageHits = np.flip(npImageHits, axis=0)

      npImage = layerNumpyImageRGBA(npImage, npImageHits)
      self.imageHits = Image.fromarray((255 * npImage).astype('uint8'))

      if printProgress:
        print(f'  {elapsedStr(start)} {Fore.YELLOW}Drawing mask at margin')

      # Draw margin mask
      npImageMask = np.zeros(npImageClean.shape, dtype=npImageClean.dtype)
      for path in self.mask.adjust(self.maskMarginPair[0][0]).paths:
        x = [((p[0] - imageMin) / (imageMax - imageMin))
             * self.resolution for p in path]
        y = [((p[1] - imageMin) / (imageMax - imageMin))
             * self.resolution for p in path]
        x = [max(0, min(self.resolution - 1, v)) for v in x]
        y = [max(0, min(self.resolution - 1, v)) for v in y]
        polygon = skimage.draw.polygon(y, x)
        npImageMask[polygon] = [1.0, 0.0, 1.0, 0.5]
      npImageMask = np.flip(npImageMask, axis=0)

      npImageMargin = layerNumpyImageRGBA(npImageMargin, npImageMask)

      if printProgress:
        print(
          f'  {elapsedStr(start)} {Fore.YELLOW}Drawing worst offending bit waveform')

      # Draw hits and offending bit waveforms
      npImageHits = np.zeros(npImageClean.shape, dtype=npImageClean.dtype)
      npImageHits[:, :, 0] = 1.0
      for h in self.maskMarginPair[0][2]:
        x = int(((h[0] - imageMin) / (imageMax - imageMin)) * self.resolution)
        y = int(((h[1] - imageMin) / (imageMax - imageMin)) * self.resolution)
        x = max(0, min(self.resolution - 1, x))
        y = max(0, min(self.resolution - 1, y))
        radius = int(max(3, self.resolution / 500))
        rr, cc = skimage.draw.circle_perimeter(y, x, radius)
        npImageHits[rr, cc, 3] = 1

      # Plot subset worst offender
      self.__drawBits(
          npImageHits, self.maskMarginPair[1], [
              self.maskMarginPair[0][1]])

      npImageHits = np.flip(npImageHits, axis=0)

      npImageMargin = layerNumpyImageRGBA(npImageMargin, npImageHits)
      self.imageMargin = Image.fromarray((255 * npImageMargin).astype('uint8'))

    else:
      self.imageMask = None
      self.imageHits = None
      self.imageMargin = None

    if printProgress:
      print(f'  {elapsedStr(start)} {Fore.GREEN}Complete image generation')

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

      if factor > 0:
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
        if t[ii - 1] < 0 or t[ii] > self.resolution:
          continue
        if min(y[ii], y[ii - 1]) < 0 or max(y[ii],
                                            y[ii - 1]) > self.resolution:
          continue
        rr, cc, val = skimage.draw.line_aa(y[ii], t[ii], y[ii - 1], t[ii - 1])
        image[rr, cc, 3] = val + (1 - val) * image[rr, cc, 3]

  def printMeasures(self) -> None:
    '''!@brief Print measures to console'''
    if not self.calculated:
      raise Exception(
        'Eye diagram must be calculated before printing measures')
    print(
      f'  yZero:        {Fore.CYAN}{metricPrefix(self.measures["yZero"], self.yUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["yZeroStdDev"], self.yUnit)}')
    print(
      f'  yCrossing:     {Fore.CYAN}{self.measures["yCrossingP"]:6.2f} %   {Fore.BLUE}σ= {self.measures["yCrossingPStdDev"]:6.2f} %')
    print(
      f'  yOne:         {Fore.CYAN}{metricPrefix(self.measures["yOne"], self.yUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["yOneStdDev"], self.yUnit)}')
    print(f'  SNR:           {Fore.CYAN}{self.measures["snr"]:6.2f}')
    print(
      f'  eyeAmplitude: {Fore.CYAN}{metricPrefix(self.measures["eyeAmplitude"], self.yUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["eyeAmplitudeStdDev"], self.yUnit)}')
    print(
      f'  eyeHeight:    {Fore.CYAN}{metricPrefix(self.measures["eyeHeight"], self.yUnit)}      {Fore.BLUE}{self.measures["eyeHeightP"]:6.2f} % ')
    print(
      f'  tBit:         {Fore.CYAN}{metricPrefix(self.measures["tBit"], self.tUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["tBitStdDev"], self.tUnit)}')
    print(
      f'  fBit:         {Fore.CYAN}{metricPrefix(self.measures["fBit"], self.tUnit)}⁻¹ {Fore.BLUE}σ={metricPrefix(self.measures["fBitStdDev"], self.tUnit)}⁻¹')
    print(
      f'  tLow:         {Fore.CYAN}{metricPrefix(self.measures["tLow"], self.tUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["tLowStdDev"], self.tUnit)}')
    print(
      f'  tHigh:        {Fore.CYAN}{metricPrefix(self.measures["tHigh"], self.tUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["tHighStdDev"], self.tUnit)}')
    print(
      f'  tRise:        {Fore.CYAN}{metricPrefix(self.measures["tRise"], self.tUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["tRiseStdDev"], self.tUnit)}')
    print(
      f'  tFall:        {Fore.CYAN}{metricPrefix(self.measures["tFall"], self.tUnit)}   {Fore.BLUE}σ={metricPrefix(self.measures["tFallStdDev"], self.tUnit)}')
    print(
      f'  tJitter:      {Fore.CYAN}{metricPrefix(self.measures["tJitterPP"], self.tUnit)}.pp  {Fore.BLUE}{metricPrefix(self.measures["tJitterRMS"], self.tUnit)}.rms')
    print(
      f'  eyeWidth:     {Fore.CYAN}{metricPrefix(self.measures["eyeWidth"], self.tUnit)}      {Fore.BLUE}{self.measures["eyeWidthP"]:6.2f} %')
    print(
      f'  dutyCycleDist: {Fore.CYAN}{self.measures["dutyCycleDistortion"]:6.2f} %   {Fore.BLUE}σ= {self.measures["dutyCycleDistortionStdDev"]:6.2f} %')
    print(
      f'  nBits:        {Fore.CYAN}{metricPrefix(self.measures["nBits"], "b")}')
    if self.mask:
      print(
        f'  nBadBits:     {Fore.CYAN}{metricPrefix(self.measures["offenderCount"], "b")}       {Fore.BLUE}{self.measures["ber"]:9.2e}')
      print(f'  maskMargin:   {Fore.CYAN}{self.measures["maskMargin"]:6.2f}%')

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


def _runnerBitExtract(edges: tuple[list, list], tZero: float, tDelta: float, tBit: float,
                      tHighOffset: float, tLowOffset: float, nBitsMax: int = 5) -> tuple[list, list]:
  '''!@brief Runner to extract bit centers from list of rising and falling edges

  @param edges tuple of rising edges, falling edges
  @param tZero Time for the first sample
  @param tDelta Time between samples
  @param tBit Time for a single bit
  @param tHighOffset Time to add to high pulses before calculating number of bits
  @param tLowOffset Time to add to low pulses before calculating number of bits
    @param nBitsMax Maximum number of consecutive bits of the same state allowed before an exception is raised
  @return tuple[list, list] tuple (bit center postitions, adjusted bit durations)
  @return tuple[list[float], list[int]] (bitCentersT, bitCentersY)
      bitCentersT is a list of centerTs. Use np.linspace(center - sliceHalfWidth * tDelta, c + shw * td, shw * 2 + 1)
      bitCentersY is a list of center indices. Use y[sliceCenter - sliceHalfWidth : c + shw + 1]
  '''
  edgesRise, edgesFall = edges
  bitCenters = []
  for i in range(1, min(len(edgesRise), len(edgesFall))):
    duration = edgesRise[i] - edgesFall[i]
    center = (edgesRise[i] + edgesFall[i]) / 2
    if duration > 0:
      # Duration is a low pulse
      duration += tLowOffset

      # Previous pulse is a high pulse
      durationPrev = edgesFall[i] - edgesRise[i - 1]
      durationPrev += tHighOffset
      centerPrev = (edgesFall[i] + edgesRise[i - 1]) / 2
    else:
      # Duration is a high pulse
      duration = -duration + tHighOffset

      # Previous pulse is a low pulse
      durationPrev = edgesRise[i] - edgesFall[i - 1]
      durationPrev += tLowOffset
      centerPrev = (edgesRise[i] + edgesFall[i - 1]) / 2

    nBits = int((duration / tBit) + 0.5)
    if nBits > nBitsMax or nBits == 0:
      errorStr = 'Number of consecutive bits of same level wrong\n'
      errorStr += f'  n: {int(nBits)}\n'
      errorStr += f'  Duration: {metricPrefix(duration)}\n'
      errorStr += f'  t={metricPrefix(edgesRise[i])}'
      raise Exception(errorStr)
    adjustedBitDuration = duration / nBits
    firstBit = center - adjustedBitDuration * (nBits - 1) / 2
    for iBit in range(nBits):
      bitCenters.append(firstBit + iBit * adjustedBitDuration)

    nBits = int((durationPrev / tBit) + 0.5)
    if nBits > nBitsMax or nBits == 0:
      errorStr = 'Number of consecutive bits of same level wrong\n'
      errorStr += f'  n: {int(nBits)}\n'
      errorStr += f'  Duration: {metricPrefix(durationPrev)}\n'
      errorStr += f'  t={metricPrefix(edgesRise[i])}'
      raise Exception(errorStr)
    adjustedBitDuration = durationPrev / nBits
    firstBit = centerPrev - adjustedBitDuration * (nBits - 1) / 2
    for iBit in range(nBits):
      bitCenters.append(firstBit + iBit * adjustedBitDuration)

  bitCentersT = []
  bitCentersY = []

  for b in bitCenters[5:-5]:
    bitCenterAdjusted = b % tDelta
    # xMin = -bitCenterAdjusted - sliceHalfWidth * tDelta
    # xMax = -bitCenterAdjusted + sliceHalfWidth * tDelta
    bitCentersT.append(-bitCenterAdjusted)

    sliceCenter = int(((b - tZero - bitCenterAdjusted) / tDelta) + 0.5)
    # sliceMin = sliceCenter - sliceHalfWidth
    # sliceMax = sliceCenter + sliceHalfWidth
    bitCentersY.append(sliceCenter)

  return bitCentersT, bitCentersY

def _runnerCollectValuesY(
  waveformY: np.ndarray, bitCentersT: list[float], bitCentersY: list[int], tDelta: float, tBit: float, yHalf: float) -> dict:
  '''!@brief Collect values from waveform: zero/cross/one vertical levels

  @param waveformY Waveform data array [y0, y1,..., yn]
  @param bitCentersT list of centerTs. Using np.linspace(center - sliceHalfWidth * tDelta, c + shw * td, shw * 2 + 1)
  @param bitCentersY list of center indices. Use y[sliceCenter - sliceHalfWidth : c + shw + 1]
  @param tDelta Time between samples
  @param tBit Time for a single bit
  @param yHalf Threshold to decide '1' or '0'
  @return dict Dictionary of collected values: zero, cross, one
  '''
  hw = int((tBit / tDelta) + 0.5)
  n = hw * 2 + 1
  t0 = np.linspace(-hw * tDelta, hw * tDelta, n).tolist()
  t0 = (t0 + tBit / 2) / tBit  # Transform to UI units

  waveformY = waveformY.tolist()

  # One level and zero level
  # Historgram mean 40% to 60%
  valuesZero = []
  valuesOne = []
  # Historgram mean -5% to 5% for bit transitions
  valuesCross = []
  for i in range(len(bitCentersT)):
    cT = bitCentersT[i] / tBit
    cY = bitCentersY[i]

    # Only look at transitions at t=0
    transition = (waveformY[cY - hw] > yHalf) != (waveformY[cY] > yHalf)

    bitValues = []

    for ii in range(n):
      t = t0[ii] + cT
      if t >= 0.4 and t <= 0.6:
        bitValues.append(waveformY[cY - hw + ii])
      if transition and t >= -0.05 and t <= 0.05:
        valuesCross.append(waveformY[cY - hw + ii])

    if np.mean(bitValues) > yHalf:
      valuesOne.extend(bitValues)
    else:
      valuesZero.extend(bitValues)

  values = {
    'zero': valuesZero,
    'cross': valuesCross,
    'one': valuesOne,
  }
  return values

def _runnerCollectValuesT(waveformY: np.ndarray, bitCentersT: list[float], bitCentersY: list[int],
                          tDelta: float, tBit: float, yZero: float, yOne: float) -> dict:
  '''!@brief Collect values from waveform: rise/fall times @ 20%, 50%, 80%

  @param waveformY Waveform data array [y0, y1,..., yn]
  @param bitCentersT list of centerTs. Using np.linspace(center - sliceHalfWidth * tDelta, c + shw * td, shw * 2 + 1)
  @param bitCentersY list of center indices. Use y[sliceCenter - sliceHalfWidth : c + shw + 1]
  @param tDelta Time between samples
  @param tBit Time for a single bit
  @param yZero Vertical value for a '0'
  @param yOne Vertical value for a '1'
  @return dict Dictionary of collected values: rise20, rise50, rise80, fall20, fall50, fall80
  '''
  hw = int((tBit / tDelta) + 0.5)
  n = hw * 2 + 1
  t0 = np.linspace(-hw * tDelta, hw * tDelta, n).tolist()

  waveformY = (waveformY - yZero) / (yOne - yZero)
  waveformY = waveformY.tolist()

  # Collect time for edges at 20%, 50%, and 80% values
  valuesRise20 = []
  valuesRise50 = []
  valuesRise80 = []
  valuesFall20 = []
  valuesFall50 = []
  valuesFall80 = []
  for i in range(len(bitCentersT)):
    cT = bitCentersT[i]
    cY = bitCentersY[i]
    # Only look at transitions at t=0
    if (waveformY[cY - hw] > 0.5) == (waveformY[cY] > 0.5):
      continue

    y = waveformY[cY - hw: cY + hw + 1]

    if waveformY[cY] > 0.5:
      ii = np.argmax(y)
      if y[ii] < 0.8:
        errorStr = 'Waveform does not reach 80%\n'
        errorStr += f'  i: {i}\n'
        errorStr += f'  yMax: {metricPrefix(y[ii])}\n'
        raise Exception(errorStr)

      t, ii = getCrossing(t0, y, ii, 0.8)
      valuesRise80.append(t + cT)
      t, ii = getCrossing(t0, y, ii, 0.5)
      valuesRise50.append(t + cT)
      t, ii = getCrossing(t0, y, ii, 0.2)
      valuesRise20.append(t + cT)
    else:
      ii = np.argmin(y)
      if y[ii] > 0.2:
        errorStr = 'Waveform does not reach 20%\n'
        errorStr += f'  i: {i}\n'
        errorStr += f'  yMin: {metricPrefix(y[ii])}\n'
        raise Exception(errorStr)

      t, ii = getCrossing(t0, y, ii, 0.2)
      valuesFall20.append(t + cT)
      t, ii = getCrossing(t0, y, ii, 0.5)
      valuesFall50.append(t + cT)
      t, ii = getCrossing(t0, y, ii, 0.8)
      valuesFall80.append(t + cT)

  values = {
    'rise20': valuesRise20,
    'rise50': valuesRise50,
    'rise80': valuesRise80,
    'fall20': valuesFall20,
    'fall50': valuesFall50,
    'fall80': valuesFall80
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

    if factor > 0:
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

  if factor > 0:
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

    if factor > 0:
      t = (tNew + cT).tolist()
      yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
      y = np.dot(yOriginal, referenceSinc).tolist()
    else:
      t = (t0 + cT).tolist()
      y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]

    while isHitting(t, y, maskAdjusted.paths) and maskMargin >= -1:
      maskMargin -= 0.0001
      maskAdjusted = mask.adjust(maskMargin)
      offender = i

  # Collect mask hits for the worst offender
  cY = bitCentersY[offender]
  cT = bitCentersT[offender] / tBit

  if factor > 0:
    t = (tNew + cT).tolist()
    yOriginal = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
    y = np.dot(yOriginal, referenceSinc).tolist()
  else:
    t = (t0 + cT).tolist()
    y = waveformY[cY - iBitWidth: cY + iBitWidth + 1]
  maskAdjusted = mask.adjust(maskMargin + 0.0001)
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

    if factor > 0:
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
