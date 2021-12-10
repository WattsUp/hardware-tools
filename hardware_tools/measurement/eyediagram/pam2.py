"""PAM2 Encoded Eye Diagram, aka NRZ
"""

from __future__ import annotations

import base64
import datetime
import io
import multiprocessing
from typing import Union

import colorama
from colorama import Fore
import numpy as np
import matplotlib.pyplot as pyplot

from hardware_tools import math, strformat
from hardware_tools.measurement.eyediagram import eyediagram


class MeasuresPAM2(eyediagram.Measures):
  """Eye Diagram Measures collecting metrics from an eye diagram

  Measures for a PAM2 (NRZ) encoded signal

  All images are base64 encoded PNGs, use save_images to save to disk

  Properties:
    n_sym: Number of symbols tested
    n_sym_bad: Number of symbols that failed, requires a mask
    transition_dist: Distribution of transitions plotted
      000, 001, 010, 011, 100, 101, 110, 111
    mask_margin: largest mask adjust without mask hits, requires a mask
      range [-1, 1], see Mask.adjust
    image_clean: layered waveforms heatmap
    image_grid: grid of UI and UA values
    image_mask: image of unadjusted mask
    image_hits: subset of mask hits and offending waveforms
    image_margin: image of margin adjusted mask

    y_0: Mean value of logical 0
    y_1: Mean value of logical 1
    y_cross: Mean value of crossing point
    y_cross_r: Ratio [0, 1] = (y_cross - y_0) / amp

    amp: Eye amplitude = y_1 - y_0
    height: Eye height = (y_1 - 3 * y_1.stddev) - (y_0 + 3 * y_0.stddev)
    height_r: Ratio [0, 1] = height / amp
    snr: Signal-to-noise ratio = amp / (y_1.stddev + y_0.stddev)

    t_sym: Mean value of symbol duration
    t_0: Mean value of logical 0 duration (y=50% histogram)
    t_1: Mean value of logical 1 duration (y=50% histogram)
    t_rise: Mean value of rising edges 20% to 80%
    t_fall: Mean value of falling edges 80% to 20%
    t_cross: Mean value of crossing point

    f_sym: Frequency of symbols = 1 / t_sym

    width: Eye width = (t_cross2 - 3 * t_cross2.stddev)
      - (t_cross1 - 3 * t_cross1.stddev)
    width_r: Ratio [0, 1] = width / t_sym
    dcd: Duty-cycle-distortion [-1, 1] = (t_1 - t_0) / (2 * t_sym)
    jitter_pp: Full width of time histogram at crossing point
    jitter_rms: Standard deviation of time histogram at crossing point

    ## The following properties are only applicable to optical signals ##
    extinction_ratio: Ratio [0, 1] = y_1 / y_0
    oma_cross: Optical Modulation Amplitude = y_1 - y_0 at crossing point
    vecp: Vertical Eye Closure Penalty in dB = 10 * log(oma_cross / a_0)
      where a_0 = (y_1, 0.05% histogram) - (y_0, 99.95% histogram)
  """

  def __init__(self, np_clean: np.ndarray, np_grid: np.ndarray,
               np_mask: np.ndarray, np_hits: np.ndarray,
               np_margin: np.ndarray) -> None:
    super().__init__(np_clean, np_grid, np_mask, np_hits, np_margin)
    # PAM2 specific measures
    self.y_0 = None
    self.y_1 = None
    self.y_cross = None
    self.y_cross_r = None

    self.amp = None
    self.height = None
    self.height_r = None
    self.snr = None

    self.t_sym = None
    self.t_0 = None
    self.t_1 = None
    self.t_rise = None
    self.t_fall = None

    self.f_sym = None

    self.width = None
    self.width_r = None
    self.dcd = None
    self.jitter_pp = None
    self.jitter_rms = None

    self.extinction_ratio = None
    # self.tdec = None TODO figure how to measure this and why
    self.oma_cross = None
    self.vecp = None


class PAM2(eyediagram.EyeDiagram):
  """Eye Diagram to layer repeated waveforms and measure the resultant heat map

  Expects PAM2 (NRZ) encoded signal
  """

  def __init__(self,
               waveforms: np.ndarray,
               clocks: np.ndarray = None,
               t_unit: str = "",
               y_unit: str = "",
               mask: eyediagram.Mask = None,
               resolution: int = 2000,
               unipolar: bool = True,
               resample: int = 50,
               y_0: float = None,
               y_1: float = None,
               **kwargs) -> None:
    """Create a new PAM2 EyeDiagram. Lazy creation, does not compute anything

    Args:
      waveforms: 3D array of waveforms, a 2D array is assumed to be a single
        waveform.
        [waveform0([[t0, t1,..., tn], [y0, y1,..., yn]]), waveform1,...]
      clocks: 3D array of clock waveforms, a 2D array is assumed to be a
        single clock waveform. None will recover the clock from the signal.
        [waveform0([[t0, t1,..., tn], [y0, y1,..., yn]]), waveform1,...]
      t_unit: Real world units of horizontal axis, used for print statements
      y_unit: Real world units of vertical axis, used for print statements
      mask: Mask object used for hit detection, None will not check for hits
      resolution: Resolution of square eye diagram image, 2UI x 2UA
      unipolar: True will plot amplitude [-0.5, 1.5]UA, False will plot [-1, 1]
      resample: n=0 will not resample, n>0 will use sinc interpolation to
        resample a single symbol to at least n segments (t_delta = t_symbol / n)
      y_0: Manually specified value for 0 symbol, None will measure waveform
      y_1: Manually specified value for 1 symbol, None will measure waveform

      **kwargs: Override default configuration, see source for options

    Raises:
      ValueError if waveforms or clocks is wrong shape
    """
    super().__init__(waveforms,
                     clocks=clocks,
                     t_unit=t_unit,
                     y_unit=y_unit,
                     mask=mask,
                     resolution=resolution,
                     unipolar=unipolar,
                     resample=resample)
    self._y_0 = y_0
    self._y_1 = y_1

    self._config = {
        # Step 1
        "hysteresis": None,  # Difference between rising and falling thresholds
        "hysteresis_ua": 0.1,  # Units of normalized amplitude, lower priority
        "hist_n_max": 50e5,  # Maximum number of points in a histogram

        # Step 2

        # Step 3

        # Step 4

        # Step 5
        "level_width": 0.2,  # Width of histogram for y_0, y_1
        "cross_width": 0.1,  # Width of histogram for y_cross
        "levels_height": 0.05,  # Height of histogram for t_0, t_1
        "cross_height": 0.05,  # Height of histogram for t_cross
        "edge_height": 0.05,  # Height of histogram for t_rise, t_fall
        "noise_floor": 0,  # Stddev of noise floor affecting extinction_ratio
    }
    for k, v in kwargs.items():
      if k not in self._config:
        raise KeyError(f"Unrecognized additional configuration: {k}={v}")
      self._config[k] = v

  def _step1_levels(self,
                    n_threads: int = 1,
                    print_progress: bool = True,
                    indent: int = 0,
                    debug_plots: str = None) -> None:
    if print_progress:
      print(f"{'':>{indent}}Starting PAM2 levels")

    y_0 = math.UncertainValue(0, 0)
    y_1 = math.UncertainValue(0, 0)
    if self._y_0 is None or self._y_1 is None or debug_plots is not None:
      # Need to measure at least one
      # yapf: disable
      args_list = [[
          self._waveforms[i][1],
          self._config["hist_n_max"]
      ] for i in range(self._waveforms.shape[0])]
      # yapf: enable
      output = self._collect_runners(_runner_levels, args_list, n_threads,
                                     print_progress, indent)

      y_min = min([o["min"] for o in output])
      y_max = max([o["max"] for o in output])
      for o in output:
        y_0 += o["0"]
        y_1 += o["1"]
      y_0 = y_0 / len(output)
      y_1 = y_1 / len(output)
    if self._y_0 is not None:
      y_0.value = self._y_0
    if self._y_1 is not None:
      y_1.value = self._y_1

    if print_progress:
      print(f"{'':>{indent}}Computing thresholds")
    if self._config["hysteresis"] is not None:
      hys = self._config["hysteresis"]
    else:
      hys = self._config["hysteresis_ua"] * (y_1 - y_0).value

    self._y_zero = y_0.value
    self._y_ua = (y_1 - y_0).value

    self._y_half = ((y_1 + y_0) / 2).value
    self._y_rising = (self._y_half + hys / 2)
    self._y_falling = (self._y_half - hys / 2)

    # Check high level is not close to low level
    if (y_1.stddev + y_0.stddev) == 0:
      self._low_snr = False
    else:
      snr = (y_1 - y_0) / (y_1.stddev + y_0.stddev)
      self._low_snr = snr < 2

    if print_progress:
      print(f"{'':>{indent}}Completed PAM2 levels")

    if debug_plots is not None:
      debug_plots += ".step1.png"

      n = 1000
      y_range = np.linspace(y_min, y_max, n)
      curve_0 = np.zeros(n)
      curve_1 = np.zeros(n)
      for o in output:
        curve_0 += o["fit_0"].compute(y_range)
        curve_1 += o["fit_1"].compute(y_range)
      pyplot.plot(y_range, curve_0 / len(output))
      pyplot.plot(y_range, curve_1 / len(output))
      pyplot.axvline(x=self._y_zero, color="g")
      pyplot.axvline(x=self._y_zero + self._y_ua, color="g")
      pyplot.axvline(x=self._y_rising, color="r")
      pyplot.axvline(x=self._y_half, color="k")
      pyplot.axvline(x=self._y_falling, color="r")
      pyplot.axvline(x=(y_0.value + y_0.stddev), color="y")
      pyplot.axvline(x=(y_0.value - y_0.stddev), color="y")
      pyplot.axvline(x=(y_1.value + y_1.stddev), color="y")
      pyplot.axvline(x=(y_1.value - y_1.stddev), color="y")

      ax = pyplot.gca()

      def tick_formatter_y(y, _):
        return strformat.metric_prefix(y, self._y_unit)

      formatter_y = pyplot.FuncFormatter(tick_formatter_y)
      ax.xaxis.set_major_formatter(formatter_y)

      pyplot.xlabel("Vertical scale")
      pyplot.ylabel("Density")
      pyplot.title("Vertical levels")

      pyplot.savefig(debug_plots, bbox_inches="tight")
      pyplot.close()
      print(f"{'':>{indent}}Saved image to {debug_plots}")

  def _step2_clock(self,
                   n_threads: int = 1,
                   print_progress: bool = True,
                   indent: int = 0,
                   debug_plots: str = None) -> None:
    # self._t_sym = float
    pass

  def _step3_sample(self,
                    n_threads: int = 1,
                    print_progress: bool = True,
                    indent: int = 0,
                    debug_plots: str = None) -> None:
    # self._centers_i = list[list[int]]
    # self._centers_t = list[list[float]]
    pass

  def _step5_measure(self,
                     n_threads: int = 1,
                     print_progress: bool = True,
                     indent: int = 0,
                     debug_plots: str = None) -> None:
    # self._measures = Measures
    pass


def _runner_levels(waveform_y: np.ndarray, n_max: int = 50e3) -> dict:
  """Calculate the high and low levels of the waveform

  Args:
    waveform_y: Waveform data array [y0, y1,..., yn]
    n_max: Maximum number of point to calculate on, None for no limit

  Returns:
    Dictionary of values:
      min: Minimum value
      max: Maximum value
      0: Most likely location for y_0
      1: Most likely location for y_1
      fit_0: GaussianMix for y_0
      fit_1: GaussianMix for y_1
  """
  if n_max is not None:
    waveform_y = math.Bin.downsample(waveform_y, n_max)

  y_min = np.amin(waveform_y)
  y_max = np.amax(waveform_y)
  y_mid = (y_min + y_max) / 2
  y_0 = math.UncertainValue(0, 0)
  y_1 = math.UncertainValue(0, 0)

  values_0 = waveform_y[np.where(waveform_y < y_mid)]
  y_0.stddev = np.std(values_0)

  values_1 = waveform_y[np.where(waveform_y > y_mid)]
  y_1.stddev = np.std(values_1)

  fit_0 = math.GaussianMix.fit_samples(values_0, n_max=3)
  fit_1 = math.GaussianMix.fit_samples(values_1, n_max=3)

  y_0.value = fit_0.center()
  y_1.value = fit_1.center()

  return {
      "min": y_min,
      "max": y_max,
      "0": y_0,
      "1": y_1,
      "fit_0": fit_0,
      "fit_1": fit_1
  }
