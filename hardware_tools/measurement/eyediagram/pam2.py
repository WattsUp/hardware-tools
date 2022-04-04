"""PAM2 Encoded Eye Diagram, aka NRZ
"""

from __future__ import annotations

import copy
from typing import List, Tuple

import colorama
from colorama import Fore
import numpy as np
from matplotlib import pyplot

from hardware_tools import strformat
from hardware_tools.math import gaussian, lines, stats
from hardware_tools.measurement.mask import Mask
from hardware_tools.measurement.eyediagram import cdr, eyediagram

try:
  from hardware_tools.measurement.eyediagram import _pam2
except ImportError:
  print(f"The cython version of {__name__} is not available")
  from hardware_tools.measurement.eyediagram import _pam2_fb as _pam2

colorama.init(autoreset=True)


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
    y_cross: Mean value of crossing point, with symbol transition
    y_cross_r: Ratio [0, 1] = (y_cross - y_0) / amp

    y_0_cross: Mean value of logical 0 at crossing point
    y_1_cross: Mean value of logical 1 at crossing point

    amp: Eye amplitude = y_1 - y_0
    height: Eye height = (y_1 - 3 * y_1.stddev) - (y_0 + 3 * y_0.stddev)
    height_r: Ratio [0, 1] = height / amp
    snr: Signal-to-noise ratio = amp / (y_1.stddev + y_0.stddev)

    t_sym: Mean value of symbol duration
    t_0: Mean value of logical 0 duration (y=50% histogram)
    t_1: Mean value of logical 1 duration (y=50% histogram)
    t_rise: Mean value of rising edges lower to upper
    t_fall: Mean value of falling edges upper to lower
    t_rise_start: Mean value of rising edge lower
    t_fall_start: Mean value of rising edge upper
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

  def __init__(self) -> None:
    super().__init__()
    # PAM2 specific measures
    self.y_0 = None
    self.y_1 = None
    self.y_cross = None
    self.y_cross_r = None

    self.y_0_cross = None
    self.y_1_cross = None

    self.amp = None
    self.height = None
    self.height_r = None
    self.snr = None

    self.t_sym = None
    self.t_0 = None
    self.t_1 = None
    self.t_rise = None
    self.t_fall = None
    self.t_rise_start = None
    self.t_fall_start = None
    self.t_cross = None

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


class PAM2Config(eyediagram.Config):
  """PAM2 Specific configuration

  Properties:
    y_0: float, manual level for logical zero
    y_1: float, manual level for logical one
  """

  def __init__(self, **kwargs) -> None:
    super().__init__()
    # Step 1
    self.y_0 = None
    self.y_1 = None

    # Step 2

    # Step 3

    # Step 4

    # Step 5

    self.consume(kwargs)


class PAM2(eyediagram.EyeDiagram):
  """Eye Diagram to layer repeated waveforms and measure the resultant heat map

  Expects PAM2 (NRZ) encoded signal
  """

  def __init__(self,
               waveforms: np.ndarray,
               clocks: np.ndarray = None,
               clock_edges: List[List[float]] = None,
               t_unit: str = "",
               y_unit: str = "",
               mask: Mask = None,
               resolution: int = 2000,
               config: PAM2Config = None) -> None:
    """Create a new PAM2 EyeDiagram. Lazy creation, does not compute anything

    Args:
      waveforms: 3D array of waveforms, a 2D array is assumed to be a single
        waveform.
        [waveform0([[t0, t1,..., tn], [y0, y1,..., yn]]), waveform1,...]
      clocks: 3D array of clock waveforms, a 2D array is assumed to be a
        single clock waveform. None will recover the clock from the signal.
        [waveform0([[t0, t1,..., tn], [y0, y1,..., yn]]), waveform1,...]
      clock_edges: 2D array of clock edges, None will recover the clock edges
        from the signal or clock waveform. Primarily pass the output of
        get_clock_edges(), see for example usage.
        [edges0([t0, t1,..., tn]), edges1,...]
      t_unit: Real world units of horizontal axis, used for print statements
      y_unit: Real world units of vertical axis, used for print statements
      mask: Mask object used for hit detection, None will not check for hits
      resolution: Resolution of square eye diagram image, 2UI x 2UA
      config: Configuration settings

    Raises:
      ValueError if waveforms or clocks is wrong shape
    """
    super().__init__(waveforms,
                     clocks=clocks,
                     clock_edges=clock_edges,
                     t_unit=t_unit,
                     y_unit=y_unit,
                     mask=mask,
                     resolution=resolution,
                     config=None)
    if config is None:
      self._config = PAM2Config()
    elif isinstance(config, PAM2Config):
      self._config = config
    else:
      raise ValueError("config must be of type PAM2Config")

  def _step1_levels(self,
                    print_progress: bool = True,
                    indent: int = 0,
                    debug_plots: str = None) -> None:
    if print_progress:
      print(f"{'':>{indent}}Starting PAM2 levels")

    y_0 = stats.UncertainValue(0, 0)
    y_1 = stats.UncertainValue(0, 0)
    if ((self._config.y_0 is None) or (self._config.y_1 is None) or
        (debug_plots is not None)):
      waveform_y = self._waveforms[:, 1].flatten()
      output = _runner_levels(waveform_y, self._config.levels_n_max)

      y_min = output["min"]
      y_max = output["max"]
      y_0 = output["0"]
      y_1 = output["1"]
    if self._config.y_0 is not None:
      y_0.value = self._config.y_0
    if self._config.y_1 is not None:
      y_1.value = self._config.y_1

    if print_progress:
      print(f"{'':>{indent}}Computing thresholds")
    if self._config.hysteresis is not None:
      hys = self._config.hysteresis
    else:
      hys = self._config.hysteresis_ua * (y_1 - y_0).value

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
      curve_0 = output["fit_0"].compute(y_range)
      curve_1 = output["fit_1"].compute(y_range)
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

      pyplot.tight_layout()
      pyplot.savefig(debug_plots, bbox_inches="tight")
      pyplot.close()
      print(f"{'':>{indent}}Saved image to {debug_plots}")

  def _step2_clock(self,
                   print_progress: bool = True,
                   indent: int = 0,
                   debug_plots: str = None) -> None:
    if self._clock_edges is None:
      if self._clocks is not None:
        self._clock_edges = []
        self._ties = []
        # Get edges from _clocks given clock_polarity
        for i in range(self._waveforms.shape[0]):
          e = lines.edges_np(self._clocks[i][0], self._clocks[i][1],
                             self._y_rising, self._y_half, self._y_falling)
          e = _filter_edge_polarity(e, self._config.clock_polarity)
          self._clock_edges.append(e.tolist())
          self._ties.append([])
      elif not self._low_snr:
        if print_progress:
          print(f"{'':>{indent}}Running clock data recovery")
        # Run CDR to generate edges
        if self._config.cdr is None:
          self._config.cdr = cdr.CDR(self._config.fallback_period)

        self._clock_edges = []
        self._ties = []
        for i in range(self._waveforms.shape[0]):
          o = _runner_cdr(self._waveforms[i][0], self._waveforms[i][1],
                          self._y_rising, self._y_half, self._y_falling,
                          copy.deepcopy(self._config.cdr),
                          self._config.clock_polarity)
          self._clock_edges.append(o[0].tolist())
          self._ties.append(o[1].tolist())
    super()._step2_clock(print_progress=print_progress,
                         indent=indent,
                         debug_plots=debug_plots)

  def _draw_grid(self, image_grid: np.ndarray) -> None:
    """Draw reference grid

    Args:
      image_grid: Image to draw onto, [x, y] coordinates
    """
    super()._draw_grid(image_grid)
    image_grid[::5, self._uia_to_image(self._config.edge_lower), 3] = 1.0
    image_grid[::5, self._uia_to_image(self._config.edge_upper), 3] = 1.0

  def _step4_measure(self,
                     print_progress: bool = True,
                     indent: int = 0,
                     debug_plots: str = None) -> None:
    m = MeasuresPAM2()
    t_sym = self._t_sym

    if self._config.skip_measures:
      m.n_sym = 0
      for i in range(self._waveforms.shape[0]):
        m.n_sym += len(self._centers_i[i])
      m.n_samples = int(m.n_sym * t_sym / self._t_delta)
      m.transition_dist = {
          "000": 0,
          "001": 0,
          "010": 0,
          "011": 0,
          "100": 0,
          "101": 0,
          "110": 0,
          "111": 0,
      }
      self._measures = m
      self._offenders = [[]] * self._waveforms.shape[0]
      self._hits = [[]] * self._waveforms.shape[0]
      return

    if print_progress:
      print(f"{'':>{indent}}Measuring waveform vertically")

    s_y_0 = []
    s_y_1 = []
    s_y_cross = []
    s_y_0_cross = []
    s_y_1_cross = []
    transitions = {
        "000": 0,
        "001": 0,
        "010": 0,
        "011": 0,
        "100": 0,
        "101": 0,
        "110": 0,
        "111": 0,
    }
    edge_dir = []
    for i in range(self._waveforms.shape[0]):
      o = _pam2.sample_vertical(self._waveforms[i][1], self._centers_t[i],
                                self._centers_i[i], self._t_delta, t_sym,
                                self._y_half, self._config.level_width,
                                self._config.cross_width)
      s_y_0.extend(o["y_0"])
      s_y_1.extend(o["y_1"])
      s_y_cross.extend(o["y_cross"])
      s_y_0_cross.extend(o["y_0_cross"])
      s_y_1_cross.extend(o["y_1_cross"])
      for t in transitions:
        transitions[t] += o["transitions"][t]
      edge_dir.append(o["edge_dir"])
    s_y_0 = np.fromiter(s_y_0, np.float64)
    s_y_1 = np.fromiter(s_y_1, np.float64)
    s_y_cross = np.fromiter(s_y_cross, np.float64)
    s_y_0_cross = np.fromiter(s_y_0_cross, np.float64)
    s_y_1_cross = np.fromiter(s_y_1_cross, np.float64)

    if s_y_0.size < 1:
      print(f"{'':>{indent}}{Fore.RED}y_0 does not have any samples")
    if s_y_1.size < 1:
      print(f"{'':>{indent}}{Fore.RED}y_1 does not have any samples")
    if s_y_cross.size < 1:
      print(f"{'':>{indent}}{Fore.RED}y_cross does not have any samples")
    if s_y_0_cross.size < 1:
      print(f"{'':>{indent}}{Fore.RED}y_0_cross does not have any samples")
    if s_y_1_cross.size < 1:
      print(f"{'':>{indent}}{Fore.RED}y_1_cross does not have any samples")

    m.n_sym = 0
    for i in range(self._waveforms.shape[0]):
      m.n_sym += len(self._centers_i[i])
    m.n_samples = int(m.n_sym * t_sym / self._t_delta)

    m.y_0 = stats.UncertainValue.samples(s_y_0)
    m.y_1 = stats.UncertainValue.samples(s_y_1)
    m.y_cross = stats.UncertainValue.samples(s_y_cross)
    m.y_0_cross = stats.UncertainValue.samples(s_y_0_cross)
    m.y_1_cross = stats.UncertainValue.samples(s_y_1_cross)
    m.transition_dist = transitions

    # Computed measures
    m.amp = m.y_1 - m.y_0
    m.height = (m.y_1 - 3 * m.y_1.stddev) - (m.y_0 + 3 * m.y_0.stddev)
    m.height_r = m.height / m.amp
    m.snr = m.amp / (m.y_1.stddev + m.y_0.stddev)
    m.y_cross_r = (m.y_cross - m.y_0) / m.amp

    # Computed measures, optical
    m.extinction_ratio = (m.y_1 - self._config.noise_floor) / (
        m.y_0 - self._config.noise_floor)
    m.oma_cross = m.y_1_cross - m.y_0_cross
    # median_unbiased: "This method is probably the best method if the sample
    # distribution function is unknown"
    if s_y_0.size < 1 or s_y_1.size < 1:
      m.vecp = stats.UncertainValue(np.nan, np.nan)
    else:
      a_0 = np.percentile(s_y_1, 0.05,
                          method="median_unbiased") - np.percentile(
                              s_y_0, 99.95, method="median_unbiased")
      vecp_linear = m.oma_cross / a_0
      m.vecp = np.log10(vecp_linear) * 10

    # Update levels if not manual
    if self._config.y_0 is None:
      self._y_zero = m.y_0.value
    if self._config.y_1 is None:
      self._y_ua = m.amp.value
    else:
      self._y_ua = self._config.y_1 - self._y_zero
    self._y_half = self._y_ua / 2 + self._y_zero

    if print_progress:
      print(f"{'':>{indent}}Measuring waveform horizontally")

    s_t_rise_lower = []
    s_t_rise_upper = []
    s_t_rise_half = []
    s_t_fall_lower = []
    s_t_fall_upper = []
    s_t_fall_half = []
    s_t_cross_left = []
    s_t_cross_right = []
    for i in range(self._waveforms.shape[0]):
      o = _pam2.sample_horizontal(
          self._waveforms[i][1], self._centers_t[i], self._centers_i[i],
          edge_dir[i], self._t_delta, t_sym, self._y_zero, self._y_ua,
          m.y_cross.value, self._config.time_height, self._config.edge_lower,
          self._config.edge_upper)
      s_t_rise_lower.extend(o["t_rise_lower"])
      s_t_rise_upper.extend(o["t_rise_upper"])
      s_t_rise_half.extend(o["t_rise_half"])
      s_t_fall_lower.extend(o["t_fall_lower"])
      s_t_fall_upper.extend(o["t_fall_upper"])
      s_t_fall_half.extend(o["t_fall_half"])
      s_t_cross_left.extend(o["t_cross_left"])
      s_t_cross_right.extend(o["t_cross_right"])
    s_t_rise_lower = np.fromiter(s_t_rise_lower, np.float64)
    s_t_rise_upper = np.fromiter(s_t_rise_upper, np.float64)
    s_t_rise_half = np.fromiter(s_t_rise_half, np.float64)
    s_t_fall_lower = np.fromiter(s_t_fall_lower, np.float64)
    s_t_fall_upper = np.fromiter(s_t_fall_upper, np.float64)
    s_t_fall_half = np.fromiter(s_t_fall_half, np.float64)
    s_t_cross_left = np.fromiter(s_t_cross_left, np.float64)
    s_t_cross_right = np.fromiter(s_t_cross_right, np.float64)

    if s_t_rise_lower.size < 1:
      print(f"{'':>{indent}}{Fore.RED}t_rise_lower does not have any samples")
    if s_t_rise_upper.size < 1:
      print(f"{'':>{indent}}{Fore.RED}t_rise_upper does not have any samples")
    if s_t_rise_half.size < 1:
      print(f"{'':>{indent}}{Fore.RED}t_rise_half does not have any samples")
    if s_t_fall_lower.size < 1:
      print(f"{'':>{indent}}{Fore.RED}t_fall_lower does not have any samples")
    if s_t_fall_upper.size < 1:
      print(f"{'':>{indent}}{Fore.RED}t_fall_upper does not have any samples")
    if s_t_fall_half.size < 1:
      print(f"{'':>{indent}}{Fore.RED}t_fall_half does not have any samples")
    if s_t_cross_left.size < 1:
      print(f"{'':>{indent}}{Fore.RED}t_cross_left does not have any samples")
    if s_t_cross_right.size < 1:
      print(f"{'':>{indent}}{Fore.RED}t_cross_right does not have any samples")

    s_t_rise_lower = s_t_rise_lower * t_sym
    s_t_rise_upper = s_t_rise_upper * t_sym
    s_t_rise_half = s_t_rise_half * t_sym
    s_t_fall_lower = s_t_fall_lower * t_sym
    s_t_fall_upper = s_t_fall_upper * t_sym
    s_t_fall_half = s_t_fall_half * t_sym
    s_t_cross_left = s_t_cross_left * t_sym
    s_t_cross_right = s_t_cross_right * t_sym

    t_rise_lower = stats.UncertainValue.samples(s_t_rise_lower)
    t_rise_upper = stats.UncertainValue.samples(s_t_rise_upper)
    t_rise_half = stats.UncertainValue.samples(s_t_rise_half)
    t_fall_lower = stats.UncertainValue.samples(s_t_fall_lower)
    t_fall_upper = stats.UncertainValue.samples(s_t_fall_upper)
    t_fall_half = stats.UncertainValue.samples(s_t_fall_half)
    t_cross_left = stats.UncertainValue.samples(s_t_cross_left)
    t_cross_right = stats.UncertainValue.samples(s_t_cross_right)

    # Computed measures
    m.t_sym = t_cross_right - t_cross_left
    m.t_0 = t_rise_half - t_fall_half + t_sym
    m.t_1 = t_fall_half - t_rise_half + t_sym
    m.t_rise = t_rise_upper - t_rise_lower
    m.t_fall = t_fall_lower - t_fall_upper
    m.t_rise_start = t_rise_lower
    m.t_fall_start = t_fall_upper
    m.t_cross = t_cross_left

    m.f_sym = stats.UncertainValue(1, 0) / m.t_sym

    m.width = (t_cross_right - 3 * t_cross_right.stddev) - (
        t_cross_left + 3 * t_cross_left.stddev)
    m.width_r = m.width / m.t_sym
    m.dcd = (m.t_1 - m.t_0) / (m.t_sym * 2)
    if s_t_cross_left.size < 1:
      m.jitter_pp = np.nan
    else:
      m.jitter_pp = s_t_cross_left.ptp()
    m.jitter_rms = t_cross_left.stddev

    if print_progress:
      print(f"{'':>{indent}}Measuring waveform mask")

    output_mask = self._sample_mask()

    if self._mask is None:
      margin = np.nan
      offender_count = np.nan
    else:
      margin = output_mask["margin"]
      offender_count = output_mask["offender_count"]

    m.mask_margin = margin
    m.n_sym_bad = offender_count

    if print_progress:
      print(f"{'':>{indent}}Generating bathtub curves")

    m.bathtub_curves = self._generate_bathtub_curves({"Eye 0": 0.5})

    self._measures = m

    if print_progress:
      print(f"{'':>{indent}}Completed PAM2 measuring")

    if debug_plots is not None:
      debug_plots += ".step4.png"

      def tick_formatter_t(t, _):
        return strformat.metric_prefix(t, self._t_unit)

      formatter_t = pyplot.FuncFormatter(tick_formatter_t)

      def tick_formatter_y(y, _):
        return strformat.metric_prefix(y, self._y_unit)

      formatter_y = pyplot.FuncFormatter(tick_formatter_y)
      _, subplots = pyplot.subplots(2, 1)

      # 00 path
      t = [-0.5 * t_sym, 0.0, 0.5 * t_sym]
      y = [m.y_0.value, m.y_0_cross.value, m.y_0.value]
      subplots[0].plot(t, y, color="r")
      # yapf: disable
      y = [
          m.y_0.value + m.y_0.stddev,
          m.y_0_cross.value + m.y_0_cross.stddev,
          m.y_0.value + m.y_0.stddev
      ]
      # yapf: enable
      subplots[0].plot(t, y, color="r", linestyle="--")
      # yapf: disable
      y = [
          m.y_0.value - m.y_0.stddev,
          m.y_0_cross.value - m.y_0_cross.stddev,
          m.y_0.value - m.y_0.stddev
      ]
      # yapf: enable
      subplots[0].plot(t, y, color="r", linestyle="--")

      # 11 path
      t = [-0.5 * t_sym, 0.0, 0.5 * t_sym]
      y = [m.y_1.value, m.y_1_cross.value, m.y_1.value]
      subplots[0].plot(t, y, color="g")
      # yapf: disable
      y = [
          m.y_1.value + m.y_1.stddev,
          m.y_1_cross.value + m.y_1_cross.stddev,
          m.y_1.value + m.y_1.stddev
      ]
      # yapf: enable
      subplots[0].plot(t, y, color="g", linestyle="--")
      # yapf: disable
      y = [
          m.y_1.value - m.y_1.stddev,
          m.y_1_cross.value - m.y_1_cross.stddev,
          m.y_1.value - m.y_1.stddev
      ]
      # yapf: enable
      subplots[0].plot(t, y, color="g", linestyle="--")

      # 01 path
      y_lower = self._config.edge_lower * self._y_ua + self._y_zero
      y_upper = self._config.edge_upper * self._y_ua + self._y_zero
      # yapf: disable
      t = [
          -0.5 * t_sym,
          t_rise_lower.value,
          m.t_cross.value,
          t_rise_upper.value,
          0.5 * t_sym
      ]
      y = [
          m.y_0.value,
          y_lower,
          m.y_cross.value,
          y_upper,
          m.y_1.value
      ]
      # yapf: enable
      slope_lower = (y[1] - y[0]) / (t[1] - t[0])
      slope_upper = (y[4] - y[3]) / (t[4] - t[3])
      subplots[0].plot(t, y, color="b")
      # yapf: disable
      t = [
          -0.5 * t_sym,
          t_rise_lower.value - t_rise_lower.stddev,
          m.t_cross.value - m.t_cross.stddev,
          t_rise_upper.value - t_rise_upper.stddev,
          0.5 * t_sym
      ]
      y = [
          m.y_0.value + m.y_0.stddev,
          m.y_0.value + m.y_0.stddev + slope_lower * (t[1] - t[0]),
          m.y_cross.value,
          m.y_1.value + m.y_1.stddev - slope_upper * (t[4] - t[3]),
          m.y_1.value + m.y_1.stddev
      ]
      # yapf: enable
      subplots[0].plot(t, y, color="b", linestyle="--")
      # yapf: disable
      t = [
          -0.5 * t_sym,
          t_rise_lower.value + t_rise_lower.stddev,
          m.t_cross.value + m.t_cross.stddev,
          t_rise_upper.value + t_rise_upper.stddev,
          0.5 * t_sym
      ]
      y = [
          m.y_0.value - m.y_0.stddev,
          m.y_0.value - m.y_0.stddev + slope_lower * (t[1] - t[0]),
          m.y_cross.value,
          m.y_1.value - m.y_1.stddev - slope_upper * (t[4] - t[3]),
          m.y_1.value - m.y_1.stddev
      ]
      # yapf: enable
      subplots[0].plot(t, y, color="b", linestyle="--")

      # 10 path
      # yapf: disable
      t = [
          -0.5 * t_sym,
          t_fall_upper.value,
          m.t_cross.value,
          t_fall_lower.value,
          0.5 * t_sym
      ]
      y = [
          m.y_1.value,
          y_upper,
          m.y_cross.value,
          y_lower,
          m.y_0.value
      ]
      # yapf: enable
      slope_upper = (y[1] - y[0]) / (t[1] - t[0])
      slope_lower = (y[4] - y[3]) / (t[4] - t[3])
      subplots[0].plot(t, y, color="y")
      # yapf: disable
      t = [
          -0.5 * t_sym,
          t_fall_upper.value - t_fall_upper.stddev,
          m.t_cross.value - m.t_cross.stddev,
          t_fall_lower.value - t_fall_lower.stddev,
          0.5 * t_sym
      ]
      y = [
          m.y_1.value - m.y_1.stddev,
          m.y_1.value - m.y_1.stddev + slope_upper * (t[1] - t[0]),
          m.y_cross.value,
          m.y_0.value - m.y_0.stddev - slope_lower * (t[4] - t[3]),
          m.y_0.value - m.y_0.stddev
      ]
      # yapf: enable
      subplots[0].plot(t, y, color="y", linestyle="--")
      # yapf: disable
      t = [
          -0.5 * t_sym,
          t_fall_upper.value + t_fall_upper.stddev,
          m.t_cross.value + m.t_cross.stddev,
          t_fall_lower.value + t_fall_lower.stddev,
          0.5 * t_sym
      ]
      y = [
          m.y_1.value + m.y_1.stddev,
          m.y_1.value + m.y_1.stddev + slope_upper * (t[1] - t[0]),
          m.y_cross.value,
          m.y_0.value + m.y_0.stddev - slope_lower * (t[4] - t[3]),
          m.y_0.value + m.y_0.stddev
      ]
      # yapf: enable
      subplots[0].plot(t, y, color="y", linestyle="--")

      subplots[0].xaxis.set_major_formatter(formatter_t)
      subplots[0].yaxis.set_major_formatter(formatter_y)
      subplots[0].set_title("Simplified Eye")

      for key, curve in m.bathtub_curves.items():
        t = curve[0]
        y = curve[1]
        subplots[1].semilogy(t, y, label=key)
      subplots[1].set_title("Bathtub curves")
      subplots[1].legend(loc="upper center")

      pyplot.tight_layout()
      pyplot.savefig(debug_plots, bbox_inches="tight")
      pyplot.close()
      print(f"{'':>{indent}}Saved image to {debug_plots}")


def _filter_edge_polarity(edges: tuple[np.ndarray],
                          polarity: eyediagram.ClockPolarity) -> np.ndarray:
  """Filter edges to fit polarity restrictions and return

  ClockPolarity.BOTH concatenates then sorts rising and falling edges

  Args:
    edges: Tuple[rising, falling] List of edge timestamps, see edges.get_np
    polarity: Polarity of edges to filter to

  Returns:
    Single dimensional array of edges
  """
  if polarity is eyediagram.ClockPolarity.RISING:
    return edges[0]
  elif polarity is eyediagram.ClockPolarity.FALLING:
    return edges[1]
  return np.sort(np.concatenate(edges))


def _runner_levels(waveform_y: np.ndarray, n_max: int) -> dict:
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
    ratio = len(waveform_y) / n_max
    if ratio > 10:
      ratio = max(1, int(ratio / 2))
      waveform_y = waveform_y[::ratio]
    waveform_y = stats.downsample(waveform_y, n_max)

  y_min = waveform_y.min()
  y_max = waveform_y.max()
  y_mid = (y_min + y_max) / 2
  y_0 = stats.UncertainValue(0, 0)
  y_1 = stats.UncertainValue(0, 0)

  values_0 = waveform_y[np.where(waveform_y < y_mid)]
  y_0.stddev = values_0.std()

  values_1 = waveform_y[np.where(waveform_y > y_mid)]
  y_1.stddev = values_1.std()

  fit_0 = gaussian.fit_mix_samples(values_0, n_max=2)
  fit_1 = gaussian.fit_mix_samples(values_1, n_max=2)

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


def _runner_cdr(
    waveform_t: np.ndarray, waveform_y: np.ndarray, y_rise: float,
    y_half: float, y_fall: float, cdr_obj: cdr.CDR,
    polarity: eyediagram.ClockPolarity) -> Tuple[np.ndarray, np.ndarray]:
  """Recover a clock from the data signal

  Args:
    waveform_t: Waveform data array [t0, t1,..., tn]
    waveform_y: Waveform data array [y0, y1,..., yn]
    y_rise: Rising threshold
    y_half: Interpolated edge value
    y_fall: Falling threshold
    cdr_obj: CDR object to execute on waveform
    polarity: Polarity of data edges to clock off of

  Returns:
    List of clock edges in the time domain.
    List of Time Interval Errors (TIEs).
  """
  data_edges = lines.edges_np(waveform_t, waveform_y, y_rise, y_half, y_fall)
  return cdr_obj.run(_filter_edge_polarity(data_edges, polarity))
