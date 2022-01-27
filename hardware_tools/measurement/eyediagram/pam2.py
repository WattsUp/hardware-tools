"""PAM2 Encoded Eye Diagram, aka NRZ
"""

from __future__ import annotations

import copy

import colorama
from colorama import Fore
import numpy as np
from matplotlib import pyplot

from hardware_tools import math, strformat
from hardware_tools.extensions import edges, intersections
from hardware_tools.measurement.mask import Mask
from hardware_tools.measurement.eyediagram import cdr, eyediagram

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


class PAM2(eyediagram.EyeDiagram):
  """Eye Diagram to layer repeated waveforms and measure the resultant heat map

  Expects PAM2 (NRZ) encoded signal
  """

  def __init__(self,
               waveforms: np.ndarray,
               clocks: np.ndarray = None,
               t_unit: str = "",
               y_unit: str = "",
               mask: Mask = None,
               resolution: int = 2000,
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
                     resolution=resolution)
    self._y_0 = y_0
    self._y_1 = y_1

    self._config = {
        # Step 1
        "hysteresis": None,  # Difference between rising and falling thresholds
        "hysteresis_ua": 0.1,  # Units of normalized amplitude, lower priority
        "hist_n_max": 50e5,  # Maximum number of points in a histogram

        # Step 2
        "clock_polarity": eyediagram.ClockPolarity.RISING,  # If given clocks
        "cdr": None,  # Clock recovery algorithm, None will use cdr.CDR
        "fallback_period": 100e-9,  # If CDR cannot run (low SNR)

        # Step 3: No configuration

        # Step 4
        "level_width": 0.2,  # Width of histogram for y_0, y_1, UI
        "cross_width": 0.1,  # Width of histogram for y_cross, UI
        "hist_height": 0.05,  # Height of histogram for time windows, UA
        "edge_location": [0.2, 0.8],  # Location of thresholds for rise/fall
        # time, UA
        "noise_floor": math.UncertainValue(
            0, 0),  # Value of noise floor affecting extinction_ratio

        # Step 5: No configuration
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
                                     print_progress, indent + 2)

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

      pyplot.tight_layout()
      pyplot.savefig(debug_plots, bbox_inches="tight")
      pyplot.close()
      print(f"{'':>{indent}}Saved image to {debug_plots}")

  def _step2_clock(self,
                   n_threads: int = 1,
                   print_progress: bool = True,
                   indent: int = 0,
                   debug_plots: str = None) -> None:
    self._clock_edges = []
    periods = []
    ties = None
    if self._clocks is not None:
      # Get edges from _clocks given clock_polarity
      # yapf: disable
      args_list = [[
          self._clocks[i][0],
          self._clocks[i][1],
          self._y_rising,
          self._y_half,
          self._y_falling
      ] for i in range(self._waveforms.shape[0])]
      # yapf: enable
      output = self._collect_runners(edges.get_np, args_list, n_threads,
                                     print_progress, indent + 2)
      for o in output:
        if self._config["clock_polarity"] is eyediagram.ClockPolarity.RISING:
          e = o[0]
        elif self._config["clock_polarity"] is eyediagram.ClockPolarity.FALLING:
          e = o[1]
        else:
          e = np.sort(np.concatenate(o))
        periods.append(np.diff(e))
        self._clock_edges.append(e.tolist())
    elif self._low_snr:
      # Generate clock pulses at the fallback_period fixed rate
      self._t_sym = math.UncertainValue(self._config["fallback_period"], 0)

      for i in range(self._waveforms.shape[0]):
        t = self._waveforms[i][0][0] + self._t_sym.value
        e = np.arange(t, self._waveforms[i][0][-1], self._t_sym.value)
        self._clock_edges.append(e.tolist())
      return
    else:
      if print_progress:
        print(f"{'':>{indent}}Running clock data recovery")
      # Run CDR to generate edges
      if self._config["cdr"] is None:
        self._config["cdr"] = cdr.CDR(self._config["fallback_period"])
      # yapf: disable
      args_list = [[
          self._waveforms[i][0],
          self._waveforms[i][1],
          self._y_rising,
          self._y_half,
          self._y_falling,
          copy.deepcopy(self._config["cdr"]),
          self._config["clock_polarity"]
      ] for i in range(self._waveforms.shape[0])]
      # yapf: enable
      output = self._collect_runners(_runner_cdr, args_list, n_threads,
                                     print_progress, indent + 2)
      ties = []
      for o in output:
        self._clock_edges.append(o["edges"])
        periods.append(np.diff(o["edges"]))
        ties.append(o["ties"])

    if print_progress:
      print(f"{'':>{indent}}Calculating symbol period")
    # Two pass average to remove outliers arising from idle time
    periods = np.concatenate(periods)
    periods = periods[periods < 10 * periods.mean()]
    self._t_sym = math.UncertainValue(periods.mean(), periods.std())

    if debug_plots is not None:
      debug_plots += ".step2.png"

      def tick_formatter_t(t, _):
        return strformat.metric_prefix(t, self._t_unit)

      formatter_t = pyplot.FuncFormatter(tick_formatter_t)

      _, subplots = pyplot.subplots(2, 1)
      subplots[0].set_title(
          "Symbol Period Deviation from " +
          strformat.metric_prefix(self._t_sym.value, self._t_unit))
      subplots[0].hist((periods - self._t_sym.value),
                       50,
                       density=True,
                       color="b",
                       alpha=0.5)
      subplots[0].axvline(x=0, color="g")
      subplots[0].axvline(x=(self._t_sym.stddev), color="y")
      subplots[0].axvline(x=(-self._t_sym.stddev), color="y")
      subplots[0].xaxis.set_major_formatter(formatter_t)
      subplots[0].set_ylabel("Density")

      subplots[1].set_title("Time Interval Errors")
      if ties is not None:
        ties = np.concatenate(ties)
        subplots[1].hist(ties, 50, density=True, color="b", alpha=0.5)
      subplots[1].xaxis.set_major_formatter(formatter_t)
      subplots[1].set_ylabel("Density")
      subplots[1].set_xlabel("Time")

      pyplot.tight_layout()
      pyplot.savefig(debug_plots, bbox_inches="tight")
      pyplot.close()
      print(f"{'':>{indent}}Saved image to {debug_plots}")

  def _step4_measure(self,
                     n_threads: int = 1,
                     print_progress: bool = True,
                     indent: int = 0,
                     debug_plots: str = None) -> None:
    m = MeasuresPAM2()

    t_sym = self._t_sym.value

    if print_progress:
      print(f"{'':>{indent}}Measuring waveform vertically")

    # yapf: disable
    args_list = [[
        self._waveforms[i][1],
        self._centers_t[i],
        self._centers_i[i],
        self._t_delta,
        t_sym,
        self._y_half,
        self._config["level_width"],
        self._config["cross_width"]
    ] for i in range(self._waveforms.shape[0])]
    # yapf: enable
    output = self._collect_runners(_runner_sample_vertical, args_list,
                                   n_threads, print_progress, indent + 2)

    s_y_0 = np.array([])
    s_y_1 = np.array([])
    s_y_cross = np.array([])
    s_y_0_cross = np.array([])
    s_y_1_cross = np.array([])
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
    for o in output:
      s_y_0 = np.append(s_y_0, o["y_0"])
      s_y_1 = np.append(s_y_1, o["y_1"])
      s_y_cross = np.append(s_y_cross, o["y_cross"])
      s_y_0_cross = np.append(s_y_0_cross, o["y_0_cross"])
      s_y_1_cross = np.append(s_y_1_cross, o["y_1_cross"])
      for t in transitions:
        transitions[t] += o["transitions"][t]
      edge_dir.append(o["edge_dir"])

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

    m.y_0 = math.UncertainValue.samples(s_y_0)
    m.y_1 = math.UncertainValue.samples(s_y_1)
    m.y_cross = math.UncertainValue.samples(s_y_cross)
    m.y_0_cross = math.UncertainValue.samples(s_y_0_cross)
    m.y_1_cross = math.UncertainValue.samples(s_y_1_cross)
    m.transition_dist = transitions

    # Computed measures
    m.amp = m.y_1 - m.y_0
    m.height = (m.y_1 - 3 * m.y_1.stddev) - (m.y_0 + 3 * m.y_0.stddev)
    m.height_r = m.height / m.amp
    m.snr = m.amp / (m.y_1.stddev + m.y_0.stddev)
    m.y_cross_r = (m.y_cross - m.y_0) / m.amp

    # Computed measures, optical
    m.extinction_ratio = (m.y_1 - self._config["noise_floor"]) / (
        m.y_0 - self._config["noise_floor"])
    m.oma_cross = m.y_1_cross - m.y_0_cross
    # median_unbiased: "This method is probably the best method if the sample
    # distribution function is unknown"
    a_0 = np.percentile(s_y_1, 0.05, method="median_unbiased") - np.percentile(
        s_y_0, 99.95, method="median_unbiased")
    m.vecp = np.log10(m.oma_cross / a_0) * 10

    # Update levels if not manual
    if self._y_0 is None:
      self._y_zero = m.y_0.value
    if self._y_1 is None:
      self._y_ua = m.amp.value
    else:
      self._y_ua = self._y_1 - self._y_zero
    self._y_half = self._y_ua / 2 + self._y_zero

    if print_progress:
      print(f"{'':>{indent}}Measuring waveform horizontally")
    # yapf: disable
    args_list = [[
        self._waveforms[i][1],
        self._centers_t[i],
        self._centers_i[i],
        edge_dir[i],
        self._t_delta,
        t_sym,
        self._y_zero,
        self._y_ua,
        m.y_cross.value,
        self._config["hist_height"],
        self._config["edge_location"]
    ] for i in range(self._waveforms.shape[0])]
    # yapf: enable
    output = self._collect_runners(_runner_sample_horizontal, args_list,
                                   n_threads, print_progress, indent + 2)

    s_t_rise_lower = np.array([])
    s_t_rise_upper = np.array([])
    s_t_rise_half = np.array([])
    s_t_fall_lower = np.array([])
    s_t_fall_upper = np.array([])
    s_t_fall_half = np.array([])
    s_t_cross_left = np.array([])
    s_t_cross_right = np.array([])
    for o in output:
      s_t_rise_lower = np.append(s_t_rise_lower, o["t_rise_lower"])
      s_t_rise_upper = np.append(s_t_rise_upper, o["t_rise_upper"])
      s_t_rise_half = np.append(s_t_rise_half, o["t_rise_half"])
      s_t_fall_lower = np.append(s_t_fall_lower, o["t_fall_lower"])
      s_t_fall_upper = np.append(s_t_fall_upper, o["t_fall_upper"])
      s_t_fall_half = np.append(s_t_fall_half, o["t_fall_half"])
      s_t_cross_left = np.append(s_t_cross_left, o["t_cross_left"])
      s_t_cross_right = np.append(s_t_cross_right, o["t_cross_right"])

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

    t_rise_lower = math.UncertainValue.samples(s_t_rise_lower)
    t_rise_upper = math.UncertainValue.samples(s_t_rise_upper)
    t_rise_half = math.UncertainValue.samples(s_t_rise_half)
    t_fall_lower = math.UncertainValue.samples(s_t_fall_lower)
    t_fall_upper = math.UncertainValue.samples(s_t_fall_upper)
    t_fall_half = math.UncertainValue.samples(s_t_fall_half)
    t_cross_left = math.UncertainValue.samples(s_t_cross_left)
    t_cross_right = math.UncertainValue.samples(s_t_cross_right)

    # Computed measures
    m.t_sym = t_cross_right - t_cross_left
    m.t_0 = t_rise_half - t_fall_half + t_sym
    m.t_1 = t_fall_half - t_rise_half + t_sym
    m.t_rise = t_rise_upper - t_rise_lower
    m.t_fall = t_fall_lower - t_fall_upper
    m.t_cross = t_cross_left

    m.f_sym = math.UncertainValue(1, 0) / m.t_sym

    m.width = (t_cross_right - 3 * t_cross_right.stddev) - (
        t_cross_left + 3 * t_cross_left.stddev)
    m.width_r = m.width / m.t_sym
    m.dcd = (m.t_1 - m.t_0) / (m.t_sym * 2)
    m.jitter_pp = s_t_cross_left.ptp()
    m.jitter_rms = t_cross_left.stddev

    if print_progress:
      print(f"{'':>{indent}}Measuring waveform mask")

    if self._mask is None:
      m.mask_margin = np.nan
      m.n_sym_bad = np.nan
    else:
      # yapf: disable
      args_list = [[
          self._waveforms[i][1],
          self._centers_t[i],
          self._centers_i[i],
          self._t_delta,
          t_sym,
          self._y_zero,
          self._y_ua,
          self._mask
      ] for i in range(self._waveforms.shape[0])]
      # yapf: enable
      output = self._collect_runners(_runner_sample_mask, args_list, n_threads,
                                     print_progress, indent + 2)

      self._offenders = []
      self._hits = []
      margin = 1.0
      offender_count = 0
      for o in output:
        self._offenders.append(o["offenders"])
        offender_count += len(o["offenders"])
        self._hits.extend(o["hits"])
        margin = min(margin, o["margin"])

      m.mask_margin = margin
      m.n_sym_bad = offender_count

    self._measures = m
    for k, v in m.to_dict().items():
      if v is None:
        print(f"{k:20}: None")
      elif isinstance(v, math.UncertainValue):
        print(f"{k:20}: {v.value:8.6e} {v.stddev:8.6e}")
      elif isinstance(v, float):
        print(f"{k:20}: {v:8.6e}")
      elif isinstance(v, int):
        print(f"{k:20}: {v:8}")
      else:
        print(f"{k:20}: Not printable")

    if debug_plots is not None:
      debug_plots += ".step4.png"

      def tick_formatter_t(t, _):
        return strformat.metric_prefix(t, self._t_unit)

      formatter_t = pyplot.FuncFormatter(tick_formatter_t)

      def tick_formatter_y(y, _):
        return strformat.metric_prefix(y, self._y_unit)

      formatter_y = pyplot.FuncFormatter(tick_formatter_y)

      # 00 path
      t = [-0.5 * t_sym, 0.0, 0.5 * t_sym]
      y = [m.y_0.value, m.y_0_cross.value, m.y_0.value]
      pyplot.plot(t, y, color="r")
      # yapf: disable
      y = [
          m.y_0.value + m.y_0.stddev,
          m.y_0_cross.value + m.y_0_cross.stddev,
          m.y_0.value + m.y_0.stddev
      ]
      # yapf: enable
      pyplot.plot(t, y, color="r", linestyle="--")
      # yapf: disable
      y = [
          m.y_0.value - m.y_0.stddev,
          m.y_0_cross.value - m.y_0_cross.stddev,
          m.y_0.value - m.y_0.stddev
      ]
      # yapf: enable
      pyplot.plot(t, y, color="r", linestyle="--")

      # 11 path
      t = [-0.5 * t_sym, 0.0, 0.5 * t_sym]
      y = [m.y_1.value, m.y_1_cross.value, m.y_1.value]
      pyplot.plot(t, y, color="g")
      # yapf: disable
      y = [
          m.y_1.value + m.y_1.stddev,
          m.y_1_cross.value + m.y_1_cross.stddev,
          m.y_1.value + m.y_1.stddev
      ]
      # yapf: enable
      pyplot.plot(t, y, color="g", linestyle="--")
      # yapf: disable
      y = [
          m.y_1.value - m.y_1.stddev,
          m.y_1_cross.value - m.y_1_cross.stddev,
          m.y_1.value - m.y_1.stddev
      ]
      # yapf: enable
      pyplot.plot(t, y, color="g", linestyle="--")

      # 01 path
      y_lower = self._config["edge_location"][0] * self._y_ua + self._y_zero
      y_upper = self._config["edge_location"][1] * self._y_ua + self._y_zero
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
      pyplot.plot(t, y, color="b")
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
      pyplot.plot(t, y, color="b", linestyle="--")
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
      pyplot.plot(t, y, color="b", linestyle="--")

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
      pyplot.plot(t, y, color="y")
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
      pyplot.plot(t, y, color="y", linestyle="--")
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
      pyplot.plot(t, y, color="y", linestyle="--")

      ax = pyplot.gca()
      ax.xaxis.set_major_formatter(formatter_t)
      ax.yaxis.set_major_formatter(formatter_y)

      pyplot.title("Simplified Eye")
      pyplot.tight_layout()
      pyplot.savefig(debug_plots, bbox_inches="tight")
      pyplot.close()
      print(f"{'':>{indent}}Saved image to {debug_plots}")


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
    waveform_y = math.Bin.downsample(waveform_y, n_max)

  y_min = waveform_y.min()
  y_max = waveform_y.max()
  y_mid = (y_min + y_max) / 2
  y_0 = math.UncertainValue(0, 0)
  y_1 = math.UncertainValue(0, 0)

  values_0 = waveform_y[np.where(waveform_y < y_mid)]
  y_0.stddev = values_0.std()

  values_1 = waveform_y[np.where(waveform_y > y_mid)]
  y_1.stddev = values_1.std()

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


def _runner_cdr(waveform_t: np.ndarray, waveform_y: np.ndarray, y_rise: float,
                y_half: float, y_fall: float, cdr_obj: cdr.CDR,
                polarity: eyediagram.ClockPolarity) -> dict:
  """Recover a clock from the data signal

  Args:
    waveform_t: Waveform data array [t0, t1,..., tn]
    waveform_y: Waveform data array [y0, y1,..., yn]
    y_rise: Rising threshold
    y_half: Interpolated edge value
    y_fall: Falling threshold
    cdr_obj: CDR object to execute on waveform

  Returns:
    Dictionary of values:
      edges: List of clock edges in time
      ties: List of Time Interval Errors (TIEs)
  """
  data_edges = edges.get_np(waveform_t, waveform_y, y_rise, y_half, y_fall)
  if polarity is eyediagram.ClockPolarity.RISING:
    data_edges = data_edges[0]
  elif polarity is eyediagram.ClockPolarity.FALLING:
    data_edges = data_edges[1]
  else:
    data_edges = np.sort(np.concatenate(data_edges))
  results = cdr_obj.run(data_edges)
  return {"edges": results[0], "ties": results[1]}


def _runner_sample_vertical(waveform_y: np.ndarray, centers_t: list[float],
                            centers_i: list[int], t_delta: float, t_sym: float,
                            y_half: float, level_width: float,
                            cross_width: float) -> dict:
  """Measure vertical parameters

  Args:
    waveform_y: Waveform data array [y0, y1,..., yn]
    centers_t: List of symbol centers in time for sub t_delta alignment.
      Grid spans [-0.5*t_sym, 1.5*t_sym] + center_t
    centers_i: List of symbol centers indices
    t_delta: Time between samples
    t_sym: Duration of one symbol
    y_half: Decision threshold for a low or high symbol
    level_width: Width of y_0, y_1 windows, UI
    cross_width: Width of y_cross window, UI

  Returns:
    Dictionary of values:
      y_0: List of samples within the y_0 window, logical 0
      y_1: List of samples within the y_1 window, logical 1
      y_cross: List of samples within the y_cross window, edge
      y_0_cross: List of samples within the y_cross window, logical 0
      y_1_cross: List of samples within the y_cross window, logical 1
      transitions: Dictionary of collected transitions, see MeasuresPAM2
      edge_dir: List of edge directions, True=rising, False=falling, None=none
  """
  i_width = int((t_sym / t_delta) + 0.5) + 2
  t_width_ui = (i_width * t_delta / t_sym)

  n = i_width * 2 + 1
  t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, n)

  abc_width = max(level_width, 1.01 * t_delta / t_sym)
  t_a_min = -0.5 - abc_width / 2
  t_a_max = -0.5 + abc_width / 2
  t_b_min = 0.5 - abc_width / 2
  t_b_max = 0.5 + abc_width / 2
  t_c_min = 1.5 - abc_width / 2
  t_c_max = 1.5 + abc_width / 2

  t_sym_min = 0.5 - level_width / 2
  t_sym_max = 0.5 + level_width / 2

  t_cross_min = 0.0 - cross_width / 2
  t_cross_max = 0.0 + cross_width / 2

  t_sym_cross_min = 0.5 - level_width / 2
  t_sym_cross_max = 0.5 + level_width / 2

  values = {
      "y_0": [],
      "y_1": [],
      "y_cross": [],
      "y_0_cross": [],
      "y_1_cross": [],
      "transitions": {
          "000": 0,
          "001": 0,
          "010": 0,
          "011": 0,
          "100": 0,
          "101": 0,
          "110": 0,
          "111": 0,
      },
      "edge_dir": []
  }

  for i in range(len(centers_t)):
    c_i = centers_i[i]
    c_t = centers_t[i] / t_sym

    samples_a = []
    samples_b = []
    samples_c = []

    samples_sym = []
    samples_cross = []
    samples_sym_cross = []
    for ii in range(n):
      t = t0[ii] + c_t
      y = waveform_y[c_i - i_width + ii]

      if t_a_min <= t <= t_a_max:
        samples_a.append(y)
      elif t_b_min <= t <= t_b_max:
        samples_b.append(y)
      elif t_c_min <= t <= t_c_max:
        samples_c.append(y)

      if t_sym_min <= t <= t_sym_max:
        samples_sym.append(y)

      if t_cross_min <= t <= t_cross_max:
        samples_cross.append(y)

      if t_sym_cross_min <= t <= t_sym_cross_max:
        samples_sym_cross.append(y)

    sym_a = np.mean(samples_a) > y_half
    sym_b = np.mean(samples_b) > y_half
    sym_c = np.mean(samples_c) > y_half

    if sym_a != sym_b:
      values["y_cross"].extend(samples_cross)
      values["edge_dir"].append(sym_b)
    else:
      values["edge_dir"].append(None)
      if sym_b:
        values["y_1_cross"].extend(samples_sym_cross)
      else:
        values["y_0_cross"].extend(samples_sym_cross)

    seq = "1" if sym_a else "0"

    if sym_b:
      values["y_1"].extend(samples_sym)
      seq += "1"
    else:
      values["y_0"].extend(samples_sym)
      seq += "0"

    seq += "1" if sym_c else "0"

    values["transitions"][seq] += 1

  return values


def _runner_sample_horizontal(waveform_y: np.ndarray, centers_t: list[float],
                              centers_i: list[int], edge_dir: list[bool],
                              t_delta: float, t_sym: float, y_zero: float,
                              y_ua: float, y_cross: float, hist_height: float,
                              edge_location: list[float]) -> dict:
  """Measure horizontal parameters

  Args:
    waveform_y: Waveform data array [y0, y1,..., yn]
    centers_t: List of symbol centers in time for sub t_delta alignment.
      Grid spans [-0.5*t_sym, 1.5*t_sym] + center_t
    centers_i: List of symbol centers indices
    edge_dir: List of edge directions, True=rising, False=falling, None=none
    t_delta: Time between samples
    t_sym: Duration of one symbol
    y_zero: Amplitude of a logical 0
    y_ua: Normalized amplitude
    y_cross: Amplitude of crossing point
    hist_height: Height of time windows, UA
    edge_location: Location of upper and lower edge windows, UA

  Returns:
    Dictionary of values, all UI:
      t_rise_lower: List of samples within the lower edge window, rising
      t_rise_upper: List of samples within the upper edge window, rising
      t_rise_half: List of samples within the 50% window, rising
      t_fall_lower: List of samples within the lower edge window, falling
      t_fall_upper: List of samples within the upper edge window, falling
      t_fall_half: List of samples within the 50% window, falling
      t_cross_left: List of samples within the cross window, left edge
      t_cross_right: List of samples within the cross window, right edge
  """
  i_width = int((t_sym / t_delta) + 0.5) + 2
  t_width_ui = (i_width * t_delta / t_sym)

  n = i_width * 2 + 1
  t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, n)
  center_offset = int(n * 1 / 4)

  waveform_y = (waveform_y - y_zero) / y_ua
  y_cross = (y_cross - y_zero) / y_ua

  y_lower_min = edge_location[0] - hist_height / 2
  y_lower_max = edge_location[0] + hist_height / 2
  y_upper_min = edge_location[1] - hist_height / 2
  y_upper_max = edge_location[1] + hist_height / 2

  y_cross_min = y_cross - hist_height / 2
  y_cross_max = y_cross + hist_height / 2

  y_half_min = 0.5 - hist_height / 2
  y_half_max = 0.5 + hist_height / 2

  values = {
      "t_rise_lower": [],
      "t_rise_upper": [],
      "t_rise_half": [],
      "t_fall_lower": [],
      "t_fall_upper": [],
      "t_fall_half": [],
      "t_cross_left": [],
      "t_cross_right": []
  }

  for i in range(len(centers_t)):
    c_i = centers_i[i]
    c_t = centers_t[i] / t_sym

    for ii in range(n):
      t = t0[ii] + c_t
      y = waveform_y[c_i - i_width + ii]

      if y_cross_min <= y <= y_cross_max:
        if -0.5 <= t <= 0.5:
          values["t_cross_left"].append(t)
        elif 0.5 <= t <= 1.5:
          values["t_cross_right"].append(t)

    if edge_dir[i] is None:
      continue
    y_front = waveform_y[c_i - i_width:c_i + 1]
    y_center = waveform_y[c_i - i_width + center_offset:c_i + center_offset + 1]

    if edge_dir[i]:
      # Rising edge starts at minimum on [-0.5, 0.5]
      ii = np.argmin(y_front)
      # Stop at maximum on [0.0, 1.0]
      ii_stop = center_offset + np.argmax(y_center)
    else:
      # Falling edge starts at maximum on [-0.5, 0.5]
      ii = np.argmax(y_front)
      # Stop at minimum on [0.0, 1.0]
      ii_stop = center_offset + np.argmin(y_center)
    while ii <= ii_stop:
      t = t0[ii] + c_t
      y = waveform_y[c_i - i_width + ii]

      if y_lower_min <= y <= y_lower_max:
        if edge_dir[i]:
          values["t_rise_lower"].append(t)
        else:
          values["t_fall_lower"].append(t)

      if y_upper_min <= y <= y_upper_max:
        if edge_dir[i]:
          values["t_rise_upper"].append(t)
        else:
          values["t_fall_upper"].append(t)

      if y_half_min <= y <= y_half_max:
        if edge_dir[i]:
          values["t_rise_half"].append(t)
        else:
          values["t_fall_half"].append(t)

      ii += 1

  return values


def _runner_sample_mask(waveform_y: np.ndarray, centers_t: list[float],
                        centers_i: list[int], t_delta: float, t_sym: float,
                        y_zero: float, y_ua: float, mask: Mask) -> dict:
  """Measure mask parameters

  Args:
    waveform_y: Waveform data array [y0, y1,..., yn]
    centers_t: List of symbol centers in time for sub t_delta alignment.
      Grid spans [-0.5*t_sym, 1.5*t_sym] + center_t
    centers_i: List of symbol centers indices
    t_delta: Time between samples
    t_sym: Duration of one symbol
    y_zero: Amplitude of a logical 0
    y_ua: Normalized amplitude
    mask: Mask to test to

  Returns:
    Dictionary of values:
      offenders: List of bit indices that hit the mask
      hits: List of mask collision coordinates [[UI, UA],...]
      margin: Largest mask margin with zero hits [-1.0, 1.0]
  """
  i_width = int((t_sym / t_delta) + 0.5) + 2
  t_width_ui = (i_width * t_delta / t_sym)

  n = i_width * 2 + 1
  t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, n)

  waveform_y = (waveform_y - y_zero) / y_ua

  values = {"offenders": [], "hits": [], "margin": 1.0}

  mask_adj = mask.adjust(values["margin"])

  for i in range(len(centers_t)):
    c_i = centers_i[i]
    c_t = centers_t[i] / t_sym

    t = t0 + c_t
    y = waveform_y[c_i - i_width:c_i + i_width + 1]

    hits = intersections.get_hits_np(t, y, mask.paths)
    if hits.size > 0:
      values["offenders"].append(i)
      values["hits"].extend(hits.tolist())

    while intersections.is_hitting_np(
        t, y, mask_adj.paths) and values["margin"] > -1.0:
      values["margin"] -= 0.001
      mask_adj = mask.adjust(values["margin"])

  return values
