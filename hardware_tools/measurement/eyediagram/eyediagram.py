"""Eye Diagram to layer repeated waveforms and measure the resultant heat map
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import datetime
from enum import Enum
import io
import json
import os
from typing import Any, List, Union

import colorama
from colorama import Fore
from matplotlib import pyplot
import numpy as np
import skimage.draw

from hardware_tools import strformat
from hardware_tools.math import image, stats
from hardware_tools.measurement.mask import Mask

try:
  from hardware_tools.measurement.eyediagram import _eyediagram
except ImportError:
  print(f"The cython version of {__name__} is not available")
  from hardware_tools.measurement.eyediagram import _eyediagram_fb as _eyediagram

colorama.init(autoreset=True)


class ClockPolarity(Enum):
  RISING = 1  # Sample on clock's rising edge
  FALLING = 2  # Sample on clock's falling edge
  BOTH = 3  # Sample on both edges


class MeasuresJSONEncoder(json.JSONEncoder):
  """JSON Encoder for Measures
  """

  def default(self, o: Any) -> Any:
    if isinstance(o, stats.UncertainValue):
      return {
          "__type__": stats.UncertainValue.__qualname__,
          "value": o.value,
          "stddev": o.stddev
      }
    elif isinstance(o, bytes):
      return o.decode(encoding="utf-8")
    elif isinstance(o, np.integer):
      return int(o)
    elif isinstance(o, np.floating):
      return float(o)
    elif isinstance(o, np.ndarray):
      return o.tolist()
    return super().default(o)


class Measures(ABC):
  """Eye Diagram Measures collecting metrics from an eye diagram

  Must be derrived to support proper signal encoding

  All images are base64 encoded PNGs, use save_images to save to disk

  Properties:
    n_samples: Number of samples collected
    n_sym: Number of symbols tested
    n_sym_bad: Number of symbols that failed, requires a mask
    transition_dist: Distribution of transitions plotted eg:
      000, 001, 010, 011, 100, 101, 110, 111
    mask_margin: largest mask adjust without mask hits, requires a mask
      range [-1, 1], see Mask.adjust
    image_clean: layered waveforms heatmap
    image_grid: grid of UI and UA values
    image_mask: image of unadjusted mask
    image_hits: subset of mask hits and offending waveforms
    image_margin: image of margin adjusted mask
    bathtub_curves: bathtub BER curves
      dict{
        slice_level_0: np.array([[t0, t1,..., tn], [ber0, ber1,..., bern]]),
        slice_level_1:...}
  """

  def __init__(self) -> None:
    """Create a new Measures object
    """
    super().__init__()

    self.n_samples = None
    self.n_sym = None
    self.n_sym_bad = None
    self.transition_dist = None
    self.mask_margin = None

    self._np_image_clean = None
    self._np_image_grid = None
    self._np_image_mask = None
    self._np_image_hits = None
    self._np_image_margin = None

    self.bathtub_curves = None

  def set_images(self, np_clean: np.ndarray, np_grid: np.ndarray,
                 np_mask: np.ndarray, np_hits: np.ndarray,
                 np_margin: np.ndarray) -> None:
    """Set images

    Images are numpy format, RGBA, shape=(rows, columns, 4), [0.0, 1.0]

    Args:
      np_clean: numpy image for image_clean
      np_grid: numpy image for image_grid
      np_mask: numpy image for image_mask
      np_hits: numpy image for image_hits
      np_margin: numpy image for image_margin
    """
    self._np_image_clean = np_clean
    self._np_image_grid = np_grid
    self._np_image_mask = np_mask
    self._np_image_hits = np_hits
    self._np_image_margin = np_margin

  def save_images(self,
                  basename: os.PathLike = "eyediagram",
                  stack: bool = False) -> None:
    """Save images to PNG file

    Args:
      basename: base file name for images.
        image_clean will be saved to "{basename}.clean.png"
        image_grid will be saved to "{basename}.grid.png"
        image_mask will be saved to "{basename}.mask.png"
        image_hits will be saved to "{basename}.hits.png"
        image_margin will be saved to "{basename}.mask_margin.png"
      stack: True will stack grid, clean, mask, and hits and save to
        Default mask: "{basename}.png"
        Margin mask: "{basename}.margin.png"
    """
    image.np_to_file(self._np_image_clean, f"{basename}.clean.png")
    image.np_to_file(self._np_image_grid, f"{basename}.grid.png")
    image.np_to_file(self._np_image_mask, f"{basename}.mask.png")
    image.np_to_file(self._np_image_hits, f"{basename}.hits.png")
    image.np_to_file(self._np_image_margin, f"{basename}.mask_margin.png")
    if stack:
      img = image.layer_rgba(self._np_image_grid, self._np_image_clean)
      img = image.layer_rgba(img, self._np_image_mask)
      img = image.layer_rgba(img, self._np_image_hits)
      image.np_to_file(img, f"{basename}.png")

      img = image.layer_rgba(self._np_image_grid, self._np_image_clean)
      img = image.layer_rgba(img, self._np_image_margin)
      image.np_to_file(img, f"{basename}.margin.png")

  def save_json(self,
                filename: os.PathLike = "eyediagram.json",
                exclude_images: bool = False,
                **kwargs) -> None:
    """Save images to PNG file

    Args:
      filename: File name for json file
      exclude_images: True will not include the base64 encoded images
      Additional arguments passed to json.dump
    """
    with open(filename, "w", encoding="utf-8") as file:
      json.dump(self.to_dict(exclude_images=exclude_images),
                file,
                cls=MeasuresJSONEncoder,
                **kwargs)

  @property
  def image_clean(self) -> bytes:
    if self._np_image_clean is None:
      return None
    return image.np_to_base64(self._np_image_clean)

  @property
  def image_grid(self) -> bytes:
    if self._np_image_grid is None:
      return None
    return image.np_to_base64(self._np_image_grid)

  @property
  def image_mask(self) -> bytes:
    if self._np_image_mask is None:
      return None
    return image.np_to_base64(self._np_image_mask)

  @property
  def image_hits(self) -> bytes:
    if self._np_image_hits is None:
      return None
    return image.np_to_base64(self._np_image_hits)

  @property
  def image_margin(self) -> bytes:
    if self._np_image_margin is None:
      return None
    return image.np_to_base64(self._np_image_margin)

  def to_dict(self, exclude_images: bool = False) -> dict:
    """Convert Measures to dictionary of values

    Args:
      exclude_images: True will not include the base64 encoded images

    Returns:
      dictionary of metrics and images (base64 encoded PNGs)
    """
    properties = dir(self)
    d = {}
    for key in properties:
      if key.startswith("_"):
        continue
      if exclude_images and key.startswith("image"):
        continue
      if hasattr(self.__class__, key):
        attr = getattr(self.__class__, key)
        if not isinstance(attr, property):
          continue
        item = getattr(self, key)
      else:
        item = getattr(self, key)
      d[key] = item
    return d

  def pretty_print(self) -> None:
    """Print measures beautifully
    """
    properties = dir(self)
    properties_filtered = []
    max_length = 0
    for key in properties:
      if key.startswith("_"):
        continue
      if hasattr(self.__class__, key):
        attr = getattr(self.__class__, key)
        if not isinstance(attr, property):
          continue
        properties_filtered.append(key)
        max_length = max(max_length, len(key))
      else:
        properties_filtered.append(key)
        max_length = max(max_length, len(key))
    format_specifiers = {
        float: "10.3G",
        np.floating: "10.3G",
        int: "10",
        np.integer: "10",
        stats.UncertainValue: "10.3G"
    }
    for key in properties_filtered:
      print(f"{key:{max_length}}: ", end="")
      if key.startswith("image"):
        print("[Image]")
        continue
      item = getattr(self, key)
      if isinstance(item, dict):
        print("")
        kk_max = max([len(kk) for kk in item])
        for kk, vv in item.items():
          print(f"  {kk:{kk_max}}: ", end="")
          if isinstance(vv, np.ndarray):
            print(f"[np.ndarray, shape={vv.shape}, dtype={vv.dtype}]")
          else:
            matched = False
            for t, f in format_specifiers.items():
              if isinstance(vv, t):
                print(f"{vv:{f}}")
                matched = True
                break
            if not matched:
              print(f"[Unknown, type={type(vv).__qualname__}]")
      else:
        matched = False
        for t, f in format_specifiers.items():
          if isinstance(item, t):
            print(f"{item:{f}}")
            matched = True
            break
        if not matched:
          print(f"[Unknown, type={type(item).__qualname__}]")


class Config():
  """EyeDiagram configuration

  Extend for additional configuration specific to derrived EyeDiagram

  Properties:
    hysteresis: float, difference between rising and falling thresholds
    hysteresis_ua: float,  units of normalized amplitude, lower priority
    levels_n_max: int, Maximum of points in levels histogram

    clock_polarity: ClockPolarity, clock off rising, falling, or both edges
    cdr: cdr.CDR, clock recovery algorithm, None will use cdr.CDR
    fallback_period: float, if CDR cannot run (low SNR) fallback period to
      generate constant clock

    skip_measures: bool, True will not measure anything
    level_width: float, width of histogram for y levels, UI
    cross_width: float, width of histogram for y cross levels, UI
    time_hight: float, height of histogram for edge times, UA
    edge_lower: float, lower threshold for edge times, UA. Rising edge start.
    edge_upper: float, upper threshold for edge times, UA. Rising edge stop.
    noise_floor: math.UncertainValue, value of noise floor affecting y levels

    point_cloud: bool, True will disable interpolation when stacking
  """

  def __init__(self, **kwargs) -> None:
    """Initialize Configuration

    Args:
      All kwargs passed to self.consume
    """
    # Step 1
    self.hysteresis = None
    self.hysteresis_ua = 0.1
    self.levels_n_max = 10e3

    # Step 2
    self.clock_polarity = ClockPolarity.RISING
    self.cdr = None
    self.fallback_period = 100e-9

    # Step 3

    # Step 4
    self.skip_measures = False
    self.level_width = 0.2
    self.cross_width = 0.1
    self.time_height = 0.05
    self.edge_lower = 0.2
    self.edge_upper = 0.8
    self.noise_floor = stats.UncertainValue(0, 0)

    # Step 5
    self.point_cloud = False

    self.consume(kwargs)

  def consume(self, config: dict) -> None:
    """Consume configuration from dictionary

    Args:
      config: Dictionary of configuration values

    Raises:
      KeyError if an configuration parameter is unrecognized
    """
    for k, v in config.items():
      if not hasattr(self, k):
        raise KeyError(f"Unrecognized additional configuration: {k}={v}")
      setattr(self, k, v)


class EyeDiagram(ABC):
  """Eye Diagram to layer repeated waveforms and measure the resultant heat map

  Must be derrived to support proper signal encoding
  """

  def __init__(self,
               waveforms: np.ndarray,
               clocks: np.ndarray = None,
               clock_edges: List[List[float]] = None,
               t_unit: str = "",
               y_unit: str = "",
               mask: Mask = None,
               resolution: int = 2000,
               config: Config = None) -> None:
    """Create a new EyeDiagram. Lazy creation, does not compute anything

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
    super().__init__()
    if len(waveforms.shape) == 2:
      waveforms = np.array([waveforms])
    elif len(waveforms.shape) != 3:
      raise ValueError("waveforms is not 2D or 3D")

    if waveforms.shape[1] != 2 or waveforms.shape[0] > waveforms.shape[2]:
      raise ValueError("waveforms should be shape [n waveforms, 2, n points]")

    if clocks is not None:
      if len(clocks.shape) == 2:
        clocks = np.array([clocks])
      elif len(clocks.shape) != 3:
        raise ValueError("clocks is not 2D or 3D")

      if clocks.shape != waveforms.shape:
        raise ValueError("clocks is not same shape as waveforms")

    if clock_edges is not None:
      if not isinstance(clock_edges, list):
        raise ValueError("clock_edges is not list")
      if len(clock_edges) != waveforms.shape[0]:
        raise ValueError(
            "clock_edges does not have the same number of waveforms")
      if not isinstance(clock_edges[0], list):
        raise ValueError("clock_edges is not 2D")

    self._waveforms = waveforms
    self._clocks = clocks

    self._t_unit = t_unit
    self._y_unit = y_unit
    self._mask = mask
    self._mask_converted = None

    self._resolution = resolution
    self._raw_heatmap = np.zeros((resolution, resolution), dtype=np.int32)
    self._measures = None

    self._y_zero = None
    self._y_ua = None
    self._centers_t = None
    self._centers_i = None
    self._t_delta = (self._waveforms[0, 0, -1] -
                     self._waveforms[0, 0, 0]) / (self._waveforms.shape[2] - 1)
    self._t_sym = None
    self._clock_edges = clock_edges
    self._ties = None

    self._calculated = False

    if config is None:
      self._config = Config()
    else:
      self._config = config

  def calculate(self,
                print_progress: bool = True,
                indent: int = 0,
                debug_plots: str = None) -> Measures:
    """Perform eye diagram calculation

    Args:
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.

    Returns:
      Measures object containing resultant metrics, specific to line encoding
    """
    start = datetime.datetime.now()

    if print_progress:
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.CYAN}"
            f"Starting eye diagram calculation")
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.YELLOW}"
            "Step 1: Finding threshold levels")
    self._step1_levels(print_progress=print_progress,
                       indent=indent + 2,
                       debug_plots=debug_plots)

    if print_progress:
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.YELLOW}"
            "Step 2: Determining receiver clock")
    self._step2_clock(print_progress=print_progress,
                      indent=indent + 2,
                      debug_plots=debug_plots)

    if print_progress:
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.YELLOW}"
            "Step 3: Aligning symbol sampling points")
    self._step3_sample(print_progress=print_progress,
                       indent=indent + 2,
                       debug_plots=debug_plots)

    if print_progress:
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.YELLOW}"
            "Step 4: Measuring waveform")
    self._step4_measure(print_progress=print_progress,
                        indent=indent + 2,
                        debug_plots=debug_plots)

    if print_progress:
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.YELLOW}"
            "Step 5: Stacking waveforms")
    self._step5_stack(print_progress=print_progress,
                      indent=indent + 2,
                      debug_plots=debug_plots)

    self._calculated = True

    if print_progress:
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.CYAN}"
            "Completed eye diagram calculation")
    return self._measures

  @abstractmethod
  def _step1_levels(self,
                    print_progress: bool = True,
                    indent: int = 0,
                    debug_plots: str = None) -> None:
    """Calculation step 1: levels

    Dependent on line encoding

    Args:
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.
    """
    # Derrived classes set these parameters
    self._y_zero = 0.0  # pragma: no cover
    self._y_ua = 0.0  # pragma: no cover

  def _step2_clock(self,
                   print_progress: bool = True,
                   indent: int = 0,
                   debug_plots: str = None) -> None:
    """Calculation step 2: clock recovery

    Dependent on line encoding and clocks (passed during initialization).
    This method will use config.fallback_period if self._clock_edges is None

    Args:
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.
    """
    if self._clock_edges is None:
      self._clock_edges = []
      self._ties = []
      # Generate clock pulses at the fallback_period fixed rate
      for i in range(self._waveforms.shape[0]):
        t = self._waveforms[i][0][0] + self._config.fallback_period
        e = np.arange(t, self._waveforms[i][0][-1],
                      self._config.fallback_period)
        self._clock_edges.append(e.tolist())
        self._ties.append([])

    if print_progress:
      print(f"{'':>{indent}}Calculating symbol period")
    periods = []
    for edges in self._clock_edges:
      periods.append(np.diff(edges))

    # Two pass average to remove outliers arising from idle time
    periods = np.concatenate(periods)
    periods = periods[periods < 10 * periods.mean()]
    t_sym = stats.UncertainValue.samples(periods)
    self._t_sym = t_sym.value

    if debug_plots is not None:
      debug_plots += ".step2.png"

      def tick_formatter_t(t, _):
        return strformat.metric_prefix(t, self._t_unit)

      formatter_t = pyplot.FuncFormatter(tick_formatter_t)

      _, subplots = pyplot.subplots(2, 1)
      subplots[0].set_title("Symbol Period Deviation from " +
                            strformat.metric_prefix(t_sym.value, self._t_unit))
      subplots[0].hist((periods - t_sym.value),
                       50,
                       density=True,
                       color="b",
                       alpha=0.5)
      subplots[0].axvline(x=0, color="g")
      subplots[0].axvline(x=(t_sym.stddev), color="y")
      subplots[0].axvline(x=(-t_sym.stddev), color="y")
      subplots[0].xaxis.set_major_formatter(formatter_t)
      subplots[0].set_ylabel("Density")

      subplots[1].set_title("Time Interval Errors")
      ties = np.concatenate(self._ties)
      if ties.size > 0:
        subplots[1].hist(ties, 50, density=True, color="b", alpha=0.5)
      subplots[1].xaxis.set_major_formatter(formatter_t)
      subplots[1].set_ylabel("Density")
      subplots[1].set_xlabel("Time")

      pyplot.tight_layout()
      pyplot.savefig(debug_plots, bbox_inches="tight")
      pyplot.close()
      print(f"{'':>{indent}}Saved image to {debug_plots}")

  def _step3_sample(self,
                    print_progress: bool = True,
                    indent: int = 0,
                    debug_plots: str = None) -> None:
    """Calculation step 3: symbol sample point

    Args:
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.
    """
    self._centers_i = []
    self._centers_t = []

    i_width = int((self._t_sym / self._t_delta) + 0.5) + 2
    max_i = self._waveforms.shape[2]

    if print_progress:
      print(f"{'':>{indent}}Starting sampling")
    for i in range(self._waveforms.shape[0]):
      centers_i = []
      centers_t = []
      t_zero = self._waveforms[i, 0, 0]
      for b in self._clock_edges[i]:
        center_t = b % self._t_delta
        center_i = int(((b - t_zero - center_t) / self._t_delta) + 0.5)

        if (center_i - i_width) < 0 or (center_i + i_width) >= max_i:
          continue
        centers_t.append(-center_t)
        centers_i.append(center_i)

      self._centers_i.append(centers_i)
      self._centers_t.append(centers_t)

    if print_progress:
      print(f"{'':>{indent}}Completed sampling")

    if debug_plots is not None:
      debug_plots += ".step3.png"

      def tick_formatter_t(t, _):
        return strformat.metric_prefix(t, self._t_unit)

      def tick_formatter_y(y, _):
        return strformat.metric_prefix(y, self._y_unit)

      formatter_t = pyplot.FuncFormatter(tick_formatter_t)
      formatter_y = pyplot.FuncFormatter(tick_formatter_y)

      hw = int((self._t_sym / self._t_delta) + 0.5)
      n = hw * 2 + 1

      for i in range(self._waveforms.shape[0]):
        n_plot = min(len(self._centers_i[i]),
                     max(1, int(20 / self._waveforms.shape[0])))
        for edge in range(n_plot):
          c_i = self._centers_i[i][edge]
          c_t = self._centers_t[i][edge]
          x = np.linspace(-hw * self._t_delta, hw * self._t_delta, n) + c_t
          y = self._waveforms[i, 1, c_i - hw:c_i + hw + 1]
          pyplot.plot(x, y, color="b")

      pyplot.axvline(x=-(self._t_sym / 2), color="g")
      pyplot.axvline(x=0, color="r")
      pyplot.axvline(x=(self._t_sym / 2), color="g")
      pyplot.axhline(y=self._y_zero, color="g")
      pyplot.axhline(y=(self._y_ua + self._y_zero), color="g")

      ax = pyplot.gca()
      ax.xaxis.set_major_formatter(formatter_t)
      ax.yaxis.set_major_formatter(formatter_y)

      pyplot.xlabel("Time")
      pyplot.ylabel("Vertical")
      pyplot.title("Select Bit Sequence")

      pyplot.tight_layout()
      pyplot.savefig(debug_plots, bbox_inches="tight")
      pyplot.close()
      print(f"{'':>{indent}}Saved image to {debug_plots}")

  @abstractmethod
  def _step4_measure(self,
                     print_progress: bool = True,
                     indent: int = 0,
                     debug_plots: str = None) -> None:
    """Calculation step 4: compute measures

    Dependent on line encoding

    Args:
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.
    """
    # Derrived classes set these parameters
    self._measures = Measures()  # pragma: no cover
    self._offenders = [[]]  # pragma: no cover List[List[indices]]
    self._hits = [[]]  # pragma: no cover List[[t_UI, y_UA]]

  def _generate_bathtub_curves(self, y_slices: dict) -> dict:
    """Generate bathtub BER curves

    Args:
      y_slices: levels to slice at, UA

    Returns:
      dict of curves, same keys as y_slices
      {
        slice_level_0: np.array([[t0, t1,..., tn], [ber0, ber1,..., bern]]),
        slice_level_1:...}
    """
    levels = list(y_slices.values())
    output = []
    for i in range(self._waveforms.shape[0]):
      o = _eyediagram.y_slice(self._waveforms[i][1], self._centers_t[i],
                              self._centers_i[i], self._t_delta, self._t_sym,
                              self._y_zero, self._y_ua, levels)
      output.append(o)

    curves = {}
    for i, key in enumerate(y_slices):
      hits = []
      for o in output:
        hits.extend(o[i])
      hits = np.sort(hits)
      hits_left = hits[hits <= 0.5]
      hits_right = hits[hits > 0.5]

      n = len(hits_left)
      t_left = hits_left
      ber_left = (1 - np.arange(n) / n)

      n = len(hits_right)
      t_right = hits_right
      ber_right = (1 - np.arange(n)[::-1] / n)

      t = np.concatenate([[0.0], t_left, t_right, [1.0]])
      ber = np.concatenate([[1], ber_left, ber_right, [1]])

      # TODO (WattsUp) Add extrapolation, tails should be gaussian (RJ)

      curves[key] = np.array([t, ber])
    return curves

  def _sample_mask(self) -> dict:
    """Measure mask parameters

    Returns:
      Dictionary of values:
        offender_count: Number of symbols that hit the mask
        margin: Largest mask margin with zero hits [-1.0, 1.0]
    """
    values = {"offender_count": 0, "margin": 1.0}
    if self._mask is None:
      return values

    self._mask_converted = self._mask.convert_mixed_units(
        self._y_ua, self._t_sym)

    mask_paths = []
    mask_margins = []
    for i in range(1000, -1001, -1):
      mask_paths.append(self._mask_converted.adjust(i / 1000).paths)
      mask_margins.append(i / 1000)

    self._offenders = []
    self._hits = []
    margin = 1.0
    offender_count = 0
    for i in range(self._waveforms.shape[0]):
      o = _eyediagram.sample_mask(self._waveforms[i][1], self._centers_t[i],
                                  self._centers_i[i], self._t_delta,
                                  self._t_sym, self._y_zero, self._y_ua,
                                  mask_paths, mask_margins)
      self._offenders.append(o["offenders"])
      offender_count += len(o["offenders"])
      self._hits.extend(o["hits"])
      margin = min(margin, o["margin"])

    return {"margin": margin, "offender_count": offender_count}

  def _draw_grid(self, image_grid: np.ndarray) -> None:
    """Draw reference grid

    Override if wanting more levels such as thresholds

    Args:
      image_grid: Image to draw onto, [x, y] coordinates
    """
    image_grid[:, :, 0:2] = 1.0  # Yellow
    image_grid[self._uia_to_image(0.0), :, 3] = 1.0
    image_grid[self._uia_to_image(0.5), ::3, 3] = 1.0
    image_grid[self._uia_to_image(1.0), :, 3] = 1.0
    image_grid[:, self._uia_to_image(0.0), 3] = 1.0
    image_grid[::3, self._uia_to_image(0.5), 3] = 1.0
    image_grid[:, self._uia_to_image(1.0), 3] = 1.0
    for u in [-0.25, 0.25, 0.75, 1.25]:
      image_grid[::10, self._uia_to_image(u), 3] = 1.0
      image_grid[self._uia_to_image(u), ::10, 3] = 1.0

  def _uia_to_image(self,
                    u: float,
                    return_list: bool = True) -> Union[List[int], int]:
    """Convert UI/UA coordinates to image coordinates

    Args:
      u: UI/UA value
      return_list: True will return position inside a list, False will return
        scaler

    Returns:
      If return_list is true, list of position, list to clip outside image
      If return_list is false, position without clipping
    """
    pos = int((u + 0.5) / 2 * self._resolution)
    if not return_list:
      return pos
    if 0 <= pos <= self._resolution:
      return [pos]
    return []

  def _step5_stack(self,
                   print_progress: bool = True,
                   indent: int = 0,
                   debug_plots: str = None) -> None:
    """Calculation step 5: stack waveforms

    Args:
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.
    """

    min_y_ua = -0.5
    max_y_ua = 1.5
    # Convert UA to real vertical units
    min_y = min_y_ua * self._y_ua + self._y_zero
    max_y = max_y_ua * self._y_ua + self._y_zero

    if print_progress:
      print(f"{'':>{indent}}Starting stacking")

    self._raw_heatmap = np.zeros((self._resolution, self._resolution),
                                 dtype=np.int32)
    for i in range(self._waveforms.shape[0]):
      _eyediagram.stack(self._waveforms[i][1], self._centers_t[i],
                        self._centers_i[i], self._t_delta, self._t_sym, min_y,
                        max_y, self._resolution, self._raw_heatmap,
                        self._config.point_cloud)

    # Normalize density such that each column has the same number of counts
    n = self._measures.n_sym
    self._raw_heatmap = self._raw_heatmap.astype(np.float32)
    for i in range(self._resolution):
      self._raw_heatmap[i] *= (n / max(1, np.sum(self._raw_heatmap[i])))

    # [x, y] coordinates to image coordinates
    self._raw_heatmap = np.rot90(self._raw_heatmap)

    if print_progress:
      print(f"{'':>{indent}}Generating images")
    image_clean = self._raw_heatmap.copy()

    # Replace 0s with nan to be colored transparent
    image_clean[image_clean == 0] = np.nan

    # Normalize heatmap to 0 to 1
    image_max = np.nanmax(image_clean)
    image_clean = image_clean / image_max

    image_clean = pyplot.cm.jet(image_clean)

    image_grid = np.zeros(image_clean.shape)
    image_mask = np.zeros(image_clean.shape)
    image_hits = np.zeros(image_clean.shape)
    image_margin = np.zeros(image_clean.shape)

    # Draw a grid for reference levels
    self._draw_grid(image_grid)

    if self._mask_converted:
      for path in self._mask_converted.paths:
        x = [self._uia_to_image(p[0], return_list=False) for p in path]
        y = [self._uia_to_image(p[1], return_list=False) for p in path]
        rr, cc = skimage.draw.polygon(x, y, shape=image_mask.shape)
        image_mask[rr, cc] = [1.0, 0.0, 1.0, 0.5]  # Magenta a=0.5

      for path in self._mask_converted.adjust(self._measures.mask_margin).paths:
        x = [self._uia_to_image(p[0], return_list=False) for p in path]
        y = [self._uia_to_image(p[1], return_list=False) for p in path]
        rr, cc = skimage.draw.polygon(x, y, shape=image_margin.shape)
        image_margin[rr, cc] = [1.0, 0.0, 1.0, 0.5]  # Magenta a=0.5

      # Draw hits and offending bit waveforms
      image_hits[:, :, 0] = 1.0  # Red
      radius = int(max(3, self._resolution / 500))
      hit_drawn = np.zeros((self._resolution, self._resolution), dtype=np.bool_)
      for h in self._hits:
        x = self._uia_to_image(h[0], return_list=False)
        y = self._uia_to_image(h[1], return_list=False)
        if not hit_drawn[x, y]:
          # Don't repeatably draw hits
          # Opacity < 1 or antialiasing would make a difference
          hit_drawn[x, y] = True
          rr, cc = skimage.draw.circle_perimeter(x,
                                                 y,
                                                 radius,
                                                 shape=image_margin.shape)
          image_hits[rr, cc, 3] = 1

      # Plot subset of offenders
      for i in range(self._waveforms.shape[0]):
        o = _runner_draw_symbols(self._waveforms[i][1], self._centers_t[i],
                                 self._centers_i[i], self._t_delta, self._t_sym,
                                 min_y, max_y, self._resolution,
                                 self._offenders[i][:2])
        image_hits = image.layer_rgba(image_hits, o)

    # [x, y] coordinates to image coordinates
    image_grid = np.rot90(image_grid)
    image_mask = np.rot90(image_mask)
    image_hits = np.rot90(image_hits)
    image_margin = np.rot90(image_margin)

    self._measures.set_images(image_clean, image_grid, image_mask, image_hits,
                              image_margin)

    if print_progress:
      print(f"{'':>{indent}}Completed stacking")

    if debug_plots is not None:
      debug_plots += ".step5.png"
      image.np_to_file(image_clean, debug_plots)
      print(f"{'':>{indent}}Saved image to {debug_plots}")

  def get_raw_heatmap(self,
                      as_string: bool = False) -> Union[np.ndarray, bytes]:
    """Get generated raw heatmap (2D histogram of waveforms), optionally as a
    base64 encoded string

    Decode base64 as follows
    with io.BytesIO(base64.b64decode(data)) as file
      with np.load(file) as zip:
        raw_heatmap = zip[zip.files[0]]

    Args:
      as_string: True will return base64 encoded string for the numpy array

    Returns:
      numpy array or base64 encoded numpy array

    Raises:
      RuntimeError if EyeDiagram is not calculated first
    """
    if not self._calculated:
      raise RuntimeError("EyeDiagram must be calculated first")
    if as_string:
      with io.BytesIO() as file:
        np.savez_compressed(file, self._raw_heatmap)
        buf = file.getvalue()
      return base64.b64encode(buf)
    return self._raw_heatmap

  def get_measures(self) -> Measures:
    """Get result metrics of EyeDiagram calculation

    Returns:
      Measures object containing resultant metrics, specific to line encoding

    Raises:
      RuntimeError if EyeDiagram is not calculated first
    """
    if not self._calculated:
      raise RuntimeError("EyeDiagram must be calculated first")
    return self._measures

  def get_clock_edges(self) -> List[List[float]]:
    """Get clock edges found/used during calculation

    Primarily used to pass into EyeDiagram construction such as with/without
    filter or for differential signal decomposition. Also useful for TIE
    analysis including jitter decomposition

    waveform_filtered = apply_filter(waveform_raw)
    eye_filtered = EyeDiagram(waveform_filtered)
    eye_filtered.calculate()

    clock_edges = eye_filtered.get_clock_edges() # Clock off filtered signal
    config = Config("same vertical levels as eye_filtered")
    eye_unfiltered = EyeDiagram(waveform_raw,
                                clock_edges=clock_edges,
                                config=config)
    eye_unfiltered.calculate() # Compare with/without filter

    Returns:
      Array of clock edges per waveform

    Raises:
      RuntimeError if EyeDiagram is not calculated first
    """
    if not self._calculated:
      raise RuntimeError("EyeDiagram must be calculated first")
    return self._clock_edges


def _runner_draw_symbols(waveform_y: np.ndarray, centers_t: List[float],
                         centers_i: List[int], t_delta: float, t_sym: float,
                         min_y: float, max_y: float, resolution: int,
                         sym_indices: List[int]) -> np.ndarray:
  """Stack waveforms and counting overlaps in a heat map

  Args:
    waveform_y: Waveform data array [y0, y1,..., yn]
    centers_t: List of symbol centers in time for sub t_delta alignment.
      Grid spans [-0.5*t_sym, 1.5*t_sym] + center_t
    centers_i: List of symbol centers indices
    t_delta: Time between samples
    t_sym: Duration of one symbol
    min_y: Lower vertical value for bottom of grid
    max_y: Upper vertical value for top of grid.
      Grid units = (y - min_y) / (max_y - min_y)
    resolution: Resolution of square eye diagram image, 2UI x 2UA
    sym_indices: Symbol indices to draw

  Returns:
    RGBA image, [x, y] coordinates
  """
  i_width = int((t_sym / t_delta) + 0.5) + 2
  t_width_ui = (i_width * t_delta / t_sym)

  t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, i_width * 2 + 1)

  waveform_y = (waveform_y - min_y) / (max_y - min_y)

  img = np.zeros((resolution, resolution, 4))
  img[:, :, 0] = 1.0  # Red

  for i in sym_indices:
    c_i = centers_i[i]
    c_t = centers_t[i] / t_sym

    t = (t0 + c_t)
    y = waveform_y[c_i - i_width:c_i + i_width + 1]

    td = (((t + 0.5) / 2) * resolution).astype(np.int32)
    yd = (y * resolution).astype(np.int32)

    for ii in range(1, len(td)):
      if td[ii] < 0 or td[ii - 1] > (resolution):
        continue
      rr, cc, val = skimage.draw.line_aa(td[ii], yd[ii], td[ii - 1], yd[ii - 1])

      # Trim to image size
      mask = (rr >= 0) & (rr < resolution) & (cc >= 0) & (cc < resolution)
      rr = rr[mask]
      cc = cc[mask]
      val = val[mask]

      img[rr, cc, 3] = val + (1 - val) * img[rr, cc, 3]

  return img
