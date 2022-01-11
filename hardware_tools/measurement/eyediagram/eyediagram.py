"""Eye Diagram to layer repeated waveforms and measure the resultant heat map
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import datetime
from enum import Enum
import io
import multiprocessing
from typing import Callable, Union

import colorama
from colorama import Fore
import numpy as np
import matplotlib.pyplot as pyplot

from hardware_tools import math, strformat
from hardware_tools.extensions import bresenham
from hardware_tools.measurement.mask import Mask


class ClockPolarity(Enum):
  RISING = 1  # Sample on clock's rising edge
  FALLING = 2  # Sample on clock's falling edge
  BOTH = 3  # Sample on both edges


class Measures(ABC):
  """Eye Diagram Measures collecting metrics from an eye diagram

  Must be derrived to support proper signal encoding

  All images are base64 encoded PNGs, use save_images to save to disk

  Properties:
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
  """

  def __init__(self, np_clean: np.ndarray, np_grid: np.ndarray,
               np_mask: np.ndarray, np_hits: np.ndarray,
               np_margin: np.ndarray) -> None:
    """Create a new Measures object

    Images are numpy format, RGBA, shape=(rows, columns, 4), [0.0, 1.0]

    Args:
      np_clean: numpy image for image_clean
      np_grid: numpy image for image_grid
      np_mask: numpy image for image_mask
      np_hits: numpy image for image_hits
      np_margin: numpy image for image_margin
    """
    super().__init__()

    self.n_sym = None
    self.n_sym_bad = None
    self.transition_dist = None
    self.mask_margin = None

    self._np_image_clean = np_clean
    self._np_image_grid = np_grid
    self._np_image_mask = np_mask
    self._np_image_hits = np_hits
    self._np_image_margin = np_margin

  def save_images(self, basename: str = "eyediagram") -> None:
    """Save images to PNG file

    Args:
      basename: base file name for images.
        image_clean will be saved to "{basename}.clean.png"
        image_grid will be saved to "{basename}.grid.png"
        image_mask will be saved to "{basename}.mask.png"
        image_hits will be saved to "{basename}.hits.png"
        image_margin will be saved to "{basename}.margin.png"
    """
    math.Image.np_to_file(self._np_image_clean, f"{basename}.clean.png")
    math.Image.np_to_file(self._np_image_grid, f"{basename}.grid.png")
    math.Image.np_to_file(self._np_image_mask, f"{basename}.mask.png")
    math.Image.np_to_file(self._np_image_hits, f"{basename}.hits.png")
    math.Image.np_to_file(self._np_image_margin, f"{basename}.margin.png")

  @property
  def image_clean(self) -> str:
    return math.Image.np_to_base64(self._np_image_clean)

  @property
  def image_grid(self) -> str:
    return math.Image.np_to_base64(self._np_image_grid)

  @property
  def image_mask(self) -> str:
    return math.Image.np_to_base64(self._np_image_mask)

  @property
  def image_hits(self) -> str:
    return math.Image.np_to_base64(self._np_image_hits)

  @property
  def image_margin(self) -> str:
    return math.Image.np_to_base64(self._np_image_margin)

  def to_dict(self) -> dict:
    """Convert Measures to dictionary of values

    Compatible with default JSONEncoder

    Returns:
      dictionary of metrics and images (base64 encoded PNGs)
    """
    properties = dir(self)
    d = {}
    for key in properties:
      if key.startswith("_"):
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


class EyeDiagram(ABC):
  """Eye Diagram to layer repeated waveforms and measure the resultant heat map

  Must be derrived to support proper signal encoding
  """

  def __init__(self,
               waveforms: np.ndarray,
               clocks: np.ndarray = None,
               t_unit: str = "",
               y_unit: str = "",
               mask: Mask = None,
               resolution: int = 2000,
               unipolar: bool = True,
               resample: int = 50) -> None:
    """Create a new EyeDiagram. Lazy creation, does not compute anything

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

    self._waveforms = waveforms
    self._clocks = clocks

    self._t_unit = t_unit
    self._y_unit = y_unit
    self._mask = mask
    self._unipolar = unipolar
    self._resample = resample

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
    self._clock_edges = None

    self._calculated = False

  def calculate(self,
                n_threads: int = 0,
                print_progress: bool = True,
                indent: int = 0,
                debug_plots: str = None) -> Measures:
    """Perform eye diagram calculation

    Args:
      n_threads: number of thread to execute across, 0=all, 1=single, or n
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.

    Returns:
      Measures object containing resultant metrics, specific to line encoding
    """
    colorama.init(autoreset=True)
    start = datetime.datetime.now()
    if n_threads < 1:
      n_threads = multiprocessing.cpu_count()
    else:
      n_threads = min(n_threads, multiprocessing.cpu_count(),
                      self._waveforms.shape[0])

    if print_progress:
      print(indent)
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.CYAN}"
            f"Starting eye diagram calculation with {n_threads} threads")
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.YELLOW}"
            "Step 1: Finding threshold levels")
    self._step1_levels(n_threads=n_threads,
                       print_progress=print_progress,
                       indent=indent + 2,
                       debug_plots=debug_plots)

    if print_progress:
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.YELLOW}"
            "Step 1: Determining receiver clock")
    self._step2_clock(n_threads=n_threads,
                      print_progress=print_progress,
                      indent=indent + 2,
                      debug_plots=debug_plots)

    if print_progress:
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.YELLOW}"
            "Step 3: Aligning symbol sampling points")
    self._step3_sample(n_threads=n_threads,
                       print_progress=print_progress,
                       indent=indent + 2,
                       debug_plots=debug_plots)

    if print_progress:
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.YELLOW}"
            "Step 4: Stacking waveforms")
    self._step4_stack(n_threads=n_threads,
                      print_progress=print_progress,
                      indent=indent + 2,
                      debug_plots=debug_plots)

    if print_progress:
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.YELLOW}"
            "Step 5: Measuring waveform")
    self._step5_measure(n_threads=n_threads,
                        print_progress=print_progress,
                        indent=indent + 2,
                        debug_plots=debug_plots)

    self._calculated = True

    if print_progress:
      print(f"{'':>{indent}}{strformat.elapsed_str(start)} {Fore.CYAN}"
            "Completed eye diagram calculation")
    return self._measures

  @abstractmethod
  def _step1_levels(self,
                    n_threads: int = 1,
                    print_progress: bool = True,
                    indent: int = 0,
                    debug_plots: str = None) -> None:
    """Calculation step 1: levels

    Dependent on line encoding

    Args:
      n_threads: number of thread to execute across
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.
    """
    # self._y_zero = float
    # self._y_ua = float
    pass  # pragma: no cover

  @abstractmethod
  def _step2_clock(self,
                   n_threads: int = 1,
                   print_progress: bool = True,
                   indent: int = 0,
                   debug_plots: str = None) -> None:
    """Calculation step 2: clock recovery

    Dependent on line encoding and clocks (passed during initialization)

    Args:
      n_threads: number of thread to execute across
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.
    """
    # self._clock_edges = list[list[float]]
    # self._t_sym = math.UncertainValue
    pass  # pragma: no cover

  def _step3_sample(self,
                    n_threads: int = 1,
                    print_progress: bool = True,
                    indent: int = 0,
                    debug_plots: str = None) -> None:
    """Calculation step 3: symbol sample point

    Args:
      n_threads: number of thread to execute across
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.
    """
    _ = n_threads
    self._centers_i = []
    self._centers_t = []

    if print_progress:
      print(f"{'':>{indent}}Starting sampling")
    for i in range(self._waveforms.shape[0]):
      centers_i = []
      centers_t = []
      t_zero = self._waveforms[i, 0, 0]
      for b in self._clock_edges[i]:
        center_t = b % self._t_delta
        centers_t.append(-center_t)

        center_i = int(((b - t_zero - center_t) / self._t_delta) + 0.5)
        centers_i.append(center_i)

      self._centers_i.append(centers_i)
      self._centers_t.append(centers_t)
      if print_progress:
        print(f"{'':>{indent}}Ran waveform #{i}")

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

      hw = int((self._t_sym.value / self._t_delta) + 0.5)
      n = hw * 2 + 1

      for i in range(self._waveforms.shape[0]):
        n_plot = min(len(self._centers_i[i]),
                     max(1, int(20 / self._waveforms.shape[0])))
        for edge in range(n_plot):
          c_i = self._centers_i[i][edge]
          c_t = self._centers_t[i][edge]
          if (c_i - hw) < 0 or (c_i + hw + 1) >= self._waveforms.shape[2]:
            continue
          x = np.linspace(-hw * self._t_delta, hw * self._t_delta, n) + c_t
          y = self._waveforms[i, 1, c_i - hw:c_i + hw + 1]
          pyplot.plot(x, y, color="b")

      pyplot.axvline(x=(-self._t_sym.value / 2), color="g")
      pyplot.axvline(x=0, color="r")
      pyplot.axvline(x=(self._t_sym.value / 2), color="g")
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

  def _step4_stack(self,
                   n_threads: int = 1,
                   print_progress: bool = True,
                   indent: int = 0,
                   debug_plots: str = None) -> None:
    """Calculation step 4: stack waveforms

    Args:
      n_threads: number of thread to execute across
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.
    """

    min_y = -0.5 if self._unipolar else -1.0
    max_y = 1.5 if self._unipolar else 1.0
    # Convert UA to real vertical units
    min_y = min_y * self._y_ua + self._y_zero
    max_y = max_y * self._y_ua + self._y_zero

    if print_progress:
      print(f"{'':>{indent}}Starting stacking")

    # yapf: disable
    args_list = [[
        self._waveforms[i][1],
        self._centers_t[i],
        self._centers_i[i],
        self._t_delta,
        self._t_sym.value,
        min_y,
        max_y,
        self._resample,
        self._resolution
    ] for i in range(self._waveforms.shape[0])]
    # yapf: enable
    output = self._collect_runners(_runner_stack, args_list, n_threads,
                                   print_progress, indent)

    self._raw_heatmap = np.zeros((self._resolution, self._resolution),
                                 dtype=np.int32)
    for o in output:
      self._raw_heatmap += o

    self._raw_heatmap = self._raw_heatmap.T[::-1, :]
    self._raw_heatmap = self._raw_heatmap.astype(np.float32)

    if print_progress:
      print(f"{'':>{indent}}Completed stacking")

    if debug_plots is not None:
      debug_plots += ".step4.png"
      image = self._raw_heatmap.copy()

      # Replace 0s with nan to be colored transparent
      image[image == 0] = np.nan

      # Normalize heatmap to 0 to 1
      image_max = np.nanmax(image)
      image = image / image_max

      image = pyplot.cm.jet(image)
      math.Image.np_to_file(image, debug_plots)
      print(f"{'':>{indent}}Saved image to {debug_plots}")

  @abstractmethod
  def _step5_measure(self,
                     n_threads: int = 1,
                     print_progress: bool = True,
                     indent: int = 0,
                     debug_plots: str = None) -> None:
    """Calculation step 5: compute measures

    Dependent on line encoding

    Args:
      n_threads: number of thread to execute across
      print_progress: True will print statements along the way, False will not.
      indent: Indent all print statements that much
      debug_plots: base filename to save debug plots to. None will not save any
        plots.
    """
    # self._measures = Measures
    pass  # pragma: no cover

  @staticmethod
  def _collect_runners(method: Callable,
                       args_list: list,
                       n_threads: int = 1,
                       print_progress: bool = True,
                       indent: int = 0) -> list:
    if n_threads <= 1:
      output = []
      for i in range(len(args_list)):
        output.append(method(*args_list[i]))
        if print_progress:
          print(f"{'':>{indent}}Ran waveform #{i}")
    else:
      with multiprocessing.Pool(n_threads) as p:
        results = []
        for i in range(len(args_list)):
          r = p.apply_async(method, args=args_list[i])
          results.append((i, r))
        output = []
        for i, r in results:
          output.append(r.get())
          if print_progress:
            print(f"{'':>{indent}}Ran waveform #{i}")
    return output

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


def _runner_stack(waveform_y: np.ndarray,
                  centers_t: list[float],
                  centers_i: list[int],
                  t_delta: float,
                  t_sym: float,
                  min_y: float,
                  max_y: float,
                  resample: int = 50,
                  resolution: int = 500) -> np.ndarray:
  """Stack waveforms and counting overlaps in a heat map

  Args:
    waveform_y: Waveform data array [y0, y1,..., yn]
    centers_t: List of symbol centers in time for sub t_delta alignment.
      Grid spans [-0.5*t_sym, 1.5*t_sym] + center_t
    centers_i: List of symbol centers indices
    t_delta: Time between samples
    min_y: Lower vertical value for bottom of grid
    max_y: Upper vertical value for top of grid.
      Grid units = (y - min_y) / (max_y - min_y)
    resample: n=0 will not resample, n>0 will use sinc interpolation to
      resample a single symbol to at least n segments
    resolution: Resolution of square eye diagram image, 2UI x 2UA

  Returns:
    2D grid of heat map counts
  """
  i_width = int((t_sym / t_delta) + 0.5)
  t_width_ui = (i_width * t_delta / t_sym)

  t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, i_width * 2 + 1)

  waveform_y = (waveform_y - min_y) / (max_y - min_y)

  factor = int(np.ceil(resample / i_width))
  if factor > 1:
    # Expand to 3 bits for better sinc interpolation at the edges
    i_width = int((t_sym / t_delta * 1.5) + 0.5)
    t_width_ui = (i_width * t_delta / t_sym)
    t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, i_width * 2 + 1)
    t_new = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui,
                        i_width * 2 * factor + 1)

    period = t0[1] - t0[0]
    sinc = np.tile(t_new, (len(t0), 1)) - \
        np.tile(t0[:, np.newaxis], (1, len(t_new)))
    reference_sinc = np.sinc(sinc / period)
  else:
    # Add an extra sample on either end
    i_width += 2
    t_width_ui = (i_width * t_delta / t_sym)
    t0 = np.linspace(0.5 - t_width_ui, 0.5 + t_width_ui, i_width * 2 + 1)

  grid = np.zeros((resolution, resolution), dtype=np.int32)

  max_i = len(waveform_y)

  for i in range(len(centers_t)):
    c_i = centers_i[i]
    c_t = centers_t[i] / t_sym

    if (c_i - i_width) < 0 or (c_i + i_width + 1) > max_i:
      continue

    if factor > 1:
      t = (t_new + c_t)
      y_original = waveform_y[c_i - i_width:c_i + i_width + 1]
      y = np.dot(y_original, reference_sinc)
    else:
      t = (t0 + c_t)
      y = waveform_y[c_i - i_width:c_i + i_width + 1]

    td = (((t + 0.5) / 2) * resolution).astype(np.int32)
    yd = (y * resolution).astype(np.int32)
    bresenham.draw(td, yd, grid)

  return grid
