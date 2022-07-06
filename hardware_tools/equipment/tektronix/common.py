"""Common functions among Tektronix scopes
"""

import datetime
import re
import struct
from typing import Any, Tuple

import numpy as np

from hardware_tools.equipment import utility
from hardware_tools.equipment.scope import SampleMode
from hardware_tools.math.lines import EdgePolarity
from hardware_tools.math.stats import Comparison

_rng = np.random.default_rng()


def parse_comparison(v: str) -> Comparison:
  """Convert a str to a Comparison

  Args:
    v: Value to convert

  Returns:
    Comparison or v if no match
  """
  d = {
      "LESSthan": Comparison.LESS,
      "MOREthan": Comparison.MORE,
      "EQual": Comparison.EQUAL,
      "UNEqual": Comparison.UNEQUAL,
      "LESSEQual": Comparison.LESSEQUAL,
      "MOREEQual": Comparison.MOREEQUAL,
      "WIThin": Comparison.WITHIN,
      "INrange": Comparison.WITHIN,
      "OUTside": Comparison.OUTSIDE,
      "OUTrange": Comparison.OUTSIDE
  }
  for kd, vd in d.items():
    kd: str
    kk_short = re.sub(r"[a-z]+$", "", kd)
    if v.startswith(kk_short):
      return vd
  return v


def parse_sample_mode(v: str) -> SampleMode:
  """Convert a str to a Comparison

  Args:
    v: Value to convert

  Returns:
    Comparison or v if no match
  """
  d = {
      "SAMple": SampleMode.SAMPLE,
      "AVErage": SampleMode.AVERAGE,
      "ENVelope": SampleMode.ENVELOPE
  }
  for kd, vd in d.items():
    kd: str
    kk_short = re.sub(r"[a-z]+$", "", kd)
    if v.startswith(kk_short):
      return vd
  return v


def parse_polarity(v: str) -> EdgePolarity:
  """Convert a str to a EdgePolarity

  Args:
    v: Value to convert

  Returns:
    EdgePolarity or v if no match
  """
  d = {
      "RISe": EdgePolarity.RISING,
      "POSitive": EdgePolarity.RISING,
      "STAYSHigh": EdgePolarity.RISING,
      "FALL": EdgePolarity.FALLING,
      "NEGative": EdgePolarity.FALLING,
      "STAYSLow": EdgePolarity.FALLING,
      "EITher": EdgePolarity.BOTH
  }
  for kd, vd in d.items():
    kd: str
    kk_short = re.sub(r"[a-z]+$", "", kd)
    if v.startswith(kk_short):
      return vd
  return v


def parse_threshold(v: str) -> float:
  """Convert a str to a threshold

  Args:
    v: Value to convert

  Returns:
    -1.3 if ECL
     1.4 if TTL
     else float(v)
  """
  d = {"ECL": -1.3, "TTL": 1.4}
  if v in d:
    return d[v]
  return float(v)


TEK_TYPES = {
    "TRIGger": {
        "A": {
            "BANDWidth": {
                "RF": {
                    "HIGH": float,
                    "LOW": float
                }
            },
            "EDGE": {
                "COUPling": str.upper,
                "SLOpe": parse_polarity,
                "SOUrce": str.upper
            },
            "HOLDoff": {
                "TIMe": float
            },
            "LEVel": {
                "AUXin": parse_threshold,
                "CH<1-4>": parse_threshold,
                "D<0-15>": parse_threshold
            },
            "MODe": str.upper,
            "PULse": {
                "CLAss": str.upper
            },
            "PULSEWidth": {
                "HIGHLimit": float,
                "LOWLimit": float,
                "POLarity": parse_polarity,
                "SOUrce": str.upper,
                "WHEn": parse_comparison,
                "WIDth": float
            },
            "TIMEOut": {
                "POLarity": parse_polarity,
                "SOUrce": str.upper,
                "TIMe": float
            },
            "TYPe": str.upper
        }
    },
    "WFMOutpre|WFMInpre|WFMPre": {
        "BYT_Nr": int,
        "BIT_Nr": int,
        "ENCdg": str.upper,
        "BN_Fmt": str.upper,
        "BYT_Or": str.upper,
        "WFId": lambda s: str.strip(s, '"'),
        "NR_Pt": int,
        "PT_Fmt": str.upper,
        "XUNit": lambda s: str.strip(s, '"'),
        "XINcr": float,
        "XZEro": float,
        "PT_Off": int,
        "PT_ORder": str.upper,
        "YUNit": lambda s: str.strip(s, '"'),
        "YMUlt": float,
        "YOFf": float,
        "YZEro": float,
        "VSCALE": float,
        "HSCALE": float,
        "VPOS": float,
        "VOFFSET": float,
        "HDELAY": float,
        "DOMain": str.upper,
        "WFMTYPe": str.upper,
        "CENTERFREQuency": float,
        "SPAN": float,
        "REFLEvel": float
    }
}


def parse_waveform_query(data: bytes,
                         raw: bool = False,
                         add_noise: bool = False) -> Tuple[np.ndarray, dict]:
  """Parse waveform query format, output of WAVFRM?, also ISF files

  Args:
    data: Raw data from WAVFRM? query or equivalent source
    raw: True will return raw ADC values, False will transform into
      real-world units
    add_noise: True will add uniform noise to the LSB for anti-aliasing

  Returns:
    Samples: [[x0, x1,..., xn], [y0, y1,..., yn]]
    Dictionary of sampling information:
      config_string: str = Human readable string describing configuration
      x_unit: str = Unit string for horizontal axis
      y_unit: str = Unit string for vertical axis
      x_incr: float = Step between horizontal axis values
      y_incr: float = Step between vertical axis values (LSB)
      y_clip_min: float = Minimum input without clipping
      y_clip_max: float = Maximum input without clipping
      clipping_top: bool = waveform exceeded ADC limits
      clipping_bottom: bool = waveform exceeded ADC limits

  Raises:
    ValueError if a parsing error was encountered
  """
  data_list = data.split(b";:CURVE ")
  if len(data_list) != 2:
    data_list = data.split(b";:CURV ")
  if len(data_list) != 2:
    raise ValueError("Missing ';:CURVE '")

  waveform_format = data_list[0].decode(encoding="ascii")
  curve = data_list[1]

  header = utility.parse_scpi(waveform_format, flat=False, types=TEK_TYPES)
  header = header.popitem()[1]  # Remove outer "WFMOutpre|WFMInpre|WFMPre"

  points = header["NR_PT"]
  x_incr = header["XINCR"]
  x_zero = header["XZERO"]
  x_unit = header["XUNIT"]

  info_dict = {
      "config_string": header["WFID"],
      "x_unit": x_unit,
      "y_unit": "ADC Counts",
      "x_incr": x_incr,
      "y_incr": 1
  }

  header_len = 2 + int(chr(curve[1]), 16)

  byte_order = ">" if header["BYT_OR"] == "MSB" else "<"
  byte_format = "i" if header["BN_FMT"] == "RI" else "u"
  byte_size = header["BYT_NR"]
  dtype = f"{byte_order}{byte_format}{byte_size}"

  wave_bin = curve[header_len:header_len + points * byte_size]
  wave_int = np.frombuffer(wave_bin, dtype=dtype)
  wave = wave_int.astype(np.float64)
  x = x_zero + x_incr * np.arange(points).astype(np.float64)
  # float32 is not accurate enough for 1e6 points (only ~7digits of precision)

  if byte_size == 1:
    min_val = -127
    max_val = 127
  elif byte_size == 2:
    min_val = -32767
    max_val = 32767
  else:
    raise ValueError("Unknown number of bytes")

  info_dict["y_clip_min"] = min_val
  info_dict["y_clip_max"] = max_val
  info_dict["clipping_top"] = wave.max() >= max_val
  info_dict["clipping_bottom"] = wave.min() <= min_val

  if add_noise:
    # Need to determine actual bit depth since a 8b ADC can store into a 16b
    all_or = np.bitwise_or.reduce(wave_int, axis=-1)
    lsb = 0
    while lsb < 8 * byte_size and ((all_or >> lsb) & 0x1) == 0x0:
      lsb += 1

    wave += _rng.uniform(-0.5 * (2**lsb), 0.5 * (2**lsb), points)
    # Don"t add noise to clipping values
    wave[np.where(wave >= max_val - 0.5)] = max_val
    wave[np.where(wave < min_val + 0.5)] = min_val

  if raw:
    return (np.stack([x, wave]), info_dict)

  y_mult = header["YMULT"]
  y_zero = header["YZERO"]
  y_off = header["YOFF"]
  y_unit = header["YUNIT"]
  y = (wave - y_off) * y_mult + y_zero

  info_dict["y_clip_min"] = (min_val - y_off) * y_mult + y_zero
  info_dict["y_clip_max"] = (max_val - y_off) * y_mult + y_zero

  info_dict["y_unit"] = y_unit
  info_dict["y_incr"] = y_mult

  return (np.stack([x, y]), info_dict)


def parse_wfm_file(data: bytes,
                   raw: bool = False,
                   add_noise: bool = False,
                   include_prepost: bool = False) -> Tuple[np.ndarray, dict]:
  """Parse wfm file format

  Args:
    data: Raw data from .wfm file or equivalent source
    raw: True will return raw ADC values, False will transform into
      real-world units
    add_noise: True will add uniform noise to the LSB for anti-aliasing
    include_prepost: True will include the precharge and postcharge samples
      normally used for interpolation engines, False will not

  Returns:
    Samples: [[x0, x1,..., xn], [y0, y1,..., yn]]
      If data has fast frames,
        samples will be shaped (n_fast_frames, 2, n_samples)
        [frame0[[x0, x1,..., xn], [y0, y1,..., yn]], frame1[[...],[...]],...]
    Dictionary of sampling information:
      config_string: str = Human readable string describing configuration
      x_unit: str = Unit string for horizontal axis
      y_unit: str = Unit string for vertical axis
      x_incr: float = Step between horizontal axis values
      y_incr: float = Step between vertical axis values (LSB)
      y_clip_min: float = Minimum input without clipping
      y_clip_max: float = Maximum input without clipping
      clipping_top: bool = waveform exceeded ADC limits
      clipping_bottom: bool = waveform exceeded ADC limits
      timestamp: datetime = Timestamp of trigger
        If data has fast frames, this will be a list of datetimes
        The resolution of datetime is µs so some resolution may be loss

  Raises:
    ValueError if a parsing error was encountered
  """
  len_data = len(data)
  len_header = 0x346
  if len_data < len_header:
    raise ValueError("Data is shorter than minimum")

  # Section: Waveform static file information
  endianness = "<"
  # TODO (WattsUp) Find a wfm file with big endianness to test against
  # if data[:2] == b"\xF0\xF0":
  #   endianness = ">"
  if data[:2] != b"\x0F\x0F":
    raise ValueError(f"Data unrecognized start {data[:2]}")

  def unpack(fmt: str, offset: int) -> Any:
    return struct.unpack_from(endianness + fmt, data, offset=offset)[0]

  version = int(unpack("3s", 0x007))
  if version != 3:
    raise ValueError("Only version 3 is supported")

  bytes_per_point: int = unpack("b", 0x00F)
  str_raw: bytes = unpack("20s", 0x028)
  label: str = str_raw.split(b"\x00", maxsplit=1)[0].decode(encoding="utf-8")
  n_fast_frames: int = unpack("L", 0x048)
  n_frames = n_fast_frames + 1

  # Section: Waveform header
  imp_dim_count: int = unpack("L", 0x072)
  exp_dim_count: int = unpack("L", 0x076)
  data_type: int = unpack("L", 0x07A)

  if ((imp_dim_count != 1) or (exp_dim_count != 1) or (data_type != 2)):
    raise ValueError("Only compatible with normal YT waveforms")

  # Section: Explicit dimension 1 and 2
  # Only care about dimension 1
  exp_dim_scale: float = unpack("d", 0x0A8)
  exp_dim_offset: float = unpack("d", 0x0B0)
  str_raw: bytes = unpack("20s", 0x0BC)
  exp_dim_units = str_raw.split(b"\x00", maxsplit=1)[0].decode(encoding="utf-8")
  exp_dim_format: int = unpack("L", 0x0F0)
  exp_dim_storage_type: int = unpack("L", 0x0F4)

  if exp_dim_storage_type != 0:
    raise ValueError("Only compatible with EXPLICIT_SAMPLE")

  # Section: Implicit dimension 1 and 2
  # Only care about dimension 1
  imp_dim_scale: float = unpack("d", 0x1E8)
  imp_dim_offset: float = unpack("d", 0x1F0)
  str_raw: bytes = unpack("20s", 0x1FC)
  imp_dim_units = str_raw.split(b"\x00", maxsplit=1)[0].decode(encoding="utf-8")

  # Section: Time Base 1 and 2 information
  time_base = [{}, {}]
  for i in range(2):
    offset = 0x00C * i
    time_base[i]["spacing"] = unpack("L", 0x2F8 + offset)
    time_base[i]["sweep"] = unpack("L", 0x2FC + offset)
    time_base[i]["type"] = unpack("L", 0x300 + offset)

  # Section: Wfm Update specification
  tt_offset: float = unpack("d", 0x314)
  frac_sec: float = unpack("d", 0x31C)
  gmt_sec: int = unpack("l", 0x324)
  timestamp = datetime.datetime.fromtimestamp(gmt_sec + frac_sec,
                                              tz=datetime.timezone.utc)

  # Section: Wfm Curve information
  offset_precharge: int = unpack("L", 0x332)
  offset_data: int = unpack("L", 0x336)
  offset_postcharge: int = unpack("L", 0x33A)
  offset_postcharge_stop: int = unpack("L", 0x33E)
  offset_end: int = unpack("L", 0x342)

  codes = {
      0: ("int16", 16, -((1 << 15) - 1), (1 << 15) - 1),
      1: ("int32", 32, -((1 << 31) - 1), (1 << 31) - 1),
      2: ("uint32", 32, 0, (1 << 32) - 1),
      3: ("uint64", 64, 0, (1 << 64) - 1),
      # TODO (WattsUp) Floats not compatible due to figuring out LSB for
      # add_noise
      # 4: ("float32", 32),
      # 5: ("float64", 64),
      6: ("uint8", 8, 0, (1 << 8) - 1),
      7: ("int8", 8, -((1 << 7) - 1), (1 << 7) - 1)
  }
  if exp_dim_format not in codes:
    raise ValueError(f"Invalid explicit dimension format: {exp_dim_format}")
  if codes[exp_dim_format][1] != (bytes_per_point * 8):
    raise ValueError("Explicit dimension format != bytes_per_point: "
                     f"{exp_dim_format} vs {bytes_per_point}B")

  # Construct time domain data
  n_precharge = (offset_data - offset_precharge) // bytes_per_point
  n_samples = (offset_postcharge - offset_data) // bytes_per_point
  n_postcharge = (offset_postcharge_stop - offset_postcharge) // bytes_per_point
  n_samples_all = n_precharge + n_samples + n_postcharge
  t = np.arange(-n_postcharge, n_samples + n_postcharge) + tt_offset
  t = t * imp_dim_scale + imp_dim_offset

  # Header + fast frames headers + curve buffer for all frames + checksum
  len_expected = len_header + (54 * n_fast_frames) + (offset_end * n_frames) + 8
  if len_data < len_expected:
    raise ValueError(
        f"Data is shorter than expected {len_data} vs {len_expected}")
  checksum_expected: int = unpack("Q", len_expected - 8)
  checksum: np.int64 = np.frombuffer(data[:(len_expected - 8)],
                                     dtype="uint8").sum(dtype=np.int64)
  if checksum != checksum_expected:
    raise ValueError("Checksum error: "
                     f"{checksum:016X} vs {checksum_expected:016X}")

  dtype = np.dtype(codes[exp_dim_format][0]).newbyteorder(endianness)
  y: np.ndarray = np.frombuffer(data,
                                dtype=dtype,
                                offset=len_header + (54 * n_fast_frames),
                                count=n_samples_all * n_frames)

  min_val = codes[exp_dim_format][2]
  max_val = codes[exp_dim_format][3]

  info_dict = {
      "config_string": f"Label: {label}, "
                       f"{n_samples_all if include_prepost else n_samples} "
                       f"points, {n_frames} frame{'s' if n_frames > 1 else ''}",
      "x_unit": imp_dim_units,
      "y_unit": "ADC Counts",
      "x_incr": imp_dim_scale,
      "y_incr": 1,
      "y_clip_min": min_val,
      "y_clip_max": max_val,
      "clipping_top": (y.max() >= max_val) if y.size > 0 else False,
      "clipping_bottom": (y.min() <= min_val) if y.size > 0 else False
  }

  if add_noise:
    # Need to determine actual bit depth since a 8b ADC can store into a 16b
    all_or = np.bitwise_or.reduce(y, axis=-1)
    lsb = 0
    while lsb < 8 * bytes_per_point and ((all_or >> lsb) & 0x1) == 0x0:
      lsb += 1

    y = y + _rng.uniform(-0.5 * (2**lsb), 0.5 *
                         (2**lsb), n_samples_all * n_frames)
    # Don"t add noise to clipping values
    y[np.where(y >= max_val - 0.5)] = max_val
    y[np.where(y < min_val + 0.5)] = min_val
  if not raw:
    y = y * exp_dim_scale + exp_dim_offset
    info_dict["y_unit"] = exp_dim_units
    info_dict["y_incr"] = exp_dim_scale
    info_dict["y_clip_min"] = min_val * exp_dim_scale + exp_dim_offset
    info_dict["y_clip_max"] = max_val * exp_dim_scale + exp_dim_offset

  if n_frames != 1:
    timestamp = [timestamp]
    y = np.split(y, n_frames)
    wave = [[t, y[0]]]
    for i in range(n_fast_frames):
      offset = 0x346 + i * 24
      tt_offset: float = unpack("d", offset + 4)
      frac_sec: float = unpack("d", offset + 12)
      gmt_sec: int = unpack("l", offset + 20)
      timestamp.append(
          datetime.datetime.fromtimestamp(gmt_sec + frac_sec,
                                          tz=datetime.timezone.utc))

      t = np.arange(-n_postcharge, n_samples + n_postcharge) + tt_offset
      t = t * imp_dim_scale + imp_dim_offset

      wave.append([t, y[i + 1]])
    wave = np.array(wave)
    if not include_prepost:
      wave = wave[:, :, n_precharge:(n_precharge + n_samples)]
  else:
    wave = np.array([t, y])

    if not include_prepost:
      wave = wave[:, n_precharge:(n_precharge + n_samples)]

  info_dict["timestamp"] = timestamp
  return wave, info_dict
