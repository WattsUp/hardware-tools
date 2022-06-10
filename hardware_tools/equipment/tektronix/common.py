"""Common functions among Tektronix scopes
"""

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
    max_val = 127
  elif byte_size == 2:
    max_val = 32767
  else:
    raise ValueError("Unknown number of bytes")

  info_dict["y_clip_min"] = -max_val
  info_dict["y_clip_max"] = max_val
  info_dict["clipping_top"] = wave.max() >= max_val
  info_dict["clipping_bottom"] = wave.min() <= -max_val

  if add_noise:
    # Need to determine actual bit depth since a 8b ADC can store into a 16b
    all_or = np.bitwise_or.reduce(wave_int, axis=-1)
    lsb = 0
    while lsb < 8 * byte_size and ((all_or >> lsb) & 0x1) == 0x0:
      lsb += 1

    wave += _rng.uniform(-0.5 * (2**lsb), 0.5 * (2**lsb), points)
    # Don"t add noise to clipping values
    wave[np.where(wave >= max_val - 0.5)] = max_val
    wave[np.where(wave < -max_val + 0.5)] = -max_val

  if raw:
    return (np.stack([x, wave]), info_dict)

  y_mult = header["YMULT"]
  y_zero = header["YZERO"]
  y_off = header["YOFF"]
  y_unit = header["YUNIT"]
  y = (wave - y_off) * y_mult + y_zero

  info_dict["y_clip_min"] = (-max_val - y_off) * y_mult + y_zero
  info_dict["y_clip_max"] = (max_val - y_off) * y_mult + y_zero

  info_dict["y_unit"] = y_unit
  info_dict["y_incr"] = y_mult

  return (np.stack([x, y]), info_dict)


def parse_wfm_file(data: bytes,
                   raw: bool = False,
                   add_noise: bool = False) -> Tuple[np.ndarray, dict]:
  """Parse wfm file format

  Args:
    data: Raw data from .wfm file or equivalent source
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
  size_static_header = 78
  if len(data) < size_static_header:
    raise ValueError("File is shorter than minimum")
  # Section: Waveform static file information

  endianness = "<"
  if data[:2] == b"\xF0\xF0":
    endianness = ">"
  elif data[:2] != b"\x0F\x0F":
    raise ValueError(f"File unrecognized start {data[:2]}")

  version = int(struct.unpack_from(f"{endianness}3s", data, 0x007)[0])
  if version > 3:
    raise ValueError("Only versions 3 and below are supported")

  digits_per_byte: int = struct.unpack_from(f"{endianness}b", data, 0x00A)[0]
  bytes_until_eof: int = struct.unpack_from(f"{endianness}l", data, 0x00B)[0]
  bytes_per_point: int = struct.unpack_from(f"{endianness}b", data, 0x00F)[0]
  buffer_offset: int = struct.unpack_from(f"{endianness}l", data, 0x010)[0]
  label: str = (struct.unpack_from(f"{endianness}32s", data,
                                   0x028)[0]).decode(encoding="utf-8")
  n_fast_frames: int = struct.unpack_from(f"{endianness}L", data, 0x048)[0]
  size_wfm_header: int = struct.unpack_from(f"{endianness}H", data, 0x04C)[0]

  if len(data) < (size_static_header + size_wfm_header):
    raise ValueError("File is shorter than minimum")
  # Section: Waveform header
  print(size_wfm_header)
  return None