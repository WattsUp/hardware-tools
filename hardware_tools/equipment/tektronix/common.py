"""Common functions among Tektronix scopes
"""

import re
from typing import Tuple

import numpy as np

from hardware_tools.equipment import utility
from hardware_tools.math.lines import EdgePolarity
from hardware_tools.math.stats import Comparison

_rng = np.random.default_rng()


def _comparison(v: str) -> Comparison:
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
      "INrange": Comparison.WITHIN,
      "OUTrange": Comparison.OUTSIDE
  }
  for kd, vd in d.items():
    kd: str
    kk_short = re.sub(r"[a-z]+$", "", kd)
    if v.startswith(kk_short):
      return vd
  return v


def _polarity(v: str) -> EdgePolarity:
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


def _threshold(v: str) -> float:
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
                "SLOpe": _polarity,
                "SOUrce": str.upper
            },
            "HOLDoff": {
                "TIMe": float
            },
            "LEVel": {
                "AUXin": _threshold,
                "CH<1-4>": _threshold,
                "D<0-15>": _threshold
            },
            "MODe": str.upper,
            "PULse": {
                "CLAss": str.upper
            },
            "PULSEWidth": {
                "HIGHLimit": float,
                "LOWLimit": float,
                "POLarity": _polarity,
                "SOUrce": str.upper,
                "WHEn": _comparison,
                "WIDth": float
            },
            "TIMEOut": {
                "POLarity": _polarity,
                "SOUrce": str.upper,
                "TIMe": float
            },
            "TYPe": str.upper
        }
    },
    "WFMOpre|WFMInpre|WFMPre": {
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


def parse_wfm(data: bytes,
              raw: bool = False,
              add_noise: bool = False) -> Tuple[np.ndarray, dict]:
  """Parse wfm file format

  Args:
    data: Raw data from .wfm file or equivalent source
    raw: True will return raw ADC values, False will transform into
      real-world units
    add_noise: True will add uniform noise to the LSB for antialiasing

  Returns:
    Samples: [[x0, x1,..., xn], [y0, y1,..., yn]]
    Dictionary of sampling information:
      config_string: str = Human readable string describing configuration
      x_unit: str = Unit string for horizontal axis
      y_unit: str = Unit string for vertical axis
      x_incr: float = Step between horizontal axis values
      y_incr: float = Step between vertical axis values (LSB)
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
  header = header.popitem()[1]  # Remove outer "WFMOpre|WFMInpre|WFMPre"

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

  info_dict["y_unit"] = y_unit
  info_dict["y_incr"] = y_mult

  return (np.stack([x, y]), info_dict)
