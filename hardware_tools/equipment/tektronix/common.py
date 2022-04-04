"""Common functions among Tektronix scopes
"""

from typing import Tuple

import numpy as np

_rng = np.random.default_rng()


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

  # If commentted out, they are not used
  header_names = {
      "BYT_NR": "BYT_NR",
      "BYT_N": "BYT_NR",
      # "BIT_NR": "BIT_NR",
      # "BIT_N": "BIT_NR",
      # "ENCDG": "ENCDG",
      # "ENC": "ENCDG",
      "BN_FMT": "BN_FMT",
      "BN_F": "BN_FMT",
      "BYT_OR": "BYT_OR",
      "BYT_O": "BYT_OR",
      "WFID": "WFID",
      "WFI": "WFID",
      "NR_PT": "NR_PT",
      "NR_P": "NR_PT",
      # "PT_FMT": "PT_FMT",
      # "PT_F": "PT_FMT",
      "XUNIT": "XUNIT",
      "XUN": "XUNIT",
      "XINCR": "XINCR",
      "XIN": "XINCR",
      "XZERO": "XZERO",
      "XZE": "XZERO",
      # "PT_OFF": "PT_OFF",
      # "PT_O": "PT_OFF",
      # "PT_ORDER": "PT_ORDER",
      # "PT_OR": "PT_ORDER",
      "YUNIT": "YUNIT",
      "YUN": "YUNIT",
      "YMULT": "YMULT",
      "YMU": "YMULT",
      "YOFF": "YOFF",
      "YOF": "YOFF",
      "YZERO": "YZERO",
      "YZE": "YZERO",
      # "VSCALE": "VSCALE",
      # "HSCALE": "HSCALE",
      # "VPOS": "VPOS",
      # "VOFFSET": "VOFFSET",
      # "HDELAY": "HDELAY",
      # "DOMAIN": "DOMAIN",
      # "DOM": "DOMAIN",
      # "WFMTYPE": "WFMTYPE",
      # "WFMTYP": "WFMTYPE",
      # "CENTERFREQUENCY": "CENTERFREQUENCY",
      # "CENTERFREQ": "CENTERFREQUENCY",
      # "REFLEVEL": "REFLEVEL",
      # "REFLE": "REFLEVEL",
      # "SPAN": "SPAN"
  }
  header_types = {
      "BYT_NR": int,
      "BIT_NR": int,
      "ENCDG": str,
      "BN_FMT": str,
      "BYT_OR": str,
      "WFID": lambda s: str.strip(s, '"'),
      "NR_PT": int,
      "PT_FMT": str,
      "XUNIT": lambda s: str.strip(s, '"'),
      "XINCR": float,
      "XZERO": float,
      "PT_OFF": int,
      "PT_ORDER": str,
      "YUNIT": lambda s: str.strip(s, '"'),
      "YMULT": float,
      "YOFF": float,
      "YZERO": float,
      "VSCALE": float,
      "HSCALE": float,
      "VPOS": float,
      "VOFFSET": float,
      "HDELAY": float,
      "DOMAIN": str,
      "WFMTYPE": str,
      "CENTERFREQUENCY": float,
      "SPAN": float,
      "REFLEVEL": float
  }

  header = {}
  for i in waveform_format.strip(";").split(";"):
    (k, v) = i.split(" ", maxsplit=1)
    k = k.upper()
    k = k.replace(":WFMOUTPRE:", "")
    k = k.replace(":WFMO:", "")
    k = k.replace(":WFMPRE:", "")
    k = k.replace(":WFMP:", "")
    if k not in header_names:
      continue
      # raise ValueError(f"Unrecognized header '{k}'")
    k = header_names[k]
    header[k] = header_types[k](v)

  # for k, v in header.items():
  #   print(f"{k}: {v}")

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
