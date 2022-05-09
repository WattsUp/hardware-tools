"""Tektronix oscilloscopes a part of the MSO4000 family

MDO4000
MSO4000
MDO3000
"""

import re
import time
from typing import Tuple

import numpy as np

from hardware_tools.equipment import utility
from hardware_tools.equipment.scope import (Scope, AnalogChannel, Channel,
                                            DigitalChannel, SampleMode, Trigger,
                                            TriggerEdge, TriggerEdgeTimeout,
                                            TriggerPulseWidth, EdgePolarity,
                                            Comparison)
from hardware_tools.equipment.tektronix import common


class _MSO4000Channel(Channel):
  """Base Channel for Tektronix Digital Oscilloscope 4000/3000 models
  MDO4000B, MDO4000, MSO4000B, DPO4000B and MDO3000
  """

  @property
  def position(self) -> float:
    return float(self._parent.ask(f"{self._alias}:POSITION?"))

  @position.setter
  def position(self, value: float) -> None:
    self._parent.send(f"{self._alias}:POSITION {float(value):.6E}")

  @property
  def label(self) -> str:
    return self._parent.ask(f"{self._alias}:LABEL?")

  @label.setter
  def label(self, value: str) -> None:
    value = value.encode("ascii", errors="ignore").decode()
    self._parent.send(f"{self._alias}:LABEL '{value[:30]}'")

  @property
  def active(self) -> bool:
    return bool(self._parent.ask(f"SELECT:{self._alias}?"))

  @active.setter
  def active(self, on: bool) -> None:
    self._parent.send(f"SELECT:{self._alias} {int(on)}")

  def read_waveform(self,
                    raw: bool = False,
                    add_noise: bool = False) -> Tuple[np.ndarray, dict]:
    self.active = True

    self._parent.send(f"DATA:SOURCE {self._alias}")
    self._parent.send("DATA:START 1")
    self._parent.send("DATA:STOP 1E9")
    self._parent.send("DATA:WIDTH 1")
    self._parent.send("DATA:ENCDG FASTEST")  # BINARY, signed

    self._parent.send("HEADER 1")
    self._parent.send("WAVFRM?")
    data = self._parent.receive()
    with open("out.wfm", "wb") as file:
      file.write(data)
    self._parent.send("HEADER 0")

    return common.parse_wfm(data, raw=raw, add_noise=add_noise)


class _MSO4000AnalogChannel(AnalogChannel, _MSO4000Channel):
  """AnalogChannel for Tektronix Digital Oscilloscope 4000/3000 models
  MDO4000B, MDO4000, MSO4000B, DPO4000B and MDO3000
  """

  @property
  def scale(self) -> float:
    return float(self._parent.ask(f"{self._alias}:SCALE?"))

  @scale.setter
  def scale(self, value: float) -> None:
    self._parent.send(f"{self._alias}:SCALE {float(value):.6E}")

  @property
  def bandwidth(self) -> float:
    return float(self._parent.ask(f"{self._alias}:BANDWIDTH?"))

  @bandwidth.setter
  def bandwidth(self, value: float) -> None:
    self._parent.send(f"{self._alias}:BANDWIDTH {float(value):.6E}")

  @property
  def dc_coupling(self) -> bool:
    return self._parent.ask(f"{self._alias}:COUPLING?") == "DC"

  @dc_coupling.setter
  def dc_coupling(self, dc: bool) -> None:
    if dc:
      self._parent.send(f"{self._alias}:COUPLING DC")
    else:
      self._parent.send(f"{self._alias}:COUPLING AC")

  @property
  def deskew(self) -> float:
    return float(self._parent.ask(f"{self._alias}:deskew?"))

  @deskew.setter
  def deskew(self, value: float) -> None:
    self._parent.send(f"{self._alias}:deskew {float(value):.6E}")

  @property
  def inverted(self) -> bool:
    return bool(self._parent.ask(f"{self._alias}:INVERT?"))

  @inverted.setter
  def inverted(self, invert: bool) -> None:
    self._parent.send(f"{self._alias}:INVERT {int(invert)}")

  @property
  def offset(self) -> float:
    return float(self._parent.ask(f"{self._alias}:OFFSET?"))

  @offset.setter
  def offset(self, value: float) -> None:
    self._parent.send(f"{self._alias}:OFFSET {float(value):.6E}")

  @property
  def termination(self) -> float:
    return float(self._parent.ask(f"{self._alias}:TERMINATION?"))

  @termination.setter
  def termination(self, value: float) -> None:
    self._parent.send(f"{self._alias}:TERMINATION {float(value):.6E}")

  @property
  def probe_gain(self) -> float:
    return float(self._parent.ask(f"{self._alias}:PROBE:GAIN?"))

  @probe_gain.setter
  def probe_gain(self, value: float) -> None:
    self._parent.send(f"{self._alias}:PROBE:GAIN {float(value):.6E}")


class _MSO4000DigitalChannel(DigitalChannel, _MSO4000Channel):
  """DigitalChannel for Tektronix Digital Oscilloscope 4000/3000 models
  MDO4000B, MDO4000, MSO4000B, DPO4000B and MDO3000
  """

  @property
  def threshold(self) -> float:
    return common._threshold(self._parent.ask(f"{self._alias}:THRESHOLD?"))  # pylint: disable=protected-access

  @threshold.setter
  def threshold(self, value: float) -> None:
    self._parent.send(f"{self._alias}:THRESHOLD {float(value):.6E}")


class MSO4000Family(Scope):
  """Tektronix Digital Oscilloscope 4000/3000 models
  MDO4000B, MDO4000, MSO4000B, DPO4000B and MDO3000
  """

  n_div_horz = 10
  n_div_vert = 10

  def _init_channels(self) -> None:
    id_str = self.ask("*IDN?")
    if id_str.startswith("TEKTRONIX,"):
      model = id_str.split(",")[1]
      if model[3] not in ["3", "4"]:
        e = (f"{self._address} did not connect to a Tektronix "
             "4000/3000 series scope\n"
             f"  '*IDN?' returned '{id_str}'")
        raise ValueError(e)
      if self._name == "":
        self._name = "-".join(id_str.split(",")[:2])
    else:
      e = (f"{self._address} did not connect to a Tektronix scope\n"
           f"  '*IDN?' returned '{id_str}'")
      raise ValueError(e)

    self.send("HEADER ON")
    self.send("VERBOSE ON")

    self.rf: AnalogChannel = None  # TODO (WattsUp) add RFChannel
    self._aux = False  # True indicate scope has auxiliary trigger connector

    channels = utility.parse_scpi(self.ask("SELECT?"))["SELECT"]
    for c in channels:
      match = re.match(r"^(CH|D)(\d+)$", c)
      if match:
        n = int(match.group(2))
        if match.group(1) == "CH":
          self._channels[n] = _MSO4000AnalogChannel(f"CH{n}", self)
        else:
          self._digitals[n] = _MSO4000DigitalChannel(f"D{n}", self)

    self.send("HEADER OFF")

    name_freq = model[4:6]
    if name_freq == "01":
      self.max_bandwidth = 100e6
    elif name_freq == "02":
      self.max_bandwidth = 200e6
    elif name_freq == "03":
      self.max_bandwidth = 350e6
    elif name_freq == "05":
      self.max_bandwidth = 500e6
    elif name_freq == "10":
      self.max_bandwidth = 1000e6
    else:
      e = f"{self} is Tektronix but unknown bandwidth\n"
      e += f"  '*IDN?' returned '{id_str}'"
      raise ValueError(e)

  @property
  def sample_rate(self) -> float:
    return float(self.ask("HORIZONTAL:SAMPLERATE?"))

  @sample_rate.setter
  def sample_rate(self, value: float) -> None:
    time_per_div = (self.time_points / value) / 10
    self.time_scale = time_per_div

  @property
  def sample_mode(self) -> SampleMode:
    mode = self.ask("ACQUIRE:MODE?")
    if mode.startswith("SAM"):
      return SampleMode.SAMPLE
    if mode.startswith("AVE"):
      return SampleMode.AVERAGE
    if mode.startswith("ENV"):
      return SampleMode.ENVELOPE
    # Could also be PEAKDETECT or HIRES but not enumerated
    return None

  @sample_mode.setter
  def sample_mode(self, mode: SampleMode) -> None:
    if mode == SampleMode.SAMPLE:
      self.send("ACQUIRE:MODE SAMPLE")
    elif mode == SampleMode.AVERAGE:
      self.send("ACQUIRE:MODE AVERAGE")
    elif mode == SampleMode.ENVELOPE:
      self.send("ACQUIRE:MODE ENVELOPE")
    else:
      raise ValueError(f"Unknown SampleMode {mode}")

  @property
  def sample_mode_n(self) -> int:
    mode = self.sample_mode
    if mode == SampleMode.AVERAGE:
      return int(self.ask("ACQUIRE:NUMAVG?"))
    if mode == SampleMode.ENVELOPE:
      return int(self.ask("ACQUIRE:NUMENV?"))
    return 1

  @sample_mode_n.setter
  def sample_mode_n(self, value: int) -> None:
    self.send(f"ACQUIRE:NUMAVG {int(value)}")
    self.send(f"ACQUIRE:NUMENV {int(value)}")

  @property
  def time_scale(self) -> float:
    return float(self.ask("HORIZONTAL:SCALE?"))

  @time_scale.setter
  def time_scale(self, value: float) -> None:
    self.send(f"HORIZONTAL:SCALE {float(value):.6E}")

  @property
  def time_offset(self) -> float:
    return float(self.ask("HORIZONTAL:DELAY:TIME?"))

  @time_offset.setter
  def time_offset(self, value: float) -> None:
    self.send("HORIZONTAL:DELAY:MODE ON")
    self.send(f"HORIZONTAL:DELAY:TIME {float(value):.6E}")

  @property
  def time_points(self) -> int:
    return int(self.ask("HORIZONTAL:RECORDLENGTH?"))

  @time_points.setter
  def time_points(self, value: int) -> None:
    self.send(f"HORIZONTAL:RECORDLENGTH {int(value)}")

  @property
  def trigger(self) -> Trigger:
    self.send("HEADER ON")
    settings = utility.parse_scpi(self.ask("TRIGGER:A?"),
                                  types=common.TEK_TYPES)["TRIGGER"]["A"]
    self.send("HEADER OFF")
    holdoff = settings["HOLDOFF"]["TIME"]

    t = None
    if re.match(r"^EDG(E)?$", settings["TYPE"]):
      src = settings["EDGE"]["SOURCE"]
      level = settings["LEVEL"][src]
      slope = settings["EDGE"]["SLOPE"]
      dc_coupling = (settings["EDGE"]["COUPLING"] == "DC")
      t = TriggerEdge(src, level, slope, dc_coupling, holdoff)
    elif re.match(r"^PULS(E)?$", settings["TYPE"]):
      if re.match(r"^TIMEO(UT)?$", settings["PULSE"]["CLASS"]):
        src = settings["TIMEOUT"]["SOURCE"]
        level = settings["LEVEL"][src]
        timeout = settings["TIMEOUT"]["TIME"]
        slope = settings["TIMEOUT"]["POLARITY"]
        t = TriggerEdgeTimeout(src, level, timeout, slope, holdoff)
      elif re.match(r"^WID(TH)?$", settings["PULSE"]["CLASS"]):
        src = settings["PULSEWIDTH"]["SOURCE"]
        comparison = settings["PULSEWIDTH"]["WHEN"]
        positive = settings["PULSEWIDTH"]["POLARITY"] == EdgePolarity.RISING
        if comparison in [
            Comparison.WITHIN, Comparison.WITHININC, Comparison.OUTSIDE,
            Comparison.OUTSIDEINC
        ]:
          lower = settings["PULSEWIDTH"]["LOWLIMIT"]
          upper = settings["PULSEWIDTH"]["HIGHLIMIT"]
          t = TriggerPulseWidth(src, level, (lower, upper), comparison,
                                positive, holdoff)
        else:
          width = settings["PULSEWIDTH"]["WIDTH"]
          t = TriggerPulseWidth(src, level, width, comparison, positive,
                                holdoff)
    return t

  @trigger.setter
  def trigger(self, value: Trigger) -> None:
    self.send(f"TRIGGER:A:HOLDOFF:TIME {float(value.holdoff):.6E}")
    src_a = [f"CH{i}" for i in self._channels]
    src_d = [f"D{i}" for i in self._digitals]
    if isinstance(value, TriggerEdge):
      self.send("TRIGGER:A:TYPE EDGE")
      sources = ["LINE"]
      sources.extend(src_a)
      sources.extend(src_d)
      if self.rf:
        sources.append("RF")
      if self._aux:
        sources.append("AUX")
      if value.src not in sources:
        raise ValueError(f"Trigger.src not available '{value.src}'")
      self.send(f"TRIGGER:A:EDGE:SOURCE {value.src}")
      if value.src == "RF":
        pass  # TODO (WattsUp) Add RF trigger
      else:
        self.send(f"TRIGGER:A:LEVEL:{value.src} {float(value.level):.6E}")

      polarities = {
          EdgePolarity.RISING: "RISE",
          EdgePolarity.FALLING: "FALL",
          EdgePolarity.BOTH: "EITHER",
      }
      self.send(f"TRIGGER:A:EDGE:SLOPE {polarities[value.slope]}")

      if value.dc_coupling:
        self.send("TRIGGER:A:EDGE:COUPLING DC")
      else:
        self.send("TRIGGER:A:EDGE:COUPLING AC")
    elif isinstance(value, TriggerEdgeTimeout):
      self.send("TRIGGER:A:TYPE PULSE")
      self.send("TRIGGER:A:PULSE:CLASS TIMEOUT")
      sources = ["LINE"]
      sources.extend(src_a)
      sources.extend(src_d)
      if self.rf:
        sources.append("RF")
      if self._aux:
        sources.append("AUX")
      if value.src not in sources:
        raise ValueError(f"Trigger.src not available '{value.src}'")
      self.send(f"TRIGGER:A:TIMEOUT:SOURCE {value.src}")
      if value.src == "RF":
        pass  # TODO (WattsUp) Add RF trigger
      else:
        self.send(f"TRIGGER:A:LEVEL:{value.src} {float(value.level):.6E}")

      self.send(f"TRIGGER:A:TIMEOUT:TIME {float(value.timeout):.6E}")
      polarities = {
          EdgePolarity.RISING: "STAYSHIGH",
          EdgePolarity.FALLING: "STAYSLOW",
          EdgePolarity.BOTH: "EITHER",
      }
      self.send(f"TRIGGER:A:TIMEOUT:POLARITY {polarities[value.slope]}")
    elif isinstance(value, TriggerPulseWidth):
      self.send("TRIGGER:A:TYPE PULSE")
      self.send("TRIGGER:A:PULSE:CLASS WIDTH")
      sources = ["LINE"]
      sources.extend(src_a)
      sources.extend(src_d)
      if self.rf:
        sources.append("RF")
      if self._aux:
        sources.append("AUX")
      if value.src not in sources:
        raise ValueError(f"Trigger.src not available '{value.src}'")
      self.send(f"TRIGGER:A:PULSEWIDTH:SOURCE {value.src}")
      if value.src == "RF":
        pass  # TODO (WattsUp) Add RF trigger
      else:
        self.send(f"TRIGGER:A:LEVEL:{value.src} {float(value.level):.6E}")

      if value.comparison in [
          Comparison.WITHIN, Comparison.WITHININC, Comparison.OUTSIDE,
          Comparison.OUTSIDEINC
      ]:
        try:
          self.send("TRIGGER:A:PULSEWIDTH:LOWLIMIT "
                    f"{float(value.width[0]):.6E}")
          self.send("TRIGGER:A:PULSEWIDTH:HIGHLIMIT "
                    f"{float(value.width[1]):.6E}")
        except TypeError as e:
          raise ValueError("TriggerPulseWidth.width should be a tuple (lower, "
                           "upper) when using WITHIN/OUTSIDE comparison") from e
      else:
        self.send(f"TRIGGER:A:PULSEWIDTH:WIDTH {float(value.width):.6E}")
      whens = {
          Comparison.LESS: "LESSTHAN",
          Comparison.LESSEQUAL: "LESSTHAN",
          Comparison.MORE: "MORETHAN",
          Comparison.MOREEQUAL: "MORETHAN",
          Comparison.EQUAL: "EQUAL",
          Comparison.UNEQUAL: "UNEQUAL",
          Comparison.WITHIN: "WITHIN",
          Comparison.WITHININC: "WITHIN",
          Comparison.OUTSIDE: "OUTSIDE",
          Comparison.OUTSIDEINC: "OUTSIDE",
      }
      self.send(f"TRIGGER:A:PULSEWIDTH:WHEN {whens[value.comparison]}")
      if value.positive:
        self.send("TRIGGER:A:PULSEWIDTH:POLARITY POSITIVE")
      else:
        self.send("TRIGGER:A:PULSEWIDTH:POLARITY NEGATIVE")
    else:
      raise ValueError(f"Unknown Trigger type {type(value)}")

  def stop(self, timeout: float = 1) -> None:
    self.send("ACQUIRE:STATE STOP")
    if timeout is not None:
      self.ask_and_wait("TRIGGER:STATE?", ["SAVE"], timeout=timeout)

  def run(self, normal: bool = True, timeout: float = 1) -> None:
    self.send("ACQUIRE:STATE STOP")
    self.send("ACQUIRE:STOPAFTER RUNSTOP")
    if normal:
      self.send("TRIGGER:A:MODE NORMAL")
    else:
      self.send("TRIGGER:A:MODE AUTO")

    self.send("ACQUIRE:STATE RUN")
    if timeout is not None:
      self.ask_and_wait("TRIGGER:STATE?", ["ARMED", "AUTO", "TRIGGER", "READY"],
                        timeout=timeout)

  def single(self,
             trigger_cmd: callable = None,
             force: bool = False,
             timeout: float = 1) -> None:

    def default_trigger_cmd():
      time.sleep(0.1)

    if trigger_cmd is None:
      trigger_cmd = default_trigger_cmd

    self.send("ACQUIRE:STATE STOP")
    self.ask_and_wait("TRIGGER:STATE?", ["SAVE"], timeout=timeout)
    self.send("ACQUIRE:STOPAFTER SEQUENCE")
    self.send("ACQUIRE:STATE RUN")
    trigger_cmd()
    self.ask_and_wait("ACQUIRE:STATE?", ["0"],
                      timeout=timeout,
                      additional_command=("TRIGGER FORCE" if force else None))
    self.ask_and_wait("TRIGGER:STATE?", ["SAVE"], timeout=timeout)
    self.ask_and_wait("ACQUIRE:NUMACQ?", ["1"])

  def force(self) -> None:
    self.send("TRIGGER FORCE")
