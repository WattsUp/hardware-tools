"""Tektronix oscilloscopes a part of the MSO4000 family

MDO4000
MSO4000
MDO3000
"""

import time
from typing import Any, Tuple

import numpy as np

from hardware_tools.equipment import scope
from hardware_tools.equipment.tektronix import common

_rng = np.random.default_rng()


class MSO4000(scope.Scope):
  """Tektronix Digital Oscilloscope model MSO4000
  """

  def __init__(self, address: str, **kwargs) -> None:
    """Initialize MSO4000 Scope

    Args:
      address: Address to the Equipment (VISA resource string)

    Raises:
      ValueError if ID does not match scope type
    """
    super().__init__(address)
    query = self.ask("*IDN?")
    if query.startswith("TEKTRONIX,"):
      self._name = "-".join(query.split(",")[:2])
    else:
      e = f"{address} did not connect to a Tektronix scope\n"
      e += f"  '*IDN?' returned '{query}'"
      raise ValueError(e)
    if "check_identity" not in kwargs or kwargs["check_identity"]:
      if not self._name.startswith("TEKTRONIX-MSO4"):
        e = f"{address} did not connect to a Tektronix MSO4000 scope\n"
        e += f"  '*IDN?' returned '{query}'"
        raise ValueError(e)

    self.send("HEADER OFF")
    self.send("VERBOSE ON")

    # Add specific settings and commands
    if self._name[16] == "2":
      self.channels = ["CH1", "CH2"]
    elif self._name[16] == "4":
      self.channels = ["CH1", "CH2", "CH3", "CH4"]
    else:
      e = f"{address} is Tektronix but unknown channel count\n"
      e += f"  '*IDN?' returned '{query}'"
      raise ValueError(e)

    self.settings.extend([])
    # TODO add other trigger options
    # TODO add acquire modes settings
    self.channel_settings.extend([])
    self.commands.extend(["AUTOSCALE", "CLEARMENU"])

    name_freq = self._name[14:16]
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
      e = f"{address} is Tektronix but unknown bandwidth\n"
      e += f"  '*IDN?' returned '{query}'"
      raise ValueError(e)

  def configure(self, setting: str, value: Any) -> Any:
    setting = setting.upper()

    if setting == "SAMPLE_RATE":
      try:
        sample_rate = value[0]
        n_points = value[1]
      except TypeError as e:
        raise ValueError(
            "SAMPLE_RATE expects value to be (sample_rate, n_points)") from e
      n_points = self.configure("TIME_POINTS", n_points)
      time_per_div = (n_points / sample_rate) / 10
      self.configure("TIME_SCALE", time_per_div)
      return float(self.ask("HORIZONTAL:SAMPLERATE?"))
    elif setting == "TIME_SCALE":
      self.send(f"HORIZONTAL:SCALE {float(value):.6E}")
      return float(self.ask("HORIZONTAL:SCALE?"))
    elif setting == "TIME_OFFSET":
      self.send("HORIZONTAL:DELAY:MODE ON")
      self.send(f"HORIZONTAL:DELAY:TIME {float(value):.6E}")
      return float(self.ask("HORIZONTAL:DELAY:TIME?"))
    elif setting == "TIME_POINTS":
      self.send(f"HORIZONTAL:RECORDLENGTH {int(value)}")
      return int(self.ask("HORIZONTAL:RECORDLENGTH?"))
    elif setting == "TRIGGER_MODE":
      value = value.upper()
      if value not in ["AUTO", "NORM", "NORMAL"]:
        raise ValueError(f"{self} cannot set trigger mode '{value}'")
      self.send(f"TRIGGER:A:MODE {value}")
      return self.ask("TRIGGER:A:MODE?")
    elif setting == "TRIGGER_SOURCE":
      value = value.upper()
      choices = ["AUX", "CH1", "CH2", "CH3", "CH4", "LINE", "RF"]
      choices.extend([f"D{i}" for i in range(16)])
      if value not in choices:
        raise ValueError(f"{self} cannot set trigger off of channel '{value}'")
      self.send("TRIGGER:A:TYPE EDGE")
      self.send(f"TRIGGER:A:EDGE:SOURCE {value}")
      return self.ask("TRIGGER:A:EDGE:SOURCE?")
    elif setting == "TRIGGER_COUPLING":
      value = value.upper()
      choices = [
          "AC", "DC", "HFR", "HFREJ", "LFR", "LFREJ", "NOISE", "NOISEREJ"
      ]
      if value not in choices:
        raise ValueError(f"{self} cannot set trigger coupling to '{value}'")
      self.send("TRIGGER:A:TYPE EDGE")
      self.send(f"TRIGGER:A:EDGE:COUPLING {value}")
      return self.ask("TRIGGER:A:EDGE:COUPLING?")
    elif setting == "TRIGGER_POLARITY":
      value = value.upper()
      if value not in ["RIS", "RISE", "FALL", "EITH", "EITHER"]:
        raise ValueError(f"{self} cannot set trigger polarity to '{value}'")
      self.send("TRIGGER:A:TYPE EDGE")
      self.send(f"TRIGGER:A:EDGE:SLOPE {value}")
      return self.ask("TRIGGER:A:EDGE:SLOPE?")
    elif setting == "ACQUIRE_MODE":
      value = value.upper()
      choices = [
          "SAM", "SAMPLE", "PEAK", "PEAKDETECT", "HIR", "HIRES", "AVE",
          "AVERAGE", "ENV", "ENVELOPE"
      ]
      if value not in choices:
        raise ValueError(f"{self} cannot set acquire mode to '{value}'")
      self.send(f"ACQUIRE:MODE {value}")
      return self.ask("ACQUIRE:MODE?")
    raise KeyError(f"{self} cannot configure setting '{setting}'")

  def configure_channel(self, channel: str, setting: str, value: Any) -> Any:
    setting = setting.upper()
    channel = channel.upper()

    if setting == "SCALE":
      if channel in ["CH1", "CH2", "CH3", "CH4"]:
        self.send(f"{channel}:SCALE {float(value):.6E}")
        return float(self.ask(f"{channel}:SCALE?"))
    elif setting == "POSITION":
      if channel in ["CH1", "CH2", "CH3", "CH4"]:
        self.send(f"{channel}:POSITION {float(value):.6E}")
        return float(self.ask(f"{channel}:POSITION?"))
    elif setting == "OFFSET":
      if channel in ["CH1", "CH2", "CH3", "CH4"]:
        self.send(f"{channel}:OFFSET {float(value):.6E}")
        return float(self.ask(f"{channel}:OFFSET?"))
    elif setting == "LABEL":
      if channel in ["CH1", "CH2", "CH3", "CH4"]:
        value = value.encode("ascii", errors="ignore").decode()
        self.send(f"{channel}:LABEL '{value[:30]}'")
        return self.ask(f"{channel}:LABEL?")
    elif setting == "BANDWIDTH":
      if channel in ["CH1", "CH2", "CH3", "CH4"]:
        if isinstance(value, str) and value in ["FULL", "FUL"]:
          self.send(f"{channel}:BANDWIDTH {value}")
        else:
          self.send(f"{channel}:BANDWIDTH {float(value):.6E}")
        return float(self.ask(f"{channel}:BANDWIDTH?"))
    elif setting == "TERMINATION":
      if channel in ["CH1", "CH2", "CH3", "CH4"]:
        self.send(f"{channel}:TERMINATION {value}")
        return float(self.ask(f"{channel}:TERMINATION?"))
    elif setting == "INVERT":
      if channel in ["CH1", "CH2", "CH3", "CH4"]:
        value = int(value)
        self.send(f"{channel}:INVERT {value}")
        return bool(self.ask(f"{channel}:INVERT?"))
    elif setting == "PROBE_ATTENUATION":
      if channel in ["CH1", "CH2", "CH3", "CH4"]:
        self.send(f"{channel}:PROBE:GAIN {1 / float(value):.6E}")
        return 1 / float(self.ask(f"{channel}:PROBE:GAIN?"))
    elif setting == "PROBE_GAIN":
      if channel in ["CH1", "CH2", "CH3", "CH4"]:
        self.send(f"{channel}:PROBE:GAIN {float(value):.6E}")
        return float(self.ask(f"{channel}:PROBE:GAIN?"))
    elif setting == "COUPLING":
      if channel in ["CH1", "CH2", "CH3", "CH4"]:
        value = value.upper()
        if value not in ["AC", "DC", "DCREJ", "DCREJECT"]:
          raise ValueError(
              f"{self} cannot set channel '{channel}' coupling to '{value}'")
        self.send(f"{channel}:COUPLING {value}")
        return self.ask(f"{channel}:COUPLING?")
    elif setting == "ACTIVE":
      value = int(value)
      self.send(f"SELECT:{channel} {value}")
      return bool(self.ask(f"SELECT:{channel}?"))
    elif setting == "TRIGGER_LEVEL":
      if value in ["ECL", "TTL"]:
        self.send(f"TRIGGER:A:LEVEL:{channel} {value}")
      else:
        self.send(f"TRIGGER:A:LEVEL:{channel} {float(value):.6E}")
      return float(self.ask(f"TRIGGER:A:LEVEL:{channel}?"))

    raise KeyError(
        f"{self} cannot configure channel '{channel}' setting '{setting}'")

  def command(self,
              command: str,
              timeout: float = 1,
              silent: bool = True,
              channel: str = None) -> None:
    command = command.upper()

    if command == "STOP":
      self.send("ACQUIRE:STATE STOP")
      self.ask_and_wait("TRIGGER:STATE?", ["SAVE"], timeout=timeout)
      return
    elif command == "RUN":
      self.send("ACQUIRE:STATE STOP")
      self.send("ACQUIRE:STOPAFTER RUNSTOP")
      self.send("ACQUIRE:STATE RUN")
      self.ask_and_wait("TRIGGER:STATE?", ["ARMED", "AUTO", "TRIGGER", "READY"],
                        timeout=timeout)
      return
    elif command == "FORCE_TRIGGER":
      self.ask_and_wait("TRIGGER:STATE?", ["READY", "AUTO", "SAVE", "TRIGGER"],
                        timeout=timeout)
      self.send("TRIGGER FORCE")
      return
    elif command == "SINGLE":
      self.send("ACQUIRE:STATE STOP")
      self.ask_and_wait("TRIGGER:STATE?", ["SAVE"], timeout=timeout)
      self.send("ACQUIRE:STOPAFTER SEQUENCE")
      self.send("ACQUIRE:STATE RUN")
      time.sleep(0.1)
      self.ask_and_wait("ACQUIRE:STATE?", ["0"], timeout=timeout)
      self.ask_and_wait("TRIGGER:STATE?", ["SAVE"], timeout=timeout)
      self.ask_and_wait("ACQUIRE:NUMACQ?", ["1"])
      return
    elif command == "SINGLE_FORCE":
      self.send("ACQUIRE:STATE STOP")
      self.ask_and_wait("TRIGGER:STATE?", ["SAVE"], timeout=timeout)
      self.send("ACQUIRE:STOPAFTER SEQUENCE")
      self.send("ACQUIRE:STATE RUN")
      time.sleep(0.1)
      self.ask_and_wait("ACQUIRE:STATE?", ["0"],
                        timeout=timeout,
                        additional_command="TRIGGER FORCE")
      self.ask_and_wait("TRIGGER:STATE?", ["SAVE"], timeout=timeout)
      self.ask_and_wait("ACQUIRE:NUMACQ?", ["1"])
      return
    elif command == "AUTOSCALE":
      if channel is None:
        raise ValueError(f"{self} cannot autoscale channel '{channel}'")
      channel = channel.upper()
      if channel not in self.channels:
        raise ValueError(f"{self} cannot autoscale channel '{channel}'")

      if not silent:
        print(f"Autoscaling channel '{channel}'")

      original_num_points = self.ask("HORIZONTAL:RECORDLENGTH?")
      self.configure("TIME_POINTS", 1e6)

      attempts = 10
      while attempts > 0:
        if attempts != 10 and not silent:
          print(f"  Remaining attempts: {attempts}")
        self.command("SINGLE_FORCE")
        data = self.read_waveform(channel=channel, raw=True)[0][1]
        data = data / 254  # -127 to 127 => -0.5 to 0.5
        attempts -= 1

        position = float(self.ask(f"{channel}:POSITION?"))
        scale = float(self.ask(f"{channel}:SCALE?"))
        new_scale = scale
        new_position = position

        data_min = data.min()
        data_max = data.max()
        data_mid = (data_min + data_max) / 2
        data_span = (data_max - data_min)

        # print(f"{data_min:.2f}, {data_mid:.2f}, {data_max:.2f}, "
        #       f"{data_span:.2f}, {position:.2f}, {scale}")

        update = False
        if data_max > 0.45:
          if not silent:
            print("    Too high")
          if data_span > 0.6:
            new_scale = scale * 4
          if data_span < 0.1:
            new_scale = scale / 4
          new_position = (position - 10 * data_mid) * scale / new_scale
          update = True
        elif data_min < -0.45:
          if not silent:
            print("    Too low")
          if data_span > 0.6:
            new_scale = scale * 4
          if data_span < 0.1:
            new_scale = scale / 4
          new_position = (position - 10 * data_mid) * scale / new_scale
          update = True
        elif data_span < 0.05:
          if not silent:
            print("    Too small")
          new_scale = scale / 2
          new_position = (position - 10 * data_mid) * scale / new_scale
          update = True
        # Covered by too high and too low
        # elif data_span > 0.9:
        #   if not silent:
        #     print("    Too large")
        #   new_scale = scale * 2
        #   new_position = (position - 10 * data_mid) * scale / new_scale
        else:
          if data_span < 0.75 or data_span > 0.85:
            if not silent:
              print("    Adjusting scale")
            new_scale = scale / (0.8 / data_span)
            update = True

          if data_mid > 0.1 or data_mid < -0.1 or new_scale != scale:
            if not silent:
              print("    Adjusting position")
            new_position = (position - 10 * data_mid) * scale / new_scale
            update = True

        if update:
          if not silent:
            print(f"  Scale: {scale:.6g}=>{new_scale:.6g}")
            print(f"  Position: {position:.2f}=>{new_position:.2f}")
          self.configure_channel(channel, "SCALE", new_scale)
          self.configure_channel(channel, "POSITION", new_position)
          continue
        else:
          if not silent:
            print("  Complete")
          break  # pragma: no cover complains this isn't reached

      self.configure("TIME_POINTS", original_num_points)

      if attempts == 0:
        raise TimeoutError(f"{self} failed to autoscale channel '{channel}'")

      return
    elif command == "CLEAR_MENU":
      self.send("CLEARMENU")
      return

    raise KeyError(f"{self} cannot perform command '{command}'")

  def read_waveform(self,
                    channel: str,
                    raw: bool = False,
                    add_noise: bool = False) -> Tuple[np.ndarray, dict]:
    channel = channel.upper()
    if channel not in self.channels:
      raise KeyError(f"{self} cannot read channel '{channel}'")
    self.configure_channel(channel, "ACTIVE", True)

    self.send(f"DATA:SOURCE {channel}")
    self.send("DATA:START 1")
    self.send("DATA:STOP 1E9")
    self.send("DATA:WIDTH 1")
    self.send("DATA:ENCDG FASTEST")  # BINARY, signed

    self.send("HEADER 1")
    self._instrument.write("WAVFRM?")
    data = self._instrument.read_raw()
    self.send("HEADER 0")

    return common.parse_wfm(data, raw=raw, add_noise=add_noise)


class MDO4000(MSO4000):
  """Tektronix Mixed Domain Oscilloscope model MDO4000
  """

  def __init__(self, address: str, **kwargs) -> None:
    super().__init__(address, check_identity=False)
    if "check_identity" not in kwargs or kwargs["check_identity"]:
      if not self._name.startswith("TEKTRONIX-MDO4"):
        query = self.ask("*IDN?")
        e = f"{address} did not connect to a Tektronix MDO4000 scope\n"
        e += f"  '*IDN?' returned '{query}'"
        raise ValueError(e)

    # TODO add other channel operations: RF


class MDO3000(MSO4000):
  """Tektronix Mixed Domain Oscilloscope model MDO3000
  """

  def __init__(self, address: str, **kwargs) -> None:
    super().__init__(address, check_identity=False)
    if "check_identity" not in kwargs or kwargs["check_identity"]:
      if not self._name.startswith("TEKTRONIX-MDO3"):
        query = self.ask("*IDN?")
        e = f"{address} did not connect to a Tektronix MDO3000 scope\n"
        e += f"  '*IDN?' returned '{query}'"
        raise ValueError(e)

    # TODO add other channel operations: RF
