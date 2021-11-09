"""Tektronix oscilloscopes a part of the MSO4000 family

MDO4000
MSO4000
MDO3000
"""

import struct
import time
from typing import Any

import numpy as np

from hardware_tools.equipment import scope


class MSO4000(scope.Scope):
  """Tektronix Digital Oscilloscope model MSO4000
  """

  def __init__(self, address: str, **kwargs) -> None:
    """Initialize MSO4000 Scope

    Args:
      address: Address to the Equipment (VISA resource string)
    """
    super().__init__(address, name="TEKTRONIX-MSO4000")
    if "check_identity" not in kwargs or kwargs["check_identity"]:
      query = self.ask("*IDN?")
      if query.startswith("TEKTRONIX,MSO4"):
        self._name = "-".join(query.split(",")[:2])
      else:
        e = f"{address} did not connect to a Tektronix MSO4000 scope\n"
        e += f"  '*IDN?' returned '{query}'"
        raise ValueError(e)

    self.send("HEADER OFF")
    self.send("VERBOSE ON")

    # Add specific settings and commands
    if self._name[-1] == "2":
      self.channels = ["CH1", "CH2"]
    else:
      self.channels = ["CH1", "CH2", "CH3", "CH4"]

    self.settings.extend([])
    # TODO add other trigger options
    # TODO add acquire modes settings
    self.channel_settings.extend([])
    self.commands.extend(["AUTOSCALE", "CLEARMENU"])

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
      if value not in self.channels:
        raise ValueError(f"{self} cannot set trigger off of chanel '{value}'")
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
              f"{self} cannot set chanel '{channel}' coupling to '{value}'")
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
    if command not in self.commands:
      raise Exception(
          f'{self.name}@{self.addr} cannot perform command \'{command}\'')

    if command == 'STOP':
      self.send('ACQUIRE:STATE STOP')
      self.waitForReply('TRIGGER:STATE?', ['SAVE'], timeout=timeout)
      return
    elif command == 'RUN':
      self.send('ACQUIRE:STATE STOP')
      self.send('ACQUIRE:STOPAFTER RUNSTOP')
      self.send('ACQUIRE:STATE RUN')
      self.waitForReply('TRIGGER:STATE?', ['ARMED', 'AUTO', 'TRIGGER', 'READY'],
                        timeout=timeout)
      return
    elif command == 'FORCE_TRIGGER':
      self.waitForReply('TRIGGER:STATE?', ['READY', 'AUTO', 'SAVE', 'TRIGGER'],
                        timeout=timeout)
      self.send('TRIGGER FORCE')
      return
    elif command == 'SINGLE':
      self.send('ACQUIRE:STATE STOP')
      self.waitForReply('TRIGGER:STATE?', ['SAVE'], timeout=timeout)
      self.send('ACQUIRE:STOPAFTER SEQUENCE')
      self.send('ACQUIRE:STATE RUN')
      time.sleep(0.1)
      self.waitForReply('ACQUIRE:STATE?', ['0'], timeout=timeout)
      self.waitForReply('TRIGGER:STATE?', ['SAVE'], timeout=timeout)
      self.waitForReply('ACQUIRE:NUMACQ?', ['1'])
      return
    elif command == 'SINGLE_FORCE':
      self.send('ACQUIRE:STATE STOP')
      self.waitForReply('TRIGGER:STATE?', ['SAVE'], timeout=timeout)
      self.send('ACQUIRE:STOPAFTER SEQUENCE')
      self.send('ACQUIRE:STATE RUN')
      time.sleep(0.1)
      try:
        self.waitForReply('ACQUIRE:STATE?', ['0'],
                          timeout=timeout,
                          repeatSend='TRIGGER FORCE')
        self.waitForReply('TRIGGER:STATE?', ['SAVE'], timeout=timeout)
        self.waitForReply('ACQUIRE:NUMACQ?', ['1'])
        return
      except Exception:
        pass

      # Needs a more help
      self.waitForReply('TRIGGER:STATE?', ['READY'], timeout=timeout)
      self.waitForReply('TRIGGER:STATE?', ['TRIGGER', 'SAVE'],
                        timeout=timeout,
                        repeatSend='TRIGGER FORCE')
      self.waitForReply('ACQUIRE:STATE?', ['0'], timeout=timeout)
      self.waitForReply('TRIGGER:STATE?', ['SAVE'], timeout=timeout)
      self.waitForReply('ACQUIRE:NUMACQ?', ['1'])
      return
    elif command == 'AUTOSCALE':
      if channel is None:
        raise Exception(
            f'{self.name}@{self.addr} cannot autoscale chanel \'None\'')
      channel = channel.upper()
      if channel not in self.channels:
        raise Exception(
            f'{self.name}@{self.addr} cannot autoscale chanel \'{channel}\'')

      if not silent:
        print(f'Autoscaling channel \'{channel}\'')

      originalNumPoints = self.ask(f'HORIZONTAL:RECORDLENGTH?')
      self.configure("TIME_POINTS", 1e6)

      attempts = 10
      while attempts > 0:
        if attempts != 10 and not silent:
          print(f' Remaining attempts: {attempts}')
        self.command('SINGLE_FORCE')
        data = self.readWaveform(channel=channel, raw=True)[0][1]
        data = data / 254  # -127 to 127
        attempts -= 1

        position = float(self.ask(f'{channel}:POSITION?'))
        scale = float(self.ask(f'{channel}:SCALE?'))
        newScale = scale
        newPosition = position

        dataMin = np.min(data)
        dataMax = np.max(data)
        dataMid = (dataMin + dataMax) / 2
        range = (dataMax - dataMin)
        # print(f'{dataMin:.2f}, {dataMid:.2f}, {dataMax:.2f}, {range:.2f}, {position:.2f}, {scale}')

        if dataMax > 0.45:
          if not silent:
            print('    Too high')
          if range > 0.6:
            newScale = scale * 4
          if range < 0.1:
            newScale = scale / 4
          newPosition = (position - 10 * dataMid) * scale / newScale
        elif dataMin < -0.45:
          if not silent:
            print('    Too low')
          if range > 0.6:
            newScale = scale * 4
          if range < 0.1:
            newScale = scale / 4
          newPosition = (position - 10 * dataMid) * scale / newScale
        elif range < 0.05:
          if not silent:
            print('    Too small')
          newScale = scale / 2
          newPosition = (position - 10 * dataMid) * scale / newScale
        elif range > 0.9:
          if not silent:
            print('    Too large')
          newScale = scale * 2
          newPosition = (position - 10 * dataMid) * scale / newScale
        else:
          if range < 0.75 or range > 0.85:
            if not silent:
              print('    Adjusting scale')
            newScale = scale / (0.8 / range)

          if dataMid > 0.1 or dataMid < -0.1 or newScale != scale:
            if not silent:
              print('    Adjusting position')
            newPosition = (position - 10 * dataMid) * scale / newScale

        if newPosition != position or newScale != scale:
          if not silent:
            print(f'  Scale: {scale:.6g}=>{newScale:.6g}')
            print(f'  Position: {position:.2f}=>{newPosition:.2f}')
          self.configureChannel(channel, 'SCALE', newScale)
          self.configureChannel(channel, 'POSITION', newPosition)
          continue

        break

      self.configure("TIME_POINTS", originalNumPoints)

      if attempts == 0:
        raise Exception(
            f'{self.name}@{self.addr} failed to autoscale channel \'{channel}\''
        )

      return
    elif command == 'CLEARMENU':
      self.send('CLEARMENU')
      return

    raise Exception(
        f'{self.name}@{self.addr} cannot perform command \'{command}\'')

  def read_waveform(self,
                    channel: str,
                    raw: bool = False,
                    addNoise: bool = False) -> tuple[np.ndarray, dict]:
    channel = channel.upper()
    if channel not in self.channels:
      raise Exception(f"{self} cannot read chanel '{channel}'")

    self.send(f"DATA:SOURCE {channel}")
    self.send("DATA:START 1")
    self.send("DATA:STOP 1E9")
    self.send("DATA:WIDTH 1")
    self.send("DATA:ENCDG FASTEST")  # BINARY, signed

    self.send("HEADER 1")
    interpretInfo = [
        i.split(" ", maxsplit=1) for i in self.ask("WFMOUTPRE?")[11:].split(";")
    ]
    interpretInfo = {i[0]: i[1] for i in interpretInfo}
    self.send("HEADER 0")

    points = int(interpretInfo["NR_PT"])
    xIncr = float(interpretInfo["XINCR"])
    xZero = float(interpretInfo["XZERO"])
    xUnit = interpretInfo["XUNIT"].replace('"', "")

    infoDict = {
        "tUnit": xUnit,
        "yUnit": "ADC Counts",
        "tIncr": xIncr,
        "yIncr": 1
    }

    self.instrument.write("CURVE?")
    data = self.instrument.read_raw()
    headerLen = 2 + int(chr(data[1]), 16)
    wave = data[headerLen:-1]
    wave = np.array(struct.unpack(">%sb" % points, wave)).astype(np.float32)
    x = np.arange(xZero, xIncr * points + xZero, xIncr).astype(np.float32)

    infoDict["clippingTop"] = np.amax(wave) >= 127
    infoDict["clippingBottom"] = np.amin(wave) <= -127

    if addNoise:
      wave += np.random.uniform(-0.5, 0.5, points)
      # Don"t add noise to clipping values
      wave[np.where(wave >= 126.5)] = 127
      wave[np.where(wave < -126.5)] = -127

    if raw:
      return (np.stack([x, wave]), infoDict)

    yMult = float(interpretInfo["YMULT"])
    yZero = float(interpretInfo["YZERO"])
    yOff = float(interpretInfo["YOFF"])
    yUnit = interpretInfo["YUNIT"].replace('"', "")
    y = (wave - yOff) * yMult + yZero

    infoDict["yUnit"] = yUnit
    infoDict["yIncr"] = yMult

    return (np.stack([x, y]), infoDict)


class MDO4000(MSO4000):
  """Tektronix Mixed Domain Oscilloscope model MDO4000
  """

  def __init__(self, address: str, **kwargs) -> None:
    super().__init__(address, check_identity=False)
    if "check_identity" not in kwargs or kwargs["check_identity"]:
      query = self.ask("*IDN?")
      if query.startswith("TEKTRONIX,MDO4"):
        self._name = "-".join(query.split(",")[:2])
      else:
        e = f"{address} did not connect to a Tektronix MDO4000 scope\n"
        e += f"  '*IDN?' returned '{query}'"
        raise ValueError(e)

    # No 2 channel options
    # if self._name[-1] == "2":
    #   self.channels = ["CH1", "CH2"]

    # TODO add other channel operations: RF


class MDO3000(MSO4000):
  """Tektronix Mixed Domain Oscilloscope model MDO3000
  """

  def __init__(self, address: str, **kwargs) -> None:
    super().__init__(address, check_identity=False)
    if "check_identity" not in kwargs or kwargs["check_identity"]:
      query = self.ask("*IDN?")
      if query.startswith("TEKTRONIX,MDO3"):
        self._name = "-".join(query.split(",")[:2])
      else:
        e = f"{address} did not connect to a Tektronix MDO3000 scope\n"
        e += f"  '*IDN?' returned '{query}'"
        raise ValueError(e)

    if self._name[-1] == "2":
      self.channels = ["CH1", "CH2"]

    # TODO add other channel operations: RF
