"""Test module hardware_tools.equipment.utility
"""

import pyvisa

from hardware_tools.equipment import utility
from hardware_tools.equipment import tektronix

from tests import base
from tests.equipment import mock_pyvisa


class TestEquipmentUtility(base.TestBase):
  """Test Equipment Utility
  """

  def setUp(self) -> None:
    super().setUp()

    mock_pyvisa.resources = {}
    mock_pyvisa.available = []

  def tearDown(self) -> None:
    super().tearDown()
    mock_pyvisa.resources = {}
    mock_pyvisa.available = []
    mock_pyvisa.no_pop = False

  def test_get_available(self):
    mock_pyvisa.available = [
        "USB::0x0000::0x0000:C000000::INSTR", "TCPIP::127.0.0.1::INSTR"
    ]

    rm = mock_pyvisa.ResourceManager()
    available = utility.get_available(rm=rm)
    self.assertListEqual(mock_pyvisa.available, available)

    try:
      utility.pyvisa = mock_pyvisa
      available = utility.get_available()
      self.assertListEqual(mock_pyvisa.available, available)
    finally:
      utility.pyvisa = pyvisa

  def test_connect(self):
    address = "USB::0x0000::0x0000:C000000::INSTR"

    equipment_types = {
        "TEKTRONIX,MSO4032,serial number": tektronix.MSO4000Family,
        "TEKTRONIX,MDO4054,serial number": tektronix.MSO4000Family,
        "TEKTRONIX,DPO4104B,serial number": tektronix.MSO4000Family,
        "TEKTRONIX,MDO3104,serial number": tektronix.MSO4000Family,
        "TEKTRONIX,MSO64B,serial number": tektronix.MSO456Family,
        "TEKTRONIX,LPD64,serial number": tektronix.MSO456Family,
    }

    mock_pyvisa.no_pop = True

    rm = mock_pyvisa.ResourceManager()
    instrument = mock_pyvisa.Resource(rm, address)
    instrument.query_map["*IDN"] = "FAKE"
    self.assertRaises(LookupError, utility.connect, address, rm=rm)

    try:
      utility.pyvisa = mock_pyvisa
      self.assertRaises(LookupError, utility.connect, address)
    finally:
      utility.pyvisa = pyvisa

    instrument.close()

    for name, class_type in equipment_types.items():
      instrument = mock_pyvisa.Resource(rm, address)
      instrument.query_map["*IDN"] = name
      if class_type == tektronix.MSO4000Family:
        instrument.query_map["CONFIGURATION"] = {
            "ANALOG": {
                "NUMCHANNELS": "1",
                "BANDWIDTH": "1.0000E+9"
            },
            "DIGITAL": {
                "NUMCHANNELS": "1"
            },
            "AUXIN": "0"
        }
        instrument.query_map["HEADER"] = (lambda _: None, "0")
        instrument.query_map["VERBOSE"] = (lambda _: None, "1")
      elif class_type == tektronix.MSO456Family:
        instrument.query_map["CONFIGURATION"] = {
            "ANALOG": {
                "BANDWIDTH": "1.0000E+9"
            }
        }
        instrument.query_map["DISPLAY"] = ":DISPLAY:GLOBAL:CH1:STATE ON"
        instrument.query_map["HEADER"] = (lambda _: None, "0")
        instrument.query_map["VERBOSE"] = (lambda _: None, "1")

      e = utility.connect(address, rm=rm)
      self.assertIsInstance(e, class_type)

      instrument.close()

  def test_parse_scpi(self):
    with open(self._DATA_ROOT.joinpath("scpi.txt"), "r",
              encoding="utf-8") as file:
      raw = file.read()

    d = utility.parse_scpi(raw, flat=False, types=tektronix.TEK_TYPES)
    self.assertIsInstance(d, dict)
    self.assertIsInstance(d["TRIGGER"], dict)

    d = utility.parse_scpi(raw, flat=True, types=tektronix.TEK_TYPES)
    self.assertIsInstance(d, dict)
    for v in d.values():
      self.assertNotIsInstance(v, dict)
    
    raw = "INCOMPLETE"
    self.assertRaises(ValueError, utility.parse_scpi, raw)
