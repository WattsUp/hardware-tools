"""Test module hardware_tools.equipment.utility
"""

from hardware_tools.equipment import equipment, utility
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
    equipment.pyvisa = mock_pyvisa
    utility.pyvisa = mock_pyvisa

  def tearDown(self) -> None:
    super().tearDown()
    mock_pyvisa.resources = {}
    mock_pyvisa.available = []
    mock_pyvisa.no_pop = False

  def test_get_available(self):
    mock_pyvisa.available = [
        "USB::0x0000::0x0000:C000000::INSTR", "TCPIP::127.0.0.1::INSTR"
    ]

    available = utility.get_available()
    self.assertListEqual(mock_pyvisa.available, available)

  def test_connect(self):
    address = "USB::0x0000::0x0000:C000000::INSTR"

    equipment_types = {
        "TEKTRONIX,MSO4032": tektronix.MSO4000Family,
        "TEKTRONIX,MDO4054": tektronix.MSO4000Family,
        "TEKTRONIX,DPO4104B": tektronix.MSO4000Family,
        "TEKTRONIX,MDO3104": tektronix.MSO4000Family,
    }

    mock_pyvisa.no_pop = True

    rm = mock_pyvisa.ResourceManager()
    instrument = mock_pyvisa.Resource(rm, address)
    instrument.query_map["*IDN?"] = "FAKE"
    self.assertRaises(LookupError, utility.connect, address)
    instrument.close()

    for name, class_type in equipment_types.items():
      instrument = mock_pyvisa.Resource(rm, address)
      instrument.query_map["*IDN?"] = name
      if class_type == tektronix.MSO4000Family:
        instrument.query_map["SELECT?"] = (":SELECT:CH1 1;MATH 0;REF1 0;D0 0;"
                                           "BUS1 0;CONTROL CH1")

      e = utility.connect(address)
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
