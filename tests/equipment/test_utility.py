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
        "TEKTRONIX,MSO4032": tektronix.MSO4000,
        "TEKTRONIX,MSO4034": tektronix.MSO4000,
        "TEKTRONIX,MSO4054": tektronix.MSO4000,
        "TEKTRONIX,MSO4104": tektronix.MSO4000,
        "TEKTRONIX,MSO4032B": tektronix.MSO4000,
        "TEKTRONIX,MSO4034B": tektronix.MSO4000,
        "TEKTRONIX,MSO4054B": tektronix.MSO4000,
        "TEKTRONIX,MSO4104B": tektronix.MSO4000,
        "TEKTRONIX,MDO4054": tektronix.MDO4000,
        "TEKTRONIX,MDO4104": tektronix.MDO4000,
        "TEKTRONIX,MDO3012": tektronix.MDO3000,
        "TEKTRONIX,MDO3014": tektronix.MDO3000,
        "TEKTRONIX,MDO3022": tektronix.MDO3000,
        "TEKTRONIX,MDO3024": tektronix.MDO3000,
        "TEKTRONIX,MDO3032": tektronix.MDO3000,
        "TEKTRONIX,MDO3034": tektronix.MDO3000,
        "TEKTRONIX,MDO3052": tektronix.MDO3000,
        "TEKTRONIX,MDO3054": tektronix.MDO3000,
        "TEKTRONIX,MDO3102": tektronix.MDO3000,
        "TEKTRONIX,MDO3104": tektronix.MDO3000,
    }

    mock_pyvisa.no_pop = True

    instrument = mock_pyvisa.Resource(address)
    instrument.query_map["*IDN?"] = "FAKE"
    self.assertRaises(LookupError, utility.connect, address)
    instrument.close()

    for name, class_type in equipment_types.items():
      instrument = mock_pyvisa.Resource(address)
      instrument.query_map["*IDN?"] = name

      e = utility.connect(address)
      self.assertIsInstance(e, class_type)

      instrument.close()
