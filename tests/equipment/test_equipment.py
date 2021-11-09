"""Test module hardware_tools.equipment.equipment
"""

import unittest

from typing import Any
import random

from hardware_tools.equipment import equipment

from . import mock_pyvisa


class Derrived(equipment.Equipment):

  def configure(self, setting: str, value: Any) -> Any:
    pass

  def command(self,
              command: str,
              timeout: float = 1,
              silent: bool = True) -> None:
    pass


class TestEquipment(unittest.TestCase):
  """Test Equipment
  """

  def setUp(self) -> None:
    super().setUp()

    mock_pyvisa.resources = {}
    mock_pyvisa.available = []
    equipment.pyvisa = mock_pyvisa
  
  def tearDown(self) -> None:
    super().tearDown()
    mock_pyvisa.resources = {}
    mock_pyvisa.available = []
    mock_pyvisa.no_pop = False

  def test_init(self):
    name = "Mock Equipment"
    address = "USB::0x0000::0x0000:C000000::INSTR"
    e = Derrived(address, name=name)
    self.assertIn(address, mock_pyvisa.resources)
    self.assertEqual(e._instrument, mock_pyvisa.resources[address])  # pylint: disable=protected-access

    self.assertEqual(f"{name} @ {address}", repr(e))

    del e
    self.assertNotIn(address, mock_pyvisa.resources)

  def test_send(self):
    name = "Mock Equipment"
    address = "USB::0x0000::0x0000:C000000::INSTR"
    e = Derrived(address, name=name)

    instrument = mock_pyvisa.resources[address]

    command = "*IDN?"
    e.send(command)

    self.assertEqual(1, len(instrument.queue_tx))
    self.assertEqual(command, instrument.queue_tx[0])

  def test_ask(self):
    name = "Mock Equipment"
    address = "USB::0x0000::0x0000:C000000::INSTR"
    e = Derrived(address, name=name)

    instrument = mock_pyvisa.resources[address]
    command = "*IDN?"
    reply = "FAKE:SERIAL_NUMBER"
    instrument.query_map[command] = reply

    self.assertEqual(reply, e.ask(command))
    self.assertEqual(1, len(instrument.queue_tx))
    self.assertEqual(command, instrument.queue_tx[0])

  def test_ask_and_wait(self):
    name = "Mock Equipment"
    address = "USB::0x0000::0x0000:C000000::INSTR"
    e = Derrived(address, name=name)

    count = random.randint(2, 10)

    instrument = mock_pyvisa.resources[address]
    command = "*IDN?"
    reply = "FAKE:SERIAL_NUMBER"
    loading = "FAKE:LOADING"
    instrument.queue_rx.extend([loading] * count)
    instrument.queue_rx.append(reply)

    self.assertEqual(reply, e.ask_and_wait(command, [reply]))
    self.assertListEqual([command] * (count + 1), instrument.queue_tx)

    instrument.queue_tx = []
    instrument.queue_rx = []
    instrument.queue_rx.extend([loading] * count)
    instrument.queue_rx.append(reply)

    self.assertEqual(
        reply, e.ask_and_wait(command, [reply], additional_command="RESET"))
    self.assertListEqual(["RESET", command] * (count + 1), instrument.queue_tx)

    instrument.queue_tx = []
    instrument.queue_rx = []
    instrument.queue_rx.extend([loading] * 10000)

    self.assertRaises(TimeoutError,
                      e.ask_and_wait,
                      command, [reply],
                      additional_command="RESET")

  def test_receive(self):
    name = "Mock Equipment"
    address = "USB::0x0000::0x0000:C000000::INSTR"
    e = Derrived(address, name=name)

    instrument = mock_pyvisa.resources[address]
    reply = b"FAKE:SERIAL_NUMBER"
    instrument.queue_rx.append(reply)

    self.assertEqual(reply, e.receive())
    self.assertEqual(0, len(instrument.queue_tx))
