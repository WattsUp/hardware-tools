"""Test module hardware_tools.signal.clock
"""

import numpy as np

from hardware_tools.signal import clock

from tests import base


class TestSignalClock(base.TestBase):
  """Test signal clock methods
  """

  def test_edges(self):
    t_sym = 8e-10
    n = int(1e6)
    ideal_edges = np.linspace(0, t_sym * (n - 1), n)

    t_rj = 0
    t_uj = 0
    t_sj = 0
    f_sj = 0
    dcd = 0
    edges = clock.edges(t_sym=t_sym,
                        t_rj=t_rj,
                        t_uj=t_uj,
                        t_sj=t_sj,
                        f_sj=f_sj,
                        dcd=dcd,
                        n=n)
    self.assertEqual(len(edges), n)
    ties = edges - ideal_edges
    self.assertEqualWithinError(0, ties.mean() / t_sym, 0.01)
    self.assertEqualWithinError(0, ties.std() / t_sym, 0.01)

    ui = 0.1
    t_rj = t_sym * ui
    t_uj = 0
    t_sj = 0
    f_sj = 0
    dcd = 0
    edges = clock.edges(t_sym=t_sym,
                        t_rj=t_rj,
                        t_uj=t_uj,
                        t_sj=t_sj,
                        f_sj=f_sj,
                        dcd=dcd,
                        n=n)
    ties = edges - ideal_edges
    self.assertEqualWithinError(0, ties.mean() / t_sym, 0.01)
    self.assertEqualWithinError(ui, ties.std() / t_sym, 0.01)

    ui = 0.1
    t_rj = 0
    t_uj = t_sym * ui
    t_sj = 0
    f_sj = 0
    dcd = 0
    edges = clock.edges(t_sym=t_sym,
                        t_rj=t_rj,
                        t_uj=t_uj,
                        t_sj=t_sj,
                        f_sj=f_sj,
                        dcd=dcd,
                        n=n)
    ties = edges - ideal_edges
    self.assertEqualWithinError(0, ties.mean() / t_sym, 0.01)
    self.assertEqualWithinError(ui, np.sqrt((ties.std() / t_sym)**2 * 12),
                                0.01)

    ui = 0.1
    mod_freq = 1 / 10000
    t_rj = 0
    t_uj = 0
    t_sj = t_sym * ui
    f_sj = 1 / t_sym * mod_freq
    dcd = 0
    edges = clock.edges(t_sym=t_sym,
                        t_rj=t_rj,
                        t_uj=t_uj,
                        t_sj=t_sj,
                        f_sj=f_sj,
                        dcd=dcd,
                        n=n)
    ties = edges - ideal_edges
    self.assertEqualWithinError(0, ties.mean() / t_sym, 0.01)
    self.assertEqualWithinError(ui, ties.max() / t_sym, 0.01)
    fft = np.abs(np.power(np.fft.rfft(ties), 2))
    freq = fft.argmax() / len(fft) / 2
    self.assertEqualWithinError(mod_freq, freq, 0.01)

    ui = 0.1
    t_rj = 0
    t_uj = 0
    t_sj = 0
    f_sj = 0
    dcd = ui
    edges = clock.edges(t_sym=t_sym,
                        t_rj=t_rj,
                        t_uj=t_uj,
                        t_sj=t_sj,
                        f_sj=f_sj,
                        dcd=dcd,
                        n=n)
    ties = edges - ideal_edges
    self.assertEqualWithinError(ui / 2, ties.mean() / t_sym, 0.01)
    self.assertEqualWithinError(0, ties.min() / t_sym, 0.01)
    self.assertEqualWithinError(ui, ties.max() / t_sym, 0.01)
    self.assertEqualWithinError(ui / 2, ties.std() / t_sym, 0.01)
