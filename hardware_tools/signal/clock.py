"""Clock generation
"""

import numpy as np

_rng = np.random.default_rng()


def edges(t_sym: float = 8e-10,
          t_rj: float = 0,
          t_uj: float = 0,
          t_sj: float = 0,
          f_sj: float = 0,
          dcd: float = 0.0,
          n: int = 1e6) -> np.ndarray:
  """Generate edges of a clock

  Args:
    t_sym: Duration of a symbol
    t_rj: Amplitude of random jitter, gaussian standard deviation
    t_uj: Amplitude of uniform jitter, uniform distribution width
    t_sj: Amplitude of sinusoidal jitter
    f_sj: Frequency of sinusoidal jitter
    dcd: Duty-cycle-distortion [-1, 1]
    n: Number of edges to generate

  Returns:
    numpy array of edges times
  """
  n = int(n)
  e = np.linspace(0, t_sym * (n - 1), n)
  e += np.sin(2 * np.pi * f_sj * e + _rng.uniform(0, 2 * np.pi)) * t_sj
  e[::2] += dcd * t_sym
  e += _rng.normal(0, t_rj, n)
  e += _rng.uniform(-t_uj / 2, t_uj / 2, n)
  return e
