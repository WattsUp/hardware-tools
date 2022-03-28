"""Test module hardware_tools.image
"""

import base64
import io

import numpy as np
import PIL
import PIL.Image

from hardware_tools.math import image

from tests import base


class TestImage(base.TestBase):
  """Test image methods
  """

  def test_image_layer(self):
    shape = self._RNG.integers(10, 100, 3)

    above = np.zeros(shape=shape, dtype=np.float32)
    below = np.zeros(shape=shape[:2], dtype=np.float32)

    self.assertNotEqual(above.shape, below.shape)
    self.assertRaises(ValueError, image.layer_rgba, below, above)

    above = np.zeros(shape=shape, dtype=np.float32)
    below = np.zeros(shape=shape, dtype=np.float32)

    self.assertEqual(above.shape, below.shape)
    self.assertRaises(ValueError, image.layer_rgba, below, above)

    shape[2] = 4
    above = np.zeros(shape=shape, dtype=np.float32)
    below = np.zeros(shape=shape, dtype=np.float32)

    above_rgba = self._RNG.uniform(0, 1, 4)
    above[:, :] = above_rgba
    below_rgba = self._RNG.uniform(0, 1, 4)
    below[:, :] = below_rgba

    out = image.layer_rgba(below, above)
    out_rgba = out[0][0]
    out_a = above_rgba[3] + below_rgba[3] * (1 - above_rgba[3])
    self.assertAlmostEqual(out_a, out_rgba[3], 3)

    out_r = (above_rgba[0] * above_rgba[3] + (below_rgba[0] * below_rgba[3]) *
             (1 - above_rgba[3])) / out_a
    self.assertAlmostEqual(out_r, out_rgba[0], 3)

    out_g = (above_rgba[1] * above_rgba[3] + (below_rgba[1] * below_rgba[3]) *
             (1 - above_rgba[3])) / out_a
    self.assertAlmostEqual(out_g, out_rgba[1], 3)

    out_b = (above_rgba[2] * above_rgba[3] + (below_rgba[2] * below_rgba[3]) *
             (1 - above_rgba[3])) / out_a
    self.assertAlmostEqual(out_b, out_rgba[2], 3)

  def test_image_base64(self):
    shape = self._RNG.integers(10, 100, 3)
    shape[2] = 4

    img = self._RNG.uniform(0.0, 1.0, size=shape)

    out = image.np_to_base64(img)
    with io.BytesIO(base64.b64decode(out)) as buf:
      img_pil = PIL.Image.open(buf, formats=["PNG"])
      img_pil.load()

    out = b"0123456789ABCDEF" + out
    with io.BytesIO(base64.b64decode(out)) as buf:
      self.assertRaises(PIL.UnidentifiedImageError,
                        PIL.Image.open,
                        buf,
                        formats=["PNG"])

  def test_image_file(self):
    shape = self._RNG.integers(10, 100, 3)
    shape[2] = 4

    img = self._RNG.uniform(0.0, 1.0, size=shape)

    path = str(self._TEST_ROOT.joinpath("image.png"))

    image.np_to_file(img, path)

    img_pil = PIL.Image.open(path, formats=["PNG"])
    img_pil.load()
