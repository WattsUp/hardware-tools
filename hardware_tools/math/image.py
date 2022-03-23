"""Collection of image processing functions
"""

import base64
import io

import numpy as np
import PIL.Image


def layer_rgba(below: np.ndarray, above: np.ndarray) -> np.ndarray:
  """Layer a RGBA image on top of another using alpha compositing

  Images are np.arrays [row, column, channel=4]

  Args:
    below: Image on bottom of stack
    above: Image on top of stack

  Returns:
    Combined image

  Raises:
    ValueError if image shapes (resolution) don't match or they are not 4
    channels
  """
  if below.shape != above.shape:
    raise ValueError(
        f"Images must be same shape {below.shape} vs. {above.shape}")
  if below.shape[2] != 4:
    raise ValueError("Image is not RGBA")

  alpha_a = above[:, :, 3]
  alpha_b = below[:, :, 3]
  alpha_out = alpha_a + np.multiply(alpha_b, 1 - alpha_a)
  out = np.zeros(below.shape, dtype=below.dtype)
  out[:, :, 0] = np.divide(
      np.multiply(above[:, :, 0], alpha_a) +
      np.multiply(np.multiply(below[:, :, 0], alpha_b), 1 - alpha_a),
      alpha_out,
      where=alpha_out != 0)
  out[:, :, 1] = np.divide(
      np.multiply(above[:, :, 1], alpha_a) +
      np.multiply(np.multiply(below[:, :, 1], alpha_b), 1 - alpha_a),
      alpha_out,
      where=alpha_out != 0)
  out[:, :, 2] = np.divide(
      np.multiply(above[:, :, 2], alpha_a) +
      np.multiply(np.multiply(below[:, :, 2], alpha_b), 1 - alpha_a),
      alpha_out,
      where=alpha_out != 0)
  out[:, :, 3] = alpha_out
  return out


def np_to_base64(image: np.ndarray) -> bytes:
  """Convert a numpy image to base64 encoded PNG

  Args:
    image: Image to convert, shape=[row, column, channels] from 0.0 to 1.0

  Returns:
    base64 encoded PNG image
  """
  image = PIL.Image.fromarray((255 * image).astype("uint8"))
  with io.BytesIO() as buf:
    image.save(buf, "PNG")
    return base64.b64encode(buf.getvalue())


def np_to_file(image: np.ndarray, path: str):
  """Save a numpy image to file

  Args:
    image: Image to save, shape=[row, column, channels] from 0.0 to 1.0
    path: Path to output file
  """
  image = PIL.Image.fromarray((255 * image).astype("uint8"))
  image.save(path)
