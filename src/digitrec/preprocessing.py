"""Stages 1–2 of the pipeline: load an image and make it look like an MNIST sample.

MNIST models expect a very specific input: a 28x28 grayscale image with a white digit on a
black background, scaled to fit a 20x20 box and centred by centre of mass. Real photos of
handwriting look nothing like that (dark ink on white paper, shadows, off-centre, any size).
Most "the model got it wrong" failures come from that mismatch, not from the model, which
is why this stage is kept separate and shared by every backend.

This module is built in small steps. So far:

* :func:`load_image`: any supported input, turned into a 2-D ``uint8`` grayscale array.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

#: Everything :func:`load_image` accepts:
#:
#: * ``str`` / ``Path``: a file on disk (PNG, JPEG, …: anything Pillow can open).
#: * ``bytes``: the encoded file contents (e.g. ``st.file_uploader(...).getvalue()``).
#: * a binary file-like object (e.g. Streamlit's ``UploadedFile``).
#: * ``PIL.Image.Image``: an already opened image.
#: * ``np.ndarray``: raw ``uint8`` pixels, shaped ``HxW`` (gray), ``HxWx3`` (RGB) or ``HxWx4``
#:   (RGBA, e.g. the Streamlit drawing canvas's ``image_data``).
type ImageSource = str | Path | bytes | BinaryIO | Image.Image | np.ndarray

# Colour transparent pixels become. White matches paper, so a transparent background
# is treated like a blank page.
_BACKGROUND_RGB = (255, 255, 255)


def _open(source: ImageSource) -> Image.Image:
    """Turn any :data:`ImageSource` into a PIL image (no colour conversion yet)."""
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8:
            raise TypeError(f"array images must be uint8 (0–255), got dtype {source.dtype}")
        if source.ndim == 2 or (source.ndim == 3 and source.shape[2] in (3, 4)):
            return Image.fromarray(source)  # mode inferred: L, RGB or RGBA
        raise ValueError(f"array must be HxW, HxWx3 or HxWx4, got shape {source.shape}")
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    # Path, str or file-like object. Image.open is lazy, so .load() reads the pixels
    # now and surfaces a bad file here, not later in the pipeline.
    image = Image.open(source)
    image.load()
    return image


def _has_transparency(image: Image.Image) -> bool:
    return image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info


def load_image(source: ImageSource) -> np.ndarray:
    """Load an image and return it as a grayscale ``uint8`` array (stage 1).

    Steps:

    1. Open the source (see :data:`ImageSource` for accepted types).
    2. Apply the EXIF orientation tag. Phone cameras often store photos sideways and
       only *tag* the rotation. Without this a "7" could reach the model on its side.
    3. Composite any transparency onto a white background.
    4. Convert to grayscale with Pillow's luma formula ``L = 0.299 R + 0.587 G + 0.114 B``.

    Args:
        source: The image to load.

    Returns:
        A ``HxW`` ``np.uint8`` array, 0 = black and 255 = white. Polarity is left
        as-is: ink is usually dark here. Inverting to MNIST's white-on-black happens
        in a later preprocessing step.

    Raises:
        TypeError / ValueError: Unsupported array dtype or shape.
        PIL.UnidentifiedImageError: The file or bytes are not a readable image.
    """
    image = _open(source)
    image = ImageOps.exif_transpose(image)  # returns a rotated copy; no-op without EXIF

    if _has_transparency(image):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*_BACKGROUND_RGB, 255))
        image = Image.alpha_composite(background, rgba)

    return np.asarray(image.convert("L"), dtype=np.uint8)
