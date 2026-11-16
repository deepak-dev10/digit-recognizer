"""Tests for digitrec.preprocessing."""

import io

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from digitrec.preprocessing import load_image

# --- load_image ---------------------------------------------------------------------------

# Pillow's EXIF tag id for orientation; value 6 means "rotate 90° clockwise to display".
_EXIF_ORIENTATION = 0x0112


def _rgb_image(width=20, height=10, color=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def _encode(image: Image.Image, fmt="PNG", **save_kwargs) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def test_returns_2d_uint8_grayscale():
    gray = load_image(_rgb_image(color=(255, 0, 0)))
    assert gray.shape == (10, 20)  # (height, width)
    assert gray.dtype == np.uint8
    assert gray[0, 0] == 76  # luma of pure red: 0.299 * 255 ≈ 76


@pytest.mark.parametrize("as_source", ["path", "str", "bytes", "filelike", "pil"])
def test_accepts_all_source_types(tmp_path, as_source):
    image = _rgb_image(color=(0, 0, 0))
    path = tmp_path / "digit.png"
    image.save(path)
    source = {
        "path": path,
        "str": str(path),
        "bytes": path.read_bytes(),
        "filelike": io.BytesIO(path.read_bytes()),
        "pil": image,
    }[as_source]
    gray = load_image(source)
    assert gray.shape == (10, 20)
    assert (gray == 0).all()


def test_accepts_gray_rgb_and_rgba_arrays():
    assert load_image(np.full((5, 7), 128, np.uint8)).shape == (5, 7)
    assert load_image(np.zeros((5, 7, 3), np.uint8)).shape == (5, 7)
    assert load_image(np.zeros((5, 7, 4), np.uint8)).shape == (5, 7)


def test_rejects_bad_arrays():
    with pytest.raises(TypeError, match="uint8"):
        load_image(np.zeros((5, 7), np.float32))
    with pytest.raises(ValueError, match="shape"):
        load_image(np.zeros((5, 7, 2), np.uint8))


def test_exif_orientation_is_applied():
    """A 20x10 photo tagged 'rotate 90°' must come out 10 wide and 20 tall."""
    exif = Image.Exif()
    exif[_EXIF_ORIENTATION] = 6
    data = _encode(_rgb_image(width=20, height=10), fmt="JPEG", exif=exif)
    assert load_image(data).shape == (20, 10)


def test_transparent_pixels_become_white():
    """Like the drawing canvas: black stroke on a fully transparent background."""
    rgba = np.zeros((4, 4, 4), np.uint8)    # transparent black everywhere…
    rgba[1, 1] = (0, 0, 0, 255)            # …except one opaque black "ink" pixel
    gray = load_image(rgba)
    assert gray[1, 1] == 0                 # ink stays black
    assert gray[0, 0] == 255               # transparent background became white paper


def test_unreadable_bytes_raise():
    with pytest.raises(UnidentifiedImageError):
        load_image(b"not an image")
