"""Tests for digitrec.preprocessing."""

import io

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from digitrec.preprocessing import binarize, load_image

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


# --- binarize -----------------------------------------------------------------------------


def _ink_on_paper(size=200, paper=None, ink=40) -> tuple[np.ndarray, np.ndarray]:
    """A fake digit photo: a few thin dark strokes on paper.

    Returns (image, ink_mask). ``paper`` and ``ink`` may each be a number or an HxW array
    (e.g. a shadow gradient).
    """
    img = np.full((size, size), 230, np.float64) if paper is None else np.asarray(paper, float).copy()
    mask = np.zeros((size, size), bool)
    mask[40:160, 95:101] = True    # vertical stroke, 6 px wide
    mask[40:46, 60:140] = True     # horizontal stroke on top: roughly a "7"/"T"
    img[mask] = ink[mask] if np.ndim(ink) else ink
    return img.clip(0, 255).astype(np.uint8), mask


def test_binarize_otsu_separates_ink_and_paper():
    rng = np.random.default_rng(0)
    img, mask = _ink_on_paper()
    noisy = (img + rng.normal(0, 12, img.shape)).clip(0, 255).astype(np.uint8)  # sensor noise

    binary = binarize(noisy)

    assert set(np.unique(binary)) <= {0, 255}
    assert (binary[mask] == 0).mean() > 0.95    # ink is black…
    assert (binary[~mask] == 255).mean() > 0.99  # …paper white, noise specks removed by the blur
    assert binary.shape == img.shape


def test_binarize_keeps_polarity_for_light_on_dark():
    img, mask = _ink_on_paper(paper=np.zeros((200, 200)), ink=255)  # white strokes on black
    binary = binarize(img)
    assert (binary[mask] == 255).mean() > 0.95
    assert (binary[~mask] == 0).mean() > 0.99


def test_adaptive_handles_shadow_where_otsu_fails():
    """Paper fading from bright (left) to a deep shadow (right), ink drawn across both."""
    shadow = np.tile(np.linspace(250, 70, 200), (200, 1))
    img, mask = _ink_on_paper(paper=shadow, ink=shadow * 0.45)  # ink is ~55% darker than local paper

    otsu = binarize(img, method="otsu")
    adaptive = binarize(img, method="adaptive")

    paper_white_otsu = (otsu[~mask] == 255).mean()
    paper_white_adaptive = (adaptive[~mask] == 255).mean()
    assert paper_white_otsu < 0.8          # global threshold blackens the shadowed side
    assert paper_white_adaptive > 0.97     # local threshold keeps all of the paper white
    assert (adaptive[mask] == 0).mean() > 0.9  # and still finds the ink


def test_binarize_without_blur():
    img, _ = _ink_on_paper()
    assert set(np.unique(binarize(img, blur_ksize=0))) == {0, 255}


def test_binarize_rejects_bad_input():
    with pytest.raises(ValueError, match="2-D uint8"):
        binarize(np.zeros((5, 5, 3), np.uint8))
    with pytest.raises(ValueError, match="unknown threshold method"):
        binarize(np.zeros((5, 5), np.uint8), method="magic")  # type: ignore[arg-type]
