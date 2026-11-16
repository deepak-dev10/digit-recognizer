"""Tests for digitrec.preprocessing."""

import io

import cv2
import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from digitrec.preprocessing import (
    NoInkFoundError,
    background_is_light,
    binarize,
    center_by_mass,
    crop_to_ink,
    ensure_white_on_black,
    estimate_stroke_width,
    fit_to_box,
    load_image,
    thicken_strokes,
)

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


# --- background_is_light / ensure_white_on_black ------------------------------------------


def _binary_digit(dark_on_light=True, size=50) -> np.ndarray:
    """Binary image with a vertical bar 'digit' in the middle."""
    paper, ink = (255, 0) if dark_on_light else (0, 255)
    img = np.full((size, size), paper, np.uint8)
    img[10:40, 22:28] = ink
    return img


def test_dark_on_light_is_inverted():
    img = _binary_digit(dark_on_light=True)
    out = ensure_white_on_black(img)
    assert background_is_light(img)
    assert out[0, 0] == 0 and out[25, 25] == 255  # black background, white digit


def test_white_on_black_is_unchanged():
    img = _binary_digit(dark_on_light=False)
    out = ensure_white_on_black(img)
    assert not background_is_light(img)
    np.testing.assert_array_equal(out, img)
    assert out is not img  # always a new array


def test_input_is_not_modified():
    img = _binary_digit(dark_on_light=True)
    before = img.copy()
    ensure_white_on_black(img)
    np.testing.assert_array_equal(img, before)


def test_stroke_touching_border_does_not_flip_decision():
    """A binary stroke running off the top and bottom edges (as on a tightly cropped canvas)."""
    img = np.full((50, 50), 255, np.uint8)
    img[:, 20:30] = 0  # ink touches the top and bottom border
    assert background_is_light(img)
    assert ensure_white_on_black(img)[0, 0] == 0


def test_median_handles_dim_paper_with_ink_on_border():
    """Grayscale case where a mean-based check would get it wrong.

    Dim paper (150) with black ink along ~1/3 of the border: the border mean drops to
    ~100 (below 127, so it would look "dark"), but most border pixels are still paper.
    """
    img = np.full((60, 60), 150, np.uint8)
    img[:, :20] = 0  # thick dark stroke along the whole left third
    t = 3            # border thickness used for a 60 px image (5%)
    border = np.concatenate([img[:t].ravel(), img[-t:].ravel(), img[:, :t].ravel(), img[:, -t:].ravel()])
    assert border.mean() < 127  # mean-based check would misjudge…
    assert background_is_light(img)  # …the median-based one gets it right


def test_works_on_grayscale():
    img = np.full((30, 30), 200, np.uint8)  # light-grey paper
    img[10:20, 10:20] = 50                  # dark-grey ink
    out = ensure_white_on_black(img)
    assert out[0, 0] == 55 and out[15, 15] == 205


def test_tiny_image_uses_at_least_one_pixel_border():
    assert background_is_light(np.full((2, 2), 255, np.uint8))
    assert not background_is_light(np.zeros((2, 2), np.uint8))


def test_background_is_light_rejects_color():
    with pytest.raises(ValueError, match="2-D"):
        background_is_light(np.zeros((5, 5, 3), np.uint8))


# --- crop_to_ink / fit_to_box -------------------------------------------------------------


def test_crop_to_ink_returns_tight_box():
    img = np.zeros((50, 40), np.uint8)
    img[10:30, 5:15] = 255   # the digit
    img[12, 16] = 80         # an anti-aliased grey edge pixel still counts as ink
    cropped = crop_to_ink(img)
    assert cropped.shape == (20, 12)  # rows 10..29, cols 5..16
    assert cropped[0, 0] == 255


def test_crop_to_ink_threshold_ignores_faint_pixels():
    img = np.zeros((50, 40), np.uint8)
    img[10:30, 5:15] = 255
    img[45, 35] = 20  # faint speck in the corner
    assert crop_to_ink(img).shape == (36, 31)  # speck stretches the box to rows 10..45, cols 5..35…
    assert crop_to_ink(img, threshold=50).shape == (20, 10)  # …unless it's below the threshold


def test_crop_to_ink_blank_image_raises():
    with pytest.raises(NoInkFoundError, match="no digit found"):
        crop_to_ink(np.zeros((28, 28), np.uint8))
    assert issubclass(NoInkFoundError, ValueError)


def test_crop_returns_copy():
    img = np.zeros((10, 10), np.uint8)
    img[2:5, 2:5] = 255
    crop_to_ink(img)[0, 0] = 0
    assert img[2, 2] == 255


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((100, 50), (20, 10)),   # tall digit: height becomes 20
        ((30, 90), (7, 20)),     # wide digit: width becomes 20, 30 * 20/90 = 6.67 rounds to 7
        ((5, 5), (20, 20)),      # small crop is enlarged
        ((200, 2), (20, 1)),     # very thin "1": short side never drops to 0
    ],
)
def test_fit_to_box_keeps_aspect_ratio(shape, expected):
    out = fit_to_box(np.full(shape, 255, np.uint8))
    assert out.shape == expected
    assert out.dtype == np.uint8


def test_fit_to_box_shrinking_creates_soft_edges():
    """A 1-px-off-grid stroke shrunk with INTER_AREA gets grey (anti-aliased) edge pixels."""
    img = np.zeros((100, 100), np.uint8)
    img[:, 45:58] = 255  # 13 px stroke; 13 * 0.2 = 2.6 px after shrinking, so edges are partial
    out = fit_to_box(img)
    assert ((out > 0) & (out < 255)).any()


def test_fit_to_box_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        fit_to_box(np.zeros((0, 5), np.uint8))


# --- center_by_mass -----------------------------------------------------------------------


def _center_of_mass(img: np.ndarray) -> tuple[float, float]:
    w = img.astype(float)
    ys, xs = np.indices(img.shape)
    return (w * ys).sum() / w.sum(), (w * xs).sum() / w.sum()


def test_center_by_mass_output_is_28x28_uint8():
    out = center_by_mass(np.full((20, 10), 255, np.uint8))
    assert out.shape == (28, 28)
    assert out.dtype == np.uint8


def test_symmetric_digit_is_centered():
    out = center_by_mass(np.full((20, 10), 255, np.uint8))
    cy, cx = _center_of_mass(out)
    assert abs(cy - 13.5) <= 0.5 and abs(cx - 13.5) <= 0.5


def test_lopsided_digit_centered_by_mass_not_box():
    """A '7'-like shape: a heavy top bar plus a thin stem. Its mass sits high in its box."""
    seven = np.zeros((20, 14), np.uint8)
    seven[:5, :] = 255       # thick top bar
    seven[5:, 6:8] = 255     # thin stem
    out = center_by_mass(seven)

    cy, cx = _center_of_mass(out)
    assert abs(cy - 13.5) <= 0.5 and abs(cx - 13.5) <= 0.5  # mass is centred…
    rows_with_ink = np.flatnonzero(out.any(axis=1))
    assert rows_with_ink[0] > 4  # …so the digit sits lower than box-centring (top row 4) would put it


def test_shift_is_clamped_so_no_ink_is_lost():
    """Nearly all mass in one corner: the ideal shift would push the digit off the frame."""
    digit = np.zeros((20, 20), np.uint8)
    digit[0, 0] = 255   # a bright dot in the top-left…
    digit[19, 19] = 1   # …and a barely visible pixel bottom-right that defines the box
    out = center_by_mass(digit)
    assert out.sum() == digit.astype(int).sum()  # every pixel survived
    assert out[0, 0] == 0 and out.max() == 255   # moved, but stayed inside


def test_pixels_are_copied_unchanged():
    rng = np.random.default_rng(1)
    digit = rng.integers(0, 256, (20, 15), dtype=np.uint8)
    out = center_by_mass(digit)
    assert sorted(out[out > 0].tolist()) == sorted(digit[digit > 0].tolist())  # no interpolation


def test_center_by_mass_rejects_bad_input():
    with pytest.raises(NoInkFoundError):
        center_by_mass(np.zeros((20, 20), np.uint8))
    with pytest.raises(ValueError, match="does not fit"):
        center_by_mass(np.full((30, 10), 255, np.uint8))
    with pytest.raises(ValueError, match="2-D"):
        center_by_mass(np.zeros((5, 5, 3), np.uint8))


# --- estimate_stroke_width / thicken_strokes ----------------------------------------------


@pytest.mark.parametrize("width", [3, 6, 12])
def test_estimate_stroke_width_on_straight_stroke(width):
    img = np.zeros((220, 60), np.uint8)
    img[10:210, 20 : 20 + width] = 255  # 200 px long vertical stroke
    assert estimate_stroke_width(img) == pytest.approx(width, rel=0.1)


def test_estimate_stroke_width_on_ring():
    """A '0': the hole's outline counts too, so a ring of width 8 measures ~8."""
    img = np.zeros((200, 200), np.uint8)
    cv2.circle(img, (100, 100), 80, 255, thickness=8)
    assert estimate_stroke_width(img) == pytest.approx(8, rel=0.15)


def test_estimate_stroke_width_blank_is_zero():
    assert estimate_stroke_width(np.zeros((10, 10), np.uint8)) == 0.0


def _thin_seven(stroke=4, size=400) -> np.ndarray:
    """A fine-pen '7' on a big crop: stroke/size = 0.01, far thinner than MNIST."""
    img = np.zeros((size, size), np.uint8)
    cv2.line(img, (0, 0), (size - 1, 0), 255, stroke)
    cv2.line(img, (size - 1, 0), (size // 3, size - 1), 255, stroke)
    return crop_to_ink(img)


def test_thin_strokes_are_thickened_to_mnist_like_ratio():
    crop = _thin_seven()
    out = thicken_strokes(crop)
    ratio = estimate_stroke_width(out) / max(out.shape)
    assert estimate_stroke_width(crop) / max(crop.shape) < 0.02  # started very thin
    assert 0.08 <= ratio <= 0.14                                  # now within MNIST's typical range


def test_thickening_keeps_strokes_after_resize():
    """The point of the step: after fit_to_box, a thickened stroke still has solid pixels."""
    crop = _thin_seven()
    assert fit_to_box(crop).max() < 128           # without: only a faint grey ghost remains
    assert fit_to_box(thicken_strokes(crop)).max() == 255


def test_thick_strokes_are_left_alone():
    img = np.zeros((100, 100), np.uint8)
    img[10:90, 40:55] = 255  # width 15 on an 80 px digit: ratio ~0.19, thicker than MNIST median
    crop = crop_to_ink(img)
    out = thicken_strokes(crop)
    np.testing.assert_array_equal(out, crop)
    assert out is not crop


def test_thickening_pads_so_edge_strokes_are_not_clipped():
    """The 7's top bar lies on the crop's top edge; dilation must grow it upwards, not cut it."""
    crop = _thin_seven()
    out = thicken_strokes(crop)
    pad = (out.shape[0] - crop.shape[0]) // 2
    h, w = crop.shape
    assert pad > 0
    assert (out[pad : pad + h, pad : pad + w][crop > 0] > 0).all()  # every original ink pixel kept
    assert out[:pad].any()  # ink grew into the padding above the original top edge


def test_thicken_blank_returns_copy():
    blank = np.zeros((10, 10), np.uint8)
    assert (thicken_strokes(blank) == 0).all()
