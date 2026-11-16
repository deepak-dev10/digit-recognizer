"""Stages 1–2 of the pipeline: load an image and make it look like an MNIST sample.

MNIST models expect a very specific input: a 28x28 grayscale image with a white digit on a
black background, scaled to fit a 20x20 box and centred by centre of mass. Real photos of
handwriting look nothing like that (dark ink on white paper, shadows, off-centre, any size).
Most "the model got it wrong" failures come from that mismatch, not from the model, which
is why this stage is kept separate and shared by every backend.

This module is built in small steps. So far:

* :func:`load_image`: any supported input, turned into a 2-D ``uint8`` grayscale array.
* :func:`binarize`: denoise and threshold to pure black/white (0 or 255).
* :func:`ensure_white_on_black`: invert if needed so the digit is white on black, like MNIST.
* :func:`crop_to_ink` + :func:`fit_to_box`: cut away empty margins, scale the digit to 20 px.
* :func:`thicken_strokes`: optional dilation of thin pen strokes, applied to the crop
  before :func:`fit_to_box` (see its docstring for why the order matters).
* :func:`center_by_mass`: place it in a 28x28 frame with its centre of mass in the middle.
* :func:`to_tensor`: turn the 28x28 image into the ``[1, 28, 28]`` float tensor models take.

The steps are chained in two halves, so a segmenter can sit in between::

    prepare_page(source)      load → binarize → invert           once per image
          │
          ▼  (segmenter: whole page for one digit, one region per digit later)
    normalize_digit(region)   crop → thicken → fit 20 px → centre 28x28 → tensor

:func:`preprocess` runs both halves for the single-digit case. Every step's image is kept
in ``stages`` so the UI and ``--show-debug`` can show exactly what the model saw.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

import cv2
import numpy as np
import torch
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


class NoInkFoundError(ValueError):
    """The image contains no digit pixels (blank page, empty canvas, or everything thresholded away).

    Subclasses ``ValueError`` so generic handlers still catch it, while the CLI and UI can
    catch it specifically and show a friendly "no digit found" message.
    """


#: Thresholding strategy for :func:`binarize`.
ThresholdMethod = Literal["otsu", "adaptive"]

# Contrast = gap between the mean brightness of Otsu's light and dark groups, measured
# after blurring AND after removing slow lighting changes (see _ink_contrast).
# Measured on synthetic 400x400 pages with sensor noise (σ 6 unless stated):
#   blank pages:  noise σ 0–25 → 0–9.5, vignette → 5.8, gradients 245→190 / 245→120 → 3.7 / 5.6,
#                 deep shadow 245→90 → 9.0
#   inked pages:  marker / fine pen → 128 / 100, pencil 4 px → 56, faint pencil 2 px → 32,
#                 very faint 2 px (ink only 30 levels darker than paper) → 18.5,
#                 pen 4 px across a deep shadow → 73
# Without the lighting removal, a blank page with a gradient measured 28–78 and overlapped faint pencil.
#: Below this contrast the image is treated as blank (~5 levels of margin on each side of the data above).
MIN_CONTRAST = 15


def _ink_contrast(blurred: np.ndarray) -> float:
    """How strongly the darkest/brightest marks stand out from their surroundings (0–255).

    1. Estimate the **background lighting** (paper brightness, shadows, vignetting) by
       heavily blurring the image, then subtract it. Thin strokes survive the subtraction;
       slow brightness changes don't. Without this, a blank page with a shadow across it
       would look like it has strong "ink".
    2. Split the flattened image with Otsu and return the gap between the two groups' means.

    The background is estimated on a small copy (shorter side 64 px) and scaled back up.
    Lighting is smooth, so nothing is lost, and big photos don't need a huge blur kernel.
    """
    h, w = blurred.shape
    scale = 64 / min(h, w)
    if scale < 1:
        small = cv2.resize(blurred, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = blurred
    small_bg = cv2.GaussianBlur(small.astype(np.float32), (0, 0), sigmaX=min(small.shape) / 8)
    background = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)

    flat = np.clip(blurred.astype(np.float32) - background + 128, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark, light = flat[flat <= t], flat[flat > t]
    return float(light.mean() - dark.mean()) if dark.size and light.size else 0.0


def _odd_at_least_3(n: int) -> int:
    """Round ``n`` up to the nearest odd number >= 3 (OpenCV kernel/block sizes must be odd)."""
    n = max(n, 3)
    return n if n % 2 else n + 1


def binarize(
    gray: np.ndarray,
    *,
    method: ThresholdMethod = "otsu",
    blur_ksize: int = 5,
    adaptive_c: int = 10,
    min_contrast: float = MIN_CONTRAST,
) -> np.ndarray:
    """Denoise a grayscale image and threshold it to pure black and white (stage 2a).

    1. **Gaussian blur** smooths sensor noise and paper texture, so stray specks don't
       survive thresholding as fake "ink".
    2. **Blank check.** Otsu *always* splits the histogram in two, even when there is
       nothing to split: on an empty page it would turn sensor noise into fake ink. So we
       first measure the **contrast**: how strongly the marks stand out once slow lighting
       changes (shadows, vignetting) are removed, see :func:`_ink_contrast`. Below
       ``min_contrast`` the image is treated as blank and :class:`NoInkFoundError` is
       raised. This applies to both methods.
    3. **Threshold** sets every pixel to 0 or 255:

       * ``"otsu"`` (default): one global threshold, chosen automatically to best separate
         the two brightness groups (ink and paper) in the histogram. Works well for evenly
         lit photos, scans and the drawing canvas.
       * ``"adaptive"``: a separate threshold per neighbourhood (Gaussian-weighted local mean
         minus ``adaptive_c``). Use it for photos with **shadows or uneven lighting**,
         where one global threshold would turn the darker part of the paper black.

    Polarity is **kept**: pixels brighter than the threshold become 255, darker ones 0, so
    dark ink on paper stays 0-on-255. Inverting to MNIST's white-on-black is a separate step.

    Args:
        gray: ``HxW`` ``uint8`` grayscale image, e.g. from :func:`load_image`.
        method: ``"otsu"`` or ``"adaptive"``, see above.
        blur_ksize: Gaussian kernel size in pixels (rounded up to an odd number);
            ``0`` disables blurring, e.g. for tiny or already clean images.
        adaptive_c: Only for ``"adaptive"``. How much darker than its surroundings a
            pixel must be to count as ink. Higher values ignore faint shading but can
            break up light pen strokes.
        min_contrast: Minimum contrast (0–255 brightness levels) to accept the image as
            containing ink; ``0`` disables the check. The default comes from the
            measurements next to :data:`MIN_CONTRAST`.

    Returns:
        ``HxW`` ``uint8`` array containing only 0 and 255.

    Raises:
        NoInkFoundError: The contrast is below ``min_contrast`` (blank or nearly blank page).

    Note:
        ``"adaptive"`` assumes **dark ink on a light background**, which is how real photos
        look. For light-on-dark images use ``"otsu"``.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"expected a 2-D uint8 grayscale image, got {gray.dtype} {gray.shape}")
    if method not in ("otsu", "adaptive"):
        raise ValueError(f"unknown threshold method {method!r}; expected 'otsu' or 'adaptive'")

    if blur_ksize > 0:
        k = _odd_at_least_3(blur_ksize)
        gray = cv2.GaussianBlur(gray, (k, k), 0)  # sigma 0 = derived from the kernel size

    contrast = _ink_contrast(gray)
    if contrast < min_contrast:
        raise NoInkFoundError(
            f"no digit found: the image looks blank (contrast {contrast:.0f} < {min_contrast:g})"
        )

    if method == "otsu":
        # With THRESH_OTSU, OpenCV ignores the given threshold (0) and computes its own.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    # Adaptive: the neighbourhood must be much larger than a pen stroke but small enough to
    # follow a shadow. A quarter of the shorter side works for typical digit photos.
    block_size = _odd_at_least_3(min(gray.shape) // 4)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, adaptive_c
    )


def background_is_light(image: np.ndarray, *, border_frac: float = 0.05) -> bool:
    """Guess whether the background is light (e.g. paper) by looking at the image border.

    The edges of a digit photo are almost always background, so the border pixels tell us
    the background colour without having to find the digit first.

    The decision is "is the **median** border pixel brighter than mid-grey?", which amounts
    to "are most border pixels light?". A stroke touching the edge (common with the drawing
    canvas or a tightly cropped photo) can't flip it unless it covers half the border.
    On a binary image this is the same as comparing the mean. On a *grayscale* image the
    median is more robust: dark strokes on the edge pull the mean down, so dim paper
    (e.g. grey 150) with ink along part of the border could fall below 127 and be misjudged.
    The median stays at the paper's value.

    Args:
        image: ``HxW`` ``uint8`` image, binary or grayscale.
        border_frac: Border thickness as a fraction of the shorter side (at least 1 px).

    Returns:
        ``True`` if the median border pixel is brighter than mid-grey (> 127).
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {image.shape}")

    t = max(1, round(min(image.shape) * border_frac))
    border = np.concatenate([
        image[:t, :].ravel(),     # top rows
        image[-t:, :].ravel(),    # bottom rows
        image[:, :t].ravel(),     # left columns (corners counted twice; harmless for a median)
        image[:, -t:].ravel(),    # right columns
    ])
    return bool(np.median(border) > 127)


def ensure_white_on_black(image: np.ndarray, *, border_frac: float = 0.05) -> np.ndarray:
    """Return the image with a white digit on a black background (stage 2b).

    MNIST digits are white (255) strokes on black (0). Photos of handwriting are the
    opposite (dark ink on light paper), while the drawing canvas may be either depending
    on its colours. This function inverts only when :func:`background_is_light` says so,
    so already-correct input passes through unchanged.

    Args:
        image: ``HxW`` ``uint8`` image, normally the output of :func:`binarize`.
        border_frac: Passed to :func:`background_is_light`.

    Returns:
        A new array (the input is never modified) with a dark background.
    """
    if background_is_light(image, border_frac=border_frac):
        return 255 - image  # uint8 subtraction never wraps here: 255 - x is in 0..255
    return image.copy()


#: MNIST scales every digit so its longest side is 20 px; the 28x28 frame adds a 4 px margin.
MNIST_DIGIT_BOX = 20


def crop_to_ink(image: np.ndarray, *, threshold: int = 0) -> np.ndarray:
    """Crop a white-on-black image to the bounding box of its ink (stage 2c).

    Removes the empty margins, so the digit's size in the photo no longer matters, only
    its shape. Every pixel brighter than ``threshold`` counts as ink, so a stray speck far
    from the digit enlarges the box. The blur in :func:`binarize` removes most specks; the
    multi-digit segmenter will filter tiny blobs explicitly.

    Args:
        image: ``HxW`` ``uint8`` image with a **white digit on black**, e.g. from
            :func:`ensure_white_on_black`.
        threshold: Pixels with value ``> threshold`` are ink.

    Returns:
        The cropped region (a copy), tight around the ink.

    Raises:
        NoInkFoundError: No pixel is brighter than ``threshold``.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {image.shape}")

    rows = np.flatnonzero((image > threshold).any(axis=1))  # indices of rows containing ink
    cols = np.flatnonzero((image > threshold).any(axis=0))  # indices of columns containing ink
    if rows.size == 0:
        raise NoInkFoundError("no digit found: the image is blank after thresholding")

    return image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def fit_to_box(image: np.ndarray, box: int = MNIST_DIGIT_BOX) -> np.ndarray:
    """Resize so the longest side is ``box`` pixels, keeping the aspect ratio (stage 2d).

    Keeping the aspect ratio matters: stretching a narrow "1" to a square would make it
    look like a thick bar, unlike anything in MNIST. The short side is rounded and kept at
    least 1 px wide.

    Interpolation:

    * shrinking uses ``INTER_AREA``, which averages pixel blocks and leaves the soft grey
      anti-aliased edges MNIST digits have (they are not pure black/white);
    * enlarging a small crop uses ``INTER_LINEAR``, which is smooth and avoids blocky pixels.

    Args:
        image: ``HxW`` ``uint8`` image, normally the output of :func:`crop_to_ink`.
        box: Target length of the longest side.

    Returns:
        ``uint8`` array whose longest side is exactly ``box``.
    """
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"expected a non-empty 2-D image, got shape {image.shape}")

    h, w = image.shape
    scale = box / max(h, w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)  # cv2 takes (width, height)


#: Side length of an MNIST image.
MNIST_SIZE = 28


def center_by_mass(digit: np.ndarray, size: int = MNIST_SIZE) -> np.ndarray:
    """Pad a digit into a ``size x size`` black frame, centred by centre of mass (stage 2e).

    The **centre of mass** is the average pixel position weighted by brightness, i.e.
    where the ink "balances". MNIST put that point, not the bounding box centre, in the
    middle of the frame. The two differ for lopsided digits: a "7"'s heavy top bar pulls
    its centre of mass up, so in MNIST the 7 sits lower than its box would suggest.
    Matching this placement is a cheap accuracy win for MNIST-trained models.

    Details:

    * The target is pixel ``size // 2`` = **(14, 14)**, matching MNIST. Measured over the
      10k MNIST test digits, the mean centre of mass is (13.99, 14.01) with a std of 0.29 px.
      It is *not* the geometric centre 13.5. Using 13.5 would place every digit ~0.7 px
      diagonally off from where MNIST-trained models expect it.
    * The shift is rounded to **whole pixels**. A sub-pixel shift would need interpolation
      and slightly blur the strokes, for at most 0.5 px of extra accuracy.
    * The shift is **clamped** so the whole digit stays inside the frame. With a 20 px digit
      in a 28 px frame there are 8 px of slack. An extremely lopsided digit may therefore end
      up slightly off-centre instead of losing strokes over the edge.

    Args:
        digit: ``HxW`` ``uint8`` white-on-black digit no larger than ``size`` in either
            direction, normally the output of :func:`fit_to_box`.
        size: Side of the output frame.

    Returns:
        ``size x size`` ``uint8`` array; every pixel of ``digit`` is copied unchanged.

    Raises:
        NoInkFoundError: ``digit`` is completely black (no mass to centre).
        ValueError: ``digit`` is not 2-D or doesn't fit in the frame.
    """
    if digit.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {digit.shape}")
    h, w = digit.shape
    if h > size or w > size:
        raise ValueError(f"digit {h}x{w} does not fit in a {size}x{size} frame; call fit_to_box first")

    weights = digit.astype(np.float64)
    total = weights.sum()
    if total == 0:
        raise NoInkFoundError("no digit found: the crop is completely black")

    # Centre of mass in the digit's own coordinates (row, column).
    cy = (weights.sum(axis=1) @ np.arange(h)) / total
    cx = (weights.sum(axis=0) @ np.arange(w)) / total

    # Where the digit's top-left corner must go so its centre of mass lands on the frame centre…
    target = size // 2  # MNIST convention, see docstring
    top, left = round(target - cy), round(target - cx)
    # …clamped so the digit stays fully inside the frame.
    top = min(max(top, 0), size - h)
    left = min(max(left, 0), size - w)

    frame = np.zeros((size, size), np.uint8)
    frame[top : top + h, left : left + w] = digit
    return frame


# Stroke width as a fraction of the digit's longest side, measured on 2,000 MNIST test
# digits (binarised at 127): 5th percentile 0.076, 10th 0.084, median 0.112.
#: Below this ratio a stroke counts as "too thin" (thinner than ~93% of MNIST digits).
MIN_STROKE_RATIO = 0.08
#: When thickening, aim for the MNIST median.
TARGET_STROKE_RATIO = 0.11


def estimate_stroke_width(binary: np.ndarray) -> float:
    """Estimate the average stroke width of a white-on-black binary image, in pixels.

    Uses ``width ≈ 2 x area / perimeter``. A stroke of width ``W`` and length ``L`` has
    area ``W·L`` and an outline of about ``2·L``, so the ratio gives ``W`` for any long thin
    shape, curved or straight, without needing to skeletonise the digit.
    Inner outlines (the holes in 0, 6, 8, 9) are included in the perimeter, as they should be.

    Args:
        binary: ``HxW`` ``uint8`` image, ink > 0 on a 0 background.

    Returns:
        Estimated width in pixels; ``0.0`` if there is no ink.
    """
    mask = (binary > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    if perimeter == 0:
        return 0.0
    return 2.0 * float(mask.sum()) / perimeter


def thicken_strokes(
    crop: np.ndarray,
    *,
    min_ratio: float = MIN_STROKE_RATIO,
    target_ratio: float = TARGET_STROKE_RATIO,
) -> np.ndarray:
    """Dilate thin strokes so they survive downscaling (optional stage 2c+).

    **Why:** a fine pen on a high-resolution photo might draw 5 px strokes on a 600 px tall
    digit. :func:`fit_to_box` shrinks that by 30x, so the stroke becomes ~0.2 px, a faint
    grey ghost the model barely sees. MNIST strokes are about 11% of the digit's size.

    **When:** call it on the tight crop from :func:`crop_to_ink`, *before* :func:`fit_to_box`.
    The stroke width has to be judged relative to the digit's own size, and once the digit
    is shrunk to 20 px the information is already lost.

    **What it does:** if the estimated width is below ``min_ratio`` x the digit's longest
    side, it dilates with an elliptical kernel (round, so strokes grow evenly in every
    direction) until the width is about ``target_ratio`` x the side. Strokes that are thick
    enough are returned unchanged (as a copy). The crop is padded first, so strokes at the
    edge grow outwards instead of being clipped. The result is therefore slightly larger.

    Args:
        crop: ``HxW`` ``uint8`` white-on-black **binary** crop (output of :func:`crop_to_ink`).
        min_ratio: Thicken only below this width/size ratio. Default from MNIST statistics.
        target_ratio: Width/size ratio to aim for when thickening (MNIST median).

    Returns:
        ``uint8`` crop, possibly larger than the input by the kernel size.
    """
    if crop.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {crop.shape}")

    side = max(crop.shape)
    width = estimate_stroke_width(crop)
    if width == 0 or width >= min_ratio * side:
        return crop.copy()

    # Dilating with a k x k kernel widens a stroke by k - 1 px. Kernel sizes must be odd
    # so the kernel has a centre pixel.
    k = _odd_at_least_3(round(target_ratio * side - width) + 1)
    pad = k // 2
    padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(padded, kernel)


def to_tensor(image28: np.ndarray) -> torch.Tensor:
    """Convert a 28x28 ``uint8`` digit to a ``[1, 28, 28]`` ``float32`` tensor in ``[0, 1]``.

    This is the format of one crop in the :class:`~digitrec.models.base.Predictor` input
    contract: values in [0, 1], white digit on black, **not** normalised (each backend
    applies its own mean/std). The pipeline stacks these into an ``[N, 1, 28, 28]`` batch.
    """
    if image28.shape != (MNIST_SIZE, MNIST_SIZE):
        raise ValueError(f"expected a {MNIST_SIZE}x{MNIST_SIZE} image, got shape {image28.shape}")
    return torch.from_numpy(image28.astype(np.float32) / 255.0).unsqueeze(0)


@dataclass(frozen=True)
class PreprocessOptions:
    """Knobs for the preprocessing chain; the defaults suit evenly lit photos and the canvas.

    Attributes:
        threshold: ``"otsu"`` or ``"adaptive"`` (photos with shadows), see :func:`binarize`.
        blur_ksize: Gaussian blur kernel before thresholding; ``0`` disables it.
        thicken: Dilate thin pen strokes, see :func:`thicken_strokes`. It does nothing to
            strokes that are already MNIST-thick, so it's on by default.
        min_contrast: Treat the image as blank below this contrast, see :func:`binarize`.
            Lower it for very faint pencil; ``0`` disables the check.
    """

    threshold: ThresholdMethod = "otsu"
    blur_ksize: int = 5
    thicken: bool = True
    min_contrast: float = MIN_CONTRAST


@dataclass
class PreprocessResult:
    """Output of preprocessing one digit.

    Attributes:
        image: The final ``28x28`` ``uint8`` MNIST-style digit (for display).
        tensor: The same as a ``[1, 28, 28]`` float tensor in ``[0, 1]`` (for the model).
        stages: Every intermediate image in pipeline order, keyed by step name:
            ``grayscale``, ``binary``, ``white_on_black`` (page half) then ``crop``,
            ``thickened``, ``fit_20px``, ``mnist_28x28`` (digit half). ``thickened`` equals
            ``crop`` when no thickening was needed.
    """

    image: np.ndarray
    tensor: torch.Tensor
    stages: dict[str, np.ndarray] = field(default_factory=dict)


def prepare_page(
    source: ImageSource, options: PreprocessOptions | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """First half of preprocessing, for the whole image: load, binarize, invert.

    Args:
        source: Any :data:`ImageSource`.
        options: Preprocessing knobs; ``None`` means the defaults.

    Returns:
        ``(white_on_black, stages)``: the binary page with white ink on black, ready
        for a segmenter, and the intermediate images (``grayscale``, ``binary``,
        ``white_on_black``).
    """
    options = options or PreprocessOptions()
    gray = load_image(source)
    binary = binarize(
        gray,
        method=options.threshold,
        blur_ksize=options.blur_ksize,
        min_contrast=options.min_contrast,
    )
    page = ensure_white_on_black(binary)
    return page, {"grayscale": gray, "binary": binary, "white_on_black": page}


def normalize_digit(region: np.ndarray, options: PreprocessOptions | None = None) -> PreprocessResult:
    """Second half of preprocessing, for one digit: crop, thicken, fit 20 px, centre, tensor.

    Args:
        region: White-on-black binary image containing exactly one digit, e.g. the whole
            page from :func:`prepare_page` (single digit) or one segmented region (multi-digit).
        options: Preprocessing knobs; ``None`` means the defaults.

    Returns:
        A :class:`PreprocessResult` whose ``stages`` holds this half's steps only.

    Raises:
        NoInkFoundError: ``region`` has no ink.
    """
    options = options or PreprocessOptions()
    crop = crop_to_ink(region)
    thick = thicken_strokes(crop) if options.thicken else crop
    fitted = fit_to_box(thick)
    image28 = center_by_mass(fitted)
    stages = {"crop": crop, "thickened": thick, "fit_20px": fitted, "mnist_28x28": image28}
    return PreprocessResult(image=image28, tensor=to_tensor(image28), stages=stages)


def preprocess(source: ImageSource, options: PreprocessOptions | None = None) -> PreprocessResult:
    """Turn a single-digit image into an MNIST-style 28x28 digit (stages 1–2 end to end).

    Example::

        result = preprocess("samples/seven.jpg")
        result.tensor.shape    # torch.Size([1, 28, 28])
        list(result.stages)    # ['grayscale', 'binary', 'white_on_black', 'crop', ...]

    Args:
        source: Any :data:`ImageSource` containing one handwritten digit.
        options: Preprocessing knobs; ``None`` means the defaults.

    Returns:
        A :class:`PreprocessResult` with the final image, tensor and *all* stages.

    Raises:
        NoInkFoundError: Nothing that looks like ink was found.
    """
    page, page_stages = prepare_page(source, options)
    result = normalize_digit(page, options)
    result.stages = {**page_stages, **result.stages}
    return result
