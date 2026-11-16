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
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Literal

import cv2
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


#: Thresholding strategy for :func:`binarize`.
ThresholdMethod = Literal["otsu", "adaptive"]


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
) -> np.ndarray:
    """Denoise a grayscale image and threshold it to pure black and white (stage 2a).

    1. **Gaussian blur** smooths sensor noise and paper texture, so stray specks don't
       survive thresholding as fake "ink".
    2. **Threshold** sets every pixel to 0 or 255:

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

    Returns:
        ``HxW`` ``uint8`` array containing only 0 and 255.

    Note:
        ``"adaptive"`` assumes **dark ink on a light background**, which is how real photos
        look. For light-on-dark images use ``"otsu"``.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"expected a 2-D uint8 grayscale image, got {gray.dtype} {gray.shape}")

    if blur_ksize > 0:
        k = _odd_at_least_3(blur_ksize)
        gray = cv2.GaussianBlur(gray, (k, k), 0)  # sigma 0 = derived from the kernel size

    if method == "otsu":
        # With THRESH_OTSU, OpenCV ignores the given threshold (0) and computes its own.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    if method == "adaptive":
        # The neighbourhood must be much larger than a pen stroke but small enough to
        # follow a shadow. A quarter of the shorter side works for typical digit photos.
        block_size = _odd_at_least_3(min(gray.shape) // 4)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, adaptive_c
        )

    raise ValueError(f"unknown threshold method {method!r}; expected 'otsu' or 'adaptive'")


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


class NoInkFoundError(ValueError):
    """The image contains no digit pixels (blank page, empty canvas, or everything thresholded away).

    Subclasses ``ValueError`` so generic handlers still catch it, while the CLI and UI can
    catch it specifically and show a friendly "no digit found" message.
    """


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
