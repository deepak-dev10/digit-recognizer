# digitrec — Handwritten Number Recognizer

Takes a photo of handwritten digits and predicts the number.

- **Phase 1:** uses an already-trained MNIST model (no training).
- **Phase 2:** a CNN designed and trained from scratch in PyTorch, swapped in via config.

See [`plan.md`](plan.md) for the full architecture and roadmap.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 (pinned in `.python-version`).

```bash
uv python install 3.13                    # uv-managed interpreter (includes lzma, needed by torchvision)
uv sync                                   # create .venv, install dependencies + the package
uv run python -c "import digitrec; print(digitrec.__version__)"
```

### Dependencies

| Package | Used for |
| --- | --- |
| `torch`, `torchvision` | model inference/training, MNIST dataset (Phase 2) |
| `transformers` | loading the pretrained Hugging Face MNIST classifier (Phase 1) |
| `opencv-python`, `pillow`, `numpy` | image loading and preprocessing |
| `streamlit`, `streamlit-drawable-canvas` | web UI with upload and draw-a-digit canvas |
| `pytest` (dev) | tests |

> **Note:** `streamlit_drawable_canvas` can only be imported inside a running Streamlit app
> (`streamlit run ...`). Importing it from plain Python raises a `StreamlitAPIException`
> because Streamlit registers the component's assets only at runtime.

On Apple Silicon, PyTorch uses the `mps` GPU backend automatically when available.

## Visual check of preprocessing

`scripts/show_stages.py` runs images through the preprocessing chain and writes one
self-contained HTML page showing every intermediate image (grayscale → binary →
white-on-black → crop → thickened → 20 px → final 28×28):

```bash
uv run python scripts/show_stages.py --open                   # built-in demo inputs
uv run python scripts/show_stages.py samples/*.jpg --open     # your own photos
uv run python scripts/show_stages.py img.jpg --threshold adaptive --no-thicken
uv run python scripts/show_stages.py --mnist 5                # + real MNIST round trips
```

Output goes to `debug/stages.html` (gitignored). The script prints one line per input and
exits non-zero if any input ended unexpectedly, e.g. an unreadable file.

## Project layout

```text
src/digitrec/   # the library: preprocessing, segmentation, models, pipeline
scripts/        # developer tools (show_stages.py: visual check of preprocessing)
tests/          # pytest suite: uv run pytest
plan.md         # design and phase checklists
```

More sections (CLI usage, Streamlit app, tests) will be added as those parts are built.
