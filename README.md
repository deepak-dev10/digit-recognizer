# digitrec — Handwritten Number Recognizer

Takes a photo of handwritten digits and predicts the number.

- **Phase 1:** uses an already-trained MNIST model (no training).
- **Phase 2:** a CNN designed and trained from scratch in PyTorch, swapped in via config.

See [`plan.md`](plan.md) for the full architecture and roadmap.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync                                   # create .venv and install the package
uv run python -c "import digitrec; print(digitrec.__version__)"
```

## Project layout

```text
src/digitrec/   # the library: preprocessing, segmentation, models, pipeline
plan.md         # design and phase checklists
```

More sections (CLI usage, Streamlit app, tests) will be added as those parts are built.
