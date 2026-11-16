# Session 1 — Handoff Notes

Date: 2026-11-16 · Last commit before this file: `afe1b8b`
Read this together with [`plan.md`](plan.md) (the full design and checklists).

## How the user wants to work

- **Tiny atomic changes.** Implement one small step, run the tests, then **stop and show it for
  review** (what changed, why, test results, proposed commit message). Commit only after the user
  says so ("commit and next" / "proceed").
- **Good documentation**: module and function docstrings that explain *why*, not just *what*;
  comments recording where numeric defaults came from.
- **Ground defaults in measurements**, not guesses (e.g. thresholds were measured on MNIST).
- The user sometimes asks for short concept explanations (e.g. what a checkpoint is, what
  `torch.device` does) before moving on. Wait for their "okay".
- The user likes **visual confirmation** of each step: `scripts/show_stages.py`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Be upfront about mistakes found along the way. The user valued the corrections below.

## Where we are in plan.md (Phase 1)

| Plan item | Status |
| --- | --- |
| Setup: uv project, src layout, `.gitignore`, deps, `config.py` | ✅ done |
| Core interfaces: `models/base.py`, `models/registry.py` | ✅ done |
| Preprocessing (load, denoise/threshold, invert, crop, 20 px, centre, dilation, normalise) | ✅ done + blank-page guard |
| Visual check script | ✅ done (extra, not in plan) |
| **Segmentation** (`Segmenter` interface, `SingleDigitSegmenter`) | ⏭️ **next step** |
| `models/pretrained.py` (Hugging Face MNIST model) | not started |
| `pipeline.py`, low-confidence flag | not started |
| `cli.py`, `app/streamlit_app.py` | not started |
| Validation: MNIST subset accuracy, own photos in `samples/`, tests | not started (unit tests exist) |
| Phase 1.5 multi-digit, Phase 2 own CNN | not started |

Tests: **102 passing** (`uv run pytest`).

## Files so far

```text
pyproject.toml / uv.lock / .python-version   uv project, Python 3.13 (uv-managed), deps pinned in lock
src/digitrec/__init__.py                     package docstring describing the 5 pipeline stages
src/digitrec/config.py                       Config (backend, checkpoint_path, device), get_device()
src/digitrec/models/base.py                  DigitPrediction, Predictor Protocol, probs_to_predictions()
src/digitrec/models/registry.py              get_predictor(config), lazy "module:function" backends, lru_cache
src/digitrec/preprocessing.py                stages 1–2, see below
scripts/show_stages.py                       HTML visual check of every preprocessing stage
tests/test_config.py, test_models_base.py, test_registry.py, test_preprocessing.py, test_show_stages.py
```

## Key design decisions (and why)

1. **Predictor input contract** (`models/base.py`): crops are `[N, 1, 28, 28]` float in **[0, 1]**,
   **white digit on black**, **NOT normalised**. Each backend applies its own normalisation
   (MNIST mean/std for our CNN, the HF image processor's stats for the pretrained model) inside
   `predict()`. This keeps preprocessing identical for every backend.
2. **`Predictor` is a `typing.Protocol`**: backends don't inherit, they just need `name` and `predict()`.
   `probs_to_predictions(probs[N,10])` is the shared helper every backend should end with.
3. **Registry**: backends are `"module:function"` strings imported lazily
   (`"pretrained": "digitrec.models.pretrained:load"`, `"scratch_cnn": "digitrec.models.cnn:load"`).
   Each module must expose `load(config: Config) -> Predictor`. Results are cached per (hashable,
   frozen) `Config`. A missing backend module gives `NotImplementedError`, while a missing *dependency*
   inside the module is re-raised unchanged.
4. **`get_device()`**: auto picks cuda → mps → cpu. An explicitly requested but unavailable device
   **raises** rather than silently falling back. The user's Mac reports `mps`.
5. **Preprocessing is split in two halves** so multi-digit segmentation can slot in between:
   ```text
   prepare_page(source)    load_image → binarize → ensure_white_on_black      (once per image)
         ▼  segmenter goes here (whole page for one digit; one region per contour later)
   normalize_digit(region) crop_to_ink → thicken_strokes → fit_to_box(20) → center_by_mass(28) → to_tensor
   ```
   `preprocess(source, options)` = both halves for the single-digit case. It returns
   `PreprocessResult(image, tensor[1,28,28], stages)`. `stages` holds every intermediate image, in order:
   `grayscale, binary, white_on_black, crop, thickened, fit_20px, mnist_28x28`.
6. `PreprocessOptions(threshold="otsu"|"adaptive", blur_ksize=5, thicken=True, min_contrast=15)`.
7. `NoInkFoundError(ValueError)` is raised for blank input. The CLI/UI should catch it and show
   "no digit found".
8. `binarize` **keeps polarity**. Inversion is a separate step that decides from the **median of the
   border pixels**.

## Measured numbers behind the defaults

| What | Value | Source |
| --- | --- | --- |
| MNIST centre-of-mass position | (13.99, 14.01), std 0.29 | 10k MNIST test digits → `center_by_mass` targets pixel **14**, not 13.5 |
| MNIST stroke width ÷ digit size | p5 0.076, p10 0.084, median 0.112 | 2k MNIST digits → thicken below **0.08**, aim for **0.11** |
| Blank-page contrast (after lighting removal) | blank ≤ 9.5, real ink ≥ 18.5 | synthetic pages → `MIN_CONTRAST = 15` |
| Round trip: MNIST digit → fake paper photo → `preprocess()` | median pixel correlation **0.978** (p5 0.824) | 300 MNIST test digits |
| Speed | ~0.4 s full preprocess of a 12 MP photo | |

## Hurdles hit and how they were solved

1. **pyenv Python 3.13.2 lacks `_lzma`** → `import torchvision` crashed. Fixed by using a uv-managed
   interpreter (`uv python install 3.13`, pinned in `.python-version`, now 3.13.18). Documented in README
   and `pyproject.toml`.
2. **`streamlit_drawable_canvas` can't be imported from plain Python** (`StreamlitAPIException`: component
   assets are registered only at runtime). It works inside `streamlit run` / `AppTest`. Not a bug.
   Remember this when writing tests for the Streamlit app.
3. **NumPy 2 removed `ndarray.ptp()`**. Use `np.ptp(arr)`. (Hit twice.)
4. **Wrong claim corrected**: the border *median* is no more robust than the mean on *binary* images.
   It only helps on grayscale. Docstring and tests were fixed accordingly.
5. **Centre-of-mass target bug**: the first version used 13.5 (geometric centre). Measuring MNIST showed 14.
   Fixing it raised round-trip correlation from **0.835 → 0.978**.
6. **Blank noisy paper produced a fake "digit"** (Otsu always splits the histogram). The first fix (global
   Otsu contrast ≥ 30) failed: a lighting gradient on blank paper (27.6) overlapped faint pencil (29.6).
   The final fix subtracts a heavily blurred background estimate (computed on a 64 px copy for speed) before
   measuring contrast, which separates the two cleanly.
7. Loading `scripts/show_stages.py` in tests via `importlib` needs `sys.modules[...]` registration before
   `exec_module`, because `@dataclass` looks up its module there.

## Environment notes

- Run everything with `uv run ...` (venv is `.venv`, Python 3.13.18).
- MNIST test set is downloaded in `data/MNIST/` (gitignored); `torchvision.datasets.MNIST(root="data")`.
- `debug/` (gitignored) holds `show_stages.py` output.
- Useful commands:
  ```bash
  uv run pytest -q
  uv run python scripts/show_stages.py --mnist 5 --open
  ```

## Next steps (in plan order)

1. **`segmentation.py`**: a `Segmenter` Protocol `segment(page: np.ndarray) -> list[np.ndarray]`
   (regions ordered left → right) and `SingleDigitSegmenter` returning `[page]`. Each region then goes
   through `normalize_digit()`. Design it so `ContourSegmenter` (Phase 1.5) drops in later. The segmenter
   may also need to return bounding boxes for the UI to draw on later.
2. **`models/pretrained.py`**: first verify the HF candidate `farleyknight-org-username/vit-base-mnist`
   (label mapping 0–9, input size/channels, licence). Fallback: `microsoft/trocr-base-handwritten`. Expose
   `load(config) -> Predictor`. Inside `predict`: resize 28 → the model's input size, convert to 3 channels,
   apply the processor's normalisation, softmax, `probs_to_predictions`.
3. `pipeline.py`: `Pipeline(segmenter, predictor).run(image) -> Result(number, digits, confidence, debug_images)`,
   plus a low-confidence flag (< 0.6).
4. `cli.py`, then the Streamlit app, then validation (MNIST subset accuracy ~98–99%, own photos in `samples/`).
