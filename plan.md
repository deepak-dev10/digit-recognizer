# Handwritten Number Recognizer — Plan

## Context
Build a Python app that takes a picture of handwritten digits and predicts the number.
- **Phase 1:** no training — use an already-trained MNIST model.
- **Phase 2:** design + train a CNN from scratch in PyTorch and swap it in.
- Start with **single-digit** images, but design it so **multi-digit** support plugs in later without a rewrite.
- Interfaces: **CLI + Streamlit web UI**. Framework: **PyTorch**. Tooling: `uv` (installed), Python 3.13 (fall back to 3.12 if a wheel is missing).
- Working dir `/Users/deepak/Projects/ml_projects` is empty → greenfield project `digit_recognizer/`.

---

## Recommended Architecture

A **modular pipeline with a pluggable model backend**. The model is just one swappable stage, so Phase 1 → Phase 2 is a config change, and single → multi-digit is a segmentation change.

```
Image (photo / upload / canvas)
   │
   ▼
[1] Loader        → PIL/OpenCV, EXIF-rotate, to grayscale
   ▼
[2] Preprocessor  → make the input look like MNIST          ← most important stage
   ▼
[3] Segmenter     → SingleDigitSegmenter (whole image)  |  ContourSegmenter (multi-digit, later)
   ▼  list of 28×28 digit crops, ordered left→right
[4] Classifier    → Predictor interface:  PretrainedPredictor (Phase 1) | ScratchCNNPredictor (Phase 2)
   ▼  per-digit (label, confidence, probs)
[5] Assembler     → join digits into "4072", overall confidence = min/product of per-digit conf
   ▼
CLI output  /  Streamlit display (crops, per-digit bars, final number)
```

**Why this architecture**
- MNIST models fail on real photos mainly because of *preprocessing mismatch* (white background, thick/thin strokes, off-centre, wrong scale) — not the model. Isolating stage 2 lets you fix that once for every backend.
- A common `Predictor` interface (`predict(batch_of_28x28) -> probs`) means the pretrained model and your own CNN are interchangeable and can be benchmarked side by side.
- Classifying each digit individually (segment → classify) is the simplest correct route to multi-digit, reusing the exact same single-digit model.

**Model architecture for Phase 2 (from scratch):** a small CNN — the standard best fit for MNIST (~99.3–99.6% test accuracy, trains in minutes on CPU/Apple MPS):
```
Input 1×28×28
Conv(1→32,3) → BN → ReLU → Conv(32→32,3) → BN → ReLU → MaxPool2 → Dropout(0.25)
Conv(32→64,3) → BN → ReLU → Conv(64→64,3) → BN → ReLU → MaxPool2 → Dropout(0.25)
Flatten → Linear(→128) → ReLU → Dropout(0.5) → Linear(128→10)
```
(Optional learning stepping stones: MLP baseline → LeNet-5 → this CNN.)

**Later multi-digit upgrade options:** (a) contour segmentation + per-digit CNN (planned, default); (b) for touching/overlapping digits: CRNN (CNN + BiLSTM + CTC loss) trained on synthetic multi-digit MNIST strings — stretch goal only.

---

## Project Structure
```
digit_recognizer/
├── pyproject.toml                 # uv project: torch, torchvision, transformers, opencv-python, pillow, numpy, streamlit, streamlit-drawable-canvas, pytest
├── README.md
├── samples/                       # your own test photos (0–9 + some multi-digit)
├── checkpoints/                   # Phase 2 trained weights (.pt), gitignored
├── src/digitrec/
│   ├── config.py                  # backend name, checkpoint path, device (cuda/mps/cpu)
│   ├── preprocessing.py           # stage 1+2
│   ├── segmentation.py            # stage 3: SingleDigitSegmenter, ContourSegmenter
│   ├── pipeline.py                # stage 4+5: Pipeline.run(image) -> Result
│   └── models/
│       ├── base.py                # Predictor Protocol + DigitPrediction dataclass
│       ├── registry.py            # get_predictor("pretrained" | "scratch_cnn")
│       ├── pretrained.py          # Phase 1 wrapper
│       └── cnn.py                 # Phase 2 nn.Module + ScratchCNNPredictor
├── training/                      # Phase 2 only
│   ├── data.py                    # MNIST (+EMNIST-digits optional) loaders, augmentation
│   ├── train.py                   # training loop, checkpointing, early stopping
│   └── evaluate.py                # test accuracy, confusion matrix, misclassified grid
├── cli.py                         # `uv run digitrec predict img.png --backend pretrained`
├── app/streamlit_app.py           # upload or draw, show preprocessed crop + prediction
└── tests/                         # preprocessing, segmentation, pipeline smoke tests
```

---

## Phase 1 — Pretrained Model (no training)

### Model choice
- **Primary:** an MNIST-fine-tuned image classifier from Hugging Face Hub loaded via `transformers` (`AutoImageProcessor` + `AutoModelForImageClassification`). Candidate to verify first: `farleyknight-org-username/vit-base-mnist` (ViT, ~99% MNIST). Check the model card: label mapping 0–9, expected input size/channels, license.
- **Fallback if no reliable hub checkpoint:** Microsoft `microsoft/trocr-base-handwritten` (OCR, reads digit strings directly — also a handy multi-digit baseline to compare against later), restricting output to digits.
- Wrap whichever is chosen in `PretrainedPredictor` so it conforms to the `Predictor` interface (it accepts our 28×28 crops and internally resizes/converts to 3-channel as the model requires).

### Checklist
**Setup**
- [ ] `uv init digit_recognizer`, add deps, set `src/` layout, `.gitignore` (checkpoints, .venv, HF cache)
- [ ] `git init`; `config.py` with device auto-detect (`mps` on Apple Silicon → `cpu`)

**Core interfaces**
- [ ] `models/base.py`: `DigitPrediction(label:int, confidence:float, probs:list[float])`, `Predictor` Protocol with `predict(crops: Tensor[N,1,28,28]) -> list[DigitPrediction]`
- [ ] `models/registry.py`: name → predictor factory, cached load

**Preprocessing (`preprocessing.py`)**
- [ ] Load any format, apply EXIF orientation, convert to grayscale
- [ ] Denoise (Gaussian blur), threshold (Otsu; adaptive threshold for uneven lighting/shadows)
- [ ] Auto-invert so digit = white on black (MNIST convention) — decide by border-pixel mean
- [ ] Crop to ink bounding box, resize longest side to 20 px preserving aspect ratio
- [ ] Pad to 28×28, shift so centre of mass is at the centre (exact MNIST recipe)
- [ ] Optional dilation for very thin pen strokes
- [ ] Normalize (MNIST mean 0.1307 / std 0.3081, or the pretrained model's processor stats)
- [ ] Return intermediate images for debugging/display

**Segmentation (`segmentation.py`)**
- [ ] `SingleDigitSegmenter`: returns the whole preprocessed image as one crop
- [ ] Define `Segmenter` interface now (`segment(binary_img) -> list[crop]`) so multi-digit drops in later

**Model + pipeline**
- [ ] `models/pretrained.py`: download/cache weights, label mapping, softmax → `DigitPrediction`
- [ ] `pipeline.py`: `Pipeline(segmenter, predictor).run(image) -> Result(number:str, digits, confidence, debug_images)`
- [ ] Low-confidence flag (e.g. < 0.6 → "uncertain")

**Interfaces**
- [ ] `cli.py` (argparse/typer): `predict <path...> [--backend] [--multi] [--show-debug]`, prints number + confidence
- [ ] `app/streamlit_app.py`: file upload **and** drawing canvas; show original → preprocessed 28×28 → prediction + probability bar chart; backend selector in sidebar; `@st.cache_resource` for model

**Validation**
- [ ] Run against MNIST test subset (sanity: should be ~98–99%)
- [ ] Collect ~30 of your own photos (paper, different pens, lighting) into `samples/`; record accuracy
- [ ] Unit tests: inversion, centering, output shape 1×28×28, pipeline smoke test

### Phase 1.5 — Multi-digit (after single digit works)
- [ ] `ContourSegmenter`: `cv2.findContours` on binary image → bounding boxes → filter by min area/aspect → merge boxes that overlap vertically (e.g. broken strokes of “5”, “7”) → sort left→right
- [ ] Each box → same MNIST preprocessing (20px + pad + centre-of-mass) → batch to predictor
- [ ] Assembler joins labels; Streamlit draws boxes on the original image with each predicted digit
- [ ] `--multi` CLI flag / UI toggle; test with multi-digit samples
- [ ] Known limit: touching digits → handled later by CRNN stretch goal

---

## Phase 2 — Train Your Own Model From Scratch

### Checklist
**Data (`training/data.py`)**
- [ ] `torchvision.datasets.MNIST` (60k train / 10k test); split 55k train / 5k validation
- [ ] Training augmentation to bridge the gap to real photos: `RandomAffine(rotation ±10°, translate 10%, scale 0.9–1.1, shear 10°)`, optional elastic distortion, random erosion/dilation (stroke thickness)
- [ ] Optional: add EMNIST-Digits (240k) for more handwriting variety
- [ ] Optional: add your own labelled photos (run through *the same* `preprocessing.py`) as a fine-tune/test set

**Model (`models/cnn.py`)**
- [ ] Implement the CNN above as `nn.Module`; verify output shape `[N,10]`; print param count (~300k)
- [ ] (Learning path) implement MLP baseline first, then LeNet-5, compare accuracy

**Training (`training/train.py`)**
- [ ] Loss: `CrossEntropyLoss` (optionally label smoothing 0.1)
- [ ] Optimizer: Adam/AdamW lr 1e-3, `OneCycleLR` or `ReduceLROnPlateau`; batch 128; 10–15 epochs
- [ ] Train/val loop with accuracy logging (print or TensorBoard), seed fixing for reproducibility
- [ ] Save best checkpoint by val accuracy: `{"state_dict", "config", "epoch", "val_acc"}` → `checkpoints/cnn_best.pt`
- [ ] Early stopping on val loss

**Evaluation (`training/evaluate.py`)**
- [ ] Test accuracy (target ≥ 99.3%), per-class accuracy, confusion matrix plot
- [ ] Grid of misclassified examples to understand failures
- [ ] Evaluate on your own `samples/` photos — this is the real-world metric

**Integration**
- [ ] `ScratchCNNPredictor` loads checkpoint, registered as `"scratch_cnn"` in registry
- [ ] CLI `--backend scratch_cnn`; Streamlit sidebar selector; side-by-side comparison mode (pretrained vs yours)
- [ ] Benchmark script: accuracy + latency of both backends on MNIST test + your samples

**Stretch goals**
- [ ] Pure NumPy implementation of an MLP (forward/backprop by hand) to understand the internals
- [ ] CRNN + CTC on synthetic multi-digit strings (concatenated MNIST digits) for touching digits
- [ ] Export to ONNX / TorchScript; Dockerize the Streamlit app

---

## Verification (end-to-end)
1. `uv run pytest` — preprocessing/segmentation/pipeline tests pass.
2. `uv run python cli.py predict samples/seven.jpg` → prints `7` with confidence.
3. `uv run streamlit run app/streamlit_app.py` → upload a photo and draw a digit; check the 28×28 debug view looks like MNIST (white digit, black background, centred) and predictions are correct.
4. Phase 1.5: `cli.py predict samples/4072.jpg --multi` → `4072`; UI shows boxes in the right order.
5. Phase 2: `uv run python training/train.py` reaches ≥ 99.3% test acc; `--backend scratch_cnn` gives predictions on the same samples; benchmark table compares both backends.

## Order of work
Setup → interfaces → preprocessing (test with debug images) → pretrained predictor → pipeline → CLI → Streamlit → own-photo evaluation → multi-digit segmentation → Phase 2 data/model/train/eval → integrate + benchmark.
