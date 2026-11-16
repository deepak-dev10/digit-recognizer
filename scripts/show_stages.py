"""Visual check of the preprocessing chain: every intermediate image, for any input.

Runs images through :func:`digitrec.preprocessing.preprocess` and writes one
self-contained HTML page (works offline) showing each stage side by side. Use it after
changing preprocessing code, or to see why a particular photo is misread.

Usage::

    uv run python scripts/show_stages.py                          # built-in demo inputs
    uv run python scripts/show_stages.py samples/*.jpg            # your own images
    uv run python scripts/show_stages.py img.jpg --threshold adaptive --no-thicken
    uv run python scripts/show_stages.py --mnist 5 --open         # + 5 MNIST round trips, open browser

Demo inputs are synthetic (a shadowed photo, a fine-pen digit, a drawing-canvas image, blank
paper) and each one shows a specific step at work. ``--mnist N`` turns real MNIST test
digits into fake paper photos and compares our 28x28 output with the original. It
downloads the MNIST test set (~11 MB) into ``data/`` the first time.

Options such as ``--threshold`` apply to images you pass. Demo inputs use their own
settings, since several are deliberate before/after comparisons.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from digitrec.config import PROJECT_ROOT
from digitrec.preprocessing import (
    ImageSource,
    NoInkFoundError,
    PreprocessOptions,
    preprocess,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "debug" / "stages.html"

# Images larger than this are shrunk for display only, which keeps the HTML file small.
_DISPLAY_MAX_SIDE = 260

# One-line explanation of each stage name returned in PreprocessResult.stages.
STAGE_INFO = {
    "grayscale": "load_image: EXIF fix, transparency → white, grayscale",
    "binary": "binarize: blur, blank check, threshold to 0/255",
    "white_on_black": "ensure_white_on_black: invert if the border is light",
    "crop": "crop_to_ink: tight box around the ink",
    "thickened": "thicken_strokes: dilate if strokes are thin for their size",
    "fit_20px": "fit_to_box: longest side → 20 px",
    "mnist_28x28": "center_by_mass: 28×28, centre of mass on (14, 14)",
}


@dataclass
class Case:
    """One input to visualise."""

    title: str
    note: str
    source: ImageSource
    options: PreprocessOptions | None = None
    reference: np.ndarray | None = None  # original MNIST digit, for round-trip comparisons
    expect_blank: bool = False           # True if NoInkFoundError is the correct outcome


# --- demo inputs -----------------------------------------------------------------------


def _jpeg(gray: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(gray).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _noisy(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return (img + rng.normal(0, sigma, img.shape)).clip(0, 255).astype(np.uint8)


def demo_cases() -> list[Case]:
    """Synthetic inputs, each chosen to show one preprocessing step doing its job."""
    rng = np.random.default_rng(0)

    # A "4" drawn across paper that fades into a deep shadow.
    shadow = np.tile(np.linspace(245, 75, 420), (420, 1))
    mask = np.zeros((420, 420), np.uint8)
    for a, b in (((250, 70), (110, 260)), ((110, 260), (300, 260)), ((250, 70), (250, 360))):
        cv2.line(mask, a, b, 255, 10)
    shadow[mask > 0] *= 0.45
    shadow_photo = _jpeg(_noisy(shadow, 5, rng))

    # A "2" written with a fine (4 px) pen on a large (900x700) photo.
    page = np.full((900, 700), 228, np.uint8)
    cv2.ellipse(page, (350, 300), (160, 170), 0, 160, 380, 35, 4)
    cv2.line(page, (500, 360), (190, 700), 35, 4)
    cv2.line(page, (190, 700), (540, 700), 35, 4)
    fine_pen = _jpeg(_noisy(page, 4, rng))

    # What the Streamlit drawing canvas returns: black strokes on transparent pixels.
    canvas = np.zeros((280, 280, 4), np.uint8)
    pts = [(90, 70), (190, 70), (130, 140), (195, 190), (150, 235), (85, 220)]  # a "3"
    for a, b in zip(pts, pts[1:]):
        cv2.line(canvas, a, b, (0, 0, 0, 255), 18)

    blank = _noisy(np.full((400, 400), 232.0), 6, rng)
    blank_shadow = _noisy(np.tile(np.linspace(245, 90, 400), (400, 1)), 6, rng)

    return [
        Case("Shadowed photo · Otsu threshold",
             "Paper fades into a deep shadow. One global threshold turns the shadowed half black, "
             "so the 28×28 output is a blob.",
             shadow_photo, PreprocessOptions(threshold="otsu")),
        Case("Shadowed photo · adaptive threshold",
             "Same photo with threshold=\"adaptive\": each neighbourhood gets its own threshold, "
             "so the paper stays white and the 4 comes through.",
             shadow_photo, PreprocessOptions(threshold="adaptive")),
        Case("Fine pen · thickening off",
             "A 4 px pen on a 900 px photo. Shrinking to 20 px leaves faint grey strokes.",
             fine_pen, PreprocessOptions(thicken=False)),
        Case("Fine pen · thickening on (default)",
             "thicken_strokes dilates the crop to an MNIST-like stroke ratio before shrinking.",
             fine_pen),
        Case("Drawing canvas (RGBA, transparent background)",
             "Transparent pixels become white paper, then the usual chain runs.",
             canvas),
        Case("Blank paper with sensor noise",
             "Expected outcome: NoInkFoundError. The blank check in binarize stops Otsu from "
             "turning noise into a fake digit.",
             blank, expect_blank=True),
        Case("Blank paper in a deep shadow",
             "Expected outcome: NoInkFoundError. Lighting changes are removed before measuring "
             "contrast, so a shadow isn't mistaken for ink.",
             blank_shadow, PreprocessOptions(threshold="adaptive"), expect_blank=True),
    ]


def mnist_cases(n: int) -> list[Case]:
    """Real MNIST test digits disguised as paper photos, to compare our output with the original."""
    from torchvision.datasets import MNIST  # imported here: only needed with --mnist

    dataset = MNIST(root=str(PROJECT_ROOT / "data"), train=False, download=True)
    rng = np.random.default_rng(1)
    cases = []
    for i in range(n):
        digit = dataset.data[i].numpy()
        big = cv2.resize(digit, None, fx=15, fy=15, interpolation=cv2.INTER_CUBIC)
        page = (255 - np.pad(big, 150)).astype(float) * 0.85 + 20  # dark ink on paper
        cases.append(Case(
            f"MNIST test #{i} as a paper photo (label {int(dataset.targets[i])})",
            "Inverted, enlarged 15×, padded with paper, noise and JPEG added. "
            "The last panel is the original digit.",
            _jpeg(_noisy(page, 8, rng)), reference=digit,
        ))
    return cases


def file_cases(paths: list[Path], options: PreprocessOptions) -> list[Case]:
    return [Case(p.name, str(p), p, options) for p in paths]


# --- running and rendering -------------------------------------------------------------


def _png_data_uri(img: np.ndarray) -> str:
    if max(img.shape[:2]) > _DISPLAY_MAX_SIDE:
        s = _DISPLAY_MAX_SIDE / max(img.shape[:2])
        size = (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run_case(case: Case) -> dict:
    """Preprocess one case and collect everything the page needs to show it."""
    out = {"title": case.title, "note": case.note, "stages": [], "reference": None, "error": None}
    try:
        result = preprocess(case.source, case.options)
    except NoInkFoundError as e:
        out["error"] = str(e)
        out["status"] = "ok" if case.expect_blank else "fail"
        return out
    except Exception as e:  # unreadable file etc.: show it instead of aborting the whole report
        out["error"] = f"{type(e).__name__}: {e}"
        out["status"] = "fail"
        return out

    out["status"] = "fail" if case.expect_blank else "ok"
    for name, img in result.stages.items():
        out["stages"].append({
            "name": name,
            "info": STAGE_INFO.get(name, ""),
            "shape": f"{img.shape[0]}×{img.shape[1]}",
            "src": _png_data_uri(img),
            "pixelated": max(img.shape) <= 40,  # draw small images with hard pixel edges
        })
    if case.reference is not None:
        ref = case.reference.ravel().astype(float)
        corr = float(np.corrcoef(ref, result.image.ravel().astype(float))[0, 1])
        out["reference"] = {"src": _png_data_uri(case.reference), "corr": round(corr, 3)}
    return out


def render_html(rows: list[dict], subtitle: str) -> str:
    data = json.dumps({"rows": rows}).replace("</", "<\\/")  # safe inside <script>
    return _TEMPLATE.replace("__SUBTITLE__", html.escape(subtitle)).replace("__DATA__", data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("images", nargs="*", type=Path, help="image files to visualise")
    parser.add_argument("--threshold", choices=["otsu", "adaptive"], default="otsu",
                        help="threshold method for your images (default: otsu)")
    parser.add_argument("--no-thicken", action="store_true", help="disable stroke thickening")
    parser.add_argument("--min-contrast", type=float, default=None,
                        help="blank-page contrast threshold (default: library default)")
    parser.add_argument("--mnist", type=int, default=0, metavar="N",
                        help="add N MNIST round-trip digits (downloads MNIST test set once)")
    parser.add_argument("--no-demo", action="store_true", help="skip the built-in demo inputs")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output HTML file (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})")
    parser.add_argument("--open", action="store_true", help="open the page in your browser")
    args = parser.parse_args(argv)

    options = PreprocessOptions(threshold=args.threshold, thicken=not args.no_thicken)
    if args.min_contrast is not None:
        options = PreprocessOptions(threshold=options.threshold, thicken=options.thicken,
                                    min_contrast=args.min_contrast)

    cases = file_cases(args.images, options)
    if not args.images and not args.no_demo:
        cases += demo_cases()
    cases += mnist_cases(args.mnist) if args.mnist else []
    if not cases:
        parser.error("nothing to show: pass images, --mnist N, or drop --no-demo")

    rows = [run_case(c) for c in cases]

    # Terminal summary, so a quick run doesn't require opening the page.
    for r in rows:
        mark = "ok  " if r["status"] == "ok" else "FAIL"
        detail = r["error"] or (f"corr {r['reference']['corr']:.3f}" if r["reference"] else "")
        print(f"[{mark}] {r['title']}" + (f" · {detail}" if detail else ""))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subtitle = f"{len(rows)} inputs · {sum(r['status'] == 'fail' for r in rows)} failing"
    args.output.write_text(render_html(rows, subtitle), encoding="utf-8")
    print(f"\nWrote {args.output}")
    if args.open:
        webbrowser.open(args.output.resolve().as_uri())
    return 1 if any(r["status"] == "fail" for r in rows) else 0


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Digit Preprocessing Stages</title>
<style>
:root {
  --bg: #f3f4f6; --surface: #fff; --fg: #16191f; --muted: #5b6270; --line: #d9dce2;
  --accent: #1f6feb; --ok: #1a7f37; --bad: #cf222e; --chip-ok: #dafbe1; --chip-bad: #ffe1e1; --plate: #0b0d10;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1216; --surface: #171b21; --fg: #e6e9ee; --muted: #9aa3b2; --line: #2a313b;
  --accent: #6aa6ff; --ok: #4ac26b; --bad: #ff7b72; --chip-ok: #12301c; --chip-bad: #3a1618; --plate: #000;
  color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0f1216; --surface: #171b21; --fg: #e6e9ee; --muted: #9aa3b2; --line: #2a313b;
  --accent: #6aa6ff; --ok: #4ac26b; --bad: #ff7b72; --chip-ok: #12301c; --chip-bad: #3a1618; --plate: #000;
  color-scheme: dark }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.55 var(--sans); }
.wrap { max-width: 1240px; margin: 0 auto; padding: 28px 16px 56px; display: flex; flex-direction: column; gap: 18px; }
h1 { margin: 0; font-size: 28px; } h2 { margin: 0; font-size: 17px; }
.sub { color: var(--muted); margin: 4px 0 0; }
code, .mono { font-family: var(--mono); font-size: .9em; }
.case { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px;
  display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.case.fail { border-color: var(--bad); }
.head { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: baseline; }
.chip { font: 500 11px var(--mono); letter-spacing: .06em; text-transform: uppercase; padding: 2px 8px; border-radius: 99px; }
.chip.ok { background: var(--chip-ok); color: var(--ok); } .chip.fail { background: var(--chip-bad); color: var(--bad); }
.note { margin: 0; color: var(--muted); font-size: 14px; max-width: 90ch; }
.error { margin: 0; font-family: var(--mono); font-size: 13px; }
.strip { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 6px; }
figure { margin: 0; flex: 0 0 auto; width: 132px; display: flex; flex-direction: column; gap: 5px; }
.plate { width: 132px; height: 132px; background: var(--plate); border-radius: 6px; display: grid; place-items: center; overflow: hidden; }
.plate img { max-width: 100%; max-height: 100%; display: block; }
.plate img.px { width: 132px; height: 132px; image-rendering: pixelated; }
.name { font: 500 12px var(--mono); display: flex; justify-content: space-between; gap: 6px; }
.name small { color: var(--muted); font-weight: 400; }
.info { font-size: 11.5px; color: var(--muted); line-height: 1.4; }
.final .plate { outline: 2px solid var(--accent); outline-offset: 2px; }
.ref .plate { outline: 2px dashed var(--muted); outline-offset: 2px; }
.arrow { flex: 0 0 auto; margin-top: 56px; color: var(--muted); }
</style></head>
<body><div class="wrap">
<header><h1>Digit preprocessing stages</h1><p class="sub">__SUBTITLE__ · generated by <code>scripts/show_stages.py</code></p></header>
<main id="rows" style="display:flex;flex-direction:column;gap:16px"></main>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
const rows = JSON.parse(document.getElementById("data").textContent).rows;
const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
function figure(src, name, shape, info, px, cls) {
  const f = el("figure", cls || ""); const p = el("div", "plate"); const i = el("img", px ? "px" : "");
  i.src = src; i.alt = name; p.append(i);
  const n = el("div", "name"); n.append(el("span", null, name), el("small", null, shape));
  f.append(p, n); if (info) { const d = el("div", "info"); d.textContent = info; f.append(d); }
  return f;
}
const root = document.getElementById("rows");
for (const r of rows) {
  const card = el("article", "case " + r.status);
  const head = el("div", "head");
  head.append(el("h2", null, r.title), el("span", "chip " + r.status, r.status === "ok" ? "as expected" : "unexpected"));
  card.append(head, el("p", "note", r.note));
  if (r.error) card.append(el("p", "error", r.error));
  const strip = el("div", "strip");
  r.stages.forEach((s, k) => {
    if (k) strip.append(el("span", "arrow", "→"));
    strip.append(figure(s.src, s.name, s.shape, s.info, s.pixelated, s.name === "mnist_28x28" ? "final" : ""));
  });
  if (r.reference) {
    strip.append(el("span", "arrow", "vs"));
    strip.append(figure(r.reference.src, "MNIST original", "28×28", "correlation with our output: " + r.reference.corr.toFixed(3), true, "ref"));
  }
  if (r.stages.length) card.append(strip);
  root.append(card);
}
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
