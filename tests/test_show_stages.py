"""Smoke tests for scripts/show_stages.py (the visual-check script)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# scripts/ is not a package, so load the script from its path. It must be registered in
# sys.modules before running, because @dataclass looks its module up there.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "show_stages.py"
_spec = importlib.util.spec_from_file_location("show_stages", _SCRIPT)
show_stages = importlib.util.module_from_spec(_spec)
sys.modules["show_stages"] = show_stages
_spec.loader.exec_module(show_stages)


def test_demo_cases_all_behave_as_expected(tmp_path):
    """Every built-in demo input ends as documented (digit found, or NoInkFoundError for blanks)."""
    out = tmp_path / "stages.html"
    exit_code = show_stages.main(["-o", str(out)])
    assert exit_code == 0  # non-zero if any case ended differently than expected
    page = out.read_text()
    assert "mnist_28x28" in page  # digit cases rendered their final stage
    assert "looks blank" in page  # the blank-paper cases report the guard's message


def test_user_images_and_unreadable_file(tmp_path):
    good = tmp_path / "seven.png"
    img = np.full((200, 200), 230, np.uint8)
    img[40:160, 95:105] = 20
    Image.fromarray(img).save(good)
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")

    out = tmp_path / "stages.html"
    exit_code = show_stages.main([str(good), str(bad), "-o", str(out)])
    assert exit_code == 1  # the unreadable file is reported as a failure…
    page = out.read_text()
    assert "seven.png" in page and "UnidentifiedImageError" in page  # …without aborting the report
