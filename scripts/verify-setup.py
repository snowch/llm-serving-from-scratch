#!/usr/bin/env python3
"""Check this machine before chapter 1, and say which tiers it can complete.

Run before starting the book:

    python3 scripts/verify-setup.py

The point is to fail here, with a clear message, rather than three chapters in with a stack trace.
Every check names what it wants and what to do about it, and a missing GPU is reported as a tier
you cannot reach rather than as an error — the whole book's default path is CPU-only by design.
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


def _line(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def check_python(problems: list[str]) -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 11):
        problems.append(f"Python {major}.{minor} is too old; the book needs 3.11 or newer")
        _line(FAIL, f"Python {major}.{minor} (need >= 3.11)")
    else:
        _line(OK, f"Python {major}.{minor} on {platform.machine()}")


def _pinned() -> dict[str, str]:
    """Everything requirements.txt pins, as name -> version."""
    pins = {}
    for raw in (ROOT / "requirements.txt").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-+]+)", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def check_dependencies(problems: list[str]) -> None:
    """Installed versions against the pins. A mismatch is a warning, not a failure.

    The book's committed results were measured against the pins, so a different version can move a
    number without breaking anything. That is worth saying out loud and not worth blocking on.
    """
    for name, wanted in sorted(_pinned().items()):
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # Only the engine's own imports are required to start the book.
            if name in {"torch", "numpy"}:
                problems.append(f"{name} is not installed (pip install -r requirements.txt)")
                _line(FAIL, f"{name} missing, and chapter 1 needs it")
            else:
                _line(WARN, f"{name} missing (needed later; pip install -r requirements.txt)")
            continue
        if found == wanted:
            _line(OK, f"{name} {found}")
        else:
            _line(WARN, f"{name} {found} installed, {wanted} pinned — figures may differ slightly")


def check_reference_model(problems: list[str]) -> None:
    """Build the model the book measures. No download, by design — see PLAN.md §5."""
    sys.path.insert(0, str(ROOT))
    try:
        from llmserve.config import REFERENCE_MODEL
        from llmserve.model import build_model
    except ImportError as exc:
        problems.append(f"cannot import llmserve ({exc}) — run from the repository root")
        _line(FAIL, f"cannot import llmserve: {exc}")
        return

    model = build_model(REFERENCE_MODEL)
    params = sum(p.numel() for p in model.parameters())
    _line(OK, f"reference model builds: {params:,} parameters, no download needed")


def check_tiers() -> None:
    """Say which hardware tiers this machine can complete."""
    _line(OK, "Tier 1 (CPU): available — this is the book's default path")

    try:
        import torch
    except ImportError:
        _line(WARN, "Tier 2/3: cannot tell without torch installed")
        return

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        _line(OK, f"Tier 2 (1 GPU): available — {count}x {name}, {memory:.0f} GB")
        if count >= 2:
            _line(OK, f"Tier 3 (multi-GPU): available — {count} devices, ch17 is reproducible")
        else:
            _line(WARN, "Tier 3 (multi-GPU): one device only; ch17 needs at least two")
    else:
        _line(WARN, "Tier 2 (1 GPU): no CUDA device — ch13 and ch17 are read-only for you")
        _line(WARN, "Tier 3 (multi-GPU): unavailable")


def check_book_build() -> None:
    import shutil

    if shutil.which("myst"):
        _line(OK, "myst found — you can build the book locally")
    else:
        version = "see package.json"
        package = ROOT / "package.json"
        if package.exists():
            match = re.search(r'"mystmd"\s*:\s*"([^"]+)"', package.read_text())
            if match:
                version = match.group(1)
        _line(WARN, f"myst not installed (npm install -g mystmd@{version}) — only needed to build")


def main() -> int:
    print("Checking this machine against what the book needs.\n")
    problems: list[str] = []
    check_python(problems)
    check_dependencies(problems)
    check_reference_model(problems)
    check_book_build()
    print()
    check_tiers()

    print()
    if problems:
        print("Not ready yet:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Ready. Start at chapter 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
