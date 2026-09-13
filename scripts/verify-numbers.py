#!/usr/bin/env python3
"""Guard against stale benchmark numbers (PLAN.md §6.3).

Every number quoted in the book must come from a committed result file in ``bench/results/``.
This script enforces two rules:

1. A chapter that names a result file must name one that exists.
2. A result file must not be older than the engine code it claims to measure — otherwise a
   library upgrade or a code change can silently leave a stale figure in the published book.

Exits non-zero on any violation so CI fails loudly rather than publishing a wrong number.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
CODE_DIRS = ["llmserve", "bench"]

# Chapters reference results as: bench/results/<something>.json
REF = re.compile(r"bench/results/([\w.\-]+\.json)")


def git_epoch(path: str) -> int:
    """Last commit time for a path, or 0 if it has never been committed."""
    out = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return int(out.stdout.strip() or 0)


def main() -> int:
    problems: list[str] = []
    sources = sorted(ROOT.glob("chapters/*.md")) + sorted(ROOT.glob("appendices/*.md"))

    referenced: set[str] = set()
    for src in sources:
        for name in REF.findall(src.read_text()):
            referenced.add(name)
            if not (RESULTS / name).exists():
                problems.append(f"{src.relative_to(ROOT)} references missing result: {name}")

    newest_code = max((git_epoch(d) for d in CODE_DIRS), default=0)
    for name in sorted(referenced):
        target = RESULTS / name
        if not target.exists():
            continue
        try:
            payload = json.loads(target.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{name} is not valid JSON: {exc}")
            continue
        for field in ("model", "hardware", "generated_at", "versions"):
            if field not in payload:
                problems.append(f"{name} is missing the required '{field}' stamp")
        if newest_code and git_epoch(f"bench/results/{name}") < newest_code:
            problems.append(
                f"{name} predates the latest change to {'/'.join(CODE_DIRS)} — regenerate it "
                f"with scripts/run-benchmarks.sh"
            )

    if problems:
        print("verify-numbers: FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"verify-numbers: OK ({len(referenced)} result file(s) referenced)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
