#!/usr/bin/env python3
"""Guard against unstamped and stale benchmark numbers (PLAN.md §6.3).

Every figure in the book comes from a result file in ``bench/results/``, rendered into a fragment
under ``chapters/_generated/`` by ``scripts/render-scorecards.py``. This script enforces three
rules, so a wrong number fails the build rather than reaching a reader:

1. Every result a scorecard cites exists.
2. Every such result carries its stamps — model, hardware, versions, date. A performance number
   whose conditions are unknown cannot be checked by anyone, including us in six months.
3. No result predates the engine code it claims to measure.

Fragment freshness is checked separately by ``render-scorecards.py --check``, which both this
script and CI run.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.scorecards import CONDITIONS, SCORECARDS  # noqa: E402

RESULTS = ROOT / "bench" / "results"
CODE_DIRS = ["llmserve", "bench/harness.py"]
REQUIRED_STAMPS = ("model", "hardware", "generated_at", "versions", "summary")

#: A decimal followed by a multiplier or a unit is almost always a measured figure, and measured
#: figures belong in a generated fragment rather than in prose — regenerating results silently
#: invalidates anything hand-typed. Integers are allowed, so "2 x n_layers" in a formula is fine.
HARDCODED_FIGURE = re.compile(r"\b\d+\.\d+\s*(?:[x\u00d7]|ms\b|s\b|GB/s\b|tok/s\b)")


def git_epoch(path: str) -> int:
    """Last commit time for a path, or 0 if it has never been committed."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return int(result.stdout.strip() or 0)


def cited_results() -> set[str]:
    names = {name for rows in SCORECARDS.values() for _, name in rows}
    return names | set(CONDITIONS.values())


def main() -> int:
    problems: list[str] = []
    names = cited_results()

    for name in sorted(names):
        path = RESULTS / f"{name}.json"
        if not path.exists():
            problems.append(f"missing result: bench/results/{name}.json (cited by a scorecard)")
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{name}.json is not valid JSON: {exc}")
            continue
        for stamp in REQUIRED_STAMPS:
            if stamp not in payload:
                problems.append(f"{name}.json is missing the required '{stamp}' stamp")

    newest_code = max((git_epoch(d) for d in CODE_DIRS), default=0)
    if newest_code:
        for name in sorted(names):
            rel = f"bench/results/{name}.json"
            if not (RESULTS / f"{name}.json").exists():
                continue
            committed = git_epoch(rel)
            if committed and committed < newest_code:
                problems.append(
                    f"{name}.json predates the latest change to {', '.join(CODE_DIRS)} — "
                    "regenerate with `python -m bench.run_v01`"
                )

    for source in sorted((ROOT / "chapters").glob("*.md")):
        for n, line in enumerate(source.read_text().splitlines(), start=1):
            if line.lstrip().startswith(("|", ":", "*Conditions")):
                continue  # generated tables and their caption lines carry real figures
            for hit in HARDCODED_FIGURE.findall(line):
                problems.append(
                    f"{source.name}:{n} has the measured figure '{hit.strip()}' typed into prose — "
                    "put it in a generated fragment instead (PLAN.md §6.3)"
                )

    if problems:
        print("verify-numbers: FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"verify-numbers: OK ({len(names)} result file(s) cited, all stamped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
