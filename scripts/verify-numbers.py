#!/usr/bin/env python3
"""Guard against unstamped, stale and hand-typed numbers (PLAN.md §6.3).

Every figure in the book comes from a result file in ``bench/results/``, rendered into a fragment
under ``chapters/_generated/`` by ``scripts/render-scorecards.py``. Four rules, so a wrong number
fails the build rather than reaching a reader:

1. Every result a scorecard cites exists — including the ones a *derived* fragment computes
   from, which are declared in ``bench.scorecards.DERIVED_SOURCES`` precisely so they cannot
   escape these checks by not appearing in a plain table.
2. Every such result carries its stamps — model, hardware, versions, date. A performance number
   whose conditions are unknown cannot be checked by anyone, including us in six months.
3. Every result was produced by the code that is checked in. This is a **content hash** over the
   shared core plus the engine module that generated it. An earlier version compared commit
   times, which cannot tell an unrelated new file apart from a change to the thing being measured
   and was wrong in both directions: it marked every result stale when any file under
   ``llmserve/`` changed, and it could not see a change committed alongside the results.
4. No measured figure is typed into chapter prose, where regenerating results would silently
   invalidate it.

Fragment freshness is checked separately by ``render-scorecards.py --check``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.harness import code_fingerprint  # noqa: E402
from bench.scorecards import CONDITIONS, DERIVED_SOURCES, SCORECARDS  # noqa: E402

RESULTS = ROOT / "bench" / "results"
REQUIRED_STAMPS = ("model", "hardware", "generated_at", "versions", "summary")

#: A decimal followed by a multiplier or a unit is almost always a measured figure, and measured
#: figures belong in a generated fragment. Integers are allowed, so "2 x n_layers" in a formula
#: is fine.
HARDCODED_FIGURE = re.compile(r"\b\d+\.\d+\s*(?:[x×]|ms\b|s\b|GB/s\b|tok/s\b)")


def cited_results() -> set[str]:
    names = {name for rows in SCORECARDS.values() for _, name in rows}
    names |= {name for sources in DERIVED_SOURCES.values() for name in sources}
    return names | set(CONDITIONS.values())


def check_results(problems: list[str]) -> None:
    """Every cited result must exist, and *every* result must be current.

    Checking only the cited ones was not enough. A runner whose default arguments no longer produce
    the files the book cites leaves those files behind, unregenerated and unnoticed — which is
    exactly what happened: ``run_v01`` defaulted to a set of arrival rates that did not include the
    ones the chapters cite, so a full regeneration silently skipped them and they rotted for
    several commits. Stamping every file in the directory costs nothing and closes that door.
    """
    for name in sorted(cited_results()):
        if not (RESULTS / f"{name}.json").exists():
            problems.append(f"missing result: bench/results/{name}.json (cited by a scorecard)")

    for path in sorted(RESULTS.glob("*.json")):
        name = path.stem
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{name}.json is not valid JSON: {exc}")
            continue

        for stamp in REQUIRED_STAMPS:
            if stamp not in payload:
                problems.append(f"{name}.json is missing the required '{stamp}' stamp")

        recorded = payload.get("code_fingerprint")
        if recorded is None:
            problems.append(f"{name}.json has no code_fingerprint — regenerate it")
        elif recorded != code_fingerprint(payload.get("code_sources")):
            problems.append(
                f"{name}.json was produced by different code than is checked in — regenerate "
                f"with the runner that writes it (`grep -rl {name} bench/run_*.py`), or delete it "
                "if nothing produces it any more"
            )


def check_prose(problems: list[str]) -> None:
    for source in sorted((ROOT / "chapters").glob("*.md")):
        for n, line in enumerate(source.read_text().splitlines(), start=1):
            # Generated tables and their caption lines legitimately carry real figures.
            if line.lstrip().startswith(("|", ":", "*Conditions")):
                continue
            for hit in HARDCODED_FIGURE.findall(line):
                problems.append(
                    f"{source.name}:{n} has the measured figure '{hit.strip()}' typed into prose "
                    "— put it in a generated fragment instead (PLAN.md §6.3)"
                )


def main() -> int:
    problems: list[str] = []
    check_results(problems)
    check_prose(problems)

    if problems:
        print("verify-numbers: FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"verify-numbers: OK ({len(cited_results())} result file(s) cited, all stamped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
