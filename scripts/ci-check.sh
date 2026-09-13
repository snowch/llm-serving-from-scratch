#!/usr/bin/env bash
# The exact checks CI runs. Run this before pushing.
#
# CI invokes this same script, so the two can never drift (PLAN.md §9, gotcha 2).
set -euo pipefail

cd "$(dirname "$0")/.."

PY_PATHS=(llmserve bench tests scripts)

echo "== ruff lint =="
ruff check "${PY_PATHS[@]}"

echo "== ruff format =="
ruff format --check "${PY_PATHS[@]}"

echo "== pytest (CPU tier only) =="
# `python3 -m pytest`, not bare `pytest`: the module form guarantees the tests run under the same
# interpreter that has the project's dependencies. A standalone pytest (pipx, uv tool) has its own
# isolated environment and cannot import torch or llmserve.
python3 -m pytest tests/ -q

echo "== benchmark result stamps =="
python3 scripts/verify-numbers.py

echo "== scorecard fragments up to date =="
python3 scripts/render-scorecards.py --check

echo "== myst content build (strict) =="
if ! command -v myst >/dev/null 2>&1; then
  if [ -n "${CI:-}" ]; then
    # A check that silently skips itself is not a check. CI installs myst, so absence here means
    # the workflow is misconfigured, and we would rather fail loudly than publish a broken link.
    echo "ERROR: myst is not installed and CI must not skip the book build." >&2
    exit 1
  fi
  echo "  myst not installed; skipping locally."
  echo "  install with: npm install -g \"mystmd@$(node -p "require('./package.json').devDependencies.mystmd")\""
  echo
  echo "All checks passed."
  exit 0
fi

# Content build WITHOUT --html. This matters: with --html, MyST downloads the site theme before
# it parses anything, so in an environment that blocks the template registry the build aborts
# having validated nothing at all. Without it, every page is parsed first and only the final
# site assembly fails — so the log still tells us whether the content is sound.
log=$(mktemp)
myst build --strict > "$log" 2>&1 || true

# Two conditions, both required. The page count proves parsing actually happened, so a build that
# died early can never be mistaken for a clean one; the warning check is the actual verdict.
if ! grep -qE "Built [0-9]+ pages" "$log"; then
  echo "ERROR: myst did not parse any pages — the build failed before validating content." >&2
  tail -25 "$log" >&2
  rm -f "$log"
  exit 1
fi
if grep -qE "⚠|⛔" "$log"; then
  echo "ERROR: content warnings (broken reference, citation or literalinclude anchor):" >&2
  grep -E "⚠|⛔" "$log" >&2
  rm -f "$log"
  exit 1
fi
echo "  $(grep -oE 'Built [0-9]+ pages' "$log" | tail -1), no warnings"
rm -f "$log"

echo "== myst themed HTML build =="
# Needs the MyST template registry and GitHub. Mandatory in CI; locally a blocked host says
# nothing about the book, since the content build above already validated everything authored.
if [ -n "${CI:-}" ]; then
  myst build --html --strict
  echo "  themed build OK"
elif myst build --html --strict > /dev/null 2>&1; then
  echo "  themed build OK"
else
  echo "  SKIPPED: cannot reach the MyST template registry from this environment."
fi

echo
echo "All checks passed."
