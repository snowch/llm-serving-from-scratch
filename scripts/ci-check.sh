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
pytest tests/ -q

echo "== benchmark number freshness =="
python3 scripts/verify-numbers.py

echo "== myst build (strict) =="
# --strict turns content warnings (broken cross-reference, unresolved citation) into a failure.
# Needs network access to the MyST template registry at api.mystmd.org to fetch the site theme;
# content is parsed and validated first, so content errors are reported even if that fetch
# fails. --execute is left to the deploy workflow.
if command -v myst >/dev/null 2>&1; then
  myst build --html --strict
elif [ -n "${CI:-}" ]; then
  # A check that silently skips itself is not a check. CI installs myst, so absence here means
  # the workflow is misconfigured, and we would rather fail loudly than publish a broken link.
  echo "ERROR: myst is not installed and CI must not skip the book build." >&2
  exit 1
else
  echo "  myst not installed; skipping locally."
  echo "  install with: npm install -g \"mystmd@$(node -p "require('./package.json').devDependencies.mystmd")\""
fi

echo
echo "All checks passed."
