# CLAUDE.md

Project notes for AI assistants working on this book.

## What this is

*LLM Serving from Scratch* — an online book that builds an LLM inference engine incrementally.
Read **PLAN.md** first: it holds the outline, the settled decisions, and the conventions. Read
**AUTHORING_GUIDE.md** before writing or editing a chapter.

## Build

Jupyter Book 2, whose CLI is `mystmd`. Pinned in `package.json` (npm), **not** in
`requirements.txt` — one source of truth.

```bash
myst start              # live preview
./scripts/ci-check.sh   # ruff + pytest + verify-numbers + myst build
```

## The two invariants

These are what the book's credibility rests on. Do not work around them.

1. **No code pasted into prose.** Use `{literalinclude}` with `:start-at:` / `:end-before:`
   text anchors (never line numbers), pointing at real files in `llmserve/`.
2. **No numbers typed into prose.** Every figure comes from a stamped JSON file in
   `bench/results/`, declared in `bench/scorecards.py`, rendered to `chapters/_generated/` by
   `scripts/render-scorecards.py`, and pulled in with `{include}`. Chapters contain **no
   executable cells** — see AUTHORING_GUIDE.md for why. Both `verify-numbers.py` and
   `render-scorecards.py --check` run in CI.

## Things that will break the build

- **Assuming model weights can be downloaded.** They cannot in every environment. The reference
  model is built from code with seeded random weights (`llmserve/model.py`) precisely so the book
  runs anywhere; do not put a Hugging Face download on the Tier 1 path.
- **Adding executable cells to chapters.** The book build is pure markdown and must stay that
  way; render figures to a fragment instead.
- **Unpinned `mystmd`.** Always install the version from `package.json`.
- **Editing `bench/harness.py` casually.** It is a dependency of every committed result; a change
  invalidates all of them, and `verify-numbers.py` fails the build until they are regenerated with
  `python -m bench.run_v01`.
- **Bare `pytest`.** Use `python3 -m pytest`, so tests run under the interpreter that has the
  project's dependencies; a standalone pytest has its own isolated environment.
- **`BASE_URL`.** This is a *project* site at `/llm-serving-from-scratch/`. The deploy workflow
  sets it from the Pages base path; without it every link 404s.

## Adding a dependency

Engine and book dependencies go in `requirements.txt`, pinned exactly, with the date verified in
the comment. Lint/test tooling goes in `requirements-dev.txt`, and the `ruff` pin there must
match `.pre-commit-config.yaml`.

## Chapter status

Chapters 1-3 are written, with measured figures. The remaining 26 chapters and all 5 appendices
are stubs carrying the template and per-chapter guidance; `[DRAFT]` in a title means outline only. See PLAN.md §11 for what ships in which release, and
CHECKPOINTS.md for the tag scheme.
