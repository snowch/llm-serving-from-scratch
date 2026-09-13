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
   `bench/results/`, rendered by an executable cell. `scripts/verify-numbers.py` enforces this
   and runs in CI.

## Things that will break the build

- **Executable cells needing a GPU or large model weights.** CI has neither. GPU work is a
  static code block plus a committed result file.
- **Unpinned `mystmd`.** Always install the version from `package.json`.
- **Editing `bench/harness.py` casually.** It is a dependency of every committed result; a
  change invalidates all of them.
- **`BASE_URL`.** This is a *project* site at `/llm-serving-from-scratch/`. The deploy workflow
  sets it from the Pages base path; without it every link 404s.

## Adding a dependency

Engine and book dependencies go in `requirements.txt`, pinned exactly, with the date verified in
the comment. Lint/test tooling goes in `requirements-dev.txt`, and the `ruff` pin there must
match `.pre-commit-config.yaml`.

## Chapter status

All 29 chapters and 5 appendices exist as stubs carrying the template and per-chapter guidance.
`[DRAFT]` in a title means outline only. See PLAN.md §11 for what ships in which release, and
CHECKPOINTS.md for the tag scheme.
