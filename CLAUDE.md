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
- **Editing `bench/harness.py` or `llmserve/{model,sampling,request,config}.py` casually.** Every
  committed result records a content hash of those files plus everything the engine that produced it
  is built from — its own module, its base classes, and any engine it wraps — so a change invalidates
  all of them and `verify-numbers.py` fails until every runner is re-run and
  `scripts/render-scorecards.py` is re-run after it. That is 20 runners and roughly 40 minutes.
  Changing one engine module invalidates only the results whose engines inherit from or wrap it.
- **A runner whose defaults do not produce what the book cites.** `verify-numbers.py` stamps *every*
  file in `bench/results/`, not just the cited ones, because a default that drifted once left the
  cited files behind on every regeneration and nothing noticed.
- **Bare `pytest`.** Use `python3 -m pytest`, so tests run under the interpreter that has the
  project's dependencies; a standalone pytest has its own isolated environment.
- **`BASE_URL`.** This is a *project* site at `/llm-serving-from-scratch/`. The deploy workflow
  sets it from the Pages base path; without it every link 404s.

## Adding a dependency

Engine and book dependencies go in `requirements.txt`, pinned exactly, with the date verified in
the comment. Lint/test tooling goes in `requirements-dev.txt`, and the `ruff` pin there must
match `.pre-commit-config.yaml`.

## Chapter status

All 29 chapters and all 5 appendices are written, with measured figures. Two chapters are explicit
about hardware the default tier does not have: ch13 contains the paged-decode algorithm, its tests
and its arithmetic but no Triton kernel (a kernel cannot be verified without a GPU, and shipping an
unverified one would contradict the book's own standard), and ch17 proves the tensor-parallel split
correct on one device and computes the collective cost rather than timing it. Both say so in a
warning box. See PLAN.md §11 for release scoping and CHECKPOINTS.md for the tag scheme.
