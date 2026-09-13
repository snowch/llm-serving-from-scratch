---
title: "Environment Setup"
short_title: "Appendix B"
---

(appendix-b)=
# Appendix B · Environment Setup

Do this before {ref}`ch01`. It takes a few minutes and tells you exactly which parts of the book
your machine can run.

## The short version

```bash
git clone https://github.com/snowch/llm-serving-from-scratch
cd llm-serving-from-scratch
python3 -m pip install -r requirements.txt -r requirements-dev.txt
python3 scripts/verify-setup.py
```

The last command prints a line per check and ends by naming the tiers you can complete. If it says
*Ready*, start reading.

## The three tiers

The book is built so that its argument survives on the cheapest hardware available, and only two
chapters need more.

| Tier | Hardware | What it runs |
|---|---|---|
| **1** | Any laptop, CPU only | Everything except {ref}`ch13` and {ref}`ch17`. Every committed number in this book was measured here. |
| **2** | One GPU (24 GB consumer card, or a cloud L4/A10G) | Adds {ref}`ch13`'s Triton kernel and lets you re-measure Parts II–III at realistic scale. |
| **3** | Two or more GPUs, rented hourly | Adds {ref}`ch17`'s tensor and pipeline parallelism. |

**Tier 1 is the default and not a compromise.** The reference model is built from code with seeded
random weights, so it runs anywhere and needs no download; and every mechanism in Parts II–VII
shows its effect at that scale, because the effects are structural rather than a property of model
size. What Tier 1 cannot show is anything that depends on a real memory hierarchy or a real
interconnect, which is precisely the two chapters it excludes.

Tier 3's cost, if you rent: an {ref}`ch17` reproduction is under an hour of work on a two-GPU
instance, so it is a single-digit-dollar exercise at current spot prices. Check before you start;
that sentence ages badly and this book will not know.

## Pinned versions

| What | Pinned in | Version | Verified |
|---|---|---|---|
| PyTorch | `requirements.txt` | 2.14.0 | 2026-09-13 |
| NumPy | `requirements.txt` | 2.4.6 | 2026-09-13 |
| ruff | `requirements-dev.txt` | 0.16.7 | 2026-09-13 |
| pytest | `requirements-dev.txt` | 9.1.1 | 2026-09-13 |
| pre-commit | `requirements-dev.txt` | 4.6.2 | 2026-09-13 |
| MyST (`mystmd`) | `package.json` | 1.10.1 | 2026-09-13 |

Two things are deliberate here.

**The book builder is pinned in npm, not pip.** `mystmd` is a Node package; pinning it in
`requirements.txt` as well would create two sources of truth that drift. Install it with
`npm install -g mystmd@1.10.1` — or read the deployed site and skip it entirely, since nothing in
the engine needs it.

**The dependency list is very short**, and it is short because the engine is written from scratch.
There is no FastAPI, no `transformers`, no plotting library. If a pin is not imported by something
in this repository, it does not belong in the file — a version claim nobody checks is worse than no
claim.

## Running the checks

Everything CI runs is in one script, so local and CI cannot drift:

```bash
./scripts/ci-check.sh
```

That is lint, format, the test suite, the result-stamp verifier, the scorecard freshness check, and
a strict MyST build. Run it before you commit anything.

Two traps worth knowing about, both of which cost a debugging session before they became rules:

- **Use `python3 -m pytest`, never bare `pytest`.** A `pytest` installed by `pipx` or `uv tool` runs
  in its own isolated environment and cannot import `torch` or `llmserve`. The module form
  guarantees the interpreter that has your dependencies is the one running the tests.
- **`myst build --html` downloads a site theme before it parses anything.** In an environment that
  blocks the template registry, that build aborts having validated nothing at all — while looking
  like a failure you can ignore. `ci-check.sh` therefore runs a content build *without* `--html`
  first, and requires both a page count and zero warnings before trusting it.

## Model weights

There are none in this repository, and none are downloaded on the Tier 1 path.

This is a design decision with a cost, and it is worth being explicit about both halves. The cost is
that the reference model is small and its outputs are not meaningful text, so nothing in this book
can show you a quality result from the serving model itself — where quality matters ({ref}`ch14`,
{ref}`ch15`, {ref}`ch19`) the book trains a small model on a synthetic corpus first, and says so
each time.

What it buys is that every measurement in the book reproduces on any machine, in any network
environment, with no account, no token and no licence acceptance. A book whose first chapter fails
behind a corporate proxy is a book nobody finishes.

To serve a real model instead, point {ref}`ch28`'s harness at any OpenAI-compatible endpoint — the
comparison methodology there is deliberately engine-agnostic, and that is the intended path from
this engine to a production one.
