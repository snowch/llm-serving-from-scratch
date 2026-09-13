"""Render committed benchmark results into the tables chapters show.

Imports nothing heavier than the standard library on purpose: chapters call this from executable
cells during the book build, and a build that has to import torch to print a table is a build
that breaks for unrelated reasons.

The rule this enforces is PLAN.md §6.3 — no number is ever typed into prose. If a figure appears
in this book, it came from a file in ``bench/results/`` that names the hardware, the model and
the library versions that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Scorecard columns, in the order the book reads them: latency first, then capacity.
COLUMNS: list[tuple[str, str, str]] = [
    ("ttft_p50", "TTFT p50", "s"),
    ("ttft_p95", "TTFT p95", "s"),
    ("itl_p50", "ITL p50", "s"),
    ("output_tokens_per_second", "Output tok/s", ""),
    ("goodput_per_second", "Goodput req/s", ""),
]


def load(name: str) -> dict:
    """Load one result file by name (with or without the .json suffix)."""
    path = RESULTS_DIR / (name if name.endswith(".json") else f"{name}.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Regenerate results with `python -m bench.run_v01`."
        )
    return json.loads(path.read_text())


def _fmt(value: object, unit: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    return f"{text}{unit}" if unit else text


def table(rows: list[tuple[str, str]], *, columns=COLUMNS) -> str:
    """Render a scorecard table.

    ``rows`` is a list of ``(label, result_name)``. The label is what the reader sees in the
    leftmost column, usually the chapter and engine that produced the row.
    """
    header = "| Configuration | " + " | ".join(c[1] for c in columns) + " |"
    divider = "|---" * (len(columns) + 1) + "|"
    lines = [header, divider]
    for label, name in rows:
        summary = load(name)["summary"]
        cells = [_fmt(summary.get(key), unit) for key, _, unit in columns]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def conditions(name: str) -> str:
    """One line naming the hardware, model and versions behind a result.

    Printed under every scorecard table. A performance number without its conditions is not a
    measurement, it is a rumour.
    """
    data = load(name)
    hw = data["hardware"]
    model = data.get("model", {})
    versions = data.get("versions", {})
    cond = data.get("conditions", {})
    params = model.get("params")
    params_text = f"{params:,} params" if isinstance(params, int) else "unknown size"
    return (
        f"{model.get('name', 'unknown model')} ({params_text}), "
        f"{hw.get('cpu_count', '?')}x {hw.get('machine', '?')} CPU, "
        f"torch {versions.get('torch', '?')}, "
        f"{cond.get('trace', 'unknown trace')}, "
        f"arrival rate {cond.get('rate_per_second', '?')}/s, "
        f"measured {data.get('generated_at', 'unknown date')}."
    )


def value(name: str, key: str):
    """Pull a single summary figure, for quoting inline in prose."""
    return load(name)["summary"][key]


def arithmetic_table(cached_result: str, naive_result: str, context_length: int = 96) -> str:
    """Render chapter 3's prediction-versus-measurement table.

    Imports the model config lazily so this module stays usable without the engine installed.
    Every predicted value is computed from the config; every measured value is read from a
    committed result file. Neither is typed in.
    """
    from llmserve.arithmetic import (
        cached_decode_flops,
        decode_bytes_per_step,
        kv_bytes_per_token,
        naive_decode_flops,
        prefill_flops_per_token,
    )
    from llmserve.config import REFERENCE_MODEL

    m = REFERENCE_MODEL

    weights = m.n_params * m.bytes_per_element
    per_step = decode_bytes_per_step(m, context_length, 1)
    cached_itl = load(cached_result)["summary"]["itl_p50"]
    naive_itl = load(naive_result)["summary"]["itl_p50"]
    tok_per_s = 1.0 / cached_itl
    flops_ratio = naive_decode_flops(m, 64, 32) / cached_decode_flops(m, 64, 32)
    measured_ratio = naive_itl / cached_itl

    rows = [
        ("Parameters", f"{m.n_params:,}", "computed from config"),
        ("Weight bytes (fp32)", f"{weights / 1e6:.2f} MB", "computed"),
        ("KV cache per token", f"{kv_bytes_per_token(m):,.0f} B", "computed"),
        ("FLOPs per token", f"{prefill_flops_per_token(m) / 1e6:.2f} MFLOP", "computed"),
        (
            f"Bytes read per decode step (ctx {context_length})",
            f"{per_step / 1e6:.2f} MB",
            f"weights are {100 * weights / per_step:.1f}% of it",
        ),
        ("Measured decode rate, one stream", f"{tok_per_s:.0f} tok/s", "measured"),
        ("Implied achieved bandwidth", f"{per_step * tok_per_s / 1e9:.2f} GB/s", "derived"),
        ("Predicted no-cache penalty", f"{flops_ratio:.1f}x", "from FLOPs alone"),
        ("Measured no-cache penalty", f"{measured_ratio:.2f}x", "measured"),
    ]
    lines = ["| Quantity | Value | Source |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(lines)


def memory_table(block_size: int = 16, n_blocks: int = 24) -> str:
    """Render chapter 8's memory comparison: reserve-max against paged, in one budget.

    Computed from the model config and the block geometry rather than measured, because it is
    arithmetic: how many sequences *fit*, not how fast they run.
    """
    from llmserve.arithmetic import kv_bytes_per_token
    from llmserve.config import REFERENCE_MODEL

    m = REFERENCE_MODEL
    slots = block_size * n_blocks
    per_token = kv_bytes_per_token(m)
    reserve_max = slots // m.max_seq_len
    typical = 96  # a prompt plus a short answer, the trace this book uses
    paged = slots // typical

    rows = [
        ("KV budget", f"{n_blocks} blocks x {block_size} tokens = {slots} slots"),
        ("KV bytes per token", f"{per_token:,.0f} B"),
        ("Budget in bytes", f"{slots * per_token / 1e6:.2f} MB"),
        ("Reserve-max: must reserve", f"{m.max_seq_len:,} slots per sequence"),
        ("Reserve-max: sequences that fit", f"{reserve_max}"),
        ("Paged: allocates on demand", f"~{typical} slots per sequence"),
        ("Paged: sequences that fit", f"~{paged}"),
    ]
    lines = ["| Quantity | Value |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(lines)
