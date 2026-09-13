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
            f"{path} does not exist. Generate it with the bench/run_*.py that writes it — "
            f"`grep -rl {name} bench/run_*.py` names the one."
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
        f"{_rate_text(cond)}, "
        f"measured {data.get('generated_at', 'unknown date')}."
    )


def _rate_text(conditions: dict) -> str:
    """How the arrival rate is stated, which is not always one number.

    A result that sweeps the arrival rate has no single rate, and printing "arrival rate ?/s" under
    it is worse than saying nothing — it reads like a missing field rather than a deliberate sweep.
    """
    if "rates" in conditions:
        rates = conditions["rates"]
        joined = ", ".join(f"{r:g}" for r in rates)
        return f"arrival rates {joined}/s"
    if conditions.get("rate_per_second") is not None:
        return f"arrival rate {conditions['rate_per_second']}/s"
    return "no arrival process"


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


def chunk_cost_table(name: str = "chunk-cost-tier1") -> str:
    """Render the cost of splitting one prefill into several passes."""
    data = load(name)
    rows = data["summary"]["passes"]
    lines = ["| Passes | Tokens per pass | Total time |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['passes']} | {row['chunk']} | {row['ms']} ms |")
    return "\n".join(lines)


def handoff_table() -> str:
    """What a prefill-to-decode KV handoff costs at different model and context sizes.

    Pure arithmetic from the chapter 3 formula, deliberately not measured: our in-process copy
    says nothing useful about a network transfer, and this is the number that actually decides
    whether disaggregation is viable.
    """
    from llmserve.arithmetic import kv_bytes_per_token
    from llmserve.config import REFERENCE_MODEL, ModelConfig

    llama8b = ModelConfig(
        vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
    )
    cases = [
        ("This book's model, 256 tokens", REFERENCE_MODEL, 256),
        ("8B GQA, 2k context", llama8b, 2048),
        ("8B GQA, 32k context", llama8b, 32768),
    ]
    links = [("10 GbE", 1.25e9), ("100 GbE", 12.5e9), ("NVLink (~400 GB/s)", 4.0e11)]

    header = "| Handoff | KV size | " + " | ".join(name for name, _ in links) + " |"
    lines = [header, "|---" * (len(links) + 2) + "|"]
    for label, model, context in cases:
        payload = kv_bytes_per_token(model) * context
        times = [f"{1000 * payload / bw:.1f} ms" for _, bw in links]
        size = f"{payload / 1e6:.1f} MB" if payload < 1e9 else f"{payload / 1e9:.2f} GB"
        lines.append(f"| {label} | {size} | " + " | ".join(times) + " |")
    return "\n".join(lines)


def attention_footprint_table() -> str:
    """What the KV-head choice costs at production scale, and what it buys in concurrency.

    Arithmetic rather than measurement: our model is too small for the KV cache to be the binding
    constraint, and this is precisely the term that becomes binding as models and contexts grow.
    """
    from llmserve.arithmetic import kv_bytes_per_token
    from llmserve.config import ModelConfig

    budget = 40 * 1024**3  # KV budget on an 80 GB accelerator after weights
    context = 32768
    rows = []
    for label, n_kv in [
        ("MHA (32 KV heads)", 32),
        ("GQA 4:1 (8 KV heads)", 8),
        ("MQA (1 KV head)", 1),
    ]:
        config = ModelConfig(
            vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=n_kv, head_dim=128, dtype="fp16"
        )
        per_token = kv_bytes_per_token(config)
        per_sequence = per_token * context
        rows.append(
            (
                label,
                f"{per_token / 1024:.0f} KiB",
                f"{per_sequence / 1e9:.2f} GB",
                f"{int(budget // per_sequence)}",
            )
        )

    lines = [
        f"| Attention | KV per token | Per 32k sequence | Sequences in {budget // 1024**3} GB |",
        "|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return "\n".join(lines)


def score_matrix_table() -> str:
    """Bytes a materialised attention score matrix occupies, per layer, at several contexts."""
    from llmserve.attention import attention_matrix_bytes

    lines = ["| Context | Score matrix, per layer (32 heads, fp32) |", "|---|---|"]
    for context in (512, 2048, 8192, 32768):
        payload = attention_matrix_bytes(context, context, 32)
        size = f"{payload / 1e6:.0f} MB" if payload < 1e9 else f"{payload / 1e9:.1f} GB"
        lines.append(f"| {context:,} | {size} |")
    return "\n".join(lines)


def quantisation_table(name: str = "quant-tier1") -> str:
    """Chapter 14's three axes in one table: quality, memory, speed."""
    data = load(name)
    rows = data["summary"]["weights"]
    baseline = rows[0]["perplexity"]
    base_bytes = rows[0]["linear_bytes"]
    lines = [
        "| Scheme | Perplexity | Change | Linear weights | Decode tok/s |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        delta = 100 * (row["perplexity"] - baseline) / baseline
        change = "baseline" if row is rows[0] else f"{delta:+.2f}%"
        size = f"{row['linear_bytes'] / 1e6:.2f} MB"
        if row is not rows[0]:
            size += f" ({base_bytes / row['linear_bytes']:.1f}x smaller)"
        lines.append(
            f"| {row['label']} | {row['perplexity']:.4f} | {change} | {size} | "
            f"{row['decode_tok_per_s']} |"
        )
    return "\n".join(lines)


def kv_quantisation_table(name: str = "quant-tier1") -> str:
    """What quantising the cache costs in quality, and saves in footprint."""
    data = load(name)
    baseline = data["conditions"]["baseline_perplexity"]
    lines = ["| KV precision | Perplexity | Change | KV per token |", "|---|---|---|---|"]
    lines.append(f"| fp32 (baseline) | {baseline:.4f} | baseline | 3,072 B |")
    for row in data["summary"]["kv_cache"]:
        delta = 100 * (row["perplexity"] - baseline) / baseline
        lines.append(
            f"| INT{row['bits']} | {row['perplexity']:.4f} | {delta:+.2f}% | "
            f"{row['kv_bytes_per_token']:,.0f} B |"
        )
    return "\n".join(lines)


def outlier_table(name: str = "quant-tier1") -> str:
    """The mechanism behind per-channel scaling, on a matrix that actually has an outlier."""
    data = load(name)["summary"]["outliers"]
    return "\n".join(
        [
            "| Scale choice | Mean error on the other channels |",
            "|---|---|",
            f"| Per-tensor (one scale) | {data['per_tensor_error']:.5f} |",
            f"| Per-channel | {data['per_channel_error']:.5f} |",
            f"| **Per-tensor is worse by** | **{data['ratio']:.0f}x** |",
        ]
    )


def speculation_table(name: str = "speculative-tier1") -> str:
    """Acceptance economics: what proposing more actually buys."""
    rows = load(name)["summary"]["sweep"]
    lines = [
        "| Proposed per round (k) | Acceptance rate | Tokens per target pass | Output identical |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['k']} | {row['acceptance_rate']:.0%} | {row['tokens_per_round']:.2f} | "
            f"{'yes' if row['identical_to_greedy'] else 'NO'} |"
        )
    return "\n".join(lines)


def speculation_distribution_table(name: str = "speculative-tier1") -> str:
    """The correctness claim, as a falsifiable measurement rather than an assertion."""
    d = load(name)["summary"]["distribution"]
    lines = [
        f"| Acceptance rule | P(drafted token) over {d['samples']:,} samples | Distance from target |",
        "|---|---|---|",
        f"| Target model, sampled directly | {d['target_probability']:.5f} | — |",
        f"| Residual rule (correct) | {d['measured_correct_rule']:.5f} | {d['correct_rule_sigma']} SE |",
        f"| Resample from p_target (bug) | {d['measured_biased_rule']:.5f} | {d['biased_rule_sigma']} SE |",
        f"| *Theory predicts for the bug* | *{d['predicted_if_biased']:.5f}* | *—* |",
    ]
    return "\n".join(lines)


def constrained_validity_table(name: str = "constrained-tier1") -> str:
    """Does constraining work? Split by whether generation was allowed to finish."""
    v = load(name)["summary"]["validity"]
    trials = v["trials"]
    completed = v["constrained_completed"]
    lines = ["| Generation | Valid JSON | Rate |", "|---|---|---|"]
    lines.append(
        f"| Unconstrained | {v['unconstrained_valid']}/{trials} | {v['unconstrained_valid'] / trials:.0%} |"
    )
    lines.append(
        f"| Constrained, all attempts | {v['constrained_valid']}/{trials} | {v['constrained_valid'] / trials:.0%} |"
    )
    lines.append(
        f"| Constrained, reached end state | {v['constrained_valid']}/{completed} | "
        f"{v['constrained_valid'] / max(completed, 1):.0%} |"
    )
    return "\n".join(lines)


def mask_cost_table(name: str = "constrained-tier1") -> str:
    """What building a mask costs, by vocabulary size.

    The second column is the cost of rebuilding it on every step of a 1,000-token generation:
    one mask in microseconds means a thousand of them in milliseconds.
    """
    rows = load(name)["summary"]["mask_cost"]["per_mask"]
    lines = [
        "| Vocabulary | One mask | Rebuilt every step, 1,000 tokens |",
        "|---|---|---|",
    ]
    for row in rows:
        per_mask = row["microseconds_per_mask"]
        # per_mask microseconds x 1,000 steps = per_mask milliseconds.
        lines.append(f"| {row['vocab_size']:,} | {per_mask:.0f} µs | {per_mask:.0f} ms |")
    return "\n".join(lines)


def routing_table() -> str:
    """Chapter 18: what each routing policy buys and what it costs in balance."""
    rows = [
        ("Round-robin", "router-round-robin-tier1"),
        ("Least outstanding tokens", "router-least-tokens-tier1"),
        ("Prefix affinity", "router-prefix-affinity-tier1"),
    ]
    lines = [
        "| Policy | TTFT p50 | Output tok/s | Goodput req/s | Prefix reuse | Load imbalance |",
        "|---|---|---|---|---|---|",
    ]
    for label, name in rows:
        data = load(name)
        s, c = data["summary"], data["conditions"]
        lines.append(
            f"| {label} | {s['ttft_p50']}s | {s['output_tokens_per_second']} | "
            f"{s['goodput_per_second']} | {c['token_reuse']:.0%} | {c['imbalance']:.2f}x |"
        )
    return "\n".join(lines)


def cold_start_table() -> str:
    """What a replica costs to start, before it can serve one token.

    Arithmetic rather than measurement, for the same reason as chapter 11's handoff table: this
    book's model loads instantly and says nothing about a real one, and the quantity that decides
    whether autoscaling can work at all is how long the *weights* take to arrive.
    """
    links = [
        ("Local NVMe (2 GB/s)", 2.0e9),
        ("10 GbE (1.25 GB/s)", 1.25e9),
        ("Object store (400 MB/s)", 4.0e8),
    ]
    cases = [
        ("8B, fp16", 8e9, 2),
        ("8B, INT4", 8e9, 0.5),
        ("70B, fp16", 70e9, 2),
        ("70B, INT4", 70e9, 0.5),
    ]
    header = "| Weights | Size | " + " | ".join(name for name, _ in links) + " |"
    lines = [header, "|---" * (len(links) + 2) + "|"]
    for label, params, bytes_per_param in cases:
        payload = params * bytes_per_param
        times = [f"{payload / bw:.0f} s" for _, bw in links]
        lines.append(f"| {label} | {payload / 1e9:.0f} GB | " + " | ".join(times) + " |")
    return "\n".join(lines)


def lora_memory_table() -> str:
    """What a tenant's fine-tune costs, as a full copy and as an adapter.

    Arithmetic, for the same reason as chapter 11's handoff table: the argument for LoRA is
    strongest at a scale this book's model cannot reach, and the shapes are all that is needed.
    """
    from llmserve.config import BYTES_PER_DTYPE, ModelConfig
    from llmserve.lora import LoRAConfig, adapter_bytes

    llama8b = ModelConfig(
        vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
    )
    full_copy = 8e9 * BYTES_PER_DTYPE["fp16"]
    ranks = (8, 16, 64)

    lines = [
        "| Per-tenant weights | Size | Tenants in 16 GB |",
        "|---|---|---|",
        f"| Full fine-tuned copy | {full_copy / 1e9:.0f} GB | {int(16e9 // full_copy)} |",
    ]
    for rank in ranks:
        size = adapter_bytes(llama8b, LoRAConfig(rank=rank))
        lines.append(f"| LoRA adapter, rank {rank} | {size / 1e6:.1f} MB | {int(16e9 // size):,} |")
    return "\n".join(lines)


def lora_quality_table(name: str = "lora-tier1") -> str:
    """Each adapter against each tenant's held-out text, with the base model as the control."""
    quality = load(name)["summary"]["quality"]
    tenants = sorted(quality)
    columns = ["base", *tenants]
    lines = [
        "| Held-out text | "
        + " | ".join(f"{c} adapter" if c != "base" else "Base" for c in columns)
        + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for tenant in tenants:
        row = quality[tenant]
        cells = " | ".join(f"{row[c]:.4f}" for c in columns)
        lines.append(f"| {tenant}'s corpus | {cells} |")
    return "\n".join(lines)


def lora_batch_table(name: str = "lora-tier1") -> str:
    """What adapter diversity in one batch costs per decode step."""
    rows = load(name)["summary"]["batch"]
    baseline = next(r["ms"] for r in rows if r["distinct"] == 0)
    lines = ["| Decode step | Distinct adapters | ms/step | Overhead |", "|---|---|---|---|"]
    for row in rows:
        overhead = (row["ms"] - baseline) / baseline * 100
        lines.append(f"| {row['case']} | {row['distinct']} | {row['ms']:.3f} | {overhead:+.1f}% |")
    return "\n".join(lines)


def fairness_table() -> str:
    """Per-tenant time to first token, first-come-first-served against weighted fair queueing."""
    runs = [("FIFO (ch09)", "tenants-fifo-tier1"), ("Fair queue (ch19)", "tenants-fair-tier1")]
    per_run = {label: load(stem)["summary"] for label, stem in runs}
    tenants = sorted(next(iter(per_run.values())).get("by_tenant", {}))

    lines = [
        "| Scheduler | " + " | ".join(f"{t} TTFT p95" for t in tenants) + " | Completed |",
        "|---" * (len(tenants) + 2) + "|",
    ]
    for label, summary in per_run.items():
        by_tenant = summary.get("by_tenant", {})
        cells = " | ".join(f"{by_tenant[t]['ttft_p95']}s" for t in tenants)
        lines.append(f"| {label} | {cells} | {summary['completed']} |")
    return "\n".join(lines)


def workload_table() -> str:
    """Part VI's framing table: one engine, one configuration, four workloads."""
    rows = [
        ("Chat", "workload-chat-tier1"),
        ("Retrieval (RAG)", "workload-rag-tier1"),
        ("Agent loop", "workload-agent-tier1"),
        ("Code completion", "workload-completion-tier1"),
    ]
    lines = [
        "| Workload | Mean prompt | Mean output | TTFT p95 | ITL p95 | Output tok/s | Prefix reuse |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, name in rows:
        data = load(name)
        s, c = data["summary"], data["conditions"]
        lines.append(
            f"| {label} | {c['mean_prompt_len']:.0f} | {c['mean_output_len']:.0f} | "
            f"{s['ttft_p95']}s | {s['itl_p95']}s | {s['output_tokens_per_second']} | "
            f"{c['token_reuse']:.0%} |"
        )
    return "\n".join(lines)


def chat_turn_table(name: str = "chat-patterns-tier1") -> str:
    """How prefix reuse behaves as a conversation deepens."""
    rows = load(name)["summary"]["turns"]
    lines = ["| Turn | Mean prompt tokens | Prefix reuse | ms per request |", "|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['turn']} | {row['prompt_tokens']:.0f} | {row['reuse']:.0%} | "
            f"{row['ms_per_request']:.1f} |"
        )
    return "\n".join(lines)


def chat_batch_table(name: str = "chat-patterns-tier1") -> str:
    """Throughput against inter-token latency, which is what a streaming client feels."""
    rows = load(name)["summary"]["batch"]
    lines = [
        "| Max batch | ITL p50 | ITL p95 | Output tok/s | Met SLO |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['max_batch_size']} | {row['itl_p50']}s | {row['itl_p95']}s | "
            f"{row['output_tokens_per_second']} | {row['goodput_fraction']:.0%} |"
        )
    return "\n".join(lines)


def rag_interference_table(name: str = "rag-tier1") -> str:
    """What a long retrieval prefill does to the short requests sharing the engine."""
    rows = load(name)["summary"]["budgets"]
    lines = [
        "| Arrivals | Prefill | Short TTFT p95 | Short ITL p95 | Long TTFT p95 | Output tok/s |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        label = (
            f"Chunked, {row['token_budget']} tokens/step"
            if row["chunked"]
            else "One pass (no chunking)"
        )
        short, long = row["by_length"]["short"], row["by_length"]["long"]
        lines.append(
            f"| {row.get('rate_per_second', '—')}/s | {label} | {short['ttft_p95']}s | "
            f"{short['itl_p95']}s | {long['ttft_p95']}s | {row['output_tokens_per_second']} |"
        )
    return "\n".join(lines)


def agent_chain_table(name: str = "agents-tier1") -> str:
    """Per-call latency resampled into chains, which is what an agent's user actually waits for."""
    rows = load(name)["summary"]["chains"]
    single = rows[0]
    lines = [
        "| Calls | Total p50 | Total p95 | Total p99 | p50 vs one call | p95 vs one call | p95/p50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        spread = row.get("p95_over_p50") or row["p95"] / row["p50"]
        lines.append(
            f"| {row['calls']} | {row['p50']}s | {row['p95']}s | {row['p99']}s | "
            f"{row['p50'] / single['p50']:.1f}x | {row['p95'] / single['p95']:.1f}x | "
            f"{spread:.2f} |"
        )
    return "\n".join(lines)


def agent_cancellation_table(name: str = "agents-tier1") -> str:
    """Work done for callers who have already gone away."""
    data = load(name)["summary"]["cancellation"]
    lines = ["| Engine | Steps taken | Tokens produced | Requests dropped |", "|---|---|---|---|"]
    for label, key in (("Ignores cancellation", "without"), ("Honours it", "with_cancellation")):
        row = data[key]
        lines.append(
            f"| {label} | {row['steps']:,} | {row['tokens_produced']:,} | {row['aborted']} |"
        )
    lines.append(
        f"| **Saved** | **{data['steps_saved']:.0%}** | | "
        f"({data['abandoned_fraction']:.0%} abandoned) |"
    )
    return "\n".join(lines)


def completion_table(name: str = "offline-tier1") -> str:
    """Code completion against a deliberately brutal time-to-first-token objective."""
    rows = load(name)["summary"]["completion"]
    lines = [
        "| Max batch | TTFT p50 | TTFT p95 | Output tok/s | Met SLO |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['max_batch_size']} | {row['ttft_p50']}s | {row['ttft_p95']}s | "
            f"{row['output_tokens_per_second']} | {row['goodput_fraction']:.0%} |"
        )
    return "\n".join(lines)


def offline_table(name: str = "offline-tier1") -> str:
    """Offline batch: no arrivals, no objective, one number."""
    rows = load(name)["summary"]["offline"]
    lines = [
        "| Max batch | Wall time | Output tok/s | Peak KV utilisation | Preemptions |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['max_batch_size']} | {row['wall_time']}s | "
            f"{row['output_tokens_per_second']} | {row['peak_kv_utilisation']:.0%} | "
            f"{row['preemptions']} |"
        )
    return "\n".join(lines)


def template_reuse_table(name: str = "api-tier1") -> str:
    """What a chat template does to the prefix cache underneath it."""
    rows = load(name)["summary"]["templates"]
    lines = ["| Template | What differs | Prefix reuse |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['description']} | {row['token_reuse']:.0%} |")
    return "\n".join(lines)


def signal_timing_table(name: str = "observability-tier1") -> str:
    """Queue depth and latency on one axis, so which moves first is a matter of record."""
    rows = load(name)["summary"]["windows"]
    lines = [
        "| From step | Mean queue depth | Peak queue depth | KV utilisation | TTFT p95 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        ttft = f"{row['ttft_p95']}s" if row.get("ttft_p95") is not None else "—"
        lines.append(
            f"| {row['step']} | {row['mean_queue_depth']} | {row['peak_queue_depth']} | "
            f"{row['kv_utilisation']:.0%} | {ttft} |"
        )
    return "\n".join(lines)


def instrumentation_cost_table(name: str = "observability-tier1") -> str:
    """What sampling the engine on every step costs."""
    data = load(name)["summary"]["overhead"]
    report = load(name)["summary"]["report"]
    rows = [
        ("Wall time, instrumented", f"{data['instrumented_wall']}s"),
        ("Wall time, bare", f"{data['bare_wall']}s"),
        ("Overhead", f"{data['overhead_fraction']:+.1%}"),
        ("Steps sampled", f"{report['steps']:,}"),
        ("Peak queue depth", f"{report['peak_queue_depth']}"),
        ("Peak KV utilisation", f"{report['peak_kv_utilisation']:.0%}"),
    ]
    lines = ["| Quantity | Value |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(lines)


def shedding_table(name: str = "reliability-tier1") -> str:
    """Goodput under overload, with and without admission control."""
    rows = load(name)["summary"]["policies"]
    lines = [
        "| Policy | Rejected | Completed | TTFT p95 | Goodput req/s | Output tok/s |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['rejected']} | {row['completed']} | {row['ttft_p95']}s | "
            f"{row['goodput_per_second']} | {row['output_tokens_per_second']} |"
        )
    return "\n".join(lines)


def drain_table(name: str = "reliability-tier1") -> str:
    """What is in flight when a deploy starts, and what killing the process would discard."""
    data = load(name)["summary"]["drain"]
    rows = [
        ("Sequences in flight when the drain began", f"{data['in_flight_at_drain']}"),
        ("Generated tokens that a hard stop would discard", f"{data['tokens_at_risk']}"),
        ("Steps to finish them", f"{data['steps_to_drain']}"),
        ("New requests refused during the drain", f"{data['new_requests_rejected']}"),
    ]
    lines = ["| Quantity | Value |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(lines)


def cost_table() -> str:
    """Cost per million tokens as a function of utilisation, which is the term people assume.

    Arithmetic over stated inputs, not a measurement and not a price list. The hourly rate and the
    throughput are the reader's to supply; what the table shows is the *shape* — that the same
    hardware and the same engine produce wildly different costs depending on a number that usually
    goes unstated.
    """
    from llmserve.cost import Deployment

    rate, throughput = 3.0, 2500.0
    lines = [
        "| Utilisation | Tokens per hour | Cost per million tokens |",
        "|---|---|---|",
    ]
    for utilisation in (0.1, 0.25, 0.4, 0.6, 0.9):
        deployment = Deployment(
            "reference",
            dollars_per_hour=rate,
            tokens_per_second=throughput,
            utilisation=utilisation,
        )
        lines.append(
            f"| {utilisation:.0%} | {deployment.tokens_per_hour / 1e6:.1f}M | "
            f"${deployment.cost_per_million_tokens:.2f} |"
        )
    lines.append("")
    lines.append(
        f": At ${rate:.2f}/hour and {throughput:,.0f} sustained output tokens per second. "
        "Substitute your own two numbers; the shape does not change."
    )
    return "\n".join(lines)


def break_even_table() -> str:
    """The volume at which a fixed hardware bill beats a per-token price."""
    from llmserve.cost import Deployment, break_even_tokens_per_month

    deployment = Deployment("reference", dollars_per_hour=3.0, tokens_per_second=2500.0)
    monthly_bill = deployment.cost_per_hour * 24 * 30
    lines = [
        "| Hosted price per million tokens | Break-even volume per month | Utilisation it implies |",
        "|---|---|---|",
    ]
    ceiling = deployment.tokens_per_second * 3600 * 24 * 30
    for price in (0.20, 0.50, 1.00, 3.00):
        volume = break_even_tokens_per_month(deployment, price)
        if volume == float("inf"):
            lines.append(f"| ${price:.2f} | beyond this fleet's capacity | — |")
        else:
            lines.append(f"| ${price:.2f} | {volume / 1e9:.2f}B tokens | {volume / ceiling:.0%} |")
    lines.append("")
    lines.append(
        f": Against a fixed bill of ${monthly_bill:,.0f}/month. Below the break-even volume the "
        "hosted API is cheaper and somebody else operates it."
    )
    return "\n".join(lines)


def generator_table(name: str = "benchmarking-tier1") -> str:
    """The same engine and workload under both load generators."""
    rows = load(name)["summary"]["generators"]
    lines = [
        "| Generator | Setting | TTFT p50 | TTFT p95 | TTFT p99 | Achieved req/s |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['generator']} | {row['detail']} | {row['ttft_p50']}s | {row['ttft_p95']}s | "
            f"{row['ttft_p99']}s | {row['requests_per_second']} |"
        )
    return "\n".join(lines)


def final_scorecard_table() -> str:
    """The whole journey in one table, with the workload column that makes it honest.

    Rows are not all comparable, and pretending otherwise would undo the discipline the rest of the
    book insists on. The first five ran on the same uniform trace at the same rate and can be read
    against each other directly. The rest exist because a later chapter needed a workload the
    earlier ones do not have — prefix caching does nothing on a trace with no shared text — so they
    are listed with their workload and read within it.
    """
    rows = [
        ("ch01", "Naive, recompute everything", "uniform, 16 req/s", "naive-rate16-tier1"),
        ("ch05", "KV cache", "uniform, 16 req/s", "cached-rate16-tier1"),
        ("ch06", "Static batching", "uniform, 16 req/s", "static-rate16-tier1"),
        ("ch07", "Continuous batching", "uniform, 16 req/s", "continuous-rate16-tier1"),
        ("ch08", "Paged attention", "uniform, 16 req/s", "paged-rate16-tier1"),
        ("ch08", "Paged attention", "chat, 8 req/s", "paged-chat-rate8-tier1"),
        ("ch09", "Prefix caching", "chat, 8 req/s", "prefix-chat-rate8-tier1"),
        ("ch10", "Chunked prefill", "mixed lengths, 8 req/s", "chunked-budget512-tier1"),
        ("ch11", "Disaggregated pools", "uniform, 16 req/s", "disagg-rate16-tier1"),
        (
            "ch18",
            "Fleet, prefix-affinity routing",
            "multi-tenant, 8 req/s",
            "router-prefix-affinity-tier1",
        ),
        ("ch19", "Fair queueing", "noisy neighbour, 4 req/s", "tenants-fair-tier1"),
    ]
    lines = [
        "| Chapter | Engine | Workload | Output tok/s | TTFT p95 | Goodput req/s |",
        "|---|---|---|---|---|---|",
    ]
    for chapter, engine, workload, name in rows:
        s = load(name)["summary"]
        lines.append(
            f"| {chapter} | {engine} | {workload} | {s['output_tokens_per_second']} | "
            f"{s['ttft_p95']}s | {s['goodput_per_second']} |"
        )
    return "\n".join(lines)


def tensor_parallel_table() -> str:
    """What the collectives cost per token, in each phase, on each interconnect.

    Arithmetic, and deliberately so: this book's tier has one device, and a simulated all-reduce on
    one device would measure a memory copy rather than a network. The quantity that decides whether
    tensor parallelism is viable is exactly computable from the model's shape and the link's two
    numbers, and it is more useful than a measurement on the wrong hardware would be.
    """
    from llmserve.config import ModelConfig
    from llmserve.parallel import all_reduce_time_seconds

    model = ModelConfig(
        vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
    )
    links = [
        ("NVLink", 4.0e11, 5e-6),
        ("PCIe 4.0 x16", 3.2e10, 1.5e-5),
        ("100 GbE", 1.25e10, 5e-5),
    ]
    lines = [
        "| Interconnect | Shards | Decode, per token | Prefill, 2k prompt |",
        "|---|---|---|---|",
    ]
    for label, bandwidth, latency in links:
        for shards in (2, 8):
            decode = all_reduce_time_seconds(
                model,
                shards,
                tokens=1,
                bandwidth_bytes_per_second=bandwidth,
                latency_seconds=latency,
            )
            prefill = all_reduce_time_seconds(
                model,
                shards,
                tokens=2048,
                bandwidth_bytes_per_second=bandwidth,
                latency_seconds=latency,
            )
            lines.append(
                f"| {label} | {shards} | {decode * 1000:.2f} ms | {prefill * 1000:.0f} ms |"
            )
    lines.append("")
    lines.append(
        ": 8B model, 32 layers, fp16. Decode all-reduces one vector per layer and is bound by the "
        "*fixed* cost of each collective, so it does not improve with fewer shards. Prefill's "
        "payload scales with the prompt and is bound by bandwidth."
    )
    return "\n".join(lines)


def pipeline_bubble_table() -> str:
    """Idle device time in a pipeline, from the schedule alone."""
    from llmserve.parallel import pipeline_bubble_fraction

    microbatches = (1, 2, 4, 8, 16, 32)
    lines = [
        "| Stages | "
        + " | ".join(f"{m} microbatch{'es' if m > 1 else ''}" for m in microbatches)
        + " |",
        "|---" * (len(microbatches) + 1) + "|",
    ]
    for stages in (2, 4, 8):
        cells = " | ".join(f"{pipeline_bubble_fraction(stages, m):.0%}" for m in microbatches)
        lines.append(f"| {stages} | {cells} |")
    lines.append("")
    lines.append(
        ": The fraction of device time spent idle while the pipeline fills and drains. Serving sits "
        "at the left of this table, because a decode step produces one token per sequence and there "
        "is little to split into microbatches."
    )
    return "\n".join(lines)


def gather_cost_table() -> str:
    """What chapter 8's decode gather costs, in the only unit that matters: the decode ceiling.

    Arithmetic over the cache layout and one bandwidth figure, which is exactly the right tool. The
    quantity is a property of the shapes, and measuring it on this book's tiny model would report
    the overhead of Python rather than the cost of the copy.
    """
    from llmserve.config import ModelConfig
    from llmserve.kernels import gather_bytes_per_step, in_place_bytes_per_step

    model = ModelConfig(
        vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
    )
    bandwidth = 3.35e12  # HBM3, an H100-class figure
    # Combinations that actually fit an 80 GB device once the weights are resident. A row whose
    # KV cache exceeds the hardware is not a stronger argument, it is a wrong one.
    cases = [(8, 2048), (32, 2048), (32, 8192), (8, 32768)]
    lines = [
        "| Batch | Context | KV cache | Gather ms/step | In place ms/step | Decode ceiling, tok/s |",
        "|---|---|---|---|---|---|",
    ]
    for batch, context in cases:
        gathered = gather_bytes_per_step(model, context, batch)
        in_place = in_place_bytes_per_step(model, context, batch)
        slow, fast = gathered / bandwidth, in_place / bandwidth
        lines.append(
            f"| {batch} | {context:,} | {in_place / 1e9:.1f} GB | {slow * 1000:.1f} | "
            f"{fast * 1000:.1f} | {batch / slow:,.0f} → {batch / fast:,.0f} |"
        )
    lines.append("")
    lines.append(
        ": 8B model, 32 layers, GQA with 8 KV heads, fp16, at 3.35 TB/s of memory bandwidth. One "
        "decode step produces one token per sequence, so every byte here is moved to generate a "
        "handful of tokens — and the gather moves the cache twice, once to collect it and once for "
        "the model to append to it."
    )
    return "\n".join(lines)
