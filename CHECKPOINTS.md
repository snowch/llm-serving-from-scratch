# Checkpoints

Every chapter that changes the engine gets a git tag, so you can start reading anywhere.

```bash
git checkout ch07-continuous-batching   # the engine exactly as it stood at the end of ch07
```

| Chapter | Tag | Engine state | Scorecard |
|---|---|---|---|
| ch01 | `ch01-naive-server` | FastAPI wrapper around `generate()`, one request at a time | baseline |
| ch02 | `ch02-harness` | unchanged engine; `bench/` harness exists | baseline, measured properly |
| ch04 | `ch04-own-decode` | our sampling and detokenisation, equivalence-tested | correctness baseline |
| ch05 | `ch05-kv-cache` | per-sequence contiguous KV cache | first speedup |
| ch06 | `ch06-static-batching` | fixed-size batches, left-padded, waste instrumented | throughput up, a third of slots discarded |
| ch07 | `ch07-continuous-batching` | iteration-level scheduler, `step()` loop | largest single jump |
| ch08 | `ch08-paged-attention` | block allocator, block tables, preemption | concurrency up |
| ch09 | `ch09-prefix-caching` | radix prefix cache with LRU eviction | TTFT down on shared prefixes |
| ch10 | `ch10-chunked-prefill` | token-budget scheduling, policy hooks | ITL p99 down |
| ch11 | `ch11-disaggregated` | separate prefill/decode pools | TTFT/ITL separation |
| ch12 | `ch12-fast-attention` | FlashAttention backend, GQA | latency and KV footprint down |
| ch13 | `ch13-triton-kernel` | Triton paged-decode kernel *(optional)* | kernel-level win |
| ch14 | `ch14-quantised` | INT8/INT4 weights, quantised KV | memory down, quality moves |
| ch15 | `ch15-speculative` | draft-and-verify decoding | ITL down at low batch |
| ch16 | `ch16-constrained` | grammar-constrained decoding | validity at near-zero cost |
| ch17 | `ch17-tensor-parallel` | multi-GPU tensor parallelism | larger models fit |
| ch18 | `ch18-multi-replica` | router with cache-aware policies | horizontal scaling |
| ch19 | `ch19-multi-tenant` | batched LoRA adapters, quotas | many adapters, one base |
| ch24 | `ch24-openai-api` | OpenAI-compatible server, streaming, cancellation | wasted work recovered |
| ch25 | `ch25-observable` | metrics and OTel traces | — |
| ch26 | `ch26-resilient` | shedding, drain, fault handling | behaviour past saturation |
| ch29 | `v1.0` | the finished engine | full table |

Tags are created as each chapter is completed; rows above without a tag in the repository yet
are planned, not published.
