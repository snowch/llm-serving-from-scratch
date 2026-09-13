"""A closed-loop load generator, built only so chapter 31 can show why it is the wrong one.

Chapter 2 chose an open loop and explained the reasoning. This is the alternative, implemented
faithfully rather than as a straw man: a fixed number of workers, each submitting one request,
waiting for it, and submitting the next. It is how most load tests are written, including several
that shipped published numbers.

The defect is structural. When the server slows down, the generator slows down with it — a worker
that is waiting is not submitting — so the offered load is *defined* by the server's throughput and
the queue can never grow. Every latency the harness records is measured from a moment the server was
already ready to receive it. This is coordinated omission, and it does not make latencies slightly
optimistic; it removes the entire tail it was supposed to measure.
"""

from __future__ import annotations

import time

from bench.harness import SLO, BenchResult, RequestRecord, engine_sources
from llmserve.request import Request
from llmserve.sampling import SamplingParams


def run_closed_loop(
    engine,
    specs: list,
    *,
    n_workers: int = 4,
    slo: SLO | None = None,
    seed: int | None = 0,
    meta: dict | None = None,
) -> BenchResult:
    """Serve ``specs`` with ``n_workers`` concurrent in-flight requests and no arrival schedule.

    Each spec's declared arrival time is *ignored*, which is the whole point: in a closed loop there
    is no arrival process, only a concurrency level. Everything else — the engine, the records, the
    summary — is identical to the open-loop harness, so the comparison isolates the generator.
    """
    slo = slo or SLO()
    records: dict[int, RequestRecord] = {}
    pending = list(specs)
    in_flight = 0
    start = time.perf_counter()

    def submit() -> None:
        nonlocal in_flight
        spec = pending.pop(0)
        prompt = (
            list(spec.tokens)
            if spec.tokens is not None
            else [(i % 250) + 1 for i in range(spec.prompt_len)]
        )
        request = Request(
            prompt_token_ids=prompt,
            params=SamplingParams(max_tokens=spec.max_tokens, seed=seed),
            tenant=spec.tenant,
        )
        engine.add_request(request)
        # Recorded at submission, which in a closed loop is always a moment the server was free.
        records[request.request_id] = RequestRecord(
            request_id=request.request_id,
            arrival=time.perf_counter() - start,
            prompt_len=spec.prompt_len,
            tenant=spec.tenant,
        )
        in_flight += 1

    while pending and in_flight < n_workers:
        submit()

    while engine.has_work():
        outputs = engine.step()
        stamp = time.perf_counter() - start
        for out in outputs:
            record = records[out.request_id]
            if record.first_token is None and out.token_ids:
                record.first_token = stamp
            record.n_output += len(out.token_ids)
            if out.finished:
                record.finish = stamp
                in_flight -= 1
                if pending:
                    submit()

    wall = time.perf_counter() - start
    return BenchResult(
        engine=getattr(engine, "name", type(engine).__name__),
        code_sources=engine_sources(engine),
        records=list(records.values()),
        wall_time=wall,
        # There is no arrival rate in a closed loop. Reporting the achieved rate as if it were an
        # offered rate is exactly the confusion the chapter is about, so it is recorded as zero and
        # the concurrency level is recorded instead.
        rate_per_second=0.0,
        slo=slo,
        meta={"generator": "closed-loop", "workers": n_workers, **(meta or {})},
    )
