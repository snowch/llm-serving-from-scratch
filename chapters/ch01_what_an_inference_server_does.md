---
title: "What an Inference Server Actually Does"
short_title: "ch01 What an Inference Server"
---

(ch01)=
# ch01 · What an Inference Server Actually Does

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | None — start here |
| **Scorecard** | Establishes the **baseline row**: every later number is relative to this one. |
:::

## The problem

A language model is a function. You give it a sequence of token ids, it gives you back a
probability distribution over what comes next. Calling it is one line.

A serving system is not a function. It has to accept requests it did not expect, from people who
are watching, at a rate it does not control, using a device that costs more per hour than the
engineer maintaining it. Between "call the model" and "serve the model" lies everything this book
is about.

So let us build the naive thing and watch it fail. Not a strawman — the actual code most people
write first, which wraps a forward pass in a loop and puts an HTTP handler in front of it.

## The idea

Before any optimisation, be clear about what a request costs. Here is the full lifecycle of one
request, from the socket to the last token:

1. **Receive** the HTTP request and parse it.
2. **Apply the chat template**, turning a list of messages into one string.
3. **Tokenise** that string into token ids.
4. **Prefill**: run all the prompt's tokens through the model in one pass, producing a
   distribution for the position after the prompt.
5. **Sample** one token from that distribution.
6. **Decode**: run that single token through the model to get the next distribution. Repeat.
7. **Detokenise** each new token into text, incrementally, as it is produced.
8. **Stream** each fragment back to the caller.
9. **Stop** on an end token, a token budget, or the caller hanging up.

Steps 4 and 6 are the only ones that touch the accelerator. Everything else is bookkeeping. That
proportion — one or two expensive steps surrounded by many cheap ones — is worth holding on to,
because a surprising amount of serving work is making sure the cheap steps never get in the way
of the expensive ones.

Two of those steps deserve their real names now, because the whole book turns on the distinction:

- **Prefill** processes the entire prompt at once. Every token can be computed in parallel,
  because they are all already known.
- **Decode** produces one token at a time, and each one depends on the last. Nothing about it is
  parallel within a single request.

They are the same arithmetic on very different shapes, and {ref}`ch03` shows they are limited by
completely different things. Almost every technique in this book exists because of that split.

## The build

The engine interface is deliberately small. An engine takes requests, and advancing it one
`step` produces some tokens:

```{literalinclude} ../llmserve/engines/base.py
:language: python
:start-at: class Engine
:end-before: def has_work
```

Chapter 1's implementation makes two choices that are both wrong, on purpose:

```{literalinclude} ../llmserve/engines/naive.py
:language: python
:start-at: class NaiveEngine
:end-before: def add_request
```

**One request at a time.** A request that arrives while another is running waits. Not for a slot,
not for a batch — for the entire preceding request to finish generating every one of its tokens.

**No KV cache.** Look closely at the decode branch:

```{literalinclude} ../llmserve/engines/naive.py
:language: python
:start-at:             # Decode without a cache
:end-before:         prev = torch.tensor
```

To produce token 200, it runs the model over all 199 previous tokens again. Every one of those
was computed on the previous step, and thrown away. {ref}`ch05` fixes exactly this.

One detail that is easy to miss: `step()` returns after a *single* token, even though this engine
serves a single request. That is not how you would naturally write it, but it is what lets the
harness in {ref}`ch02` distinguish the time to the first token from the time between subsequent
ones. Those two numbers describe completely different experiences for the person waiting, and an
engine that reports only "the request took 4 seconds" hides which one went wrong.

## The measurement

Here is the baseline. Same trace, same model, same machine, at three arrival rates:

```{include} _generated/ch01-baseline.md
```

Read the first column downwards. At one request per second the server copes. By four, the median
user waits seconds before a single character appears — and the output token rate has almost
stopped climbing. From there, doubling the load again barely moves it.

That flat output rate is the engine's capacity. Beyond it, extra load does not produce extra
work; it produces a queue. And because the SLO in this run allows one second to the first token,
almost nothing at high load meets it.

Look at the goodput column as the rate climbs, though. It goes **down**.

```{include} _generated/ch02-goodput-collapse.md
```

This is the single most important shape in the book, and the reason {ref}`ch02` is about
measurement rather than optimisation. A server past its capacity does not merely stop improving.
It gets *worse*, because it spends its fixed throughput on requests whose owners have already
given up waiting. Throughput says the machine is busy. Goodput says the work is wasted.

## The cost

Nothing has been traded away yet — this is the baseline. But name what is wrong now, because each
item is a chapter:

| What it does wrong | Cost | Fixed in |
|---|---|---|
| Recomputes the whole prefix every token | Quadratic work per request | {ref}`ch05` |
| Serves one request at a time | Device mostly idle; queue grows without bound | {ref}`ch06`, {ref}`ch07` |
| Holds a request's memory for its whole lifetime | Concurrency capped far below what memory allows | {ref}`ch08` |
| Re-processes prompts it has already seen | Duplicate prefill across requests | {ref}`ch09` |

## Key takeaways

- Serving is the bookkeeping around two expensive operations, not the operations themselves.
- **Prefill** processes a whole prompt in parallel; **decode** produces one token at a time,
  serially. They behave so differently that the rest of the book keeps them apart.
- Time-to-first-token and inter-token latency describe different experiences. An engine that
  reports only total request time cannot tell you which one is failing.
- Past its capacity, a server's goodput falls while its throughput holds steady. Optimising the
  number that stays flat is how you end up with a busy machine serving nobody.

## Looking ahead

Every number above came from a harness that has not been described yet, and the claims in this
chapter are only worth as much as that harness is. {ref}`ch02` builds it: what to measure, why
means are useless here, and why a load generator written the obvious way will quietly tell you
your overloaded server is healthy.

## Further reading

The lifecycle above is the one every production engine implements. For how the mature systems
structure it, the vLLM and SGLang papers in Appendix E are the primary sources, and both are
readable once you have built the naive version yourself.
