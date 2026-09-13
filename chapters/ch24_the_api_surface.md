---
title: "The API Surface"
short_title: "ch24 The API Surface"
---

(ch24)=
# ch24 · The API Surface

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch04](#ch04), [ch09](#ch09), [ch22](#ch22) |
| **Scorecard** | No engine change. One line of the API layer costs a fifth of the prefix cache. |
:::

## The problem

Everything so far has been the engine. This chapter is the layer between it and an HTTP client, and
that layer has an unusual property: **it is where a surprising share of production incidents live,
and none of them are visible in the engine's own metrics.**

A chat template with a stray value, a usage count that bills the wrong number, a stream that never
terminates. Each is a small piece of ordinary data-handling code, each is easy to get wrong, and
each is invisible to everything in {ref}`ch25`'s dashboard.

There is no web framework in this chapter. Request translation, the chat template, event framing and
usage accounting are all ordinary data transformations, and they are where the interesting failures
are. Bolting them to an ASGI server is the easy half and teaches nothing.

## The measurement: a template change costs a fifth of the cache

The chat template decides the exact bytes the model sees:

```{literalinclude} ../llmserve/api.py
:language: python
:start-at: class ChatTemplate
:end-before:     #: opens a turn, per role
```

Because {ref}`ch09`'s cache is content-addressed, the template also decides how much of the prompt
can be reused. The same conversations, rendered three ways:

```{include} _generated/ch24-templates.md
```

**A timestamp in the system prompt costs a fifth of the reuse.** It is one line, it looks helpful,
it changes nothing about the model's behaviour that anyone would notice, and it makes the system
prompt different on every single request — so the cache misses on the very first block and
everything after it.

The variant that moves the system prompt to the end is worse still and for the same reason: the
shared text is no longer a *prefix*, and a prefix cache can only reuse a prefix. The content is
identical; the position is what matters.

Both of these are the kind of change that ships in a pull request titled "add request context to
system prompt", passes review, passes tests, and shows up a week later as a cost regression nobody
can attribute. **Anything that varies per request belongs after the shared text, not before it** —
and ideally not in the prompt at all.

## Streaming, and the way it is usually broken

A streaming response owns the incremental detokenizer, because that is the only place that can be
correct: a multi-byte character split across two tokens must not become two replacement characters,
and a stateless per-token decode cannot see that it is mid-character.

```{literalinclude} ../llmserve/api.py
:language: python
:start-at: class StreamingResponse
:end-before:     def __init__
```

The failure mode that matters is at the end of the stream:

```{literalinclude} ../llmserve/api.py
:language: python
:start-at: def done_event
:end-before:     return "data: [DONE]
```

A client that never sees the sentinel either hangs until its own timeout or reports a truncated
response — and the server sees a completed request, so nothing alerts. That asymmetry is the whole
reason this is worth a section: the failure is entirely on the client's side of a boundary the
server is not watching.

The same applies to cancellation. {ref}`ch22` built abort into the engine and made the point there:
a cancelled request still needs a terminal message. At this layer, that means a client disconnect
must reach the engine — a disconnect that is noticed but not acted on is a request that keeps
generating for nobody.

## Usage accounting

```{literalinclude} ../llmserve/api.py
:language: python
:start-at: class Usage
:end-before:     prompt_tokens: int = 0
```

Counting from the token ids the engine processed, never from the text. Re-tokenizing the rendered
prompt to count it is the classic way to bill a different number than you served, because the count
then depends on a code path the engine never ran — and the two diverge exactly where tokenization is
subtle, which is where the expensive prompts are.

This matters beyond billing. Usage counts are what every downstream quota, rate limit and capacity
model is built on ({ref}`ch19`, {ref}`ch27`), so a systematic error here propagates into decisions
that look unrelated.

## The rest of the surface, briefly

Four things this chapter builds no code for, in rough order of how often they cause an incident:

**Backpressure.** A server that accepts everything and queues it has moved the queue from the client
to itself, where it is less visible and harder to shed. {ref}`ch26` is the answer.

**Request size limits.** An unbounded prompt is an unbounded prefill, which is an unbounded stall
for everyone else ({ref}`ch21`). The limit belongs at the edge, in tokens rather than bytes, and it
has to be checked before the request reaches the scheduler.

**Timeouts that match the engine's behaviour.** A client timeout shorter than the queueing delay
turns a slow period into a retry storm, and a retry storm turns a slow period into an outage.

**Compatibility that is honest.** An "OpenAI-compatible" endpoint that silently ignores parameters
it does not implement produces output the caller did not ask for and cannot debug. Accepting a
subset is fine; pretending is not:

```{literalinclude} ../llmserve/api.py
:language: python
:start-at: class CompletionRequest
:end-before:     messages: list[ChatMessage]
```

## The cost

- **The template is now a performance-critical interface**, owned by whoever writes prompts rather
  than whoever runs the server. It needs a test that pins the rendered bytes, or it will drift.
- **Streaming makes every request stateful** for its whole duration: a detokenizer, a connection and
  a place to send tokens. That is a per-request cost the engine does not see and a per-request leak
  if termination is ever missed.
- **Cancellation plumbing crosses every layer** — client disconnect, HTTP handler, engine abort — and
  is only correct if all three agree. {ref}`ch22` lists the races.
- **Usage counting is a correctness surface with money attached**, which makes it the one place in
  this book where being approximately right is worse than being slow.
- **Compatibility constrains the engine.** Adopting somebody else's API means adopting their
  parameters, including the ones that are awkward to implement well, and the pressure to accept and
  ignore them is constant.

## Key takeaways

- **The chat template is part of the serving stack.** A per-request value in the system prompt costs
  a fifth of the prefix cache, and nothing in the engine's metrics will say so.
- Anything that varies per request goes *after* the shared text. A prefix cache can only reuse a
  prefix, so position matters as much as content.
- A stream must end with an explicit sentinel. Silence is not termination, and the resulting failure
  is invisible from the server.
- Detokenize incrementally, in one place. Per-token decoding corrupts multi-byte characters.
- Count usage from the tokens the engine processed, not by re-tokenizing text.
- Limit request size in tokens at the edge. An unbounded prompt is an unbounded stall for everyone.

## Looking ahead

Every failure in this chapter is invisible from the engine's own metrics — which is a statement about
the metrics. {ref}`ch25` builds the signals that would have caught these, and measures which one
moves first when the engine gets into trouble.

## Further reading

There is no good paper here; the material is API design and the sources are the specifications of the
APIs people actually implement against. Read a real OpenAI-compatible server's request handling — vLLM's
is readable — with attention to what it does with parameters it does not support, which is the honest
part of the problem.
