# NotebookLM video prompts

One prompt per chapter, for NotebookLM **Video Overview → Customise**.

## How to use

1. Make a notebook per chapter and add that chapter's page as the source (URL, or the
   corresponding pages of `llm-serving-from-scratch.pdf`). One chapter per notebook — a notebook
   holding the whole book produces a survey, not an explainer.
2. Video Overview → Customise → paste the prompt → Generate.
3. Watch the figures. Every number in this book comes from a stamped result file and is true of a
   stated workload on stated hardware; a generated narration will happily drop the workload and
   turn a conditional result into a slogan. Each prompt below pushes back on that, but it is worth
   checking the ones that carry a number.

**The recurring failure mode to watch for:** this book reports results that went *against* the
technique — quantisation that slowed decode down, an offload tier that made latency worse, a
bounded window that appeared to improve quality for the wrong reason. A summariser's instinct is to
launder those into success stories. Most prompts below name the specific result that must survive.

---

## Part I — The Serving Problem

### ch01 · What an Inference Server Actually Does

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. Work
> through the chapter's full arc — the problem, the idea, the build, the measurement, the cost —
> and do not compress the middle. The spine is the split between **prefill** (processes the whole
> prompt in parallel) and **decode** (one token at a time, serially): establish it carefully and
> early, because the rest of the book depends on it. Spend real time on the closing result: past
> capacity, **goodput falls while throughput holds steady**, so an engine optimised on throughput
> can be busy and serving nobody. Make clear that serving is the bookkeeping around two expensive
> operations rather than the operations themselves. Quote figures with the workload they were
> measured on, and do not present this chapter as a tutorial — it is a framing chapter.

### ch02 · Measuring What Matters

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, on how
> to measure a serving system without fooling yourself. Distinguish TTFT, ITL, throughput and
> goodput precisely, and be explicit that **only goodput means anything on its own, and only
> against a stated SLO**. Devote a substantial segment to **coordinated omission**: a closed-loop
> load generator slows down in sympathy with the server, so it cannot observe overload and makes a
> broken system look healthy. Explain why percentiles and never means — in serving, the tail is the
> product. Close on the same trap as chapter one, from the measurement side: past saturation
> throughput holds while goodput collapses, so a throughput-only benchmark reports that collapse is
> fine. Keep every figure tied to its offered rate and length distribution.

### ch03 · The Arithmetic of Inference

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. This
> chapter is the load-bearing one: **prefill is compute-bound, decode is memory-bandwidth-bound**,
> and every technique later in the book attacks one or the other. Derive that slowly and make the
> reader able to reproduce it. Walk through the KV-cache-per-token formula
> (`2 × n_layers × n_kv_heads × head_dim × bytes`) term by term, and explain that this is the
> budget everything later competes over. Then the consequence that justifies batching: decode
> throughput is bounded by bandwidth divided by bytes read per step, and weights are read once per
> step regardless of batch size. End on the methodological point — a FLOPs count predicts prefill
> reasonably and decode badly, and when the two disagree the scarce resource is bytes.

---

## Part II — The Decode Loop

### ch04 · Generating Tokens Correctly

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, on
> getting the sampler right before optimising anything. Treat sampling as a **pipeline whose stage
> order is part of its definition** — applying the stages out of order yields a plausible-looking
> distribution that is not the one that was configured. Give full weight to the sign bug: dividing
> a *negative* logit raises its probability, so a repetition penalty that ignores the sign
> encourages exactly what it was meant to suppress. Cover why stop reasons are not interchangeable
> (a caller needs to know whether the model finished or the budget ran out), and the streaming
> detail that decoding each token as it arrives breaks multi-byte characters — including why
> *incomplete* and *invalid* must be told apart, or the stream stalls forever on a corrupt byte.
> Land the payoff: greedy determinism plus seeded reproducibility is what makes every later
> optimisation testable.

### ch05 · The KV Cache

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. Build
> the argument from why keys and values can be cached at all — they depend only on a token and its
> position — while queries cannot and need not be. Be concrete that this **converts quadratic
> recomputation into linear memory**, and that memory then becomes the limit on concurrency, which
> sets up all of Part III. Give real time to the failure mode: passing the wrong position during
> decode produces no error, just quietly worse output. Explain how RoPE makes a cached key
> self-describing, and flag that this is what later allows cache entries to be shared between
> different requests. Close on the methodological result — a FLOPs-based prediction overstated this
> speedup by about 10× because the operation was bound by bytes moved, not arithmetic.

### ch06 · Static Batching and Its Limits

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. Lead
> with the strongest economic fact in serving: decode reads the weights once per step regardless of
> batch size, so **batching is nearly free throughput**. Then the friction — batching needs a
> rectangle and real requests are ragged, so padding brings position and masking bugs that produce
> worse output rather than errors. Spend proper time on one of them: mask with a finite floor, not
> `-inf`, because a fully-masked row softmaxes to NaN and `0 × NaN` contaminates every other
> sequence in the batch. Then show how static batching gives much of its gain back — finished
> sequences hold their slots, arriving requests wait for a whole batch to drain. End on the setup
> for the next chapter: both problems are **scheduling** problems, and neither needs a faster
> kernel.

---

## Part III — Building the Engine

### ch07 · Continuous Batching

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, on the
> single observation that a decode step is indifferent to which sequences are in it — so the
> running set can change every step. Show the two wastes it removes separately: retiring finished
> sequences immediately kills ragged-completion waste, admitting into freed slots kills
> batch-formation delay. Be precise about **where the gain shows up — mostly in TTFT** — because
> this is a queueing improvement rather than a compute one, and goodput follows TTFT. Give the
> operational reading rule real airtime: flat TTFT as load rises means capacity in hand, rising
> TTFT means queueing, whatever throughput reports. Close on what the engine has become — a
> scheduler — and that its remaining problems are scheduling and memory problems, not arithmetic.

### ch08 · Paged Attention and the Block Manager

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. Start
> from why contiguous allocation cannot work: a sequence's final length is unknowable at admission,
> so you either over-reserve or copy, and both waste memory nobody else can use. Explain fixed-size
> blocks plus a per-sequence block table as the fix — external fragmentation gone entirely,
> internal fragmentation capped at one partial block. Cover how memory exhaustion becomes a
> *scheduling decision* (preempt and recompute) rather than a failure, and the correctness rule
> that preemption must discard the cache and keep the output, because discarding the output makes a
> caller receive the same tokens twice. Include the scheduler subtlety: count memory already
> promised this step, not just what is free now. **Do not end on a clean win** — the chapter's
> honest result is that paging costs throughput when the gather is naive, and only a kernel
> resolves the tension.

### ch09 · Prefix Caching

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems.
> Identical prompt prefixes produce identical keys and values, and paging already supplied the
> mechanism — build that carefully. Then spend most of the video on the design rules, each of which
> is a real bug avoided: key a block by **every token up to and including it**, because keying by
> its own contents lets unrelated prompts collide and produce wrong output with no error; stop a
> lookup at the first miss, since a later hit across a gap is unusable; publish a block when it
> *fills*, not when its author finishes, or concurrent requests sharing a prompt all miss — the
> exact case that matters. Give a full segment to the deadlock: cached blocks hold references, so a
> warm cache is indistinguishable from a full pool, and the engine stalls with a full queue and an
> empty batch. **A cache that can refuse to yield is not a cache, it is a leak.** Note honestly
> that on shared-prefix traffic this is the best return in the book and the only optimisation so
> far with no trade attached.

### ch10 · Chunked Prefill and Scheduling Policy

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, on
> putting a token budget on each step so one long prefill cannot monopolise the engine. Explain the
> ordering rule and why it holds — decode before prefill, because those tokens have someone waiting
> on them. Then be scrupulous about the result, which cuts both ways: **chunking redistributes
> latency rather than removing it, and the budget has a wrong answer in both directions** — a
> moderate budget beat no chunking on every axis here, an aggressive one lost on every axis. Do not
> round that into "chunking is good". Include the counter-intuitive finding that splitting a long
> prefill can be *faster* than one pass, because a single enormous attention matrix has terrible
> locality. Close on the engineering lesson: represent scheduler state explicitly — both bugs in
> this chapter came from inferring "is this sequence still prefilling?" from a quantity that
> changes for other reasons.

### ch11 · Disaggregating Prefill and Decode

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. The
> architecture follows from chapter three: prefill and decode are bound by different resources, so
> one machine serving both is a compromise on both. Explain what separation buys — interference
> removed by construction, pools sized and specialised independently — and then price it honestly:
> the entire KV cache moves between pools once per request, a cost that scales with model size and
> context length and is measured in gigabytes at production scale. Make the conclusion explicit:
> **interconnect bandwidth, not scheduling, decides whether this architecture works**, and below
> the scale where you have a fast fabric and enough traffic for two pools it is pure overhead.
> Treat the chapter's methodological move as a feature, not a caveat: the test rig structurally
> cannot show a benefit, the chapter says so, and makes its argument with arithmetic instead.

### ch12 · Offloading the KV Cache

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, on the
> idea that **eviction does not have to mean destruction** — a block moved down a tier costs a
> transfer to get back, a block dropped costs a whole prefill. Cover the design rules and why each
> one is forced: key the tier by **content, not block number**, because the physical slot is the
> one thing about a demoted block that will not survive; clone on the way out, because a view into
> a block the allocator is about to reuse is silent corruption; restore *before* the lookup so
> there is only one prefill path to test. Then handle the result with care and do not smooth it
> over: **prefix reuse is restored and latency on this rig gets worse anyway**, because the premise
> of the technique is a second device and there isn't one. Present the mechanism working and the
> economics as two separate claims, measured separately — and give the arithmetic its own segment,
> since break-even bandwidth is low enough that host memory, NVMe and even a network all clear it.
> The real question is how many tiers, not whether.

---

## Part IV — Making the Math Cheaper

### ch13 · Attention at Speed

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. Two
> threads: **online softmax** and **GQA/MQA**. For the first, establish that the score matrix is
> the quadratic term and at long context is larger than the model itself, then walk through how the
> running-maximum correction makes the incremental result exactly equal to the batch one — and is
> also what prevents overflow. State plainly that FlashAttention is **IO-aware, not
> FLOP-reducing**: it attacks bytes moved, which is what attention was limited by. For the second,
> derive the KV saving straight from chapter three's formula — the group ratio is the whole
> mechanism. Then be honest about the measurement: **GQA barely mattered here, and that is the
> correct result** on a small model with short contexts where weights dominate; at 8B and 32k the
> same choice is 2 concurrent sequences versus 80. End on the generalisable lesson — distrust any
> single measurement's generality, this book's included, and check which term dominates in your own
> regime.

### ch14 · Writing a Paged Attention Kernel in Triton

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. Be
> upfront in the opening that **this chapter deliberately ships no Triton kernel**: it was written
> without a GPU, a kernel cannot be verified without one, and publishing an unverified one would
> contradict the book's own standard. What it gives instead is the algorithm, the tests and the
> arithmetic — the kernel's specification. Frame that as the point, not an apology. Cover why the
> kernel is needed at all: chapter eight's gather roughly halves the decode ceiling, and the cost
> is pure data movement rather than arithmetic. Then the key structural insight — **online softmax
> is what makes a paged kernel possible**, because an algorithm that never needs all the scores at
> once never needs the blocks to be adjacent. Include the traps: map grouped-query heads to KV
> heads rather than expanding the cache (expanding restores the copy), mask partial blocks with a
> finite floor, special-case the first block's rescale — both failures are NaN and both are silent.
> Close on the discipline: write the reference and its tests first, and note that correctness means
> bounded error, not bitwise equality.

### ch15 · Quantisation for Serving

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, and
> lead with the reporting rule: **quality, speed and memory are one measurement, and reporting a
> subset is how quantisation gets oversold.** Explain that weight-only quantisation attacks bytes
> read, so it helps decode and does nothing for prefill. Give a substantial segment to scale
> selection being more important than bit width — per-channel is not a refinement, and without it a
> single outlier channel destroys every other channel. Mention that quantising the KV cache is
> often the better return at long context and the most frequently forgotten. Then handle the
> headline result carefully and **do not turn it into a success story**: on this rig, decode got
> *slower* at every bit width, because dequantisation overhead dominates on hardware that was never
> memory-bound for a model this small. That is a fact about the test rig, not about the technique,
> and the chapter says so. End on the general rule: when an end-to-end measurement and the
> established advice disagree, find the mechanism before believing either.

### ch16 · Bounding the Context

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. Open by
> placing this chapter apart from the rest: every other cache optimisation makes entries cheaper,
> this one makes there be **fewer of them** — the only one that turns a linear cache into a
> constant one — and it is **the first optimisation in the book that changes what the model says**,
> so it is measured on quality rather than latency. Then be extremely careful with the result,
> because the naive reading is backwards: at a length the model was trained for, a window costs
> almost nothing; past that length bounding appears to *help*, and it appears to help because the
> unbounded baseline is extrapolating, not because the window is good. State the conclusion
> explicitly — **a quality-neutral claim about bounded context is meaningless without the training
> sequence length**. Cover attention sinks with the same discipline: this model has them, measurably
> and by an order of magnitude, and keeping them buys nothing at this window size. Explain why
> measuring the mechanism separately from the outcome is what lets you tell "no sinks" from "sinks
> are not the binding loss". Include why position renumbering requires caching keys *before*
> rotation — a model-level change, not a cache policy.

### ch17 · Speculative Decoding

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. Build
> the mechanism from the asymmetry: verification is prefill-shaped and therefore parallel, so *k*
> proposals cost one target forward pass, and a round always yields at least one token — so
> speculation can never reduce tokens per pass. Then make the correctness argument the centrepiece.
> The **residual acceptance rule** makes the output distribution exactly the target's; the
> obvious-looking simplification of resampling from the target on rejection double-counts the
> drafted token and biases the result invisibly. Give full weight to how the chapter *proves* this
> rather than asserting it — it implements the wrong rule as a negative control and measures both,
> and the correct rule lands a fraction of a standard error from target while the biased one lands
> many away. Draw the general lesson: a correctness argument that cannot be falsified is not
> evidence. Close on the boundaries — acceptance depends on the workload and should be quoted with
> its trace, and speculation helps at low batch and hurts at high batch. It is a latency technique,
> not a throughput one.

### ch18 · Constrained and Structured Decoding

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, on
> masking illegal tokens so that invalid output is **unreachable rather than unlikely**. Explain
> the grammar as a finite-state machine over bytes, and be clear that the genuinely hard part is
> *lifting* the machine from characters to tokens — which byte-level tokenization lets this book
> skip, and which is where the production libraries spend their complexity. Then give the central
> segment to the result that matters most: **a grammar constrains shape, not length.** Generation
> that runs out of budget produces unparseable output while every individual token was legal, so
> the accepting state has to be wired to the stopping condition. Make the distinction sharp — the
> guarantee on offer is "no illegal token is ever emitted", the guarantee people assume they are
> buying is "the output parses", and those differ whenever `max_tokens` exists. Include the
> implementation notes: mask with a finite floor never `-inf`, build masks by iterating the allowed
> set rather than the vocabulary, cache per state. End on the integration cost — constraints
> interact badly with both speculation and prefix caching, and both need to know about the grammar.

---

## Part V — Scaling Out

### ch19 · Multi-GPU: Tensor, Pipeline and Expert Parallelism

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. Take
> the tensor-parallel split slowly: cut each layer's pair of matrices **column-then-row** so one
> collective per pair suffices, and be explicit that any other cut doubles the communication for
> nothing. Stress that the split must be *exactly* correct and that this is testable on a single
> device with no GPU at all — which is what the chapter does, and it should be presented as a
> strength rather than a limitation. Then the result worth the most: **decode's collective cost is
> latency, prefill's is bandwidth** — different problems, different fixes, and chapter three's split
> showing up again. Cover how grouped-query attention caps tensor parallelism at the KV head count,
> and why pipeline parallelism communicates far less but idles instead, so serving's shortage of
> microbatches makes it a cross-node tool rather than a within-node one. End on the conclusion that
> reframes the whole chapter: **parallelism is how you fit a model, not how you make it fast.** If
> it already fits, run replicas. Note that the collective cost here is computed from shape and link
> characteristics rather than timed, and say so plainly.

### ch20 · Multi-Replica: Routing, Autoscaling and Cold Starts

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. The
> headline is that **round-robin is the wrong LLM load balancer**: it balances request counts, which
> correlate with neither cost nor cache locality, and it actively destroys the prefix cache. Work
> through least-outstanding-*tokens* as the right generic policy — a token is nearly a unit of work,
> a connection is not — but report honestly that on this trace it bought a fraction of what
> abandoning balance did. **Balancing better is not the win.** Give the main segment to cache-aware
> routing, and make the paradox explicit: it works *by refusing to balance*, and concentrating a
> prefix on one replica turned zero goodput into real goodput. Then the necessary guard — without an
> imbalance limit, one hot prefix pins the whole fleet to a single replica. Cover autoscaling on
> queue depth and KV utilisation rather than CPU (decode is memory-bound, so CPU is not the
> constraint and therefore not the signal), and cold start being dominated by weight transfer —
> which makes quantisation an availability feature and scale-to-zero incompatible with a tight TTFT
> objective.

### ch21 · Multi-Tenancy and LoRA at Serving Time

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, in three
> movements: the weights, serving them together, and fairness. For the weights, a fine-tune is a
> low-rank change so a tenant's weights can be slivers rather than a copy — the memory argument is
> three orders of magnitude and is not close — and `B` must be initialised to zero so a new adapter
> is exactly the identity, otherwise an untrained adapter and a broken one look the same. For
> serving, the result that changes procurement decisions: **cost scales with how many *distinct*
> adapters are in a batch, not how many exist**. One adapter is nearly free, eight is not. Note that
> merging removes all serving overhead and all multi-tenancy, making it right for exactly one
> tenant. For fairness, show how little it takes — one field on the request, one method on the
> scheduler — and why you charge on **admission, not completion**, since a scheduler that waits for
> completion cannot see a burst until it has already served it. Close on the honest framing: fair
> queueing does not create capacity, it decides who waits, and that is the whole point.

---

## Part VI — Serving Patterns by Workload

### ch22 · Chat and Assistants

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. Open on
> the framing result for the whole of Part VI: **prompt length does not predict cost — reusable
> prompt length does**, and the agent row has the longest prompts and the best latency. Then what
> makes chat distinctive: a conversation's prompt grows monotonically and almost all the growth is
> already cached, so later turns are *cheaper* than earlier ones and turn one is the expensive one.
> Give the main segment to batch size, and be careful with it: inter-token latency is user-visible,
> so batch size stops being a free dial — but the ITL-optimal batch fails the objective outright
> because it destroys time to first token. The rule is to pick the smallest batch that saturates
> throughput. Flag the contrast the book makes later: a workload with no human in it reads the same
> table and picks the largest batch that fits. Close on the operational consequence — chat leans on
> the prefix cache harder than anything except agents, which makes session affinity and cache warmth
> operational concerns rather than optimisations.

### ch23 · RAG and Long Context

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, and
> structure it as the chapter does: what does *not* fix retrieval traffic, then what does. Establish
> first why prefix caching is worth far less here than on chat — retrieval prompts share an
> instruction and then diverge, which is by design and not a defect. Then the carefully bounded
> result: **chunked prefill helps below saturation and does nothing above it**, because scheduling
> redistributes latency and does not create capacity. Do not overstate it in either direction. Make
> the highest-value change clear — the binding constraint is KV memory, so quantising the cache is
> the biggest lever available and it is a configuration change rather than an architecture one.
> Include the cheap fix people miss: fix the passage *order* before reaching for a cleverer cache,
> since a stable order restores a prefix that a shuffled one destroys. End on co-serving —
> embedding and reranking models are prefill-shaped and burst in sync with the generator's worst
> case, so co-serve them under a token budget or not at all — and on `top_k` being a serving
> parameter, meaning whoever tunes retrieval quality is also tuning the prefill bill.

### ch24 · Agents and Tool Use

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. Start
> from why agent traffic is the best case for prefix caching in this book: it replays its
> transcript, so step *n* contains step *n−1*, and the hit rate is the thing to protect. Then give
> the centre of the video to the statistical result, because it inverts standard advice: **a
> chain's latency is a sum, and sums concentrate.** The median grows linearly, the tail grows more
> slowly, the relative spread shrinks — and therefore you should **optimise the median for agents**,
> the reverse of the advice for every other workload in the book. Make sure that reversal is stated
> explicitly rather than implied. Cover the reporting failure it implies: report *chain* latency,
> because a per-call p95 that sounds healthy can be a twenty-second wait and no server-side
> dashboard will show it. Finish on cancellation as a **throughput feature** — it recovers a large
> fraction of steps and is nearly free on a paged engine — plus the two silent failures, that a
> cancelled request needs a terminal message and that abort must be idempotent.

### ch25 · Code Completion and Offline Batch

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, built
> around one axis: **is anyone waiting?** That single question decides nearly every other setting,
> and the chapter puts the two extremes side by side. For code completion, correct the common
> reflex: small batches are wrong, because queueing dominates time to first token, so the batch that
> minimises per-request work maximises per-request latency. Say plainly that completion's real wins
> are prefix caching, speculation, cancellation and a smaller model — not scheduler tuning. For
> offline batch, show what removing the arrival process buys: no need for an open-loop harness, and
> a licence for every throughput-for-latency trade in the book, plus sorting by length, which
> removes the padding waste chapter six could only mitigate. Land the conclusion hard, because it is
> the point of the whole part: **the same engine, tuned two ways, produces opposite answers from the
> same table — so a benchmark without a stated workload is not a result.**

---

## Part VII — Running It in Production

### ch26 · The API Surface

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, on the
> claim that **the chat template is part of the serving stack**. Lead with the measurement: a
> per-request value in the system prompt — a timestamp, say — costs about a fifth of the prefix
> cache, and nothing in the engine's metrics will tell you. Derive the rule from the mechanism:
> anything that varies per request goes *after* the shared text, because a prefix cache can only
> reuse a prefix, so position matters as much as content. Then streaming, and the way it is usually
> broken — a stream must end with an explicit sentinel, because silence is not termination and the
> resulting failure is invisible from the server. Cover detokenizing incrementally in one place
> rather than per token, counting usage from the tokens the engine actually processed rather than by
> re-tokenizing text, and limiting request size in tokens at the edge, since an unbounded prompt is
> an unbounded stall for everyone.

### ch27 · Observability for Serving Engines

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, opening
> on why host metrics cannot explain a serving engine — and why **utilisation is actively
> misleading**, since decode is memory-bound. Lay out the five causes worth exporting alongside the
> outcome: queue depth, KV utilisation, preemption rate, batch size, cache hit rate, against TTFT
> and ITL. Give a full segment to **sampling per step rather than per scrape**, because a scrape
> interval cannot see the events that produce your tail. Then the result that changes how incidents
> are read: **queue depth leads and latency lags** — latency keeps rising after the cause has
> cleared, which is why incidents feel longer than they are. Be concrete about alerting: page on SLO
> burn rate and on queue growth, never on utilisation. Close on tracing, and on putting tenant and
> chain identifiers on traces, without which two of the book's real problems are invisible.

### ch28 · Reliability and Operations

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, on
> deliberately refusing work. Lead with the counter-intuitive measurement and give it room:
> **refusing half the offered requests doubled goodput and cut the tail fourfold.** An overloaded
> engine that accepts everything serves nothing well. Be precise about the signals — shed on queue
> depth and KV utilisation, never on CPU — and about the manner: **reject explicitly and fast**,
> because a dropped connection is indistinguishable from a hang, and the client's retry is what
> turns a spike into an outage. Cover draining before stopping, since a killed process produces
> truncated responses that nothing retries and nothing alerts on. Give real weight to the worst
> failure mode in the chapter: a config change that moves *quality* while every latency and
> throughput signal stays flat — only tests catch it. End on making rejection rate a first-class
> metric, or a fleet that sheds well looks healthy while turning away traffic indefinitely.

### ch29 · Cost and Capacity Planning

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems. The
> headline is blunt: **utilisation dominates cost per token**, by nearly an order of magnitude
> across a realistic range, so measure yours before optimising anything else. Then the sizing error
> the chapter exists to prevent — a benchmark's throughput is a *peak*, and sizing from it produces
> a fleet that meets its throughput target and misses its latency one. Explain why prompt tokens and
> output tokens cost different amounts of machine time, so your traffic mix changes your cost per
> token with nothing else changing. Handle build-versus-buy carefully: the break-even volume against
> a hosted API depends on the fixed hardware bill and the hosted price rather than on utilisation,
> while the cost of what you serve depends on almost nothing else — and say plainly that the
> arithmetic leaves out operations, which at the volumes where self-hosting first looks attractive
> is usually the larger number. Close on the sizing rule: size from the batch size that meets the
> SLO, not the one that maximises throughput.

### ch30 · Choosing a Serving Framework

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, on how
> to choose between serving frameworks. The thesis: **capability fit and operational maturity decide
> framework choices; throughput rarely does.** Walk through the rubric — it has one row per
> mechanism in the book — and frame what that means: every row is a question the reader can now ask
> *and understand the answer to*, which is what the previous twenty-nine chapters bought them. Give
> a substantial segment to filling it in honestly: check the issue tracker, then the source, then
> your own workload, in that order of cost and of reliability; and read the **mechanism**, not the
> feature list — "supports prefix caching" is a claim, *when a block gets published* is the truth.
> Be explicit and unapologetic about why the chapter contains **no vendor comparison table**: it
> could not be verified from the environment the book was written in, and any such table — including
> one the author could have written — is wrong within months. The method outlasts it. Close on the
> two-stage process: score on paper to eliminate the field, benchmark the survivors to decide.

### ch31 · Benchmarking and Verifying a Serving Stack

> Make a long, in-depth explainer for engineers who build or operate LLM inference systems, on
> producing a serving benchmark nobody can dismiss. Lead with the chapter's strongest finding:
> **the load generator is part of the measurement** — the same engine doing the same work reported
> a tail two orders of magnitude apart between an open and a closed loop. Explain why a closed loop
> cannot report an offered rate at all, and that "N concurrent users" is a concurrency level whose
> latencies are not what they look like. Then the reporting standard, stated as a checklist: goodput
> against a stated SLO, with percentiles, at a stated offered rate, on a stated length distribution
> — anything less is not checkable. Give correctness its own segment, because speed comparisons
> usually skip it: a faster engine that changes the output is not a faster engine, so use
> equivalence tests for deterministic paths and distributional tests for sampled ones. Close on the
> two rules that make a comparison fair — tune both sides, or you are benchmarking documentation;
> and publish the harness, because a number nobody can reproduce is an assertion.

---

## Capstone

### ch32 · The Finished Engine

> Make a long, in-depth retrospective for engineers who build or operate LLM inference systems,
> pulling the whole book together. The central claim: **scheduling and memory policy beat
> arithmetic** — almost nothing here made a matrix multiply faster, and throughput still rose by
> most of an order of magnitude. Show how **the two-phase model explains nearly everything**:
> prefill compute-bound and parallel, decode memory-bound and sequential, with most of the book's
> surprising results following directly. Walk the journey table, and be explicit about why it needs
> a *workload* column — **a number without its workload and its SLO is not a result**, and half the
> book's tables exist to make that concrete. Give a proper segment to which optimisations paid and
> which did not, including the ones that lost on this hardware, and to the chapter's sharpest line:
> **every optimisation has a trade, and the chapters that found none were the suspicious ones** —
> prefix caching genuinely has none; paging, chunking, quantisation, speculation, affinity routing
> and fair queueing all do, and knowing which one you are paying is the job. Cover what was
> deliberately not built and why. End on the honest assessment: the gap between this engine and a
> production one is mostly kernels and operations, not ideas.
