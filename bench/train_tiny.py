"""Train the reference model just enough for quality measurements to mean something.

Chapter 15 needs a model whose output can get *worse*. Random weights cannot show that: their
perplexity is already at chance, so quantising them changes a meaningless number into a different
meaningless number.

Rather than download a trained model — which would put a network dependency on the book's default
path, and a Hugging Face CDN host on your allowlist — we train briefly on a deterministic
synthetic corpus. The corpus has real structure to learn, so perplexity drops well below chance
and quantisation damage becomes visible. It takes a couple of minutes on a laptop and is
bit-reproducible from a seed.

It is not a substitute for evaluating a production model. It *is* enough to answer the question
chapter 15 actually asks: does this quantisation scheme degrade output, and by how much relative
to what it saves?
"""

from __future__ import annotations

import torch
from torch import nn

from llmserve.config import REFERENCE_MODEL, ModelConfig
from llmserve.model import TinyGPT, build_model

#: A Zipf-distributed word process over a synthetic vocabulary.
#:
#: An earlier version used four sentence templates. The model learned them to a perplexity of
#: about 1.2 — essentially perfect — and quantising it to four bits changed nothing, because the
#: task left so much capacity spare that damaged weights still solved it. A quality measurement
#: needs a model working near its limit, so the corpus has a large vocabulary, a realistic
#: frequency distribution, and no templates to memorise.
SYLLABLES = [
    "ba",
    "ke",
    "tor",
    "mi",
    "sen",
    "lu",
    "dra",
    "fi",
    "gon",
    "ra",
    "zel",
    "pu",
    "nex",
    "vi",
    "cro",
    "tam",
    "qui",
    "oth",
    "lam",
    "yer",
]
VOCAB_SIZE = 600
CORPUS_WORDS = 60_000


def make_corpus(seed: int = 0) -> str:
    """Deterministic pseudo-natural text: many word types, Zipf-ish frequencies, no templates."""
    rng = torch.Generator().manual_seed(seed)

    def word() -> str:
        length = 2 + int(torch.randint(3, (1,), generator=rng))
        return "".join(
            SYLLABLES[int(torch.randint(len(SYLLABLES), (1,), generator=rng))]
            for _ in range(length)
        )

    vocabulary = [word() for _ in range(VOCAB_SIZE)]
    # 1/rank frequencies: a few words dominate and the tail is genuinely hard, as in real text.
    frequencies = torch.tensor([1.0 / (i + 1) for i in range(VOCAB_SIZE)])
    draws = torch.multinomial(frequencies, CORPUS_WORDS, replacement=True, generator=rng)
    return " ".join(vocabulary[int(i)] for i in draws)


def train_reference_model(
    config: ModelConfig | None = None,
    *,
    steps: int = 600,
    batch_size: int = 16,
    seq_len: int = 128,
    lr: float = 3e-3,
    seed: int = 0,
    log_every: int = 200,
) -> tuple[TinyGPT, float]:
    """Train and return the model plus its final perplexity on held-out text."""
    config = config or REFERENCE_MODEL
    torch.manual_seed(seed)
    model = build_model(config)
    model.train()

    corpus = torch.tensor(list(make_corpus(seed=seed).encode("utf-8")), dtype=torch.long)
    split = int(0.9 * len(corpus))
    train_data, eval_data = corpus[:split], corpus[split:]

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    for step in range(steps):
        starts = torch.randint(len(train_data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([train_data[s : s + seq_len] for s in starts])
        y = torch.stack([train_data[s + 1 : s + seq_len + 1] for s in starts])

        with torch.enable_grad():
            logits, _ = model._forward(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        if log_every and step % log_every == 0:
            print(f"  step {step:>4}  loss {loss.item():.4f}")

    model.eval()
    return model, evaluate_perplexity(model, eval_data, seq_len=seq_len)


@torch.inference_mode()
def evaluate_perplexity(model: TinyGPT, data: torch.Tensor, seq_len: int = 128) -> float:
    """Mean perplexity over non-overlapping windows of ``data``."""
    config = model.cfg
    losses = []
    for start in range(0, len(data) - seq_len - 1, seq_len):
        x = data[start : start + seq_len].unsqueeze(0)
        y = data[start + 1 : start + seq_len + 1].unsqueeze(0)
        logits, _ = model(x)
        losses.append(
            nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1)).item()
        )
    return float(torch.exp(torch.tensor(losses).mean()))


def held_out_data(seed: int = 0) -> torch.Tensor:
    corpus = torch.tensor(list(make_corpus(seed=seed).encode("utf-8")), dtype=torch.long)
    return corpus[int(0.9 * len(corpus)) :]
