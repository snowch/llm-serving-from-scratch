"""Chapter 21: low-rank adapters, and serving several of them at once."""

import pytest
import torch

from llmserve.config import REFERENCE_MODEL, ModelConfig
from llmserve.lora import (
    LoRAConfig,
    LoRALinear,
    adapter_bytes,
    attach,
    detach,
    make_adapter,
    merge,
    set_active,
    set_row_adapters,
)
from llmserve.model import build_model


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


@pytest.fixture
def fresh(model):
    """A model with adapters attached, restored afterwards so tests cannot leak into each other."""
    adapters = [make_adapter(model, name, seed=i) for i, name in enumerate(("alice", "bob"))]
    layers = attach(model, adapters)
    yield model, {a.name: a for a in adapters}, layers
    detach(model)


def _randomise(adapter, scale=0.05, seed=0):
    """Give an adapter a non-zero B so it stops being the identity."""
    generator = torch.Generator().manual_seed(seed)
    for _, b in adapter.weights.values():
        b.normal_(0.0, scale, generator=generator)


# -- the adapter itself ----------------------------------------------------------------


def test_a_new_adapter_is_exactly_the_identity(fresh):
    """B starts at zero, so an untrained adapter cannot be told from no adapter."""
    model, _, layers = fresh
    ids = torch.randint(1, 250, (2, 8))
    set_active(layers, None)
    base, _ = model(ids)
    set_active(layers, "alice")
    with_adapter, _ = model(ids)
    assert torch.allclose(base, with_adapter)


def test_adapters_share_the_base_weights(fresh):
    """The whole memory argument: n tenants cost one model, not n models."""
    model, _, layers = fresh
    for name, layer in layers.items():
        assert isinstance(layer, LoRALinear)
        original = dict(model.named_modules())[f"{name}.base"]
        assert layer.base.weight.data_ptr() == original.weight.data_ptr()


def test_adapter_is_a_small_fraction_of_the_base(fresh):
    model, adapters, _ = fresh
    base_params = sum(p.numel() for p in model.parameters())
    assert adapters["alice"].n_parameters < base_params / 100


def test_making_an_adapter_needs_a_projection_that_exists(model):
    with pytest.raises(ValueError, match="no projections"):
        make_adapter(model, "nope", LoRAConfig(targets=("not_a_projection",)))


def test_attached_adapters_must_agree_on_scaling(model):
    """Two adapters with different alpha/rank cannot share one layer's scale factor."""
    a = make_adapter(model, "a", LoRAConfig(rank=8, alpha=16.0))
    b = make_adapter(model, "b", LoRAConfig(rank=4, alpha=16.0))
    with pytest.raises(ValueError, match="alpha/rank"):
        attach(model, [a, b])


# -- serving several at once ------------------------------------------------------------


def test_a_mixed_batch_gives_each_row_its_own_adapter(fresh):
    """The claim the whole chapter rests on, checked row by row against the homogeneous path."""
    model, adapters, layers = fresh
    _randomise(adapters["alice"], seed=1)
    _randomise(adapters["bob"], seed=2)
    ids = torch.randint(1, 250, (4, 8))

    set_active(layers, "alice")
    all_alice, _ = model(ids)
    set_active(layers, "bob")
    all_bob, _ = model(ids)

    set_row_adapters(layers, ["alice", "bob", "alice", "bob"])
    mixed, _ = model(ids)

    assert torch.allclose(mixed[0], all_alice[0], atol=1e-5)
    assert torch.allclose(mixed[2], all_alice[2], atol=1e-5)
    assert torch.allclose(mixed[1], all_bob[1], atol=1e-5)
    assert torch.allclose(mixed[3], all_bob[3], atol=1e-5)


def test_a_row_with_no_adapter_gets_the_base_model(fresh):
    """A batch may legitimately mix tenants with adapters and callers using the base model."""
    model, adapters, layers = fresh
    _randomise(adapters["alice"], seed=3)
    ids = torch.randint(1, 250, (2, 8))

    set_active(layers, None)
    base, _ = model(ids)
    set_row_adapters(layers, ["alice", None])
    mixed, _ = model(ids)

    assert torch.allclose(mixed[1], base[1], atol=1e-6)
    assert not torch.allclose(mixed[0], base[0], atol=1e-6)


def test_merging_matches_applying(fresh):
    """Merging is an optimisation, so it has to produce the same model, not a similar one."""
    model, adapters, layers = fresh
    _randomise(adapters["alice"], seed=4)
    ids = torch.randint(1, 250, (2, 8))

    set_active(layers, "alice")
    applied, _ = model(ids)
    merge(layers, "alice")
    merged, _ = model(ids)

    assert torch.allclose(applied, merged, atol=1e-5)
    assert "alice" not in next(iter(layers.values())).adapters


def test_detach_restores_the_model_exactly(model):
    ids = torch.randint(1, 250, (2, 8))
    before, _ = model(ids)
    adapter = make_adapter(model, "temp", seed=9)
    _randomise(adapter, seed=5)
    layers = attach(model, [adapter])
    set_active(layers, "temp")
    detach(model)
    after, _ = model(ids)
    assert torch.allclose(before, after)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())


# -- arithmetic --------------------------------------------------------------------------


def test_adapter_bytes_matches_a_real_adapter(model):
    """The arithmetic the chapter's memory table uses must agree with the code that allocates."""
    config = LoRAConfig(rank=8)
    adapter = make_adapter(model, "check", config)
    predicted = adapter_bytes(REFERENCE_MODEL, config)
    assert predicted == pytest.approx(adapter.n_parameters * REFERENCE_MODEL.bytes_per_element)


def test_adapter_cost_is_linear_in_rank():
    config = ModelConfig(
        vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
    )
    small = adapter_bytes(config, LoRAConfig(rank=8))
    large = adapter_bytes(config, LoRAConfig(rank=64))
    assert large == pytest.approx(8 * small)
