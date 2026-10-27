"""Unit tests for custom LoRA implementation."""

import pytest
import torch
import torch.nn as nn
from fedmedlora.models.lora import (
    LoRALinear,
    inject_lora_to_linear,
    extract_lora_state,
    load_lora_state,
    count_lora_parameters,
    make_fresh_lora_state,
)


def test_lora_initialization():
    base = nn.Linear(64, 128, bias=True)
    lora = LoRALinear(base, r=8, lora_alpha=16.0, lora_dropout=0.0)

    # Base weights must be frozen
    assert not lora.base_layer.weight.requires_grad
    assert not lora.base_layer.bias.requires_grad

    # LoRA parameters must be trainable
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad
    assert lora.lora_A.shape == (8, 64)
    assert lora.lora_B.shape == (128, 8)

    # B initialized to zero, so initially delta_w is zero
    assert torch.all(lora.lora_B == 0)
    delta_w = lora.get_delta_weight()
    assert torch.allclose(delta_w, torch.zeros_like(delta_w))


def test_lora_forward_equivalence_at_init():
    base = nn.Linear(32, 64)
    x = torch.randn(4, 32)
    expected_base_out = base(x)

    lora = LoRALinear(base, r=4, lora_alpha=8.0)
    out = lora(x)

    # Since B is zero at init, output matches base layer exactly
    assert torch.allclose(out, expected_base_out, atol=1e-6)


def test_lora_merged_vs_unmerged():
    torch.manual_seed(42)
    base = nn.Linear(32, 64)
    lora = LoRALinear(base, r=4, lora_alpha=8.0, lora_dropout=0.0)

    # Populate B with non-zero weights
    nn.init.normal_(lora.lora_B, std=0.1)

    x = torch.randn(8, 32)

    # Unmerged forward
    out_unmerged = lora(x)

    # Merge weights
    lora.merge()
    assert lora.merged
    out_merged = lora(x)

    # Merged and unmerged must produce identical outputs
    assert torch.allclose(out_merged, out_unmerged, atol=1e-5)

    # Unmerge weights
    lora.unmerge()
    assert not lora.merged
    out_restored = lora(x)
    assert torch.allclose(out_restored, out_unmerged, atol=1e-5)


def test_get_and_set_factors():
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4, lora_alpha=8.0)

    A_orig, B_orig = lora.get_lora_factors()
    assert A_orig.shape == (4, 16)
    assert B_orig.shape == (32, 4)

    # Modify and set
    new_A = torch.randn(4, 16)
    new_B = torch.randn(32, 4)
    lora.set_lora_factors(new_A, new_B)

    A_cur, B_cur = lora.get_lora_factors()
    assert torch.allclose(A_cur, new_A)
    assert torch.allclose(B_cur, new_B)


def test_dynamic_rank_change():
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4, lora_alpha=8.0)

    # Change rank from 4 to 8
    new_A = torch.randn(8, 16)
    new_B = torch.randn(32, 8)
    lora.set_lora_factors(new_A, new_B)

    assert lora.r == 8
    assert lora.lora_A.shape == (8, 16)
    assert lora.lora_B.shape == (32, 8)
    assert lora.scaling == (8.0 / 8)


def test_inject_and_extract_lora():
    class ToyViTBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)
            self.k_proj = nn.Linear(32, 32)
            self.v_proj = nn.Linear(32, 32)
            self.out_proj = nn.Linear(32, 32)

        def forward(self, x):
            return self.out_proj(self.q_proj(x) + self.v_proj(x))

    model = ToyViTBlock()
    injected = inject_lora_to_linear(model, target_names=["q_proj", "v_proj"], r=4)

    assert len(injected) == 2
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.v_proj, LoRALinear)
    assert isinstance(model.k_proj, nn.Linear) and not isinstance(model.k_proj, LoRALinear)

    # Check parameter count helper
    trainable, total, pct = count_lora_parameters(model)
    assert trainable > 0
    assert trainable < total

    # Check extract & load
    state = extract_lora_state(model)
    assert "q_proj" in state
    assert "v_proj" in state

    # Mutate state and reload
    state["q_proj"]["A"].fill_(1.0)
    load_lora_state(model, state)
    A_loaded, _ = model.q_proj.get_lora_factors()
    assert torch.allclose(A_loaded, torch.ones_like(A_loaded))


def test_fresh_lora_state_is_identical_across_clients():
    """Every client must start from the same adapters (seeded), with B = 0."""
    template = {"q": {"A": torch.zeros(4, 16), "B": torch.zeros(32, 4), "r": 4, "scaling": 2.0}}
    s1 = make_fresh_lora_state(template, seed=7)
    s2 = make_fresh_lora_state(template, seed=7)
    s3 = make_fresh_lora_state(template, seed=8)
    assert torch.equal(s1["q"]["A"], s2["q"]["A"])
    assert not torch.equal(s1["q"]["A"], s3["q"]["A"])
    assert torch.all(s1["q"]["B"] == 0)
