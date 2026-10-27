"""Unit tests for federated PEFT baseline aggregations."""

import pytest
import torch
import torch.nn as nn
from fedmedlora.methods.baselines import (
    aggregate_fedavg_lora,
    aggregate_fedex_lora,
    aggregate_fedsa_lora,
    aggregate_ffa_lora,
    aggregate_flora,
    aggregate_hetlora,
    compute_fedprox_loss,
)
from fedmedlora.models.lora import LoRALinear


def test_fedavg_lora():
    A1 = torch.ones(4, 16)
    B1 = torch.ones(32, 4) * 2.0
    A2 = torch.ones(4, 16) * 3.0
    B2 = torch.ones(32, 4) * 4.0

    factors = [(A1, B1), (A2, B2)]
    weights = [0.5, 0.5]

    A_g, B_g = aggregate_fedavg_lora(factors, weights)
    assert torch.allclose(A_g, torch.ones(4, 16) * 2.0)
    assert torch.allclose(B_g, torch.ones(32, 4) * 3.0)


def test_ffa_lora():
    B1 = torch.ones(32, 4) * 2.0
    B2 = torch.ones(32, 4) * 6.0
    weights = [0.75, 0.25]

    B_g = aggregate_ffa_lora([B1, B2], weights)
    # 0.75 * 2 + 0.25 * 6 = 1.5 + 1.5 = 3.0
    assert torch.allclose(B_g, torch.ones(32, 4) * 3.0)


def test_flora_stacking():
    torch.manual_seed(42)
    in_f, out_f, r = 16, 32, 4
    A1, B1 = torch.randn(r, in_f), torch.randn(out_f, r)
    A2, B2 = torch.randn(r, in_f), torch.randn(out_f, r)

    factors = [(A1, B1), (A2, B2)]
    weights = [0.6, 0.4]

    A_stack, B_stack = aggregate_flora(factors, weights)
    assert A_stack.shape == (8, in_f)
    assert B_stack.shape == (out_f, 8)

    # Product of stacked representation must equal the exact sum
    expected_dW = 0.6 * (B1 @ A1) + 0.4 * (B2 @ A2)
    stacked_dW = B_stack @ A_stack
    assert torch.allclose(stacked_dW, expected_dW, atol=1e-5)


def test_hetlora_zero_padding_and_sparsity():
    in_f, out_f = 16, 32
    # Client 1: rank 4
    A1 = torch.ones(4, in_f)
    B1 = torch.ones(out_f, 4)
    # Client 2: rank 8
    A2 = torch.ones(8, in_f) * 2.0
    B2 = torch.ones(out_f, 8) * 2.0

    factors = [(A1, B1), (A2, B2)]
    weights = [0.5, 0.5]

    A_g, B_g = aggregate_hetlora(factors, weights, max_rank=8)
    assert A_g.shape == (8, in_f)
    assert B_g.shape == (out_f, 8)

    # Coordinates 0-3 have both clients active: 0.5*1 + 0.5*2 = 1.5
    assert torch.allclose(A_g[:4], torch.ones(4, in_f) * 1.5)
    # Coordinates 4-7 have only client 2 active; zero-padded entries contribute zeros and are
    # not renormalised (HetLoRA aggregates padded matrices directly): 0.5*2 = 1.0
    assert torch.allclose(A_g[4:], torch.ones(4, in_f) * 1.0)


def test_flora_stacking_folds_scaling():
    torch.manual_seed(0)
    A1, B1 = torch.randn(4, 16), torch.randn(32, 4)
    A2, B2 = torch.randn(8, 16), torch.randn(32, 8)
    A_s, B_s = aggregate_flora([(A1, B1), (A2, B2)], [0.5, 0.5], scalings=[4.0, 2.0])
    expected = 0.5 * 4.0 * (B1 @ A1) + 0.5 * 2.0 * (B2 @ A2)
    assert torch.allclose(B_s @ A_s, expected, atol=1e-4)


def test_fedex_lora_is_exact():
    torch.manual_seed(0)
    factors = [(torch.randn(4, 16), torch.randn(32, 4)) for _ in range(3)]
    weights = [0.2, 0.3, 0.5]
    s = 2.0
    A_g, B_g, residual = aggregate_fedex_lora(factors, weights, scaling=s)
    ideal = sum(w * s * (B @ A) for (A, B), w in zip(factors, weights))
    assert torch.allclose(residual + s * (B_g @ A_g), ideal, atol=1e-4)


def test_fedsa_lora_averages_A_only():
    A_g = aggregate_fedsa_lora([torch.ones(4, 8), torch.ones(4, 8) * 3.0], [0.5, 0.5])
    assert torch.allclose(A_g, torch.ones(4, 8) * 2.0)


def test_fedprox_loss():
    base = nn.Linear(16, 32)
    model = LoRALinear(base, r=4)
    A, B = model.get_lora_factors()

    # If global matches local, prox loss is 0
    state_match = {"": {"A": A.clone(), "B": B.clone()}}
    loss_0 = compute_fedprox_loss(model, state_match, mu=0.01)
    assert loss_0.item() == 0.0

    # If drifted, prox loss > 0
    state_drift = {"": {"A": A + 1.0, "B": B + 1.0}}
    loss_drift = compute_fedprox_loss(model, state_drift, mu=0.1)
    assert loss_drift.item() > 0.0
