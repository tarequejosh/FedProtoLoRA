"""Aggregation rules for the federated PEFT baselines.

See docs/baselines.md for the exact interpretation of each paper and known deviations.
FlexLoRA (Bai et al., NeurIPS 2024) reuses `aggregate_product_space_svd` from methods/fedmedlora.py.

Baselines handled elsewhere (no aggregation function needed):
- local_only, linear_probe, centralized: no server aggregation (see fl/simulator.py).
- fedavg_full: weighted average of full encoder state dicts (FederatedServer.aggregate_full_state).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def aggregate_fedavg_lora(
    client_factors: List[Tuple[torch.Tensor, torch.Tensor]],
    weights: List[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Naive FedAvg-LoRA: independent weighted averaging of A and B.

    A_g = sum_k w_k A_k,  B_g = sum_k w_k B_k.  Note (sum w B)(sum w A) != sum w (B A) in general.
    """
    assert len(client_factors) == len(weights)
    A_global = torch.zeros_like(client_factors[0][0])
    B_global = torch.zeros_like(client_factors[0][1])
    for (A_k, B_k), w_k in zip(client_factors, weights):
        A_global += A_k * float(w_k)
        B_global += B_k * float(w_k)
    return A_global, B_global


def aggregate_ffa_lora(client_B_factors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    """FFA-LoRA: A is a shared, frozen random matrix; only B is averaged (exact since dW is linear in B)."""
    assert len(client_B_factors) == len(weights)
    B_global = torch.zeros_like(client_B_factors[0])
    for B_k, w_k in zip(client_B_factors, weights):
        B_global += B_k * float(w_k)
    return B_global


def aggregate_fedex_lora(
    client_factors: List[Tuple[torch.Tensor, torch.Tensor]],
    weights: List[float],
    scaling: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FedEx-LoRA (Singhal et al., ACL 2025): FedAvg-LoRA factors plus an exact residual.

    residual = s * [sum_k w_k B_k A_k - (sum_k w_k B_k)(sum_k w_k A_k)] is added to the frozen base
    weight on every client, so W0 + residual + s * B_g A_g equals the ideal product-space average.
    """
    A_g, B_g = aggregate_fedavg_lora(client_factors, weights)
    exact = torch.zeros((B_g.shape[0], A_g.shape[1]), dtype=torch.float32, device=A_g.device)
    for (A_k, B_k), w_k in zip(client_factors, weights):
        exact += (B_k.float() @ A_k.float()) * float(w_k)
    residual = (exact - B_g.float() @ A_g.float()) * scaling
    return A_g, B_g, residual


def aggregate_flora(
    client_factors: List[Tuple[torch.Tensor, torch.Tensor]],
    weights: List[float],
    scalings: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FLoRA stacking: B_s = [w_1 s_1 B_1, ..., w_K s_K B_K], A_s = [A_1; ...; A_K]

    so that B_s A_s = sum_k w_k s_k B_k A_k (the LoRA scalings s_k are folded into B; use scaling 1).

    No rank compression is applied (that would make FLoRA identical to product-space SVD). Following
    the FLoRA paper, the stacked update is merged into each client's base weights and the local LoRA
    modules are re-initialised every round (handled by the simulator). Ranks may differ per client.
    """
    assert len(client_factors) == len(weights)
    if scalings is None:
        scalings = [1.0] * len(weights)
    B_stacked = torch.cat(
        [B_k * (float(w_k) * float(s_k)) for (_, B_k), w_k, s_k in zip(client_factors, weights, scalings)], dim=1
    )
    A_stacked = torch.cat([A_k for (A_k, _) in client_factors], dim=0)
    return A_stacked, B_stacked


def hetlora_sparsity_weights(client_factors: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[float]:
    """HetLoRA sparsity-weighted aggregation weights: p_k proportional to ||B_k A_k||_F (normalised)."""
    norms = [float(torch.linalg.norm(B.float() @ A.float())) for A, B in client_factors]
    total = sum(norms)
    if total <= 0:
        return [1.0 / len(norms)] * len(norms)
    return [n / total for n in norms]


def aggregate_hetlora(
    client_factors: List[Tuple[torch.Tensor, torch.Tensor]],
    weights: List[float],
    max_rank: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HetLoRA: zero-pad heterogeneous-rank factors to max_rank and take the weighted sum.

    `weights` should be the sparsity weights from `hetlora_sparsity_weights`. Zero-padded entries
    contribute zeros (the paper aggregates padded matrices directly; no per-coordinate renormalisation).
    Each client then receives the global factors truncated to its own rank (`truncate_factors`).
    """
    assert len(client_factors) == len(weights)
    out_features = client_factors[0][1].shape[0]
    in_features = client_factors[0][0].shape[1]
    device = client_factors[0][0].device
    A_sum = torch.zeros((max_rank, in_features), device=device)
    B_sum = torch.zeros((out_features, max_rank), device=device)
    for (A_k, B_k), w_k in zip(client_factors, weights):
        r_k = A_k.shape[0]
        A_sum[:r_k] += A_k.float() * float(w_k)
        B_sum[:, :r_k] += B_k.float() * float(w_k)
    return A_sum, B_sum


def truncate_factors(A: torch.Tensor, B: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keeps the first `rank` LoRA components (HetLoRA distribution step)."""
    return A[:rank].clone(), B[:, :rank].clone()


def aggregate_fedsa_lora(client_A_factors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    """FedSA-LoRA (Guo et al., ICLR 2025): only A is shared and averaged; every B stays on its client."""
    assert len(client_A_factors) == len(weights)
    A_global = torch.zeros_like(client_A_factors[0])
    for A_k, w_k in zip(client_A_factors, weights):
        A_global += A_k * float(w_k)
    return A_global


def compute_fedprox_loss(
    model: nn.Module,
    global_lora_state: Dict[str, Dict[str, torch.Tensor]],
    mu: float = 0.01,
) -> torch.Tensor:
    """FedProx proximal term on the trainable LoRA factors:

    0.5 * mu * sum_l (||A_l - A_l^g||_F^2 + ||B_l - B_l^g||_F^2)
    """
    device = next(model.parameters()).device
    if mu <= 0.0 or not global_lora_state:
        return torch.zeros((), device=device)
    prox = torch.zeros((), device=device)
    for name, module in model.named_modules():
        if name in global_lora_state and getattr(module, "lora_A", None) is not None:
            g = global_lora_state[name]
            if g["A"].shape != module.lora_A.shape:
                continue
            prox = prox + torch.sum((module.lora_A - g["A"].to(device)) ** 2)
            prox = prox + torch.sum((module.lora_B - g["B"].to(device)) ** 2)
    return 0.5 * mu * prox
