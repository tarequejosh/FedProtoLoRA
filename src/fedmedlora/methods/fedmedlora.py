"""FedProtoLoRA: Heterogeneity-Aware Federated Low-Rank Adaptation.

Implements:
- Component A: Product-space aggregation with truncated SVD re-projection.
- Component B: Domain-similarity-aware client weighting (mean CLS embedding of the frozen backbone).
- Component C: Client-adaptive upload rank, chosen from the spectral energy of each client's round update.
- Personalized (per-client similarity-weighted) aggregation option.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


def svd_refactor(
    delta_w: torch.Tensor,
    target_rank: int,
    target_alpha: float = 16.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Optimal (Eckart-Young) rank-`target_rank` factorisation of a dense update.

    Returns (A, B, scaling) with scaling = target_alpha / target_rank such that
    scaling * B @ A is the best rank-`target_rank` approximation of `delta_w` in Frobenius norm.
    Singular values are split symmetrically between B and A; rows beyond rank(delta_w) are zero-padded.
    """
    target_scaling = (target_alpha / target_rank) if target_rank > 0 else 1.0
    out_features, in_features = delta_w.shape
    device = delta_w.device

    # U: (out_features, min_dim), S: (min_dim,), Vh: (min_dim, in_features)
    U, S, Vh = torch.linalg.svd(delta_w.float(), full_matrices=False)

    k_rank = min(target_rank, len(S))
    sqrt_S = torch.diag(torch.sqrt(torch.clamp(S[:k_rank], min=0.0)))
    scale_factor = 1.0 / (target_scaling ** 0.5)

    B_global = (U[:, :k_rank] @ sqrt_S) * scale_factor
    A_global = (sqrt_S @ Vh[:k_rank, :]) * scale_factor

    # If target_rank > k_rank, pad with zeros
    if target_rank > k_rank:
        pad_r = target_rank - k_rank
        B_global = torch.cat([B_global, torch.zeros((out_features, pad_r), dtype=B_global.dtype, device=device)], dim=1)
        A_global = torch.cat([A_global, torch.zeros((pad_r, in_features), dtype=A_global.dtype, device=device)], dim=0)

    return A_global, B_global, target_scaling


def aggregate_product_space_svd(
    client_factors: List[Tuple[torch.Tensor, torch.Tensor, float]],  # List of (A_k, B_k, scaling_k)
    weights: List[float],
    target_rank: int,
    target_alpha: float = 16.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Computes exact weighted product-space sum dW = sum_k w_k * s_k * (B_k @ A_k)

    and re-factorizes it into optimal rank-target_rank factors (A_global, B_global) via truncated SVD.
    Client ranks may differ (the sum is formed in the dense weight space).

    Returns:
        (A_global, B_global, target_scaling)
    """
    assert len(client_factors) == len(weights)
    out_features = client_factors[0][1].shape[0]
    in_features = client_factors[0][0].shape[1]
    device = client_factors[0][0].device

    # Accumulate product-space delta
    delta_w_total = torch.zeros((out_features, in_features), dtype=torch.float32, device=device)
    for (A_k, B_k, s_k), w_k in zip(client_factors, weights):
        delta_w_total += (B_k.float() @ A_k.float()) * (s_k * float(w_k))

    return svd_refactor(delta_w_total, target_rank, target_alpha)


def select_rank_by_energy(
    singular_values: torch.Tensor,
    energy_tau: float = 0.95,
    rank_candidates: Tuple[int, ...] = (4, 8, 16, 32),
) -> int:
    """Smallest candidate rank r with sum_{i<=r} sigma_i^2 / sum_i sigma_i^2 >= tau."""
    candidates = sorted(rank_candidates)
    energy = singular_values.float() ** 2
    total_energy = energy.sum()
    if total_energy <= 1e-12:
        return candidates[0]

    cum_energy = torch.cumsum(energy, dim=0) / total_energy
    for r_cand in candidates:
        if r_cand >= len(cum_energy) or cum_energy[r_cand - 1] >= energy_tau:
            return r_cand
    return candidates[-1]


def compute_spectral_adaptive_rank(
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float,
    energy_tau: float = 0.95,
    rank_candidates: Tuple[int, ...] = (4, 8, 16, 32),
) -> int:
    """Selects the smallest rank r_k from rank_candidates that preserves at least tau of the spectral energy.

    Spectral energy is defined as sum_{i=1}^r sigma_i^2 / sum_{all} sigma_i^2 of (B @ A).
    """
    with torch.no_grad():
        dW = (B.float() @ A.float()) * scaling
        return select_rank_by_energy(torch.linalg.svdvals(dW), energy_tau, rank_candidates)


def compress_update_adaptive_rank(
    delta_w: torch.Tensor,
    lora_alpha: float,
    energy_tau: float = 0.95,
    rank_candidates: Tuple[int, ...] = (4, 8, 16),
) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
    """Component C: compresses a client's round update to the smallest candidate rank keeping tau of its energy.

    Returns (A_k, B_k, scaling_k, r_k) with scaling_k * B_k @ A_k ~= delta_w.
    """
    with torch.no_grad():
        r_k = select_rank_by_energy(torch.linalg.svdvals(delta_w.float()), energy_tau, rank_candidates)
        A_k, B_k, s_k = svd_refactor(delta_w, r_k, lora_alpha)
    return A_k, B_k, s_k, r_k


def compute_domain_similarity_weights(
    client_embeddings: Dict[str, torch.Tensor],
    sample_counts: Dict[str, int],
    target_client: Optional[str] = None,
    blend_alpha: float = 0.5,
    metric: str = "cosine",
) -> Dict[str, float]:
    """Computes blended aggregation weights combining sample counts with domain feature similarity.

    Args:
        client_embeddings: {client_id: mean_cls_embedding_tensor (dim,)}
        sample_counts: {client_id: num_samples}
        target_client: If personalized, calculate similarity relative to target_client.
        blend_alpha: 0.0 = pure sample count, 1.0 = pure similarity.
        metric: 'cosine' or 'euclidean'.

    Returns:
        Dict mapping client_id to normalized aggregation weight.
    """
    clients = list(client_embeddings.keys())
    total_samples = sum(sample_counts[c] for c in clients)
    sample_weights = {c: sample_counts[c] / total_samples for c in clients}

    if blend_alpha <= 0.0 or len(clients) <= 1:
        return sample_weights

    # Stack embeddings
    embed_matrix = torch.stack([client_embeddings[c].float() for c in clients])  # (K, D)

    if target_client is not None:
        target_idx = clients.index(target_client)
        ref_vec = embed_matrix[target_idx]  # (D,)
        if metric == "cosine":
            ref_norm = ref_vec / (torch.norm(ref_vec) + 1e-8)
            all_norm = embed_matrix / (torch.norm(embed_matrix, dim=1, keepdim=True) + 1e-8)
            sims = (all_norm @ ref_norm).clamp(min=0.0)
        else:
            dists = torch.norm(embed_matrix - ref_vec, dim=1)
            sims = torch.exp(-dists / (dists.mean() + 1e-8))
    else:
        # Centroid similarity across all modalities
        centroid = embed_matrix.mean(dim=0)
        ref_norm = centroid / (torch.norm(centroid) + 1e-8)
        all_norm = embed_matrix / (torch.norm(embed_matrix, dim=1, keepdim=True) + 1e-8)
        sims = (all_norm @ ref_norm).clamp(min=0.0)

    sim_sum = sims.sum()
    if sim_sum > 0:
        sim_weights = {c: float(sims[i] / sim_sum) for i, c in enumerate(clients)}
    else:
        sim_weights = sample_weights

    # Blend
    blended = {}
    for c in clients:
        blended[c] = (1.0 - blend_alpha) * sample_weights[c] + blend_alpha * sim_weights[c]

    # Re-normalize
    b_sum = sum(blended.values())
    return {c: blended[c] / b_sum for c in clients}
