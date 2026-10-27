"""Federated aggregation methods and baselines for FedMedLoRA."""

from .fedmedlora import (
    aggregate_product_space_svd,
    compress_update_adaptive_rank,
    compute_spectral_adaptive_rank,
    compute_domain_similarity_weights,
    select_rank_by_energy,
    svd_refactor,
)
from .baselines import (
    aggregate_fedavg_lora,
    aggregate_fedex_lora,
    aggregate_fedsa_lora,
    aggregate_ffa_lora,
    aggregate_flora,
    aggregate_hetlora,
    hetlora_sparsity_weights,
    truncate_factors,
    compute_fedprox_loss,
)

__all__ = [
    "aggregate_product_space_svd",
    "compress_update_adaptive_rank",
    "select_rank_by_energy",
    "svd_refactor",
    "compute_spectral_adaptive_rank",
    "compute_domain_similarity_weights",
    "aggregate_fedavg_lora",
    "aggregate_fedex_lora",
    "aggregate_fedsa_lora",
    "aggregate_ffa_lora",
    "aggregate_flora",
    "aggregate_hetlora",
    "hetlora_sparsity_weights",
    "truncate_factors",
    "compute_fedprox_loss",
]
