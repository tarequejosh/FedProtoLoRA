"""Federated server coordinating aggregation, domain weighting, and per-client downloads."""

from typing import Dict, List, Optional, Tuple
import torch

from ..methods.fedmedlora import (
    compute_domain_similarity_weights,
    svd_refactor,
)
from ..methods.baselines import (
    aggregate_fedavg_lora,
    aggregate_fedex_lora,
    aggregate_fedsa_lora,
    aggregate_ffa_lora,
    aggregate_flora,
    aggregate_hetlora,
    hetlora_sparsity_weights,
    truncate_factors,
)

LoRAState = Dict[str, Dict[str, torch.Tensor]]

# Methods whose server aggregates in the dense product space and re-factorises by truncated SVD.
PRODUCT_SPACE_METHODS = ("fedmedlora", "flexlora")
LORA_AGGREGATION_METHODS = PRODUCT_SPACE_METHODS + (
    "fedavg_lora", "fedprox_lora", "ffa_lora", "fedsa_lora", "fedex_lora", "flora", "hetlora",
)
NO_AGGREGATION_METHODS = ("local_only", "linear_probe")


class FederatedServer:
    """Central federated coordinator for FedProtoLoRA and all PEFT baselines.

    `aggregate` returns a per-client download dict {client_id: LoRA state}. For non-personalised
    methods every client receives the same object.
    """

    def __init__(
        self,
        method: str = "fedmedlora",
        target_rank: int = 16,
        target_alpha: float = 16.0,
        blend_alpha: float = 0.5,
        similarity_metric: str = "cosine",
        personalized: bool = False,
        device: str = "cpu",
        client_groups: Optional[Dict[str, str]] = None,
    ):
        self.method = method.lower()
        self.target_rank = target_rank
        self.target_alpha = target_alpha
        self.blend_alpha = blend_alpha
        self.similarity_metric = similarity_metric
        self.personalized = personalized
        # Oracle baseline: aggregate only within known groups (e.g. modality), sample-count weighted.
        self.client_groups = client_groups
        self.device = device
        self.global_lora_state: LoRAState = {}
        # Dense Delta W each client last received (product-space methods); deltas are uploaded against it.
        self.client_ref_delta_w: Dict[str, Dict[str, torch.Tensor]] = {}
        self.last_weights: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------ weights
    def _sample_weights(self, clients: List[str], sample_counts: Dict[str, int]) -> Dict[str, float]:
        total = sum(sample_counts[c] for c in clients)
        return {c: sample_counts[c] / total for c in clients}

    def _client_weights(
        self,
        clients: List[str],
        sample_counts: Dict[str, int],
        domain_embeddings: Optional[Dict[str, torch.Tensor]],
        target_client: Optional[str] = None,
    ) -> Dict[str, float]:
        if self.method == "fedmedlora" and domain_embeddings and self.blend_alpha > 0.0:
            return compute_domain_similarity_weights(
                {c: domain_embeddings[c] for c in clients},
                {c: sample_counts[c] for c in clients},
                target_client=target_client,
                blend_alpha=self.blend_alpha,
                metric=self.similarity_metric,
            )
        return self._sample_weights(clients, sample_counts)

    # ------------------------------------------------------------------ aggregation
    def aggregate(
        self,
        client_updates: Dict[str, LoRAState],
        sample_counts: Dict[str, int],
        domain_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        round_idx: int = 0,
    ) -> Dict[str, LoRAState]:
        """Aggregates client LoRA uploads and returns the download for every client."""
        clients = list(client_updates.keys())
        if not clients or self.method in NO_AGGREGATION_METHODS:
            return {}
        if self.method not in LORA_AGGREGATION_METHODS:
            raise ValueError(f"Unknown aggregation method: {self.method}")

        if self.method in PRODUCT_SPACE_METHODS:
            return self._aggregate_product_space(client_updates, sample_counts, domain_embeddings)
        if self.method == "hetlora":
            return self._aggregate_hetlora(client_updates)

        weights_dict = self._client_weights(clients, sample_counts, None)
        self.last_weights = {"shared": weights_dict}
        weights = [weights_dict[c] for c in clients]
        scaling = self.target_alpha / self.target_rank
        first_client = clients[0]

        aggregated_state: LoRAState = {}
        for layer_name in client_updates[first_client].keys():
            entries = [client_updates[c][layer_name] for c in clients]
            factors = [(e["A"].to(self.device), e["B"].to(self.device)) for e in entries]

            if self.method in ("fedavg_lora", "fedprox_lora"):
                A_g, B_g = aggregate_fedavg_lora(factors, weights)
                aggregated_state[layer_name] = {"A": A_g.cpu(), "B": B_g.cpu(), "r": self.target_rank, "scaling": scaling}

            elif self.method == "ffa_lora":
                B_g = aggregate_ffa_lora([B for _, B in factors], weights)
                # A is the shared frozen initialisation (identical on every client).
                aggregated_state[layer_name] = {
                    "A": entries[0]["A"].cpu(), "B": B_g.cpu(), "r": self.target_rank, "scaling": scaling,
                }

            elif self.method == "fedsa_lora":
                A_g = aggregate_fedsa_lora([A for A, _ in factors], weights)
                aggregated_state[layer_name] = {"A": A_g.cpu(), "r": self.target_rank, "scaling": scaling}

            elif self.method == "fedex_lora":
                A_g, B_g, residual = aggregate_fedex_lora(factors, weights, scaling)
                aggregated_state[layer_name] = {
                    "A": A_g.cpu(), "B": B_g.cpu(), "residual": residual.cpu(), "r": self.target_rank, "scaling": scaling,
                }

            elif self.method == "flora":
                A_s, B_s = aggregate_flora(factors, weights, scalings=[e["scaling"] for e in entries])
                aggregated_state[layer_name] = {
                    "A": A_s.cpu(), "B": B_s.cpu(), "r": A_s.shape[0], "scaling": 1.0, "merge": True,
                }

        self.global_lora_state = aggregated_state
        return {c: aggregated_state for c in clients}

    def _aggregate_product_space(
        self,
        client_updates: Dict[str, LoRAState],
        sample_counts: Dict[str, int],
        domain_embeddings: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, LoRAState]:
        """FedProtoLoRA / FlexLoRA: dense weighted sum of client Delta W_k, then truncated SVD to target rank.

        Uploads flagged `is_delta` (component C) are reconstructed as Delta W_k = ref_k + s_k B_k A_k,
        where ref_k is the dense update the server last sent to client k.
        """
        clients = list(client_updates.keys())
        layer_names = list(client_updates[clients[0]].keys())

        if self.client_groups:
            targets = {}
            for c in clients:
                peers = [k for k in clients if self.client_groups[k] == self.client_groups[c]]
                total = sum(sample_counts[k] for k in peers)
                targets[c] = {k: (sample_counts[k] / total if k in peers else 0.0) for k in clients}
        elif self.personalized:
            targets = {c: self._client_weights(clients, sample_counts, domain_embeddings, target_client=c) for c in clients}
        else:
            shared = self._client_weights(clients, sample_counts, domain_embeddings)
            targets = {"shared": shared}
        self.last_weights = targets

        downloads: Dict[str, LoRAState] = {key: {} for key in targets}
        new_refs: Dict[str, Dict[str, torch.Tensor]] = {c: {} for c in clients}

        for layer_name in layer_names:
            dense = {}
            for c in clients:
                e = client_updates[c][layer_name]
                dw = (e["B"].to(self.device).float() @ e["A"].to(self.device).float()) * float(e["scaling"])
                if e.get("is_delta", False):
                    ref = self.client_ref_delta_w.get(c, {}).get(layer_name)
                    if ref is not None:
                        dw = dw + ref.to(self.device)
                dense[c] = dw

            for key, w in targets.items():
                delta_w = sum(dense[c] * float(w[c]) for c in clients)
                A_g, B_g, s_g = svd_refactor(delta_w, self.target_rank, self.target_alpha)
                downloads[key][layer_name] = {"A": A_g.cpu(), "B": B_g.cpu(), "r": self.target_rank, "scaling": s_g}
                received = (B_g @ A_g) * s_g
                for c in (clients if key == "shared" else [key]):
                    new_refs[c][layer_name] = received

        self.client_ref_delta_w = new_refs
        if self.personalized or self.client_groups:
            self.global_lora_state = {}
            return downloads
        self.global_lora_state = downloads["shared"]
        return {c: downloads["shared"] for c in clients}

    def _aggregate_hetlora(self, client_updates: Dict[str, LoRAState]) -> Dict[str, LoRAState]:
        """HetLoRA: zero-pad to the max rank, sparsity-weighted sum, truncate back to each client's rank."""
        clients = list(client_updates.keys())
        max_rank = max(e["A"].shape[0] for u in client_updates.values() for e in u.values())
        downloads: Dict[str, LoRAState] = {c: {} for c in clients}
        for layer_name in client_updates[clients[0]].keys():
            entries = [client_updates[c][layer_name] for c in clients]
            factors = [(e["A"].to(self.device), e["B"].to(self.device)) for e in entries]
            weights = hetlora_sparsity_weights(factors)
            A_g, B_g = aggregate_hetlora(factors, weights, max_rank=max_rank)
            for c, e in zip(clients, entries):
                A_c, B_c = truncate_factors(A_g, B_g, e["A"].shape[0])
                downloads[c][layer_name] = {"A": A_c.cpu(), "B": B_c.cpu(), "r": A_c.shape[0], "scaling": e["scaling"]}
        self.global_lora_state = {}
        return downloads

    @staticmethod
    def aggregate_full_state(
        client_states: Dict[str, Dict[str, torch.Tensor]],
        sample_counts: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """FedAvg over full encoder state dicts (FedAvg-Full baseline); non-float buffers are copied."""
        clients = list(client_states.keys())
        total = sum(sample_counts[c] for c in clients)
        averaged = {}
        for key, ref in client_states[clients[0]].items():
            if torch.is_floating_point(ref):
                averaged[key] = sum(client_states[c][key].float() * (sample_counts[c] / total) for c in clients).to(ref.dtype)
            else:
                averaged[key] = ref.clone()
        return averaged
