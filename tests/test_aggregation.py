"""Unit tests for product-space SVD aggregation and domain weighting math."""

import pytest
import torch
from fedmedlora.methods.fedmedlora import (
    aggregate_product_space_svd,
    compute_spectral_adaptive_rank,
    compute_domain_similarity_weights,
    compress_update_adaptive_rank,
    svd_refactor,
)
from fedmedlora.fl.server import FederatedServer


def test_svd_aggregation_identical_A():
    """Theorem test: When clients share an identical A matrix,

    product-space aggregation dW = sum w_k (B_k @ A) exactly equals (sum w_k B_k) @ A.
    """
    torch.manual_seed(42)
    in_features = 64
    out_features = 128
    rank = 8
    alpha = 16.0
    scaling = alpha / rank

    # Shared A matrix across 3 clients
    A_shared = torch.randn(rank, in_features)

    # 3 distinct B matrices
    B1 = torch.randn(out_features, rank)
    B2 = torch.randn(out_features, rank)
    B3 = torch.randn(out_features, rank)

    weights = [0.5, 0.3, 0.2]
    client_factors = [
        (A_shared, B1, scaling),
        (A_shared, B2, scaling),
        (A_shared, B3, scaling),
    ]

    # Expected exact delta_W
    B_weighted = B1 * weights[0] + B2 * weights[1] + B3 * weights[2]
    expected_dW = (B_weighted @ A_shared) * scaling

    # SVD product aggregation
    A_glob, B_glob, glob_scaling = aggregate_product_space_svd(
        client_factors, weights, target_rank=rank, target_alpha=alpha
    )
    reproj_dW = (B_glob @ A_glob) * glob_scaling

    # Since rank(expected_dW) <= rank, SVD re-projection with target_rank=rank is mathematically exact
    assert torch.allclose(reproj_dW, expected_dW, atol=1e-4)


def test_svd_reprojection_error():
    """Tests that low-rank SVD re-projection error decreases as target rank increases."""
    torch.manual_seed(42)
    in_features = 64
    out_features = 64
    r = 16

    # 2 clients with arbitrary different A and B
    A1 = torch.randn(r, in_features)
    B1 = torch.randn(out_features, r)
    A2 = torch.randn(r, in_features)
    B2 = torch.randn(out_features, r)

    weights = [0.6, 0.4]
    client_factors = [(A1, B1, 1.0), (A2, B2, 1.0)]

    dW_true = 0.6 * (B1 @ A1) + 0.4 * (B2 @ A2)

    errors = []
    for test_rank in [4, 8, 16, 32]:
        A_g, B_g, s_g = aggregate_product_space_svd(
            client_factors, weights, target_rank=test_rank, target_alpha=float(test_rank)
        )
        dW_reproj = (B_g @ A_g) * s_g
        err = torch.norm(dW_true - dW_reproj).item()
        errors.append(err)

    # Error strictly decreases as rank increases
    assert errors[0] > errors[1] >= errors[2]
    # For test_rank=32 (which equals max possible rank of sum of two rank-16 updates), error should be near zero
    assert errors[-1] < 5e-4


def test_spectral_adaptive_rank_selection():
    """Tests that spectral rank selection picks a smaller rank when update has low effective rank."""
    torch.manual_seed(42)
    in_features = 32
    out_features = 32

    # Construct an update that only has 4 dominant singular values
    U = torch.randn(out_features, 4)
    V = torch.randn(4, in_features)
    # Factorize into A (r=16) and B (r=16) where only 4 directions have energy
    dW = U @ V  # rank 4
    U_full, S, Vh = torch.linalg.svd(dW, full_matrices=False)

    A = torch.zeros(16, in_features)
    B = torch.zeros(out_features, 16)
    for i in range(4):
        B[:, i] = U_full[:, i] * (S[i] ** 0.5)
        A[i, :] = Vh[i, :] * (S[i] ** 0.5)

    chosen_r = compute_spectral_adaptive_rank(
        A, B, scaling=1.0, energy_tau=0.95, rank_candidates=(4, 8, 16, 32)
    )
    assert chosen_r == 4


def test_domain_similarity_weights():
    """Tests domain similarity weighting with blend_alpha."""
    torch.manual_seed(42)
    # 3 clients: derm, histo, cxr
    # derm and histo have very close embeddings, cxr is distant
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([0.9, 0.1, 0.0])
    v3 = torch.tensor([0.0, 0.0, 1.0])

    embeds = {"c1": v1, "c2": v2, "c3": v3}
    counts = {"c1": 100, "c2": 100, "c3": 100}

    # At blend_alpha = 0.0, pure uniform sample weights
    w_pure = compute_domain_similarity_weights(embeds, counts, blend_alpha=0.0)
    assert pytest.approx(w_pure["c1"]) == 1/3
    assert pytest.approx(w_pure["c2"]) == 1/3
    assert pytest.approx(w_pure["c3"]) == 1/3

    # At blend_alpha = 0.5 with target_client='c1', c2 should have higher weight than c3
    w_blended = compute_domain_similarity_weights(embeds, counts, target_client="c1", blend_alpha=0.5)
    assert w_blended["c2"] > w_blended["c3"]


def test_svd_refactor_matches_scaling_and_padding():
    torch.manual_seed(0)
    dW = torch.randn(32, 6) @ torch.randn(6, 24)  # rank 6
    A, B, s = svd_refactor(dW, target_rank=8, target_alpha=16.0)
    assert A.shape == (8, 24) and B.shape == (32, 8) and s == 2.0
    assert torch.allclose(s * B @ A, dW, atol=1e-4)


def test_adaptive_rank_compression_picks_low_rank():
    torch.manual_seed(0)
    dW = torch.randn(64, 3) @ torch.randn(3, 64)
    A, B, s, r = compress_update_adaptive_rank(dW, lora_alpha=16.0, energy_tau=0.95, rank_candidates=(4, 8, 16))
    assert r == 4
    assert torch.allclose(s * B @ A, dW, atol=1e-3)


def _random_state(rank, scaling, seed):
    g = torch.Generator().manual_seed(seed)
    return {"layer": {"A": torch.randn(rank, 24, generator=g), "B": torch.randn(32, rank, generator=g), "r": rank, "scaling": scaling}}


def test_server_delta_uploads_equal_full_uploads():
    """Uploading exact round deltas against the last download must give the same aggregate as full factors."""
    counts = {"c1": 10, "c2": 30}
    full = FederatedServer(method="flexlora", target_rank=8, target_alpha=16.0)
    delta = FederatedServer(method="flexlora", target_rank=8, target_alpha=16.0)

    # Round 1 (reference = 0): both servers see identical full uploads.
    r1 = {"c1": _random_state(8, 2.0, 1), "c2": _random_state(8, 2.0, 2)}
    d_full = full.aggregate(r1, counts)
    delta.aggregate(r1, counts)

    # Round 2: local states after training; the delta server receives exact (uncompressed) differences.
    r2 = {"c1": _random_state(8, 2.0, 3), "c2": _random_state(8, 2.0, 4)}
    d_full = full.aggregate(r2, counts)
    deltas = {}
    for c, st in r2.items():
        e = st["layer"]
        diff = e["scaling"] * e["B"] @ e["A"] - delta.client_ref_delta_w[c]["layer"]
        A, B, s = svd_refactor(diff, 16, 16.0)
        deltas[c] = {"layer": {"A": A, "B": B, "r": 16, "scaling": s, "is_delta": True}}
    d_delta = delta.aggregate(deltas, counts)

    for c in counts:
        a, b = d_full[c]["layer"], d_delta[c]["layer"]
        assert torch.allclose(a["scaling"] * a["B"] @ a["A"], b["scaling"] * b["B"] @ b["A"], atol=1e-3)


def test_personalized_aggregation_differs_per_client():
    counts = {"c1": 10, "c2": 10, "c3": 10}
    emb = {"c1": torch.tensor([1.0, 0.0]), "c2": torch.tensor([0.9, 0.1]), "c3": torch.tensor([0.0, 1.0])}
    server = FederatedServer(method="fedmedlora", target_rank=8, blend_alpha=1.0, personalized=True)
    ups = {c: _random_state(8, 2.0, i) for i, c in enumerate(counts)}
    downs = server.aggregate(ups, counts, domain_embeddings=emb)
    assert not torch.allclose(downs["c1"]["layer"]["B"], downs["c3"]["layer"]["B"])
    assert server.last_weights["c1"]["c3"] < server.last_weights["c1"]["c2"]


def test_oracle_modality_clusters_aggregate_within_group_only():
    counts = {"derm_0": 10, "derm_1": 30, "oct_0": 20}
    groups = {"derm_0": "derm", "derm_1": "derm", "oct_0": "oct"}
    server = FederatedServer(method="flexlora", target_rank=8, client_groups=groups)
    ups = {c: _random_state(8, 2.0, i) for i, c in enumerate(counts)}
    downs = server.aggregate(ups, counts)
    assert server.last_weights["derm_0"] == {"derm_0": 0.25, "derm_1": 0.75, "oct_0": 0.0}
    assert server.last_weights["oct_0"]["oct_0"] == 1.0
    e = ups["oct_0"]["layer"]  # a singleton group receives exactly its own (rank-8) update back
    d = downs["oct_0"]["layer"]
    assert torch.allclose(d["scaling"] * d["B"] @ d["A"], e["scaling"] * e["B"] @ e["A"], atol=1e-3)
