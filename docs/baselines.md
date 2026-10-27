# Baselines and implementation details

All methods share the same splits, backbone, LoRA targets, optimiser (AdamW, one optimiser per local round),
learning rates, number of rounds/local epochs, and evaluation protocol. Every client starts from the **same**
seeded LoRA initialisation (Kaiming-uniform A, zero B), and every client keeps a **private classification head**
(heterogeneous label spaces). Only encoder adapters are federated.

**Evaluation protocol (all methods).** After each round's aggregation, each client applies its download and is
evaluated with the resulting encoder and its own head. Validation curves and the final test numbers therefore
describe the same model state. The test split is evaluated once, after the last round, and is never used for selection.

| Key | Method | What is uploaded / downloaded | Notes and deviations |
|---|---|---|---|
| `linear_probe` | Frozen backbone + local head | nothing | No adapters (r = 0). |
| `local_only` | Local LoRA, no federation | nothing | Same budget as federated runs. |
| `centralized` | Pooled-data LoRA (privacy-violating reference) | — | One shared encoder, per-client heads, randomly interleaved batches. |
| `fedavg_full` | FedAvg over the full encoder | full encoder state | Full fine-tuning with `lr_full` (default 1e-5). |
| `fedavg_lora` | FedIT / FedAvg-LoRA | A, B / averaged A, B | Averages A and B independently (inexact). |
| `fedprox_lora` | FedProx on LoRA factors | as FedAvg-LoRA | Proximal term on A, B, anchored to the last download, `fedprox_mu` = 0.01. |
| `ffa_lora` | FFA-LoRA (Sun et al., ICLR 2024) | B / averaged B | A frozen at the shared initialisation. Original paper also tunes a larger alpha; we keep the common alpha. |
| `fedsa_lora` | FedSA-LoRA (Guo et al., ICLR 2025) | A / averaged A | B never leaves the client (personalised). |
| `fedex_lora` | FedEx-LoRA (Singhal et al., ACL 2025) | A, B / averaged A, B + dense residual | Residual folded into the frozen weight makes the update exact. Residual bytes are counted. |
| `flora` | FLoRA (Wang et al., NeurIPS 2024) | A, B / stacked factors (rank K·r) | Stacked update merged into base weights; adapters re-initialised (shared seed) each round. |
| `hetlora` | HetLoRA (Cho et al., 2024) | rank-r_k A, B / truncated global | Client ranks cycle through {16, 8, 4} (≤ r) unless `client_ranks` is given. α_k = α·r_k/r_max so every client uses scaling α/r_max. Sparsity weights ∝ ‖B_k A_k‖_F per layer. The rank self-pruning of the paper is not implemented. |
| `flexlora` | FlexLoRA (Bai et al., NeurIPS 2024) | A, B / SVD of Σ w_k s_k B_k A_k | Homogeneous ranks here, so FlexLoRA ≡ FedMedLoRA component A alone (`ablation_A_only`). |
| `fedmedlora` | Proposed | see below | |

## FedMedLoRA components

* **A — product-space aggregation + truncated SVD.** Identical to FlexLoRA's aggregation step. This is
  FlexLoRA's aggregation step; FedMedLoRA builds on it.
* **B — domain-similarity weighting.** Each client sends the mean CLS embedding of its non-augmented
  training images under the frozen backbone, computed once before training. Weights are
  `(1-β)·n_k/N + β·sim_k/Σ sim`, where sim is the clamped cosine similarity to the centroid (global
  model), or to the target client (`personalized_aggregate: true`, one SVD per client).
* **C — adaptive upload rank.** After local training, each client computes its round update
  δ_k = s·B_k A_k − (dense update it last downloaded). It uploads the rank-r_k truncated SVD of δ_k, with r_k the
  smallest candidate (≤ r) keeping `adaptive_energy_tau` of its spectral energy. The server adds δ̂_k back
  to its stored copy of that download. This reduces **upload** bytes only; local compute and download size are unchanged.

## Fairness protocol

* **Learning rates are tuned per method** on the validation split (seed 42, 10 rounds): `lr` ∈ {3e-4, 1e-3, 3e-3}
  for LoRA methods, `lr_full` ∈ {1e-5, 3e-5, 1e-4} for FedAvg-Full, `lr_head` ∈ {1e-3, 3e-3, 1e-2} for the linear probe.
  The criterion is mean validation balanced accuracy over the last 3 rounds (`scripts/select_hparams.py` writes
  `configs/tuned_hparams.json`). The test split is never used for selection. Ablation, τ and rank variants inherit FedMedLoRA's
  (or FlexLoRA's) rate.
* **Equal budget:** 20 rounds × 1 local epoch, batch 32, rank 16, same backbone and LoRA targets.
  Every client is capped at 6,000 training / 1,500 validation images by a label-stratified subsample, identical for
  all methods (affects OCT and dermoscopy). In S2 the cap applies per modality before partitioning. Test sets are never subsampled.
* **Same loss:** cross-entropy with inverse-frequency class weights computed from each client's own training labels
  (local information only; HAM10000 is 67% nevi). It is applied identically to every method (`fl.class_weighted_loss`).
* **Same precision:** bf16 autocast on CUDA for every method; aggregation, SVD and residuals in fp32.
* **Communication** is counted at fp16 (2 bytes/parameter) for every method, including dense FedEx-LoRA residuals and FedAvg-Full states.

## Settings

* **S1** — one client per modality (5 clients).
* **S2** — each modality (except histopathology) is split into 3 institutions by whole patient groups, with Dirichlet(α)
  label skew (α ∈ {0.1, 0.5}). The same per-class proportions apply to an institution's train, val and test, so each institution is
  evaluated on its own label distribution (`fl.setting: S2`, `src/fedmedlora/data/partition.py`). S2 tests whether
  similarity weighting helps when several institutions share a modality.

Not implemented: covariance-based domain distances, partial client participation, and learning-rate schedules
(constant learning rate, full participation in every round).
