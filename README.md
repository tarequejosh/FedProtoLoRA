# FedMedLoRA

Code, data splits and run logs for

> **FedMedLoRA: Similarity-aware federated low-rank adaptation of vision foundation models across heterogeneous
> medical imaging modalities** (under review).

FedMedLoRA is a federated LoRA method for institutions that image different organs with different modalities. Each
client fine-tunes LoRA adapters on a frozen vision foundation model (DINOv2 or BiomedCLIP) and keeps a private
classification head. The server sums the client updates exactly in weight space, $\overline{\Delta W}=\sum_k w_k s_k B_k A_k$,
and re-factorizes the sum by truncated SVD, as in FlexLoRA. FedMedLoRA chooses the weights from data volume and
domain similarity:

$$w_k^{(t)} = (1-\beta)\,\frac{n_k}{N} + \beta\,\frac{\mathrm{sim}_{k\to t}}{\sum_j \mathrm{sim}_{j\to t}},\qquad \beta = 0.5,$$

where $\mathrm{sim}_{k\to t}$ is the cosine similarity between the mean frozen embeddings of clients $k$ and $t$. Each
client sends this 384-dimensional vector once, before training. The weights are either shared by all clients
(global) or computed for each target client (personalized).

## Contents

```
src/fedmedlora/      library: LoRA layers, backbones, data loading and partitioning, FL simulator, methods, metrics
scripts/             data download and curation, split generation, experiment runners, result aggregation
analysis/            tables and figures of the paper, computed from results/
configs/             tuned_hparams.json (selected learning rates) and the smoke-test config
data/splits/         split manifests <client>_<seed>.csv: image, label, group, MD5, pHash, split
data/processed/      curated per-dataset manifests (input to the splits)
data/dedup_reports/  every removed or merged duplicate image, with the reason
results/<run>/       logs of each of the 260 runs: summary.json, metrics_history.jsonl, communication_cost.csv
results/summary.csv  all final metrics, one row per run and client
tests/               unit tests and the leakage audit
```

Run names have the form `<method>_<backbone>_seed<seed>`. Prefixes and infixes mark the experiment: `tune_`
(learning-rate selection), `s2_a0.1_` and `s2_a0.5_` (setting S2), `ablation_`, `rank_sweep_`, `beta_`, and `_lr…`
(learning-rate controls). [DATASETS.md](DATASETS.md) describes the datasets and the curation, and
[docs/baselines.md](docs/baselines.md) describes how each method is implemented.

## Installation

Python 3.10 or newer. The exact versions used for the paper are pinned in `requirements-lock.txt` (PyTorch 2.11.0, CUDA 12.8).

```bash
git clone https://github.com/tarequejosh/FedMedLoRA.git
cd FedMedLoRA
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

On Windows, some run folders exceed the default path-length limit; clone with
`git clone -c core.longpaths=true https://github.com/tarequejosh/FedMedLoRA.git`.

All commands below are run from the repository root.

## Tables and figures from the released logs

No GPU or image data is needed; this takes under a minute.

```bash
python analysis/make_tables.py     # LaTeX tables -> outputs/tables/; prints the statistics quoted in the paper
python analysis/make_figures.py    # Figs. 2-5 as PDF and 600-dpi PNG -> outputs/figures/
```

## Full reproduction

**1. Data.** The images are not redistributed. `download_data.py` fetches them from Kaggle into `data/raw/`, which
requires a Kaggle API token (`~/.kaggle/kaggle.json`, or the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables).

```bash
python scripts/download_data.py
```

**2. Splits.** Use the released manifests and check them:

```bash
pytest tests/test_no_leakage.py
```

or rebuild them from the raw images (grouping, MD5 and pHash de-duplication, 70/15/15 group-stratified splits for
seeds 42, 101, 2024, 7 and 1234):

```bash
python scripts/preprocess.py
python scripts/make_splits.py
pytest tests/test_no_leakage.py
```

**3. Experiments.**

```bash
python scripts/build_tensor_cache.py   # pre-decoded 224-px image cache (about 15 GB), built once
python scripts/run_pipeline.py         # learning-rate tuning, setting S1, ablations, setting S2, tau and rank sweeps
python scripts/run_controls.py         # matched learning rates, modality oracle, beta, BiomedCLIP, S2 seeds 7 and 1234
```

Both runners skip runs that already have `results/<run>/summary.json`, so they can be stopped and restarted. Single
stages are selected with `--stages`. One run with the paper's settings, for example FedMedLoRA in S1 with seed 42:

```bash
python scripts/run_all.py --suite main --methods fedmedlora --seeds 42
```

**4. Tables and figures.**

```bash
python scripts/aggregate_results.py    # -> results/summary.csv
python analysis/make_tables.py
python analysis/make_figures.py
```

**Checks.** `pytest` runs all 52 tests (about 7 minutes). `python scripts/run_pipeline.py --smoke` runs every stage
on a tiny toy backbone with 12 images per client, writing to `results_smoke/`.

## Compute

The 260 runs (including 39 tuning runs) took about 88 h on one workstation:

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5060, 8 GB |
| CPU | Intel Core Ultra 7 265F, 20 cores |
| Memory | 64 GB DDR5-4800 |
| System storage | 512 GB NVMe SSD |
| Data storage | 1 TB SATA hard disk |
| Software | Windows 11, Python 3.12.10, PyTorch 2.11.0, CUDA 12.8, cuDNN 9.19 |

A 20-round run takes about 20 min with DINOv2 ViT-S/14 and about 40 min with BiomedCLIP ViT-B/16. The seed fixes
the split, the Dirichlet partition, the adapter initialization and the data order, and cuDNN runs in deterministic
mode. Other GPUs or library versions may still give slightly different numbers, because bfloat16 arithmetic and some
GPU kernels are not bit-identical across hardware.

## Methods

| Method | Key in `run_all.py` |
|---|---|
| FedMedLoRA, global / personalized | `fedmedlora` (`personalized_aggregate: true`) |
| FlexLoRA | `flexlora` |
| FedAvg-LoRA (FedIT) | `fedavg_lora` |
| FedProx-LoRA | `fedprox_lora` |
| FFA-LoRA | `ffa_lora` |
| FedSA-LoRA | `fedsa_lora` |
| FedEx-LoRA | `fedex_lora` |
| FLoRA | `flora` |
| HetLoRA | `hetlora` |
| FedAvg, full fine-tuning | `fedavg_full` |
| Local-only LoRA, centralized LoRA, linear probe | `local_only`, `centralized`, `linear_probe` |
| Modality oracle (S2) | `flexlora` with `oracle_modality_clusters: true` |

## Privacy

Clients exchange LoRA adapters and, for FedMedLoRA, one mean embedding per client. No images leave a client, but
neither the adapters nor the embedding carry a formal privacy guarantee.

## Citation

The citation will be added when the paper is published.

## License

The code is released under the MIT License. The datasets keep their original terms; the files in `data/` contain only
file names, labels, group identifiers and hashes.
