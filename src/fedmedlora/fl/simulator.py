"""Federated simulation loop orchestrating clients, server, rounds, evaluation, and logging.

Evaluation protocol (identical for every method): after each round's aggregation, every client applies
its download and is evaluated with the resulting shared/personalised encoder and its own local head.
Validation curves and the final test metrics therefore measure the same model state. For methods
without aggregation (local_only, linear_probe, centralized) the current local model is evaluated.
"""

import platform
import random
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.backbones import VisionFoundationModel, FederatedClientModel, get_backbone_normalization
from ..models.lora import LoRALinear, extract_lora_state, make_fresh_lora_state
from ..data.dataset import MedicalImageDataset, get_medical_transforms
from ..data.partition import dirichlet_group_partition, stratified_subsample
from ..methods.baselines import truncate_factors
from ..utils.seed import set_seed
from ..utils.comm import CommunicationTracker
from ..utils.logging import ExperimentLogger
from .client import FederatedClient
from .server import FederatedServer, NO_AGGREGATION_METHODS

# Modality class count mapping
CLIENT_NUM_CLASSES = {
    "derm": 7,
    "oct": 4,
    "histo": 2,
    "cxr": 2,
    "brain": 4,
}

KNOWN_METHODS = (
    "fedmedlora", "flexlora", "fedavg_lora", "fedprox_lora", "ffa_lora", "fedsa_lora", "fedex_lora",
    "flora", "hetlora", "local_only", "linear_probe", "fedavg_full", "centralized",
)


def default_hetlora_ranks(client_names: List[str], max_rank: int, candidates=(4, 8, 16, 32)) -> Dict[str, int]:
    """Deterministic heterogeneous rank assignment (cycles through candidates <= max_rank, largest first)."""
    usable = sorted([c for c in candidates if c <= max_rank], reverse=True) or [max_rank]
    return {c: usable[i % len(usable)] for i, c in enumerate(client_names)}


def _train_centralized_round(clients: Dict[str, FederatedClient], local_epochs: int, lr: float, lr_head: float, weight_decay: float) -> float:
    """Pooled-data upper bound: one shared encoder (LoRA), per-client heads, randomly interleaved batches."""
    first = next(iter(clients.values()))
    lora_params = [p for m in first._lora_modules().values() for p in (m.lora_A, m.lora_B)]
    head_params = [p for c in clients.values() for p in c.model.head.parameters()]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ])
    total, n = 0.0, 0
    for c in clients.values():
        c.model.train()
    for _ in range(local_epochs):
        schedule = [name for name, c in clients.items() for _ in range(len(c.train_loader))]
        random.shuffle(schedule)
        iters = {name: iter(c.train_loader) for name, c in clients.items()}
        for name in schedule:
            images, targets, _ = next(iters[name])
            client = clients[name]
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=client.amp):
                logits = client.model(client.prepare_batch(images, train=True)).float()
            loss = client.criterion(logits, targets.to(client.device))  # client's own (class-weighted) loss
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
    return total / max(1, n)


def run_federated_simulation(config: Dict) -> Dict:
    """Executes a full federated training run from a configuration dictionary.

    Returns:
        Dict: Final test metrics, communication statistics, and metadata.
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    is_smoke = config.get("smoke", False)
    fl_cfg = config.get("fl", {})
    model_cfg = config.get("model", {})
    method_cfg = config.get("method", {})

    method_name = method_cfg.get("name", "fedmedlora").lower()
    if method_name not in KNOWN_METHODS:
        raise ValueError(f"Unknown method '{method_name}'. Known: {KNOWN_METHODS}")
    num_rounds = fl_cfg.get("num_rounds", 2 if is_smoke else 20)
    local_epochs = fl_cfg.get("local_epochs", 1 if is_smoke else 2)
    batch_size = fl_cfg.get("batch_size", 4 if is_smoke else 32)
    # On Windows every DataLoader worker re-imports torch (slow spawn, large commit charge): with the in-memory
    # cache, 0 workers is both the fastest and the safest default.
    num_workers = fl_cfg.get("num_workers", 0)
    cache_in_memory = fl_cfg.get("cache_in_memory", True)
    # Batched flip/rotation/normalisation on the GPU (datasets yield uint8); CPU augmentation was the bottleneck.
    gpu_augment = fl_cfg.get("gpu_augment", True) and device == "cuda"
    eval_num_workers = fl_cfg.get("eval_num_workers", 0)
    max_train_samples = fl_cfg.get("max_train_samples_per_client")
    max_val_samples = fl_cfg.get("max_val_samples_per_client")
    setting = fl_cfg.get("setting", "S1")
    lr = fl_cfg.get("lr", 0.0005)
    lr_head = fl_cfg.get("lr_head", 0.001)
    lr_full = fl_cfg.get("lr_full", 1e-5)
    weight_decay = fl_cfg.get("weight_decay", 0.01)

    target_rank = model_cfg.get("r", 4 if is_smoke else 16)
    lora_alpha = model_cfg.get("lora_alpha", 16.0)
    lora_dropout = model_cfg.get("lora_dropout", 0.0)
    head_hidden_dim = model_cfg.get("head_hidden_dim")
    backbone_name = model_cfg.get("backbone", "toy" if is_smoke else "dinov2_small")

    # FedProtoLoRA components (FlexLoRA = product-space SVD with plain sample-count weights, no B / C).
    is_fedmedlora = method_name == "fedmedlora"
    use_domain_similarity = is_fedmedlora and method_cfg.get("use_domain_similarity", True)
    use_adaptive_rank = is_fedmedlora and method_cfg.get("use_adaptive_rank", True)
    blend_alpha = method_cfg.get("blend_alpha", 0.5) if use_domain_similarity else 0.0
    personalized = is_fedmedlora and method_cfg.get("personalized_aggregate", False)
    rank_candidates = [r for r in method_cfg.get("rank_candidates", [4, 8, 16, 32]) if r <= target_rank] or [target_rank]
    adaptive_cfg = {
        "lora_alpha": lora_alpha,
        "energy_tau": method_cfg.get("adaptive_energy_tau", 0.95),
        "rank_candidates": rank_candidates,
    } if use_adaptive_rank else None

    exp_name = config.get("experiment_name", "unnamed_run")
    results_dir = Path(config.get("results_dir", "results")) / exp_name
    logger = ExperimentLogger(results_dir)
    comm_tracker = CommunicationTracker(precision_bytes=config.get("comm_precision_bytes", 2))

    modalities = fl_cfg.get("clients", ["brain", "cxr"])
    splits_dir = Path(config.get("splits_dir", "data/splits"))
    use_amp = fl_cfg.get("amp", True) and device == "cuda"
    # Class imbalance (e.g. HAM10000: 67% nevi) is handled locally with inverse-frequency loss weights.
    class_weighted_loss = fl_cfg.get("class_weighted_loss", True)

    # Shared pre-decoded tensor caches (scripts/build_tensor_cache.py); only used with GPU augmentation.
    tensor_cache_dir = Path(fl_cfg.get("tensor_cache_dir", "data/tensor_cache"))

    def _load_tensor_cache(modality: str):
        npy, idx = tensor_cache_dir / f"{modality}_224.npy", tensor_cache_dir / f"{modality}_224_index.csv"
        if not (gpu_augment and npy.exists() and idx.exists()):
            return None
        index = pd.read_csv(idx)
        return np.load(npy, mmap_mode="r"), dict(zip(index["image_path"].astype(str), index["row"].astype(int)))

    def _cap_modality(manifest: pd.DataFrame) -> pd.DataFrame:
        """S2: apply the per-client caps at modality level *before* partitioning (equal total budget to S1)."""
        parts = []
        for split, cap in (("train", max_train_samples), ("val", max_val_samples), ("test", None)):
            parts.append(stratified_subsample(manifest[manifest["split"] == split], cap, seed))
        return pd.concat(parts).reset_index(drop=True)

    # Client specs: S1 = one client per modality; S2 = Dirichlet label-skewed sub-clients per modality.
    client_specs: Dict[str, Dict] = {}
    tensor_caches = {}
    for modality in modalities:
        manifest_path = splits_dir / f"{modality}_{seed}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        tensor_caches[modality] = _load_tensor_cache(modality)
        if setting == "S2":
            manifest = _cap_modality(manifest)
            parts = dirichlet_group_partition(
                manifest,
                num_clients=fl_cfg.get("sub_clients_per_modality", 3),
                alpha=fl_cfg.get("dirichlet_alpha", 0.5),
                seed=seed,
            )
            for k, part in enumerate(parts):
                client_specs[f"{modality}_{k}"] = {"modality": modality, "manifest": part}
        else:
            client_specs[modality] = {"modality": modality, "manifest": manifest}
    client_names = list(client_specs.keys())

    hetlora_ranks = {}
    if method_name == "hetlora":
        hetlora_ranks = method_cfg.get("client_ranks") or default_hetlora_ranks(client_names, target_rank)

    # Setup clients
    mean, std = get_backbone_normalization(backbone_name)
    train_transforms = get_medical_transforms(image_size=224, is_train=True, mean=list(mean), std=list(std))
    eval_transforms = get_medical_transforms(image_size=224, is_train=False, mean=list(mean), std=list(std))

    shared_encoder: Optional[VisionFoundationModel] = None
    clients: Dict[str, FederatedClient] = {}
    print(f"\n[{exp_name}] Initializing {len(client_names)} clients on device={device} (method={method_name})...")
    for c_name, spec in client_specs.items():
        manifest = spec["manifest"]
        num_classes = CLIENT_NUM_CLASSES.get(spec["modality"], 2)

        train_df = stratified_subsample(manifest[manifest["split"] == "train"], max_train_samples, seed)
        val_df = stratified_subsample(manifest[manifest["split"] == "val"], max_val_samples, seed)
        # Decoded-once uint8 cache for the splits read every round (augmentation still applied per access).
        tcache = tensor_caches[spec["modality"]]
        # With the shared tensor cache the per-run in-memory cache is redundant (the OS page cache serves reads).
        ds_kw = {"cache_in_memory": cache_in_memory and tcache is None, "return_uint8": gpu_augment, "tensor_cache": tcache}
        train_ds = MedicalImageDataset(train_df, transform=train_transforms, **ds_kw)
        summary_ds = MedicalImageDataset(train_df, transform=eval_transforms, **ds_kw)
        val_ds = MedicalImageDataset(val_df, transform=eval_transforms, **ds_kw)
        test_ds = MedicalImageDataset(manifest, split="test", transform=eval_transforms, return_uint8=gpu_augment,
                                      tensor_cache=tcache)
        if is_smoke:
            limit = fl_cfg.get("smoke_samples_per_client", 12)
            # Random (not head()) subsample: manifests are sorted by class, head() would give one class.
            for ds in (train_ds, summary_ds, val_ds, test_ds):
                ds.df = ds.df.sample(n=min(limit, len(ds.df)), random_state=seed).reset_index(drop=True)
        summary_ds.cached_images = train_ds.cached_images  # same rows, same order -> share the decoded cache

        def _loader(ds, shuffle, workers):
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                              pin_memory=device == "cuda", persistent_workers=workers > 0)

        # Persistent workers only for loaders used every round (train / val); one-off loaders stay in-process.
        train_loader = _loader(train_ds, True, num_workers)
        summary_loader = _loader(summary_ds, False, 0)
        val_loader = _loader(val_ds, False, eval_num_workers)
        test_loader = _loader(test_ds, False, 0)

        # Per-method encoder configuration
        enc_r, enc_alpha, full_ft = target_rank, lora_alpha, False
        if method_name == "linear_probe":
            enc_r = 0
        elif method_name == "fedavg_full":
            enc_r, full_ft = 0, True
        elif method_name == "hetlora":
            enc_r = hetlora_ranks[c_name]
            # Keep scaling = alpha / r_max on every client so zero-padded factors are commensurable.
            enc_alpha = lora_alpha * enc_r / target_rank

        if method_name == "centralized" and shared_encoder is not None:
            encoder = shared_encoder
        else:
            encoder = VisionFoundationModel(
                backbone_name=backbone_name,
                pretrained=not is_smoke,
                r=enc_r,
                lora_alpha=enc_alpha,
                lora_dropout=lora_dropout,
                is_smoke=is_smoke,
                full_finetune=full_ft,
            )
            if method_name == "centralized":
                shared_encoder = encoder
        client_model = FederatedClientModel(encoder=encoder, num_classes=num_classes, hidden_dim=head_hidden_dim)

        clients[c_name] = FederatedClient(
            client_id=c_name,
            model=client_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=lr,
            lr_head=lr_head,
            weight_decay=weight_decay,
            device=device,
            summary_loader=summary_loader,
            amp=use_amp,
            class_weighted_loss=class_weighted_loss,
            norm_mean=list(mean),
            norm_std=list(std),
        )
        print(f"  Client '{c_name}': train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, r={enc_r}")

    first_client = clients[client_names[0]]

    # Common starting point: identical frozen backbone (matters for randomly initialised smoke backbones)
    # and identical LoRA initialisation on every client.
    if method_name != "centralized":
        base_state = {k: v for k, v in first_client.model.encoder.state_dict().items() if "lora_" not in k}
        for client in clients.values():
            client.model.encoder.load_state_dict(base_state, strict=False)

    lora_template = {}
    if method_name not in ("linear_probe", "fedavg_full"):
        if method_name == "hetlora":
            max_r = max(hetlora_ranks.values())
            template = {
                name: {"A": torch.zeros(max_r, m.in_features), "B": torch.zeros(m.out_features, max_r), "r": max_r, "scaling": m.scaling}
                for name, m in first_client._lora_modules().items()
            }
            init_state = make_fresh_lora_state(template, seed)
            for c_name, client in clients.items():
                client.set_lora_factors({
                    name: dict(zip(("A", "B"), truncate_factors(e["A"], e["B"], hetlora_ranks[c_name])))
                    for name, e in init_state.items()
                })
        else:
            lora_template = extract_lora_state(first_client.model.encoder)
            init_state = make_fresh_lora_state(lora_template, seed)
            for client in clients.values():
                client.set_lora_factors(init_state)

    # Instantiate server
    server = FederatedServer(
        method=method_name,
        target_rank=target_rank,
        target_alpha=lora_alpha,
        blend_alpha=blend_alpha,
        similarity_metric=method_cfg.get("similarity_metric", "cosine"),
        personalized=personalized,
        device=device,
        # Oracle modality clustering (FlexLoRA within each modality); modality is known metadata in S2.
        client_groups={c: spec["modality"] for c, spec in client_specs.items()}
        if method_cfg.get("oracle_modality_clusters", False) else None,
    )

    # Domain summaries are computed once, before training (B = 0, i.e. frozen foundation-model features).
    domain_embeddings = {}
    if use_domain_similarity:
        print(f"[{exp_name}] Extracting frozen-backbone domain embeddings for similarity weighting...")
        for c_name, client in clients.items():
            domain_embeddings[c_name] = client.compute_domain_summary()

    # Training rounds loop
    print(f"\n[{exp_name}] Starting training ({num_rounds} rounds x {local_epochs} local epochs, method={method_name})...")
    upload_rank_history = []
    for r in range(1, num_rounds + 1):
        round_uploads: Dict[str, int] = {}
        round_downloads: Dict[str, int] = {}
        client_updates = {}
        full_states = {}
        sample_counts = {c: len(client.train_loader.dataset) for c, client in clients.items()}
        round_losses = {}

        # 1. Local training + upload
        if method_name == "centralized":
            round_losses["pooled"] = _train_centralized_round(clients, local_epochs, lr, lr_head, weight_decay)
        else:
            for c_name, client in clients.items():
                round_losses[c_name] = client.train_round(
                    local_epochs=local_epochs,
                    method=method_name,
                    mu=method_cfg.get("fedprox_mu", 0.01),
                    freeze_A=(method_name == "ffa_lora"),
                    lr_encoder_full=lr_full,
                )
                if method_name == "fedavg_full":
                    full_states[c_name] = client.get_encoder_state()
                    round_uploads[c_name] = comm_tracker.calculate_state_bytes(full_states[c_name])
                elif method_name not in NO_AGGREGATION_METHODS:
                    update = client.get_lora_update(adaptive_rank=adaptive_cfg)
                    if method_name == "fedsa_lora":
                        upload = {k: {"A": v["A"]} for k, v in update.items()}  # B never leaves the client
                    else:
                        upload = update
                    client_updates[c_name] = update
                    round_uploads[c_name] = comm_tracker.calculate_lora_bytes(upload)

        if adaptive_cfg is not None:
            upload_rank_history.append({
                c: float(np.mean(list(client.last_upload_ranks.values()))) for c, client in clients.items()
            })

        # 2. Server aggregation + download
        if method_name == "fedavg_full":
            avg_state = server.aggregate_full_state(full_states, sample_counts)
            for c_name, client in clients.items():
                client.load_encoder_state(avg_state)
                round_downloads[c_name] = comm_tracker.calculate_state_bytes(avg_state)
        elif client_updates:
            downloads = server.aggregate(client_updates, sample_counts, domain_embeddings=domain_embeddings, round_idx=r)
            for c_name, client in clients.items():
                client.apply_download(downloads[c_name])
                round_downloads[c_name] = comm_tracker.calculate_lora_bytes(downloads[c_name])
            if method_name == "flora":
                # FLoRA: global update is merged into the base weights; adapters restart from a shared fresh init.
                fresh = make_fresh_lora_state(lora_template, seed * 1000 + r)
                for client in clients.values():
                    client.set_lora_factors(fresh)

        # 3. Evaluate the post-download model on validation, track communication & log
        round_val_metrics = {c_name: client.evaluate(use_test=False) for c_name, client in clients.items()}
        comm_tracker.log_round(r, round_uploads, round_downloads)
        avg_val_acc = np.mean([round_val_metrics[c]["accuracy"] for c in clients])
        avg_val_bacc = np.mean([round_val_metrics[c]["balanced_accuracy"] for c in clients])
        avg_val_f1 = np.mean([round_val_metrics[c]["macro_f1"] for c in clients])
        log_entry = {
            "avg_val_accuracy": float(avg_val_acc),
            "avg_val_balanced_acc": float(avg_val_bacc),
            "avg_val_macro_f1": float(avg_val_f1),
            "train_loss": {k: float(v) for k, v in round_losses.items()},
            "clients": round_val_metrics,
        }
        if upload_rank_history:
            log_entry["mean_upload_rank"] = upload_rank_history[-1]
        if server.last_weights:
            log_entry["aggregation_weights"] = server.last_weights
        logger.log_round(r, log_entry)
        print(f"  Round {r}/{num_rounds} | Avg Val Acc: {avg_val_acc*100:.2f}% | Bal Acc: {avg_val_bacc*100:.2f}% | Macro-F1: {avg_val_f1:.4f}")

    # Final Test Evaluation (same model state as the last validation evaluation)
    print(f"\n[{exp_name}] Training complete. Evaluating on test splits...")
    test_results = {}
    for c_name, client in clients.items():
        t_metrics = client.evaluate(use_test=True)
        test_results[c_name] = t_metrics
        print(f"  Test [{c_name}]: Acc: {t_metrics['accuracy']*100:.2f}% | Bal Acc: {t_metrics['balanced_accuracy']*100:.2f}% | F1: {t_metrics['macro_f1']:.4f} | AUROC: {t_metrics['macro_auroc']:.4f}")

    trainable_params = sum(
        p.numel() for p in first_client.model.encoder.parameters() if p.requires_grad
    )
    summary = {
        "experiment_name": exp_name,
        "suite": config.get("suite", "single"),
        "setting": setting,
        "dirichlet_alpha": fl_cfg.get("dirichlet_alpha") if setting == "S2" else None,
        "method": method_name,
        "seed": seed,
        "backbone": backbone_name,
        "rank": target_rank,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "clients": list(clients.keys()),
        "method_config": {
            "use_domain_similarity": use_domain_similarity,
            "use_adaptive_rank": use_adaptive_rank,
            "blend_alpha": blend_alpha,
            "personalized_aggregate": personalized,
            "rank_candidates": rank_candidates if use_adaptive_rank else None,
            "hetlora_ranks": hetlora_ranks or None,
        },
        "trainable_encoder_params_per_client": int(trainable_params),
        "test_metrics": test_results,
        "mean_test_accuracy": float(np.mean([test_results[c]["accuracy"] for c in clients])),
        "mean_test_balanced_acc": float(np.mean([test_results[c]["balanced_accuracy"] for c in clients])),
        "mean_test_macro_f1": float(np.mean([test_results[c]["macro_f1"] for c in clients])),
        "mean_test_auroc": float(np.nanmean([test_results[c]["macro_auroc"] for c in clients])),
        "total_comm_bytes": int(comm_tracker.round_history[-1]["cumulative_total_bytes"]) if comm_tracker.round_history else 0,
        "mean_upload_rank_final": upload_rank_history[-1] if upload_rank_history else None,
        "client_sizes": {c: {"train": len(cl.train_loader.dataset), "val": len(cl.val_loader.dataset),
                             "test": len(cl.test_loader.dataset)} for c, cl in clients.items()},
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "amp_bf16": use_amp,
            "class_weighted_loss": class_weighted_loss,
            "gpu_augment": gpu_augment,
        },
    }
    logger.save_summary(summary)
    comm_tracker.to_dataframe().to_csv(results_dir / "communication_cost.csv", index=False)
    print(f"[{exp_name}] Summary saved to {results_dir / 'summary.json'}")

    return summary
