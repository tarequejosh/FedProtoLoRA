"""Federated client implementation for local training, evaluation, and update extraction."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.lora import (
    LoRALinear,
    extract_lora_state,
    load_lora_state,
)
from ..models.backbones import FederatedClientModel
from ..methods.baselines import compute_fedprox_loss
from ..methods.fedmedlora import compress_update_adaptive_rank
from ..eval.metrics import compute_classification_metrics
from ..data.dataset import gpu_batch_transform, IMAGENET_MEAN, IMAGENET_STD


class FederatedClient:
    """Manages local model state, training loop, evaluation, and factor updates for a client."""

    def __init__(
        self,
        client_id: str,
        model: FederatedClientModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        lr: float = 0.0005,
        lr_head: float = 0.001,
        weight_decay: float = 0.01,
        device: str = "cpu",
        summary_loader: Optional[DataLoader] = None,
        amp: bool = False,
        class_weighted_loss: bool = False,
        norm_mean=IMAGENET_MEAN,
        norm_std=IMAGENET_STD,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # Non-augmented view of the training split, used for domain statistics.
        self.summary_loader = summary_loader if summary_loader is not None else train_loader
        self.device = device
        # bf16 autocast (no loss scaling needed); aggregation maths stays in fp32.
        self.amp = amp and str(device).startswith("cuda")
        self.norm_mean, self.norm_std = norm_mean, norm_std
        self.criterion = nn.CrossEntropyLoss(weight=self._class_weights() if class_weighted_loss else None)

        self.lr = lr
        self.lr_head = lr_head
        self.weight_decay = weight_decay

        # State received in the last download: FedProx anchor and reference for delta (adaptive-rank) uploads.
        self.anchor_state: Dict[str, Dict[str, torch.Tensor]] = {}
        self.ref_delta_w: Dict[str, torch.Tensor] = {}
        self.last_upload_ranks: Dict[str, int] = {}

    def prepare_batch(self, images: torch.Tensor, train: bool) -> torch.Tensor:
        """Moves a batch to the device; uint8 batches (GPU-augmentation mode) are augmented/normalised there."""
        images = images.to(self.device, non_blocking=True)
        if images.dtype == torch.uint8:
            images = gpu_batch_transform(images, self.norm_mean, self.norm_std, train=train)
        return images

    def _class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights from the local training labels (normalised to mean 1 over present classes)."""
        num_classes = self.model.head.num_classes
        labels = torch.as_tensor(self.train_loader.dataset.df["label_idx"].to_numpy(), dtype=torch.long)
        counts = torch.bincount(labels, minlength=num_classes).float()
        weights = torch.where(counts > 0, counts.sum() / (num_classes * counts.clamp(min=1)), torch.zeros_like(counts))
        present = counts > 0
        weights[present] = weights[present] / weights[present].mean()
        return weights.to(self.device)

    def _lora_modules(self) -> Dict[str, LoRALinear]:
        return {
            name: module
            for name, module in self.model.encoder.named_modules()
            if isinstance(module, LoRALinear) and module.r > 0
        }

    def compute_domain_summary(self, max_samples: int = 512) -> torch.Tensor:
        """Mean CLS embedding over (up to `max_samples`) non-augmented local training images.

        Must be called before any training: at that point B = 0, so the features are those of the
        frozen foundation model.
        """
        self.model.eval()
        embeddings = []
        n_seen = 0
        with torch.no_grad():
            for images, _, _ in self.summary_loader:
                images = self.prepare_batch(images, train=False)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                    features = self.model.encoder.extract_features(images)
                embeddings.append(features.float().cpu())
                n_seen += images.shape[0]
                if n_seen >= max_samples:
                    break

        return torch.cat(embeddings, dim=0)[:max_samples].mean(dim=0)

    def train_round(
        self,
        local_epochs: int = 1,
        method: str = "fedmedlora",
        mu: float = 0.01,
        freeze_A: bool = False,
        lr_encoder_full: float = 1e-5,
    ) -> float:
        """Runs `local_epochs` local epochs with a single optimizer (state is kept across epochs within a round)."""
        self.model.train()

        lora_params = []
        for module in self._lora_modules().values():
            module.lora_A.requires_grad = not freeze_A
            if not freeze_A:
                lora_params.append(module.lora_A)
            lora_params.append(module.lora_B)

        # Full fine-tuning (FedAvg-Full / centralized-full): every non-LoRA encoder parameter that is trainable.
        lora_ids = {id(p) for m in self._lora_modules().values() for p in (m.lora_A, m.lora_B)}
        full_params = [p for p in self.model.encoder.parameters() if p.requires_grad and id(p) not in lora_ids]

        param_groups = [{"params": list(self.model.head.parameters()), "lr": self.lr_head, "weight_decay": self.weight_decay}]
        if lora_params:
            param_groups.append({"params": lora_params, "lr": self.lr, "weight_decay": self.weight_decay})
        if full_params:
            param_groups.append({"params": full_params, "lr": lr_encoder_full, "weight_decay": self.weight_decay})
        optimizer = torch.optim.AdamW(param_groups)

        total_loss = 0.0
        total_batches = 0
        for _ in range(local_epochs):
            for images, targets, _ in self.train_loader:
                images = self.prepare_batch(images, train=True)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                    loss = self.criterion(self.model(images).float(), targets)

                if method == "fedprox_lora" and self.anchor_state:
                    loss = loss + compute_fedprox_loss(self.model.encoder, self.anchor_state, mu=mu)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        return total_loss / max(1, total_batches)

    def evaluate(self, use_test: bool = False) -> Dict[str, float]:
        """Evaluates client model on validation or test split."""
        self.model.eval()
        loader = self.test_loader if use_test else self.val_loader

        all_logits = []
        all_targets = []

        with torch.no_grad():
            for images, targets, _ in loader:
                images = self.prepare_batch(images, train=False)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                    logits = self.model(images)
                all_logits.append(logits.float().cpu())
                all_targets.append(targets)

        if not all_logits:
            return {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0, "macro_auroc": float("nan")}

        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        return compute_classification_metrics(logits_cat, targets_cat, num_classes=self.model.head.num_classes)

    def get_lora_update(self, adaptive_rank: Optional[Dict] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extracts the LoRA update to upload.

        Default: the full local factors (A, B). With `adaptive_rank` = {"lora_alpha", "energy_tau",
        "rank_candidates"} (FedProtoLoRA component C), each layer instead uploads the round update
        delta = s * B A - (state downloaded at the start of the round), compressed by truncated SVD to
        the smallest candidate rank that keeps `energy_tau` of its spectral energy.
        """
        if adaptive_rank is None:
            return extract_lora_state(self.model.encoder)

        update = {}
        self.last_upload_ranks = {}
        with torch.no_grad():
            for name, module in self._lora_modules().items():
                dense = module.get_delta_weight().float()
                ref = self.ref_delta_w.get(name)
                delta = dense - ref.to(dense.device) if ref is not None else dense
                A_k, B_k, s_k, r_k = compress_update_adaptive_rank(
                    delta,
                    lora_alpha=adaptive_rank["lora_alpha"],
                    energy_tau=adaptive_rank["energy_tau"],
                    rank_candidates=tuple(adaptive_rank["rank_candidates"]),
                )
                update[name] = {"A": A_k.cpu(), "B": B_k.cpu(), "r": r_k, "scaling": s_k, "is_delta": True}
                self.last_upload_ranks[name] = r_k
        return update

    def apply_download(self, state: Dict[str, Dict[str, torch.Tensor]]):
        """Applies a server download to the local encoder.

        Entry kinds: plain factors {A, B}; A-only (FedSA-LoRA); factors plus a dense `residual` folded
        into the frozen base weight (FedEx-LoRA); or `merge` (FLoRA), where scaling * B @ A is folded into
        the base weight and the caller re-initialises the adapters afterwards.
        """
        modules = self._lora_modules()
        to_load = {}
        for name, entry in state.items():
            module = modules.get(name)
            if module is None:
                continue
            if entry.get("merge", False):
                module.add_to_base_weight((entry["B"].float() @ entry["A"].float()) * entry["scaling"])
                continue
            if "residual" in entry:
                module.add_to_base_weight(entry["residual"])
            to_load[name] = entry
        if to_load:
            load_lora_state(self.model.encoder, to_load)
        self._refresh_reference()

    def set_lora_factors(self, lora_state: Dict[str, Dict[str, torch.Tensor]]):
        """Loads LoRA factors into the local encoder (initial broadcast / FLoRA re-initialisation)."""
        load_lora_state(self.model.encoder, lora_state)
        self._refresh_reference()

    def _refresh_reference(self):
        self.anchor_state = extract_lora_state(self.model.encoder)
        with torch.no_grad():
            self.ref_delta_w = {name: m.get_delta_weight().detach().float().clone() for name, m in self._lora_modules().items()}

    # ---- Full-parameter federation (FedAvg-Full baseline) ----
    def get_encoder_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.encoder.state_dict().items()}

    def load_encoder_state(self, state: Dict[str, torch.Tensor]):
        self.model.encoder.load_state_dict(state)
