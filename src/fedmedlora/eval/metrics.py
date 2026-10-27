"""Evaluation metrics for medical classification: accuracy, balanced accuracy, F1, AUROC, ECE, NLL."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    log_loss,
)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE) for multi-class or binary predictions."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_classification_metrics(
    logits: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Computes comprehensive clinical validation metrics.

    Returns:
        Dict with keys: accuracy, balanced_accuracy, macro_f1, macro_auroc, ece, nll.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)

    if num_classes is None:
        num_classes = probs.shape[1]

    acc = float(accuracy_score(targets, preds))
    b_acc = float(balanced_accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds, average="macro", zero_division=0))

    # AUROC
    try:
        if num_classes == 2:
            auroc = float(roc_auc_score(targets, probs[:, 1]))
        else:
            auroc = float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
    except Exception:
        auroc = float("nan")

    # Negative Log Likelihood
    try:
        nll = float(log_loss(targets, probs, labels=list(range(num_classes))))
    except Exception:
        nll = float("nan")

    # Calibration ECE
    ece = compute_ece(probs, targets, n_bins=15)

    return {
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "macro_f1": f1,
        "macro_auroc": auroc,
        "ece": ece,
        "nll": nll,
    }
