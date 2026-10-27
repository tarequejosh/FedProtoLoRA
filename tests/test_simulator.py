"""End-to-end smoke tests: every method runs through the simulator on a tiny synthetic federation."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from fedmedlora.fl.simulator import KNOWN_METHODS, run_federated_simulation


@pytest.fixture(scope="module")
def synthetic_splits(tmp_path_factory):
    root = tmp_path_factory.mktemp("fed")
    rng = np.random.default_rng(0)
    for client, n_classes in (("brain", 4), ("cxr", 2)):
        rows = []
        for i in range(24):
            label = i % n_classes
            path = root / f"{client}_{i}.png"
            arr = (rng.random((32, 32, 3)) * 60 + label * 40).astype(np.uint8)
            Image.fromarray(arr).save(path)
            split = "train" if i < 12 else ("val" if i < 18 else "test")
            rows.append({"image_path": str(path), "label": str(label), "label_idx": label,
                         "group_id": f"g{i}", "md5": f"m{i}", "split": split})
        pd.DataFrame(rows).to_csv(root / f"{client}_42.csv", index=False)
    return root


@pytest.mark.parametrize("method", KNOWN_METHODS)
def test_every_method_runs(method, synthetic_splits):
    cfg = {
        "experiment_name": f"smoke_{method}",
        "seed": 42,
        "device": "cpu",
        "smoke": True,
        "splits_dir": str(synthetic_splits),
        "results_dir": str(synthetic_splits / "results"),
        "method": {"name": method, "blend_alpha": 0.5, "personalized_aggregate": method == "fedmedlora"},
        "model": {"r": 4, "lora_alpha": 8.0},
        "fl": {"num_rounds": 2, "local_epochs": 1, "batch_size": 4, "clients": ["brain", "cxr"],
               "smoke_samples_per_client": 12},
    }
    summary = run_federated_simulation(cfg)
    assert 0.0 <= summary["mean_test_accuracy"] <= 1.0
    if method in ("local_only", "linear_probe", "centralized"):
        assert summary["total_comm_bytes"] == 0
    else:
        assert summary["total_comm_bytes"] > 0



def test_s2_dirichlet_subclients_run(synthetic_splits):
    cfg = {
        "experiment_name": "smoke_s2", "seed": 42, "device": "cpu", "smoke": True,
        "splits_dir": str(synthetic_splits), "results_dir": str(synthetic_splits / "results"),
        "method": {"name": "fedmedlora", "personalized_aggregate": True},
        "model": {"r": 4, "lora_alpha": 8.0},
        "fl": {"setting": "S2", "sub_clients_per_modality": 2, "dirichlet_alpha": 0.5, "num_rounds": 1,
               "local_epochs": 1, "batch_size": 4, "clients": ["brain", "cxr"]},
    }
    summary = run_federated_simulation(cfg)
    assert set(summary["clients"]) == {"brain_0", "brain_1", "cxr_0", "cxr_1"}
