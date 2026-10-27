"""Tests for S2 Dirichlet sub-client partitioning and stratified subsampling."""

import pandas as pd

from fedmedlora.data.partition import dirichlet_group_partition, stratified_subsample


def _manifest():
    rows = []
    for g in range(300):
        split = "train" if g < 210 else ("val" if g < 255 else "test")
        for k in range(2):
            rows.append({"group_id": f"p{g}", "label": "ab"[g % 2], "label_idx": g % 2, "split": split})
    return pd.DataFrame(rows)


def test_dirichlet_partition_is_group_disjoint_and_complete():
    df = _manifest()
    parts = dirichlet_group_partition(df, num_clients=3, alpha=0.3, seed=0)
    assert sum(len(p) for p in parts) == len(df)
    seen = [set(p["group_id"]) for p in parts]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    for p in parts:  # every sub-client has data in every split
        assert set(p["split"]) == {"train", "val", "test"}


def test_dirichlet_partition_skews_labels():
    parts = dirichlet_group_partition(_manifest(), num_clients=3, alpha=0.1, seed=1)
    fracs = [(p["label"] == "a").mean() for p in parts]
    assert max(fracs) - min(fracs) > 0.2


def test_stratified_subsample_keeps_label_ratio():
    df = _manifest()
    sub = stratified_subsample(df, 100, seed=0)
    assert abs(len(sub) - 100) <= 2
    assert abs(sub["label_idx"].mean() - df["label_idx"].mean()) < 0.03
    assert stratified_subsample(df, None, seed=0) is df
