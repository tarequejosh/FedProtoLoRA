"""Automated data leakage tests for FedMedLoRA.

Validates that:
1. No patient / subject / lesion group ID appears in more than one split (train, val, test) within a client.
2. No exact MD5 hash appears in more than one split within a client.
3. No perceptual hash collision (Hamming distance <= 4) exists across splits within a client.
4. Generates an synthetic test if no real split manifests exist yet, so test suite passes in CI/dev.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fedmedlora.data.splits import (
    extract_patient_id_cxr,
    extract_patient_id_histo,
    make_grouped_stratified_split,
)

PHASH_THRESHOLD = 4


def _phash_bits(hex_hashes) -> np.ndarray:
    return np.array([[b == "1" for b in bin(int(h, 16))[2:].zfill(64)] for h in hex_hashes], dtype=bool)


def test_patient_id_extraction():
    # Kermany NORMAL images of one patient share the IM-xxxx prefix.
    assert extract_patient_id_cxr("IM-0115-0001.jpeg") == extract_patient_id_cxr("IM-0115-0002.jpeg") == "IM-0115"
    assert extract_patient_id_cxr("NORMAL2-IM-1427-0001.jpeg") == "NORMAL2-IM-1427"
    assert extract_patient_id_cxr("person1946_bacteria_4874.jpeg") == "person1946"
    # BreaKHis: patient 14-13412 appears under two subtypes; slide letters belong to the same patient.
    assert extract_patient_id_histo("SOB_M_DC-14-13412-100-001.png") == extract_patient_id_histo("SOB_M_LC-14-13412-40-001.png")
    assert extract_patient_id_histo("SOB_B_A-14-22549AB-100-001.png") == extract_patient_id_histo("SOB_B_A-14-22549G-40-002.png")


def test_split_is_label_stratified():
    rows = [{"group_id": f"p{i}", "label": "a" if i % 4 else "b"} for i in range(400)]
    df = make_grouped_stratified_split(pd.DataFrame(rows), group_col="group_id", label_col="label", seed=1)
    frac = df.groupby("split")["label"].apply(lambda s: (s == "b").mean())
    assert frac.max() - frac.min() < 0.05
    sizes = df["split"].value_counts(normalize=True)
    assert abs(sizes["train"] - 0.70) < 0.03 and abs(sizes["test"] - 0.15) < 0.03


def test_split_leakage_on_synthetic_data():
    """Unit test verifying that make_grouped_stratified_split strictly guarantees disjoint groups."""
    data = []
    # 20 patients, each with 3 images
    for pid in range(1, 21):
        for img_id in range(3):
            data.append({
                "image_path": f"/dummy/patient_{pid}_img_{img_id}.png",
                "label": "benign" if pid % 2 == 0 else "malignant",
                "group_id": f"patient_{pid}",
                "md5": f"hash_{pid}_{img_id}",
            })
    df = pd.DataFrame(data)

    split_df = make_grouped_stratified_split(df, group_col="group_id", label_col="label", seed=42)

    train_groups = set(split_df[split_df["split"] == "train"]["group_id"])
    val_groups = set(split_df[split_df["split"] == "val"]["group_id"])
    test_groups = set(split_df[split_df["split"] == "test"]["group_id"])

    # Disjoint assertions
    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0


def test_no_leakage_in_existing_manifests():
    """Tests all generated CSV manifests in data/splits for group, MD5, and pHash leakage."""
    splits_dir = Path("data/splits")
    if not splits_dir.exists():
        pytest.skip("No data/splits directory found yet.")

    split_files = list(splits_dir.glob("*.csv"))
    if not split_files:
        pytest.skip("No split CSV manifests found in data/splits.")

    for manifest_path in split_files:
        df = pd.read_csv(manifest_path)
        assert "split" in df.columns, f"Missing 'split' column in {manifest_path}"
        assert "group_id" in df.columns, f"Missing 'group_id' column in {manifest_path}"
        assert "md5" in df.columns, f"Missing 'md5' column in {manifest_path}"

        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]

        # 1. Group ID overlap
        train_groups = set(train_df["group_id"].astype(str))
        val_groups = set(val_df["group_id"].astype(str))
        test_groups = set(test_df["group_id"].astype(str))

        assert not (train_groups & val_groups), f"Train/Val group leakage in {manifest_path.name}: {train_groups & val_groups}"
        assert not (train_groups & test_groups), f"Train/Test group leakage in {manifest_path.name}: {train_groups & test_groups}"
        assert not (val_groups & test_groups), f"Val/Test group leakage in {manifest_path.name}: {val_groups & test_groups}"

        # 2. Exact MD5 hash overlap
        train_md5 = set(train_df["md5"].astype(str))
        val_md5 = set(val_df["md5"].astype(str))
        test_md5 = set(test_df["md5"].astype(str))

        assert not (train_md5 & val_md5), f"Train/Val MD5 leakage in {manifest_path.name}"
        assert not (train_md5 & test_md5), f"Train/Test MD5 leakage in {manifest_path.name}"
        assert not (val_md5 & test_md5), f"Val/Test MD5 leakage in {manifest_path.name}"

        # 3. Perceptual near-duplicates across splits (Hamming distance <= threshold)
        if "phash" in df.columns:
            bits = {s: _phash_bits(d["phash"].astype(str)) for s, d in (("train", train_df), ("val", val_df), ("test", test_df))}
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                if len(bits[a]) == 0 or len(bits[b]) == 0:
                    continue
                for chunk in np.array_split(bits[b], max(1, len(bits[b]) // 256)):
                    dists = (chunk[:, None, :] != bits[a][None, :, :]).sum(-1)
                    assert dists.min() > PHASH_THRESHOLD, f"{a}/{b} pHash near-duplicate leakage in {manifest_path.name}"
