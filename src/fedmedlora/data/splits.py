"""Patient and subject-grouped leakage-safe dataset splitting for FedMedLoRA."""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def extract_patient_id_oct(filename: str) -> str:
    """Extracts patient ID from Retinal OCT filenames like CNV-1016042-1.jpeg -> 1016042.

    The class prefix is dropped on purpose: the same patient can appear under several classes
    (e.g. CNV-1016042-1 and DRUSEN-1016042-3) and must stay within one split.
    """
    match = re.search(r"^[A-Z]+-(\d+)-", filename)
    if match:
        return match.group(1)
    # Fallback to prefix before hyphen
    parts = filename.split("-")
    return parts[1] if len(parts) > 1 else filename


def extract_patient_id_cxr(filename: str) -> str:
    """Extracts patient ID from Kermany chest X-ray filenames.

    PNEUMONIA images: person1946_bacteria_4874.jpeg -> person1946
    NORMAL images:    IM-0115-0001.jpeg -> IM-0115, NORMAL2-IM-1427-0001.jpeg -> NORMAL2-IM-1427
    (NORMAL images of the same patient share the IM-xxxx prefix; treating each file as its own
    group would leak patients across splits.)
    """
    match = re.search(r"(person\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    match = re.search(r"^((?:NORMAL2-)?IM-\d+)-", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return filename


def extract_patient_id_histo(filename: str) -> str:
    """Extracts the BreaKHis patient number, e.g. SOB_B_A-14-22549AB-100-001.png -> 14-22549.

    Grouping is deliberately conservative: the tumour-subtype code and slide letter suffix are dropped
    because (i) patient 14-13412 appears under both the DC and LC subtypes and (ii) slide variants such
    as 22549AB / 22549CD / 22549G belong to the same patient number.
    """
    match = re.search(r"SOB_[MB]_[A-Z]+-(\d+-\d+)[A-Z]*-", filename)
    if match:
        return match.group(1)
    return filename


def make_grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    n_folds: int = 20,
) -> pd.DataFrame:
    """Splits a dataset by group (patient/lesion) with zero group overlap and approximate label stratification.

    The data are partitioned into `n_folds` stratified, group-disjoint folds (StratifiedGroupKFold);
    whole folds are then assigned to test, val and train according to the requested ratios
    (20 folds -> 5% granularity, so 70/15/15 is reproduced exactly at the fold level).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1."
    df = df.copy().reset_index(drop=True)
    labels = df[label_col].astype(str)
    groups = df[group_col].astype(str)

    # Majority label per group: used to judge whether image-level fold stratification is feasible.
    group_label = labels.groupby(groups).agg(lambda s: s.value_counts().index[0])
    min_groups_per_class = int(group_label.value_counts().min())

    if min_groups_per_class >= 2 * n_folds:
        n_test = max(1, int(round(test_ratio * n_folds)))
        n_val = max(1, int(round(val_ratio * n_folds)))
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_of = np.empty(len(df), dtype=int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for fold_idx, (_, held_idx) in enumerate(sgkf.split(df, y=labels, groups=groups)):
                fold_of[held_idx] = fold_idx

        df["split"] = "train"
        df.loc[fold_of < n_test, "split"] = "test"
        df.loc[(fold_of >= n_test) & (fold_of < n_test + n_val), "split"] = "val"
    else:
        # Few groups (e.g. BreaKHis: 70 patients): stratify *patients* by their label and split by patient
        # count, so every split contains patients of every class (image fractions then vary with patient size).
        rng = np.random.default_rng(seed)
        split_of_group = {}
        for _, g_ids in group_label.groupby(group_label):
            ids = np.array(g_ids.index, copy=True)  # writable (pandas copy-on-write)
            rng.shuffle(ids)
            n = len(ids)
            n_test = max(1, int(round(test_ratio * n)))
            n_val = max(1, int(round(val_ratio * n))) if n - n_test > 1 else 0
            for i, gid in enumerate(ids):
                split_of_group[gid] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
        df["split"] = groups.map(split_of_group)

    # Leakage verification assertion
    train_groups = set(df[df["split"] == "train"][group_col])
    val_groups = set(df[df["split"] == "val"][group_col])
    test_groups = set(df[df["split"] == "test"][group_col])

    assert len(train_groups.intersection(val_groups)) == 0, "Leakage detected: train and val share groups!"
    assert len(train_groups.intersection(test_groups)) == 0, "Leakage detected: train and test share groups!"
    assert len(val_groups.intersection(test_groups)) == 0, "Leakage detected: val and test share groups!"

    return df
