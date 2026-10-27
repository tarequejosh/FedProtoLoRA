"""Client construction helpers: Dirichlet label-skew sub-clients (setting S2) and stratified subsampling."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def dirichlet_group_partition(
    df: pd.DataFrame,
    num_clients: int,
    alpha: float,
    seed: int,
    group_col: str = "group_id",
    label_col: str = "label",
    min_groups_per_client: int = 1,
) -> List[pd.DataFrame]:
    """Splits one modality's manifest into `num_clients` sub-clients with Dirichlet label skew.

    Whole patient groups are assigned (no patient is shared between sub-clients), separately within
    every split (train / val / test), using the *same* per-class proportions p_c ~ Dir(alpha) for a
    sub-client across its splits, so each sub-client's val/test reflect its own label distribution.
    """
    rng = np.random.default_rng(seed)
    group_label = df.groupby(group_col)[label_col].agg(lambda s: s.value_counts().index[0])
    classes = sorted(group_label.unique().tolist())
    proportions = {c: rng.dirichlet(alpha * np.ones(num_clients)) for c in classes}

    assignment: Dict[str, int] = {}
    for split in df["split"].unique():
        split_groups = group_label[group_label.index.isin(df.loc[df["split"] == split, group_col].unique())]
        for c in classes:
            ids = np.array(split_groups[split_groups == c].index, copy=True)  # writable (pandas copy-on-write)
            rng.shuffle(ids)
            # Cut points from the cumulative proportions (multinomial-style allocation of whole groups).
            cuts = (np.cumsum(proportions[c]) * len(ids)).round().astype(int)[:-1]
            for k, part in enumerate(np.split(ids, cuts)):
                for gid in part:
                    assignment[gid] = k

        # Guarantee every sub-client at least `min_groups_per_client` groups in this split.
        split_ids = list(split_groups.index)
        counts = np.bincount([assignment[g] for g in split_ids], minlength=num_clients)
        for k in range(num_clients):
            while counts[k] < min_groups_per_client:
                donor = int(np.argmax(counts))
                if counts[donor] <= min_groups_per_client:
                    break
                movable = [g for g in split_ids if assignment[g] == donor]
                gid = movable[rng.integers(len(movable))]
                assignment[gid] = k
                counts[donor] -= 1
                counts[k] += 1

    client_of_row = df[group_col].map(assignment)
    return [df[client_of_row == k].reset_index(drop=True) for k in range(num_clients)]


def stratified_subsample(df: pd.DataFrame, max_samples: Optional[int], seed: int, label_col: str = "label_idx") -> pd.DataFrame:
    """Label-stratified random subsample to at most `max_samples` rows (no-op when smaller or None).

    Used to cap very large clients (e.g. Kermany OCT, ~84k images) so every method runs on the same budget.
    Subsampling inside one split cannot create cross-split leakage.
    """
    if not max_samples or len(df) <= max_samples:
        return df
    frac = max_samples / len(df)
    parts = [g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed) for _, g in df.groupby(label_col)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
