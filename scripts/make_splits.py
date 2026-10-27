"""Generate deterministic, patient/lesion-grouped leakage-free train/val/test splits."""

import argparse
from pathlib import Path
import pandas as pd
from fedmedlora.data.splits import make_grouped_stratified_split

def create_splits_for_client(
    client: str,
    processed_dir: Path,
    splits_dir: Path,
    seeds: list = [42],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    proc_file = processed_dir / f"{client}.csv"
    if not proc_file.exists():
        print(f"[{client}] Processed manifest {proc_file} does not exist, skipping.")
        return

    df = pd.read_csv(proc_file)
    print(f"[{client}] Loaded {len(df)} samples across {df['group_id'].nunique()} groups.")

    splits_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        split_df = make_grouped_stratified_split(
            df=df,
            group_col="group_id",
            label_col="label",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        out_path = splits_dir / f"{client}_{seed}.csv"
        split_df.to_csv(out_path, index=False)
        train_count = (split_df['split'] == 'train').sum()
        val_count = (split_df['split'] == 'val').sum()
        test_count = (split_df['split'] == 'test').sum()
        print(f"[{client}] Seed {seed}: train={train_count}, val={val_count}, test={test_count} -> {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Create leakage-free grouped splits")
    parser.add_argument("--clients", nargs="+", default=["brain", "cxr", "derm", "histo", "oct"])
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024, 7, 1234])
    args = parser.parse_args()

    proc_dir = Path(args.processed_dir)
    splits_dir = Path(args.splits_dir)

    for client in args.clients:
        create_splits_for_client(client, proc_dir, splits_dir, seeds=args.seeds)

if __name__ == "__main__":
    main()
