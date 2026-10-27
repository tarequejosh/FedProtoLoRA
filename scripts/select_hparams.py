"""Select each method's learning rate from the tuning runs (validation split only; test is never read).

Criterion: mean validation balanced accuracy over clients, averaged over the last 3 rounds (less noisy
than the final round alone). Writes configs/tuned_hparams.json, which scripts/run_all.py picks up.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def select(results_dir: str = "results", out: str = "configs/tuned_hparams.json", last_k: int = 3) -> dict:
    candidates = {}
    for summary_file in Path(results_dir).glob("tune_*/summary.json"):
        summary = json.loads(summary_file.read_text())
        if summary.get("suite") != "tune":
            continue
        history_file = summary_file.parent / "metrics_history.jsonl"
        history = [json.loads(line) for line in history_file.read_text().splitlines() if line]
        score = float(np.mean([h["avg_val_balanced_acc"] for h in history[-last_k:]]))
        fl = summary["config"]["fl"]
        method = summary["method"]
        key = next(k for k in ("lr_head", "lr_full", "lr") if summary_file.parent.name.startswith(f"tune_{method}_{k}"))
        candidates.setdefault(method, []).append((score, key, fl[key]))

    if not candidates:
        raise SystemExit("No tuning runs found (expected results/tune_*/summary.json). Run: python scripts/run_all.py --suite tune")

    tuned = {}
    for method, rows in sorted(candidates.items()):
        rows.sort(reverse=True)
        best_score, key, value = rows[0]
        tuned[method] = {key: value}
        grid = ", ".join(f"{k}={v:g}: {s:.4f}" for s, k, v in sorted(rows, key=lambda r: r[2]))
        print(f"{method:14s} -> {key}={value:g}  (val bal-acc {best_score:.4f}; grid: {grid})")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tuned, indent=2) + "\n")
    print(f">> Wrote {out_path}")
    return tuned


def main():
    parser = argparse.ArgumentParser(description="Select per-method learning rates from tuning runs")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out", type=str, default="configs/tuned_hparams.json")
    parser.add_argument("--last_k", type=int, default=3)
    args = parser.parse_args()
    select(args.results_dir, args.out, args.last_k)


if __name__ == "__main__":
    main()
