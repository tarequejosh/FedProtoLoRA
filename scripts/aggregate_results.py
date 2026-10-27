"""Aggregate all experiment runs from results/ into a consolidated summary CSV."""

import json
from pathlib import Path
import pandas as pd


def aggregate_experiment_results(results_dir: Path = Path("results")) -> pd.DataFrame:
    records = []
    if not results_dir.exists():
        print(f"Results directory '{results_dir}' does not exist.")
        return pd.DataFrame()

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        summary_file = exp_dir / "summary.json"
        if not summary_file.exists():
            continue

        try:
            with open(summary_file, "r") as f:
                data = json.load(f)

            config = data.get("config", {})
            rec = {
                "experiment_name": data.get("experiment_name", exp_dir.name),
                "suite": data.get("suite", "unknown"),
                "method": data.get("method", "unknown"),
                # S2 runs distinguish e.g. fedmedlora vs fedmedlora_personalized by a variant label.
                "variant": config.get("variant", data.get("method", "unknown")),
                "setting": data.get("setting", "S1"),
                "dirichlet_alpha": data.get("dirichlet_alpha"),
                "backbone": data.get("backbone", "unknown"),
                "lr": config.get("fl", {}).get("lr"),
                "blend_alpha": data.get("method_config", {}).get("blend_alpha"),
                "oracle_modality": bool(config.get("method", {}).get("oracle_modality_clusters", False)),
                "rank": data.get("rank", -1),
                "seed": data.get("seed", -1),
                "num_rounds": data.get("num_rounds", 0),
                "mean_test_accuracy": data.get("mean_test_accuracy", float("nan")),
                "mean_test_balanced_acc": data.get("mean_test_balanced_acc", float("nan")),
                "mean_test_macro_f1": data.get("mean_test_macro_f1", float("nan")),
                "mean_test_auroc": data.get("mean_test_auroc", float("nan")),
                "tau": config.get("method", {}).get("adaptive_energy_tau", 0.95)
                if data.get("method_config", {}).get("use_adaptive_rank") else None,
                "mean_upload_rank": (sum(data["mean_upload_rank_final"].values()) / len(data["mean_upload_rank_final"]))
                if data.get("mean_upload_rank_final") else None,
                "total_comm_bytes": data.get("total_comm_bytes", 0),
                "total_comm_mb": data.get("total_comm_bytes", 0) / (1024 ** 2),
            }

            # Per-client breakdown
            for c_name, c_metrics in data.get("test_metrics", {}).items():
                rec[f"{c_name}_accuracy"] = c_metrics.get("accuracy", float("nan"))
                rec[f"{c_name}_balanced_acc"] = c_metrics.get("balanced_accuracy", float("nan"))
                rec[f"{c_name}_macro_f1"] = c_metrics.get("macro_f1", float("nan"))
                rec[f"{c_name}_auroc"] = c_metrics.get("macro_auroc", float("nan"))
                rec[f"{c_name}_ece"] = c_metrics.get("ece", float("nan"))

            records.append(rec)
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=["suite", "method", "seed"]).reset_index(drop=True)
        out_csv = results_dir / "summary.csv"
        df.to_csv(out_csv, index=False)
        print(f">> Aggregated {len(df)} experiment runs into {out_csv}")
    else:
        print(">> No completed experiment summaries found to aggregate.")

    return df


if __name__ == "__main__":
    aggregate_experiment_results()
