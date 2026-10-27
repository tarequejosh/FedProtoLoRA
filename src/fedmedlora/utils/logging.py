"""Logging utilities for experiment runs: JSON metrics, console output, and CSV summaries."""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd


class ExperimentLogger:
    """Logs per-round metrics, client evaluations, and final results to disk."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self.history_file = self.output_dir / "metrics_history.jsonl"
        # Start a fresh history: appending to a previous (e.g. --force) run would mix two runs.
        self.history_file.write_text("")

    def log_round(self, round_idx: int, metrics: Dict[str, Any]):
        entry = {"round": round_idx, **metrics}
        self.metrics_history.append(entry)
        with open(self.history_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_summary(self, summary: Dict[str, Any]):
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics_history)
