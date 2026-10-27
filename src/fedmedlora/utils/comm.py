"""Exact communication cost accounting for federated PEFT."""

from typing import Dict, List, Optional
import pandas as pd


class CommunicationTracker:
    """Tracks exact uploaded and downloaded bytes across federated rounds.

    Accounts for client-specific rank r_k, fp16/fp32 precision, and domain statistics.
    """

    def __init__(self, precision_bytes: int = 2):  # default 2 for fp16
        self.precision_bytes = precision_bytes
        self.round_history: List[Dict] = []
        self.cumulative_uploaded: Dict[str, int] = {}
        self.cumulative_downloaded: Dict[str, int] = {}

    def calculate_lora_bytes(
        self,
        lora_state: Dict[str, Dict[str, any]],
        precision_bytes: Optional[int] = None,
    ) -> int:
        """Calculates total bytes for a given LoRA state dictionary."""
        p_bytes = precision_bytes if precision_bytes is not None else self.precision_bytes
        total_params = 0
        for name, entry in lora_state.items():
            # A-only (FedSA-LoRA) entries have no B; FedEx-LoRA additionally ships a dense residual.
            for key in ("A", "B", "residual"):
                if entry.get(key) is not None:
                    total_params += entry[key].numel()
        return total_params * p_bytes

    def calculate_state_bytes(self, state: Dict[str, any], precision_bytes: Optional[int] = None) -> int:
        """Bytes of a full (floating-point) model state dict, e.g. for the FedAvg-Full baseline."""
        p_bytes = precision_bytes if precision_bytes is not None else self.precision_bytes
        return sum(t.numel() for t in state.values() if hasattr(t, "is_floating_point") and t.is_floating_point()) * p_bytes

    def log_round(
        self,
        server_round: int,
        client_uploads: Dict[str, int],    # {client_id: bytes_uploaded}
        client_downloads: Dict[str, int],  # {client_id: bytes_downloaded}
    ):
        """Logs communication for a completed federated round."""
        round_up_total = sum(client_uploads.values())
        round_down_total = sum(client_downloads.values())

        for c, b in client_uploads.items():
            self.cumulative_uploaded[c] = self.cumulative_uploaded.get(c, 0) + b
        for c, b in client_downloads.items():
            self.cumulative_downloaded[c] = self.cumulative_downloaded.get(c, 0) + b

        record = {
            "round": server_round,
            "round_uploaded_bytes": round_up_total,
            "round_downloaded_bytes": round_down_total,
            "round_total_bytes": round_up_total + round_down_total,
            "cumulative_uploaded_bytes": sum(self.cumulative_uploaded.values()),
            "cumulative_downloaded_bytes": sum(self.cumulative_downloaded.values()),
            "cumulative_total_bytes": sum(self.cumulative_uploaded.values()) + sum(self.cumulative_downloaded.values()),
        }
        for c in client_uploads:
            record[f"client_{c}_up_bytes"] = client_uploads[c]
            record[f"client_{c}_down_bytes"] = client_downloads.get(c, 0)
            record[f"client_{c}_cum_bytes"] = self.cumulative_uploaded[c] + self.cumulative_downloaded.get(c, 0)

        self.round_history.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.round_history)
