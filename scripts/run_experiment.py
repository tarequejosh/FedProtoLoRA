"""CLI entrypoint to execute a single federated learning experiment."""

import argparse
import sys
import time
from pathlib import Path
import yaml

from fedmedlora.fl.simulator import run_federated_simulation


def main():
    parser = argparse.ArgumentParser(description="Run FedProtoLoRA experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test mode")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cuda' or 'cpu')")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of federated rounds")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.smoke:
        config["smoke"] = True
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device
    if args.rounds is not None:
        config["fl"]["num_rounds"] = args.rounds

    t0 = time.time()
    summary = run_federated_simulation(config)
    elapsed = time.time() - t0

    print(f"\n>> Experiment completed in {elapsed:.2f} seconds.")
    print(f">> Mean Test Accuracy: {summary['mean_test_accuracy']*100:.2f}%")
    print(f">> Mean Test Macro-F1: {summary['mean_test_macro_f1']:.4f}")
    print(f">> Total Communication: {summary['total_comm_bytes'] / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
