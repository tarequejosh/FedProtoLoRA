"""Resumable runner for the FedProtoLoRA experiment grid.

Compute budget (identical for every method): 20 rounds x 1 local epoch, label-stratified caps of 6,000 train /
1,500 val images per client, bf16, batch 32. Suites are ordered seed-major so that a partially finished grid
still yields complete tables with fewer seeds. Redundant runs are not repeated: ablation "A only" is FlexLoRA and
"A+B+C" is FedProtoLoRA from the main suite; the rank sweep and tau sweep reuse r=16 / tau=0.95 from main.

Recommended: python scripts/run_pipeline.py, then python scripts/run_controls.py
Single suites: python scripts/run_all.py --suite {tune,main,ablations,s2,tau,ranks}
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from fedmedlora.fl.simulator import run_federated_simulation


ALL_METHODS = [
    "fedmedlora",
    "flexlora",
    "fedavg_lora",
    "fedprox_lora",
    "ffa_lora",
    "fedsa_lora",
    "fedex_lora",
    "flora",
    "hetlora",
    "local_only",
    "linear_probe",
    "fedavg_full",
    "centralized",
]

ALL_CLIENTS = ["derm", "oct", "histo", "cxr", "brain"]
MAIN_SEEDS = [42, 101, 2024]
EXTRA_SEEDS = [7, 1234]

# Only the ablation variants that are NOT already main-suite runs:
#   ablation_A_only   == flexlora     (main suite)
#   ablation_full_A_B_C == fedmedlora (main suite)
ABLATION_VARIANTS = {
    "ablation_A_plus_B": {"use_domain_similarity": True, "use_adaptive_rank": False, "blend_alpha": 0.5},
    "ablation_A_plus_C": {"use_domain_similarity": False, "use_adaptive_rank": True},
    "ablation_full_personalized": {"use_domain_similarity": True, "use_adaptive_rank": True, "blend_alpha": 0.5,
                                   "personalized_aggregate": True},
}

# Setting S2: 3 label-skewed, patient-disjoint sub-clients per modality. Histopathology is excluded: its 70
# patients (16 benign) cannot be split into 3 sub-clients with usable val/test sets.
S2_CLIENTS = ["derm", "oct", "cxr", "brain"]
S2_ALPHAS = [0.5, 0.1]
S2_METHODS = {
    "fedmedlora": {},
    "fedmedlora_personalized": {"personalized_aggregate": True},
    "flexlora": {},
    "fedavg_lora": {},
    "fedsa_lora": {},
    "local_only": {},
}

TAU_VALUES = [0.8, 0.9]          # 0.95 = main-suite FedProtoLoRA
SWEEP_RANKS = [4, 8, 32]         # 16 = main suite

# Learning-rate grids searched on the validation split (seed 42, 10 rounds).
TUNE_GRID = {
    "linear_probe": ("lr_head", [1e-3, 3e-3, 1e-2]),
    "fedavg_full": ("lr_full", [1e-5, 3e-5, 1e-4]),
}
DEFAULT_TUNE = ("lr", [3e-4, 1e-3, 3e-3])
TUNE_ROUNDS = 10

TUNED_PATH = Path("configs/tuned_hparams.json")
RESULTS_DIR = Path("results")
FAILED_LOG = RESULTS_DIR / "failed_runs.log"


def load_tuned() -> dict:
    path = Path(TUNED_PATH)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def build_config(
    exp_name: str,
    method_name: str,
    seed: int,
    backbone: str = "dinov2_small",
    rank: int = 16,
    num_rounds: int = 20,
    clients: list = None,
    method_kwargs: dict = None,
    smoke: bool = False,
    suite: str = "main",
    fl_overrides: dict = None,
    tuned: dict = None,
) -> dict:
    if clients is None:
        clients = list(ALL_CLIENTS)

    m_cfg = {"name": method_name}
    if method_kwargs:
        m_cfg.update(method_kwargs)

    fl = {
        "setting": "S1",
        "num_rounds": num_rounds,
        "local_epochs": 1,
        "batch_size": 32 if not smoke else 4,
        "lr": 0.001,
        "lr_head": 0.001,
        "lr_full": 3e-5,
        "weight_decay": 0.01,
        "clients": clients,
        "smoke_samples_per_client": 12,
        # Equal compute budget for every method: label-stratified caps (OCT 50k, derm 6.9k train images).
        "max_train_samples_per_client": 6000,
        "max_val_samples_per_client": 1500,
        "amp": True,
        "class_weighted_loss": True,
        # GPU augmentation + shared pre-decoded tensor cache; no worker processes (slow / memory-hungry on Windows).
        "gpu_augment": True,
        "cache_in_memory": True,
        "num_workers": 0,
        "eval_num_workers": 0,
    }
    # Tuned learning rates (from select_hparams.py) are keyed by the base method name.
    if tuned and method_name in tuned:
        fl.update(tuned[method_name])
    if fl_overrides:
        fl.update(fl_overrides)

    return {
        "experiment_name": exp_name,
        "results_dir": str(RESULTS_DIR),  # module-level so run_pipeline.py can redirect a dry run
        "suite": suite,
        "seed": seed,
        "device": "auto",
        "smoke": smoke,
        "method": m_cfg,
        "model": {
            "backbone": backbone,
            "r": rank,
            "lora_alpha": 16.0,
            "lora_dropout": 0.05,
        },
        "fl": fl,
    }


def execute_grid_run(config: dict, force: bool = False) -> bool:
    """Runs one experiment unless it already finished. A failure is logged and does not stop the grid."""
    exp_name = config["experiment_name"]
    results_dir = Path(config.get("results_dir", RESULTS_DIR))
    summary_path = results_dir / exp_name / "summary.json"

    if summary_path.exists() and not force:
        print(f">> [SKIPPING] Result already exists for '{exp_name}'", flush=True)
        return True

    print("\n=======================================================", flush=True)
    print(f">> Starting Experiment: {exp_name}", flush=True)
    print("=======================================================", flush=True)
    t0 = time.time()
    try:
        run_federated_simulation(config)
        print(f">> [DONE] {exp_name} in {(time.time() - t0) / 60:.1f} min", flush=True)
        return True
    except Exception:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / FAILED_LOG.name, "a") as f:
            f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {exp_name}\n{traceback.format_exc()}\n")
        print(f">> [FAILED] {exp_name} (see {results_dir / FAILED_LOG.name})", flush=True)
        return False


# ----------------------------------------------------------------------------- suites
def suite_tune(backbone="dinov2_small", seed=MAIN_SEEDS[0], methods=None, smoke=False, force=False):
    for method in methods or ALL_METHODS:
        key, grid = TUNE_GRID.get(method, DEFAULT_TUNE)
        for value in grid:
            cfg = build_config(f"tune_{method}_{key}{value:g}_{backbone}_seed{seed}", method, seed,
                               backbone=backbone, num_rounds=TUNE_ROUNDS, suite="tune",
                               fl_overrides={key: value}, smoke=smoke)
            execute_grid_run(cfg, force)


def suite_main(seeds, backbone="dinov2_small", methods=None, smoke=False, force=False):
    tuned = load_tuned()
    for seed in seeds:  # seed-major: every method gets seed 42 before any method gets seed 101
        for method in methods or ALL_METHODS:
            cfg = build_config(f"{method}_{backbone}_seed{seed}", method, seed, backbone=backbone,
                               smoke=smoke, tuned=tuned)
            execute_grid_run(cfg, force)


def suite_ablations(seeds, backbone="dinov2_small", smoke=False, force=False):
    tuned = load_tuned()
    for seed in seeds:
        for var_name, var_kwargs in ABLATION_VARIANTS.items():
            cfg = build_config(f"{var_name}_{backbone}_seed{seed}", "fedmedlora", seed, backbone=backbone,
                               method_kwargs=var_kwargs, suite="ablations", smoke=smoke, tuned=tuned)
            execute_grid_run(cfg, force)


def suite_s2(seeds, alphas=S2_ALPHAS, backbone="dinov2_small", smoke=False, force=False):
    tuned = load_tuned()
    for alpha in alphas:
        for seed in seeds:
            for variant, kwargs in S2_METHODS.items():
                base_method = variant.split("_personalized")[0]
                cfg = build_config(f"s2_a{alpha:g}_{variant}_{backbone}_seed{seed}", base_method, seed,
                                   backbone=backbone, clients=S2_CLIENTS, method_kwargs=kwargs, suite="s2",
                                   fl_overrides={"setting": "S2", "sub_clients_per_modality": 3,
                                                 "dirichlet_alpha": alpha},
                                   smoke=smoke, tuned=tuned)
                cfg["variant"] = variant
                execute_grid_run(cfg, force)


def suite_tau(seeds, backbone="dinov2_small", smoke=False, force=False):
    tuned = load_tuned()
    for seed in seeds:
        for tau in TAU_VALUES:
            cfg = build_config(f"tau_sweep_t{tau:g}_{backbone}_seed{seed}", "fedmedlora", seed, backbone=backbone,
                               method_kwargs={"adaptive_energy_tau": tau}, suite="tau", smoke=smoke, tuned=tuned)
            execute_grid_run(cfg, force)


def suite_ranks(seeds, backbone="dinov2_small", smoke=False, force=False):
    tuned = load_tuned()
    for seed in seeds:
        for method in ["fedmedlora", "flexlora"]:
            for r in SWEEP_RANKS:
                cfg = build_config(f"rank_sweep_{method}_r{r}_{backbone}_seed{seed}", method, seed, rank=r,
                                   backbone=backbone, suite="ranks", smoke=smoke, tuned=tuned)
                execute_grid_run(cfg, force)


def main():
    parser = argparse.ArgumentParser(description="Run FedProtoLoRA experimental suites")
    parser.add_argument("--suite", type=str, default="main", choices=["tune", "main", "ablations", "s2", "tau", "ranks"])
    parser.add_argument("--seeds", nargs="+", type=int, default=MAIN_SEEDS)
    parser.add_argument("--backbone", type=str, default="dinov2_small")
    parser.add_argument("--methods", nargs="+", default=None, help="Restrict the main/tune suites to these methods")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    args = parser.parse_args()

    common = dict(backbone=args.backbone, smoke=args.smoke, force=args.force)
    if args.suite == "tune":
        suite_tune(seed=args.seeds[0], methods=args.methods, **common)
    elif args.suite == "main":
        suite_main(args.seeds, methods=args.methods, **common)
    elif args.suite == "ablations":
        suite_ablations(args.seeds, **common)
    elif args.suite == "s2":
        suite_s2(args.seeds, **common)
    elif args.suite == "tau":
        suite_tau(args.seeds, **common)
    elif args.suite == "ranks":
        suite_ranks(args.seeds, **common)


if __name__ == "__main__":
    main()
