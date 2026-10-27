"""Control experiments, run after scripts/run_pipeline.py.

R1  lr_control   FlexLoRA at FedMedLoRA's learning rate (1e-3), S1, 5 seeds   -> method vs. learning-rate effect
R3  s2           FlexLoRA at 1e-3 in S2 (both alphas, 3 seeds)                 -> same question for S2
R4  s2           Oracle modality clustering (FlexLoRA within each modality)    -> does similarity weighting beat known metadata?
R2  lr_control   FedMedLoRA at FlexLoRA's tuned learning rate (3e-4), S1        -> robustness of the comparison to the LR
R5  beta         personalized FedMedLoRA, S2 alpha=0.1, beta in {0.25,0.75,1}   -> sensitivity to the blend weight
R6  backbone2    BiomedCLIP ViT-B/16: five key methods at a common LR, S1       -> second (medical) foundation model
R7  ranks_lr     FlexLoRA rank sweep (4, 8, 32) at 1e-3, S1, 3 seeds             -> LR-matched rank comparison
R8  s2           S2 seeds 7 and 1234 for all eight S2 variants                   -> S2 at five seeds, like S1
                 (decided on 2026-10-26, before any of these runs; all five seeds are reported)

Same resumable / failure-tolerant / keep-awake behaviour as run_pipeline.py.
Usage: python scripts/run_controls.py [--stages R1 R3 ...]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402
from run_all import build_config, execute_grid_run, load_tuned  # noqa: E402
from run_pipeline import keep_awake, log, make_report, snapshot_code  # noqa: E402

STAGES = ["R1", "R3", "R4", "R2", "R5", "R6", "R7", "R8"]
S2_EXTRA_SEEDS = run_all.EXTRA_SEEDS
S1_SEEDS = run_all.MAIN_SEEDS + run_all.EXTRA_SEEDS
S2_SEEDS = run_all.MAIN_SEEDS
S2_FL = lambda a: {"setting": "S2", "sub_clients_per_modality": 3, "dirichlet_alpha": a}  # noqa: E731


def r1(backbone):
    for seed in S1_SEEDS:
        execute_grid_run(build_config(f"flexlora_lr0.001_{backbone}_seed{seed}", "flexlora", seed, backbone=backbone,
                                      suite="lr_control", fl_overrides={"lr": 1e-3}))


def r2(backbone):
    for seed in S1_SEEDS:
        execute_grid_run(build_config(f"fedmedlora_lr0.0003_{backbone}_seed{seed}", "fedmedlora", seed, backbone=backbone,
                                      suite="lr_control", fl_overrides={"lr": 3e-4}))


def s2_runs(variant, method, kwargs, lr, backbone, seeds=S2_SEEDS):
    for alpha in run_all.S2_ALPHAS:
        for seed in seeds:
            cfg = build_config(f"s2_a{alpha:g}_{variant}_{backbone}_seed{seed}", method, seed, backbone=backbone,
                               clients=run_all.S2_CLIENTS, method_kwargs=kwargs, suite="s2",
                               fl_overrides={**S2_FL(alpha), "lr": lr})
            cfg["variant"] = variant
            execute_grid_run(cfg)


def r5(backbone):
    tuned = load_tuned()
    for beta in [0.25, 0.75, 1.0]:
        for seed in S2_SEEDS:
            cfg = build_config(f"beta_b{beta:g}_s2_a0.1_fedmedlora_personalized_{backbone}_seed{seed}", "fedmedlora", seed,
                               backbone=backbone, clients=run_all.S2_CLIENTS, suite="beta", tuned=tuned,
                               method_kwargs={"personalized_aggregate": True, "blend_alpha": beta},
                               fl_overrides=S2_FL(0.1))
            cfg["variant"] = f"fedmedlora_personalized_beta{beta:g}"
            execute_grid_run(cfg)


def r6():
    # A common learning rate (1e-3, the value selected for most LoRA methods on DINOv2) avoids per-method
    # re-tuning on the second backbone and removes learning-rate differences from the comparison.
    for seed in S2_SEEDS:
        for method in ["fedmedlora", "flexlora", "fedavg_lora", "fedsa_lora", "local_only"]:
            execute_grid_run(build_config(f"{method}_biomedclip_seed{seed}", method, seed, backbone="biomedclip",
                                          suite="backbone2", fl_overrides={"lr": 1e-3}))


def r7(backbone):
    for seed in S2_SEEDS:
        for r in run_all.SWEEP_RANKS:
            cfg = build_config(f"rank_sweep_flexlora_lr0.001_r{r}_{backbone}_seed{seed}", "flexlora", seed, rank=r,
                               backbone=backbone, suite="ranks_lr", fl_overrides={"lr": 1e-3})
            cfg["variant"] = "flexlora_lr1e-3"
            execute_grid_run(cfg)


def main():
    parser = argparse.ArgumentParser(description="Control experiments")
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    parser.add_argument("--smoke", action="store_true", help="Toy-backbone dry run into results_smoke/")
    args = parser.parse_args()
    if args.smoke:
        global build_config
        run_all.RESULTS_DIR = Path("results_smoke")
        _orig = run_all.build_config
        build_config = lambda *a, **k: _orig(*a, smoke=True, **k)  # noqa: E731
    backbone = "dinov2_small"
    results_dir = Path(run_all.RESULTS_DIR)
    log_file = results_dir / "controls_progress.log"

    keep_awake(True)
    try:
        log(f"controls start: stages={args.stages}; code snapshot={snapshot_code(results_dir)}", log_file)
        t_start = time.time()
        for stage in args.stages:
            t0 = time.time()
            log(f"stage {stage} started", log_file)
            if stage == "R1":
                r1(backbone)
            elif stage == "R2":
                r2(backbone)
            elif stage == "R3":
                s2_runs("flexlora_lr1e-3", "flexlora", {}, 1e-3, backbone)
            elif stage == "R4":
                s2_runs("oracle_modality", "flexlora", {"oracle_modality_clusters": True}, 1e-3, backbone)
            elif stage == "R5":
                r5(backbone)
            elif stage == "R6":
                r6()
            elif stage == "R7":
                r7(backbone)
            elif stage == "R8":
                s2_runs("flexlora_lr1e-3", "flexlora", {}, 1e-3, backbone, seeds=S2_EXTRA_SEEDS)
                s2_runs("oracle_modality", "flexlora", {"oracle_modality_clusters": True}, 1e-3, backbone,
                        seeds=S2_EXTRA_SEEDS)
                run_all.suite_s2(S2_EXTRA_SEEDS, backbone=backbone, smoke=args.smoke)
            log(f"stage {stage} finished in {(time.time() - t0) / 3600:.2f} h (elapsed {(time.time() - t_start) / 3600:.2f} h)", log_file)
            make_report(results_dir, args.smoke, log_file)
        log(f"controls finished in {(time.time() - t_start) / 3600:.2f} h", log_file)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
