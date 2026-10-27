"""Main experiment pipeline: every stage in order, with a quick-look report (reports/) after each stage.

Stages (the control experiments follow in scripts/run_controls.py):
  1. tune          per-method learning rate on validation (seed 42, 10 rounds)  -> configs/tuned_hparams.json
  2. main          13 methods x seeds 42/101/2024                               (Table: main, per-client, significance)
  3. ablations     A+B, A+C, personalized x 3 seeds                             (A-only / A+B+C reuse main runs)
  4. s2_a0.5       three label-skewed institutions per modality, Dirichlet alpha=0.5, 6 variants x 3 seeds
  5. tau           adaptive-rank energy threshold 0.8 / 0.9 x 3 seeds          (0.95 = main)
  6. s2_a0.1       stronger skew, 6 variants x 3 seeds
  7. ranks         FedMedLoRA & FlexLoRA at r = 4 / 8 / 32 x 3 seeds           (16 = main)
  8. main_extra    seeds 7 and 1234 for all 13 methods (5 seeds in total)
Every run is resumable (finished runs are skipped), failures are logged to results/failed_runs.log and skipped.

Usage:  python scripts/run_pipeline.py                 # full pipeline
        python scripts/run_pipeline.py --smoke         # toy-backbone dry run into results_smoke/
        python scripts/run_pipeline.py --stages main   # selected stages only
"""

import argparse
import ctypes
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all  # noqa: E402
from aggregate_results import aggregate_experiment_results  # noqa: E402
from select_hparams import select  # noqa: E402

STAGES = ["tune", "main", "ablations", "s2_a0.5", "tau", "s2_a0.1", "ranks", "main_extra"]


def keep_awake(enable: bool):
    """Prevents Windows from sleeping while this process runs (no system settings are changed)."""
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0))


def log(msg: str, log_file: Path):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def snapshot_code(results_dir: Path):
    """Copies the code and configs used for a stage into results/_code_snapshot/ (provenance of every run)."""
    dest = results_dir / "_code_snapshot" / time.strftime("%Y%m%d_%H%M%S")
    for folder in ("src", "scripts", "configs"):
        shutil.copytree(folder, dest / folder, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    for f in ("requirements-lock.txt", "pyproject.toml"):
        if Path(f).exists():
            shutil.copy2(f, dest / f)
    return dest


def make_report(results_dir: Path, smoke: bool, log_file: Path):
    import subprocess
    df = aggregate_experiment_results(results_dir)
    if df.empty:
        return
    out = "reports_smoke" if smoke else "reports"
    res = subprocess.run([sys.executable, "scripts/make_report.py", "--summary_csv", str(results_dir / "summary.csv"),
                          "--results_dir", str(results_dir), "--output_dir", out], capture_output=True, text=True)
    log(f"report regenerated in {out}/ (exit {res.returncode}, {len(df)} runs)", log_file)


def main():
    parser = argparse.ArgumentParser(description="Run the full FedMedLoRA experiment pipeline")
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    parser.add_argument("--backbone", default="dinov2_small")
    parser.add_argument("--smoke", action="store_true", help="Toy-backbone dry run into results_smoke/")
    args = parser.parse_args()

    if args.smoke:
        run_all.RESULTS_DIR = Path("results_smoke")
        run_all.TUNED_PATH = Path("results_smoke/tuned_hparams.json")
    results_dir = Path(run_all.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "pipeline_progress.log"

    keep_awake(True)
    try:
        snap = snapshot_code(results_dir)
        log(f"pipeline start: stages={args.stages} smoke={args.smoke} code snapshot={snap}", log_file)
        common = dict(backbone=args.backbone, smoke=args.smoke)
        seeds = run_all.MAIN_SEEDS[:1] if args.smoke else run_all.MAIN_SEEDS
        t_start = time.time()

        for stage in args.stages:
            t0 = time.time()
            log(f"stage '{stage}' started", log_file)
            if stage == "tune":
                run_all.suite_tune(seed=run_all.MAIN_SEEDS[0], **common)
                tuned = select(str(results_dir), str(run_all.TUNED_PATH))
                log(f"tuned learning rates: {json.dumps(tuned)}", log_file)
            elif stage == "main":
                if not Path(run_all.TUNED_PATH).exists():
                    log("WARNING: no tuned_hparams.json - main suite uses default learning rates", log_file)
                run_all.suite_main(seeds, **common)
            elif stage == "ablations":
                run_all.suite_ablations(seeds, **common)
            elif stage.startswith("s2_a"):
                run_all.suite_s2(seeds, alphas=[float(stage[len("s2_a"):])], **common)
            elif stage == "tau":
                run_all.suite_tau(seeds, **common)
            elif stage == "ranks":
                run_all.suite_ranks(seeds, **common)
            elif stage == "main_extra":
                if not args.smoke:
                    run_all.suite_main(run_all.EXTRA_SEEDS, **common)
            log(f"stage '{stage}' finished in {(time.time() - t0) / 3600:.2f} h "
                f"(elapsed {(time.time() - t_start) / 3600:.2f} h)", log_file)
            make_report(results_dir, args.smoke, log_file)

        failed = results_dir / "failed_runs.log"
        n_failed = failed.read_text().count("=== ") if failed.exists() else 0
        log(f"pipeline finished in {(time.time() - t_start) / 3600:.2f} h; failed runs: {n_failed}", log_file)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
