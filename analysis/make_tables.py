"""Builds the result tables of the paper (LaTeX) from results/summary.csv and the data manifests, and prints
the statistics quoted in the text (seed-level paired tests, exact sign-flip permutation tests, per-seed differences,
and robustness checks).

Run from the repository root:  python analysis/make_tables.py [--out outputs/tables]
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_report import ci95, holm, paired_tests  # noqa: E402

OUT = ROOT / "outputs" / "tables"
OURS = "fedmedlora"

NAMES = {
    "linear_probe": "Linear probe", "local_only": "Local-only LoRA", "centralized": "Centralized LoRA$^\\ddagger$",
    "fedavg_full": "FedAvg (full FT)", "fedavg_lora": "FedAvg-LoRA (FedIT)", "fedprox_lora": "FedProx-LoRA",
    "ffa_lora": "FFA-LoRA", "fedsa_lora": "FedSA-LoRA", "fedex_lora": "FedEx-LoRA", "flora": "FLoRA",
    "hetlora": "HetLoRA", "flexlora": "FlexLoRA", "flexlora_lr1e-3": "FlexLoRA, $\\eta=10^{-3}$$^\\S$",
    "fedmedlora": "\\textbf{FedProtoLoRA (ours)}",
}
GROUPS = [
    ("Non-federated references", ["linear_probe", "local_only", "centralized"]),
    ("Federated baselines", ["fedavg_full", "fedavg_lora", "fedprox_lora", "ffa_lora", "fedsa_lora", "fedex_lora",
                             "flora", "hetlora", "flexlora", "flexlora_lr1e-3"]),
    ("Proposed", ["fedmedlora"]),
]
CLIENTS = [("derm", "Derm."), ("oct", "OCT"), ("histo", "Histo."), ("cxr", "CXR"), ("brain", "Brain MRI")]
S2_ORDER = [("local_only", "Local-only LoRA"), ("fedsa_lora", "FedSA-LoRA"), ("flexlora", "FlexLoRA"),
            ("flexlora_lr1e-3", "FlexLoRA, $\\eta=10^{-3}$$^\\S$"), ("fedavg_lora", "FedAvg-LoRA (FedIT)"),
            ("oracle_modality", "Oracle modality clustering$^\\P$"), ("fedmedlora", "FedProtoLoRA (global)"),
            ("fedmedlora_personalized", "\\textbf{FedProtoLoRA (personalized)}")]


def pm(values, scale=100.0, d=1, bold=False):
    v = np.asarray(values, dtype=float) * scale
    m, c = np.nanmean(v), ci95(v)
    s = f"{m:.{d}f}" if np.isnan(c) else f"{m:.{d}f} {{\\scriptsize$\\pm${c:.{d}f}}}"
    return f"\\textbf{{{s}}}" if bold else s


def sg(x):
    """Signed number with a typographic minus."""
    return f"{x:+.2f}".replace("-", "$-$")


def sg_pm(mean_d, ci_d):
    """Signed difference with +/- 95% CI over seeds."""
    base = sg(mean_d)
    if np.isnan(ci_d):
        return base
    return f"{base} {{\\scriptsize$\\pm${ci_d:.2f}}}"


def pval(p):
    if np.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def exact_sign_flip_p(diffs):
    """Exact two-sided paired sign-flip (permutation) test over n paired seed differences."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return float("nan")
    obs = abs(np.mean(diffs))
    count = 0
    total = 0
    for signs in product([-1.0, 1.0], repeat=n):
        total += 1
        if abs(np.mean(diffs * np.array(signs))) >= obs - 1e-12:
            count += 1
    return count / total


def seed_paired_stats(df, a, b, key="method", metric="mean_test_balanced_acc"):
    """Seed-level paired statistics between `a` and `b` on client-averaged `metric`.

    Returns:
        dict with keys: mean_d, sd_d, ci_d, p_t, p_perm, wins, n_seeds, diffs
    """
    da = df[df[key] == a].set_index("seed")[metric]
    db = df[df[key] == b].set_index("seed")[metric]
    seeds = sorted(set(da.index) & set(db.index))
    d = 100.0 * (da.loc[seeds].to_numpy(float) - db.loc[seeds].to_numpy(float))
    n = len(d)
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    ci_d = float(ci95(d))
    p_t = float(stats.ttest_rel(da.loc[seeds], db.loc[seeds]).pvalue) if n > 1 and np.any(d != 0) else 1.0
    p_perm = float(exact_sign_flip_p(d)) if n > 1 else float("nan")
    wins = int(np.sum(d > 0))
    return {
        "mean_d": mean_d,
        "sd_d": sd_d,
        "ci_d": ci_d,
        "p_t": p_t,
        "p_perm": p_perm,
        "wins": wins,
        "n_seeds": n,
        "diffs": d,
    }


def write(name, lines):
    """Writes a table; every tabular is wrapped so it never exceeds the text width."""
    out = []
    for line in lines:
        if line.startswith(r"\begin{tabular}"):
            out.append(r"\begin{adjustbox}{max width=\textwidth}")
        out.append(line)
        if line == r"\end{tabular}":
            out.append(r"\end{adjustbox}")
    lines = out
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f">> {OUT / name}")


def mean_ece(df):
    cols = [f"{c}_ece" for c, _ in CLIENTS if f"{c}_ece" in df.columns]
    return df[cols].mean(axis=1)


# ----------------------------------------------------------------------------- tables
def table_datasets():
    meta = {
        "derm": ("Dermoscopy", "HAM10000 \\citep{tschandl2018ham10000}", 7, "lesion"),
        "oct": ("Retinal OCT", "Kermany OCT \\citep{kermany2018identifying}", 4, "patient"),
        "histo": ("Histopathology", "BreaKHis \\citep{spanhol2016breakhis}", 2, "patient"),
        "cxr": ("Chest X-ray", "Kermany CXR \\citep{kermany2018identifying}", 2, "patient"),
        "brain": ("Brain MRI", "Brain Tumor MRI \\citep{nickparvar2021brain}", 4, "near-dup.\\ group"),
    }
    sizes = json.loads((ROOT / "results" / f"{OURS}_dinov2_small_seed42" / "summary.json").read_text())["client_sizes"]
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\caption{Federated benchmark datasets after curation. Images: after exact-duplicate (MD5) removal and "
        "near-duplicate (64-bit pHash, Hamming distance $\\le 4$) handling. Groups: units kept disjoint across splits "
        "(lesion ID for HAM10000; parsed patient ID for OCT, BreaKHis and CXR; perceptual near-duplicate clusters for "
        "Brain MRI, where patient IDs are unavailable). Train/val/test: seed-42 split counts used in S1 training; "
        "training and validation sets exceeding 6{,}000/1{,}500 images (Dermoscopy, OCT) are subsampled by per-class "
        "proportional rounding (producing $\\pm 1$ image rounding totals such as 5{,}999 and 6{,}001/1{,}501), while test "
        "sets are evaluated in full.}",
        "\\label{tab:datasets}",
        "\\begin{tabular}{llrrlr}", "\\toprule",
        "Client & Dataset & Classes & Images & Groups (unit) & Train/val/test \\\\", "\\midrule",
    ]
    for key, (mod, ds, ncls, unit) in meta.items():
        proc = pd.read_csv(ROOT / "data" / "processed" / f"{key}.csv", usecols=["group_id"])
        s = sizes[key]
        lines.append(f"{mod} & {ds} & {ncls} & {len(proc):,} & {proc['group_id'].nunique():,} ({unit}) & "
                     f"{s['train']:,}/{s['val']:,}/{s['test']:,} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    write("datasets.tex", lines)


def table_main(main):
    feds = [m for _, ms in GROUPS[1:] for m in ms]
    best = {col: max(feds, key=lambda m: main[main.method == m][col].mean())
            for col in ["mean_test_balanced_acc", "mean_test_macro_f1", "mean_test_auroc"]}
    best_ece = min(feds, key=lambda m: mean_ece(main[main.method == m]).mean())
    rows = []
    for _, ms in GROUPS:
        for m in ms:
            if m != OURS:
                rows.append((m, seed_paired_stats(main, OURS, m)))
    adj = dict(zip([r[0] for r in rows], holm([r[1]["p_t"] for r in rows])))
    raw = {r[0]: r[1] for r in rows}
    n_seeds = main.groupby("method").seed.nunique().min()
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        f"\\caption{{Setting S1 (one client per modality): test performance averaged over the five clients, mean "
        f"{{\\scriptsize$\\pm$95\\% CI}} over {n_seeds} independent seeds. Bold: best federated method per column. "
        "Comm.: total LoRA/model upload + download over 20 rounds (MiB, fp16; excludes FedProtoLoRA's one-time 3.8\\,KiB "
        "embedding upload). $\\Delta$: paired seed-level difference (FedProtoLoRA minus method) in balanced accuracy "
        f"(percentage points, mean {{\\scriptsize$\\pm$95\\% CI}} over {n_seeds} seeds); $p_t$ ($p_{{\\text{{Holm}}}}$): "
        f"two-sided paired $t$-test over the {n_seeds} seed-level means, unadjusted and Holm-adjusted across all "
        f"{len(rows)} comparisons (exact two-sided sign-flip test has a minimum attainable $p_{{\\text{{perm}}}}=0.0625$ "
        f"at $n={n_seeds}$ seeds). $^\\ddagger$Pooled data (not privacy-preserving). $^\\S$Learning-rate control: "
        "FlexLoRA at FedProtoLoRA's learning rate instead of its validation-selected one (Section~\\ref{sec:lrcontrol}).}",
        "\\label{tab:main}",
        "\\begin{tabular}{lcccccrr}", "\\toprule",
        "Method & Bal.\\ acc.\\ (\\%) & Macro-F1 (\\%) & AUROC (\\%) & ECE $\\downarrow$ & Comm.\\ (MiB) & $\\Delta$ (pp) & $p_t$ ($p_{\\text{Holm}}$) \\\\",
    ]
    for gname, ms in GROUPS:
        lines += ["\\midrule", f"\\multicolumn{{8}}{{l}}{{\\textit{{{gname}}}}} \\\\"]
        for m in ms:
            r = main[main.method == m]
            ece = mean_ece(r)
            cells = [
                pm(r.mean_test_balanced_acc, bold=best["mean_test_balanced_acc"] == m),
                pm(r.mean_test_macro_f1, bold=best["mean_test_macro_f1"] == m),
                pm(r.mean_test_auroc, bold=best["mean_test_auroc"] == m),
                (f"\\textbf{{{ece.mean():.3f}}}" if best_ece == m else f"{ece.mean():.3f}"),
                f"{r.total_comm_mb.mean():,.1f}" if r.total_comm_mb.mean() > 0 else "--",
            ]
            if m == OURS:
                cells += ["--", "--"]
            else:
                st = raw[m]
                cells += [sg_pm(st["mean_d"], st["ci_d"]), f"{pval(st['p_t'])} ({pval(adj[m])})"]
            lines.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    write("main_s1.tex", lines)
    return raw, adj


def table_per_client(main):
    order = [m for _, ms in GROUPS for m in ms]
    best = {c: max([m for m in order if m not in ("centralized",)],
                   key=lambda m: main[main.method == m][f"{c}_balanced_acc"].mean()) for c, _ in CLIENTS}
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\caption{Setting S1: per-client test balanced accuracy (\\%), mean {\\scriptsize$\\pm$95\\% CI} over 5 seeds. "
        "Bold: best per client (excluding the pooled reference). Histopathology has only 70 patients total "
        "($\\approx$10--11 test patients per seed), hence its wide intervals.}",
        "\\label{tab:perclient}",
        "\\begin{tabular}{l" + "c" * len(CLIENTS) + "}", "\\toprule",
        "Method & " + " & ".join(n for _, n in CLIENTS) + " \\\\", "\\midrule",
    ]
    for m in order:
        r = main[main.method == m]
        lines.append(f"{NAMES[m]} & " + " & ".join(pm(r[f'{c}_balanced_acc'], bold=best[c] == m) for c, _ in CLIENTS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    write("per_client.tex", lines)


def table_s2(s2):
    alphas = sorted(s2.dirichlet_alpha.dropna().unique())
    ours = "fedmedlora_personalized"
    tests = []
    for a in alphas:
        sub = s2[s2.dirichlet_alpha == a]
        for key, _ in S2_ORDER:
            if key != ours:
                tests.append(((a, key), seed_paired_stats(sub, ours, key, key="variant")))
    adj = dict(zip([t[0] for t in tests], holm([t[1]["p_t"] for t in tests])))
    raw = {t[0]: t[1] for t in tests}
    n_seeds = s2.groupby("variant").seed.nunique().min()
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{$\\alpha={a:g}$}}" for a in alphas)
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        f"\\caption{{Setting S2 (three simulated label-skewed, group-disjoint sub-clients per modality; 12 clients): "
        f"test balanced accuracy (\\%), mean {{\\scriptsize$\\pm$95\\% CI}} over {n_seeds} independent seeds. Smaller "
        "Dirichlet $\\alpha$ = stronger label skew. $\\Delta$: paired seed-level difference (personalized FedProtoLoRA "
        f"minus method, percentage points, mean {{\\scriptsize$\\pm$95\\% CI}} over {n_seeds} seeds); $p_t$ "
        f"($p_{{\\text{{Holm}}}}$): two-sided paired $t$-test over the {n_seeds} seed-level means, unadjusted and "
        f"Holm-adjusted across all {len(tests)} S2 comparisons (within-regime 7-comparison Holm adjustment gives "
        "$p_{\\text{Holm}}=0.020$ for FlexLoRA at $\\eta=10^{-3}$ under $\\alpha=0.5$; exact sign-flip test gives "
        "$p_{\\text{perm}}=0.0625$ whenever all 5 seeds agree in sign). $^\\S$FlexLoRA at FedProtoLoRA's learning rate. "
        "$^\\P$Uses ground-truth modality labels: FlexLoRA run separately within each modality.}",
        "\\label{tab:s2}",
        "\\begin{tabular}{l" + "crr" * len(alphas) + "}", "\\toprule",
        f"Method & {head} \\\\",
        " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(alphas))),
        " & " + " & ".join(["Bal.\\ acc. & $\\Delta$ (pp) & $p_t$ ($p_{\\text{Holm}}$)"] * len(alphas)) + " \\\\", "\\midrule",
    ]
    best = {a: max([k for k, _ in S2_ORDER], key=lambda k: s2[(s2.variant == k) & (s2.dirichlet_alpha == a)]
                    .mean_test_balanced_acc.mean()) for a in alphas}
    for key, name in S2_ORDER:
        cells = []
        for a in alphas:
            v = s2[(s2.variant == key) & (s2.dirichlet_alpha == a)]
            cells.append(pm(v.mean_test_balanced_acc, bold=best[a] == key))
            if key == ours:
                cells += ["--", "--"]
            else:
                st = raw[(a, key)]
                cells += [sg_pm(st["mean_d"], st["ci_d"]), f"{pval(st['p_t'])} ({pval(adj[(a, key)])})"]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    write("s2.tex", lines)
    return raw, adj


def table_ablation(main, abl, fx):
    """fx: FlexLoRA at the learning rate of the other rows (10^-3), so that the ablation changes one factor at a time."""
    seeds = sorted(abl.seed.unique())
    m3 = main[main.seed.isin(seeds)]
    rows = [
        ("A: product-space SVD aggregation ($\\equiv$ FlexLoRA)", "\\checkmark", "", "", "", fx[fx.seed.isin(seeds)]),
        ("A + B: + domain-similarity weighting", "\\checkmark", "\\checkmark", "", "", abl[abl.experiment_name.str.startswith("ablation_A_plus_B_")]),
        ("A + C: + adaptive upload rank", "\\checkmark", "", "\\checkmark", "", abl[abl.experiment_name.str.startswith("ablation_A_plus_C_")]),
        ("A + B + C (FedProtoLoRA, global)", "\\checkmark", "\\checkmark", "\\checkmark", "", m3[m3.method == OURS]),
        ("A + B + C, personalized", "\\checkmark", "\\checkmark", "\\checkmark", "\\checkmark", abl[abl.experiment_name.str.startswith("ablation_full_personalized_")]),
    ]
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{Component ablation in setting S1 (seeds {', '.join(map(str, seeds))}; mean {{\\scriptsize$\\pm$95\\% CI}}). "
        "A: product-space aggregation; B: similarity weighting; C: spectral-energy upload rank; P: personalized aggregation. "
        "All rows use the same learning rate ($10^{-3}$).}",
        "\\label{tab:ablation}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{ccccccc}", "\\toprule",
        "A & B & C & P & Bal.\\ acc.\\ (\\%) & Macro-F1 (\\%) & Comm.\\ (MiB) \\\\", "\\midrule",
    ]
    for _, a, b, c, p, v in rows:
        lines.append(f"{a} & {b} & {c} & {p} & {pm(v.mean_test_balanced_acc)} & {pm(v.mean_test_macro_f1)} & "
                     f"{v.total_comm_mb.mean():.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("ablation.tex", lines)


def table_tau(main, tau):
    seeds = sorted(tau.seed.unique())
    m3 = main[main.seed.isin(seeds)]
    rows = [("FlexLoRA (full-rank upload)", m3[m3.method == "flexlora"], "16.0")]
    for t, v in [(0.95, m3[m3.method == OURS])] + [(t, tau[tau.tau == t]) for t in sorted(tau.tau.dropna().unique(), reverse=True)]:
        rows.append((f"FedProtoLoRA, $\\tau={t:g}$", v, f"{v.mean_upload_rank.mean():.1f}"))
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{Spectral-energy upload rank (component C) in setting S1: energy threshold $\\tau$ vs.\\ accuracy and "
        f"communication (seeds {', '.join(map(str, seeds))}). Mean upload rank: average over clients and adapted layers in the "
        "final round (maximum 16). Note: FlexLoRA is shown at its validation-selected $\\eta=3\\times10^{-4}$; at matched "
        "$\\eta=10^{-3}$ without component C (row A+B in Table~\\ref{tab:ablation}), balanced accuracy is 87.1\\% at 112.5\\,MiB.}",
        "\\label{tab:tau}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "Configuration & Bal.\\ acc.\\ (\\%) & Upload rank & Comm.\\ (MiB) \\\\", "\\midrule",
    ]
    for name, v, rank in rows:
        lines.append(f"{name} & {pm(v.mean_test_balanced_acc)} & {rank} & {v.total_comm_mb.mean():.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tau.tex", lines)


def table_tuned():
    tuned = json.loads((ROOT / "configs" / "tuned_hparams.json").read_text())
    names = {k: v.replace("\\textbf{", "").replace("}", "").replace("$^\\ddagger$", "") for k, v in NAMES.items()}
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Learning rates selected on the validation split (seed 42, 10 rounds; criterion: mean validation balanced "
        "accuracy over the last three rounds). Grids: LoRA $\\{3\\times10^{-4}, 10^{-3}, 3\\times10^{-3}\\}$; full fine-tuning "
        "$\\{10^{-5}, 3\\times10^{-5}, 10^{-4}\\}$; linear-probe head $\\{10^{-3}, 3\\times10^{-3}, 10^{-2}\\}$.}",
        "\\label{tab:tuned}",
        "\\begin{tabular}{llc}", "\\toprule", "Method & Parameter & Selected value \\\\", "\\midrule",
    ]
    for _, ms in GROUPS:
        for m in ms:
            if m not in tuned:  # control rows (e.g. flexlora_lr1e-3) were not tuned
                continue
            (k, v), = tuned[m].items()
            mant, exp = f"{v:.0e}".split("e")
            val = (f"{mant}\\times" if mant != "1" else "") + f"10^{{{int(exp)}}}"
            lines.append(f"{names[m]} & \\texttt{{{k.replace('_', '\\_')}}} & ${val}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tuned_lr.tex", lines)


def tuning_score(method, lr):
    """Validation criterion used for learning-rate selection: mean val balanced accuracy over the last three rounds."""
    path = ROOT / "results" / f"tune_{method}_lr{lr:g}_dinov2_small_seed42" / "metrics_history.jsonl"
    hist = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return 100 * np.mean([h["avg_val_balanced_acc"] for h in hist[-3:]])


def seed_test(a, b, metric="mean_test_balanced_acc"):
    """(mean diff in pp, ci95 of diff, paired-t p over seeds, exact sign-flip p over seeds, n seeds with a > b, n seeds)."""
    a_s, b_s = a.set_index("seed")[metric], b.set_index("seed")[metric]
    s = sorted(set(a_s.index) & set(b_s.index))
    d = 100.0 * (a_s.loc[s].to_numpy(float) - b_s.loc[s].to_numpy(float))
    return float(d.mean()), float(ci95(d)), float(stats.ttest_rel(a_s.loc[s], b_s.loc[s]).pvalue), float(exact_sign_flip_p(d)), int((d > 0).sum()), len(s)


SCI = {3e-4: "3\\times10^{-4}", 1e-3: "10^{-3}"}


def table_lr_control(main, lrc):
    """2x2 learning-rate control in S1: FlexLoRA and FedProtoLoRA, each at both learning rates."""
    cells = {("flexlora", 3e-4): main[main.method == "flexlora"], ("fedmedlora", 1e-3): main[main.method == OURS],
             ("flexlora", 1e-3): lrc[lrc.method == "flexlora"], ("fedmedlora", 3e-4): lrc[lrc.method == OURS]}
    selected = {"flexlora": 3e-4, "fedmedlora": 1e-3}
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Learning-rate control in setting S1 (five independent seeds). Val.: tuning criterion on the validation "
        "split (seed 42, 10 rounds) that selected the learning rate $\\eta$; $^\\ast$selected value. Lower block: paired "
        "seed-level differences in test balanced accuracy (mean {\\scriptsize$\\pm$95\\% CI}); $p_t$: two-sided paired "
        "$t$-test over the 5 seeds; $p_{\\text{perm}}$: exact two-sided paired sign-flip test over the 5 seeds; wins: "
        "seeds on which the first configuration exceeded the second.}",
        "\\label{tab:lrcontrol}",
        "\\begin{tabular}{llcccc}", "\\toprule",
        "Method & $\\eta$ & Val.\\ (\\%) & Bal.\\ acc.\\ (\\%) & Macro-F1 (\\%) & AUROC (\\%) \\\\", "\\midrule",
    ]
    for m in ["flexlora", OURS]:
        for lr in [3e-4, 1e-3]:
            v = cells[(m, lr)]
            star = "$^\\ast$" if selected[m] == lr else ""
            name = "FlexLoRA" if m == "flexlora" else "FedProtoLoRA"
            lines.append(f"{name} & ${SCI[lr]}${star} & {tuning_score(m, lr):.2f} & {pm(v.mean_test_balanced_acc)} & "
                         f"{pm(v.mean_test_macro_f1)} & {pm(v.mean_test_auroc)} \\\\")
    lines += ["\\midrule", "\\multicolumn{2}{l}{Paired comparison (5 seeds)} & $\\Delta$ (pp) & $p_t$ & $p_{\\text{perm}}$ & wins \\\\",
              "\\midrule"]
    out = {}
    for lr in [3e-4, 1e-3]:
        a, b = cells[(OURS, lr)], cells[("flexlora", lr)]
        d, ci_d, p_t, p_perm, wins, n = seed_test(a, b)
        out[lr] = (d, ci_d, p_t, p_perm)
        lines.append(f"\\multicolumn{{2}}{{l}}{{FedProtoLoRA $-$ FlexLoRA ($\\eta={SCI[lr]}$)}} & {sg_pm(d, ci_d)} & {pval(p_t)} & {pval(p_perm)} & {wins}/{n} \\\\")
    for m, label in [("flexlora", "FlexLoRA"), (OURS, "FedProtoLoRA")]:
        a, b = cells[(m, 1e-3)], cells[(m, 3e-4)]
        d, ci_d, p_t, p_perm, wins, n = seed_test(a, b)
        out[f"{m}_lr"] = (d, ci_d, p_t, p_perm)
        lines.append(f"\\multicolumn{{2}}{{l}}{{{label} ($\\eta=10^{{-3}} - 3\\times10^{{-4}}$)}} & {sg_pm(d, ci_d)} & {pval(p_t)} & {pval(p_perm)} & {wins}/{n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("lr_control.tex", lines)
    return out


def table_beta(s2, beta):
    """Sensitivity to the blend weight beta (personalized FedProtoLoRA, S2, alpha = 0.1). beta = 0 is FlexLoRA."""
    a01 = s2[(s2.dirichlet_alpha == 0.1) & s2.seed.isin(beta.seed.unique())]  # same seeds as the beta sweep
    rows = [(0.0, a01[a01.variant == "flexlora_lr1e-3"], " (= FlexLoRA)")]
    for b in sorted(beta.blend_alpha.unique()):
        rows.append((b, beta[beta.blend_alpha == b], ""))
    rows.append((0.5, a01[a01.variant == "fedmedlora_personalized"], " (default)"))
    rows.sort(key=lambda r: r[0])
    ref = rows[0][1]
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Sensitivity to the blend weight $\\beta$ of Eq.~\\eqref{eq:weights}: personalized FedProtoLoRA in setting "
        "S2 with strong label skew ($\\alpha=0.1$), seeds " + ", ".join(str(s) for s in sorted(beta.seed.unique())) +
        ", $\\eta=10^{-3}$. $\\beta=0$ reduces to FlexLoRA, $\\beta=1$ "
        "uses similarity only. $\\Delta$: paired seed-level difference to $\\beta=0$ in balanced accuracy (pp, mean "
        "{\\scriptsize$\\pm$95\\% CI}); $p_t$: paired $t$-test over the 3 seeds (minimum attainable two-sided sign-flip "
        "$p_{\\text{perm}}=0.250$ at $n=3$). $\\beta=0.5$ was fixed before any experiment and not tuned. On these three "
        "seeds the default gains $+4.67$\\,pp over FlexLoRA vs.\\ $+2.83$\\,pp across all five seeds (Table~\\ref{tab:s2}).}",
        "\\label{tab:beta}",
        "\\begin{tabular}{lccccc}", "\\toprule",
        "$\\beta$ & Bal.\\ acc.\\ (\\%) & Macro-F1 (\\%) & $\\Delta$ (pp) & $p_t$ & wins \\\\", "\\midrule",
    ]
    for b, v, note in rows:
        if b == 0.0:
            tail = "-- & -- & --"
        else:
            d, ci_d, p_t, _, wins, n = seed_test(v, ref)
            tail = f"{sg_pm(d, ci_d)} & {pval(p_t)} & {wins}/{n}"
        lines.append(f"{b:g}{note} & {pm(v.mean_test_balanced_acc)} & {pm(v.mean_test_macro_f1)} & {tail} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("beta.tex", lines)


def table_backbone2(bb):
    """Second backbone (BiomedCLIP ViT-B/16), S1, three seeds, common learning rate."""
    order = [("local_only", "Local-only LoRA"), ("fedavg_lora", "FedAvg-LoRA (FedIT)"), ("fedsa_lora", "FedSA-LoRA"),
             ("flexlora", "FlexLoRA"), (OURS, "\\textbf{FedProtoLoRA (ours)}")]
    best = {c: max([m for m, _ in order], key=lambda m: bb[bb.method == m][c].mean())
            for c in ["mean_test_balanced_acc", "mean_test_macro_f1", "mean_test_auroc"]}
    stats_map = {}
    for m, _ in order:
        if m != OURS:
            stats_map[m] = seed_test(bb[bb.method == OURS], bb[bb.method == m])
    adj_map = dict(zip(list(stats_map.keys()), holm([stats_map[m][2] for m in stats_map])))
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Second backbone: BiomedCLIP ViT-B/16 in setting S1 (three seeds: 42, 101, 2024; common learning rate "
        "$10^{-3}$, LoRA on the fused query-key-value projections). $\\Delta$: paired seed-level difference (FedProtoLoRA "
        "minus method) in balanced accuracy (pp, mean {\\scriptsize$\\pm$95\\% CI}); $p_t$ ($p_{\\text{Holm}}$): two-sided "
        "paired $t$-test over the 3 seeds, unadjusted and Holm-adjusted over the 4 comparisons (minimum attainable "
        "two-sided sign-flip $p_{\\text{perm}}=0.250$ at $n=3$). Bold: best per column.}",
        "\\label{tab:biomedclip}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccccc}", "\\toprule",
        "Method & Bal.\\ acc.\\ (\\%) & Macro-F1 (\\%) & AUROC (\\%) & $\\Delta$ (pp) & $p_t$ ($p_{\\text{Holm}}$) \\\\", "\\midrule",
    ]
    for m, name in order:
        v = bb[bb.method == m]
        cells = [pm(v[c], bold=best[c] == m) for c in ["mean_test_balanced_acc", "mean_test_macro_f1", "mean_test_auroc"]]
        if m == OURS:
            cells += ["--", "--"]
        else:
            d, ci_d, p_t, _, _, _ = stats_map[m]
            cells += [sg_pm(d, ci_d), f"{pval(p_t)} ({pval(adj_map[m])})"]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("biomedclip.tex", lines)


def print_s2_facts(s2):
    """Every S2 number the prose relies on: means, per-seed differences and win counts (for cross-checking the text)."""
    pairs = [("fedmedlora_personalized", "flexlora_lr1e-3"), ("fedmedlora_personalized", "flexlora"),
             ("fedmedlora_personalized", "oracle_modality"), ("fedmedlora_personalized", "fedavg_lora"),
             ("fedmedlora_personalized", "fedmedlora"), ("fedmedlora", "flexlora_lr1e-3"),
             ("oracle_modality", "flexlora_lr1e-3"), ("flexlora_lr1e-3", "fedavg_lora")]
    for a in sorted(s2.dirichlet_alpha.dropna().unique()):
        sub = s2[s2.dirichlet_alpha == a]
        means = sub.groupby("variant").mean_test_balanced_acc.agg(["mean", "count"])
        print(f"\nS2 alpha={a:g} means (%):", {k: (round(100 * m, 2), int(n)) for k, (m, n) in means.iterrows()})
        for x, y in pairs:
            d, ci_d, p_t, p_perm, wins, n = seed_test(sub[sub.variant == x], sub[sub.variant == y])
            per = (100 * (sub[sub.variant == x].set_index("seed").mean_test_balanced_acc
                          - sub[sub.variant == y].set_index("seed").mean_test_balanced_acc)).dropna().round(2)
            print(f"  {x} - {y}: {d:+.2f} +/- {ci_d:.2f} pp, p_t={p_t:.4f}, p_perm={p_perm:.4f}, wins {wins}/{n}, per seed {per.to_dict()}")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="output folder for the .tex tables")
    OUT = parser.parse_args().out.resolve()
    df = pd.read_csv(ROOT / "results" / "summary.csv")
    main_df = df[df.suite == "main"]
    lrc = df[df.suite == "lr_control"]
    fx_matched = lrc[lrc.method == "flexlora"]
    # The S1 table also lists the learning-rate-matched FlexLoRA control as its own row.
    main_ext = pd.concat([main_df, fx_matched.assign(method="flexlora_lr1e-3")])
    s1_raw, s1_adj = table_main(main_ext)
    table_datasets()
    table_per_client(main_ext)
    s2_raw, s2_adj = table_s2(df[df.suite == "s2"])
    table_ablation(main_df, df[df.suite == "ablations"], fx_matched)
    table_tau(main_df, df[df.suite == "tau"])
    table_tuned()
    lr_out = table_lr_control(main_df, lrc)
    table_beta(df[df.suite == "s2"], df[df.suite == "beta"])
    table_backbone2(df[df.suite == "backbone2"])
    print_s2_facts(df[df.suite == "s2"])

    # Key numbers used in the text, printed for cross-checking against the prose.
    print("\nS1 Delta/p_t_raw/p_t_holm:", {k: (round(v["mean_d"], 2), round(v["p_t"], 4), round(s1_adj[k], 4)) for k, v in s1_raw.items()})
    print("LR control (Delta, ci95, p_t, p_perm):", {k: tuple(round(x, 4) for x in v) for k, v in lr_out.items()})
    print("S2 Delta/p_t_raw/p_t_holm:", {f"{a}-{k}": (round(v["mean_d"], 2), round(v["p_t"], 4), round(s2_adj[(a, k)], 4)) for (a, k), v in s2_raw.items()})
    print("S1 seed SD (pp):", {m: round(100 * main_df[main_df.method == m].mean_test_balanced_acc.std(), 2)
                               for m in ["fedmedlora", "flexlora", "fedavg_lora", "fedsa_lora", "fedex_lora", "flora"]})
    print("S1 ECE:", {m: round(mean_ece(main_df[main_df.method == m]).mean(), 3) for m in NAMES if m in main_ext.method.values})


if __name__ == "__main__":
    main()
