"""Generate publication LaTeX tables, statistics, and figures from real experimental runs.

Statistics: mean ± 95% CI (t-distribution over seeds). Significance against the reference method
(FlexLoRA by default — FedProtoLoRA's direct parent) uses (i) a paired t-test over seeds on the
client-averaged metric and (ii) a Wilcoxon signed-rank test over (seed, client) pairs. With 3 seeds the
seed-level test has little power; report both and prefer 5 seeds for the final paper.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import numpy as np
import pandas as pd
from scipy import stats


METHODS_DISPLAY_ORDER = [
    ("linear_probe", "Linear Probe"),
    ("local_only", "Local-Only LoRA"),
    ("centralized", "Centralized LoRA (pooled, reference)"),
    ("fedavg_full", "FedAvg Full FT"),
    ("fedavg_lora", "FedAvg-LoRA"),
    ("fedprox_lora", "FedProx-LoRA"),
    ("ffa_lora", "FFA-LoRA"),
    ("fedsa_lora", "FedSA-LoRA"),
    ("fedex_lora", "FedEx-LoRA"),
    ("flora", "FLoRA"),
    ("hetlora", "HetLoRA"),
    ("flexlora", "FlexLoRA"),
    ("fedmedlora", "\\textbf{FedProtoLoRA (Ours)}"),
]
def plain(name: str) -> str:
    """Strip a \\textbf{...} wrapper (and nothing else, so math such as $10^{-3}$ survives)."""
    return re.sub(r"^\\textbf\{(.*)\}$", r"\1", name)


PLAIN_NAMES = {k: plain(v) for k, v in METHODS_DISPLAY_ORDER}

ABLATIONS_DISPLAY_ORDER = [
    ("ablation_A_only", "A: Product SVD, sample weights, fixed rank ($\\equiv$ FlexLoRA)"),
    ("ablation_A_plus_B", "A + B: Product SVD + Domain Similarity"),
    ("ablation_A_plus_C", "A + C: Product SVD + Adaptive Upload Rank"),
    ("ablation_full_A_B_C", "\\textbf{A + B + C: Full Proposed Method}"),
    ("ablation_full_personalized", "A + B + C, personalized aggregation"),
]

S2_DISPLAY_ORDER = [
    ("local_only", "Local-Only LoRA"),
    ("fedavg_lora", "FedAvg-LoRA"),
    ("fedsa_lora", "FedSA-LoRA"),
    ("flexlora", "FlexLoRA"),
    ("flexlora_lr1e-3", "FlexLoRA, lr $10^{-3}$"),
    ("oracle_modality", "Oracle modality clustering"),
    ("fedmedlora", "FedProtoLoRA (global)"),
    ("fedmedlora_personalized", "\\textbf{FedProtoLoRA (personalized)}"),
]

# Validated categorical slots (dataviz reference palette, light mode); everything else is neutral gray.
HIGHLIGHT = {
    "fedmedlora": ("#2a78d6", "-", "o"),
    "flexlora": ("#eb6834", "--", "s"),
    "fedavg_lora": ("#1baf7a", "-.", "^"),
    "local_only": ("#6250d6", ":", "D"),
}
NEUTRAL = "#a3a29c"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


# ----------------------------------------------------------------------------- statistics
def ci95(values) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def fmt(values, scale=100.0, digits=2) -> str:
    v = np.asarray(values, dtype=float) * scale
    mean = np.nanmean(v) if len(v) else float("nan")
    ci = ci95(v)
    if np.isnan(mean):
        return "\\textsc{NOT RUN}"
    return f"{mean:.{digits}f}" if np.isnan(ci) else f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"


def client_columns(df: pd.DataFrame, metric: str = "balanced_acc"):
    # Only clients present in these runs (summary.csv also holds S2 sub-client columns, NaN for S1 runs).
    return [c for c in df.columns if c.endswith(f"_{metric}") and not c.startswith("mean_") and df[c].notna().any()]


def paired_tests(df: pd.DataFrame, a: str, b: str, key: str = "method", metric: str = "balanced_acc"):
    """(mean diff a-b in %, p paired-t over seeds, p Wilcoxon over (seed, client) pairs, n pairs)."""
    da, db = df[df[key] == a].set_index("seed"), df[df[key] == b].set_index("seed")
    seeds = sorted(set(da.index) & set(db.index))
    if len(seeds) < 2:
        return None
    mean_col = "mean_test_balanced_acc" if metric == "balanced_acc" else f"mean_test_{metric}"
    xa, xb = da.loc[seeds, mean_col].to_numpy(float), db.loc[seeds, mean_col].to_numpy(float)
    p_t = float(stats.ttest_rel(xa, xb).pvalue) if np.any(xa != xb) else 1.0
    cols = [c for c in client_columns(df, metric) if c in da.columns and c in db.columns]
    pa = da.loc[seeds, cols].to_numpy(float).ravel()
    pb = db.loc[seeds, cols].to_numpy(float).ravel()
    ok = ~(np.isnan(pa) | np.isnan(pb))
    pa, pb = pa[ok], pb[ok]
    p_w = float(stats.wilcoxon(pa, pb).pvalue) if len(pa) >= 6 and np.any(pa != pb) else float("nan")
    return 100 * float(np.mean(xa - xb)), p_t, p_w, int(ok.sum())


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (family-wise error control; NaNs are left as NaN)."""
    p = np.asarray(pvalues, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def write_lines(lines, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f">> Saved {out_path}")


# ----------------------------------------------------------------------------- tables
def generate_main_table(df: pd.DataFrame, out_path: Path, reference: str):
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\caption{Test performance across modality-diverse federated clients (setting S1). Mean $\\pm$ 95\\% CI over seeds "
        "of the client-averaged metric. $^\\dagger$: Wilcoxon signed-rank $p<0.05$ vs.\\ " + PLAIN_NAMES.get(reference, reference)
        + " over (seed, client) pairs. Comm.: total uploaded + downloaded MiB (fp16).}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{lccccc}", "\\toprule",
        "\\textbf{Method} & \\textbf{Accuracy (\\%)} & \\textbf{Balanced Acc (\\%)} & \\textbf{Macro-F1 (\\%)} & \\textbf{Macro-AUROC (\\%)} & \\textbf{Comm. (MiB)} \\\\",
        "\\midrule",
    ]
    for key, name in METHODS_DISPLAY_ORDER:
        m = df[df["method"] == key]
        if m.empty:
            lines.append(f"{name} & " + " & ".join(["\\textsc{NOT RUN}"] * 5) + " \\\\")
            continue
        mark = ""
        if key != reference:
            res = paired_tests(df, key, reference)
            if res and not np.isnan(res[2]) and res[2] < 0.05:
                mark = "$^\\dagger$"
        lines.append(
            f"{name}{mark} & {fmt(m['mean_test_accuracy'])} & {fmt(m['mean_test_balanced_acc'])} & "
            f"{fmt(m['mean_test_macro_f1'])} & {fmt(m['mean_test_auroc'])} & {m['total_comm_mb'].mean():.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    write_lines(lines, out_path)


def generate_per_client_table(df: pd.DataFrame, out_path: Path):
    cols = client_columns(df)
    clients = [c[: -len("_balanced_acc")] for c in cols]
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\caption{Per-client test balanced accuracy (\\%, mean $\\pm$ 95\\% CI over seeds), setting S1.}",
        "\\label{tab:per_client}",
        "\\begin{tabular}{l" + "c" * len(clients) + "}", "\\toprule",
        "\\textbf{Method} & " + " & ".join(f"\\textbf{{{c}}}" for c in clients) + " \\\\", "\\midrule",
    ]
    for key, name in METHODS_DISPLAY_ORDER:
        m = df[df["method"] == key]
        if m.empty:
            continue
        lines.append(name + " & " + " & ".join(fmt(m[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    write_lines(lines, out_path)


def generate_significance_table(df: pd.DataFrame, out_path: Path, ours: str = "fedmedlora"):
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{FedProtoLoRA minus each method in test balanced accuracy (percentage points). $p_t$: paired t-test over seeds; "
        "$p_W$: Wilcoxon signed-rank over (seed, client) pairs; $p_W^{\\mathrm{Holm}}$: Holm-adjusted over all comparisons.}",
        "\\label{tab:significance}",
        "\\begin{tabular}{lccccc}", "\\toprule",
        "\\textbf{Comparison} & $\\Delta$ (pp) & $p_t$ & $p_W$ & $p_W^{\\mathrm{Holm}}$ & pairs \\\\", "\\midrule",
    ]
    rows = []
    for key, _ in METHODS_DISPLAY_ORDER:
        if key == ours:
            continue
        res = paired_tests(df, ours, key)
        if res is not None:
            rows.append((key, *res))
    adj = holm([r[3] for r in rows])
    for (key, d, p_t, p_w, n), p_h in zip(rows, adj):
        lines.append(f"vs.\\ {PLAIN_NAMES[key]} & {d:+.2f} & {p_t:.3f} & {p_w:.3f} & {p_h:.3f} & {n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_lines(lines, out_path)


# Ablation rows that are main-suite runs (not re-run): A only == FlexLoRA, A+B+C == FedProtoLoRA.
ABLATION_FROM_MAIN = {"ablation_A_only": "flexlora", "ablation_full_A_B_C": "fedmedlora"}


def generate_ablation_table(df: pd.DataFrame, out_path: Path, main_df: pd.DataFrame = None):
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Ablation of FedProtoLoRA components (mean $\\pm$ 95\\% CI over seeds).}",
        "\\label{tab:ablations}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "\\textbf{Configuration} & \\textbf{Bal Acc (\\%)} & \\textbf{Macro-F1 (\\%)} & \\textbf{AUROC (\\%)} & \\textbf{Comm (MiB)} \\\\",
        "\\midrule",
    ]
    for var_key, var_name in ABLATIONS_DISPLAY_ORDER:
        if var_key in ABLATION_FROM_MAIN and main_df is not None:
            v = main_df[main_df["method"] == ABLATION_FROM_MAIN[var_key]]
        else:
            v = df[df["experiment_name"].str.startswith(var_key + "_")] if not df.empty else df
        if v.empty:
            lines.append(f"{var_name} & " + " & ".join(["\\textsc{NOT RUN}"] * 4) + " \\\\")
            continue
        lines.append(f"{var_name} & {fmt(v['mean_test_balanced_acc'])} & {fmt(v['mean_test_macro_f1'])} & "
                     f"{fmt(v['mean_test_auroc'])} & {v['total_comm_mb'].mean():.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_lines(lines, out_path)


def generate_tau_table(tau_df: pd.DataFrame, main_df: pd.DataFrame, out_path: Path):
    """Component C trade-off: energy threshold tau vs. accuracy and communication (tau=0.95 is the main run)."""
    rows = [(0.95, main_df[main_df["method"] == "fedmedlora"])]
    for tau in sorted(tau_df["tau"].dropna().unique()):
        rows.append((float(tau), tau_df[tau_df["tau"] == tau]))
    ref = main_df[main_df["method"] == "flexlora"]
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Adaptive upload rank: spectral-energy threshold $\\tau$ vs.\\ test balanced accuracy and total communication "
        "(mean $\\pm$ 95\\% CI over seeds). FlexLoRA (full-rank uploads) for reference.}",
        "\\label{tab:tau}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "\\textbf{Setting} & \\textbf{Bal Acc (\\%)} & \\textbf{Mean upload rank} & \\textbf{Comm (MiB)} \\\\", "\\midrule",
    ]
    if not ref.empty:
        lines.append(f"FlexLoRA & {fmt(ref['mean_test_balanced_acc'])} & {int(ref['rank'].iloc[0])} & {ref['total_comm_mb'].mean():.1f} \\\\")
    for tau, v in sorted(rows, key=lambda r: -r[0]):
        if v.empty:
            continue
        rank = v["mean_upload_rank"].mean() if "mean_upload_rank" in v else float("nan")
        lines.append(f"FedProtoLoRA, $\\tau={tau:g}$ & {fmt(v['mean_test_balanced_acc'])} & {rank:.1f} & {v['total_comm_mb'].mean():.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_lines(lines, out_path)


def generate_s2_significance(df: pd.DataFrame, out_path: Path, ours: str = "fedmedlora_personalized"):
    """S2: personalized FedProtoLoRA minus each variant, per alpha (paired over seeds / (seed, sub-client) pairs)."""
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Setting S2: personalized FedProtoLoRA minus each method in test balanced accuracy (percentage points). "
        "$p_t$: paired t-test over seeds; $p_W$: Wilcoxon signed-rank over (seed, sub-client) pairs; "
        "$p_W^{\\mathrm{Holm}}$: Holm-adjusted over all S2 comparisons.}",
        "\\label{tab:s2_significance}",
        "\\begin{tabular}{llccccc}", "\\toprule",
        "$\\alpha$ & \\textbf{Comparison} & $\\Delta$ (pp) & $p_t$ & $p_W$ & $p_W^{\\mathrm{Holm}}$ & pairs \\\\", "\\midrule",
    ]
    rows = []
    names = dict(S2_DISPLAY_ORDER)
    for a in sorted(df["dirichlet_alpha"].dropna().unique()):
        sub = df[df["dirichlet_alpha"] == a]
        sub = sub[[c for c in sub.columns if not c.endswith("_balanced_acc") or sub[c].notna().any()]]
        for key, _ in S2_DISPLAY_ORDER:
            if key == ours:
                continue
            res = paired_tests(sub, ours, key, key="variant")
            if res is None:
                continue
            rows.append((a, plain(names[key]), *res))
    adj = holm([r[4] for r in rows])
    for (a, label, d, p_t, p_w, n), p_h in zip(rows, adj):
        lines.append(f"{a:g} & vs.\\ {label} & {d:+.2f} & {p_t:.3f} & {p_w:.3f} & {p_h:.3f} & {n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_lines(lines, out_path)


def generate_s2_table(df: pd.DataFrame, out_path: Path):
    alphas = sorted(df["dirichlet_alpha"].dropna().unique()) if not df.empty else []
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Setting S2 (3 label-skewed sub-clients per modality): test balanced accuracy (\\%, mean $\\pm$ 95\\% CI) "
        "by Dirichlet concentration $\\alpha$ (smaller = more skew).}",
        "\\label{tab:s2}",
        "\\begin{tabular}{l" + "c" * max(1, len(alphas)) + "}", "\\toprule",
        "\\textbf{Method} & " + (" & ".join(f"$\\alpha={a:g}$" for a in alphas) or "--") + " \\\\", "\\midrule",
    ]
    for key, name in S2_DISPLAY_ORDER:
        cells = []
        for a in alphas:
            v = df[(df["variant"] == key) & (df["dirichlet_alpha"] == a)]
            cells.append(fmt(v["mean_test_balanced_acc"]) if not v.empty else "\\textsc{NOT RUN}")
        lines.append(f"{name} & " + (" & ".join(cells) or "\\textsc{NOT RUN}") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_lines(lines, out_path)


# ----------------------------------------------------------------------------- figures
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f">> Saved {path.with_suffix('.pdf')}")


def _spread_labels(ys, min_gap):
    """Nudges sorted end-of-line label positions apart by at least `min_gap` (data units), keeping order."""
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)
    for k in range(1, len(order)):
        prev, cur = order[k - 1], order[k]
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    return placed


def _place_point_labels(ax, fig, xs, ys, labels):
    """Greedy label placement around scatter points: first candidate offset not overlapping earlier labels/points."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken = []
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    point_boxes = [(px - 5, py - 5, px + 5, py + 5) for px, py in pts]
    candidates = [((5, 3), "left", "bottom"), ((5, -3), "left", "top"), ((-5, 3), "right", "bottom"),
                  ((-5, -3), "right", "top"), ((5, 9), "left", "bottom"), ((5, -9), "left", "top")]

    def overlaps(b, boxes):
        return any(not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3]) for o in boxes)

    for i in np.argsort(-np.asarray(ys)):
        for offset, ha, va in candidates:
            t = ax.annotate(labels[i], (xs[i], ys[i]), xytext=offset, textcoords="offset points",
                            fontsize=7, color=INK, ha=ha, va=va)
            bb = t.get_window_extent(renderer)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            others = [p for j, p in enumerate(point_boxes) if j != i]
            if not overlaps(box, taken) and not overlaps(box, others):
                taken.append(box)
                break
            t.remove()
        else:  # no free slot: keep the default position
            ax.annotate(labels[i], (xs[i], ys[i]), xytext=(5, 3), textcoords="offset points", fontsize=7, color=INK)


def plot_accuracy_vs_communication(df: pd.DataFrame, figures_dir: Path):
    agg = df.groupby("method").agg(acc=("mean_test_balanced_acc", "mean"), mb=("total_comm_mb", "mean")).reset_index()
    agg = agg[agg["mb"] > 0]  # methods without communication have no x position on a log axis
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for _, row in agg.iterrows():
        color, _, marker = HIGHLIGHT.get(row["method"], (NEUTRAL, "-", "o"))
        ax.scatter(row["mb"], row["acc"] * 100, s=46, color=color, marker=marker, edgecolors="white", linewidths=1.5, zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Total communication (MiB, log scale)", fontsize=9, color=INK)
    ax.set_ylabel("Mean test balanced accuracy (%)", fontsize=9, color=INK)
    _style(ax)
    ax.margins(x=0.12, y=0.08)
    _place_point_labels(ax, fig, agg["mb"].to_numpy(), (agg["acc"] * 100).to_numpy(),
                        [PLAIN_NAMES.get(m, m) for m in agg["method"]])
    _save(fig, figures_dir / "accuracy_vs_communication")


def plot_convergence(df: pd.DataFrame, results_dir: Path, figures_dir: Path):
    curves = {}
    for _, row in df.iterrows():
        hist_file = results_dir / row["experiment_name"] / "metrics_history.jsonl"
        if not hist_file.exists():
            continue
        hist = [json.loads(line) for line in hist_file.read_text().splitlines() if line]
        curves.setdefault(row["method"], []).append([h["avg_val_balanced_acc"] for h in hist])
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ends = {}
    # Gray context lines first, highlighted methods on top with direct end labels.
    for method in sorted(curves, key=lambda m: m in HIGHLIGHT):
        runs = curves[method]
        n = min(len(r) for r in runs)
        y = np.mean([r[:n] for r in runs], axis=0) * 100
        x = np.arange(1, n + 1)
        color, ls, marker = HIGHLIGHT.get(method, (NEUTRAL, "-", None))
        lw = 2.0 if method in HIGHLIGHT else 1.0
        ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, marker=marker, markersize=4,
                markevery=max(1, n // 5), zorder=3 if method in HIGHLIGHT else 2)
        if method in HIGHLIGHT:
            ends[method] = (x[-1], y[-1])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Communication round", fontsize=9, color=INK)
    ax.set_ylabel("Mean validation balanced accuracy (%)", fontsize=9, color=INK)
    ax.text(0.01, 0.01, "gray: other baselines", transform=ax.transAxes, fontsize=7, color=INK_MUTED)
    _style(ax)
    if ends:
        lo, hi = ax.get_ylim()
        methods = list(ends)
        placed = _spread_labels([ends[m][1] for m in methods], min_gap=(hi - lo) * 0.035)
        for m, y_lab in zip(methods, placed):
            x_end, y_end = ends[m]
            ax.annotate(PLAIN_NAMES.get(m, m), (x_end, y_end), xytext=(x_end + 0.6, y_lab), textcoords="data",
                        fontsize=7, color=INK, va="center", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.5, shrinkA=0, shrinkB=2))
    _save(fig, figures_dir / "convergence_curves")


def plot_rank_sweep(df: pd.DataFrame, figures_dir: Path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    dodge = {"fedmedlora": 0.96, "flexlora": 1.04}  # multiplicative offset on the log2 axis so bars do not overlap
    for method in ("flexlora", "fedmedlora"):
        m = df[df["method"] == method]
        if m.empty:
            continue
        # Plot only ranks with every seed present, so a partially finished sweep never shows a biased mean.
        n_seeds = m.groupby("rank")["seed"].nunique()
        m = m[m["rank"].isin(n_seeds[n_seeds == n_seeds.max()].index)]
        g = m.groupby("rank")["mean_test_balanced_acc"]
        ranks = np.array(sorted(g.groups))
        mean = g.mean().loc[ranks].to_numpy() * 100
        err = np.array([ci95(g.get_group(r) * 100) for r in ranks])
        color, ls, marker = HIGHLIGHT[method]
        xs = ranks * dodge[method]
        ax.errorbar(xs, mean, yerr=np.nan_to_num(err), color=color, linestyle=ls, marker=marker, markersize=5,
                    linewidth=2, elinewidth=1.0, capsize=3, zorder=3 if method == "fedmedlora" else 2)
        ax.annotate(PLAIN_NAMES[method], (xs[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=7, color=INK, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xticklabels(["4", "8", "16", "32"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(3.3, 52)
    ax.set_xlabel("LoRA rank r", fontsize=9, color=INK)
    ax.set_ylabel("Mean test balanced accuracy (%)", fontsize=9, color=INK)
    _style(ax)
    _save(fig, figures_dir / "accuracy_vs_rank")


def plot_s2(df: pd.DataFrame, figures_dir: Path):
    """S2 dot plot: test balanced accuracy (mean ± 95% CI over seeds) per method at each Dirichlet alpha."""
    if df.empty:
        return
    alphas = sorted(df["dirichlet_alpha"].dropna().unique())
    alpha_style = {a: c for a, c in zip(alphas, ["#2a78d6", "#eb6834"])}  # validated slots 1-2
    rows = [(k, plain(n)) for k, n in S2_DISPLAY_ORDER if (df["variant"] == k).any()]
    fig, ax = plt.subplots(figsize=(5.6, 0.45 * len(rows) + 1.0))
    for i, (key, _) in enumerate(rows):
        for j, a in enumerate(alphas):
            v = df[(df["variant"] == key) & (df["dirichlet_alpha"] == a)]["mean_test_balanced_acc"] * 100
            if v.empty:
                continue
            y = i + (j - (len(alphas) - 1) / 2) * 0.22
            ax.errorbar(v.mean(), y, xerr=np.nan_to_num(ci95(v)), fmt="o" if j == 0 else "s", color=alpha_style[a],
                        markersize=5, elinewidth=1.0, capsize=2, markeredgecolor="white", markeredgewidth=1.0,
                        label=f"$\\alpha={a:g}$" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for _, n in rows], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Mean test balanced accuracy (%), setting S2", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(alphas))
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    _save(fig, figures_dir / "s2_label_skew")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables, statistics and publication figures")
    parser.add_argument("--summary_csv", type=str, default="results/summary.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--reference", type=str, default="flexlora", help="Method used for the dagger significance marks")
    args = parser.parse_args()

    csv_path = Path(args.summary_csv)
    out_dir = Path(args.output_dir)
    base_cols = ["experiment_name", "suite", "method", "variant", "setting", "dirichlet_alpha", "rank", "seed",
                 "tau", "mean_upload_rank",
                 "mean_test_accuracy", "mean_test_balanced_acc", "mean_test_macro_f1", "mean_test_auroc", "total_comm_mb"]
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    # Runs from before the `suite` field existed (or no runs at all) still yield NOT RUN tables, not a crash.
    for col in base_cols:
        if col not in df.columns:
            df[col] = pd.Series(dtype=object if col in ("experiment_name", "suite", "method", "variant", "setting") else float)

    def suite(name):
        return df[df["suite"] == name]

    # Only main-suite runs in the main table: ablation and rank-sweep runs also carry method == "fedmedlora".
    main_df = suite("main")
    tables = out_dir / "tables"
    generate_main_table(main_df, tables / "main_results.tex", args.reference)
    if not main_df.empty:
        generate_per_client_table(main_df, tables / "per_client.tex")
        generate_significance_table(main_df, tables / "significance.tex")
    # Rows reused from the main suite are restricted to the seeds the sweep itself ran (same-seed comparison).
    def same_seeds(sweep):
        return main_df[main_df["seed"].isin(sweep["seed"].unique())] if not sweep.empty else main_df

    generate_ablation_table(suite("ablations"), tables / "ablations.tex", same_seeds(suite("ablations")))
    generate_tau_table(suite("tau"), same_seeds(suite("tau")), tables / "tau_tradeoff.tex")
    generate_s2_table(suite("s2"), tables / "s2_results.tex")
    generate_s2_significance(suite("s2"), tables / "s2_significance.tex")

    figures = out_dir / "figures"
    if not main_df.empty:
        plot_accuracy_vs_communication(main_df, figures)
        plot_convergence(main_df, Path(args.results_dir), figures)
    # Rank 16 of the sweep is the main-suite run.
    rank_main = same_seeds(suite("ranks"))
    rank_df = pd.concat([suite("ranks"), rank_main[rank_main["method"].isin(["fedmedlora", "flexlora"])]])
    # Learning-rate-matched FlexLoRA sweep (control R7), when available, replaces the FlexLoRA curve.
    ranks_lr = suite("ranks_lr")
    if not ranks_lr.empty:
        lrc = suite("lr_control")
        fx16 = lrc[(lrc["method"] == "flexlora") & lrc["seed"].isin(ranks_lr["seed"].unique())]
        rank_df = pd.concat([rank_df[rank_df["method"] != "flexlora"], ranks_lr, fx16])
    plot_rank_sweep(rank_df, figures)
    plot_s2(suite("s2"), figures)


if __name__ == "__main__":
    main()
