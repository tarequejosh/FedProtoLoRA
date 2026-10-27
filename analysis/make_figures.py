"""Figures of the FedMedLoRA paper, drawn from the run logs in results/.

Run from the repository root:  python analysis/make_figures.py [--out outputs/figures]
Writes vector PDFs and 600-dpi PNGs. Every figure is drawn at its printed size (text width 6.48 in), and each
method keeps the same colour in every figure.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_report import ci95  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "outputs" / "figures"
TEXT_WIDTH = 6.48  # inches (elsarticle 3p)

# One colour per entity, used in every figure (validated all-pairs for colour-vision deficiency).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ORACLE = "#3a3936"          # reference that uses extra information: dark ink, diamond marker
GREY, GREY_DARK = "#b4b3ad", "#6f6e69"
INK, INK2, MUTED, GRID = "#1a1a19", "#52514e", "#898781", "#e6e5e0"

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": MUTED, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "text.color": INK, "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "savefig.dpi": 600,
})

S1_NAMES = {
    "local_only": "Local-only LoRA", "centralized": "Centralized LoRA (pooled)", "fedavg_full": "FedAvg (full fine-tuning)",
    "fedavg_lora": "FedAvg-LoRA", "fedprox_lora": "FedProx-LoRA", "ffa_lora": "FFA-LoRA", "fedsa_lora": "FedSA-LoRA",
    "fedex_lora": "FedEx-LoRA", "flora": "FLoRA", "hetlora": "HetLoRA", "flexlora": "FlexLoRA",
    "fedmedlora": "FedMedLoRA",
}
ETA_LO, ETA_HI = r"$\eta = 3{\times}10^{-4}$", r"$\eta = 10^{-3}$"


def style(ax, grid_axis="both"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel(ax, letter, title, x=-0.02, y=1.035):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")
    ax.text(x + 0.02, y, title, transform=ax.transAxes, fontsize=8, va="bottom", ha="left", color=INK)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)
    print(f">> {OUT / name}.pdf / .png")


def history(experiment):
    lines = (RES / experiment / "metrics_history.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def jitter(n, width=0.16, seed=0):
    return np.random.default_rng(seed).uniform(-width, width, n)


# ----------------------------------------------------------------------------------------------------------- S1
def fig_s1(df):
    main = df[df.suite == "main"]
    lrc = df[df.suite == "lr_control"]
    acc = lambda d: d.mean_test_balanced_acc.to_numpy() * 100  # noqa: E731

    # Rows of panel a: (label, values, colour, filled)
    rows = []
    for m in S1_NAMES:
        v = acc(main[main.method == m])
        colour = BLUE if m == "fedmedlora" else ORANGE if m == "flexlora" else GREY_DARK
        label = S1_NAMES[m] + (f"  ({ETA_LO})" if m == "flexlora" else "")
        rows.append((label, v, colour, m != "flexlora", m))
    rows.append((f"FlexLoRA  ({ETA_HI})", acc(lrc[lrc.method == "flexlora"]), ORANGE, True, "flexlora_hi"))
    federated = [r for r in rows if r[4] not in ("local_only", "centralized")]
    federated.sort(key=lambda r: np.mean(r[1]))
    reference = [r for r in rows if r[4] in ("local_only", "centralized")]
    rows = reference + federated  # bottom-to-top order on the y axis

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1.15], wspace=0.34, hspace=0.6,
                          left=0.215, right=0.975, top=0.935, bottom=0.1)
    ax = fig.add_subplot(gs[:, 0])
    for i, (label, v, colour, filled, key) in enumerate(rows):
        lo, hi = np.mean(v) - ci95(v), np.mean(v) + ci95(v)
        ax.plot([lo, hi], [i, i], color=colour, linewidth=1.1, solid_capstyle="butt", zorder=2)
        ax.scatter(v, i + jitter(len(v), 0.13, i), s=5, color=colour, alpha=0.35, linewidths=0, zorder=2)
        ax.scatter(np.mean(v), i, s=24, marker="o", facecolor=colour if filled else "white", edgecolor=colour,
                   linewidths=1.1, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[4] == "fedmedlora":
            t.set_fontweight("bold")
    ax.axhline(len(reference) - 0.5, color=MUTED, linewidth=0.5)
    ax.text(1.0, len(reference) - 0.38, "federated", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=6.5, color=MUTED, style="italic")
    ax.text(1.0, len(reference) - 0.62, "non-federated", transform=ax.get_yaxis_transform(), ha="right", va="top",
            fontsize=6.5, color=MUTED, style="italic")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Test balanced accuracy (%)")
    style(ax, "x")
    ax.tick_params(axis="y", length=0)
    panel(ax, "a", "Setting S1: five clients, one per modality", x=-0.02)

    # Panel b: the learning-rate control, seed by seed.
    bx = fig.add_subplot(gs[0, 1])
    pairs = {
        "FlexLoRA": (main[main.method == "flexlora"], lrc[lrc.method == "flexlora"], ORANGE, -0.06),
        "FedMedLoRA": (lrc[lrc.method == "fedmedlora"], main[main.method == "fedmedlora"], BLUE, 0.06),
    }
    for name, (lo_df, hi_df, colour, dx) in pairs.items():
        lo_s = lo_df.set_index("seed").mean_test_balanced_acc * 100
        hi_s = hi_df.set_index("seed").mean_test_balanced_acc * 100
        for s in lo_s.index:
            bx.plot([0 + dx, 1 + dx], [lo_s[s], hi_s[s]], color=colour, linewidth=0.5, alpha=0.3, zorder=2)
        bx.plot([0 + dx, 1 + dx], [lo_s.mean(), hi_s.mean()], color=colour, linewidth=1.8, zorder=3,
                marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6)
        bx.annotate(f"{name}\n{lo_s.mean():.1f} → {hi_s.mean():.1f}", (1 + dx, hi_s.mean()),
                    xytext=(9, 7 if name == "FedMedLoRA" else -9), textcoords="offset points",
                    va="center", ha="left", fontsize=6.5, color=colour if name == "FedMedLoRA" else "#c4521f")
    bx.set_xticks([0, 1])
    bx.set_xticklabels([r"$3{\times}10^{-4}$", r"$10^{-3}$"])
    bx.set_xlim(-0.3, 1.95)
    bx.set_xlabel(r"Learning rate $\eta$")
    bx.set_ylabel("Balanced acc. (%)")
    style(bx, "y")
    panel(bx, "b", "Learning-rate control (lines: seeds)", x=-0.02)

    # Panel c: accuracy against total communication.
    cx = fig.add_subplot(gs[1, 1])
    comm = main.groupby("method").agg(acc=("mean_test_balanced_acc", "mean"), mb=("total_comm_mb", "mean"))
    comm = comm[comm.mb > 0]
    comm.loc["flexlora_hi"] = [lrc[lrc.method == "flexlora"].mean_test_balanced_acc.mean(),
                               lrc[lrc.method == "flexlora"].total_comm_mb.mean()]
    # Label anchors. Five methods share ~112 MiB, so the upper three get leader lines into the empty area
    # to their right; the rest sit beside their point. Positions are in data coordinates (x in MiB).
    place = {
        "fedavg_lora": ((200, 87.78), "left"), "fedmedlora": ((200, 87.48), "left"), "flexlora_hi": ((200, 87.18), "left"),
        "fedsa_lora": ((40, 87.82), "center"), "hetlora": ((63, 86.33), "right"), "fedex_lora": ((960, 86.85), "left"),
        "flora": ((410, 86.66), "left"), "fedprox_lora": ((135, 86.18), "left"), "flexlora": ((135, 85.8), "left"),
        "ffa_lora": ((135, 85.47), "left"), "fedavg_full": ((8400, 85.5), "center"),
    }
    leader = {"fedavg_lora", "fedmedlora", "flexlora_hi", "fedsa_lora"}
    names = dict(S1_NAMES, flexlora_hi="FlexLoRA, " + ETA_HI, flexlora="FlexLoRA, " + ETA_LO,
                 fedavg_full="FedAvg (full FT)")
    for m, r in comm.iterrows():
        colour = BLUE if m == "fedmedlora" else ORANGE if m.startswith("flexlora") else GREY_DARK
        filled = m != "flexlora"
        cx.scatter(r.mb, r.acc * 100, s=20, facecolor=colour if filled else "white", edgecolor=colour,
                   linewidths=1.0, zorder=3)
        (tx, ty), ha = place[m]
        text_colour = BLUE if m == "fedmedlora" else "#c4521f" if m.startswith("flexlora") else INK2
        cx.annotate(names[m], (r.mb, r.acc * 100), xytext=(tx, ty), textcoords="data", fontsize=6, ha=ha,
                    va="top" if m == "fedavg_full" else "center", color=text_colour,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5, shrinkA=1, shrinkB=3)
                    if m in leader else None)
    cx.set_xscale("log")
    cx.xaxis.set_major_locator(FixedLocator([30, 100, 300, 1000, 3000, 10000]))
    cx.set_xticklabels(["30", "100", "300", "1k", "3k", "10k"])
    cx.xaxis.set_minor_formatter(NullFormatter())
    cx.set_xlim(14, 20000)
    cx.set_xlabel("Communication over 20 rounds (MiB, log scale)")
    cx.set_ylim(85.1, 88.0)
    cx.set_ylabel("Balanced acc. (%)")
    style(cx)
    panel(cx, "c", "Accuracy and communication", x=-0.02)
    save(fig, "fig_s1")


# ----------------------------------------------------------------------------------------------------------- S2
S2_ROWS = [  # key, label, colour, filled, marker
    ("local_only", "Local-only LoRA", GREY_DARK, True, "o"),
    ("fedsa_lora", "FedSA-LoRA", GREY_DARK, True, "o"),
    ("flexlora", "FlexLoRA, " + ETA_LO, ORANGE, False, "o"),
    ("flexlora_lr1e-3", "FlexLoRA, " + ETA_HI, ORANGE, True, "o"),
    ("fedavg_lora", "FedAvg-LoRA", AQUA, True, "o"),
    ("fedmedlora", "FedMedLoRA (global)", BLUE, False, "o"),
    ("fedmedlora_personalized", "FedMedLoRA (personalized)", BLUE, True, "o"),
    ("oracle_modality", "Oracle: FlexLoRA within modality", ORACLE, True, "D"),
]


def fig_s2(df):
    s2 = df[df.suite == "s2"]
    alphas = [0.1, 0.5]
    fig = plt.figure(figsize=(TEXT_WIDTH, 4.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], wspace=0.08, hspace=0.55,
                          left=0.255, right=0.985, top=0.93, bottom=0.09)
    top = [fig.add_subplot(gs[0, j]) for j in range(2)]
    bot = [fig.add_subplot(gs[1, j]) for j in range(2)]
    for j, a in enumerate(alphas):
        d = s2[s2.dirichlet_alpha == a]
        ax = top[j]
        for i, (key, label, colour, filled, marker) in enumerate(S2_ROWS):
            v = d[d.variant == key].mean_test_balanced_acc.to_numpy() * 100
            m, h = v.mean(), ci95(v)
            ax.plot([m - h, m + h], [i, i], color=colour, linewidth=1.1, zorder=2)
            ax.scatter(v, i + jitter(len(v), 0.14, i), s=5, color=colour, alpha=0.4, linewidths=0, zorder=2)
            ax.scatter(m, i, s=26 if marker == "o" else 22, marker=marker, facecolor=colour if filled else "white",
                       edgecolor=colour, linewidths=1.1, zorder=3)
        ax.set_yticks(range(len(S2_ROWS)))
        ax.set_yticklabels([r[1] for r in S2_ROWS] if j == 0 else [])
        if j == 0:
            for t, r in zip(ax.get_yticklabels(), S2_ROWS):
                if r[0] == "fedmedlora_personalized":
                    t.set_fontweight("bold")
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(S2_ROWS) - 0.4)
        ax.set_xlim(66, 97)
        ax.set_xlabel("Test balanced accuracy (%)")
        style(ax, "x")
        skew = "strong" if a == 0.1 else "mild"
        panel(ax, "ab"[j], rf"$\alpha = {a:g}$, {skew} label skew", x=-0.01)

        # Bottom: per-seed gain of each method over learning-rate-matched FlexLoRA.
        bx = bot[j]
        ref = d[d.variant == "flexlora_lr1e-3"].set_index("seed").mean_test_balanced_acc
        comps = [("fedavg_lora", "FedAvg-LoRA", AQUA, "o", True), ("fedmedlora", "FedMedLoRA (global)", BLUE, "o", False),
                 ("fedmedlora_personalized", "FedMedLoRA (personalized)", BLUE, "o", True),
                 ("oracle_modality", "Oracle: FlexLoRA within modality", ORACLE, "D", True)]
        for i, (key, label, colour, marker, filled) in enumerate(comps):
            g = (d[d.variant == key].set_index("seed").mean_test_balanced_acc - ref).dropna() * 100
            bx.scatter(g.to_numpy(), np.full(len(g), i) + jitter(len(g), 0.12, 10 + i), s=9, color=colour,
                       alpha=0.55, linewidths=0, zorder=2)
            bx.plot([g.mean()] * 2, [i - 0.3, i + 0.3], color=colour, linewidth=1.8, zorder=3)
            bx.text(1.0, i, f"{g.mean():+.1f}", transform=bx.get_yaxis_transform(), ha="right", va="center",
                    fontsize=6.8, color=INK, fontweight="bold" if key == "fedmedlora_personalized" else "normal")
        bx.axvline(0, color=INK2, linewidth=0.7, zorder=1)
        bx.set_yticks(range(len(comps)))
        bx.set_yticklabels([c[1] for c in comps] if j == 0 else [])
        bx.tick_params(axis="y", length=0)
        bx.set_ylim(-0.6, len(comps) - 0.4)
        bx.set_xlim(-5, 10.5)
        bx.set_xticks([-4, -2, 0, 2, 4, 6, 8])
        bx.text(1.0, len(comps) - 0.45, "mean", transform=bx.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=6.3, color=MUTED, style="italic")
        bx.set_xlabel("Difference in balanced accuracy (percentage points)")
        style(bx, "x")
        panel(bx, "cd"[j], rf"Gain over FlexLoRA ({ETA_HI}), $\alpha = {a:g}$", x=-0.01)
    save(fig, "fig_s2")


# ------------------------------------------------------------------------------------------------------ weights
MOD_LABEL = {"derm": "Derm", "oct": "OCT", "cxr": "CXR", "brain": "MRI"}


def same_modality_share(weights):
    """Mean over targets of the total weight given to clients of the target's own modality (incl. itself)."""
    if "shared" in weights:  # global aggregate: every target sees the same weights
        weights = {t: weights["shared"] for t in weights["shared"]}
    return np.mean([sum(w for src, w in row.items() if src.split("_")[0] == tgt.split("_")[0])
                    for tgt, row in weights.items()])


def fig_weights(df):
    s2 = df[(df.suite == "s2") & (df.dirichlet_alpha == 0.1)]
    beta = df[df.suite == "beta"]
    seeds = sorted(beta.seed.unique())

    runs = s2[s2.variant == "fedmedlora_personalized"].experiment_name
    order, mats = None, []
    for e in runs:
        w = history(e)[0]["aggregation_weights"]
        order = order or sorted(w, key=lambda c: (list(MOD_LABEL).index(c.split("_")[0]), c))
        mats.append([[w[t][s] for s in order] for t in order])
    W = np.mean(mats, axis=0)
    labels = [f"{MOD_LABEL[c.split('_')[0]]} {int(c.split('_')[1]) + 1}" for c in order]

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.75, wspace=0.42,
                          left=0.085, right=0.985, top=0.9, bottom=0.2)
    ax = fig.add_subplot(gs[:, 0])
    cmap = mpl.colors.LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    im = ax.imshow(W, cmap=cmap, vmin=0, vmax=W.max(), aspect="equal")
    for k in range(1, 4):
        ax.axhline(3 * k - 0.5, color="white", linewidth=1.6)
        ax.axvline(3 * k - 0.5, color="white", linewidth=1.6)
    for k in range(4):
        ax.add_patch(mpl.patches.Rectangle((3 * k - 0.42, 3 * k - 0.42), 2.84, 2.84, fill=False, edgecolor=INK,
                                           linewidth=0.9, zorder=5))
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("Source institution (whose update is used)")
    ax.set_ylabel("Target institution")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_ticks([0, 0.05, 0.10, 0.15, 0.20])
    cb.set_label("Aggregation weight", fontsize=7)
    panel(ax, "a", r"Personalized weights, $\alpha = 0.1$ (mean of 5 seeds)", x=0.0)

    # Panels b/c: accuracy and same-modality weight share as functions of beta (seeds of the beta sweep).
    pts = {0.0: s2[(s2.variant == "flexlora_lr1e-3") & s2.seed.isin(seeds)],
           0.5: s2[(s2.variant == "fedmedlora_personalized") & s2.seed.isin(seeds)]}
    for b in sorted(beta.blend_alpha.unique()):
        pts[b] = beta[beta.blend_alpha == b]
    bs = sorted(pts)
    acc = {b: pts[b].mean_test_balanced_acc.to_numpy() * 100 for b in bs}
    share = {b: np.array([same_modality_share(history(e)[0]["aggregation_weights"]) for e in pts[b].experiment_name]) * 100
             for b in bs}

    bx = fig.add_subplot(gs[0, 1])
    cx = fig.add_subplot(gs[1, 1], sharex=bx)
    for axis, data, ylabel in ((bx, acc, "Balanced acc. (%)"), (cx, share, "Same-modality\nweight (%)")):
        means = [data[b].mean() for b in bs]
        axis.plot(bs, means, color=BLUE, linewidth=1.4, zorder=2)
        for b in bs:
            axis.scatter(np.full(len(data[b]), b) + jitter(len(data[b]), 0.018, int(b * 100)), data[b], s=6,
                         color=BLUE, alpha=0.4, linewidths=0, zorder=2)
            axis.scatter(b, data[b].mean(), s=20, facecolor=ORANGE if b == 0 else BLUE, edgecolor="white",
                         linewidths=0.6, zorder=3)
        axis.set_ylabel(ylabel)
        style(axis)
    bx.annotate("FlexLoRA", (0, acc[0].mean()), xytext=(6, -2), textcoords="offset points", fontsize=6.3,
                color="#c4521f", va="top")
    bx.set_ylim(81, 91.5)
    bx.set_yticks([82, 84, 86, 88, 90])
    bx.annotate(r"default ($\beta = 0.5$)", (0.5, acc[0.5].mean()), xytext=(8, 2), textcoords="offset points",
                fontsize=6.3, color=INK2, ha="left", va="bottom")
    cx.axhline(25, color=MUTED, linewidth=0.7, zorder=1)
    cx.text(1.0, 25.8, "uniform weights (25%)", fontsize=6.3, color=INK2, ha="right", va="bottom")
    cx.set_ylim(15, 58)
    cx.set_xticks(bs)
    cx.set_xticklabels([f"{b:g}" for b in bs])
    cx.set_xlabel(r"Blend weight $\beta$")
    plt.setp(bx.get_xticklabels(), visible=False)
    panel(bx, "b", rf"Accuracy vs. $\beta$ ($\alpha = 0.1$, {len(seeds)} seeds)", x=-0.02, y=1.07)
    panel(cx, "c", "Weight on same-modality institutions", x=-0.02, y=1.07)
    save(fig, "fig_weights")
    print(f"   same-modality weight share at beta=0.5 (5 seeds): {np.mean([same_modality_share(history(e)[0]['aggregation_weights']) for e in runs]):.3f}")
    for b in bs:
        print(f"   beta={b:g}: acc {acc[b].mean():.2f}, same-modality share {share[b].mean():.1f}%")


# ------------------------------------------------------------------------------------------- rank + convergence
def fig_rank_convergence(df):
    main, lrc = df[df.suite == "main"], df[df.suite == "lr_control"]
    seeds3 = [42, 101, 2024]
    fed = pd.concat([df[(df.suite == "ranks") & (df.method == "fedmedlora")],
                     main[(main.method == "fedmedlora") & main.seed.isin(seeds3)]])
    flex = pd.concat([df[df.suite == "ranks_lr"], lrc[(lrc.method == "flexlora") & lrc.seed.isin(seeds3)]])

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.4], wspace=0.28, left=0.075, right=0.985, top=0.91, bottom=0.42)
    ax = fig.add_subplot(gs[0, 0])
    for d, colour, dx, name in ((flex, ORANGE, 1.05, "FlexLoRA"), (fed, BLUE, 0.95, "FedMedLoRA")):
        g = d.groupby("rank").mean_test_balanced_acc
        ranks = np.array(sorted(g.groups))
        assert all(len(g.get_group(r)) == 3 for r in ranks), "rank sweep must have three seeds per rank"
        means = np.array([g.get_group(r).mean() for r in ranks]) * 100
        for r in ranks:  # individual seeds; with three seeds a t-interval would mostly show the t quantile
            ax.scatter(np.full(3, r * dx), g.get_group(r).to_numpy() * 100, s=7, color=colour, alpha=0.45,
                       linewidths=0, zorder=2)
        ax.plot(ranks * dx, means, color=colour, linewidth=1.5, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        print(f"   rank {name}: " + ", ".join(f"r={r}: {m:.1f}" for r, m in zip(ranks, means)))
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([4, 8, 16, 32]))
    ax.set_xticklabels(["4", "8", "16", "32"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(3.3, 39)
    ax.set_xlabel("LoRA rank $r$")
    ax.set_ylabel("Test balanced accuracy (%)")
    style(ax)
    panel(ax, "a", rf"LoRA rank (both {ETA_HI}, 3 seeds)", x=-0.02)
    rank_handles = [Line2D([], [], color=BLUE, marker="o", markersize=4, linewidth=1.5, label="FedMedLoRA"),
                    Line2D([], [], color=ORANGE, marker="o", markersize=4, linewidth=1.5, label="FlexLoRA"),
                    Line2D([], [], color=MUTED, marker="o", markersize=2.5, linewidth=0, alpha=0.6, label="single seed")]
    ax.legend(handles=rank_handles, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False,
              handlelength=2.0, columnspacing=1.2)

    bx = fig.add_subplot(gs[0, 1])
    curves = {}
    for _, row in pd.concat([main, lrc.assign(method=lrc.method + "_lrc")]).iterrows():
        curves.setdefault(row.method, []).append([h["avg_val_balanced_acc"] * 100 for h in history(row.experiment_name)])
    highlight = [  # key, label, colour, linestyle, linewidth
        ("fedmedlora", f"FedMedLoRA, {ETA_HI}", BLUE, "-", 1.7),
        ("flexlora_lrc", f"FlexLoRA, {ETA_HI}", ORANGE, "-", 1.5),
        ("flexlora", f"FlexLoRA, {ETA_LO}", ORANGE, (0, (3.5, 1.8)), 1.4),
        ("fedavg_lora", f"FedAvg-LoRA, {ETA_HI}", AQUA, "-", 1.4),
        ("fedsa_lora", "FedSA-LoRA (half of each adapter local)", GREY_DARK, (0, (5, 1.5, 1, 1.5)), 1.2),
        ("local_only", "Local-only LoRA (no communication)", INK, (0, (1, 1.3)), 1.4),
    ]
    keys = {h[0] for h in highlight}
    others = [m for m in curves if m not in keys and m in S1_NAMES and m != "centralized"]
    rounds = np.arange(1, 21)
    other_means = np.array([np.mean(curves[m], axis=0) for m in others])
    bx.fill_between(rounds, other_means.min(axis=0), other_means.max(axis=0), color="#ebeae6", linewidth=0, zorder=1)
    handles = []
    for key, label, colour, ls, lw in highlight:
        bx.plot(rounds, np.mean(curves[key], axis=0), color=colour, linestyle=ls, linewidth=lw, zorder=3,
                solid_capstyle="round")
        handles.append(Line2D([], [], color=colour, linestyle=ls, linewidth=lw, label=label))
    handles.append(mpl.patches.Patch(color="#ebeae6", label=f"Range of the other {len(others)} federated baselines"))
    bx.set_xlim(1, 20)
    bx.set_xticks([1, 5, 10, 15, 20])
    bx.set_ylim(60, 91)
    bx.set_xlabel("Communication round")
    bx.set_ylabel("Validation balanced accuracy (%)")
    style(bx)
    bx.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.27), ncol=2, frameon=False,
              handlelength=2.8, columnspacing=1.5, labelspacing=0.45)
    panel(bx, "b", "Training curves in S1 (mean of 5 seeds)", x=-0.02)
    save(fig, "fig_rank_conv")
    for key, *_ in highlight:
        y = np.mean(curves[key], axis=0)
        print(f"   {key}: round1 {y[0]:.1f}, round3 {y[2]:.1f}, round10 {y[9]:.1f}, round20 {y[19]:.1f}")
    for m in others + ["fedsa_lora"]:
        if m in curves:
            y = np.mean(curves[m], axis=0)
            print(f"   (other) {m}: round1 {y[0]:.1f}, round3 {y[2]:.1f}, round20 {y[19]:.1f}")


# ------------------------------------------------------------------------------- parts of the overview (Fig. 1)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="output folder for the figures")
    OUT = parser.parse_args().out.resolve()
    df = pd.read_csv(RES / "summary.csv")
    fig_s1(df)
    fig_s2(df)
    fig_weights(df)
    fig_rank_convergence(df)


if __name__ == "__main__":
    main()
