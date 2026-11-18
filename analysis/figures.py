"""Quantitative paper figures from the benchmark CSV (see figures/FIGURE_PLAN.md).

    python figures.py RESULTS_DIR [--out ../generated]

Writes vector PDFs (fonts embedded as TrueType) 174 mm wide:
    fig_feasibility.pdf   share of feasible plans vs implement width, per strategy, by obstacle density
    fig_decision.pdf      winning strategy (min cost C) per density x width tile
    fig_tradeoff.pdf      field time vs missed area (feasible plans), per width
    fig_sc_effect.pdf     overlap with SC off -> on per strategy (dumbbell) + product saved
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
W2 = 174 * MM
STRAT = ["B1", "B2", "B3"]
NAME = {"A": "A reference", "B1": "B1 holes", "B2": "B2 decomposition", "B3": "B3 drive-around"}
COL = {"A": "#8a8f96", "B1": "#2a78d6", "B2": "#eb6834", "B3": "#1baf7a"}
INK, INK2, MUTED, GRID = "#1f2933", "#52514e", "#8a8f96", "#e6e4dd"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def load(results):
    df = pd.read_csv(os.path.join(results, "runs.csv"))
    df["error"] = df["error"].fillna("")
    df["feasible"] = df["feasible"].astype(str).str.lower().isin(["true", "1"]) & (df["error"] == "")
    df["sc"] = df["sc"].astype(int)
    df = df[df["gate_type"].astype(str).str.contains("road")]
    return df


def _strategy_legend(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[s], marker="o", ms=4, mec="white", mew=0.8) for s in STRAT]
    fig.legend(h, [NAME[s] for s in STRAT], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.8, columnspacing=1.6)


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom", color=INK)


def fig_feasibility(df, out):
    d = df[(df.density_per_ha > 0) & (df.sc == 1) & df.strategy.isin(STRAT)]
    dens = sorted(d.density_per_ha.unique())
    widths = sorted(d.width_m.unique())
    fig, axs = plt.subplots(1, len(dens), figsize=(W2, 55 * MM), sharey=True)
    for j, (ax, de) in enumerate(zip(np.atleast_1d(axs), dens)):
        for s in STRAT:
            g = d[(d.density_per_ha == de) & (d.strategy == s)].groupby("width_m").feasible.agg(["mean", "size"])
            p = g["mean"] * 100
            # 95 % Wilson interval for a proportion
            n, ph = g["size"], g["mean"]
            z = 1.96
            den = 1 + z ** 2 / n
            ctr = (ph + z ** 2 / (2 * n)) / den
            half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
            ax.fill_between(p.index, 100 * (ctr - half), 100 * (ctr + half), color=COL[s], alpha=0.12, lw=0)
            ax.plot(p.index, p.values, "-o", color=COL[s], ms=4, mec="white", mew=0.8)
        ax.set_title(f"Obstacle density {de:g} ha$^{{-1}}$", color=INK)
        ax.set_xticks(widths)
        ax.set_xlabel("Implement width (m)")
        ax.set_ylim(-3, 103)
        panel_label(ax, "abc"[j])
    np.atleast_1d(axs)[0].set_ylabel("Feasible plans (%)")
    _strategy_legend(fig)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out, "fig_feasibility.pdf"))
    plt.close(fig)


def _text_colour(hex_):
    r, g, b = (int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    return "white" if lum < 0.25 else INK


def fig_decision(df, out):
    d = df[(df.density_per_ha > 0) & (df.sc == 1) & df.strategy.isin(STRAT)]
    sel = d[d.selected == 1]
    dens = sorted(d.density_per_ha.unique())
    widths = sorted(d.width_m.unique())
    n_inst = d.groupby(["density_per_ha", "width_m"]).instance_id.nunique()
    fig, ax = plt.subplots(figsize=(110 * MM, 62 * MM))
    ax.grid(False)
    for i, de in enumerate(dens):
        for j, w in enumerate(widths):
            s = sel[(sel.density_per_ha == de) & (sel.width_m == w)].strategy.value_counts()
            n = n_inst.get((de, w), 0)
            none = n - s.sum()
            if s.empty:
                win, share = None, 0
            else:
                win, share = s.index[0], 100 * s.iloc[0] / n
            fc = COL[win] if win else "#f0efec"
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fc=fc, ec="white", lw=2))
            tc = _text_colour(fc)
            ax.text(j + 0.5, i + 0.58, NAME[win].split()[0] if win else "none", ha="center", va="center",
                    color=tc, fontsize=9, fontweight="bold")
            sub = f"{share:.0f} % of paddocks" + (f"\n{100 * none / n:.0f} % none feasible" if none else "")
            ax.text(j + 0.5, i + 0.3, sub, ha="center", va="center", color=tc, fontsize=6.5, linespacing=1.1)
    ax.set_xlim(0, len(widths)); ax.set_ylim(0, len(dens))
    ax.set_xticks(np.arange(len(widths)) + 0.5, [f"{w:g} m" for w in widths])
    ax.set_yticks(np.arange(len(dens)) + 0.5, [f"{x:g}" for x in dens])
    ax.set_xlabel("Implement width"); ax.set_ylabel("Obstacles per ha")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[s]) for s in STRAT]
    ax.legend(handles, [NAME[s] for s in STRAT], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
              handlelength=1.2, columnspacing=1.2)
    fig.savefig(os.path.join(out, "fig_decision.pdf"))
    plt.close(fig)


def fig_tradeoff(df, out):
    d = df[(df.density_per_ha > 0) & (df.sc == 1)]
    a = d[d.strategy == "A"].set_index(["instance_id", "width_m"]).time_min.rename("time_A")
    f = d[d.strategy.isin(STRAT) & d.feasible].join(a, on=["instance_id", "width_m"])
    f = f.assign(dtime=100 * (f.time_min - f.time_A) / f.time_A)
    widths = sorted(f.width_m.unique())
    fig, axs = plt.subplots(1, len(widths), figsize=(W2, 62 * MM), sharey=True)
    for j, (ax, w) in enumerate(zip(np.atleast_1d(axs), widths)):
        for s in STRAT:
            g = f[(f.width_m == w) & (f.strategy == s)].groupby("density_per_ha")
            if not len(g):
                continue
            m, se, n = g[["dtime", "missed_pct"]].mean(), g[["dtime", "missed_pct"]].sem(), g.size()
            ax.errorbar(m.dtime, m.missed_pct, xerr=1.96 * se.dtime, yerr=1.96 * se.missed_pct, fmt="-o",
                        color=COL[s], ms=4, mec="white", mew=0.8, elinewidth=0.7, capsize=0, lw=1.0)
            de = max(m.index); row = m.loc[de]
            off = {"B1": (5, 4), "B2": (5, 4), "B3": (-5, 6)}[s]
            ax.annotate(f"{NAME[s].split()[0]} ({de:g} ha$^{{-1}}$)", (row.dtime, row.missed_pct), xytext=off,
                        textcoords="offset points", fontsize=6.5, color=INK, ha="left" if off[0] > 0 else "right")
        ax.set_title(f"w = {w:g} m", color=INK)
        ax.set_xlabel("Field time vs A (%)")
        panel_label(ax, "abc"[j])
    np.atleast_1d(axs)[0].set_ylabel("Missed target area, SC on (%)")
    _strategy_legend(fig)
    fig.text(0.5, -0.04, "Points: obstacle density 0.5 → 1 → 2 ha$^{-1}$ (labelled at 2); bars: 95 % CI; "
             "feasible plans only", ha="center", fontsize=6.5, color=INK2)
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out, "fig_tradeoff.pdf"))
    plt.close(fig)


def fig_sc_effect(df, out):
    d = df[(df.density_per_ha > 0) & df.strategy.isin(["A"] + STRAT)]
    d = d[d.feasible | (d.strategy == "A")]
    g = d.groupby(["strategy", "sc"]).overlap_pct.mean().unstack()
    app = d.groupby(["strategy", "sc"]).applied_ha.mean().unstack()
    saved = 100 * (1 - app[1] / app[0])
    order = ["A"] + STRAT
    fig, ax = plt.subplots(figsize=(110 * MM, 50 * MM))
    for i, s in enumerate(order[::-1]):
        off, on = g.loc[s, 0], g.loc[s, 1]
        ax.plot([on, off], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.scatter([off], [i], s=36, facecolor="white", edgecolor=COL[s], lw=1.4, zorder=3)
        ax.scatter([on], [i], s=36, color=COL[s], edgecolor="white", lw=0.8, zorder=3)
        ax.text(max(g[0]) * 1.06, i, f"{saved[s]:.1f} % less product", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(order)), [NAME[s] for s in order[::-1]])
    ax.set_xlabel("Overlap (% of target applied ≥ 2 times)")
    ax.set_xlim(0, max(g[0]) * 1.45)
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=36, facecolor="white", edgecolor=INK2, label="SC off")
    ax.scatter([], [], s=36, color=INK2, label="SC on")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, handletextpad=0.3, fontsize=7)
    fig.savefig(os.path.join(out, "fig_sc_effect.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "generated"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.results)
    for f in (fig_feasibility, fig_decision, fig_tradeoff, fig_sc_effect):
        f(df, a.out)
        print("wrote", f.__name__ + ".pdf")


if __name__ == "__main__":
    main()
