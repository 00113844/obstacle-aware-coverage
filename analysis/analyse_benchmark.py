"""Statistical analysis of the synthetic benchmark (paper Section 4).

Usage:
    python analyse_benchmark.py RESULTS_DIR [--out ../generated]

Reads RESULTS_DIR/runs.csv (schema: tools/f2c_bench.py COLUMNS) and RESULTS_DIR/run_meta.json and writes:
    generated/tab_*.tex       LaTeX tables (booktabs) included by sections/results.tex
    generated/fig_*.pdf       figures (174 mm wide, sans-serif, embedded fonts)
    generated/numbers.tex     \\newcommand macros for every number quoted in the text
    generated/models.txt      full mixed-model output

Conventions (from the coding session's delivery notes):
- Density-0 instances contain only A rows. In obstacle instances A is the infeasible reference.
- B3 rows repeat A's t_hg..t_pp; B3's own planning time is t_detour, so B3's total = A's total + t_detour.
- The `selected` column marks the strategy (B1/B2/B3) that minimises crit_C (C = c_t T + c_p applied_SC + c_m missed).
- Rows with a non-empty `error` are counted and excluded.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

STRATS = ["A", "B1", "B2", "B3"]
FACT = ["shape", "density_per_ha", "obstacle_type", "pattern", "width_m"]
MM = 1 / 25.4


def load(results):
    df = pd.read_csv(os.path.join(results, "runs.csv"))
    meta = {}
    p = os.path.join(results, "run_meta.json")
    if os.path.exists(p):
        with open(p, encoding="utf8") as fh:
            meta = json.load(fh)
    df["error"] = df["error"].fillna("")
    # Rows with an error ("no swaths", GEOS exceptions, ...) are planning failures, not missing
    # data: keep them and count them as infeasible so feasibility rates are not inflated.
    n_err = int((df["error"] != "").sum())
    df["feasible"] = df["feasible"].astype(str).str.lower().isin(["true", "1"]) & (df["error"] == "")
    df["sc"] = df["sc"].astype(int)
    df["strategy"] = df["strategy"].astype(str).str.split().str[0].str.rstrip("·:")  # tolerate labels like "B1 · holes"
    df["t_plan_s"] = df[["t_hg_s", "t_sg_s", "t_rp_s", "t_pp_s"]].fillna(0).sum(1)
    b3 = df["strategy"] == "B3"
    df.loc[b3, "t_plan_s"] = df.loc[b3, "t_plan_s"] + df.loc[b3, "t_detour_s"].fillna(0)
    return df, meta, n_err


def main_set(df):
    """Factorial runs on the road-side gate (the gate subset is analysed separately)."""
    if "gate_type" in df:
        df = df[df["gate_type"].astype(str).str.contains("road")]
    return df


def rel_to_A(df, cols=("time_min", "L_total_m", "L_turn_m", "turns")):
    """Per instance/width/SC: each strategy relative to the open-paddock reference A (%)."""
    key = ["instance_id", "width_m", "sc"]
    a = df[df.strategy == "A"].set_index(key)[list(cols)]
    out = df[df.strategy != "A"].join(a, on=key, rsuffix="_A")
    for c in cols:
        out[f"d_{c}_pct"] = 100 * (out[c] - out[f"{c}_A"]) / out[f"{c}_A"]
    return out


def feasibility_table(df, out):
    """Share of plans (SC on, obstacle instances) that are feasible, and why the others are not."""
    d = df[(df.density_per_ha > 0) & (df.sc == 1)].copy()
    d["planning_error"] = d["error"] != ""
    d["unplaced_detour"] = d["n_unplaced_detours"].fillna(0) > 0
    d["clearance_work"] = d.get("min_clearance_work_m", pd.Series(np.nan, index=d.index)) < 0
    d["clearance_turn"] = d.get("min_clearance_turn_m", pd.Series(np.nan, index=d.index)) < 0
    d["outside_field"] = d["outside_paddock_m"].fillna(0) > 1.0
    cols = ["feasible", "planning_error", "unplaced_detour", "clearance_work", "clearance_turn", "outside_field"]
    g = d.groupby(["strategy", "width_m"])[cols].mean().mul(100).round(0)
    with open(os.path.join(out, "tab_feasibility.tex"), "w", encoding="utf8") as fh:
        fh.write(g.to_latex(escape=True))
    return g


def summary_table(df, out):
    """Feasibility over all plans; performance means over feasible plans only (SC on)."""
    d = df[(df.density_per_ha > 0) & (df.sc == 1)]
    feas = d.groupby(["strategy", "density_per_ha"]).feasible.mean().mul(100).rename("feasible_pct")
    f = d[d.feasible]
    g = f.groupby(["strategy", "density_per_ha"]).agg(
        n=("time_min", "size"), time_min=("time_min", "mean"), turns=("turns", "mean"),
        L_turn_m=("L_turn_m", "mean"), missed=("missed_pct", "mean"), overlap=("overlap_pct", "mean"))
    g = g.join(feas, how="outer")
    with open(os.path.join(out, "tab_summary.tex"), "w", encoding="utf8") as fh:
        fh.write(g.round(2).to_latex(escape=True))
    return g


def sc_table(df, out):
    """SC effect over the same population as the text: feasible plans (and the A reference)."""
    d = df[(df.density_per_ha > 0) & (df.feasible | (df.strategy == "A"))]
    g = d.pivot_table(index=["strategy", "width_m"], columns="sc",
                      values=["missed_pct", "overlap_pct", "offtarget_m2", "applied_ha"], aggfunc="mean")
    with open(os.path.join(out, "tab_sc.tex"), "w", encoding="utf8") as fh:
        fh.write(g.round(2).to_latex(escape=True))
    return g


def decision_map(df, out):
    """Share of instances in which each strategy minimises C, by density x width (SC on)."""
    d = df[(df.density_per_ha > 0) & (df.sc == 1) & (df.strategy != "A")]
    sel = d[d.selected == 1]
    tab = (sel.groupby(["density_per_ha", "width_m"]).strategy.value_counts(normalize=True)
           .mul(100).unstack(fill_value=0).round(0))
    with open(os.path.join(out, "tab_decision.tex"), "w", encoding="utf8") as fh:
        fh.write(tab.to_latex(escape=True))
    return tab


def mixed_models(df, out):
    """time_min and missed_pct ~ strategy * density * width (* sc) + (1 | instance)."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels missing: skipping mixed models", file=sys.stderr)
        return
    d = df[(df.density_per_ha > 0) & (df.strategy != "A") & df.feasible].copy()
    d["density"] = d.density_per_ha.astype(str)
    d["width"] = d.width_m.astype(str)
    lines = []
    for resp, form in [
        ("time_min", "time_min ~ C(strategy) * C(density) * C(width)"),
        ("missed_pct", "missed_pct ~ C(strategy) * C(sc) * C(width)"),
        ("overlap_pct", "overlap_pct ~ C(strategy) * C(sc) * C(width)"),
    ]:
        dd = d if resp != "time_min" else d[d.sc == 1]
        # random intercept per paddock LAYOUT (the same layout is reused across widths and SC states)
        layout = dd["instance_id"].str.replace(r"_w\d+_", "_", regex=True)
        m = smf.mixedlm(form, dd, groups=layout).fit(reml=True)
        lines += [f"===== {resp} =====", str(m.summary()), ""]
    with open(os.path.join(out, "models.txt"), "w", encoding="utf8") as fh:
        fh.write("\n".join(lines))


def friedman(df, out):
    """Non-parametric check: strategies B1-B3 compared on time within each instance-width block (SC on)."""
    from scipy.stats import friedmanchisquare
    d = df[(df.density_per_ha > 0) & (df.sc == 1) & df.strategy.isin(["B1", "B2", "B3"]) & df.feasible]
    w = d.pivot_table(index=["instance_id", "width_m"], columns="strategy", values="time_min").dropna()
    if len(w) > 5:
        s, p = friedmanchisquare(*[w[c] for c in w.columns])
        with open(os.path.join(out, "models.txt"), "a", encoding="utf8") as fh:
            fh.write(f"\nFriedman time_min B1/B2/B3: chi2={s:.2f}, p={p:.3g}, n_blocks={len(w)}\n")


def figures(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    col = {"A": "#7f8c8d", "B1": "#2f6db5", "B2": "#8e5cb8", "B3": "#e07b39"}
    d = df[(df.density_per_ha > 0) & (df.sc == 1) & (df.feasible | (df.strategy == "A"))]

    # Fig: time and missed area vs density, by strategy, one panel per width
    widths = sorted(d.width_m.unique())
    fig, axs = plt.subplots(2, len(widths), figsize=(174 * MM, 100 * MM), sharex=True)
    axs = np.atleast_2d(axs)
    for j, w in enumerate(widths):
        for s in STRATS:
            g = d[(d.width_m == w) & (d.strategy == s)].groupby("density_per_ha")
            if not len(g):
                continue
            for i, c in enumerate(["time_min", "missed_pct"]):
                m, se = g[c].mean(), g[c].sem()
                axs[i, j].errorbar(m.index, m.values, yerr=1.96 * se.values, marker="o", ms=3, lw=1, capsize=2,
                                   color=col[s], label=s)
        axs[0, j].set_title(f"w = {w:g} m")
        axs[1, j].set_xlabel("Obstacle density (ha$^{-1}$)")
    axs[0, 0].set_ylabel("Field time (min)")
    axs[1, 0].set_ylabel("Missed area (%)")
    axs[0, -1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_time_missed.pdf"))
    plt.close(fig)

    # Fig: overlap with vs without SC
    e = df[(df.density_per_ha > 0) & (df.strategy != "A") & df.feasible]
    g = e.groupby(["strategy", "sc"]).overlap_pct.mean().unstack()
    fig, ax = plt.subplots(figsize=(84 * MM, 60 * MM))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.get(0, 0), 0.4, label="SC off", color="#bbbbbb")
    ax.bar(x + 0.2, g.get(1, 0), 0.4, label="SC on", color=[col[s] for s in g.index])
    ax.set_xticks(x, g.index)
    ax.set_ylabel("Overlap (% of target)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_overlap_sc.pdf"))
    plt.close(fig)


def numbers(df, meta, n_err, out):
    """LaTeX macros for numbers quoted in the text (so text and data never diverge)."""
    r = rel_to_A(df[(df.density_per_ha > 0) & (df.sc == 1)]); r = r[r.feasible]
    nums = {
        "NInstances": df.instance_id.nunique(),
        "NRuns": len(df),
        "NErrors": n_err,
        **{f"Feas{dict(B1='BOne', B2='BTwo', B3='BThree')[s]}W{dict([(12, 'Twelve'), (24, 'TwentyFour'), (36, 'ThirtySix')])[w]}Pct": 100 * df[(df.density_per_ha > 0) & (df.sc == 1) & (df.strategy == s) & (df.width_m == w)].feasible.mean()
           for s, w in [(s, w) for s in ("B1", "B2", "B3") for w in (12, 24, 36)]},
        "BThreeTimeVsAPct": r[r.strategy == "B3"].d_time_min_pct.mean(),
        "BOneTimeVsAPct": r[r.strategy == "B1"].d_time_min_pct.mean(),
        "BTwoTimeVsAPct": r[r.strategy == "B2"].d_time_min_pct.mean(),
        "BTwoDecompFailPct": 100 * df[df.strategy == "B2"].decomp_failed.fillna(0).astype(float).mean(),
        "BThreeUnplacedPct": 100 * (df[df.strategy == "B3"].n_unplaced_detours.fillna(0) > 0).mean(),
    }
    with open(os.path.join(out, "numbers.tex"), "w", encoding="utf8") as fh:
        fh.write("% generated by analysis/analyse_benchmark.py; do not edit\n")
        for k, v in nums.items():
            v = f"{v:.1f}" if isinstance(v, float) else str(v)
            v = r"\ensuremath{-}" + v[1:] if v.startswith("-") else v
            fh.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
    return nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "generated"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df_all, meta, n_err = load(a.results)
    df = main_set(df_all)
    print(f"{df.instance_id.nunique()} instances, {len(df)} rows ({n_err} rows with planning errors, counted as infeasible)")
    print(feasibility_table(df, a.out))
    print(summary_table(df, a.out))
    sc_table(df, a.out)
    print(decision_map(df, a.out))
    mixed_models(df, a.out)
    friedman(df, a.out)
    figures(df, a.out)
    print(numbers(df, meta, n_err, a.out))
    print(extra_numbers(df_all, a.out))


# ---------------------------------------------------------------------------
# Additional macros quoted in the Results text (appended to numbers.tex)
# ---------------------------------------------------------------------------
_S = {"A": "A", "B1": "BOne", "B2": "BTwo", "B3": "BThree"}
_W = {12: "Twelve", 24: "TwentyFour", 36: "ThirtySix"}


def extra_numbers(df_all, out):
    df = main_set(df_all)
    o = df[df.density_per_ha > 0]
    on = o[o.sc == 1]
    f = on[on.feasible]
    nums = {}
    a = on[on.strategy == "A"].set_index(["instance_id", "width_m"]).time_min.rename("tA")
    ff = f.join(a, on=["instance_id", "width_m"])
    ff = ff.assign(dt=100 * (ff.time_min - ff.tA) / ff.tA)
    for (s, w), v in ff.groupby(["strategy", "width_m"]).dt.mean().items():
        nums[f"Time{_S[s]}W{_W[int(w)]}Pct"] = v
    for (s, w), v in f.groupby(["strategy", "width_m"]).missed_pct.mean().items():
        nums[f"Missed{_S[s]}W{_W[int(w)]}Pct"] = v
    e = o[o.feasible | (o.strategy == "A")]
    g = e.groupby(["strategy", "sc"]).agg(ov=("overlap_pct", "mean"), ap=("applied_ha", "mean")).unstack()
    for s in g.index:
        nums[f"Overlap{_S[s]}Off"] = g.loc[s, ("ov", 0)]
        nums[f"Overlap{_S[s]}On"] = g.loc[s, ("ov", 1)]
        nums[f"Saved{_S[s]}Pct"] = 100 * (1 - g.loc[s, ("ap", 1)] / g.loc[s, ("ap", 0)])
    b = on[on.strategy != "A"]
    grp = b.groupby(["instance_id", "width_m"]).feasible.any()
    nums["NPlansNoFeasible"] = int((~grp).sum())
    nums["NPlansObstacle"] = int(len(grp))
    sel = on[on.selected == 1].strategy.value_counts()
    for s in ("B1", "B2", "B3"):
        nums[f"Selected{_S[s]}"] = int(sel.get(s, 0))
    # gate subset: random vs road gate, same instances (paired), relative time change
    if (df_all.gate_type != "road").any():
        key = ["shape", "obstacle_type", "pattern", "width_m", "density_per_ha"]
        ids = df_all[df_all.gate_type != "road"][key].drop_duplicates()
        s2 = df_all[df_all.sc == 1].merge(ids, on=key)
        p = s2.pivot_table(index=key + ["seed", "strategy"], columns="gate_type", values="time_min").dropna()
        p["d"] = 100 * (p["random"] - p["road"]) / p["road"]
        for s, v in p.groupby("strategy").d.mean().items():
            nums[f"Gate{_S[s]}Pct"] = v
        nums["GateMinPct"], nums["GateMaxPct"] = p.d.min(), p.d.max()
    t = on.assign(tplan=on[["t_hg_s", "t_sg_s", "t_rp_s", "t_pp_s"]].fillna(0).sum(axis=1))
    nums["PlanTimeMedianS"] = t.tplan.median()
    nums["DetourTimeMedianS"] = on[on.strategy == "B3"].t_detour_s.median()
    nums["SimTimeMedianS"] = on.t_sim_s.median()
    with open(os.path.join(out, "numbers.tex"), "a", encoding="utf8") as fh:
        for k, v in nums.items():
            v = f"{v:.1f}" if isinstance(v, float) else str(v)
            v = r"\ensuremath{-}" + v[1:] if v.startswith("-") else v
            fh.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
    return nums


if __name__ == "__main__":
    main()
