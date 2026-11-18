"""2D top-view figures for the appendix: what goes wrong with each obstacle strategy.

    python fig_topview.py GEOJSON_DIR STEM PANELS --out FILE.pdf [--runs RUNS.csv] [--sc 1] [--ncols 2]

GEOJSON_DIR holds STEM.geojson (paths, obstacles, zone, gate, paddock) and STEM_coverage.npz
(application counts per 1 m cell, keys "<strategy>_sc<0|1>", plus "extent").
PANELS is a comma list of strategy labels in the geojson (e.g. A,B1,B2,B3 or N_sort,N_rp,N_dec,B3).

Each panel shows the coverage raster (missed target in red, overlap in blue), working passes, turns and
headland passes, and the path sections that violate the clearance (Eq. clearance: closer than w/2 to an
obstacle) as thick red lines.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from shapely.geometry import shape, Point  # noqa: E402
from shapely.ops import unary_union  # noqa: E402
from shapely import contains_xy  # noqa: E402

from figures import INK, INK2, MUTED, MM, W2, panel_label  # noqa: E402

TITLE = {
    "A": "A  open-paddock reference (ignores obstacles)",
    "B1": "B1  holes with headland rings",
    "B2": "B2  Boustrophedon decomposition",
    "B3": "B3  drive-around (this work)",
    "B4": "B4  straddle low outcrops (this work)",
    "N_sort": "Native F2C: holes, boustrophedon order",
    "N_rp": "Native F2C: holes, OR-Tools route",
    "N_dec": "Native F2C: Boustrophedon decomp., OR-Tools",
    "N_trap": "Native F2C: trapezoidal decomp., OR-Tools",
}
MISSED, ONE, TWO, THREE = "#d03b3b", "#e3eefb", "#86b6ef", "#2a78d6"
TREE, ROCK, ZONE = "#2f6b3a", "#8c7b6b", "#ece9e2"
VIOL = "#b0171f"


def load(gdir, stem):
    g = json.load(open(os.path.join(gdir, stem + ".geojson")))
    w = float(g.get("properties", {}).get("width", 0) or 0)
    feats = g["features"]
    npz = os.path.join(gdir, stem + "_coverage.npz")
    cov = np.load(npz) if os.path.exists(npz) else None
    return w, feats, cov


def kind(f):
    p = f["properties"]
    return p.get("kind") or p.get("layer")


def panel(ax, w, feats, cov, strat, sc, info=""):
    pad = [shape(f["geometry"]) for f in feats if kind(f) == "paddock"][0]
    zone = unary_union([shape(f["geometry"]) for f in feats if kind(f) == "zone"] or [Point(0, 0).buffer(0)])
    trees = [shape(f["geometry"]) for f in feats if kind(f) == "tree"]
    rocks = [shape(f["geometry"]) for f in feats if kind(f) == "outcrop"]
    obst = unary_union(trees + rocks) if trees or rocks else None

    # coverage raster: target = paddock minus safety zone
    key = f"{strat}_sc{sc}"
    if cov is not None and key in cov.files:
        c = cov[key].astype(int)
        x0, x1, y0, y1 = cov["extent"]
        ny, nx = c.shape
        xs = np.linspace(x0 + 0.5 * (x1 - x0) / nx, x1 - 0.5 * (x1 - x0) / nx, nx)
        ys = np.linspace(y0 + 0.5 * (y1 - y0) / ny, y1 - 0.5 * (y1 - y0) / ny, ny)
        X, Y = np.meshgrid(xs, ys)
        target = contains_xy(pad, X, Y) & ~contains_xy(zone, X, Y)
        cls = np.full(c.shape, np.nan)
        cls[target] = np.clip(c[target], 0, 3)
        cmap = ListedColormap([MISSED, ONE, TWO, THREE])
        ax.imshow(cls, origin="lower", extent=(x0, x1, y0, y1), cmap=cmap, vmin=-0.5, vmax=3.5,
                  interpolation="nearest", zorder=1)
    for z in getattr(zone, "geoms", [zone]):
        if not z.is_empty:
            ax.fill(*z.exterior.xy, color=ZONE, lw=0, zorder=2)
    for t in trees:
        ax.fill(*t.exterior.xy, color=TREE, lw=0, zorder=5)
    for r in rocks:
        ax.fill(*r.exterior.xy, color=ROCK, lw=0, zorder=5)
    ax.plot(*pad.exterior.xy, color=INK, lw=0.9, zorder=6)

    lines = {"swath": [], "turn": [], "headland_pass": []}
    for f in feats:
        if f["properties"].get("strategy") == strat and kind(f) in lines:
            lines[kind(f)].append(shape(f["geometry"]))
    for ln in lines["headland_pass"]:
        ax.plot(*ln.xy, color=MUTED, lw=0.35, zorder=3)
    for ln in lines["swath"]:
        ax.plot(*ln.xy, color=INK2, lw=0.45, zorder=4)
    for ln in lines["turn"]:
        ax.plot(*ln.xy, color="#6f6c66", lw=0.45, ls=(0, (2, 1)), zorder=4)
    segs = []
    if obst is not None and w > 0:
        bad = obst.buffer(w / 2 - 1e-6)
        for k in ("swath", "turn"):
            for ln in lines[k]:
                seg = ln.intersection(bad)
                segs += [s for s in getattr(seg, "geoms", [seg]) if not s.is_empty and s.length > 0.5]
    for s in segs:
        ax.plot(*s.xy, color=VIOL, lw=2.0, solid_capstyle="round", zorder=7)
        if strat != "A" and len(segs) <= 6:   # call out isolated violations; many speak for themselves
            c = s.interpolate(0.5, normalized=True)
            ax.plot(c.x, c.y, marker="o", ms=13, mfc="none", mec=VIOL, mew=1.1, zorder=7)
    n_viol = len(segs)
    for f in feats:
        if kind(f) == "gate":
            p = shape(f["geometry"])
            ax.plot(p.x, p.y, marker="v", ms=6, color=INK, mec="white", mew=0.8, zorder=8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(TITLE.get(strat, strat) + (f"\n{info}" if info else ""), fontsize=7, color=INK, loc="left")
    return n_viol


def legend(fig):
    h = [Patch(color=MISSED, label="missed target"), Patch(color=ONE, label="treated once"),
         Patch(color=TWO, label="twice"), Patch(color=THREE, label="three times or more"),
         Patch(color=ZONE, label="safety zone"), Patch(color=TREE, label="tree"), Patch(color=ROCK, label="rock outcrop"),
         Line2D([], [], color=INK2, lw=0.8, label="working pass"),
         Line2D([], [], color="#6f6c66", lw=0.8, ls=(0, (2, 1)), label="turn"),
         Line2D([], [], color=VIOL, lw=2.0, label="clearance violated"),
         Line2D([], [], ls="", marker="v", color=INK, label="gate")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.6, columnspacing=1.2)


def info_from_runs(runs, stem, strat, sc):
    if runs is None:
        return ""
    inst = stem
    r = runs[(runs.instance_id == inst) & (runs.strategy == strat) & (runs.sc == sc)]
    if r.empty:
        return ""
    r = r.iloc[0]
    feas = "feasible" if str(r.feasible).lower() in ("true", "1") else "infeasible"
    s = f"{feas}; {r.time_min:.0f} min; missed {r.missed_pct:.1f} %; overlap {r.overlap_pct:.1f} %"
    oc = r.get("over_crop_m", 0)
    if pd.notna(oc) and oc >= 50:
        s += f"; {oc / 1000:.1f} km over crop"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gdir"); ap.add_argument("stem"); ap.add_argument("panels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--sc", type=int, default=1)
    ap.add_argument("--ncols", type=int, default=2)
    a = ap.parse_args()
    w, feats, cov = load(a.gdir, a.stem)
    runs = pd.read_csv(a.runs) if a.runs else None
    P = a.panels.split(",")
    nr = int(np.ceil(len(P) / a.ncols))
    fig, axs = plt.subplots(nr, a.ncols, figsize=(W2, nr * 72 * MM + 14 * MM))
    axs = np.atleast_1d(axs).ravel()
    for i, (ax, s) in enumerate(zip(axs, P)):
        nv = panel(ax, w, feats, cov, s, a.sc, info_from_runs(runs, a.stem, s, a.sc))
        panel_label(ax, "abcdefgh"[i])
        print(s, "violating sections:", nv)
    for ax in axs[len(P):]:
        ax.axis("off")
    legend(fig)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08, wspace=0.05, hspace=0.18)
    fig.savefig(a.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
