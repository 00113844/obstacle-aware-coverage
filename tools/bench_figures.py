"""R6 figures from benchmark geometry dumps (no Fields2Cover needed).

    python bench_figures.py RESULTS/geojson/<instance_id>.geojson --out FIGDIR
writes fig_paths.pdf, fig_coverage.pdf, fig_detour_sc.pdf (174 mm wide, sans-serif 8 pt,
fonts embedded as TrueType/Type 42, RGB, coverage maps rasterised inside the vector PDF).
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import shape, LineString
from shapely.plotting import plot_polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_geom import Track, simulate_coverage  # noqa: E402

MM = 1 / 25.4
FULL_W = 174 * MM
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.5, "savefig.dpi": 600,
})
C_SW, C_TN, C_OBS, C_ZONE = "#2166ac", "#e08214", "#4d4d4d", "#bababa"
CMAP = ListedColormap(["#d73027", "#a6d96a", "#fdae61"])
STRAT = ["A", "B1", "B2", "B3"]
LABEL = {"A": "A (open-paddock plan)", "B1": "B1 holes", "B2": "B2 decomposition", "B3": "B3 drive-around"}


def load(path):
    gj = json.load(open(path))
    d = {"width": gj.get("properties", {}).get("width"), "obstacles": [], "paths": {}, "extra": {}}
    for f in gj["features"]:
        p, g = f["properties"], shape(f["geometry"])
        if p["layer"] == "paddock":
            d["field"] = g
        elif p["layer"] == "gate":
            d["gate"] = (g.x, g.y)
        elif p["layer"] == "obstacle":
            d["obstacles"].append((p["kind"], g))
        elif p["layer"] == "zone":
            d["zones"] = g
        elif p["layer"] == "path":
            d["paths"].setdefault(p["strategy"], []).append((p["seq"], g, p["kind"]))
        elif p["layer"] == "headland_pass":
            d["extra"].setdefault(p["strategy"], []).append(g)
    d["tracks"] = {s: Track([(g, k) for _, g, k in sorted(v, key=lambda t: t[0])]) for s, v in d["paths"].items()}
    npz = path.replace(".geojson", "_coverage.npz")
    d["cov"] = dict(np.load(npz)) if os.path.exists(npz) else {}
    return d


def base_ax(ax, d, zones=True):
    plot_polygon(d["field"], ax=ax, add_points=False, facecolor="#f7f7f7", edgecolor="k", linewidth=0.6)
    if zones and d.get("zones") is not None:
        plot_polygon(d["zones"], ax=ax, add_points=False, facecolor="none", edgecolor=C_ZONE, linewidth=0.4,
                     linestyle="--")
    for kind, g in d["obstacles"]:
        plot_polygon(g, ax=ax, add_points=False, facecolor="#1b7837" if kind == "tree" else "#8c6d31",
                     edgecolor="none", linewidth=0, zorder=4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_track(ax, tr, lw=0.45):
    segs = [np.asarray(g.coords) for g, _ in tr.segs]
    cols = [C_SW if k == "swath" else C_TN for _, k in tr.segs]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, zorder=5))


def fig_paths(d, out):
    strats = [s for s in STRAT if s in d["tracks"]]
    n = len(strats)
    cols = 2 if n > 2 else n
    rows = int(np.ceil(n / cols))
    fw, fh = d["field"].bounds[2] - d["field"].bounds[0], d["field"].bounds[3] - d["field"].bounds[1]
    fig, axs = plt.subplots(rows, cols, figsize=(FULL_W, FULL_W / cols * fh / fw * rows + 0.35), squeeze=False)
    for ax, s in zip(axs.ravel(), strats):
        base_ax(ax, d)
        draw_track(ax, d["tracks"][s])
        ax.plot(*d["gate"], marker="*", ms=7, mfc="gold", mec="k", mew=0.4)
        ax.set_title(f"({'abcd'[strats.index(s)]}) {LABEL[s]}", loc="left")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=C_SW, lw=1, label="working"), Line2D([], [], color=C_TN, lw=1, label="turn / headland"),
                        Patch(color="#1b7837", label="tree"), Patch(color="#8c6d31", label="rock outcrop"),
                        Line2D([], [], color=C_ZONE, ls="--", lw=0.8, label="safety zone"),
                        Line2D([], [], marker="*", ls="", mfc="gold", mec="k", label="gate")],
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out); plt.close(fig)


def cov_map(ax, d, key, field_mask=True):
    cnt = d["cov"][key].astype(float)
    ext = d["cov"]["extent"]
    sim_mask = _target_mask(d, cnt.shape, ext)
    im = ax.imshow(np.ma.masked_where(~sim_mask, np.minimum(cnt, 2)), origin="lower", extent=ext, cmap=CMAP,
                   vmin=-0.5, vmax=2.5, interpolation="nearest", rasterized=True, zorder=2)
    return im


_MASKS = {}


def _target_mask(d, shp, ext):
    import shapely
    key = (id(d), shp)
    if key not in _MASKS:
        ny, nx = shp
        xs = np.linspace(ext[0], ext[1], nx, endpoint=False) + (ext[1] - ext[0]) / nx / 2
        ys = np.linspace(ext[2], ext[3], ny, endpoint=False) + (ext[3] - ext[2]) / ny / 2
        X, Y = np.meshgrid(xs, ys)
        m = shapely.contains_xy(d["field"], X, Y)
        if d.get("zones") is not None:
            m &= ~shapely.contains_xy(d["zones"], X, Y)
        _MASKS[key] = m
    return _MASKS[key]


def fig_coverage(d, out):
    strats = [s for s in STRAT if f"{s}_sc0" in d["cov"] and (s != "A" or len(d["cov"]) <= 3)]
    n = len(strats)
    fw, fh = d["field"].bounds[2] - d["field"].bounds[0], d["field"].bounds[3] - d["field"].bounds[1]
    fig, axs = plt.subplots(2, n, figsize=(FULL_W, FULL_W / n * fh / fw * 2 + 0.45), squeeze=False)
    for j, s in enumerate(strats):
        for i, sc in enumerate((0, 1)):
            ax = axs[i, j]
            base_ax(ax, d, zones=False)
            cov_map(ax, d, f"{s}_sc{sc}")
            cnt, m = d["cov"][f"{s}_sc{sc}"], _target_mask(d, d["cov"][f"{s}_sc{sc}"].shape, d["cov"]["extent"])
            ax.set_title(f"{s} · {'SC on' if sc else 'SC off'}\nmissed {100 * (m & (cnt == 0)).sum() / m.sum():.1f}% · "
                         f"overlap {100 * (m & (cnt >= 2)).sum() / m.sum():.1f}%", loc="left")
    fig.legend(handles=[Patch(color=CMAP(0), label="missed (0)"), Patch(color=CMAP(1), label="once (1)"),
                        Patch(color=CMAP(2), label="overlap (≥2)")], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out); plt.close(fig)


def fig_detour(d, out, n_sections=None, step=2.0):
    """B3 detour zoom: coverage without SC (left) and section ON/OFF footprints with SC (right)."""
    tr, w = d["tracks"]["B3"], float(d["width"])
    n_sections = n_sections or {12: 6, 24: 12, 36: 18}.get(int(w), 6)
    # obstacle whose safety zone is crossed by the most working pieces
    best = max(d["obstacles"], key=lambda o: sum(g.distance(o[1]) < w for g in tr.pieces("swath")))[1]
    win = best.buffer(2.2 * w).envelope
    sims = {sc: simulate_coverage(tr, d["field"], d.get("zones"), w, n_sections, 1.0, section_control=sc,
                                  extra_passes=d["extra"].get("B3", []), record_window=win if sc else None)
            for sc in (False, True)}
    x0, y0, x1, y1 = win.bounds
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, FULL_W / 2 * (y1 - y0) / (x1 - x0) + 0.4))
    for ax, sc in zip(axs, (False, True)):
        base_ax(ax, d)
        r = sims[sc]
        ax.imshow(np.ma.masked_where(~r["target"], np.minimum(r["count"], 2)), origin="lower",
                  extent=r["extent"], cmap=CMAP, vmin=-0.5, vmax=2.5, interpolation="nearest",
                  alpha=0.35 if sc else 1.0, rasterized=True, zorder=2)
        if sc:  # boom ribbon of the pass that switches sections most often in the window
            by_pass = {}
            for pid, x, y, dx, dy, on in r["states"]:
                lst = by_pass.setdefault(pid, [])
                if not lst or np.hypot(x - lst[-1][0], y - lst[-1][1]) >= step:
                    lst.append((x, y, dx, dy, on))
            switches = {pid: sum(int((a[4] != b[4]).sum()) for a, b in zip(s[:-1], s[1:])) for pid, s in by_pass.items()}
            pid = max(switches, key=switches.get)
            on_q, off_q, edges = [], [], np.linspace(-w / 2, w / 2, n_sections + 1)
            s = by_pass[pid]
            for (x, y, dx, dy, on), (x2, y2, dx2, dy2, _) in zip(s[:-1], s[1:]):
                c, c2 = np.array([x, y]), np.array([x2, y2])
                n1, n2 = np.array([-dy, dx]), np.array([-dy2, dx2])
                for j in range(n_sections):
                    a, b = edges[j], edges[j + 1]
                    (on_q if on[j] else off_q).append([c + a * n1, c + b * n1, c2 + b * n2, c2 + a * n2])
            ax.add_collection(PolyCollection(off_q, facecolors="#bdbdbd", edgecolors="none", rasterized=True, zorder=3))
            ax.add_collection(PolyCollection(on_q, facecolors="#4393c3", edgecolors="none", rasterized=True, zorder=3))
            # section dividers
            for j in range(1, n_sections):
                pts = np.array([(x - edges[j] * dy, y + edges[j] * dx) for x, y, dx, dy, _ in s])
                ax.plot(pts[:, 0], pts[:, 1], color="white", lw=0.25, zorder=3.5)
        draw_track(ax, tr, lw=0.8)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_title("(a) SC off: coverage count" if not sc else f"(b) SC on: coverage + {n_sections}-section ON/OFF of one pass", loc="left")
    fig.legend(handles=[Patch(color="#4393c3", label="section ON (one pass)"), Patch(color="#bdbdbd", label="section OFF"),
                        Patch(color=CMAP(0), label="missed"), Patch(color=CMAP(1), label="once"),
                        Patch(color=CMAP(2), label="overlap"), Line2D([], [], color=C_SW, label="B3 path")],
               loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geojson")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load(a.geojson)
    fig_paths(d, os.path.join(a.out, "fig_paths.pdf"))
    fig_coverage(d, os.path.join(a.out, "fig_coverage.pdf"))
    if "B3" in d["tracks"]:
        fig_detour(d, os.path.join(a.out, "fig_detour_sc.pdf"))
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
