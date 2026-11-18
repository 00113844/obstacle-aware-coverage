"""E4 plan-view figures: base-lane failures on real plans (reads <out>/geojson, recomputes lanes locally).

    python ctf_figs.py --geo ../results/ctf_v1/geojson --out ../results/ctf_v1/figs
Writes
  cusp_w36_<id>.png       B3 lanes at 3:1 with an outer-fillet cusp vs B3c / B3n at the same spot
  crossing_w<w>_<id>.png  B3 adjacent-pass lane crossing vs B3n at the same spot
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import shape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_geom import Track  # noqa: E402
import ctf_geom as cg  # noqa: E402

R_CTF = max(cg.CTF_DEFAULTS["r_min_seeder"], cg.CTF_DEFAULTS["r_min_harvester"])


def load(path):
    g = json.load(open(path))
    feats = g["features"]
    w = g["properties"]["width"]
    tracks = {}
    for f in sorted((f for f in feats if f["properties"]["layer"] == "path"), key=lambda f: f["properties"]["seq"]):
        p = f["properties"]
        tracks.setdefault(p["strategy"], []).append((shape(f["geometry"]), p["kind"]))
    obs = [dict(poly=shape(f["geometry"]), type=f["properties"]["kind"]) for f in feats if f["properties"]["layer"] == "obstacle"]
    field = next(shape(f["geometry"]) for f in feats if f["properties"]["layer"] == "paddock")
    return w, field, obs, {k: Track(v) for k, v in tracks.items()}


def lanes(track, w):
    return cg.make_lanes(track.pieces("swath"), w, 12.0, "shared", R_CTF)


def first_crossing(L):
    lines = [l["line"] for l in L]
    tree = cg.shapely.STRtree(lines)
    a, b = tree.query(lines, predicate="intersects")
    for i, j in zip(a, b):
        if i < j:
            x = lines[i].intersection(lines[j])
            return np.asarray(x.centroid.coords[0])
    return None


def panel(ax, L, obs, centre, half, title, track=None):
    for o in obs:
        ax.fill(*o["poly"].exterior.xy, color="#6b4423" if o["type"] == "tree" else "#7f7f7f", zorder=3)
    for l in L:
        col = "#2a9d4b" if l["valid"] else "#d62728"
        ax.plot(*l["line"].xy, color=col, lw=1.0, zorder=2)
    if track is not None:
        for g in track.pieces("swath"):
            ax.plot(*g.xy, color="k", lw=0.5, ls="--", zorder=1)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--half", type=float, default=110)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    done = set()
    for path in sorted(glob.glob(os.path.join(a.geo, "*d1_w*_road_s0.geojson")) + sorted(glob.glob(os.path.join(a.geo, "*d2_w*_road_s0.geojson")))):
        iid = os.path.basename(path)[:-8]
        w, field, obs, T = load(path)
        if not {"B3", "B3n"} <= set(T):
            continue
        L3 = lanes(T["B3"], w)
        if w == 36 and "cusp" not in done:
            bad = [l for l in L3 if l["cusp"]]
            if bad:
                c = np.asarray(bad[0]["worst_xy"])
                fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
                panel(axs[0], L3, obs, c, a.half, "B3 (rho = 6 m): cusped / too-tight lanes in red", T["B3"])
                panel(axs[1], lanes(T["B3c"], w), obs, c, a.half, "B3c (rho = 20 m, lane envelope)", T["B3c"])
                panel(axs[2], lanes(T["B3n"], w), obs, c, a.half, "B3n (nested, s_min = 6 m)", T["B3n"])
                fig.suptitle(f"{iid}: 3:1 base lanes (12 m, shared layout); dashed = sprayer passes", fontsize=11)
                plt.tight_layout()
                fig.savefig(os.path.join(a.out, f"cusp_w36_{iid}.png"), dpi=110)
                plt.close(fig)
                done.add("cusp")
        key = f"cross{w}"
        if key not in done and w in (24, 36):
            c = first_crossing(L3)
            if c is not None:
                fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.8))
                panel(axs[0], L3, obs, c, a.half, "B3: lanes of adjacent passes cross", T["B3"])
                panel(axs[1], lanes(T["B3n"], w), obs, c, a.half, "B3n: nested passes, no crossing", T["B3n"])
                fig.suptitle(f"{iid}: {w // 12}:1 base lanes (12 m, shared layout)", fontsize=11)
                plt.tight_layout()
                fig.savefig(os.path.join(a.out, f"crossing_w{w}_{iid}.png"), dpi=110)
                plt.close(fig)
                done.add(key)
    print("figures:", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
