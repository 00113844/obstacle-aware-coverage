"""Local tests for ctf_geom (no Fields2Cover): lane layouts, cusps, B3c / B3n, crossings, timing."""
import os, sys, time
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from bench_geom import (Track, make_drive_around_poly, make_field, make_obstacles, stable_seed, simulate_coverage)
from ctf_geom import *
from test_bench_geom import mock_track  # noqa: E402  (runs the bench_geom tests too)

OUT = os.environ.get("CTF_TEST_OUT", os.path.dirname(os.path.abspath(__file__)))
CD = CTF_DEFAULTS
r_ctf = max(CD["r_min_seeder"], CD["r_min_harvester"])

# 1. layouts
assert lane_offsets(1, 12, "shared") == [0] and lane_offsets(2, 12, "shared") == [0, 12]
assert lane_offsets(3, 12, "shared") == [-12, 0, 12] and lane_offsets(2, 12, "centred") == [-6, 6]
assert d_max(2, 12, "shared") == 12 and d_max(2, 12, "centred") == 6 and d_max(3, 12, "shared") == 12
p = ctf_params(24, 12, "shared", r_ctf); assert p["h_env"] == 18 and p["rho"] == 20, p
p = ctf_params(24, 12, "centred", r_ctf); assert p["h_env"] == 12 and p["rho"] == 14, p
print("layouts ok")

# 2. one tree in a rectangle, all widths/layouts
F = box(0, 0, 600, 400)
tree = [{"type": "tree", "poly": Point(300, 203).buffer(6, quad_segs=32), "r": 6.0}]
rows = []
for w_s in (12, 24, 36):
    main = F.buffer(-2 * w_s, join_style=2)
    trA = mock_track(main, w_s)
    for layout in ("shared", "centred"):
        if layout == "centred" and w_s != 24:
            continue
        cp = ctf_params(w_s, 12, layout, r_ctf)
        trB3, _ = make_drive_around_poly(trA, tree, 3.0, w_s, 6.0, F.buffer(-w_s / 2))
        trC, uc = make_b3c(trA, tree, 3.0, w_s, cp["rho"], cp["h_env"], F.buffer(-w_s / 2))
        t0 = time.time()
        trN, un, st = make_b3n(trA, tree, 3.0, w_s, 12, cp["rho"], cp["h_env"], r_ctf, cp["d_max"], CD["s_min"],
                               F.buffer(-w_s / 2))
        tn = time.time() - t0
        for name, tr in (("B3", trB3), ("B3c", trC), ("B3n", trN)):
            L = make_lanes(tr.pieces("swath"), w_s, 12, layout, r_ctf)
            lines = [l["line"] for l in L]
            det = [l for l in L if l["detoured"]]
            sp = spacing(lines, 12)
            rows.append((w_s, layout, name, len(L), sum(l["cusp"] for l in L), sum(not l["valid"] for l in det), len(det),
                         crossings(lines), round(float(sp.min()), 2), round(lane_clearance(lines, tree, 6.0), 2),
                         st if name == "B3n" else ""))
        print(f"  w={w_s} {layout}: B3n {tn:.2f}s {st}")
print("w layout strat n_lanes cusps invalid_detoured/detoured crossings spacing_min tool_clear")
for r in rows:
    print(*r)
by = {(r[0], r[1], r[2]): r for r in rows}
assert by[(36, "shared", "B3")][4] > 0, "B3 at 3:1 should have a cusp (rho=6, d=12)"
assert by[(36, "shared", "B3c")][4] == 0 and by[(24, "shared", "B3c")][4] == 0
assert by[(36, "shared", "B3n")][7] == 0 and by[(24, "shared", "B3n")][7] == 0, "B3n lanes must not cross"
assert by[(36, "shared", "B3c")][9] >= -1e-6, "B3c lanes must clear the tree with the base tool"
print("single-tree checks ok")

# 3. realistic instance (timing + all metrics)
for shape, otype, w_s in (("L", "mixed", 24), ("rectangle", "mixed", 36), ("convex", "outcrop", 12)):
    rng = np.random.default_rng(stable_seed(shape, 0)); F2 = make_field(shape, 20, rng)
    obs = make_obstacles(F2.buffer(-(2 * 36 + 10)), 20, otype, "poisson",
                         np.random.default_rng(stable_seed(shape, 1.0, otype, "poisson", 0)))
    main = F2.buffer(-2 * w_s, join_style=2)
    trA = mock_track(main, w_s, ang=17.0)
    cp = ctf_params(w_s, 12, "shared", r_ctf)
    t0 = time.time(); trB3, u3 = make_drive_around_poly(trA, obs, 3.0, w_s, 6.0, F2.buffer(-w_s / 2)); t3 = time.time() - t0
    t0 = time.time(); trC, uc = make_b3c(trA, obs, 3.0, w_s, cp["rho"], cp["h_env"], F2.buffer(-w_s / 2)); tc = time.time() - t0
    t0 = time.time()
    trN, un, st = make_b3n(trA, obs, 3.0, w_s, 12, cp["rho"], cp["h_env"], r_ctf, cp["d_max"], CD["s_min"], F2.buffer(-w_s / 2))
    tn = time.time() - t0
    zones = unary_union([o["poly"].buffer(3.0) for o in obs])
    for name, tr, u in (("B3", trB3, u3), ("B3c", trC, uc), ("B3n", trN, un)):
        t0 = time.time()
        L = make_lanes(tr.pieces("swath"), w_s, 12, "shared", r_ctf)
        lines = [l["line"] for l in L]
        hl = headland_lanes(F2, 12, 2 * w_s)
        ncross = crossings(lines); sp = spacing(lines, 12)
        lt = Track([(l, "swath") for l in lines])
        sim = simulate_coverage(lt, F2, zones, 12, 6, 1.0, True, extra_passes=hl)
        WR = WheelRaster(F2.buffer(20).bounds, CD["traffic_res"])
        net = WR.paint(WR.blank(), lines + hl, CD["gauge"], CD["off_lane_tol"])
        off = WR.off_length(net, path_lines(tr), CD["gauge"])
        traf = WR.paint(WR.blank(), path_lines(tr) + lines + hl, CD["gauge"], CD["tyre_width"] / 2).sum() * CD["traffic_res"] ** 2 / 1e4
        tl = time.time() - t0
        det = [l for l in L if l["detoured"]]
        print(f"{shape:9s} {otype:7s} w={w_s} {name:3s}: unplaced {u} lanes {len(L)} detoured {len(det)} "
              f"invalid(det) {sum(not l['valid'] for l in det)} cusps {sum(l['cusp'] for l in L)} crossings {ncross} "
              f"spacing [{sp.min():.1f},{sp.max():.1f}] sown missed {sim['missed_pct']:.2f}% overlap {sim['overlap_pct']:.2f}% "
              f"off-lane {off:.0f} m trafficked {traf:.2f} ha | lanes+metrics {tl:.1f}s"
              + (f" | B3n {tn:.1f}s {st}" if name == "B3n" else f" | detour {t3 if name == 'B3' else tc:.1f}s"))

# 4. figure: B3 vs B3n at 3:1 around the single tree, lanes coloured by validity
fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
w_s = 36; main = F.buffer(-2 * w_s, join_style=2); trA = mock_track(main, w_s)
cp = ctf_params(w_s, 12, "shared", r_ctf)
trB3, _ = make_drive_around_poly(trA, tree, 3.0, w_s, 6.0, F.buffer(-w_s / 2))
trC, _ = make_b3c(trA, tree, 3.0, w_s, cp["rho"], cp["h_env"], F.buffer(-w_s / 2))
trN, _, st = make_b3n(trA, tree, 3.0, w_s, 12, cp["rho"], cp["h_env"], r_ctf, cp["d_max"], CD["s_min"], F.buffer(-w_s / 2))
for ax, (name, tr) in zip(axs, (("B3 (rho = 6 m)", trB3), ("B3c (rho = 20 m, lane envelope)", trC), ("B3n (nested, s_min = 6 m)", trN))):
    for l in make_lanes(tr.pieces("swath"), w_s, 12, "shared", r_ctf):
        ax.plot(*l["line"].xy, color="#2a9d4b" if l["valid"] else "#d62728", lw=0.9)
    for g in tr.pieces("swath"):
        ax.plot(*g.xy, color="k", lw=0.5, ls="--")
    ax.fill(*tree[0]["poly"].exterior.xy, color="#6b4423")
    ax.set_xlim(180, 420); ax.set_ylim(100, 330); ax.set_aspect("equal"); ax.set_title(name)
fig.suptitle("3:1 (w_s = 36 m, w_b = 12 m, shared layout): base lanes green = drivable, red = cusp / too tight; dashed = sprayer")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "ctf_test.png"), dpi=80)
print("figure ->", os.path.join(OUT, "ctf_test.png"))
