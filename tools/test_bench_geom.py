"""Local tests for bench_geom (no Fields2Cover needed)."""
import os, sys, time
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_geom import *
from nb_geom import detour_profile
from shapely.plotting import plot_polygon

def mock_track(main, w, ang=0.0):
    rot = lambda g, a: affinity.rotate(g, a, origin=main.centroid)
    m = rot(main, -ang); segs = []; y = m.bounds[1] + w / 2; k = 0
    while y < m.bounds[3]:
        gg = LineString([(m.bounds[0] - 10, y), (m.bounds[2] + 10, y)]).intersection(m)
        for s in getattr(gg, "geoms", [gg]):
            if s.is_empty or s.length < 5: continue
            c = list(rot(s, ang).coords); c = c[::-1] if k % 2 else c
            if segs: segs.append((LineString([segs[-1][0].coords[-1], c[0]]), "turn"))
            segs.append((LineString(c), "swath")); k += 1
        y += w
    return Track(segs)

# 1. disc case agrees with the analytic profile
R, rho, a = 12.0, 6.0, 4.0
ana, _ = detour_profile(a, R, rho, 0.25)
S = Point(50, -a).buffer(R, quad_segs=256)
res = detour_line_convex(LineString([(0, 0), (100, 0)]), [S], rho, lambda p: True, 0.25)[0]
la = LineString(ana).length + (100 - (ana[-1, 0] - ana[0, 0]))
print(f"disc: analytic {la:.2f} m vs general {res.length:.2f} m")

# 2. instances
print("shape areas:", {s: round(make_field(s, 20, np.random.default_rng(1)).area / 1e4, 2) for s in ["rectangle", "convex", "L"]})
fig, axs = plt.subplots(2, 3, figsize=(18, 11))
for ax, (shape, otype, pat, w) in zip(axs.ravel(), [("rectangle", "tree", "poisson", 12), ("convex", "outcrop", "thomas", 24),
                                                  ("L", "mixed", "poisson", 36), ("rectangle", "mixed", "thomas", 12),
                                                  ("convex", "tree", "thomas", 36), ("L", "outcrop", "poisson", 24)]):
    rng = np.random.default_rng(stable_seed(shape, 0)); F = make_field(shape, 20, rng)
    region = F.buffer(-(2 * 36 + 10))
    obs = make_obstacles(region, 40 if w == 12 else 20, otype, pat, np.random.default_rng(stable_seed(shape, otype, pat, 0)))
    main = F.buffer(-2 * w, join_style=2)
    tr = mock_track(main, w, ang=17.0)
    t0 = time.time()
    trB3, unpl = make_drive_around_poly(tr, obs, 3.0, w, 6.0, F.buffer(-w / 2))
    dt = time.time() - t0
    cl = clearance(trB3, obs, w)
    # regression (v1.0 bug): detours must never double back over an earlier detour on the same swath
    for g in trB3.pieces("swath"):
        assert g.is_simple, "self-crossing swath"
    work_cl = min(g.distance(o["poly"]) for g in trB3.pieces("swath") for o in obs) - w / 2
    assert unpl > 0 or work_cl >= -1e-6, f"placed detours touch an obstacle ({work_cl:.2f})"
    kmax = max(max_curvature(g) for g in trB3.pieces("swath"))
    zones = unary_union([o["poly"].buffer(3.0) for o in obs])
    t1 = time.time(); sim = simulate_coverage(trB3, F, zones, w, w // 2, 1.0, True); ts = time.time() - t1
    print(f"{shape:9s} {otype:7s} {pat:7s} w={w}: n={len(obs)} groups={len(obstacle_groups(obs, 3))} unplaced={unpl} "
          f"min clearance={min(cl):.2f} (of placed) kmax={kmax:.3f} detour {dt:.1f}s sim {ts:.1f}s missed {sim['missed_pct']:.1f}%")
    ax.imshow(np.ma.masked_where(~sim["target"], np.minimum(sim["count"], 2)), origin="lower", extent=sim["extent"],
              cmap=matplotlib.colors.ListedColormap(["#e74c3c", "#9fd18b", "#f39c12"]), vmin=0, vmax=2, interpolation="nearest")
    for o in obs: plot_polygon(o["poly"], ax=ax, add_points=False, facecolor="#2f6b2f", edgecolor="k", linewidth=0.3)
    for g, kd in trB3.segs: ax.plot(*g.xy, color="#2f6db5" if kd == "swath" else "#e07b39", lw=0.5)
    ax.plot(*road_gate(F), "*", ms=14, mfc="gold", mec="k"); ax.set_aspect("equal")
    ax.set_title(f"{shape}/{otype}/{pat}/w{w}: unplaced {unpl}", fontsize=9)
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "bench_geom_test.png", dpi=80, bbox_inches="tight")
