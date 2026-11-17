import sys, time, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_geom import *
from shapely.geometry import Polygon, LineString, Point, box
from shapely.plotting import plot_polygon

# --- 1. overlap sanity: two parallel passes 6 m wide, 4 m apart -> 2 m overlap strip
field = box(0, 0, 100, 40)
t = Track([(LineString([(10, 10), (90, 10)]), "swath"), (LineString([(90, 10), (90, 14)]), "turn"),
           (LineString([(90, 14), (10, 14)]), "swath")])
r0 = simulate_coverage(t, field, None, 6, 6, 1.0, section_control=False)
r1 = simulate_coverage(t, field, None, 6, 6, 1.0, section_control=True)
ov_area = (r0["target"] & (r0["count"] >= 2)).sum() * 1.0
print("overlap m2 noSC", ov_area, "expected", 2 * 80, "| SC overlap m2", (r1["target"] & (r1["count"] >= 2)).sum() * 1.0,
      "| applied", r0["applied_ha"], r1["applied_ha"])

# --- 2. drive-around on the tutorial paddock
paddock = [(0, 0), (410, -25), (465, 110), (450, 290), (300, 330), (215, 265), (95, 310), (-25, 175), (0, 0)]
trees = [(105, 70, 7.0), (190, 150, 9.0), (300, 65, 6.0), (355, 205, 8.0), (80, 205, 5.5), (260, 235, 6.5), (395, 60, 5.0)]
W, M, RMIN = 6.0, 3.0, 6.0
F = Polygon(paddock); main = F.buffer(-18, join_style=2)
segs = []; y = main.bounds[1] + 3; k = 0
while y < main.bounds[3]:
    g = LineString([(-100, y), (600, y)]).intersection(main)
    for s in (g.geoms if hasattr(g, "geoms") else [g]):
        if s.is_empty or s.length < 5: continue
        c = list(s.coords)
        if k % 2: c = c[::-1]
        if segs: segs.append((LineString([segs[-1][0].coords[-1], c[0]]), "turn"))
        segs.append((LineString(c), "swath")); k += 1
    y += 6
trA = Track(segs)
t0 = time.time()
trB3, issues = make_drive_around(trA, trees, M, W, RMIN, F.buffer(-W / 2))
print("B3 built in %.2fs, issues:" % (time.time() - t0), issues)
curv = max(max_curvature(g) for g in trB3.pieces("swath"))
print("max curvature %.3f vs 1/rmin %.3f" % (curv, 1 / RMIN))
print("min clearance (edge) per tree:", np.round(tree_clearance(trB3, trees, W), 2))
# continuity: consecutive segment endpoints coincide
gaps = [np.hypot(*(np.array(a.coords[-1]) - np.array(b.coords[0]))) for (a, _), (b, _) in zip(trB3.segs, trB3.segs[1:])]
print("max gap between segments", max(gaps))
print("swath length A %.0f  B3 %.0f  (+%.0f m detours)" % (trA.length('swath'), trB3.length('swath'), trB3.length('swath') - trA.length('swath')))
obst = unary_union([Point(x, y).buffer(r + M) for x, y, r in trees])
for sc in (False, True):
    t0 = time.time(); r = simulate_coverage(trB3, F, obst, W, 6, 1.0, section_control=sc)
    print("SC=%s missed %.1f%% overlap %.2f%% offtarget %.0f m2 applied %.2f ha  (%.1fs)" % (sc, r["missed_pct"], r["overlap_pct"], r["offtarget_m2"], r["applied_ha"], time.time() - t0))

# --- plot zoom + full
fig, axs = plt.subplots(1, 2, figsize=(16, 7))
for ax, (cx, cy, rr) in zip(axs, [(190, 150, 45), (0, 0, 0)]):
    ax.imshow(np.ma.masked_where(~r["target"], np.minimum(r["count"], 2)), origin="lower", extent=r["extent"],
              cmap=matplotlib.colors.ListedColormap(["#e74c3c", "#9fd18b", "#f39c12"]), vmin=0, vmax=2, interpolation="nearest")
    for g, kd in trB3.segs:
        ax.plot(*g.xy, color="#2f6db5" if kd == "swath" else "#e07b39", lw=0.6)
    for x, y, r_ in trees:
        ax.add_patch(plt.Circle((x, y), r_, color="#2f6b2f"))
    ax.set_aspect("equal")
    if cx: ax.set_xlim(cx - rr, cx + rr); ax.set_ylim(cy - rr, cy + rr)
fig.savefig("b3_test.png", dpi=110, bbox_inches="tight")
