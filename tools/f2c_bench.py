"""Headless benchmark runner for obstacle-aware coverage planning (R2-R5).

Runs in Colab (Fields2Cover needs Linux). Typical use from the driver notebook:
    python f2c_bench.py design  --out RESULTS            # writes RESULTS/design.csv + run_meta.json
    python f2c_bench.py run     --out RESULTS --workers 2  # plans every pending instance (resumable)
    python f2c_bench.py merge   --out RESULTS            # RESULTS/runs.csv
Each instance is planned in its own subprocess, so a C++ crash only loses that instance
(its error is written to RESULTS/parts/<id>.err).
"""
import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH_VERSION = "benchmark_v1 / f2c_bench 1.0"

CONFIG = dict(
    area_ha=20.0,
    shapes=["rectangle", "convex", "L"],
    densities=[0.0, 0.5, 1.0, 2.0],             # obstacles per ha
    obstacle_types=["tree", "outcrop", "mixed"],
    patterns=["poisson", "thomas"],
    widths={12: 6, 24: 12, 36: 18},             # implement width -> number of sections
    seeds=[0, 1, 2, 3, 4],
    gate_subset=dict(shape="rectangle", densities=[0.5, 2.0], width=24, otype="tree", pattern="poisson"),
    r_min=6.0, max_diff_curv=0.1, cruise_vel=2.5, turn_vel=1.2, reverse_penalty_s=10.0,
    margin=3.0, k_min=2, rp_time=10, rp_optimum=True, allow_reverse=True,
    turn_discretization=0.2,   # m between turn points (F2C default 0.01 -> ~20x fewer path points)
    raster_res=1.0, cover_headlands=True, placement_inset=10.0,
    # SC-aware selection criterion C = c_t*T[min] + c_p*applied_SC[ha] + c_m*missed_area[ha]  (R4)
    c_t=2.5,       # cost per minute of machine time
    c_p=40.0,      # cost per hectare of product applied
    c_m=300.0,     # cost per hectare left untreated
    angle_by_C=False,   # also pick the swath angle by C (A and B3), from {NSwath, SwathLength, NSwath+90°}
    geojson_seeds=[0],  # dump geometry for instances with these seeds
    instance_timeout_s=1800,
)

COLUMNS = ("instance_id seed shape area_ha density_per_ha n_obstacles obstacle_type pattern width_m cov_width_m "
           "n_sections r_min_m gate_type strategy sc feasible n_unplaced_detours decomp_failed swath_angle_deg "
           "n_swaths turns reversals L_work_m L_turn_m L_total_m eta_pct time_min over_crop_m outside_paddock_m "
           "min_clearance_m max_curvature missed_pct overlap_pct offtarget_m2 applied_ha product_saved_pct "
           "sections_on_pct t_hg_s t_sg_s t_rp_s t_pp_s t_detour_s t_sim_s "
           "crit_C selected turn_model k_headland gate_x gate_y n_groups error").split()


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------
def make_design(cfg):
    rows = []

    def add(shape, d, otype, pat, w, gate, seed):
        iid = f"{shape}_{otype}_{pat}_d{d:g}_w{w}_{gate}_s{seed}"
        rows.append(dict(instance_id=iid, shape=shape, density=d, otype=otype, pattern=pat, width=w,
                         n_sections=cfg["widths"][w], gate_type=gate, seed=seed))

    for shape in cfg["shapes"]:
        for w in cfg["widths"]:
            for seed in cfg["seeds"]:
                for d in cfg["densities"]:
                    if d == 0:
                        add(shape, 0.0, "none", "none", w, "road", seed)
                        continue
                    for otype in cfg["obstacle_types"]:
                        for pat in cfg["patterns"]:
                            add(shape, d, otype, pat, w, "road", seed)
    gs = cfg["gate_subset"]
    for d in gs["densities"]:
        for seed in cfg["seeds"]:
            add(gs["shape"], d, gs["otype"], gs["pattern"], gs["width"], "random", seed)
    return rows


def run_meta(cfg):
    meta = {"bench_version": BENCH_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": platform.python_version(), "config": cfg}
    try:
        import fields2cover as f2c
        meta["fields2cover"] = getattr(f2c, "__version__", "unknown")
    except Exception:
        meta["fields2cover"] = "not importable here"
    try:
        import shapely
        meta["shapely"] = shapely.__version__
    except Exception:
        pass
    try:
        meta["git"] = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip() or "not a git repo"
    except Exception:
        meta["git"] = "not a git repo"
    import hashlib
    meta["source_sha256"] = {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()[:16]
                             for f in ("f2c_bench.py", "bench_geom.py", "nb_geom.py")}
    return meta


# ---------------------------------------------------------------------------
# Fields2Cover helpers (copied from the tutorial; see SWIG memory note there)
# ---------------------------------------------------------------------------
def _f2c():
    import fields2cover as f2c
    return f2c


def to_shapely(g):
    from shapely import wkt, force_2d
    return force_2d(wkt.loads(g.exportToWkt()))


def make_ring(f2c, coords):
    coords = [(float(x), float(y)) for x, y in coords]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return f2c.LinearRing(f2c.VectorPoint([f2c.Point(x, y) for x, y in coords]))


def polys(g):
    return list(g.geoms) if hasattr(g, "geoms") else [g]


def track_from_path(f2c, path):
    from bench_geom import Track
    pts, kinds, rev = [], [], []
    for i in range(path.size()):
        s = path.getState(i)
        pts.append((s.point.getX(), s.point.getY()))
        kinds.append("swath" if s.type == f2c.PathSectionType_SWATH else "turn")
        rev.append(s.dir == f2c.PathDirection_BACKWARD)
    e = path.atEnd()
    pts.append((e.getX(), e.getY()))
    segs = [([pts[i], pts[i + 1]], kinds[i]) for i in range(len(kinds)) if pts[i] != pts[i + 1]]
    n_rev = sum(1 for i, r in enumerate(rev) if r and (i == 0 or not rev[i - 1]))
    return Track(segs, n_rev)


class Planner:
    """All F2C stages for one instance, with per-stage timers."""

    def __init__(self, cfg, field_shp, width, gate):
        self.f2c = f2c = _f2c()
        self.cfg, self.field_shp, self.w, self.gate = cfg, field_shp, width, gate
        r = cfg["r_min"]
        d_omega = r + math.sqrt(max(4 * r * r - (r + width / 2) ** 2, 0)) if width < 2 * r else r
        self.k = max(cfg["k_min"], math.ceil((d_omega + width / 2) / width))
        rb = f2c.Robot(width, width)
        rb.setMinTurningRadius(r)
        rb.setMaxDiffCurv(cfg["max_diff_curv"])
        rb.setCruiseVel(cfg["cruise_vel"])
        rb.setTurnVel(cfg["turn_vel"])
        self.robot = rb
        self.hl = f2c.HG_Const_gen()
        self.bf = f2c.SG_BruteForce()
        self.pp = f2c.PP_PathPlanning()

    def cells(self, holes=()):
        f2c = self.f2c
        cells = f2c.Cells(f2c.Cell(make_ring(f2c, list(self.field_shp.exterior.coords))))
        for h in holes:
            cells.addRing(0, make_ring(f2c, list(h.exterior.coords)))
        return cells

    def headlands(self, cells):
        t0 = time.perf_counter()
        mid = self.hl.generateHeadlands(cells, 0.5 * self.k * self.w)
        no = self.hl.generateHeadlands(cells, self.k * self.w)
        return mid, no, time.perf_counter() - t0

    def headland_passes(self, cells):
        from shapely.geometry import LineString
        if not self.cfg["cover_headlands"]:
            return []
        rings = self.hl.generateHeadlandSwaths(cells, self.w, max(1, int(self.k)), False)
        out = []
        for i in range(len(rings)):
            for p in polys(to_shapely(rings[i])):
                out += [LineString(p.exterior.coords)] + [LineString(r.coords) for r in p.interiors]
        return out

    def swaths(self, no_hl, angle=None):
        t0 = time.perf_counter()
        obj = self.f2c.OBJ_NSwathModified()
        big = max(range(no_hl.size()), key=lambda i: no_hl.getGeometry(i).area())
        if angle is None:
            angle = self.bf.computeBestAngle(obj, self.w, no_hl.getGeometry(big))
            sw = self.bf.generateBestSwaths(obj, self.w, no_hl)
        else:
            sw = self.bf.generateSwaths(angle, self.w, no_hl)
        return sw, angle, time.perf_counter() - t0

    def route(self, mid_hl, swaths):
        from shapely.geometry import Point
        f2c = self.f2c
        t0 = time.perf_counter()
        poly = max(polys(to_shapely(mid_hl)), key=lambda p: p.area)
        q = poly.exterior.interpolate(poly.exterior.project(Point(self.gate)))
        rp = f2c.RP_RoutePlannerBase()
        rp.setStartAndEndPoint(f2c.Point(q.x, q.y))
        route = rp.genRoute(mid_hl, swaths, False, 1e-4, True, int(self.cfg["rp_time"]), bool(self.cfg["rp_optimum"]))
        return route, time.perf_counter() - t0

    def path(self, route):
        from bench_geom import length_outside
        f2c = self.f2c
        t0 = time.perf_counter()
        models = {"Dubins CC": f2c.PP_DubinsCurvesCC()}
        if self.cfg["allow_reverse"] and self.w < 2 * self.cfg["r_min"]:
            models["Reeds-Shepp HC"] = f2c.PP_ReedsSheppCurvesHC()
        best = None
        for name, tm in models.items():
            tm.setDiscretization(float(self.cfg.get("turn_discretization", 0.01)))
            tr = track_from_path(f2c, self.pp.planPath(self.robot, route, tm))
            score = (tr.length("turn") / self.cfg["turn_vel"] + self.cfg["reverse_penalty_s"] * tr.n_rev
                     + 6.0 * length_outside(tr, self.field_shp))
            if best is None or score < best[0]:
                best = (score, name, tr)
        return best[2], best[1], time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(cfg, track, *, field_shp, zones, crop, obstacles, width, n_sections, extra):
    """Path metrics (shared by both SC rows) + one simulation per SC setting."""
    from bench_geom import turn_length_inside, length_outside, clearance, max_curvature, simulate_coverage
    L_sw, L_tn = track.length("swath"), track.length("turn")
    L_hl = sum(g.length for g in extra)
    T = (L_sw + L_hl) / cfg["cruise_vel"] + L_tn / cfg["turn_vel"] + cfg["reverse_penalty_s"] * track.n_rev
    base = dict(turns=track.n_turns, reversals=track.n_rev, L_work_m=round(L_sw + L_hl, 1), L_turn_m=round(L_tn, 1),
                L_total_m=round(L_sw + L_hl + L_tn, 1), eta_pct=round(100 * (L_sw + L_hl) / (L_sw + L_hl + L_tn), 2),
                time_min=round(T / 60, 2), over_crop_m=round(turn_length_inside(track, crop.buffer(-1.0)), 1),
                outside_paddock_m=round(length_outside(track, field_shp), 1),
                min_clearance_m=round(min(clearance(track, obstacles, width)), 3) if obstacles else "",
                max_curvature=round(max((max_curvature(g) for g in track.pieces("swath")), default=0.0), 4))
    sims = {}
    for sc in (False, True):
        t0 = time.perf_counter()
        r = simulate_coverage(track, field_shp, zones, width, n_sections, cfg["raster_res"], section_control=sc,
                              extra_passes=extra)
        sims[sc] = dict(missed_pct=round(r["missed_pct"], 3), overlap_pct=round(r["overlap_pct"], 3),
                        offtarget_m2=round(r["offtarget_m2"], 1), applied_ha=round(r["applied_ha"], 4),
                        sections_on_pct=round(r["sections_on_pct"], 2), t_sim_s=round(time.perf_counter() - t0, 3),
                        _target_ha=r["target_ha"], _count=r["count"], _extent=r["extent"])
    saved = 100 * (1 - sims[True]["applied_ha"] / sims[False]["applied_ha"]) if sims[False]["applied_ha"] else 0.0
    for sc in sims:
        s = sims[sc]
        missed_ha = s["missed_pct"] / 100 * s["_target_ha"]
        s["crit_C"] = round(cfg["c_t"] * base["time_min"] + cfg["c_p"] * s["applied_ha"] + cfg["c_m"] * missed_ha, 2)
        s["product_saved_pct"] = round(saved, 3)
    return base, sims


# ---------------------------------------------------------------------------
# One instance
# ---------------------------------------------------------------------------
def build_instance(cfg, spec):
    from bench_geom import (make_field, make_obstacles, road_gate, random_vertex_gate, stable_seed)
    field = make_field(spec["shape"], cfg["area_ha"], np.random.default_rng(stable_seed(spec["shape"], spec["seed"])))
    wmax = max(cfg["widths"])
    region = field.buffer(-(cfg["k_min"] * wmax + cfg["placement_inset"]))  # same layout for every width
    n = int(round(spec["density"] * cfg["area_ha"]))
    obs = make_obstacles(region, n, spec["otype"], spec["pattern"],
                         np.random.default_rng(stable_seed(spec["shape"], spec["density"], spec["otype"],
                                                           spec["pattern"], spec["seed"])))
    gate = road_gate(field) if spec["gate_type"] == "road" else \
        random_vertex_gate(field, np.random.default_rng(stable_seed("gate", spec["shape"], spec["seed"])))
    return field, obs, gate


def run_instance(cfg, spec, out_dir):
    from shapely.geometry import mapping, Point
    from shapely.ops import unary_union
    from bench_geom import make_drive_around_poly, obstacle_groups

    field, obs, gate = build_instance(cfg, spec)
    w, ns = int(spec["width"]), int(spec["n_sections"])
    P = Planner(cfg, field, w, gate)
    zones_list = [o["poly"].buffer(cfg["margin"]) for o in obs]
    zones = unary_union(zones_list) if obs else None
    holes = [p.simplify(0.2) for p in polys(zones)] if obs else []
    common = dict(instance_id=spec["instance_id"], seed=spec["seed"], shape=spec["shape"],
                  area_ha=round(field.area / 1e4, 3), density_per_ha=spec["density"], n_obstacles=len(obs),
                  obstacle_type=spec["otype"], pattern=spec["pattern"], width_m=w, cov_width_m=w, n_sections=ns,
                  r_min_m=cfg["r_min"], gate_type=spec["gate_type"], k_headland=P.k,
                  gate_x=round(gate[0], 2), gate_y=round(gate[1], 2),
                  n_groups=len(obstacle_groups(obs, cfg["margin"])) if obs else 0)
    rows, geo = [], {}

    def emit(strategy, track, crop, extra, timings, angle, n_sw, turn_model, unplaced="", decomp_failed="", error=""):
        if track is None:
            for sc in (0, 1):
                rows.append({**common, "strategy": strategy, "sc": sc, "feasible": False,
                             "decomp_failed": decomp_failed, "error": error, **timings})
            return
        base, sims = evaluate(cfg, track, field_shp=field, zones=zones, crop=crop, obstacles=obs, width=w,
                              n_sections=ns, extra=extra)
        ok_clear = base["min_clearance_m"] == "" or base["min_clearance_m"] >= 0
        feasible = bool(ok_clear and not unplaced)
        for sc in (False, True):
            s = sims[sc]
            rows.append({**common, "strategy": strategy, "sc": int(sc), "feasible": feasible,
                         "n_unplaced_detours": unplaced, "decomp_failed": decomp_failed,
                         "swath_angle_deg": round(math.degrees(angle) % 180, 2) if angle is not None else "",
                         "n_swaths": n_sw, **base, **{k: v for k, v in s.items() if not k.startswith("_")},
                         **timings, "turn_model": turn_model, "error": error})
        geo[strategy] = dict(track=track, extra=extra, sims=sims)

    # ---- A (open-paddock plan; reference, infeasible if it drives through obstacles) ----
    cells_A = P.cells()
    mid_A, no_A, t_hg = P.headlands(cells_A)
    HL_A = P.headland_passes(cells_A)
    no_A_shp = to_shapely(no_A)
    cand_angles = [None]
    if cfg["angle_by_C"]:
        big = max(range(no_A.size()), key=lambda i: no_A.getGeometry(i).area())
        a1 = P.bf.computeBestAngle(P.f2c.OBJ_NSwathModified(), w, no_A.getGeometry(big))
        a2 = P.bf.computeBestAngle(P.f2c.OBJ_SwathLength(), w, no_A.getGeometry(big))
        cand_angles = [a1, a2, (a1 + math.pi / 2) % math.pi]
    best_A = None
    for ang in cand_angles:
        sw, angle, t_sg = P.swaths(no_A, ang)
        route, t_rp = P.route(mid_A, sw)
        track, model, t_pp = P.path(route)
        base, sims = evaluate(cfg, track, field_shp=field, zones=zones, crop=no_A_shp, obstacles=obs,
                              width=w, n_sections=ns, extra=HL_A) if len(cand_angles) > 1 else (None, {True: {"crit_C": 0}})
        if best_A is None or sims[True]["crit_C"] < best_A[0]:
            best_A = (sims[True]["crit_C"], track, model, angle, sw.sizeTotal(), t_sg, t_rp, t_pp)
    _, track_A, model_A, angle_A, nsw_A, t_sg, t_rp, t_pp = best_A
    tim_A = dict(t_hg_s=round(t_hg, 3), t_sg_s=round(t_sg, 3), t_rp_s=round(t_rp, 3), t_pp_s=round(t_pp, 3), t_detour_s=0.0)
    crop_A = no_A_shp.difference(zones) if obs else no_A_shp
    emit("A", track_A, crop_A, HL_A, tim_A, angle_A, nsw_A, model_A)

    if obs:
        # ---- B1: obstacles as holes with headland rings ----
        try:
            cells_B = P.cells(holes)
            mid_B, no_B, t_hgB = P.headlands(cells_B)
            HL_B = P.headland_passes(cells_B)
            no_B_shp = to_shapely(no_B)
            sw, angle, t_sgB = P.swaths(no_B)
            route, t_rpB = P.route(mid_B, sw)
            track, model, t_ppB = P.path(route)
            emit("B1", track, no_B_shp, HL_B, dict(t_hg_s=round(t_hgB, 3), t_sg_s=round(t_sgB, 3),
                 t_rp_s=round(t_rpB, 3), t_pp_s=round(t_ppB, 3), t_detour_s=0.0), angle, sw.sizeTotal(), model)
        except Exception as e:
            emit("B1", None, None, None, {}, None, "", "", error=f"{type(e).__name__}: {e}")
            mid_B = None

        # ---- B2: Boustrophedon decomposition ----
        try:
            if mid_B is None:
                raise RuntimeError("B1 headlands unavailable")
            t0 = time.perf_counter()
            dec = P.f2c.DECOMP_Boustrophedon()
            dec.setSplitAngle(angle_A)
            dcells = dec.decompose(mid_B)
            no_B2 = P.f2c.HG_Corridor_gen().generateHeadlands(dcells, 0.5 * P.k * w)
            t_hg2 = time.perf_counter() - t0
            sw, angle, t_sg2 = P.swaths(no_B2)
            route, t_rp2 = P.route(mid_B, sw)
            track, model, t_pp2 = P.path(route)
            emit("B2", track, no_B_shp, HL_B, dict(t_hg_s=round(t_hg2, 3), t_sg_s=round(t_sg2, 3),
                 t_rp_s=round(t_rp2, 3), t_pp_s=round(t_pp2, 3), t_detour_s=0.0), angle, sw.sizeTotal(), model,
                 decomp_failed=False)
        except Exception as e:
            emit("B2", None, None, None, {}, None, "", "", decomp_failed=True, error=f"{type(e).__name__}: {e}")

        # ---- B3: A's plan + detours (t_hg..t_pp are A's, since B3 reuses A's plan) ----
        t0 = time.perf_counter()
        track_B3, unplaced = make_drive_around_poly(track_A, obs, cfg["margin"], w, cfg["r_min"], field.buffer(-w / 2))
        emit("B3", track_B3, crop_A, HL_A, {**tim_A, "t_detour_s": round(time.perf_counter() - t0, 3)},
             angle_A, nsw_A, model_A, unplaced=unplaced)

    # ---- selection by C among feasible obstacle strategies (SC on) ----
    cands = [r for r in rows if r["sc"] == 1 and r.get("feasible") and (r["strategy"] != "A" or not obs)]
    best = min(cands, key=lambda r: r["crit_C"])["strategy"] if cands else ""
    for r in rows:
        r["selected"] = int(r["strategy"] == best)

    # ---- outputs ----
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    part = os.path.join(out_dir, "parts", spec["instance_id"] + ".csv")
    with open(part + ".tmp", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    os.replace(part + ".tmp", part)
    if os.path.exists(part[:-4] + ".err"):
        os.remove(part[:-4] + ".err")
    if int(spec["seed"]) in cfg["geojson_seeds"]:
        write_geojson(os.path.join(out_dir, "geojson", spec["instance_id"]), field, obs, zones, gate, geo, w)
    return rows


def write_geojson(stem, field, obs, zones, gate, geo, width):
    """Geometry dump + coverage-count rasters (npz) for figures (R3, R6)."""
    from shapely.geometry import mapping, Point
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    feats = [{"type": "Feature", "properties": {"layer": "paddock"}, "geometry": mapping(field)},
             {"type": "Feature", "properties": {"layer": "gate"}, "geometry": mapping(Point(gate))}]
    feats += [{"type": "Feature", "properties": {"layer": "obstacle", "kind": o["type"], "id": i},
               "geometry": mapping(o["poly"])} for i, o in enumerate(obs)]
    if zones is not None:
        feats.append({"type": "Feature", "properties": {"layer": "zone"}, "geometry": mapping(zones)})
    arrays = {}
    for strat, g in geo.items():
        feats += [{"type": "Feature", "properties": {"layer": "path", "strategy": strat, "kind": k, "seq": j},
                   "geometry": mapping(ln)} for j, (ln, k) in enumerate(g["track"].segs)]
        feats += [{"type": "Feature", "properties": {"layer": "headland_pass", "strategy": strat},
                   "geometry": mapping(ln)} for ln in g["extra"]]
        for sc in (False, True):
            arrays[f"{strat}_sc{int(sc)}"] = g["sims"][sc]["_count"]
            arrays["extent"] = np.array(g["sims"][sc]["_extent"])
    with open(stem + ".geojson", "w") as fh:
        json.dump({"type": "FeatureCollection", "properties": {"width": width}, "features": feats}, fh)
    np.savez_compressed(stem + "_coverage.npz", **arrays)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_cfg(out):
    p = os.path.join(out, "run_meta.json")
    if os.path.exists(p):
        cfg = json.load(open(p))["config"]
        cfg["widths"] = {int(k): v for k, v in cfg["widths"].items()}
        return cfg
    return CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "run", "one", "merge"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--id")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="plan at most N pending instances")
    ap.add_argument("--filter", default="", help="only instance ids containing this text")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.cmd == "design":
        rows = make_design(CONFIG)
        with open(os.path.join(a.out, "design.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
        json.dump(run_meta(CONFIG), open(os.path.join(a.out, "run_meta.json"), "w"), indent=1)
        print(f"{len(rows)} instances -> {a.out}/design.csv")

    elif a.cmd == "one":
        cfg = load_cfg(a.out)
        spec = next(r for r in csv.DictReader(open(os.path.join(a.out, "design.csv"))) if r["instance_id"] == a.id)
        spec["density"] = float(spec["density"]); spec["seed"] = int(spec["seed"])
        try:
            run_instance(cfg, spec, a.out)
        except Exception:
            os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
            open(os.path.join(a.out, "parts", a.id + ".err"), "w").write(traceback.format_exc())
            raise

    elif a.cmd == "run":
        cfg = load_cfg(a.out)
        design = list(csv.DictReader(open(os.path.join(a.out, "design.csv"))))
        done = lambda i: os.path.exists(os.path.join(a.out, "parts", i + ".csv"))
        todo = [r["instance_id"] for r in design if a.filter in r["instance_id"] and not done(r["instance_id"])]
        if a.limit:
            todo = todo[:a.limit]
        print(f"{len(todo)} pending instances, {a.workers} workers", flush=True)
        t_start = time.time()

        def job(iid):
            t0 = time.time()
            try:
                p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--out", a.out, "--id", iid],
                                   capture_output=True, text=True, timeout=cfg["instance_timeout_s"])
                status = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
                if p.returncode != 0 and not os.path.exists(os.path.join(a.out, "parts", iid + ".err")):
                    os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
                    open(os.path.join(a.out, "parts", iid + ".err"), "w").write(p.stderr[-5000:])
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
                open(os.path.join(a.out, "parts", iid + ".err"), "w").write("timeout")
            return iid, status, time.time() - t0

        with ThreadPoolExecutor(a.workers) as ex:
            for n, (iid, status, dt) in enumerate(ex.map(job, todo), 1):
                el = time.time() - t_start
                print(f"[{n}/{len(todo)}] {iid}: {status} ({dt:.0f}s) | ETA {el / n * (len(todo) - n) / 60:.0f} min",
                      flush=True)
        merge(a.out)

    elif a.cmd == "merge":
        merge(a.out)


def merge(out):
    parts = sorted(f for f in os.listdir(os.path.join(out, "parts")) if f.endswith(".csv")) \
        if os.path.isdir(os.path.join(out, "parts")) else []
    with open(os.path.join(out, "runs.csv"), "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=COLUMNS)
        wr.writeheader()
        for f in parts:
            wr.writerows(csv.DictReader(open(os.path.join(out, "parts", f))))
    errs = [f for f in os.listdir(os.path.join(out, "parts")) if f.endswith(".err")] if parts else []
    print(f"merged {len(parts)} instances -> {out}/runs.csv ({len(errs)} with errors)")


if __name__ == "__main__":
    main()
