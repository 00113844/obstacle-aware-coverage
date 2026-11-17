"""E2 (R8): native Fields2Cover pipelines vs the extended workflow (gate + B1-B3 + SC).

Native = obstacles (⊕ margin) as holes, F2C stages only, full factorial per instance over
  decomposition  {none, boustrophedon, trapezoidal}      (tutorial 7.2 recipe: decompose the mid
                  headland, open corridors with HG_Corridor_gen; split angle = the field's best angle)
  SG objective   {NSwath, NSwathModified, SwathLength}   (SG_BruteForce.generateBestSwaths, per cell)
  route order    {boustrophedon, snake, spiral(k), rp_first, rp_gls}
                 sorters: single-cell F2C orders; per cell the start-corner variant (0-3) closest to
                 where the machine is (gate, then the previous cell's end) is used, cells in F2C order.
                 Consecutive swaths are joined by PP_PathPlanning turns only (no headland graph), which
                 is F2C's native behaviour. Gate legs are measured along the mid-headland ring.
                 rp_*: RP_RoutePlannerBase with setStartAndEndPoint(gate), first solution or GLS 10 s.
  turn model     {Dubins, DubinsCC, ReedsShepp, ReedsSheppHC}
One work unit = (instance, decomp, objective, order) -> 4 turn models x 2 SC rows. Each unit runs in
its own subprocess with a timeout (F2C 2.1.1's route graph is slow on hole-heavy fields), and a
timeout is recorded as a result (status=timeout).

    python native_bench.py design --out ../results/native_v1
    python native_bench.py run    --out ../results/native_v1 --workers 8
    python native_bench.py merge  --out ../results/native_v1 --ext ../results/benchmark_v1_2/runs.csv
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f2c_bench as fb  # noqa: E402

NATIVE_VERSION = "native_v1 / native_bench 1.0.2"
NCONFIG = dict(
    shapes=["rectangle", "L"], densities=[0.0, 1.0], otype="mixed", pattern="poisson", widths=[12, 24],
    seeds=[0, 1, 2, 3, 4], gate="road",
    decomps=["none", "boustrophedon", "trapezoidal"],
    objectives=["NSwath", "NSwathModified", "SwathLength"],
    orders=["boustrophedon", "snake", "spiral", "rp_first", "rp_gls"],
    turn_models=["Dubins", "DubinsCC", "ReedsShepp", "ReedsSheppHC"],
    spiral_k=4, rp_time_gls=10, unit_timeout_s=900,
)
NCOLS = ("instance_id seed shape area_ha density_per_ha n_obstacles obstacle_type pattern width_m n_sections r_min_m "
         "gate_type k_headland pipeline decomp sg_obj route_order turn_model spiral_k sc status feasible n_cells "
         "n_swaths n_passes turns reversals n_manoeuvres L_work_m L_turn_m L_gate_m L_total_m eta_pct time_min "
         "over_crop_m outside_paddock_m min_clearance_m min_clearance_work_m min_clearance_turn_m max_curvature "
         "missed_pct overlap_pct offtarget_m2 applied_ha product_saved_pct crit_C "
         "t_hg_s t_decomp_s t_sg_s t_rp_s t_pp_s t_sim_s error").split()


def units(nc):
    out = []
    for spec in instances(nc):
        for d in nc["decomps"]:
            for o in nc["objectives"]:
                for r in nc["orders"]:
                    out.append(dict(spec, unit_id=f"{spec['instance_id']}__{d}__{o}__{r}", decomp=d, sg_obj=o,
                                    route_order=r))
    return out


def instances(nc):
    rows = []
    for shape in nc["shapes"]:
        for w in nc["widths"]:
            for seed in nc["seeds"]:
                for d in nc["densities"]:
                    ot, pat = ("none", "none") if d == 0 else (nc["otype"], nc["pattern"])
                    iid = f"{shape}_{ot}_{pat}_d{d:g}_w{w}_{nc['gate']}_s{seed}"   # same ids as f2c_bench
                    rows.append(dict(instance_id=iid, shape=shape, density=d, otype=ot, pattern=pat, width=w,
                                     n_sections=fb.CONFIG["widths"][w], gate_type=nc["gate"], seed=seed))
    return rows


# ---------------------------------------------------------------------------
def ring_leg(ring, p, q):
    """Travel p -> q along the mid-headland ring (to the ring, along the shorter arc, off the ring)."""
    from shapely.geometry import Point
    a, b = ring.project(Point(p)), ring.project(Point(q))
    arc = abs(a - b)
    arc = min(arc, ring.length - arc)
    return Point(p).distance(ring) + arc + Point(q).distance(ring)


def gate_legs_from_track(track):
    """Non-working length before the first and after the last working piece (RP routes: gate legs)."""
    kinds = [k for _, k in track.segs]
    if "swath" not in kinds:
        return 0.0
    i0, i1 = kinds.index("swath"), len(kinds) - 1 - kinds[::-1].index("swath")
    return sum(g.length for g, _ in track.segs[:i0]) + sum(g.length for g, _ in track.segs[i1 + 1:])


def sorted_swaths(f2c, swaths_by_cells, order, k, gate):
    """Single-cell F2C sorter per cell; variant (start corner) chosen greedily from the gate."""
    import numpy as np
    sorter = {"boustrophedon": f2c.RP_Boustrophedon, "snake": f2c.RP_Snake}.get(order)
    out = f2c.Swaths()
    here = np.asarray(gate, float)
    for ci in range(swaths_by_cells.size()):
        cell_sw = swaths_by_cells.at(ci)
        if cell_sw.size() == 0:
            continue
        best = None
        for v in range(4):
            if cell_sw.size() == 1:
                # F2C 2.1.1 RP_Snake segfaults on a 1-swath Swaths (verified on Nectar); every order is
                # identical for one swath, so use the Boustrophedon sorter (only the variant/direction differs)
                s = f2c.RP_Boustrophedon()
            elif order == "spiral":
                s = f2c.RP_Spiral()
                s.setSpiralSize(int(min(k, max(2, cell_sw.size()))))
            else:
                s = sorter()
            srt = s.genSortedSwaths(cell_sw.clone(), v)
            p0 = srt.at(0).startPoint()
            dist = float(np.hypot(p0.getX() - here[0], p0.getY() - here[1]))
            if best is None or dist < best[0]:
                best = (dist, srt)
        srt = best[1]
        e = srt.at(srt.size() - 1).endPoint()
        here = np.array([e.getX(), e.getY()])
        out.append(srt)
    return out


def run_unit(cfg, nc, u, out_dir):
    from shapely.ops import unary_union
    from bench_geom import length_outside, turn_length_inside, clearance, max_curvature
    f2c = fb._f2c()
    spec = dict(u, density=float(u["density"]), seed=int(u["seed"]))
    field, obs, gate = fb.build_instance(cfg, spec)
    w, ns = int(spec["width"]), int(spec["n_sections"])
    P = fb.Planner(cfg, field, w, gate)
    zones = unary_union([o["poly"].buffer(cfg["margin"]) for o in obs]) if obs else None
    holes = [p.simplify(0.2) for p in fb.polys(zones)] if obs else []
    common = dict(instance_id=spec["instance_id"], seed=spec["seed"], shape=spec["shape"],
                  area_ha=round(field.area / 1e4, 3), density_per_ha=spec["density"], n_obstacles=len(obs),
                  obstacle_type=spec["otype"], pattern=spec["pattern"], width_m=w, n_sections=ns,
                  r_min_m=cfg["r_min"], gate_type=spec["gate_type"], k_headland=P.k, pipeline="native",
                  decomp=u["decomp"], sg_obj=u["sg_obj"], route_order=u["route_order"],
                  spiral_k=nc["spiral_k"] if u["route_order"] == "spiral" else "")
    rows = []
    T = {}
    try:
        t0 = time.perf_counter()
        cells = P.cells(holes)
        mid = P.hl.generateHeadlands(cells, 0.5 * P.k * w)
        no = P.hl.generateHeadlands(cells, P.k * w)
        T["t_hg_s"] = time.perf_counter() - t0
        if no.size() == 0 or no.area() < 1.0:
            raise fb.NoSwaths("no swaths: empty mainland")
        obj = getattr(f2c, "OBJ_" + u["sg_obj"])()
        t0 = time.perf_counter()
        if u["decomp"] == "none":
            work, n_cells = no, no.size()
        else:
            big = max(range(no.size()), key=lambda i: no.getGeometry(i).area())
            ang = P.bf.computeBestAngle(obj, w, no.getGeometry(big))
            dec = f2c.DECOMP_Boustrophedon() if u["decomp"] == "boustrophedon" else f2c.DECOMP_TrapezoidalDecomp()
            dec.setSplitAngle(ang)
            dcells = dec.decompose(mid)
            work = f2c.HG_Corridor_gen().generateHeadlands(dcells, 0.5 * P.k * w)
            n_cells = dcells.size()
        T["t_decomp_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        sw = P.bf.generateBestSwaths(obj, w, work)
        T["t_sg_s"] = time.perf_counter() - t0
        n_sw = sw.sizeTotal()
        if n_sw == 0:
            raise fb.NoSwaths("no swaths")
        t0 = time.perf_counter()
        ring = max(fb.polys(fb.to_shapely(mid)), key=lambda p: p.area).exterior
        if u["route_order"].startswith("rp_"):
            from shapely.geometry import Point
            q = ring.interpolate(ring.project(Point(gate)))
            rp = f2c.RP_RoutePlannerBase()
            rp.setStartAndEndPoint(f2c.Point(q.x, q.y))
            gls = u["route_order"] == "rp_gls"
            plan_in = rp.genRoute(mid, sw, False, 1e-4, True, int(nc["rp_time_gls"]) if gls else 1, gls)
            leg_extra = None
        else:
            plan_in = sorted_swaths(f2c, sw, u["route_order"], nc["spiral_k"], gate)
            s0, s1 = plan_in.at(0).startPoint(), plan_in.at(plan_in.size() - 1).endPoint()
            leg_extra = (ring_leg(ring, gate, (s0.getX(), s0.getY())) + ring_leg(ring, (s1.getX(), s1.getY()), gate))
        T["t_rp_s"] = time.perf_counter() - t0
        HL = P.headland_passes(cells)
        crop = fb.to_shapely(no)
        sims = None
        for tm_name in nc["turn_models"]:
            tm = getattr(f2c, {"Dubins": "PP_DubinsCurves", "DubinsCC": "PP_DubinsCurvesCC",
                               "ReedsShepp": "PP_ReedsSheppCurves", "ReedsSheppHC": "PP_ReedsSheppCurvesHC"}[tm_name])()
            tm.setDiscretization(float(cfg["turn_discretization"]))
            t0 = time.perf_counter()
            try:
                track = fb.track_from_path(f2c, P.pp.planPath(P.robot, plan_in, tm))
            except Exception as e:     # e.g. F2C CubicSpline "x array is not sorted": fail this turn model only
                rows += [{**common, "turn_model": tm_name, "sc": sc, "status": "error", "feasible": False,
                          "error": f"{type(e).__name__}: {e}"[:300], **{k: round(v, 3) for k, v in T.items()}}
                         for sc in (0, 1)]
                continue
            t_pp = time.perf_counter() - t0
            if sims is None:           # coverage depends on the working pieces only: same for every turn model
                t0 = time.perf_counter()
                _, sims = fb.evaluate(cfg, track, field_shp=field, zones=zones, crop=crop, obstacles=obs, width=w,
                                      n_sections=ns, extra=HL)
                t_sim = time.perf_counter() - t0
            L_sw, L_tn = track.length("swath"), track.length("turn")
            L_hl = sum(g.length for g in HL)
            L_gate = gate_legs_from_track(track) if leg_extra is None else leg_extra
            L_nw = L_tn + (leg_extra or 0.0)
            Tsec = (L_sw + L_hl) / cfg["cruise_vel"] + L_nw / cfg["turn_vel"] + cfg["reverse_penalty_s"] * track.n_rev
            mc = round(min(clearance(track, obs, w)), 3) if obs else ""
            base = dict(status="ok", n_cells=n_cells, n_swaths=n_sw, n_passes=n_sw + P.k, turns=track.n_turns,
                        reversals=track.n_rev, n_manoeuvres=track.n_turns + track.n_rev,
                        L_work_m=round(L_sw + L_hl, 1), L_turn_m=round(L_nw, 1), L_gate_m=round(L_gate, 1),
                        L_total_m=round(L_sw + L_hl + L_nw, 1),
                        eta_pct=round(100 * (L_sw + L_hl) / (L_sw + L_hl + L_nw), 2), time_min=round(Tsec / 60, 2),
                        over_crop_m=round(turn_length_inside(track, crop.buffer(-1.0)), 1),
                        outside_paddock_m=round(length_outside(track, field), 1), min_clearance_m=mc,
                        min_clearance_work_m=fb._part_clear(track, obs, w, "swath"),
                        min_clearance_turn_m=fb._part_clear(track, obs, w, "turn"),
                        max_curvature=round(max((max_curvature(g) for g in track.pieces("swath")), default=0.0), 4),
                        feasible=bool(mc == "" or mc >= 0), turn_model=tm_name,
                        **{k: round(v, 3) for k, v in T.items()}, t_pp_s=round(t_pp, 3), t_sim_s=round(t_sim, 3))
            for sc in (False, True):
                s = sims[sc]
                missed_ha = s["missed_pct"] / 100 * s["_target_ha"]
                crit = cfg["c_t"] * base["time_min"] + cfg["c_p"] * s["applied_ha"] + cfg["c_m"] * missed_ha
                rows.append({**common, **base, "sc": int(sc), "missed_pct": s["missed_pct"],
                             "overlap_pct": s["overlap_pct"], "offtarget_m2": s["offtarget_m2"],
                             "applied_ha": s["applied_ha"], "product_saved_pct": s["product_saved_pct"],
                             "crit_C": round(crit, 2)})
    except Exception as e:
        err = str(e) if isinstance(e, fb.NoSwaths) else f"{type(e).__name__}: {e}"
        rows = [{**common, "turn_model": tm, "sc": sc, "status": "error", "feasible": False, "error": err[:300],
                 **{k: round(v, 3) for k, v in T.items()}} for tm in nc["turn_models"] for sc in (0, 1)]
    write_part(out_dir, u["unit_id"], rows)


def write_part(out_dir, uid, rows):
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    part = os.path.join(out_dir, "parts", uid + ".csv")
    with open(part + ".tmp", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=NCOLS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    os.replace(part + ".tmp", part)


def merge(out, ext=None):
    import glob
    parts = sorted(glob.glob(os.path.join(out, "parts", "*.csv")))
    rows = [r for p in parts for r in csv.DictReader(open(p))]
    ids = {r["instance_id"] for r in rows}
    n_ext = 0
    if ext and os.path.exists(ext):
        for r in csv.DictReader(open(ext)):
            if r["instance_id"] in ids:
                rows.append({**{k: r.get(k, "") for k in NCOLS}, "pipeline": "extended", "decomp": "",
                             "sg_obj": "", "route_order": r["strategy"], "status": "error" if r.get("error") else "ok",
                             "n_passes": "", "n_manoeuvres": (int(float(r["turns"] or 0)) + int(float(r["reversals"] or 0)))
                             if r.get("turns") else "", "L_gate_m": ""})
                n_ext += 1
    with open(os.path.join(out, "runs.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=NCOLS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"merged {len(parts)} units ({len(rows) - n_ext} native rows) + {n_ext} extended rows -> {out}/runs.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "run", "one", "merge"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--id")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--filter", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ext", default="", help="merge: extended-workflow runs.csv (f2c_bench) to append")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta_p = os.path.join(a.out, "run_meta.json")
    if a.cmd == "design":
        us = units(NCONFIG)
        with open(os.path.join(a.out, "design.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(us[0]))
            wr.writeheader(); wr.writerows(us)
        meta = fb.run_meta(fb.CONFIG)
        meta.update(native_version=NATIVE_VERSION, native_config=NCONFIG)
        meta["source_sha256"]["native_bench.py"] = __import__("hashlib").sha256(
            open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        json.dump(meta, open(meta_p, "w"), indent=1)
        print(f"{len(us)} units ({len(instances(NCONFIG))} instances) -> {a.out}/design.csv")
        return
    meta = json.load(open(meta_p))
    cfg, nc = meta["config"], meta["native_config"]
    cfg["widths"] = {int(k): v for k, v in cfg["widths"].items()}
    if a.cmd == "one":
        u = next(r for r in csv.DictReader(open(os.path.join(a.out, "design.csv"))) if r["unit_id"] == a.id)
        run_unit(cfg, nc, u, a.out)
    elif a.cmd == "run":
        design = list(csv.DictReader(open(os.path.join(a.out, "design.csv"))))
        done = lambda i: os.path.exists(os.path.join(a.out, "parts", i + ".csv"))
        todo = [r for r in design if a.filter in r["unit_id"] and not done(r["unit_id"])]
        todo = todo[:a.limit] if a.limit else todo
        print(f"{len(todo)} pending units, {a.workers} workers", flush=True)
        t_start = time.time()

        def job(u):
            t0 = time.time()
            try:
                p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--out", a.out, "--id", u["unit_id"]],
                                   capture_output=True, text=True, timeout=nc["unit_timeout_s"])
                status = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
                if p.returncode != 0:      # crash (e.g. OR-Tools CHECK abort): record it as a result
                    write_part(a.out, u["unit_id"], [dict(_unit_common(u), turn_model=tm, sc=sc, status="crash",
                                                          feasible=False, error=(p.stderr or "")[-300:])
                                                     for tm in nc["turn_models"] for sc in (0, 1)])
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
                write_part(a.out, u["unit_id"], [dict(_unit_common(u), turn_model=tm, sc=sc, status="timeout",
                                                      feasible=False, error=f"unit timeout {nc['unit_timeout_s']} s")
                                                 for tm in nc["turn_models"] for sc in (0, 1)])
            return u["unit_id"], status, time.time() - t0

        with ThreadPoolExecutor(a.workers) as ex:
            for n, (uid, status, dt) in enumerate(ex.map(job, todo), 1):
                el = time.time() - t_start
                print(f"[{n}/{len(todo)}] {uid}: {status} ({dt:.0f}s) | ETA {el / n * (len(todo) - n) / 60:.0f} min",
                      flush=True)
        merge(a.out, a.ext)
    elif a.cmd == "merge":
        merge(a.out, a.ext)


def _unit_common(u):
    return dict(instance_id=u["instance_id"], seed=u["seed"], shape=u["shape"], density_per_ha=u["density"],
                obstacle_type=u["otype"], pattern=u["pattern"], width_m=u["width"], n_sections=u["n_sections"],
                gate_type=u["gate_type"], pipeline="native", decomp=u["decomp"], sg_obj=u["sg_obj"],
                route_order=u["route_order"])


if __name__ == "__main__":
    main()
