"""E4 / R10: controlled-traffic tramline analysis on the E1 (+E3 B4) sprayer plans.

The sprayer plan is the tramline master (w = 12/24/36 m -> 1:1, 2:1, 3:1 on a 12 m base module). This
driver re-plans the E1 design with f2c_bench (same plans as 1.5: A, B1, B2, B3, B4) and, through
f2c_bench.HOOKS, adds
  Part 2   B3c       B3 with fillet rho = r_ctf + d_max and lane-envelope clearance h_env = max(w_s/2, d_max + w_b/2)
  Part 2b  B3n       nested drive-around (s_min = 6 m) and B3n_s12 (s_min = w_b, sensitivity)
           at 2:1 also B3c_ctr / B3n_ctr with the centred-layout d_max (sensitivity)
  Part 1   base-lane geometry per plan x lane layout (shared = primary, centred = sensitivity):
           validity (cusp, radius, simple), clearances, crossings, spacing, sown coverage (12 m seeder,
           with / without 6 sections), off-lane wheel travel, trafficked area
  Part 3   (subset) seeder and harvester routes constrained to the base lanes: RP_RoutePlannerBase from
           the gate (GLS 10 s) + PP DubinsCC; time, turns, eta, off-lane travel, season trafficked area.
One row per (instance, strategy, lane_layout) in <out>/runs.csv; the sprayer rows (both SC settings) are
kept in <out>/sprayer_runs.csv (f2c_bench schema).

    python ctf_bench.py design --out ../results/ctf_v1
    python ctf_bench.py run    --out ../results/ctf_v1 --workers 8
    python ctf_bench.py merge  --out ../results/ctf_v1
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f2c_bench as fb  # noqa: E402
import ctf_geom as cg  # noqa: E402

CTF_VERSION = "ctf_v1 / ctf_bench 1.0"
CTF_COLS = ("instance_id seed shape area_ha density_per_ha n_obstacles obstacle_type pattern width_m gate_type "
            "k_headland strategy lane_layout module_ratio n_ratio w_b_m d_max_m rho_m h_env_m s_min_m "
            "feasible n_unplaced_detours time_min L_total_m missed_pct overlap_pct crit_C min_clearance_m "
            "n_lanes n_lanes_detoured n_edge_lanes lane_valid_pct lane_valid_detoured_pct lane_valid_seeder_pct "
            "lane_valid_seeder_detoured_pct n_cusp_lanes n_nonsimple_lanes n_tight_lanes n_clear_fail_lanes "
            "lane_min_radius_m lane_clear_tool_min_m lane_clear_wheel_min_m lane_outside_field_m "
            "n_lane_crossings n_lane_overlaps n_lane_breaks n_lane_break_ends n_lane_hl_crossings lane_spacing_min_m lane_spacing_max_m lane_squeeze_share_pct lane_squeeze_len_m "
            "lane_gap_len_m sown_missed_pct sown_overlap_pct sown_missed_sc_pct sown_overlap_sc_pct "
            "off_lane_wheel_sprayer_m unique_trafficked_ha n_nested_passes max_nest_depth nest_depth_by_group "
            "nest_iterations nest_conflicts part3 seeder_time_min seeder_turns seeder_eta_pct seeder_L_total_m seeder_L_turn_m "
            "seeder_off_lane_m harvester_time_min harvester_turns harvester_eta_pct harvester_L_total_m "
            "harvester_L_turn_m harvester_off_lane_m unique_trafficked_season_ha t_ctf_s error").split()

STATE = {}


def _ratio(w, ctf):
    return int(round(w / ctf["w_b"]))


def ctf_variants(w, ctf):
    """(strategy, layout_for_params, s_min, max nest depth) of the CTF drive-around variants for sprayer width w."""
    n = _ratio(w, ctf)
    out = [("B3c", "shared", None, None), ("B3n", "shared", ctf["s_min"], None),
           ("B3n_s12", "shared", ctf["s_min_sens"], None), ("B3n_k2", "shared", ctf["s_min"], ctf["nest_k"])]
    if n % 2 == 0:
        out += [("B3c_ctr", "centred", None, None), ("B3n_ctr", "centred", ctf["s_min"], None)]
    return out


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def hook_strategies(c):
    cfg, obs, w, field = c["cfg"], c["obs"], c["w"], c["field"]
    ctf = cfg["ctf"]
    r_ctf = max(ctf["r_min_seeder"], ctf["r_min_harvester"])
    STATE["params"], STATE["nest"] = {}, {}
    allowed = field.buffer(-w / 2)
    for name, layout, s_min, kmax in ctf_variants(w, ctf):
        cp = cg.ctf_params(w, ctf["w_b"], layout, r_ctf)
        STATE["params"][name] = dict(cp, s_min=s_min, layout=layout)
        t0 = time.perf_counter()
        try:
            if s_min is None:
                tr, un = cg.make_b3c(c["track_A"], obs, cfg["margin"], w, cp["rho"], cp["h_env"], allowed)
            else:
                tr, un, st = cg.make_b3n(c["track_A"], obs, cfg["margin"], w, ctf["w_b"], cp["rho"], cp["h_env"], r_ctf,
                                         cp["d_max"], s_min, allowed, max_depth=kmax)
                STATE["nest"][name] = st
        except Exception as e:
            c["emit"](name, None, None, None, {}, None, "", "", error=f"{type(e).__name__}: {e}"[:300])
            continue
        c["emit"](name, tr, c["crop_A"], c["HL_A"], {**c["tim_A"], "t_detour_s": round(time.perf_counter() - t0, 3)},
                  c["angle_A"], c["nsw_A"], c["model_A"], unplaced=un)


def _is_ring(ln):
    c = ln.coords
    return len(c) > 3 and np.allclose(c[0], c[-1])


def _part3_subset(spec, ctf):
    f = ctf["part3_filter"]
    return (spec["otype"] == f["otype"] and spec["pattern"] == f["pattern"] and float(spec["density"]) == f["density"]
            and spec["gate_type"] == f["gate_type"] and spec["shape"] in f["shapes"])


def route_machine(P, cfg, ctf, mid, lanes, r_min, gate):
    """RP (gate, GLS) + PP DubinsCC over fixed base lanes for a w_b machine. Returns Track."""
    from shapely.geometry import Point
    f2c = P.f2c
    w_b = ctf["w_b"]
    sw = f2c.Swaths()
    for i, ln in enumerate(lanes):
        ls = f2c.LineString()
        for x, y in ln.coords:
            ls.addPoint(float(x), float(y))
        sw.emplace_back(ls, float(w_b), i)
    sbc = f2c.SwathsByCells()
    sbc.push_back(sw)
    poly = max(fb.polys(fb.to_shapely(mid)), key=lambda p: p.area)
    q = poly.exterior.interpolate(poly.exterior.project(Point(gate)))
    rp = f2c.RP_RoutePlannerBase()
    rp.setStartAndEndPoint(f2c.Point(q.x, q.y))
    route = rp.genRoute(mid, sbc, False, 1e-4, True, int(ctf["part3_rp_time"]), True)
    rb = f2c.Robot(w_b, w_b)
    rb.setMinTurningRadius(float(r_min))
    rb.setMaxDiffCurv(cfg["max_diff_curv"])
    rb.setCruiseVel(cfg["cruise_vel"])
    rb.setTurnVel(cfg["turn_vel"])
    tm = f2c.PP_DubinsCurvesCC()
    tm.setDiscretization(float(cfg["turn_discretization"]))
    return fb.track_from_path(f2c, P.pp.planPath(rb, route, tm))


def hook_post(c):
    from shapely.geometry import Point
    from bench_geom import clean_poly
    cfg, spec, P, field, obs, geo, rows, w = (c[k] for k in ("cfg", "spec", "P", "field", "obs", "geo", "rows", "w"))
    ctf = cfg["ctf"]
    w_b, n = ctf["w_b"], _ratio(w, ctf)
    r_ctf = max(ctf["r_min_seeder"], ctf["r_min_harvester"])
    part3 = bool(obs) and _part3_subset(spec, ctf)
    WR = cg.WheelRaster(field.buffer(w + 10).bounds, ctf["traffic_res"])
    common = dict(instance_id=spec["instance_id"], seed=spec["seed"], shape=spec["shape"],
                  area_ha=round(field.area / 1e4, 3), density_per_ha=spec["density"], n_obstacles=len(obs),
                  obstacle_type=spec["otype"], pattern=spec["pattern"], width_m=w, gate_type=spec["gate_type"],
                  k_headland=P.k, module_ratio=f"{n}:1", n_ratio=n, w_b_m=w_b)
    regions = {}
    out = []
    for strat, g in geo.items():
        fam = "B" if strat in ("B1", "B2") else "A"
        if fam not in regions:
            cells = c["cells_B"] if fam == "B" else c["cells_A"]
            regions[fam] = None if cells is None else clean_poly(fb.to_shapely(cells))
        region = regions[fam]
        mid = c["mid_B"] if fam == "B" else c["mid_A"]
        sp = next((r for r in rows if r["strategy"] == strat and r["sc"] == 1), {})
        prm = STATE.get("params", {}).get(strat, {})
        nest = STATE.get("nest", {}).get(strat, {})
        rings = [l for l in g["extra"] if _is_ring(l)]
        open_passes = [l for l in g["extra"] if not _is_ring(l)]        # B2 corridor passes
        passes = list(g["track"].pieces("swath")) + open_passes
        hl = cg.headland_lanes(region, w_b, P.k * w) if region is not None else []
        spray_lines = cg.path_lines(g["track"], rings)
        layouts = ["shared", "centred"]
        cache = {}
        for layout in layouts:
            t0 = time.perf_counter()
            row = {**common, "strategy": strat, "lane_layout": layout,
                   "d_max_m": prm.get("d_max", cg.d_max(n, w_b, layout)), "rho_m": prm.get("rho", cfg["r_min"]),
                   "h_env_m": prm.get("h_env", w / 2), "s_min_m": prm.get("s_min") or "",
                   **{k: sp.get(k, "") for k in ("feasible", "n_unplaced_detours", "time_min", "L_total_m", "missed_pct",
                                                  "overlap_pct", "crit_C", "min_clearance_m")},
                   "n_nested_passes": nest.get("n_nested_passes", ""), "max_nest_depth": nest.get("max_depth", ""),
                   "nest_depth_by_group": nest.get("depth_by_group", ""), "nest_iterations": nest.get("iterations", ""), "nest_conflicts": nest.get("conflicts", ""),
                   "part3": int(part3)}
            if not passes:
                out.append({**row, "error": "no passes"})
                continue
            if n % 2 == 1 and "shared" in cache:      # odd n: both layouts identical -> copy
                out.append({**cache["shared"], "lane_layout": layout})
                continue
            try:
                L = cg.make_lanes(passes, w, w_b, layout, r_ctf, ctf["curv_ds"], ctf["curv_window"], ctf["radius_tol"])
                lines = [l["line"] for l in L]
                det = [l for l in L if l["detoured"]]
                seeder_ok = lambda l: (not l["cusp"]) and l["simple"] and l["min_radius"] >= ctf["r_min_seeder"] * (1 - ctf["radius_tol"])
                tool_cl = [min((l["line"].distance(o["poly"]) for o in obs), default=np.inf) - w_b / 2 for l in L]
                spc = cg.spacing(lines, w_b)
                # tramline stops: lane ends that are not at the outer headland (holes, corridors), plus lanes that
                # run into an obstacle (tool clearance < 0, e.g. unplaced detours) and must stop there
                band = P.k * w + w_b
                n_break_ends = sum(int(Point(e).distance(field.exterior) > band)
                                   for l in lines for e in (l.coords[0], l.coords[-1]))
                stepv = 2.0
                inner = field.buffer(-w_b / 2 + 0.05)
                lt = fb.Track if False else None  # noqa
                from bench_geom import Track, simulate_coverage
                ltrack = Track([(l, "swath") for l in lines])
                zones = c["zones"]
                sims = {sc: simulate_coverage(ltrack, field, zones, w_b, ctf["seeder_sections"], cfg["raster_res"],
                                              section_control=sc, extra_passes=hl) for sc in (False, True)}
                net = WR.paint(WR.blank(), lines + hl, ctf["gauge"], ctf["off_lane_tol"])
                traffic = WR.paint(WR.blank(), spray_lines + lines + hl, ctf["gauge"], ctf["tyre_width"] / 2)
                row.update(
                    n_lanes=len(L), n_lanes_detoured=len(det), n_edge_lanes=sum(l["edge"] for l in L),
                    lane_valid_pct=round(100 * np.mean([l["valid"] for l in L]), 3),
                    lane_valid_detoured_pct=round(100 * np.mean([l["valid"] for l in det]), 3) if det else "",
                    lane_valid_seeder_pct=round(100 * np.mean([seeder_ok(l) for l in L]), 3),
                    lane_valid_seeder_detoured_pct=round(100 * np.mean([seeder_ok(l) for l in det]), 3) if det else "",
                    n_cusp_lanes=sum(l["cusp"] for l in L), n_nonsimple_lanes=sum(not l["simple"] for l in L),
                    n_tight_lanes=sum((not l["cusp"]) and l["min_radius"] < r_ctf * (1 - ctf["radius_tol"]) for l in L),
                    n_clear_fail_lanes=sum(t < 0 for t in tool_cl) if obs else "",
                    lane_min_radius_m=round(min(l["min_radius"] for l in L), 3) if det else "",
                    lane_clear_tool_min_m=round(min(tool_cl), 3) if obs else "",
                    lane_clear_wheel_min_m=round(cg.lane_clearance(lines, obs, ctf["gauge"] / 2), 3) if obs else "",
                    lane_outside_field_m=round(sum(l.difference(inner).length for l in lines), 1),
                    n_lane_crossings=cg.crossings(lines), n_lane_overlaps=cg.overlaps(lines),
                    n_lane_break_ends=n_break_ends,
                    n_lane_breaks=int(round(n_break_ends / 2)) + (sum(t < 0 for t in tool_cl) if obs else 0),
                    n_lane_hl_crossings=int(sum(len(cg.shapely.STRtree(hl).query(l, predicate="crosses")) for l in lines)) if hl else 0,
                    lane_spacing_min_m=round(float(spc.min()), 3) if len(spc) else "",
                    lane_spacing_max_m=round(float(spc.max()), 3) if len(spc) else "",
                    lane_squeeze_share_pct=round(100 * float((spc < 0.75 * w_b).mean()), 3) if len(spc) else "",
                    lane_squeeze_len_m=round(float((spc < w_b - 0.5).sum() * stepv), 1),
                    lane_gap_len_m=round(float((spc > w_b + 0.5).sum() * stepv), 1),
                    sown_missed_pct=round(sims[False]["missed_pct"], 3), sown_overlap_pct=round(sims[False]["overlap_pct"], 3),
                    sown_missed_sc_pct=round(sims[True]["missed_pct"], 3), sown_overlap_sc_pct=round(sims[True]["overlap_pct"], 3),
                    off_lane_wheel_sprayer_m=round(WR.off_length(net, spray_lines, ctf["gauge"]), 1),
                    unique_trafficked_ha=round(traffic.sum() * ctf["traffic_res"] ** 2 / 1e4, 4))
                if part3 and strat != "A" and mid is not None:
                    season = traffic.copy()
                    for mach, rmin in (("seeder", ctf["r_min_seeder"]), ("harvester", ctf["r_min_harvester"])):
                        try:
                            tr = route_machine(P, cfg, ctf, mid, lines, rmin, c["gate"])
                            L_sw, L_tn = tr.length("swath"), tr.length("turn")
                            L_hl = sum(l.length for l in hl)
                            T = (L_sw + L_hl) / cfg["cruise_vel"] + L_tn / cfg["turn_vel"]
                            row.update({f"{mach}_time_min": round(T / 60, 2), f"{mach}_turns": tr.n_turns,
                                        f"{mach}_eta_pct": round(100 * (L_sw + L_hl) / (L_sw + L_hl + L_tn), 2),
                                        f"{mach}_L_total_m": round(L_sw + L_hl + L_tn, 1),
                                        f"{mach}_L_turn_m": round(L_tn, 1),
                                        f"{mach}_off_lane_m": round(WR.off_length(net, tr.pieces(), ctf["gauge"]), 1)})
                            WR.paint(season, tr.pieces("turn"), ctf["gauge"], ctf["tyre_width"] / 2)
                        except Exception as e:
                            row["error"] = (row.get("error", "") + f" {mach}: {type(e).__name__}: {e}")[:300]
                    row["unique_trafficked_season_ha"] = round(season.sum() * ctf["traffic_res"] ** 2 / 1e4, 4)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:300] + " | " + traceback.format_exc()[-200:]
            row["t_ctf_s"] = round(time.perf_counter() - t0, 2)
            cache[layout] = row
            out.append(row)
    os.makedirs(os.path.join(c["out_dir"], "ctf_parts"), exist_ok=True)
    part = os.path.join(c["out_dir"], "ctf_parts", spec["instance_id"] + ".csv")
    with open(part + ".tmp", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CTF_COLS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(out)
    os.replace(part + ".tmp", part)


fb.HOOKS.update(strategies=hook_strategies, post=hook_post)


# ---------------------------------------------------------------------------
def merge(out):
    fb.merge(out)
    os.replace(os.path.join(out, "runs.csv"), os.path.join(out, "sprayer_runs.csv"))
    d = os.path.join(out, "ctf_parts")
    parts = sorted(f for f in os.listdir(d) if f.endswith(".csv")) if os.path.isdir(d) else []
    n = 0
    with open(os.path.join(out, "runs.csv"), "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=CTF_COLS)
        wr.writeheader()
        for f in parts:
            for r in csv.DictReader(open(os.path.join(d, f))):
                wr.writerow(r)
                n += 1
    print(f"merged {len(parts)} instances, {n} CTF rows -> {out}/runs.csv (sprayer rows -> sprayer_runs.csv)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "run", "one", "merge"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--id")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--filter", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "design":
        cfg = dict(fb.CONFIG, b4=True, detours_only=False, geojson_seeds=[0], ctf=cg.CTF_DEFAULTS)
        rows = fb.make_design(cfg)
        with open(os.path.join(a.out, "design.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
        meta = fb.run_meta(cfg)
        meta.update(ctf_version=CTF_VERSION, ctf_definitions={
            "layouts": "shared d_j=(j-floor((n-1)/2))*w_b (primary; sprayer on a base lane; even n: edge lane added on the "
                       "side without a neighbour pass); centred d_j=(j-(n-1)/2)*w_b (sensitivity). d measured along the "
                       "canonical normal of the pass chord.",
            "r_ctf": "max(r_min_seeder, r_min_harvester)",
            "B3c": "rho = r_ctf + d_max; clearance half-width h_env = max(w_s/2, d_max + w_b/2) for every obstacle (class I for "
                   "the lanes); set corners eps = max(m + h_env, rho); hard set obstacle + B(h_env)",
            "B3n": "B3c plus nesting: neighbour pass on the bulge side avoids conv(bump) + B(c_sep), c_sep = max(w_s - w_b + s_min, "
                   "r_ctf + d_max), forced side, iterated to a fixed point; depth = nested passes per obstacle group",
            "variants": "B3c, B3n (s_min), B3n_s12 (s_min_sens), B3n_k2 (B3n with chains stopped after nest_k pushed passes); even n also B3c_ctr, B3n_ctr with the centred-layout d_max",
            "lane_valid": "no cusp (1 - d*kappa <= 0 anywhere), lane simple, min radius >= r_ctf*(1-radius_tol); kappa from tangent "
                          "angle change over curv_window on a curv_ds resampling. *_seeder_*: same with r_min_seeder",
            "lane_clear_tool_min_m": "min over lanes/obstacles dist(lane, obstacle) - w_b/2 (ground-engaging tool, all obstacles); "
                                     "n_clear_fail_lanes counts lanes < 0",
            "lane_clear_wheel_min_m": "min dist(lane, obstacle) - gauge/2",
            "spacing": "nearest other swath lane, sampled every 2 m excluding w_b/2 at lane ends; squeeze < w_b-0.5, gap > w_b+0.5 "
                       "(length, m); lane_squeeze_share_pct = share of sampled lane length with spacing < 0.75 w_b",
            "network": "swath lanes (+B2 corridor lanes) + base-machine headland rings at (i+0.5) w_b filling the k*w_s band of "
                       "the strategy's cells (outer boundary and B1/B2 holes)",
            "n_lane_overlaps": "pairs of lanes within 0.5 m of each other for > 24 m (duplicated tramline)",
            "n_lane_breaks": "tramline stops = round(lane ends farther than k*w_s + w_b from the paddock boundary / 2) + lanes "
                             "with tool clearance < 0 (they must stop at the obstacle)",
            "off_lane": "mean over the two wheels (gauge) of wheel-line length farther than off_lane_tol from network wheel tracks",
            "trafficked": "raster traffic_res, tyre_width footprints; unique_trafficked_ha = sprayer path + headland passes + all "
                          "lane wheel tracks; _season adds the Part-3 seeder and harvester turns",
            "sown": "simulate_coverage with the w_b seeder along all lanes + base headland rings; *_sc_* with seeder_sections",
            "part3": "subset part3_filter, every width; strategies except A; RP_RoutePlannerBase from the gate projected on the "
                     "strategy's mid headland, GLS part3_rp_time s; PP DubinsCC; robot w_b, r_min per machine, same speeds as "
                     "the sprayer; time = (lanes + base headland rings)/cruise + turns/turn_vel; harvester has no SC",
        })
        json.dump(meta, open(os.path.join(a.out, "run_meta.json"), "w"), indent=1)
        print(f"{len(rows)} instances -> {a.out}/design.csv")
        return
    cfg = fb.load_cfg(a.out)
    if a.cmd == "one":
        spec = next(r for r in csv.DictReader(open(os.path.join(a.out, "design.csv"))) if r["instance_id"] == a.id)
        spec["density"] = float(spec["density"]); spec["seed"] = int(spec["seed"])
        try:
            fb.run_instance(cfg, spec, a.out)
        except Exception:
            os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
            open(os.path.join(a.out, "parts", a.id + ".err"), "w").write(traceback.format_exc())
            raise
    elif a.cmd == "run":
        design = list(csv.DictReader(open(os.path.join(a.out, "design.csv"))))
        done = lambda i: os.path.exists(os.path.join(a.out, "ctf_parts", i + ".csv"))
        ctf = cfg["ctf"]
        rows_ = [r for r in design if a.filter in r["instance_id"] and not done(r["instance_id"])]
        # slowest first (Part-3 instances route two machines per strategy), for load balance
        rows_.sort(key=lambda r: (0 if _part3_subset(dict(r, density=float(r["density"])), ctf) else 1))
        todo = [r["instance_id"] for r in rows_]
        # Part-3 instances first-ish is not needed; heavy (w=12, d=2) ones are spread by the design order
        todo = todo[:a.limit] if a.limit else todo
        print(f"{len(todo)} pending instances, {a.workers} workers", flush=True)
        t_start = time.time()

        def job(iid):
            t0 = time.time()
            try:
                p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--out", a.out, "--id", iid],
                                   capture_output=True, text=True, timeout=cfg["instance_timeout_s"] * 2)
                status = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
                if p.returncode != 0 and not os.path.exists(os.path.join(a.out, "parts", iid + ".err")):
                    os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
                    open(os.path.join(a.out, "parts", iid + ".err"), "w").write(p.stderr[-5000:])
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
                os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
                open(os.path.join(a.out, "parts", iid + ".err"), "w").write("timeout")
            return iid, status, time.time() - t0

        with ThreadPoolExecutor(a.workers) as ex:
            for k, (iid, status, dt) in enumerate(ex.map(job, todo), 1):
                el = time.time() - t_start
                print(f"[{k}/{len(todo)}] {iid}: {status} ({dt:.0f}s) | ETA {el / k * (len(todo) - k) / 60:.0f} min", flush=True)
        merge(a.out)
    elif a.cmd == "merge":
        merge(a.out)


if __name__ == "__main__":
    main()
