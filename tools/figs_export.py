"""Appendix figure export (paper request, 2026-11-18): native F2C vs the extended workflow, top view.

One GeoJSON + coverage npz per instance, same schema as benchmark_v1_2/geojson (f2c_bench.write_geojson).
Strategies:
  A, B1, B2, B3, B4      extended workflow (f2c_bench.run_instance, config of native_v1/run_meta.json)
  N_sort  decomp none          + boustrophedon order + DubinsCC
  N_rp    decomp none          + rp_gls              + DubinsCC
  N_dec   decomp boustrophedon + rp_gls              + DubinsCC
  N_trap  decomp trapezoidal   + rp_gls              + DubinsCC
All native plans use sg_obj = NSwath. Plans are re-generated (OR-Tools GLS has a wall-clock limit, so
routes can differ slightly from the rows in native_v1 / benchmark_v1_2); each plan's metric rows are
written to <out>/parts (extended) and <out>/native_parts (native) so the figure can quote its own numbers.

    python figs_export.py --out ../results/figs_v1 --meta ../results/native_v1/run_meta.json \
        L_mixed_poisson_d1_w12_road_s0 L_mixed_poisson_d1_w24_road_s0 rectangle_mixed_poisson_d1_w12_road_s3
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f2c_bench as fb  # noqa: E402
import native_bench as nb  # noqa: E402

NATIVE = {"N_sort": ("none", "boustrophedon"), "N_rp": ("none", "rp_gls"),
          "N_dec": ("boustrophedon", "rp_gls"), "N_trap": ("trapezoidal", "rp_gls")}
SG_OBJ, TURN = "NSwath", "DubinsCC"


def parse_id(iid):
    m = re.fullmatch(r"(\w+?)_(\w+?)_(\w+?)_d([\d.]+)_w(\d+)_(road|random)_s(\d+)", iid)
    shape, ot, pat, d, w, gate, seed = m.groups()
    return dict(instance_id=iid, shape=shape, otype=ot, pattern=pat, density=float(d), width=int(w),
                gate_type=gate, seed=int(seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    cfg, nc = meta["config"], dict(meta["native_config"])
    cfg["widths"] = {int(k): v for k, v in cfg["widths"].items()}
    cfg.update(b4=True, detours_only=False)
    nc["turn_models"] = [TURN]
    os.makedirs(a.out, exist_ok=True)

    captured = {}
    write_geojson = fb.write_geojson
    fb.write_geojson = lambda stem, *args: captured.update(args=args)

    for iid in a.ids:
        spec = parse_id(iid)
        spec["n_sections"] = cfg["widths"][spec["width"]]
        cfg["geojson_seeds"] = [spec["seed"]]
        captured.clear()
        fb.run_instance(cfg, spec, a.out)
        field, obs, zones, gate, geo, w = captured["args"]
        for label, (dec, order) in NATIVE.items():
            u = dict(spec, unit_id=f"{iid}__{dec}__{SG_OBJ}__{order}", decomp=dec, sg_obj=SG_OBJ, route_order=order)
            cap = {}
            nb.run_unit(cfg, nc, u, os.path.join(a.out, "native_parts"), capture=cap)
            if TURN in cap:
                geo[label] = cap[TURN]
            else:
                print(f"  {iid} {label}: no plan (see native_parts/{u['unit_id']}.csv)", flush=True)
        write_geojson(os.path.join(a.out, iid), field, obs, zones, gate, geo, w)
        print(f"{iid}: {sorted(geo)} -> {a.out}/{iid}.geojson", flush=True)
    json.dump(dict(meta, figs_export=dict(native=NATIVE, sg_obj=SG_OBJ, turn_model=TURN)),
              open(os.path.join(a.out, "run_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
