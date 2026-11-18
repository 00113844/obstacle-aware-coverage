"""Combine the per-plan result rows of the figure export (results/figs_v1) into one CSV with strategy labels
matching the GeoJSON (A, B1..B4, N_sort, N_rp, N_dec, N_trap), so fig_topview.py quotes these exact plans.

    python combine_figs_runs.py ../../results/figs_v1
"""
import glob
import os
import sys

import pandas as pd

LAB = {("none", "boustrophedon"): "N_sort", ("none", "rp_gls"): "N_rp",
       ("boustrophedon", "rp_gls"): "N_dec", ("trapezoidal", "rp_gls"): "N_trap"}

root = sys.argv[1]
ext = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(root, "parts", "*.csv"))], ignore_index=True)
nat = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(root, "native_parts", "parts", "*.csv"))], ignore_index=True)
nat = nat[nat.turn_model.astype(str).str.replace(" ", "").eq("DubinsCC")]
nat["strategy"] = [LAB.get((d, o)) for d, o in zip(nat.decomp, nat.route_order)]
nat = nat.dropna(subset=["strategy"])
cols = ["instance_id", "strategy", "sc", "feasible", "time_min", "missed_pct", "overlap_pct", "over_crop_m"]
out = pd.concat([ext[cols[:-1]].assign(over_crop_m=ext.get("over_crop_m")), nat[cols]], ignore_index=True)
out.to_csv(os.path.join(root, "figs_runs.csv"), index=False)
print(out[out.sc == 1].sort_values(["instance_id", "strategy"]).to_string(index=False))
