"""Generate Benchmark_Runner.ipynb: a Colab driver for f2c_bench.py (R2-R6).

Run:  python build_bench_notebook.py <output.ipynb>
The four modules are embedded with %%writefile, so the notebook is self-contained.
"""
import json
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
md = lambda s: cells.append(("markdown", textwrap.dedent(s).strip("\n")))
code = lambda s: cells.append(("code", textwrap.dedent(s).strip("\n")))

md(r"""
# Obstacle-aware coverage benchmark — runner (benchmark_v1_2, f2c_bench 1.4)

This notebook runs `f2c_bench.py`, a headless batch runner that plans strategies **A, B1, B2 and B3** with Fields2Cover over the factorial design.
It writes one CSV row per (instance, strategy, SC) and GeoJSON/coverage dumps for the seed-0 instances.

* Results go to **Google Drive** (`MyDrive/F2C_benchmark/benchmark_v1_2`), so they survive a disconnect. **Re-run the "run" cell to resume**: finished instances are skipped.
* Each instance is planned in its own subprocess, so a crash inside Fields2Cover only loses that instance (see `parts/*.err`).
* Expected cost is ≈2.5k route plans at a 10 s OR-Tools budget, i.e. **≈5–6 h with 2 workers**. Colab may disconnect, so plan on 1–2 resumes.

The same `.py` files also run headless on a Linux VM (see the last cell).
""")

code(r'''
# ---- 1. Install Fields2Cover (same recipe as the tutorial, idempotent) ----
import importlib.util, os, subprocess, time, glob
ORTOOLS_URL = ("https://github.com/google/or-tools/releases/download/v9.9/"
               "or-tools_amd64_ubuntu-22.04_cpp_v9.9.3963.tar.gz")
def sh(cmd, env=None):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, env={**os.environ, **(env or {})})
    if r.returncode: print(r.stdout[-3000:], r.stderr[-3000:]); raise RuntimeError(cmd)
if not os.path.exists("/opt/or-tools/lib/cmake/ortools"):
    sh(f"wget -q -O /tmp/ortools.tgz {ORTOOLS_URL} && mkdir -p /opt/or-tools && "
       "tar -xzf /tmp/ortools.tgz -C /opt/or-tools --strip-components=1 && "
       "echo /opt/or-tools/lib > /etc/ld.so.conf.d/ortools.conf && ldconfig")
if importlib.util.find_spec("fields2cover") is None:
    t0 = time.time()
    sh("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y build-essential cmake git "
       "libgdal-dev libgeos-dev libeigen3-dev libtbb-dev libtinyxml2-dev nlohmann-json3-dev libboost-dev")
    wheels = sorted(glob.glob("/content/drive/MyDrive/F2C_cache/fields2cover-*.whl"))
    if not wheels:
        sh("pip wheel --no-deps -w /content/f2c_wheels 'git+https://github.com/Fields2Cover/Fields2Cover.git@v2.1.1'",
           env={"CMAKE_PREFIX_PATH": "/opt/or-tools", "CMAKE_BUILD_PARALLEL_LEVEL": str(os.cpu_count())})
        wheels = sorted(glob.glob("/content/f2c_wheels/fields2cover-*.whl"))
    sh(f"pip install -q --no-deps {wheels[-1]}")
    print(f"installed in {(time.time()-t0)/60:.1f} min")
import fields2cover as f2c
print("Fields2Cover", getattr(f2c, "__version__", "?"))
''')

code(r'''
# ---- 2. Google Drive for persistent results ----
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/F2C_benchmark/benchmark_v1_2"
os.makedirs(OUT, exist_ok=True)
os.makedirs("/content/bench", exist_ok=True)
%cd /content/bench
''')

md("## Benchmark code (written to `/content/bench`)")
for f in ["nb_geom.py", "bench_geom.py", "f2c_bench.py", "bench_figures.py"]:
    with open(os.path.join(HERE, f), encoding="utf-8") as fh:
        code(f"%%writefile /content/bench/{f}\n" + fh.read())

md(r"""
## Design, pilot, full run
`design` writes `design.csv` and `run_meta.json`. The meta file holds the full CONFIG (including the cost weights c_t, c_p and c_m), the package
versions and the source hashes, which makes the run reproducible. The CONFIG is read from `run_meta.json` on every resume, so **do not edit it mid-run**.
""")

code(r'''
import json
if not os.path.exists(f"{OUT}/design.csv"):
    subprocess.run(["python", "f2c_bench.py", "design", "--out", OUT], check=True)
meta = json.load(open(f"{OUT}/run_meta.json"))
print({k: meta[k] for k in ("bench_version", "fields2cover", "created")}, "| costs:",
      {k: meta["config"][k] for k in ("c_t", "c_p", "c_m")})
''')

code(r'''
# Pilot: 4 instances (about 3-5 min). Check runs.csv before starting the full run.
!python f2c_bench.py run --out "{OUT}" --workers 2 --limit 4 --filter "_s0"
import pandas as pd
df = pd.read_csv(f"{OUT}/runs.csv"); display(df.head(12)); print(df.shape)
!ls "{OUT}/parts" | grep err || echo "no errors"
''')

code(r'''
# Full run (resumable). If Colab disconnects, re-run cells 1, 2, the %%writefile cells and this one.
!python f2c_bench.py run --out "{OUT}" --workers 2
''')

code(r'''
# Merge + quick summary
!python f2c_bench.py merge --out "{OUT}"
df = pd.read_csv(f"{OUT}/runs.csv")
print(df.shape, "instances:", df.instance_id.nunique())
display(df[df.sc == 1].groupby(["strategy"])[["time_min", "missed_pct", "overlap_pct", "product_saved_pct", "crit_C"]].median())
''')

code(r'''
# R6 figures for one illustrative instance (vector PDF, 174 mm, embedded fonts)
inst = "rectangle_mixed_thomas_d1_w24_road_s0"
!python bench_figures.py "{OUT}/geojson/{inst}.geojson" --out "{OUT}/figures/{inst}"
!ls "{OUT}/figures/{inst}"
''')

md(r"""
## Headless on a Linux VM (e.g. Nectar)
```bash
bash docs/nectar_setup.sh                       # builds OR-Tools + F2C into a venv
source ~/f2c-venv/bin/activate                  # (the path printed by the setup script)
cd tools
python f2c_bench.py design --out ../results/benchmark_v1
nohup python f2c_bench.py run --out ../results/benchmark_v1 --workers $(nproc) > ../results/benchmark_v1/run.log 2>&1 &
python bench_figures.py ../results/benchmark_v1/geojson/<instance>.geojson --out ../results/benchmark_v1/figures
```
""")


def to_lines(s):
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": [], "name": "Benchmark_Runner.ipynb"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "cells": []}
for i, (kind, src) in enumerate(cells):
    c = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": to_lines(src)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)
out = sys.argv[1] if len(sys.argv) > 1 else "Benchmark_Runner.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"wrote {out} with {len(cells)} cells")
