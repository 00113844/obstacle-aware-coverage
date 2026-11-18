# obstacle-aware-coverage

Code, benchmark and data for the article

> Togeiro de Alckmin, G. *Drive around or cut holes? Obstacle-aware coverage paths and controlled-traffic tramlines for paddocks with scattered trees and rock outcrops.* Submitted to *Precision Agriculture*.

The repository extends the open-source coverage planner [Fields2Cover](https://github.com/Fields2Cover/Fields2Cover) (F2C, v2.1.1) for paddocks with scattered in-field obstacles. It adds:

- **gate-anchored routing**: the route starts and ends at a gate on the access track;
- an **obstacle model** for trees (canopy circles) and rock outcrops (polygons), with a per-machine passability class;
- four **obstacle strategies**: holes (B1), Boustrophedon decomposition (B2), curvature-bounded **drive-around** detours (B3) and **straddling** of low obstacles (B4);
- a raster **section-control** model for missed area, overlap, off-target application and product saving;
- a **controlled-traffic (CTF) tramline model**: base lanes derived from the sprayer plan, CTF-compatible detours (B3c, B3n) and seeder and harvester routes constrained to the lanes.

Archived versions are on Zenodo (see the DOI badge of the release). Please cite the archived version (`CITATION.cff`).

## Contents

| Path | What |
|---|---|
| `tools/f2c_bench.py` | Benchmark runner for E1 (B1–B3) and E3 (B4): synthetic paddocks, planning, section-control simulation |
| `tools/native_bench.py` | E2: native F2C configurations (decompositions, swath objectives, route orders, turn models) |
| `tools/ctf_bench.py`, `tools/ctf_geom.py` | E4: CTF tramline layouts, B3c/B3n, constrained seeder and harvester routes |
| `tools/bench_geom.py`, `tools/nb_geom.py` | Pure numpy/shapely geometry: paddocks, obstacle patterns, detours, clearance, coverage raster |
| `tools/figs_export.py` | Plan-view exports used for the appendix figures |
| `tools/test_*.py` | Tests of the geometry modules (run without F2C) |
| `analysis/` | Scripts that regenerate every number, table and figure of the article from `data/` |
| `data/` | Per-run results of all experiments (CSV) with their configurations (`run_meta.json`) |
| `Fields2Cover_Paddock_Trees.ipynb`, `handout/` | Google Colab tutorial and its LaTeX handout (gate, drive-around, section control on one paddock) |
| `Benchmark_Runner.ipynb` | Colab runner for the benchmark |
| `docs/` | Project notes, F2C performance notes and the VM setup script |

## Data

| Folder | Experiment | Rows |
|---|---|---|
| `data/benchmark_v1_2/` | E1: 285 layouts × 3 widths, strategies A, B1–B3, SC on/off | one row per instance × strategy × SC state |
| `data/native_v1/` | E2: 180 native F2C configurations against the extended workflow | one row per instance × configuration × SC state |
| `data/b4_v1/` | E3: B3 and B4 on the paddocks with rock outcrops | one row per instance × strategy × SC state |
| `data/ctf_v1/` | E4: tramline layouts per module system (1:1, 2:1, 3:1) and strategy; stage-2 seeder and harvester routes | one row per instance × strategy × lane layout |
| `data/figs_v1/` | Geometries and coverage rasters of the plans shown in the appendix figures | |

Column definitions are in `COLUMNS` (`tools/f2c_bench.py`) and `NCOLS` (`tools/native_bench.py`), and the E4 definitions are in `data/ctf_v1/run_meta.json` (`ctf_definitions`).

## Reproduce the analysis

Requires Python ≥ 3.10. With [uv](https://docs.astral.sh/uv/):

```
uv run --no-project --with-requirements analysis/requirements.txt python analysis/analyse_benchmark.py data/benchmark_v1_2 --out generated
uv run --no-project --with-requirements analysis/requirements.txt python analysis/figures.py          data/benchmark_v1_2 --out generated
uv run --no-project --with-requirements analysis/requirements.txt python analysis/analyse_e2_b4.py    data --out generated
uv run --no-project --with-requirements analysis/requirements.txt python analysis/sensitivity_cost.py data --out generated
uv run --no-project --with-requirements analysis/requirements.txt python analysis/analyse_e4.py       data/ctf_v1 --out generated
```

Each script writes LaTeX macros (`numbers*.tex`), tables (`tab_*.tex`) and figures (`fig_*.pdf`).

## Re-run the benchmark

The planners need Fields2Cover v2.1.1 built with its Python bindings and OR-Tools; `docs/nectar_setup.sh` installs both on Ubuntu 22.04. Then, for example:

```
cd tools
python f2c_bench.py design --out ../results/benchmark_v1_2
python f2c_bench.py run    --out ../results/benchmark_v1_2 --workers 8
python f2c_bench.py merge  --out ../results/benchmark_v1_2
```

`native_bench.py` and `ctf_bench.py` follow the same `design` / `run` / `merge` pattern (see their docstrings). The routing search stops at a wall-clock limit, so re-runs reproduce the results statistically, not bit for bit.

## Tutorial notebook (Google Colab)

Open `Fields2Cover_Paddock_Trees.ipynb` in Google Colab and run the **Part 0** cell, which builds Fields2Cover v2.1.1 and OR-Tools in the Colab VM (10–15 min the first time). Then choose *Runtime → Run all*. The handout (`handout/f2c_tutorial.pdf`) explains each step. The notebook is generated by `tools/build_notebook.py`; `tools/check_nb.py` validates it.

## Licence

BSD 3-Clause (see `LICENSE`). Fields2Cover is also BSD 3-Clause.

## Acknowledgements

This research was supported by use of the Nectar Research Cloud, a collaborative Australian research platform supported by the NCRIS-funded Australian Research Data Commons (ARDC). The simulation experiments were run on a Nectar Research Cloud virtual machine.
