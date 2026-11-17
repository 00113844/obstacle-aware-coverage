# Handover: Fields2Cover obstacle/gate/section-control project

_Last updated 2026-11-17. Read this first if you are a person or an agent picking up the work, especially on the Nectar VM._

## 1. Goal
Extend [Fields2Cover](https://github.com/Fields2Cover/Fields2Cover) (F2C v2.1.1, BSD-3) for real paddocks:

- **Gate**: a fixed start point and a separate end point (the end point is new; F2C allows only one start=end depot).
- **Obstacles** (trees, rock outcrops). Three strategies are compared: B1 holes, B2 Boustrophedon decomposition, B3 drive-around detours.
- **Section control (SC)**: partial implement width, used both to *evaluate* a path and to *choose* between paths.
- **Contour objective**: swaths that follow terrain contours (see `docs/ATTACK_PLAN.md` §4).

Then publish this as an original research article in *Precision Agriculture* (Springer), with a synthetic factorial benchmark.

## 2. Who does what
| Stream | Owner | Location |
|---|---|---|
| Code: notebook, geometry, benchmark | Claude session `plan-v2-tree-path-optimisation` | `tools/`, `Fields2Cover_Paddock_Trees.ipynb` |
| Paper | Claude session `obstacle-aware-coverage-paper` | `paper/` |
| Running F2C (Colab today, Nectar next) | Gustavo | Colab / Nectar VM |

The paper session only **reads** code outputs. Requests from the paper to the code are tracked in `paper/results_needed.md` (R1–R7).

## 3. Repository map
| Path | What |
|---|---|
| `Fields2Cover_Paddock_Trees.ipynb` | Colab tutorial notebook (generated, do not hand-edit) |
| `tools/build_notebook.py` | Notebook generator. Part 0 holds the F2C build recipe. |
| `tools/nb_geom.py` | Pure numpy/shapely: `Track`, B3 circle detours, raster SC simulator `simulate_coverage` |
| `tools/bench_geom.py` | Benchmark geometry: synthetic paddocks, Poisson/Thomas obstacles, convex-set detours (polygons), clearance |
| `tools/test_*.py` | Local tests (no F2C needed) |
| `tools/f2c_bench.py` | _in progress_ (headless benchmark runner, CSV/GeoJSON) |
| `handout/` | LaTeX tutorial handout |
| `paper/` | Manuscript (sn-jnl, author-year). Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`. |
| `paper/refs.bib`, `paper/lit_matrix.md` | 48 verified references and the novelty analysis |
| `docs/` | This handover, `nectar_setup.sh`, `ROADMAP_performance.md` (includes the F2C source analysis), `ATTACK_PLAN.md` |

## 4. State of play
- **Done**
  - Gate projected onto the headland and passed to `setStartAndEndPoint`. In F2C, the gate legs are part of the OR-Tools cost (paper R7).
  - Auto headland width from the Ω-turn depth.
  - Strategies B1/B2/B3 (B3 for circles and convex polygons).
  - Raster SC simulator.
  - Paper: Methods, Introduction and Related Work drafted; the manuscript compiles cleanly.
- **In progress**: the headless benchmark runner (R2/R3/R5) and the SC-aware selection cost C = c_t·T + c_p·applied + c_m·missed (R4).
- **Blocked**: F2C cannot run on the Windows laptop (no WSL or Docker), so everything that calls F2C ran on Colab. **The Nectar VM removes this block.**

## 5. Getting going on Nectar
1. Launch an **Ubuntu 22.04** instance. This matches the OR-Tools v9.9 tarball.
   - Suggested size: at least 16 vCPUs and at least 32 GB RAM. The benchmark parallelises across instances, so cores matter most.
   - Add a volume for `results/`.
   - Open SSH in the security group.
2. Copy the repository over (`scp -r` or `git`; the project is not yet a git repo, so `git init` it first, see §7).
3. `bash docs/nectar_setup.sh` builds OR-Tools, the F2C wheel and a venv in about 10–15 min, then runs a smoke test.
   The F2C source is kept in `~/f2c-build/Fields2Cover`, so it can be patched and rebuilt for the features in `ATTACK_PLAN.md`.
4. Run the benchmark inside `tmux`, one worker process per core, with `OMP_NUM_THREADS=1`. OR-Tools routing is single-threaded per solve, so parallelism comes from instances.
   `runs.csv` is appended row by row and the runner resumes after interruption.
5. Optional: install Claude Code on the VM (`npm i -g @anthropic-ai/claude-code`, Node ≥ 18) so that a session can run and debug F2C natively.

## 6. Known pitfalls
- **Python bindings**: never call `.at(i)` / `.getState(i)` on a temporary (e.g. `x.flatten().at(0)`). It leaves a dangling pointer and segfaults the kernel. Always `.clone()` or bind to a variable first.
- When `w_c < 2 r_min`, forward-only turns become Ω-loops. The planner tries Reeds–Shepp turns (reversing) and picks the faster one.
- The OR-Tools routing cost has **no turn cost**: turns are planned after routing.
- `DECOMP_Boustrophedon` can fail on some geometries. B2 is wrapped in try/except and flagged in the CSV.
- Real paddocks: plan in UTM (`f2c.Transform.transformToUTM`), then transform back.
- Drive file IDs for the Colab notebook and handout are in the agent memory (`f2c-tutorial-colab-drive`). The Drive connector cannot replace file content, so a new version means a new file.

## 7. Recommended housekeeping before scaling up
- Turn the folder into a git repository with a remote (GitHub or UWA GitLab), so that the laptop, Colab and Nectar share one source of truth. Fork F2C for the C++ changes.
- Pin versions: F2C tag or commit, OR-Tools 9.9, Python package versions (`pip freeze > requirements.lock`).
- Archive the benchmark outputs and code on Zenodo at submission (this is required by the paper's data and code availability statements).

## 8. Next documents
- `docs/ROADMAP_performance.md`: how to make F2C (and our layer) faster.
- `docs/ATTACK_PLAN.md`: how to add start/end points, section control and the contour objective.
