# Roadmap: computational efficiency of Fields2Cover (and our layer)

_2026-11-17. Based on reading the F2C v2.1.1 source (tag tarball). The `develop` and `v3.0` branches were also checked.
File references are to `src/fields2cover/…` in v2.1.1. Speed-ups are **expected, not measured**. Step 0 measures them._

## TL;DR: ranked opportunities
| # | Where | Problem | Fix | Expected impact | Effort |
|---|---|---|---|---|---|
| 1 | `types/Graph.cpp` `initializeMatrices` (≈l.132–175) | All-pairs shortest paths by **Floyd–Warshall, O(V³)**, over every headland-ring vertex, swath end and split node. Each tree ring adds ~36 vertices, so V reaches thousands with obstacles. | Run Dijkstra (sparse graph) only from the 2N+1 *terminal* nodes (swath ends + depot): O(T·E log V). | **Very large** on paddocks with many obstacles (10–1000×, since V³ ≫ T·E log V). Likely the dominant RP cost. | M |
| 2 | same function, "Reconstruct paths" loop (≈l.162–174) | Walks every i→j path and stores nothing. Pure O(V²·L) waste. | Delete it (keep only the INF fix-up). | Large on big graphs, trivial change | S |
| 3 | `route_planning/route_planner_base.cpp` `computeBestRoute` | The OR-Tools arc-cost callback does **4 hash-map lookups per call** (`getCostFromEdge`: `count`+`at` on nested `unordered_map`). GLS calls it millions of times. | Precompute a dense `std::vector<int64_t>` cost matrix and use `RegisterTransitMatrix` or an indexed lambda. | 2–5× faster local search per second, so better routes within the same time limit | S |
| 4 | `createCoverageGraph` | 3 nodes per swath (start, mid, end) plus 4 edges per swath pair (O(N²) inserts into nested hash maps). | Build the matrix directly from (1) without the `Graph2D` object. Optionally model each swath as 2 nodes with a forced-adjacency constraint. | 2–4× on graph building, smaller routing model | M |
| 5 | `computeBestRoute` search settings | `AUTOMATIC` first solution and a pure time limit (we use 10–30 s per route). | Seed with a boustrophedon or snake order (`ReadAssignmentFromRoutes`). Stop on stagnation (`improvement_limit_parameters`) or a solution limit. Skip OR-Tools for single-cell fields, where boustrophedon is already near-optimal. | Wall-clock per instance 10–30 s → 1–5 s, with similar quality (verify) | S–M |
| 6 | `transformSolutionToRoute` | For each route step it linearly scans all swaths with `Point ==` (O(N²)) and `clone()`s every swath it scans. | Use a hash map from end point to (swath id, orientation). | Small–medium for big N | S |
| 7 | `swath_generator/brute_force.cpp` `computeBestAngle` | Sweeps **2π** in 1° steps (360 angles), but swath cost is direction-symmetric, so π is enough. Each angle rotates the polygon and intersects all lines (GEOS) unless the objective has a "fast" path (only `OBJ_NSwathModified` does). | Sweep [0, π). Search coarse-to-fine (5°, then ±5° at 0.25°). Make sure the build links **TBB**, otherwise `ALLOW_PARALLELIZATION` falls back to serial. | 2× (π) × 3–5× (coarse-to-fine) × cores (TBB) | S |
| 8 | `path_planning/turning_base.h` | Turn discretisation defaults to **0.01 m**, so every turn has hundreds of points. That bloats the `Path`, the SWIG transfers and all downstream shapely work in our Python layer. | Call `setDiscretization(0.1–0.25)` on the turn planner. The turn cache (`using_cache=true`) is already on. | 10–25× fewer path points, which speeds our coverage/clearance metrics a lot | XS |
| 9 | `createShortestGraph` "connect nodes near other edges" | Runs twice, each time over a **copy** of the edge map × all nodes (O(E·V) distance checks). | Use an STRtree / grid spatial index for the node-to-segment checks. | Medium on detailed boundaries | M |
| 10 | Build | Release is the default. No `-march=native` or LTO. TBB is optional. | `-O3 -march=native -flto`, and require TBB on Nectar. | 10–30% overall | XS |

**Our Python layer** (`tools/nb_geom.py`, `tools/bench_geom.py`):
- The SC raster simulator is vectorised numpy and fine at 1 m.
- With (8) its input shrinks a lot.
- Next steps:
  - optional `numba` for the per-pass loop;
  - reuse one planned path for SC on/off (already done);
  - avoid WKT round-trips (`to_shapely`) in hot loops by reading coordinates directly.

**Benchmark throughput on Nectar**: OR-Tools routing is single-threaded per solve, so run one worker process per vCPU with `OMP_NUM_THREADS=1` and TBB limited to 1 thread inside workers. With items 1–5 and a 16-vCPU VM, the ~2–3k-run benchmark should take hours rather than days (estimate).

## Step 0: measure before changing (week 1)
1. **Profiling harness** `bench/perf/` with a fixed set of fields:
   - rectangle 20 ha with 0, 5, 20, 50, 100 trees;
   - L-shape;
   - one real WA boundary.
   Log per-stage time (HG, SG, RP-graph, RP-solve, PP), V and E of the shortest graph, N swaths, and the route cost at 1/5/10/30 s.
2. Profile the C++ with `perf record -g` and the Python with `py-spy`, to confirm that FW (1) dominates RP when obstacles are many.
3. Freeze a **reference output** (route cost, path length) for every field, so optimisations can be regression-tested. Results must be identical for 1, 2, 6 and 8. Items 3 and 5 can change the route found within the time limit, so for those compare the cost distribution.

## Phase 1: quick wins (week 1–2, no API change)
Items 2, 3, 6, 7 (π range + TBB check), 8, 10. Each is a small, separately testable commit on our fork `f2c-fork/perf-*`, validated by the F2C gtest suite (`tests/cpp`, `ctest`) plus the reference outputs.

## Phase 2: algorithmic (week 2–4)
Item 1 (Dijkstra from terminals), item 4 (matrix-direct coverage model), item 5 (seeded search plus a stagnation stop), item 9 (spatial index).
Add gtests that compare Dijkstra and FW costs on random graphs.

## Phase 3: upstream and paper
- Open PRs to Fields2Cover for 1–3, 6–8 (BSD-3; follow `CONTRIBUTING`/code style; one PR per item with benchmark numbers). The maintainers are active: `develop` has had many fixes since 2.1.1.
- Before porting, check the **`v3.0` branch**. It restructures route planning (transit lanes, robot-aware `genRoute` overloads), so prefer patches that apply to both, or target `develop`.
- For the paper: a short "computational performance" subsection or supplementary table (runtime per stage against the number of obstacles, before and after).

## Risks
- Dijkstra and FW tie-breaking can choose different equal-cost paths, so routes may differ while costs stay the same. Compare costs, not geometry.
- Seeding and early stopping trade quality for time. Report both.
- Diverging from upstream: keep the patches small and rebase often.
