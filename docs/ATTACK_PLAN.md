# Attack plan: start point, end point, section control, contour objective

_2026-11-17. Target: a fork of Fields2Cover v2.1.1 (C++ core + SWIG Python), built and tested on the Nectar VM (`docs/nectar_setup.sh`).
Our Python prototypes (`tools/nb_geom.py`, `tools/bench_geom.py`) remain the reference behaviour. Each feature is first prototyped in Python, then moved into C++ where speed or upstreaming matters._

## 0. Ground rules
- **Fork**: `Fields2Cover` → `<your-github>/Fields2Cover`, branch `feat/*` off tag `v2.1.1`. Keep the patches small and rebasable onto `develop` and `v3.0`.
- **Tests first**: each feature gets gtests (`tests/cpp/...`) plus a Python test (`tests/python`). Run with `ctest` on Nectar.
- **The paper drives priorities.** Start/end and section control are in the manuscript now. The contour objective is new: either a second paper or an extension (decide after Phase C).
- **Performance work** (`docs/ROADMAP_performance.md`, item 1 Dijkstra and item 3 dense matrix) touches the same file as features A and B, `route_planner_base.cpp`. Do roadmap Phase 1 first, then A, to avoid merge pain.

---

## A. Separate start point and end point (gate in, gate out)
**Today (v2.1.1)**:
- `setStartAndEndPoint(p)` stores one point. `createCoverageGraph` makes it the single depot.
- `computeBestRoute` builds `RoutingIndexManager(numNodes, 1, depot)` and *assumes the depot is the last node* (`depot_id = numNodes()-1`).
- The edge from the point to the headland graph is a straight line to `closestPointOnBorderTo` and is **not checked against obstacles**.

**Design**:
1. API (backwards compatible):
   - `setStartPoint(p)` and `setEndPoint(q)`;
   - `setStartAndEndPoint(p)` becomes `setStartPoint(p); setEndPoint(p)`;
   - optional `setStartPose(p, heading)` so the first turn leaves the gate with the right heading.
2. Graph:
   - add `start` and `end` as two distinct nodes and record their **indices explicitly** instead of relying on "last node";
   - connect `start` to every swath end (and every swath end to `end`) with the headland shortest-path cost;
   - if only one of the two is given, keep the dummy zero-cost depot for the other side.
3. OR-Tools: use the multi-depot constructor `RoutingIndexManager(n, 1, {start_idx}, {end_idx})`. The start→first-swath and last-swath→end legs then sit in the objective, as in Eq. (route) of the paper.
4. Access legs: connect `p` to the headland through an **obstacle-aware** link, i.e. the shortest path inside `field ∪ access corridor` that avoids the obstacle buffers. Our Python version projects onto the outer headland track. In C++, add the link edge only if `LineString(p, proj)` does not cross any interior ring, otherwise go through the nearest ring vertex chain.
5. `transformSolutionToRoute`: prepend `shortestPath(start, first)` and append `shortestPath(last, end)`.
6. Tests:
   - start = end gives the same cost as before;
   - start ≠ end on a rectangle checks that an open route ends at the right corner;
   - an obstacle between the gate and the headland must not be crossed.
7. Paper/benchmark: add a factor "same gate vs entry ≠ exit" to the gate-effect subset.

**Effort**: ~3–5 days, including tests. **Risk**: low.

---

## B. Section control (planning and evaluation)
**Today**:
- F2C assumes full width. `Swath` has a single `width_`, and `PathState` holds `{point, angle, len, dir, type, velocity}` with no implement state.
- `OBJ_Overlaps` and `OBJ_FieldCoverage` exist for swath generation but know nothing about sections.
- Our Python raster simulator (`simulate_coverage`) gives M, O, off-target area and the saving S for any path.

**Design in three layers**:
1. **Evaluation (C++ port of our simulator)**:
   - `f2c::sc::SectionModel{n_sections, widths, look_ahead_m, look_behind_m, on_threshold}`;
   - `f2c::sc::simulate(path, field, obstacles, model, res) → {count_raster, M, O, A_off, applied_area, section_masks}`.
   Also add `std::vector<bool> sections` to a parallel per-`PathState` structure (not to `PathState` itself, to keep the ABI), so the section ON/OFF schedule travels with the path and can be exported.
   Model switching **latency** with look-ahead/behind distances (cite `sharda2011nozzle`).
2. **Planning with SC**:
   - (a) **SC-aware strategy choice**: the cost C = c_t·T + c_p·applied + c_m·missed (paper Eq. 9, request R4) ranks A/B1/B2/B3. This is done in our Python layer.
   - (b) **SC-aware swath objective**: `OBJ_SCAware(c_t, c_p, model)`. For an angle θ its cost is the turn time plus the *product* that the generated swaths would apply under SC, computed from the geometric overlap between swaths and with the headland passes. It plugs into `SG_BruteForce` like any `SGObjective` via `computeCost(const F2CCell&, const F2CSwaths&)`.
   - (c) **Partial last swath and drive-around overlap**: with SC, overlap of the last swath (`allow_overlap=true`) and B3 detour overlap become cheap. The planner may then prefer fewer, overlapping swaths over an extra pass. Evaluate this in the benchmark.
3. **Export**: write the section schedule per path point (CSV/GeoJSON first). ISOXML TaskData (`TIM`/`TLG` or a prescription with section states) is optional future work.

**Steps**:
1. Freeze the Python simulator as the test oracle.
2. Port it to C++ with gtests that compare against the oracle on 5 fields (tolerance 0.1 % of area).
3. Write `OBJ_SCAware`.
4. Add a benchmark factor for n_sections (1, 6, 12, 18, 48).
5. Cite and compare with `plessen2025multisection` and `luck2011fieldshape`.

**Effort**: evaluation port ~1 week; objective and benchmark ~1 week. **Risk**: medium. The raster resolution trades accuracy against time, so the C++ port should probably use polygon clipping (GEOS) for exact areas, with the raster only for maps.

---

## C. Contour objective
**Interpretation (to confirm)**: swaths that follow **terrain contours**, so the machine works across the slope. This reduces erosion and runoff and lets a broadacre machine work level. F2C has no elevation support beyond `Point::getZ()`, and no objective uses slope.
_Alternative reading_: "contour" = concentric swaths that follow the field boundary (a headland-style spiral). See C3.

- **C1: straight swaths with a slope objective (do first; small).**
  New `OBJ_SlopeAlongTrack(DEM)`. For direction θ with unit vector u(θ), cost(θ) = ∫_F |∇z · u(θ)| dA, i.e. the mean along-track slope. It is independent of the swath geometry, so `isFastCompAvailable()` returns true and the 1° sweep costs O(raster) once, then O(1) per angle via a precomputed gradient histogram.
  - Combined objective: J(θ) = α·N_swaths(θ) + β·slope(θ), where α and β trade turns against erosion.
  - Inputs: a DEM GeoTIFF read through GDAL, which F2C already depends on. Data source for WA paddocks: the national 5 m LiDAR DEM from ELVIS / Geoscience Australia.
- **C2: curved contour-following swaths (larger; a novel contribution).**
  New `SG_Contour` generator:
  1. pick a *key contour* (e.g. the longest contour through the cell at its median elevation, or the keyline);
  2. generate swaths as successive **parallel offsets** (GEOS `offsetCurve`) at w_c spacing on both sides;
  3. clip to the cell;
  4. re-key when the offset drifts from the true contour by more than a tolerance (contours are not parallel on uneven ground).
  The route planner already treats swaths as linestrings. Path planning must respect curvature: smooth the offsets so that κ ≤ 1/r_min, and flag violations.
  Metrics: mean |slope along track|, swath curvature, overlap/miss between non-parallel neighbours (**section control matters here**, because the wedges between offsets are exactly where SC saves product, which links C2 to B).
- **C3 (alternative reading): concentric or boundary-following swaths.** F2C has `RP_Spiral`, but that is only a route *order* over straight swaths. A concentric generator means repeated inward offsets of the boundary (like the headland generator iterated to the centre). It is easy to prototype in Python with shapely `buffer(-k·w_c)`.

**Effort**: C1 ~3–4 days; C2 ~3–4 weeks (research-grade); C3 ~1 week (Python prototype). **Risk**: C2 is high, because of offset degeneracies and curvature on rough terrain.

---

## D. Sequencing (proposed)
| Week | Work | Output |
|---|---|---|
| 1 | Nectar VM up, fork, roadmap Step 0 (profiling harness) + Phase 1 quick wins | Baseline timings, faster F2C build |
| 2 | **A** start/end + roadmap item 1 (Dijkstra) in `route_planner_base.cpp` | PR-ready branch, gtests |
| 2–3 | Run the paper benchmark on Nectar (parallel workers) | `results/benchmark_v1/runs.csv`, which unblocks the paper Results |
| 3–4 | **B** SC evaluation in C++ + `OBJ_SCAware` | SC-aware planning in the paper (Methods 3.6) |
| 5 | **C1** slope objective + DEM input | Demo on a WA paddock with LiDAR DEM |
| 6+ | **C2** contour swath generator (or C3, after confirmation) | Second paper / extension |

## E. Coordination
- The code is owned by the `plan-v2-tree-path-optimisation` session (or by whoever runs on Nectar). The paper session tracks needs in `paper/results_needed.md`.
- Each feature PR updates `docs/HANDOVER.md` §4 "State of play".
