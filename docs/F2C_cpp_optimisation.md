# Fields2Cover C++ core: code-level optimisation guide

_2026-11-17. Source reviewed: Fields2Cover **v2.1.1** (tag tarball), C++ core under `include/fields2cover/` and `src/fields2cover/`.
Build checked on the Nectar VM: Ubuntu 22.04, GDAL 3.4, GEOS 3.10.2, OR-Tools 9.9.
Companion to `docs/ROADMAP_performance.md`, which covers the algorithm and module level. This file goes down to classes, allocations and inner loops, and gives step-by-step instructions._

> **Status of the claims.** Every finding below was read in the source and cites a file and function. The **speed-ups are estimates** until measured with the protocol in §1. Items marked **VERIFY** depend on library internals (GDAL/GEOS) that were not inspected.

---

## 0. Summary

| ID | Area | Finding | Fix | Expected gain | Effort | Risk |
|---|---|---|---|---|---|---|
| **C1** | `types/Graph.cpp` | Floyd–Warshall, O(V³) time and **2 × V² int64 memory** in `vector<vector>` (V = 5000 needs ≈ 400 MB). A dead path-reconstruction loop follows it. | Dijkstra from terminal nodes only, flat storage, delete the dead loop | Very large for fields with many obstacles | M | Low |
| **C2** | `objectives/sg_obj/n_swath_modified.cpp` | The 360-angle sweep evaluates a function whose **exact minimum is at an edge direction**. Each evaluation allocates three `Point`s per vertex. | Evaluate only the n edge directions, on precomputed `(L_i, φ_i)` | 10–100× for SG; exact instead of 1° | S | Low |
| **C3** | Build | `ALLOW_PARALLELIZATION=ON`, but the built module **does not link TBB** (`ldd` on Nectar), so `std::execution::par_unseq` runs serially | Make CMake find oneTBB, or replace with OpenMP | ≈ cores × SG sweep | XS | Low |
| **C4** | `types/Point.*`, `Geometry.h` | Every `Point` is a heap `OGRPoint` behind a `shared_ptr` with a custom deleter (2 allocations). Its copy constructor deep-clones (2 more allocations). `operator->()` returns `shared_ptr` **by value** (2 atomic ops per call). | Return `T*` from `operator->`. Add a POD fast path (`Vec2`) for hot loops. | 1.5–3× on geometry-heavy loops | S (operator->) / L (POD) | Low / Medium |
| **C5** | `types/Geometries_impl.hpp` | The non-const iterator heap-allocates `Private`, copies each child on deref, and on every `++` and destructor **re-fetches and compares** (`update()`) to write back changes | Const iteration in library code, or index loops over raw OGR arrays | 2–5× on ring and line loops | S | Low |
| **C6** | `swath_generator_base.cpp` `generateSwaths` | Per angle it rotates the whole polygon (a new `Point` per vertex, `sin`/`cos` per vertex), builds lines, rotates them back, then does a GEOS `Intersection` (OGR→GEOS conversion of the polygon each time) | Analytic scanline clipping of parallel lines against the edge list | 5–20× per angle for swath-based objectives | M | Medium |
| **C7** | `route_planner_base.cpp` `computeBestRoute` | The OR-Tools arc callback does 4 hash lookups per call | Dense `vector<int64_t>` matrix, indexed lambda | 2–5× more local-search moves per second | S | Low |
| **C8** | `createCoverageGraph` | 4·N² `addEdge` calls into `unordered_map<size_t, unordered_map<size_t,int64>>`. Each call hashes two `Point`s. | Fill the dense matrix directly from the C1 distances | 2–4× graph building | S | Low |
| **C9** | `createShortestGraph` | Edge map **copied** twice. O(E·V) GEOS/`Point` distance checks. `getCellWherePoint` does a GEOS `distance` per cell per swath end. | Spatial grid/STRtree; pass the owning cell index with each swath | Medium on detailed boundaries | M | Low |
| **C10** | `transformSolutionToRoute` | O(route × N) linear search with `Point ==`, cloning each scanned swath | `unordered_map<endpoint, (id, dir)>` | Small–medium | XS | Low |
| **C11** | `types/Path.cpp` | `operator+=` appends without `reserve`. Each `PathState` holds a heap `Point`. The default turn discretisation is 0.01 m. | `reserve`. Coarser discretisation (we already use 0.2 m). Store `PathState` as POD. | Memory and SWIG transfer | XS–M | Low |
| **C12** | `LineString::closestPointTo`, `Cell(s)::closestPointOnBorderTo` | Build `vector<Point>` and `vector<double>` of every candidate (allocations), then take the argmin | Running argmin on raw coordinates | Small, but called per swath end | XS | Low |

**Correctness issues found while reading** (fix alongside the above; they interact with hashing and sorting):
- **K1.** `Point::operator<` compares exactly, but `operator==` uses a 1e-7 tolerance, so `<` is not a strict weak ordering consistent with `==`. `Swaths::sort` and `std::sort` rely on it.
- **K2.** `std::hash<Point>` rounds to 1e-7 with `llround`. Two points within 1e-7 that straddle a rounding boundary compare equal but hash differently, so a `Graph2D` lookup can create duplicate nodes (rare, but silent).
- **K3.** `computeBestRoute` assumes the depot is node `numNodes()-1`. This holds only because the depot is inserted last. Make it explicit; it is also needed for separate start and end points (`ATTACK_PLAN.md` §A).

---

## 1. Measurement protocol (do this first)

1. **Fork and branch.** Fork `Fields2Cover/Fields2Cover`, branch `perf/baseline` from `v2.1.1`. Build on Nectar:
   ```bash
   cd ~/f2c-build/Fields2Cover
   cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_PYTHON=ON -DBUILD_TESTING=ON \
         -DCMAKE_PREFIX_PATH=/opt/or-tools -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer"
   cmake --build build -j$(nproc) && (cd build && ctest --output-on-failure)
   ```
2. **Micro-benchmarks.** Add `bench/` using Google Benchmark (`apt install libbenchmark-dev`) with these fixtures:
   - `rect20ha`;
   - `rect20ha_trees{5,20,50,100}` (circles with 36 vertices);
   - `Lshape20ha`;
   - one digitised WA boundary with more than 500 vertices.

   Benchmarks:
   - `BM_SG_bestAngle_NSwathModified`;
   - `BM_SG_bestAngle_SwathLength`;
   - `BM_SG_generateSwaths`;
   - `BM_RP_shortestGraph`;
   - `BM_RP_coverageGraph`;
   - `BM_RP_solve_{1,5,10}s` (reports route cost too);
   - `BM_PP_planPath`;
   - `BM_Point_copy`;
   - `BM_Ring_iterate`.
3. **Profiles.** Run `perf record -g --call-graph dwarf python tools/f2c_bench.py one …` on 3 representative instances, then `perf report --no-children`. Save the flame graphs to `bench/profiles/`.
4. **Golden outputs.** For each fixture, store the best angle, swath count, route cost (int64), path length and number of states.
   - Items that must reproduce them **bit-exactly or within 1e-9**: C1, C3, C4, C5, C8, C9, C10, C11, C12, K3.
   - Items that change results by design: C2 (angle differs by < 1°; cost must be ≤ baseline), C6 (swaths equal within 1e-6 m) and C7 (route cost distribution within the same time limit; compare 30 seeds).
5. **Report.** A table of before and after (median of 10 runs) per benchmark goes in the PR description and into the paper's supplementary material.

---

## 2. Instructions per item

### C1: replace Floyd–Warshall with terminal Dijkstra
**Where:** `src/fields2cover/types/Graph.cpp`, `shortestPathsAndCosts`, `initializeMatrices`, `getCosts`, `getPaths`, `shortestPath`, `shortestPathCost`.

1. Delete the "Reconstruct paths" double loop in `initializeMatrices`. It walks every i→j path and stores nothing. Keep only the `next == -1 ⇒ dist = INF` fix-up, folded into the FW loop.
2. Add a CSR (compressed sparse row) snapshot of `edges_`: `std::vector<size_t> offs, nbr; std::vector<int64_t> w;`, rebuilt lazily when edges change (the existing cache-invalidation spot in `addDirectedEdge`).
3. Add `Graph::dijkstraFrom(size_t s)`. It uses a binary heap (`std::priority_queue` on `(dist, node)` with lazy deletion) and writes `dist_[s][*]` and `pred_[s][*]` into **row caches keyed by source**: `std::unordered_map<size_t, Row>`, where `Row` is `{vector<int64_t> dist; vector<int32_t> pred;}`.
4. Re-implement `shortestPathCost(from, to)` as `row(from).dist[to]` and `shortestPath(from, to)` as walking `row(from).pred` backwards. Only the sources actually queried (swath ends and the depot, 2N+1 of them) are ever computed: O(T·E log V) instead of O(V³), and memory O(T·V) instead of O(V²).
5. Keep `getCosts()` and `getPaths()` for API compatibility, but mark them `[[deprecated]]`. They fall back to running Dijkstra from every node.
6. Store `dist` contiguously (`vector<int64_t>` of size V, not `vector<vector>`).
7. **Test:** a gtest with 200 random sparse graphs, asserting that the FW and Dijkstra costs are equal for all pairs. Run the existing `RoutePlanner` tests unchanged.
8. Optional: parallelise the T Dijkstra runs (independent sources) with TBB/OpenMP. Thread safety needs one row cache per thread, merged at the end.

### C2: exact and fast angle for `OBJ_NSwathModified`
**Where:** `src/fields2cover/objectives/sg_obj/n_swath_modified.cpp`, `swath_generator/brute_force.cpp`.

**Why it is exact.** The cost is J(θ) = (1/2w) Σᵢ Lᵢ |sin(θ − φᵢ)| over all ring edges (length Lᵢ, direction φᵢ).
- On any interval of θ where no θ − φᵢ crosses a multiple of π, each term is a non-negative multiple of sin on a half-period, hence concave.
- The sum is therefore concave on every interval between consecutive breakpoints {φᵢ mod π}.
- A concave function on a closed interval attains its minimum at an endpoint.

So **θ\* ∈ {φᵢ mod π}**: the optimum is parallel to some edge (a known property of the "number of swaths" objective). The current 1° sweep can miss it by up to 0.5°.

Steps:
1. Add `NSwathModified::precompute(const F2CCell&)`, which returns `std::vector<std::pair<double,double>> {L_i, φ_i}`. Read raw coordinates with `OGRLinearRing::getX(i)` / `getY(i)` (no `Point` objects).
2. Add `NSwathModified::bestAngle(const F2CCell&, double w)`. It evaluates J at each distinct φᵢ mod π (deduplicated to 1e-9) and returns the argmin. Complexity is O(n²) trig ops (n = 500 gives 250 k operations, well under a millisecond), or O(n log n) with a sorted sweep that updates the sum incrementally.
3. In `BruteForce::computeBestAngle`, if `obj` is `NSwathModified`, return `bestAngle(...)` (a `dynamic_cast` or a new virtual `bool hasExactBestAngle()` plus `exactBestAngle()` on `SGObjective`).
4. Also rewrite the per-angle `computeCost(ang, …)` on the precomputed pairs, so that other callers such as `computeCostOfAngle` benefit.
5. Tests:
   - for 1000 random polygons, `J(exact) ≤ J(sweep at 0.01°) + 1e-9`;
   - the angle matches the sweep's within 1° on the existing tests.
6. Separately in `BruteForce`: sweep **[0, π)** instead of [0, 2π). Swath sets for θ and θ+π are identical up to direction, so this halves the work for every other objective.

### C3: make the parallel sweep actually parallel
**Where:** `CMakeLists.txt` (≈ l.104–114) and `brute_force.cpp`.

1. On Ubuntu 22.04, `libtbb-dev` (oneTBB 2021.5) ships `TBBConfig.cmake`, but `find_package(TBB CONFIG QUIET)` failed silently in our pip build, and `ldd _fields2cover_python.so` shows no `libtbb`. Make the failure loud: `find_package(TBB CONFIG REQUIRED)` when `ALLOW_PARALLELIZATION` is ON, and print `TBB_DIR`.
2. Check the result: `ldd …/_fields2cover_python.so | grep tbb`. A micro-benchmark `BM_SG_bestAngle_SwathLength` should scale with cores.
3. Alternative without TBB: replace `std::transform(std::execution::par_unseq, …)` with an `#pragma omp parallel for` over the angle ids and link `OpenMP::OpenMP_CXX`.
4. **Thread safety (VERIFY):** `computeCostOfAngle` calls GEOS through OGR. GDAL/OGR geometry ops use a per-thread GEOS context (`OGRGeometry::createGEOSContext`), which should be safe, but **test under TSan** (`-fsanitize=thread`) before enabling.
5. **In the benchmark:** when running 8 worker processes on 8 cores, set `TBB_NUM_THREADS=1` / `OMP_NUM_THREADS=1` in the workers to avoid oversubscription. Use the threads only for single large runs.

### C4: cheaper `Point` and geometry handles
**Where:** `include/fields2cover/types/Geometry.h`, `Geometry_impl.hpp`, `Point.h` / `Point.cpp`.

1. **Quick win (ABI-neutral for users):** change
   ```cpp
   std::shared_ptr<T> operator->();              // copies shared_ptr: 2 atomic RMW per call
   std::shared_ptr<const T> operator->() const;
   ```
   to
   ```cpp
   T* operator->() { return data_.get(); }
   const T* operator->() const { return data_.get(); }
   ```
   This is behaviour-identical, because callers only dereference.
2. Allocate the control block and the object together where possible. Today it is `shared_ptr<T>(createGeometry(R), deleter)`, which is 2 allocations. For `Point`, construct as `std::make_shared<OGRPoint>(x, y, z)`. The default deleter is fine because `OGRPoint` is a complete type with a virtual destructor. This needs `OGRGeometryFactory::destroyGeometry` semantics to be the same as `delete`, which holds for GDAL-built objects in the same module (**VERIFY on Windows/MSVC builds**, where cross-DLL heaps differ; keep the factory path under `#ifdef _WIN32`).
3. `Point(const Point&)` deep-clones on purpose: points are mutable value types. Keep it, but **add move-awareness**. `Point& operator=(Point&&)` currently copies the coordinates instead of swapping `data_`; implement it as `data_.swap(p.data_)`.
4. **Medium-term:** add a header-only POD `struct Vec2 { double x, y; }` with arithmetic, and use it internally in hot loops: objectives, swath generation, `closestPoint*`, Dubins sampling and `PathState`. Convert to and from `Point` at API boundaries only. Do this file by file, guided by the profiles from §1.
5. Test: the full gtest suite, plus `BM_Point_copy` and `BM_Ring_iterate` before and after.

### C5: iterator overhead
**Where:** `include/fields2cover/types/Geometries_impl.hpp` (`Iterator`, `ConstIterator`).

The non-const `Iterator`:
- holds a heap `Private`;
- on `operator*` it copies the child into `m_oChild`;
- on `++` and in the destructor `update()` fetches the child **again** into a fresh object and compares it, writing back if it changed.

So a `for (auto&& p : ring)` over a non-const ring costs about 2 allocations, 2 copies and a compare per element.

1. In library code, iterate **const** objects (`for (auto&& p : std::as_const(ring))`), or use index loops on the raw OGR object: `ring->getX(i)`, `ring->getY(i)`, `ring->getNumPoints()`.
2. Grep for hot non-const range-for loops: `grep -rn "for (auto&& .* : \*this)" src/` and loops over `Cell`, `LinearRing` and `LineString`. Profile-guided priorities: `NSwathModified::computeCost`, `createShortestGraph` (ring edges), `Cell::closestPointOnBorderTo`, `Point::rotateFromPoint<T>`.
3. Optional: keep the write-back iterator, but make `update()` skip the re-fetch when `operator*` was never called, and allocate `Private` inline (no `new`).

### C6: analytic swath generation (scanline clipping)
**Where:** `src/fields2cover/swath_generator/swath_generator_base.cpp`, `generateSwaths(angle, width, cell)`.

Today, per angle, the code:
1. rotates the cell (a new geometry and a `Point` per vertex, with `sin`/`cos` recomputed per vertex);
2. builds K long lines;
3. rotates the lines back;
4. calls GEOS `Intersection(lines, cell)`, which converts both to GEOS every call.

Replacement:
1. Extract the edges once per cell as raw `(x0, y0, x1, y1)` arrays (all rings).
2. For angle θ, compute the unit direction d and normal n. For each edge, compute its normal-coordinate interval `[s_min, s_max]` = `[min(n·p0, n·p1), max(…)]`.
3. Swath offsets are `s_k = s_min(field) + (k − ½)·w`, matching the current `curve_y` logic including `allow_overlap`. For each `s_k`, collect edges whose interval contains `s_k`, i.e. a sweep with edges sorted by `s_min`, plus an active set. Compute the crossing parameter `t = d·p` and sort the crossings. Even-odd pairs are the inside intervals: this is standard polygon scan conversion, and holes work automatically.
4. Emit each interval as a 2-point `LineString`. Handle degenerate cases explicitly: a vertex exactly on a scanline is counted once (half-open rule), and horizontal edges are skipped.
5. Keep the GEOS path as `generateSwathsReference()` and test equivalence on random polygons with holes. Each interval endpoint must match within 1e-6 m, and the swath count must match exactly.
6. Expected cost per angle: O(n log n + K + crossings), instead of GEOS overlay on the whole polygon.

### C7: OR-Tools callback on a dense matrix
**Where:** `route_planner_base.cpp`, `computeBestRoute`.
```cpp
const int n = cov_graph.numNodes();
std::vector<int64_t> M(static_cast<size_t>(n) * n, INF);
for (auto& [from, tos] : cov_graph.getEdges()) for (auto& [to, c] : tos) M[from * n + to] = c;
const int cb = routing.RegisterTransitCallback(
  [&M, &manager, n](int64_t i, int64_t j) {
    return M[manager.IndexToNode(i).value() * n + manager.IndexToNode(j).value()];
  });
```
- `getEdges()` returns the map **by value** (a full copy). Add `const map_to_map_to_int& edges() const`.
- Cache `numNodes()`: it builds an `unordered_set` every call.
- Optionally use `RegisterTransitMatrix` (OR-Tools ≥ 9.x), which stores the matrix inside the solver.
- Test: same route cost on fixed seeds with `search_for_optimum=false` (deterministic first solution). With GLS, compare the distribution.

### C8: build the coverage model without `Graph2D`
**Where:** `createCoverageGraph`.
Number the terminals directly: swath s has nodes 2s (start) and 2s+1 (end), and the depot or start/end come last. Fill the matrix from C1 rows (`row(src).dist[dst]`) for all terminal pairs. No `Point` hashing, no nested maps. Keep a parallel `std::vector<Point>` for `transformSolutionToRoute`.
Optionally drop the mid-node per swath (3 nodes become 2) by using a disjunction/pickup-delivery pair. Test this one separately, because it changes the model.

### C9: spatial indexing in `createShortestGraph`
- Replace the "connect nodes that are near other edges" double loop (it runs twice over a **copy** of `getEdges()`) with a uniform grid hash or `GEOSSTRtree` of segments. Query each node's d_tol neighbourhood: O((V+E) log E).
- `getCellWherePoint(p, d_tol)`, called for every swath end, performs a GEOS `distance` per cell. Carry the cell index with each swath (`SwathsByCells` already groups them), and only fall back to the search when needed.

### C10: route reconstruction
**Where:** `transformSolutionToRoute`.
Pre-build `std::unordered_map<std::pair<int,int>, std::pair<size_t,bool>>` from (start node, end node) to (swath id, reversed). Remove the per-step `clone()` of every scanned swath; clone only the matched one.

### C11: paths
- `Path::operator+=`: call `states_.reserve(states_.size() + path.size())` and use `insert(end, begin, end)`.
- Turn discretisation: the default `0.01 m` (`turning_base.h`) produces about 100 states per metre of turn. Change the default to `0.1 m`, or document it prominently. **Our benchmark already sets 0.2 m.**
- Longer term, make `PathState::point` a `Vec2`/`Vec3` (see C4). Paths are the biggest objects crossing the SWIG boundary.

### C12: argmin helpers
**Where:** `LineString::closestPointTo`, `Cell::closestPointOnBorderTo`, `Cells::closestPointOnBorderTo`.
Replace the "build `vector<Point>` and `vector<double>`, then `min_element`" pattern with a running best on raw coordinates, using `ring->getX(i)` (no `at(i)` copies). Return one `Point` at the end.

### K1–K3: correctness fixes
- **K1:** define `operator<` lexicographically on coordinates *quantised the same way as the hash* (`llround(x·1e7)`, then y, then z), so that `<`, `==` and the hash agree.
- **K2:** make `operator==` compare the quantised integers too (exact equality after `llround`). This changes `==` semantics only for points within 1e-7 of a rounding boundary. Document it in the CHANGELOG.
- **K3:** store `depot_idx_` / `start_idx_` / `end_idx_` explicitly and use `RoutingIndexManager(n, 1, {start}, {end})`.

---

## 3. Order of work
1. §1 measurement harness and golden outputs (1–2 days).
2. Low-risk, high-yield: **C3** (build), **C4 step 1** (`operator->`), **C7**, **C10**, **C11**, **C12**, **C1 step 1** (delete the dead loop) (2–3 days).
3. **C1** (Dijkstra) + **C8** (2–4 days), then **C2** (exact angle) (1–2 days).
4. **C5** and C4 steps 2–3, profile-guided (2–3 days).
5. **C6** scanline swaths (4–6 days, the largest change).
6. **C9** spatial index (2 days). Then K1–K3, together with the start/end feature (`ATTACK_PLAN.md` §A).

Each step is one commit with its benchmark delta. Group them into upstream PRs by theme (graph, SG, geometry core, build). Target `develop` (2.1.x), and check that the patches apply to the `v3.0` branch, which restructures route planning.

## 4. What not to change
- Do not change the default objective semantics or the public Python API in the same PRs as the performance work.
- Do not drop the GEOS reference implementations. Keep them behind `…Reference()` functions for tests.
- Do not parallelise inside benchmark workers. Parallelism across instances already saturates the VM.
