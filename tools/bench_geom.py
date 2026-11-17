"""Pure-geometry layer of the benchmark (numpy + shapely only, no Fields2Cover).

- synthetic paddocks (rectangle / convex irregular / L-shape) at a target area
- obstacles: trees (discs) and rock outcrops (polygons), Poisson or Thomas-cluster placement
- general B3 detour around a smooth convex set S (covers discs and buffered polygon hulls)
- polygon-aware clearance
"""
import zlib
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union, nearest_points, substring

from nb_geom import Track, simulate_coverage, turn_length_inside, length_outside, max_curvature  # noqa: F401


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Paddocks
# ---------------------------------------------------------------------------
def make_field(shape, area_ha, rng):
    """Polygon of `area_ha` hectares with the road along its southern (min-y) side."""
    A = area_ha * 1e4
    if shape == "rectangle":
        poly = box(0, 0, np.sqrt(A * 1.25), np.sqrt(A / 1.25))
    elif shape == "L":
        poly = Polygon([(0, 0), (600, 0), (600, 250), (300, 250), (300, 500), (0, 500)])
    elif shape == "convex":
        n = 7
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = rng.uniform(0.8, 1.2, n)
        pts = np.c_[1.4 * rad * np.cos(ang), rad * np.sin(ang)]
        poly = Polygon(pts).convex_hull
    else:
        raise ValueError(shape)
    poly = affinity.scale(poly, np.sqrt(A / poly.area), np.sqrt(A / poly.area), origin=(0, 0))
    minx, miny, _, _ = poly.bounds
    return shapely.normalize(affinity.translate(poly, -minx, -miny))


def road_gate(field):
    """Road-side gate: the boundary point nearest to a road far to the south."""
    minx, miny, maxx, _ = field.bounds
    p = nearest_points(field.exterior, Point((minx + maxx) / 2, miny - 1e4))[0]
    return (p.x, p.y)


def random_vertex_gate(field, rng):
    c = list(field.exterior.coords)[:-1]
    return tuple(c[rng.integers(len(c))])


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------
def _tree(c, rng):
    r = rng.uniform(4, 9)
    return {"type": "tree", "poly": Point(c).buffer(r, quad_segs=16), "r": r}


def _outcrop(c, rng):
    R = rng.uniform(6, 15)
    nv = int(rng.integers(6, 10))
    ang = np.sort(rng.uniform(0, 2 * np.pi, nv))
    rad = R * rng.uniform(0.55, 1.0, nv)
    poly = Polygon(np.c_[c[0] + rad * np.cos(ang), c[1] + rad * np.sin(ang)]).buffer(0)
    return {"type": "outcrop", "poly": poly, "r": R}


def make_obstacles(region, n, otype, pattern, rng, sigma=15.0, mu=4):
    """n obstacles with centres in `region`; pattern 'poisson' (uniform) or 'thomas' (clustered)."""
    if n <= 0 or region.is_empty:
        return []
    minx, miny, maxx, maxy = region.bounds

    def uniform_pt():
        while True:
            p = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if region.contains(Point(p)):
                return p

    if pattern == "poisson":
        centres = [uniform_pt() for _ in range(n)]
    elif pattern == "thomas":
        parents = [uniform_pt() for _ in range(max(1, int(round(n / mu))))]
        centres = []
        while len(centres) < n:
            px, py = parents[rng.integers(len(parents))]
            p = (px + rng.normal(0, sigma), py + rng.normal(0, sigma))
            if region.contains(Point(p)):
                centres.append(p)
    else:
        raise ValueError(pattern)
    out = []
    for c in centres:
        kind = otype if otype in ("tree", "outcrop") else ("tree" if rng.random() < 0.5 else "outcrop")
        out.append(_tree(c, rng) if kind == "tree" else _outcrop(c, rng))
    return out


def obstacle_groups(obstacles, margin):
    """Connected components of overlapping safety zones (obstacle ⊕ B(margin))."""
    zones = [o["poly"].buffer(margin) for o in obstacles]
    parent = list(range(len(zones)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = shapely.STRtree(zones)
    for i, z in enumerate(zones):
        for j in tree.query(z, predicate="intersects"):
            if j > i:
                parent[find(i)] = find(int(j))
    groups = {}
    for i in range(len(zones)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def half_widths(obstacles, width, track_width=None):
    """Half-width that must clear each obstacle. Class I (default, and all trees): the implement,
    w/2. Class II (B4 straddle, outcrops when track_width is given): only the running gear, g/2;
    the boom passes over with its sections off."""
    if track_width is None:
        return [width / 2] * len(obstacles)
    return [width / 2 if o["type"] == "tree" else track_width / 2 for o in obstacles]


def clearance(track, obstacles, width, half=None):
    """Edge clearance δ = dist(path, obstacle polygon) − half-width for each obstacle (canopy / rock)."""
    ln = track.line
    half = half_widths(obstacles, width) if half is None else half
    return [ln.distance(o["poly"]) - h for o, h in zip(obstacles, half)]


# ---------------------------------------------------------------------------
# General convex detour
# ---------------------------------------------------------------------------
def _arc(cx, cy, r, a0, a1, step):
    n = max(2, int(np.ceil(abs(a1 - a0) * r / step)) + 1)
    t = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def _detour_local(S, L, rho, step):
    """Detour in the swath frame (swath = segment (0,0)-(L,0), bulge towards +y) around the
    smooth convex set S. Returns (points, x1, x2) or None if S does not touch the swath line
    or the detour does not fit between the swath ends."""
    G = S.buffer(rho, quad_segs=32)
    xmin, _, xmax, _ = G.bounds
    hl = LineString([(xmin - 1, rho), (xmax + 1, rho)])
    inter = G.exterior.intersection(hl)
    xs = [p.x for p in getattr(inter, "geoms", [inter]) if not p.is_empty and p.geom_type == "Point"]
    if len(xs) < 2:
        return None
    x1, x2 = min(xs), max(xs)
    if x1 < 0 or x2 > L:
        return None
    c1, c2 = np.array([x1, rho]), np.array([x2, rho])
    T1 = np.array(nearest_points(S, Point(c1))[0].coords[0])
    T2 = np.array(nearest_points(S, Point(c2))[0].coords[0])
    th1 = np.arctan2(*(T1 - c1)[::-1])
    while th1 <= -np.pi / 2:
        th1 += 2 * np.pi
    f1 = _arc(x1, rho, rho, -np.pi / 2, th1, step)
    th2 = np.arctan2(*(T2 - c2)[::-1])
    e2 = -np.pi / 2
    while e2 <= th2:
        e2 += 2 * np.pi
    f2 = _arc(x2, rho, rho, th2, e2, step)
    ring = LineString(S.exterior.coords)
    d1, d2 = ring.project(Point(T1)), ring.project(Point(T2))
    lo, hi = sorted((d1, d2))
    a = substring(ring, lo, hi)
    b_parts = [substring(ring, hi, ring.length), substring(ring, 0, lo)]
    b = LineString(np.vstack([np.asarray(p.coords) for p in b_parts if p.length > 0]))
    top = max((a, b), key=lambda g: np.asarray(g.coords)[:, 1].mean())
    tc = np.asarray(top.coords)
    if np.hypot(*(tc[0] - T1)) > np.hypot(*(tc[-1] - T1)):
        tc = tc[::-1]
    return np.vstack([f1, tc, f2]), x1, x2


def detour_line_convex(line, sets, rho, is_valid, step=0.5, mergeable=None, max_iter=12):
    """Splice detours into straight swath `line` around every smooth convex set in `sets`
    (shapely polygons) that touches it. Returns (LineString, n_unplaced).

    All merging happens BEFORE any detour is committed, so a merged hull can never reach back
    over a detour that was already placed (that produced a 180-degree backtrack in v1.0):
      1. sets whose detour intervals could overlap along the swath are merged into their hull;
      2. every set is tried on both sides; if a set fails, the sets of `mergeable` near it are
         merged into it and the whole list is re-checked from step 1;
      3. a set that still fails is left unplaced (the swath drives straight; it is flagged).
    The committed detours are verified to be strictly ordered along the swath."""
    p0, p1 = np.asarray(line.coords[0], float), np.asarray(line.coords[-1], float)
    L = np.hypot(*(p1 - p0))
    u = (p1 - p0) / L
    n = np.array([-u[1], u[0]])
    ang = np.degrees(np.arctan2(u[1], u[0]))
    to_local = lambda g: affinity.rotate(affinity.translate(g, -p0[0], -p0[1]), -ang, origin=(0, 0))
    to_world = lambda xy: p0 + np.outer(xy[:, 0], u) + np.outer(xy[:, 1], n)
    axis = LineString([(-1e6, 0), (1e6, 0)])
    on_swath = lambda s: s.intersects(axis) and s.bounds[2] > -rho and s.bounds[0] < L + rho

    def merge_overlapping(hs):
        hs = sorted(hs, key=lambda s: s.bounds[0])
        out = []
        for s in hs:
            if out and out[-1].bounds[2] + 2 * rho > s.bounds[0]:
                out[-1] = unary_union([out[-1], s]).convex_hull
            else:
                out.append(s)
        return out if len(out) == len(hs) else merge_overlapping(out)

    def best_side(S):
        best = None
        for flip in (1, -1):
            res = _detour_local(affinity.scale(S, 1, flip, origin=(0, 0)), L, rho, step)
            if res is None:
                continue
            prof = res[0] * np.array([1, flip])
            piece = LineString(to_world(prof))
            if is_valid(piece) and (best is None or piece.length < best[0]):
                best = (piece.length, prof, res[1], res[2])
        return best

    loc_merge = [to_local(g) for g in (mergeable or [])]
    hits = [s for s in (to_local(s) for s in sets) if on_swath(s)]
    unplaced_sets = []
    for _ in range(max_iter):
        hits = merge_overlapping(hits)
        placed, failed = [], None
        for S in hits:
            b = best_side(S)
            if b is None:
                failed = S
                break
            placed.append(b)
        if failed is None:
            break
        extra = [m for m in loc_merge if m.intersects(failed.buffer(2 * rho)) and not m.within(failed)]
        grown = unary_union([failed] + extra).convex_hull if extra else None
        hits = [s for s in hits if s is not failed]
        if grown is not None and grown.area > failed.area * 1.001:
            hits.append(grown)            # retry with the grown hull (re-merged in step 1)
        else:
            unplaced_sets.append(failed)  # give up on this one: straight through, flagged
    else:
        placed = [b for b in (best_side(S) for S in merge_overlapping(hits)) if b is not None]

    placed.sort(key=lambda b: b[2])
    pts, x_prev = [np.zeros(2)], 0.0
    for _, prof, x1, x2 in placed:
        if x1 < x_prev - 1e-6:            # should be impossible after merging; never backtrack
            unplaced_sets.append(None)
            continue
        pts.extend(prof)
        x_prev = x2
    pts.append(np.array([L, 0.0]))
    P = np.asarray(pts)
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-6]   # drop duplicate vertices at joins
    return LineString(to_world(P[keep])), len(unplaced_sets)


def detour_sets(obstacles, margin, width, r_min, quad_segs=16, half=None):
    """Smooth convex clearance set per obstacle group:
    discs (single tree): disc of radius max(r + m + h, r_min);
    otherwise: convex hull of the group ⊕ B(max(m + h, r_min)).
    h is the half-width that must clear the obstacle (w/2, or g/2 for B4 class-II outcrops).
    In a group with mixed h, the hull is taken over each member ⊕ B(max(m + h_i, r_min)); its
    corners are arcs of radius ≥ r_min, so the fillet/curvature argument still holds."""
    half = half_widths(obstacles, width) if half is None else half
    sets = []
    for g in obstacle_groups(obstacles, margin):
        eps = [max(margin + half[i], r_min) for i in g]
        if len(g) == 1 and obstacles[g[0]]["type"] == "tree":
            o = obstacles[g[0]]
            c = o["poly"].centroid
            sets.append(c.buffer(max(o["r"] + margin + half[g[0]], r_min), quad_segs=quad_segs * 2))
        elif len(set(eps)) == 1:
            hull = unary_union([obstacles[i]["poly"] for i in g]).convex_hull
            sets.append(hull.buffer(eps[0], quad_segs=quad_segs))
        else:
            sets.append(unary_union([obstacles[i]["poly"].buffer(e, quad_segs=quad_segs)
                                     for i, e in zip(g, eps)]).convex_hull)
    return sets


def make_drive_around_poly(track, obstacles, margin, width, r_min, allowed_poly, step=0.5, half=None):
    """Strategy B3 (half=None: everything class I) or B4 (half from half_widths(..., track_width)).
    Returns (Track, n_unplaced_detours)."""
    # B4 sets hug arcs of exactly r_min, so they need a finer arc (64 segments/quarter) to keep the
    # discretised curvature at the B3 level (~0.171 vs 1/r_min = 0.167); B3 keeps 16 (as in v1_2).
    q = 16 if half is None else 64
    half = half_widths(obstacles, width) if half is None else half
    sets = detour_sets(obstacles, margin, width, r_min, quad_segs=q, half=half)
    hard = [o["poly"].buffer(h - 1e-6) for o, h in zip(obstacles, half)]
    hard_tree = shapely.STRtree(hard) if hard else None

    def is_valid(piece):
        if not allowed_poly.covers(piece):
            return False
        return hard_tree is None or len(hard_tree.query(piece, predicate="intersects")) == 0

    segs, unplaced = [], 0
    for g, kind in track.segs:
        if kind != "swath":
            segs.append((g, kind))
            continue
        c = list(g.coords)
        for a, b in zip(c[:-1], c[1:]):
            g2, u = detour_line_convex(LineString([a, b]), sets, r_min, is_valid, step, mergeable=sets)
            segs.append((g2, kind))
            unplaced += u
    return Track(segs, track.n_rev), unplaced


def corridor_passes(region, width, min_area=None):
    """Working passes that cover strip-shaped `region` pieces (e.g. the corridors F2C's
    HG_Corridor_gen cuts between decomposed cells, which no swath covers). Each piece gets
    ceil(strip width / w) passes along its long axis (minimum rotated rectangle)."""
    out = []
    min_area = width * width if min_area is None else min_area
    for piece in getattr(region, "geoms", [region]):
        if piece.is_empty or piece.area < min_area:
            continue
        rect = np.asarray(piece.minimum_rotated_rectangle.exterior.coords)[:4]
        e1, e2 = rect[1] - rect[0], rect[2] - rect[1]
        long_e = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
        u = long_e / np.hypot(*long_e)
        nrm = np.array([-u[1], u[0]])
        xy = np.asarray(piece.exterior.coords)
        a, b = (xy @ nrm).min(), (xy @ nrm).max()
        k = max(1, int(np.ceil((b - a) / width - 1e-6)))
        for j in range(k):
            c = (a + b) / 2 + (j - (k - 1) / 2) * width
            ln = LineString([c * nrm - 1e5 * u, c * nrm + 1e5 * u]).intersection(piece.buffer(0.01))
            for seg in getattr(ln, "geoms", [ln]):
                if seg.geom_type == "LineString" and seg.length > width:
                    out.append(seg)
    return out


def clean_poly(g, grid=1e-6):
    """Snap to a fine grid and repair: F2C/OGR output can carry repeated or near-coincident vertices
    (zero-length segments) that make GEOS overlay ops raise 'non-noded intersection'."""
    g = shapely.make_valid(shapely.set_precision(g, grid))
    polys_ = [p for p in getattr(g, "geoms", [g]) if p.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(polys_) if polys_ else Polygon()
