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


def clearance(track, obstacles, width):
    """Edge clearance δ = dist(path, obstacle polygon) − w/2 for each obstacle (canopy / rock)."""
    ln = track.line
    return [ln.distance(o["poly"]) - width / 2 for o in obstacles]


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


def detour_line_convex(line, sets, rho, is_valid, step=0.5, mergeable=None):
    """Splice detours into straight swath `line` around every smooth convex set in `sets`
    (shapely polygons) that touches it. Sets whose detour intervals overlap are merged into
    their convex hull. If a detour is invalid on both sides, sets hit by it are merged in
    (when `mergeable` is given) and the detour is retried. Returns (LineString, n_unplaced)."""
    p0, p1 = np.asarray(line.coords[0], float), np.asarray(line.coords[-1], float)
    L = np.hypot(*(p1 - p0))
    u = (p1 - p0) / L
    n = np.array([-u[1], u[0]])
    ang = np.degrees(np.arctan2(u[1], u[0]))
    to_local = lambda g: affinity.rotate(affinity.translate(g, -p0[0], -p0[1]), -ang, origin=(0, 0))
    to_world = lambda xy: p0 + np.outer(xy[:, 0], u) + np.outer(xy[:, 1], n)
    axis = LineString([(-1e6, 0), (1e6, 0)])

    loc = [to_local(s) for s in sets]
    hits = [s for s in loc if s.intersects(axis) and s.bounds[2] > -rho and s.bounds[0] < L + rho]
    changed = True
    while changed and len(hits) > 1:
        changed = False
        hits.sort(key=lambda s: s.bounds[0])
        for i in range(len(hits) - 1):
            if hits[i].bounds[2] + 2 * rho > hits[i + 1].bounds[0]:
                hits[i:i + 2] = [unary_union(hits[i:i + 2]).convex_hull]
                changed = True
                break

    pts, unplaced = [np.zeros(2)], 0
    for S in sorted(hits, key=lambda s: s.bounds[0]):
        placed = False
        for _attempt in range(3):
            for flip in (1, -1):              # +y side, then -y side (shorter one first)
                Sf = affinity.scale(S, 1, flip, origin=(0, 0))
                res = _detour_local(Sf, L, rho, step)
                if res is None:
                    continue
                prof = res[0] * np.array([1, flip])
                piece = LineString(to_world(prof))
                if is_valid(piece):
                    cand = (piece.length, prof)
                    if not placed or cand[0] < best[0]:
                        best = cand
                    placed = True
            if placed or mergeable is None:
                break
            extra = [m for m in (to_local(g) for g in mergeable) if m.intersects(S.buffer(2 * rho)) and not m.within(S)]
            if not extra:
                break
            S = unary_union([S] + extra).convex_hull
        if placed:
            pts.extend(best[1])
        else:
            unplaced += 1
    pts.append(np.array([L, 0.0]))
    return LineString(to_world(np.asarray(pts))), unplaced


def detour_sets(obstacles, margin, width, r_min, quad_segs=16):
    """Smooth convex clearance set per obstacle group:
    discs (single tree): disc of radius max(r + m + w/2, r_min);
    otherwise: convex hull of the group ⊕ B(max(m + w/2, r_min))."""
    eps = margin + width / 2
    sets = []
    for g in obstacle_groups(obstacles, margin):
        if len(g) == 1 and obstacles[g[0]]["type"] == "tree":
            o = obstacles[g[0]]
            c = o["poly"].centroid
            sets.append(c.buffer(max(o["r"] + eps, r_min), quad_segs=quad_segs * 2))
        else:
            hull = unary_union([obstacles[i]["poly"] for i in g]).convex_hull
            sets.append(hull.buffer(max(eps, r_min), quad_segs=quad_segs))
    return sets


def make_drive_around_poly(track, obstacles, margin, width, r_min, allowed_poly, step=0.5):
    """Strategy B3 for discs and polygons. Returns (Track, n_unplaced_detours)."""
    sets = detour_sets(obstacles, margin, width, r_min)
    hard = [o["poly"].buffer(width / 2 - 1e-6) for o in obstacles]
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
