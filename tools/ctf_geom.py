"""Controlled-traffic (CTF) tramline geometry for E4 / R10 (pure numpy + shapely, no Fields2Cover).

The sprayer plan (width w_s = n * w_b) is the tramline master. Each sprayer working pass carries base
lanes (seeder / header, width w_b) at lateral offsets d measured along a CANONICAL normal (so that lanes of
passes driven in opposite directions line up):
    shared   d_j = (j - floor((n-1)/2)) * w_b     sprayer on a base lane (CTF in the wheel sense; primary)
    centred  d_j = (j - (n-1)/2) * w_b            lanes tile the sprayer strip symmetrically (sensitivity)
For even n the shared layout has a lane on one strip boundary only; the outermost pass on the other side
gets the missing boundary lane (edge lane), so the lanes tile the field.

Lane validity uses the master's curvature kappa (window-smoothed, robust to polygonised arcs): an offset at
signed distance d (left = +) has radius (1 - d*kappa)/|kappa|; 1 - d*kappa <= 0 is a cusp.

Also here: B3c / B3n (CTF-compatible drive-around), headland base rings, wheel-track rasters (off-lane
travel, trafficked area) and lane crossing / spacing statistics.
"""
import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union

from bench_geom import Track, detour_line_convex, detour_sets, half_widths  # noqa: F401

CTF_DEFAULTS = dict(
    w_b=12.0,                 # base module (seeder, header)
    gauge=3.0,                # common wheel-track gauge
    tyre_width=0.6,
    r_min_seeder=6.0, r_min_harvester=8.0,
    s_min=6.0,                # B3n: minimum centreline separation between lanes of adjacent passes
    s_min_sens=12.0,          # B3n sensitivity (= w_b: no strip overlap -> bump propagates)
    nest_k=2,                 # B3n_k2: at most this many pushed passes per chain
    off_lane_tol=0.5,         # wheel travel farther than this from any network wheel track is off-lane
    traffic_res=0.2,          # raster for wheel footprints
    curv_ds=0.5, curv_window=2.0, radius_tol=0.03,
    seeder_sections=6,
    part3_filter=dict(otype="mixed", pattern="poisson", density=1.0, gate_type="road", shapes=["rectangle", "L"]),
    part3_turn_model="DubinsCC", part3_rp_time=10,
)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def lane_offsets(n, w_b, layout):
    if layout == "shared":
        return [(j - (n - 1) // 2) * w_b for j in range(n)]
    return [(j - (n - 1) / 2) * w_b for j in range(n)]


def d_max(n, w_b, layout):
    return max(abs(d) for d in lane_offsets(n, w_b, layout))


def canonical(u):
    """Unit direction with angle in [0, pi) -> (u_c, n_c) with n_c the left normal of u_c."""
    u = np.asarray(u, float) / np.hypot(*u)
    if u[1] < -1e-12 or (abs(u[1]) <= 1e-12 and u[0] < 0):
        u = -u
    return u, np.array([-u[1], u[0]])


def _chord(line):
    c = np.asarray(line.coords)
    return c[0], c[-1]


# ---------------------------------------------------------------------------
# Curvature and offsets
# ---------------------------------------------------------------------------
def sample(line, ds):
    L = line.length
    m = max(2, int(np.ceil(L / ds)) + 1)
    d = np.linspace(0, L, m)
    return shapely.get_coordinates(shapely.line_interpolate_point(line, d)), L / (m - 1)


def frame(pts, ds, window):
    """Unit normals (left) and signed curvature at each sample, smoothed over `window` metres."""
    seg = np.diff(pts, axis=0)
    th = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    th_pt = np.r_[th[0], 0.5 * (th[1:] + th[:-1]), th[-1]]          # tangent angle at the samples
    k = max(1, int(round(window / (2 * ds))))
    i = np.arange(len(pts))
    lo, hi = np.clip(i - k, 0, len(pts) - 1), np.clip(i + k, 0, len(pts) - 1)
    span = np.maximum((hi - lo) * ds, 1e-9)
    kappa = (th_pt[hi] - th_pt[lo]) / span
    nrm = np.c_[-np.sin(th_pt), np.cos(th_pt)]
    return nrm, kappa


def offset_lane(pts, nrm, kappa, d, r_check, tol):
    """Lane at signed offset d (left +) of a sampled master. Returns dict(line, min_radius, cusp, simple, valid)."""
    q = pts + d * nrm
    f = 1 - d * kappa
    cusp = bool((f <= 0).any())
    curved = np.abs(kappa) > 1e-6
    rad = np.where(curved, f / np.maximum(np.abs(kappa), 1e-12), np.inf)
    min_r = float(rad.min()) if len(rad) else np.inf
    worst = q[int(np.argmin(rad))] if len(rad) else q[0]
    line = LineString(q)
    simple = bool(line.is_simple)
    line = line.simplify(0.01) if simple else line   # straight runs -> 2 vertices (keeps sampling linear)
    valid = (not cusp) and simple and min_r >= r_check * (1 - tol)
    return dict(line=line, min_radius=min_r, cusp=cusp, simple=simple, valid=valid, worst_xy=worst)


def make_lanes(passes, w_s, w_b, layout, r_check, ds=0.5, window=2.0, tol=0.03):
    """Base lanes of every working pass. Returns a list of lane dicts
    (line, d, pass_idx, detoured, min_radius, cusp, simple, valid, edge)."""
    n = int(round(w_s / w_b))
    D = lane_offsets(n, w_b, layout)
    info = []
    for line in passes:
        a, b = _chord(line)
        if np.hypot(*(b - a)) < 1e-6:
            continue
        u_c, n_c = canonical(b - a)
        sign = 1.0 if np.dot(b - a, u_c) > 0 else -1.0       # local left normal = sign * n_c
        mid = 0.5 * (a + b)
        info.append(dict(line=line, u=u_c, n=n_c, sign=sign, c=float(mid @ n_c),
                         x0=float(min(a @ u_c, b @ u_c)), x1=float(max(a @ u_c, b @ u_c))))

    def has_neighbour(i, side):
        p = info[i]
        for j, q in enumerate(info):
            if j == i or abs(p["u"] @ q["n"]) > 0.02:
                continue
            dc = (q["c"] - p["c"]) * side
            if 0.5 * w_s < dc < 1.5 * w_s and min(p["x1"], q["x1"]) - max(p["x0"], q["x0"]) > w_b:
                return True
        return False

    lanes = []
    for i, p in enumerate(info):
        offs = [(d, False) for d in D]
        if layout == "shared" and n % 2 == 0 and not has_neighbour(i, -1):
            offs += [(d - w_s, True) for d in D if abs(d - w_s) <= w_s / 2 + 1e-9]
        pts, dsr = sample(p["line"], ds)
        nrm, kappa = frame(pts, dsr, window)
        detoured = bool(np.abs(kappa).max() > 1e-4)
        for d, edge in offs:
            lane = offset_lane(pts, nrm, kappa, p["sign"] * d, r_check, tol)
            lane.update(d=d, pass_idx=i, detoured=detoured, edge=edge)
            lanes.append(lane)
    return lanes


def headland_lanes(region, w_b, band):
    """Base-machine headland rounds: rings at (i + 0.5) w_b inside `region` filling a band of width `band`
    (k * w_s), around the outer boundary and every hole."""
    out = []
    for i in range(int(round(band / w_b))):
        g = region.buffer(-(i + 0.5) * w_b)
        for p in getattr(g, "geoms", [g]):
            if p.is_empty or p.geom_type != "Polygon":
                continue
            out += [LineString(p.exterior.coords)] + [LineString(r.coords) for r in p.interiors]
    return out


# ---------------------------------------------------------------------------
# Lane statistics
# ---------------------------------------------------------------------------
def crossings(lines):
    """Pairs of distinct lanes that cross (interiors meet in points)."""
    if len(lines) < 2:
        return 0
    tree = shapely.STRtree(lines)
    a, b = tree.query(lines, predicate="crosses")
    return int(((a < b)).sum())


def overlaps(lines, tol=0.5, min_len=24.0):
    """Pairs of distinct lanes that run on top of each other (within tol for > min_len, i.e. duplicated
    tramline, e.g. edge lanes); short near-tangent crossings are counted by crossings() only."""
    if len(lines) < 2:
        return 0
    tree = shapely.STRtree(lines)
    n = 0
    for i, ln in enumerate(lines):
        for j in tree.query(ln.buffer(tol)):
            if j > i and lines[j].intersection(ln.buffer(tol)).length > min_len:
                n += 1
    return n


def spacing(lines, w_b, step=2.0, end_excl=None):
    """Nearest-other-lane distance sampled along every lane (lane ends excluded)."""
    end_excl = w_b / 2 if end_excl is None else end_excl
    if len(lines) < 2:
        return np.array([])
    tree = shapely.STRtree(lines)
    out = []
    for i, ln in enumerate(lines):
        if ln.length <= 2 * end_excl:
            continue
        cand = [j for j in tree.query(ln.buffer(2.5 * w_b)) if j != i]
        if not cand:
            continue
        s = np.arange(end_excl, ln.length - end_excl, step)
        pts = shapely.line_interpolate_point(ln, s)
        out.append(shapely.distance(pts, MultiLineString([lines[j] for j in cand])))
    return np.concatenate(out) if out else np.array([])


def lane_clearance(lines, obstacles, half):
    if not obstacles or not lines:
        return ""
    ml = MultiLineString([l for l in lines])
    return float(min(ml.distance(o["poly"]) - half for o in obstacles))


# ---------------------------------------------------------------------------
# Wheel tracks (raster)
# ---------------------------------------------------------------------------
class WheelRaster:
    def __init__(self, bounds, res):
        minx, miny, maxx, maxy = bounds
        self.minx, self.miny, self.res = minx, miny, res
        self.nx, self.ny = int(np.ceil((maxx - minx) / res)), int(np.ceil((maxy - miny) / res))

    def blank(self):
        return np.zeros((self.ny, self.nx), bool)

    def _idx(self, xy):
        ix = ((xy[:, 0] - self.minx) / self.res).astype(int)
        iy = ((xy[:, 1] - self.miny) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, ok

    def wheel_samples(self, line, gauge, step):
        """Sample points and unit normals along both wheel lines of a path."""
        if line.length < 1e-6:
            return []
        pts, ds = sample(line, step)
        if len(pts) < 2:
            return []
        seg = np.diff(pts, axis=0)
        t = np.r_[seg[:1], 0.5 * (seg[1:] + seg[:-1]), seg[-1:]]
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-12)[:, None]
        nrm = np.c_[-t[:, 1], t[:, 0]]
        return [(pts + s * gauge / 2 * nrm, nrm, ds) for s in (-1, 1)]

    def paint(self, mask, lines, gauge, half_width, step=None):
        step = self.res / 2 if step is None else step
        lat = np.arange(-half_width, half_width + 1e-9, self.res / 2)
        for ln in lines:
            for w, nrm, _ in self.wheel_samples(ln, gauge, step):
                for o in lat:
                    ix, iy, ok = self._idx(w + o * nrm)
                    mask[iy[ok], ix[ok]] = True
        return mask

    def off_length(self, mask, lines, gauge, step=0.25):
        """Mean (over the two wheels) length of wheel line whose sample cell is not in `mask`."""
        tot = 0.0
        for ln in lines:
            for w, _, ds in self.wheel_samples(ln, gauge, step):
                ix, iy, ok = self._idx(w)
                off = ~ok
                off[ok] = ~mask[iy[ok], ix[ok]]
                tot += off.sum() * ds
        return tot / 2


def path_lines(track, extra=()):
    return list(track.pieces()) + list(extra)


# ---------------------------------------------------------------------------
# CTF drive-around variants
# ---------------------------------------------------------------------------
def ctf_params(w_s, w_b, layout, r_ctf):
    """Lane-envelope clearance half-width and fillet / corner radius for B3c and B3n."""
    n = int(round(w_s / w_b))
    dm = d_max(n, w_b, layout)
    h_env = max(w_s / 2, dm + w_b / 2)
    return dict(n=n, d_max=dm, h_env=h_env, rho=r_ctf + dm)


def _ctf_sets_and_valid(obstacles, margin, w_s, rho, h_env, allowed_poly):
    half = [h_env] * len(obstacles)
    sets = detour_sets(obstacles, margin, w_s, rho, quad_segs=64, half=half)
    hard = [o["poly"].buffer(h_env - 1e-6) for o in obstacles]
    hard_tree = shapely.STRtree(hard) if hard else None

    def is_valid(piece):
        if not allowed_poly.covers(piece):
            return False
        return hard_tree is None or len(hard_tree.query(piece, predicate="intersects")) == 0
    return sets, is_valid


def make_b3c(track, obstacles, margin, w_s, rho, h_env, allowed_poly, step=0.5):
    """B3 with fillet rho and lane-envelope clearance h_env (every obstacle class I for the lanes)."""
    sets, is_valid = _ctf_sets_and_valid(obstacles, margin, w_s, rho, h_env, allowed_poly)
    segs, unplaced = [], 0
    for g, kind in track.segs:
        if kind != "swath":
            segs.append((g, kind))
            continue
        c = list(g.coords)
        for a, b in zip(c[:-1], c[1:]):
            g2, u = detour_line_convex(LineString([a, b]), sets, rho, is_valid, step, mergeable=sets)
            segs.append((g2, kind))
            unplaced += u
    return Track(segs, track.n_rev), unplaced


def make_b3n(track, obstacles, margin, w_s, w_b, rho, h_env, r_ctf, d_max_, s_min, allowed_poly,
             step=0.5, max_iter=40, max_depth=None):
    """Nested drive-around: a detouring pass pushes its neighbour on the bulge side out of
    conv(bump) + B(c_sep), c_sep = max(w_s - w_b + s_min, r_ctf + d_max), on the forced side, so lanes of
    adjacent passes stay >= s_min apart and never cross. The bump decays by ~(w_b - s_min) per pass.
    Returns (Track, n_unplaced, stats) with stats = dict(n_nested_passes, max_depth, depth_by_group)."""
    sets, is_valid = _ctf_sets_and_valid(obstacles, margin, w_s, rho, h_env, allowed_poly)
    c_sep = max(w_s - w_b + s_min, r_ctf + d_max_)
    idx = [i for i, (g, k) in enumerate(track.segs) if k == "swath"]
    lines = [LineString([track.segs[i][0].coords[0], track.segs[i][0].coords[-1]]) for i in idx]
    geo = []
    for ln in lines:
        a, b = map(np.asarray, (ln.coords[0], ln.coords[-1]))
        u_c, n_c = canonical(b - a)
        geo.append(dict(u=u_c, n=n_c, c=float(0.5 * (a + b) @ n_c), x0=float(min(a @ u_c, b @ u_c)),
                        x1=float(max(a @ u_c, b @ u_c))))
    group_of_set = list(range(len(sets)))
    forced = {i: [] for i in range(len(lines))}     # line -> [(S', dir, depth, root group)]
    result = [None] * len(lines)

    def neighbour(i, v, bump):
        p = geo[i]
        bx = shapely.get_coordinates(bump) @ p["u"]
        lo, hi = bx.min(), bx.max()
        best = None
        for j, q in enumerate(geo):
            if j == i or abs(p["u"] @ q["n"]) > 0.02:
                continue
            dc = (q["c"] - p["c"]) * float(np.sign(v @ p["n"]))
            if 0.25 * w_s < dc < 1.75 * w_s and min(hi, q["x1"]) > max(lo, q["x0"]):
                if best is None or dc < best[0]:
                    best = (dc, j)
        return None if best is None else best[1]

    def plan(i):
        fs = forced[i]
        det = []
        g2, u = detour_line_convex(lines[i], sets + [f[0] for f in fs], rho, is_valid, step, mergeable=sets,
                                   force=[None] * len(sets) + [f[1] for f in fs], details=det)
        return g2, u, det

    def root_of(prof):
        hits = [k for k, S in enumerate(sets) if S.buffer(rho + 0.5).intersects(prof)]
        return group_of_set[hits[0]] if hits else -1

    # 1. every pass with its own obstacle detours (= B3c); 2. outward-only propagation: a bump pushes the
    # neighbour on its bulge side; the neighbour's new bump in the SAME direction is propagated further.
    # Chains only move away from their source, so this terminates (at the latest at the field edge).
    unpl, dets = {}, {}
    for i in range(len(lines)):
        result[i], unpl[i], dets[i] = plan(i)
    queue = [(i, d, 0, root_of(d["profile"])) for i in range(len(lines)) for d in dets[i]]
    depth_by_group, conflicts, steps = {}, 0, 0
    for _, _, dep, rt in queue:
        depth_by_group.setdefault(rt, 0)
    while queue and steps < max_iter * max(1, len(lines)):
        i, dtl, depth, root = queue.pop(0)
        steps += 1
        if max_depth is not None and depth >= max_depth:
            continue                         # B3n_k: chain stops; remaining conflicts stay as crossings
        j = neighbour(i, dtl["dir"], dtl["profile"])
        if j is None:
            continue
        Sp = dtl["profile"].convex_hull.buffer(c_sep - 1e-6, quad_segs=32)
        if not Sp.intersects(lines[j]):
            continue
        forced[j].append((Sp, dtl["dir"], depth + 1, root))
        result[j], unpl[j], dets[j] = plan(j)
        near = Sp.buffer(rho + 1.0)
        for d in dets[j]:
            if not d["profile"].intersects(near):
                continue
            if float(np.dot(d["dir"], dtl["dir"])) > 0:
                queue.append((j, d, depth + 1, root))
                depth_by_group[root] = max(depth_by_group.get(root, 0), depth + 1)
            else:
                conflicts += 1
    unplaced = sum(unpl.values())
    segs, k = [], 0
    for i, (g, kind) in enumerate(track.segs):
        if kind == "swath":
            segs.append((result[k], kind))
            k += 1
        else:
            segs.append((g, kind))
    stats = dict(n_nested_passes=sum(1 for v in forced.values() if v),
                 max_depth=max(depth_by_group.values(), default=0),
                 depth_by_group=";".join(f"{g}:{d}" for g, d in sorted(depth_by_group.items())),
                 iterations=steps, conflicts=conflicts)
    return Track(segs, track.n_rev), unplaced, stats
