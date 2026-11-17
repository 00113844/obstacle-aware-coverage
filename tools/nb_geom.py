# ---- Track, drive-around detours & section-control simulator (pure numpy/shapely) ----
# This cell does not depend on Fields2Cover: it post-processes any path given as a Track.
import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import unary_union


class Track:
    """An ordered drivable path: list of (LineString, kind) with kind in {'swath', 'turn'}.
    Consecutive pieces of the same kind are merged. `n_rev` counts reversing manoeuvres."""

    def __init__(self, segs, n_rev=0):
        merged = []
        for geom, kind in segs:
            pts = np.asarray(geom.coords if hasattr(geom, "coords") else geom, float)[:, :2]
            if len(pts) < 2 or LineString(pts).length < 1e-6:
                continue
            if merged and merged[-1][1] == kind:
                prev = merged[-1][0]
                if np.allclose(prev[-1], pts[0]):
                    pts = pts[1:]
                merged[-1][0] = np.vstack([prev, pts])
            else:
                merged.append([pts, kind])
        self.segs = [(LineString(p), k) for p, k in merged if len(p) >= 2]
        self.n_rev = n_rev

    def pieces(self, kind=None):
        return [g for g, k in self.segs if kind is None or k == kind]

    def length(self, kind=None):
        return sum(g.length for g in self.pieces(kind))

    @property
    def n_turns(self):
        kinds = [k for _, k in self.segs]
        return sum(1 for i in range(1, len(kinds) - 1)
                   if kinds[i] == "turn" and kinds[i - 1] == "swath" and kinds[i + 1] == "swath")

    @property
    def line(self):
        pts = [np.asarray(self.segs[0][0].coords)]
        for g, _ in self.segs[1:]:
            c = np.asarray(g.coords)
            pts.append(c[1:] if np.allclose(pts[-1][-1], c[0]) else c)
        return LineString(np.vstack(pts))

    @property
    def start(self):
        return self.segs[0][0].coords[0]

    @property
    def end(self):
        return self.segs[-1][0].coords[-1]


# ---------------------------------------------------------------------------
# Drive-around detour: straight -> fillet (rho) -> arc around tree (R) -> fillet -> straight
# ---------------------------------------------------------------------------
def _arc(cx, cy, r, a0, a1, step):
    n = max(2, int(np.ceil(abs(a1 - a0) * r / step)) + 1)
    t = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def detour_profile(a, R, rho, step=0.5):
    """Detour in the tree frame: tree at the origin, the swath is the line y = a (|a| < R),
    travelling in +x; the detour bulges towards +y. Returns (points, half_length xf)."""
    h = a + rho
    xf = np.sqrt((R + rho) ** 2 - h ** 2)
    th1 = np.arctan2(-h, xf)                     # fillet 1: CCW from -pi/2 to th1
    while th1 <= -np.pi / 2:
        th1 += 2 * np.pi
    f1 = _arc(-xf, h, rho, -np.pi / 2, th1, step)
    phi1 = np.arctan2(h, -xf)                    # tree arc: CW from phi1 to phi2
    if phi1 < 0:
        phi1 += 2 * np.pi
    phi2 = np.pi - phi1
    while phi2 >= phi1:
        phi2 -= 2 * np.pi
    tc = _arc(0.0, 0.0, R, phi1, phi2, step)
    th2 = np.arctan2(-h, -xf)                    # fillet 2: CCW from th2 to -pi/2
    e2 = -np.pi / 2
    while e2 <= th2:
        e2 += 2 * np.pi
    f2 = _arc(xf, h, rho, th2, e2, step)
    return np.vstack([f1, tc[1:], f2[1:]]), xf


def _enclose(c1, c2):
    """Smallest circle (x, y, R) enclosing two circles."""
    (x1, y1, r1), (x2, y2, r2) = c1, c2
    d = np.hypot(x2 - x1, y2 - y1)
    if d + r2 <= r1:
        return c1
    if d + r1 <= r2:
        return c2
    R = (d + r1 + r2) / 2
    t = (R - r1) / d
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1), R)


def detour_line(line, circles, rho, is_valid, step=0.5):
    """Splice detours into a straight swath `line` around every clearance circle (x, y, R)
    it crosses. Tries the short side first, then the long side. Returns (LineString, issues)."""
    p0, p1 = np.asarray(line.coords[0], float), np.asarray(line.coords[-1], float)
    L = np.hypot(*(p1 - p0))
    u = (p1 - p0) / L
    n = np.array([-u[1], u[0]])
    to_world = lambda xy: p0 + np.outer(xy[:, 0], u) + np.outer(xy[:, 1], n)

    hits = []
    for c in circles:
        tx, ty = (np.array(c[:2]) - p0) @ u, (np.array(c[:2]) - p0) @ n
        if abs(ty) < c[2] and -c[2] < tx < L + c[2]:
            hits.append(list(c))
    # merge circles whose detours would overlap along the swath
    changed = True
    while changed and len(hits) > 1:
        changed = False
        hits.sort(key=lambda c: (np.array(c[:2]) - p0) @ u)
        for i in range(len(hits) - 1):
            a, b = hits[i], hits[i + 1]
            ta, tb = (np.array(a[:2]) - p0) @ u, (np.array(b[:2]) - p0) @ u
            if tb - ta < a[2] + b[2] + 2 * rho:
                hits[i:i + 2] = [list(_enclose(tuple(a), tuple(b)))]
                changed = True
                break

    pts, issues = [p0], []
    for cx, cy, R in sorted(hits, key=lambda c: (np.array(c[:2]) - p0) @ u):
        tx, ty = (np.array([cx, cy]) - p0) @ u, (np.array([cx, cy]) - p0) @ n
        done = False
        for a in (abs(ty), -abs(ty)):          # short side, then long side
            if abs(a) >= R:
                continue
            s = (-ty / a) if abs(a) > 1e-9 else (1.0 if a >= 0 else -1.0)
            prof, xf = detour_profile(a, R, rho, step)
            if tx - xf < 0 or tx + xf > L:     # detour does not fit inside this swath
                continue
            local = np.c_[tx + prof[:, 0], ty + s * prof[:, 1]]
            piece = to_world(local)
            if is_valid(LineString(piece)):
                pts.extend(piece)
                done = True
                break
        if not done:
            issues.append((cx, cy, R))
    pts.append(p1)
    return LineString(np.asarray(pts)), issues


def make_drive_around(track, trees, margin, width, r_min, allowed_poly, step=0.5):
    """Strategy B3: keep every swath of `track` and drive around trees on the swath itself."""
    circles = [(x, y, max(r + margin + width / 2, r_min)) for x, y, r in trees]
    hard = [(x, y, r + width / 2) for x, y, r in trees]          # canopy + half implement

    def is_valid(piece):
        if not allowed_poly.covers(piece):
            return False
        return all(piece.distance(Point(x, y)) >= R - 1e-6 for x, y, R in hard)

    segs, issues = [], []
    for g, kind in track.segs:
        if kind == "swath":
            c = list(g.coords)
            for a, b in zip(c[:-1], c[1:]):          # each straight piece of the swath run
                g2, iss = detour_line(LineString([a, b]), circles, r_min, is_valid, step)
                segs.append((g2, kind))
                issues += iss
        else:
            segs.append((g, kind))
    return Track(segs, track.n_rev), issues


def max_curvature(line, ds=1.0):
    """Approximate max curvature [1/m] of a polyline (resampled every ds metres)."""
    d = np.arange(0, line.length, ds)
    p = shapely.get_coordinates(shapely.line_interpolate_point(line, d))
    if len(p) < 3:
        return 0.0
    v1, v2 = p[1:-1] - p[:-2], p[2:] - p[1:-1]
    ang = np.abs(np.arctan2(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0], (v1 * v2).sum(1)))
    return float(ang.max() / ds)


# ---------------------------------------------------------------------------
# Section-control simulator (raster)
# ---------------------------------------------------------------------------
def simulate_coverage(track, field_shp, obstacles, width, n_sections=6, res=1.0,
                      section_control=True, extra_passes=(), record_window=None):
    """Drive the boom along every working piece of `track` (+ `extra_passes`) and count, per
    raster cell, how many distinct passes applied product to it.
    A section is ON if (section control) the majority of its footprint is inside the field,
    outside every obstacle and not yet covered by an earlier pass; otherwise it is always ON."""
    minx, miny, maxx, maxy = field_shp.buffer(width).bounds
    nx, ny = int(np.ceil((maxx - minx) / res)), int(np.ceil((maxy - miny) / res))
    xs, ys = minx + (np.arange(nx) + 0.5) * res, miny + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(field_shp, X, Y)
    obst = (shapely.contains_xy(obstacles, X, Y)
            if obstacles is not None and not obstacles.is_empty else np.zeros_like(inside))
    target = inside & ~obst
    count = np.zeros((ny, nx), np.int16)
    h = res / 4                     # boom & track sampled at 1/4 cell -> 16 hits per full cell
    full = (res / h) ** 2
    offs = -width / 2 + h / 2 + np.arange(int(round(width / h))) * h
    sec = np.minimum(((offs + width / 2) // (width / n_sections)).astype(int), n_sections - 1)
    sec_n = np.bincount(sec, minlength=n_sections)
    sections_on = sections_total = 0
    states = []                      # (pass id, x, y, dx, dy, on per section) inside record_window

    passes = list(track.pieces("swath")) + list(extra_passes)
    for pid, line in enumerate(passes):
        L = line.length
        if L < res:
            continue
        d = np.arange(0, L, h)
        p = shapely.get_coordinates(shapely.line_interpolate_point(line, d))
        q = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(d + h, L)))
        dv = q - p
        dv[-1] = dv[-2] if len(dv) > 1 else dv[-1]
        dv /= np.maximum(np.hypot(dv[:, 0], dv[:, 1]), 1e-12)[:, None]
        bx = p[:, :1] - dv[:, 1:2] * offs            # (samples, boom points)
        by = p[:, 1:2] + dv[:, 0:1] * offs
        ix, iy = ((bx - minx) / res).astype(int), ((by - miny) / res).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
        if section_control:   # decisions use passes completed before this one
            good = ok & target[iy, ix] & (count[iy, ix] == 0)
            frac = np.stack([good[:, sec == s].mean(1) for s in range(n_sections)], 1)
            on = (frac >= 0.5)[:, sec]
        else:
            on = np.ones_like(ok)
        if record_window is not None:
            m = shapely.contains_xy(record_window, p[:, 0], p[:, 1])
            for i in np.flatnonzero(m):
                states.append((pid, p[i, 0], p[i, 1], dv[i, 0], dv[i, 1], on[i, sec_n.cumsum() - 1].copy()))
        sections_on += on[:, sec_n.cumsum() - 1].sum()
        sections_total += on.shape[0] * n_sections
        flat = (iy * nx + ix)[on & ok]
        cells, hits = np.unique(flat, return_counts=True)
        # applied if > half the cell is under an ON section; an exact half goes to the first pass
        claim = (hits > full / 2) | ((hits == full / 2) & (count.flat[cells] == 0))
        count.flat[cells[claim]] += 1

    cell = res * res
    t = target.sum()
    return dict(
        count=count, target=target, inside=inside, obst=obst,
        extent=(minx, minx + nx * res, miny, miny + ny * res),
        target_ha=t * cell / 1e4,
        missed_pct=100 * (target & (count == 0)).sum() / t,
        overlap_pct=100 * (target & (count >= 2)).sum() / t,
        offtarget_m2=float(((~target) & (count > 0)).sum() * cell),
        applied_ha=float(count.sum() * cell / 1e4),
        sections_on_pct=100 * sections_on / max(sections_total, 1),
        states=states, offs=offs, sec=sec,
    )


def turn_length_inside(track, poly):
    """Non-working distance driven inside `poly` (e.g. over the crop)."""
    return sum(g.intersection(poly).length for g in track.pieces("turn"))


def length_outside(track, poly):
    """Distance driven outside `poly` (e.g. outside the paddock)."""
    return sum(g.difference(poly).length for g, _ in track.segs)


def tree_clearance(track, trees, width):
    """Edge clearance (distance to trunk - canopy radius - half implement) per tree."""
    ln = track.line
    return [ln.distance(Point(x, y)) - r - width / 2 for x, y, r in trees]
