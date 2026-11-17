"""Generate Fields2Cover_Paddock_Trees.ipynb (Colab tutorial), v2.

Run:  python build_notebook.py <output.ipynb>
The pure-geometry helpers are read verbatim from nb_geom.py (tested locally with test_geom).
"""
import json
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []


def md(src):
    cells.append(("markdown", textwrap.dedent(src).strip("\n")))


def code(src):
    cells.append(("code", textwrap.dedent(src).strip("\n")))


# =============================================================================
# Title
# =============================================================================
md(r"""
# 🚜 Optimal Coverage Path Planning for a Paddock with Trees
### A hands-on Fields2Cover tutorial (Google Colab) — v2

This notebook uses the open-source library [**Fields2Cover**](https://github.com/Fields2Cover/Fields2Cover) (F2C, v2.1.1) to plan an
**optimal coverage path** for an agricultural machine. It starts with a plain paddock and then plans the same paddock with **pinpointed paddock trees**.
It also adds what F2C does not provide: **section control**, a **field entrance (gate)**, and an alternative **drive-around** strategy for trees.

| Part | What you will do |
|---|---|
| **0** | Build and install Fields2Cover inside Colab (one-off, ≈10–15 min) |
| **1** | Learn the maths of Coverage Path Planning (CPP), turn feasibility and section control |
| **2** | Configure the paddock, the machine, the gate and the trees, and load the helpers |
| **A** | Plan an **open paddock**: headlands → swath angle → route from the gate → feasible turns |
| **B** | Plan around **trees** with three strategies: **B1** holes, **B2** cell decomposition, **B3** drive-around + section control |
| **C** | Compare all of them (distance, time, driving over crop, coverage and overlap with and without section control), validate and export |

> **Companion handout.** `f2c_tutorial.pdf` (LaTeX) explains every step, with schematics and exercises.

**Reference.** G. Mier, J. Valente, S. de Bruin (2023). *Fields2Cover: An open-source coverage path planning library for
unmanned agricultural vehicles.* IEEE RA-L 8(4), 2166–2172.
""")

# =============================================================================
# Part 0
# =============================================================================
md(r"""
---
## Part 0 · Install Fields2Cover in Colab

F2C is C++ with Python (SWIG) bindings. PyPI ships only the **source**, so Colab compiles it. The setup cell:
1. installs the Linux dependencies (GDAL, GEOS, Eigen, TBB, TinyXML2, nlohmann-json);
2. downloads the pre-built **Google OR-Tools C++** release (the route optimiser) into `/opt/or-tools`;
3. builds a wheel of **F2C v2.1.1** from GitHub and installs it.

⏱️ The first run takes **≈10–15 min**. The cell does nothing if `fields2cover` is already importable, for example after a kernel restart.
💾 Set `USE_DRIVE_CACHE = True` to keep the wheel and OR-Tools in `MyDrive/F2C_cache/`. Later sessions then install in under a minute.
""")

code(r'''
# ---- Part 0: build & install Fields2Cover (idempotent) ---------------------
import importlib.util, os, subprocess, sys, time, glob, shutil

F2C_GIT_TAG     = "v2.1.1"
ORTOOLS_URL     = ("https://github.com/google/or-tools/releases/download/v9.9/"
                   "or-tools_amd64_ubuntu-22.04_cpp_v9.9.3963.tar.gz")  # version F2C v2.1.1 is tested with
USE_DRIVE_CACHE = False
CACHE_DIR       = "/content/drive/MyDrive/F2C_cache"

def sh(cmd, env=None):
    t0 = time.time()
    print(f"$ {cmd[:110]}{'...' if len(cmd) > 110 else ''}")
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, env={**os.environ, **(env or {})})
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:])
        raise RuntimeError(f"Command failed ({r.returncode}): {cmd}")
    print(f"  ✓ done in {time.time()-t0:.0f}s")

def ensure_ortools(cache):
    if os.path.exists("/opt/or-tools/lib/cmake/ortools"):
        return
    tgz = os.path.join(cache, "ortools.tar.gz") if cache else None
    if not (tgz and os.path.exists(tgz)):
        tgz = "/tmp/ortools.tar.gz"
        sh(f"wget -q -O {tgz} {ORTOOLS_URL}")
        if cache: shutil.copy(tgz, os.path.join(cache, "ortools.tar.gz"))
    sh(f"mkdir -p /opt/or-tools && tar -xzf {tgz} -C /opt/or-tools --strip-components=1")
    sh("echo /opt/or-tools/lib > /etc/ld.so.conf.d/ortools.conf && ldconfig")

if importlib.util.find_spec("fields2cover") is None:
    T0 = time.time(); cache = None
    if USE_DRIVE_CACHE:
        from google.colab import drive
        drive.mount("/content/drive"); os.makedirs(CACHE_DIR, exist_ok=True); cache = CACHE_DIR
    sh("cat /etc/os-release | grep PRETTY_NAME")
    sh("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y "
       "build-essential cmake git libgdal-dev libgeos-dev libeigen3-dev libtbb-dev "
       "libtinyxml2-dev nlohmann-json3-dev libboost-dev")
    ensure_ortools(cache)
    wheels = sorted(glob.glob(os.path.join(cache, "fields2cover-*.whl"))) if cache else []
    if not wheels:
        env = {"CMAKE_PREFIX_PATH": "/opt/or-tools", "CMAKE_BUILD_PARALLEL_LEVEL": str(os.cpu_count() or 2)}
        sh("pip wheel --no-deps -w /content/f2c_wheels "
           f"'git+https://github.com/Fields2Cover/Fields2Cover.git@{F2C_GIT_TAG}'", env=env)
        wheels = sorted(glob.glob("/content/f2c_wheels/fields2cover-*.whl"))
        if cache: [shutil.copy(w, cache) for w in wheels]
    else:
        ensure_ortools(cache)
    sh(f"pip install -q --no-deps {wheels[-1]}")
    print(f"\nInstalled in {(time.time()-T0)/60:.1f} min")
else:
    ensure_ortools(None)   # the OR-Tools runtime libraries must exist in every new VM

import fields2cover as f2c
print("Fields2Cover version:", getattr(f2c, "__version__", "unknown"))
''')

code(r'''
!pip -q install "shapely>=2.0" matplotlib pandas
''')

# =============================================================================
# Part 1 – Theory
# =============================================================================
md(r"""
---
## Part 1 · The maths of Coverage Path Planning

F2C splits Coverage Path Planning into four modules that run one after another. Each module has its own objective:

$$
\underbrace{F}_{\text{field}}\xrightarrow{\textbf{HG}}\underbrace{F_{\text{in}}}_{\text{mainland}}
\xrightarrow{\textbf{SG}}\underbrace{\mathcal S=\{s_1,\dots,s_N\}}_{\text{swaths}}
\xrightarrow{\textbf{RP}}\underbrace{\pi}_{\text{order}}
\xrightarrow{\textbf{PP}}\underbrace{P(t)}_{\text{drivable path}}
$$

### 1.1 Headlands (HG)
$F_{\text{in}}=F\ominus\mathcal B(k\,w)$, where $w$ is the implement width and $\ominus$ is erosion (a negative buffer). The ring at $\tfrac{k}{2}w$ is the
**headland track** the machine drives along between swaths.

### 1.2 Swaths (SG)
For a driving direction $\theta$, $\;N(\theta)\approx\lceil W_\perp(\theta)/w_c\rceil$ and $\;\theta^*=\arg\min_{\theta\in[0,\pi)}J(\theta)$.
Here $J$ is either `OBJ_NSwathModified` (the number of swaths, i.e. turns) or `OBJ_SwathLength`.

### 1.3 Route (RP) and the gate
`RP_RoutePlannerBase` orders the swaths using OR-Tools (a travelling-salesman-like problem on the headland graph):

$$\min_\pi\; d_{hl}(g,\text{start}(s_{\pi_1}))+\sum_{j=1}^{N-1} d_{hl}\big(\text{end}(s_{\pi_j}),\text{start}(s_{\pi_{j+1}})\big)+d_{hl}(\text{end}(s_{\pi_N}),g)$$

$g$ is the **gate** (field entrance), set with `setStartAndEndPoint`. With `search_for_optimum=True`, OR-Tools uses *guided local search* for
`time_limit_seconds` instead of stopping at the first solution.

### 1.4 Turns (PP) and when they become loops
The machine has a curvature limit $|\kappa|\le 1/r_{\min}$. A forward-only U-turn between neighbouring swaths needs the swaths to be at least $2r_{\min}$ apart.
When $w_c<2r_{\min}$ the planner has to drive an **Ω-turn** (a bulb), which extends into the headland by

$$D_\Omega = r_{\min}+\sqrt{4r_{\min}^2-\left(r_{\min}+\tfrac{w_c}{2}\right)^2}.$$

The headland must contain the bulb plus half the implement: $\;k\,w\ \ge\ D_\Omega+\tfrac{w}{2}\;\Rightarrow\;k=\max\!\left(3,\left\lceil\frac{D_\Omega+w/2}{w}\right\rceil\right)$.
The alternative is **Reeds–Shepp** turns, which allow reversing and so replace the bulb with a three-point turn.

$$L=\sum L_{\text{swath}}+\sum L_{\text{turn}},\qquad T=\frac{L_{\text{swath}}}{v_{\text{cruise}}}+\frac{L_{\text{turn}}}{v_{\text{turn}}}+t_{\text{rev}}\,n_{\text{rev}},\qquad \eta=\frac{L_{\text{swath}}}{L}$$

### 1.5 Trees
Each tree is a disc $B_i=\{\mathbf p:\lVert\mathbf p-\mathbf t_i\rVert\le r_i+m\}$, and the workable field is $F'=F\setminus\bigcup_iB_i$. The implement edge must clear the canopy:
$\min_t\lVert P(t)-\mathbf t_i\rVert\ge r_i+\tfrac w2$.

### 1.6 Section control (not modelled by F2C)
The boom of width $w_c$ is split into $n_s$ sections $S_1,\dots,S_{n_s}$. Section $j$ is **on** at time $t$ if most of its footprint is over target ground
that has not yet been covered:

$$\text{on}_j(t)=\mathbb 1\!\left[\frac{|S_j(t)\cap F'\cap U(t)|}{|S_j(t)|}\ge\tfrac12\right],\qquad U(t)=\text{area not yet covered at time }t.$$

Writing $c(\mathbf p)$ for how many passes applied product at point $\mathbf p$, we measure

$$\text{missed }M=\frac{|\{c=0\}\cap F'|}{|F'|},\qquad \text{overlap }O=\frac{|\{c\ge2\}\cap F'|}{|F'|},\qquad \text{product saved}=1-\frac{\int c_{\text{SC}}}{\int c_{\text{no SC}}}.$$
""")

# =============================================================================
# Part 2 – Config & helpers
# =============================================================================
md(r"""
---
## Part 2 · Configuration and helpers
Every tunable value lives in `CONFIG`. Coordinates are **local metres** (x = East, y = North). The **gate** is where the machine enters and leaves the paddock.
""")

code(r'''
import math, json, time
import numpy as np
import pandas as pd

CONFIG = dict(
    paddock=[(0, 0), (410, -25), (465, 110), (450, 290), (300, 330),
             (215, 265), (95, 310), (-25, 175), (0, 0)],
    gate=(0.0, 0.0),           # field entrance (x, y); the route starts and ends here
    width=6.0,                 # implement width w [m]
    cov_width=6.0,             # coverage width w_c [m]
    min_turn_radius=6.0,       # r_min [m]
    max_diff_curv=0.1,         # sigma_max [1/m^2] (continuous-curvature turns)
    cruise_vel=2.5,            # working speed [m/s]
    turn_vel=1.2,              # turning speed [m/s]
    reverse_penalty_s=10.0,    # time lost per reversal [s]
    headland_factor="auto",    # k in w_hl = k*w, or "auto" (from the Omega-turn depth)
    allow_reverse=True,        # also try Reeds-Shepp (3-point) turns when w_c < 2 r_min
    rp_time=30,                # OR-Tools time budget per route [s]
    rp_optimum=True,           # guided local search (better) instead of the first solution
    robot_tuning={},           # optional corner smoothing, e.g. {"setMaxCornerCut": 4.0, "setTurnSlack": 0.5}
    trees=[(105, 70, 7.0), (190, 150, 9.0), (300, 65, 6.0), (355, 205, 8.0),
           (80, 205, 5.5), (260, 235, 6.5), (395, 60, 5.0)],
    tree_margin=3.0,           # safety margin m [m]
    random_trees=0,            # >0 -> N random trees instead of the list above
    n_sections=6,              # boom sections for the section-control simulation
    raster_res=1.0,            # coverage raster cell [m]
    cover_headlands=True,      # add headland rounds (driven last) to the coverage analysis
    # strategy selection C = c_t*T[min] + c_p*applied_with_SC[ha] + c_m*missed_area[ha]
    c_t=2.5,                   # cost per minute of machine time
    c_p=40.0,                  # cost per hectare of product applied
    c_m=300.0,                 # cost per hectare left untreated
    seed=42,
)
rng = np.random.default_rng(CONFIG["seed"])
W, WC, RMIN = CONFIG["width"], CONFIG["cov_width"], CONFIG["min_turn_radius"]
print(f"w = {W} m, w_c = {WC} m, r_min = {RMIN} m  ->  "
      + ("w_c < 2 r_min: forward U-turns become Omega loops" if WC < 2 * RMIN else "w_c >= 2 r_min: plain U-turns fit"))
''')

md(r"""
The next cell holds the **pure-geometry toolbox** (numpy and shapely only):
- `Track`: a drivable path stored as working and non-working pieces;
- `make_drive_around`: Strategy B3;
- `simulate_coverage`: the section-control simulator;
- metrics for driving over the crop and driving outside the paddock.
""")

with open(os.path.join(HERE, "nb_geom.py"), encoding="utf-8") as fh:
    code(fh.read())

code(r'''
# ---- F2C <-> shapely / Track helpers ------------------------------------------
from shapely import wkt, force_2d
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import unary_union
from types import SimpleNamespace

def to_shapely(g):
    """Any F2C geometry -> shapely geometry (via WKT), dropping Z."""
    return force_2d(wkt.loads(g.exportToWkt()))

def make_ring(coords):
    coords = [(float(x), float(y)) for x, y in coords]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return f2c.LinearRing(f2c.VectorPoint([f2c.Point(x, y) for x, y in coords]))

def circle_coords(x, y, r, n=36):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = [(x + r * np.cos(a), y + r * np.sin(a)) for a in t]
    return pts + [pts[0]]

# SWIG memory rule: .at(i) / .getState(i) return references into C++ containers. If the container
# is a temporary (e.g. .flatten()), it is freed and the kernel crashes. Always clone / copy out.
def swath_list(sw):
    if isinstance(sw, f2c.Route):
        return [s for k in range(sw.sizeVectorSwaths()) for s in swath_list(sw.getSwaths(k))]
    if isinstance(sw, f2c.SwathsByCells):
        return [s for c in range(sw.size()) for s in swath_list(sw.at(c))]
    return [sw.at(i).clone() for i in range(sw.size())]

def swath_lines(sw):
    return [to_shapely(s.getPath()) for s in swath_list(sw)]

def path_states(path):
    out = []
    for i in range(path.size()):
        s = path.getState(i)
        out.append(SimpleNamespace(x=s.point.getX(), y=s.point.getY(), len=float(s.len),
                                   is_swath=(s.type == f2c.PathSectionType_SWATH),
                                   is_rev=(s.dir == f2c.PathDirection_BACKWARD)))
    return out

def track_from_path(path):
    """F2C Path -> Track (working / non-working pieces + number of reversals)."""
    st = path_states(path)
    e = path.atEnd()
    pts = [(s.x, s.y) for s in st] + [(e.getX(), e.getY())]
    segs = [([pts[i], pts[i + 1]], "swath" if s.is_swath else "turn")
            for i, s in enumerate(st) if pts[i] != pts[i + 1]]
    rev = [s.is_rev for s in st]
    n_rev = sum(1 for i, r in enumerate(rev) if r and (i == 0 or not rev[i - 1]))
    return Track(segs, n_rev)

def as_track(x):
    return x if isinstance(x, Track) else track_from_path(x)
''')

code(r'''
# ---- Plotting helpers ---------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle, Polygon as MplPolygon
from matplotlib.lines import Line2D
from shapely.plotting import plot_polygon

COL = dict(bg="#f4f1e8", paddock="#cfe3b0", stripe="#b8d593", headland="#eadfb8",
           border="#3b4a2a", swath="#2f6db5", turn="#e07b39", hl_track="#8e5cb8",
           canopy="#2f6b2f", canopy_hi="#4f8f3f", trunk="#6b4a2b", buffer="#a0522d", gate="#111111")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.facecolor": COL["bg"], "figure.facecolor": "white",
                     "animation.embed_limit": 60})

def _polys(geom):
    return list(geom.geoms) if hasattr(geom, "geoms") else [geom]

def draw_cells(ax, cells_or_shp, fc=COL["paddock"], ec=COL["border"], lw=1.5, alpha=1.0, z=1, **kw):
    g = cells_or_shp if hasattr(cells_or_shp, "geom_type") else to_shapely(cells_or_shp)
    for p in _polys(g):
        if not p.is_empty:
            plot_polygon(p, ax=ax, add_points=False, facecolor=fc, edgecolor=ec,
                         linewidth=lw, alpha=alpha, zorder=z, **kw)

def draw_swaths(ax, sw, color=None, cmap="viridis", lw=1.4, numbers=False, z=4):
    lines = swath_lines(sw) if not isinstance(sw, list) else sw
    cm = plt.get_cmap(cmap)
    for i, ln in enumerate(lines):
        c = color or cm(i / max(1, len(lines) - 1))
        x, y = ln.xy
        ax.plot(x, y, color=c, lw=lw, zorder=z, solid_capstyle="butt")
        if numbers and (i % max(1, len(lines) // 12) == 0):
            ax.text(x[0], y[0], str(i + 1), fontsize=7, color=c, ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=c, lw=0.8), zorder=z + 1)

def draw_coverage(ax, lines, width, z=2):
    for i, ln in enumerate(lines):
        plot_polygon(ln.buffer(width / 2, cap_style="flat"), ax=ax, add_points=False, linewidth=0,
                     zorder=z, facecolor=COL["stripe"] if i % 2 else "#c4dc9f")

def draw_path(ax, path_or_track, lw=1.3, arrows=True, z=5):
    tr = as_track(path_or_track)
    segs = [np.asarray(g.coords) for g, _ in tr.segs]
    cols = [COL["swath"] if k == "swath" else COL["turn"] for _, k in tr.segs]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, zorder=z, capstyle="round"))
    if arrows:
        sw = [g for g in tr.pieces("swath") if g.length > 20]
        for g in sw[::max(1, len(sw) // 25)]:
            a, b = g.interpolate(0.5, normalized=True), g.interpolate(0.5 * g.length + 3)
            ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y),
                        arrowprops=dict(arrowstyle="-|>", color=COL["swath"], lw=0.8), zorder=z + 1)
    ax.plot(*tr.start, marker="o", ms=9, mfc="#27ae60", mec="white", zorder=z + 2)
    ax.plot(*tr.end, marker="s", ms=8, mfc="#c0392b", mec="white", zorder=z + 2)
    ax.plot(*CONFIG["gate"], marker="*", ms=16, mfc="gold", mec=COL["gate"], zorder=z + 3)

def draw_trees(ax, trees, margin, z=6):
    for (x, y, r) in trees:
        ax.add_patch(Circle((x, y), r + margin, fc="none", ec=COL["buffer"], lw=1, ls="--", zorder=z))
        ax.add_patch(Circle((x, y), r, fc=COL["canopy"], ec="#1d4a1d", lw=0.8, alpha=0.95, zorder=z))
        ax.add_patch(Circle((x - 0.25 * r, y + 0.25 * r), 0.55 * r, fc=COL["canopy_hi"], ec="none", alpha=0.8, zorder=z))
        ax.add_patch(Circle((x, y), max(0.6, 0.12 * r), fc=COL["trunk"], ec="none", zorder=z + 1))

def add_scalebar(ax, length=50, loc=(0.04, 0.05)):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    X, Y = x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0)
    ax.add_patch(Rectangle((X, Y), length / 2, 3, fc="black", zorder=20))
    ax.add_patch(Rectangle((X + length / 2, Y), length / 2, 3, fc="white", ec="black", lw=0.8, zorder=20))
    ax.text(X + length / 2, Y + 6, f"{length} m", ha="center", fontsize=8, zorder=20)

def add_north(ax, loc=(0.95, 0.88)):
    ax.annotate("N", xy=loc, xytext=(loc[0], loc[1] - 0.10), xycoords="axes fraction", ha="center",
                fontsize=11, fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black"), zorder=20)

def setup_ax(ax, title="", pad=25, bounds=None):
    ax.set_aspect("equal"); ax.set_title(title)
    if bounds is not None:
        minx, miny, maxx, maxy = bounds
        ax.set_xlim(minx - pad, maxx + pad); ax.set_ylim(miny - pad, maxy + pad)
    ax.set_xlabel("Easting [m]"); ax.set_ylabel("Northing [m]")
    ax.grid(color="white", lw=0.6, alpha=0.7); ax.set_axisbelow(True)

def draw_coverage_map(ax, sim, title):
    """Application count map: red = missed, green = once, orange = overlap; white = not target."""
    c = np.minimum(sim["count"], 2).astype(float)
    ax.imshow(np.ma.masked_where(~sim["target"], c), origin="lower", extent=sim["extent"],
              cmap=ListedColormap(["#e74c3c", "#9fd18b", "#f39c12"]), vmin=0, vmax=2, interpolation="nearest")
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

LEGEND_PATH = [Line2D([], [], color=COL["swath"], lw=2, label="Working (swath)"),
               Line2D([], [], color=COL["turn"], lw=2, label="Turn / headland travel"),
               Line2D([], [], marker="*", ls="", mfc="gold", mec="k", ms=12, label="Gate"),
               Line2D([], [], marker="o", ls="", mfc="#27ae60", mec="white", ms=8, label="Start"),
               Line2D([], [], marker="s", ls="", mfc="#c0392b", mec="white", ms=8, label="End")]
LEGEND_TREE = [Line2D([], [], marker="o", ls="", mfc=COL["canopy"], mec="#1d4a1d", ms=10, label="Tree canopy"),
               Line2D([], [], color=COL["buffer"], ls="--", label="Safety buffer $r_i+m$")]
''')

code(r'''
# ---- Metrics (distance, time, validity, coverage with / without section control) ----
SIMS = {}

def scenario_metrics(name, track, *, field_shp, crop, obstacles=None, trees=(), extra_passes=(),
                     coverage=True):
    tr = as_track(track)
    L_sw, L_tn = tr.length("swath"), tr.length("turn")
    L_hl = sum(g.length for g in extra_passes)
    T = (L_sw + L_hl) / CONFIG["cruise_vel"] + L_tn / CONFIG["turn_vel"] + CONFIG["reverse_penalty_s"] * tr.n_rev
    m = {"scenario": name, "turns": tr.n_turns, "reversals": tr.n_rev,
         "working [m]": round(L_sw + L_hl), "non-working [m]": round(L_tn),
         "total [m]": round(L_sw + L_hl + L_tn),
         "efficiency η [%]": round(100 * (L_sw + L_hl) / (L_sw + L_hl + L_tn), 1),
         "time [min]": round(T / 60, 1),
         "over crop [m]": round(turn_length_inside(tr, crop.buffer(-1.0))),
         "outside paddock [m]": round(length_outside(tr, field_shp))}
    if len(trees):
        m["min tree clearance [m]"] = round(min(tree_clearance(tr, trees, W)), 2)
    if coverage:
        for sc in (False, True):
            r = simulate_coverage(tr, field_shp, obstacles, WC, CONFIG["n_sections"], CONFIG["raster_res"],
                                  section_control=sc, extra_passes=extra_passes)
            SIMS[(name, sc)] = r
            tag = "SC" if sc else "no SC"
            m[f"missed [%] ({tag})"] = round(r["missed_pct"], 2)
            m[f"overlap [%] ({tag})"] = round(r["overlap_pct"], 2)
            m[f"applied [ha] ({tag})"] = round(r["applied_ha"], 2)
        m["product saved by SC [%]"] = round(100 * (1 - m["applied [ha] (SC)"] / m["applied [ha] (no SC)"]), 1)
    return m
''')

# =============================================================================
# Part A
# =============================================================================
md(r"""
---
## Part A · Open paddock (no obstacles)

### A.1 Field, machine and gate
""")

code(r'''
field = f2c.Cells(f2c.Cell(make_ring(CONFIG["paddock"])))
field_shp = to_shapely(field)
BOUNDS = field_shp.bounds

robot = f2c.Robot(W, WC)
robot.setMinTurningRadius(RMIN)
robot.setMaxDiffCurv(CONFIG["max_diff_curv"])
robot.setCruiseVel(CONFIG["cruise_vel"])
robot.setTurnVel(CONFIG["turn_vel"])
for setter, value in CONFIG["robot_tuning"].items():      # optional corner-smoothing parameters
    getattr(robot, setter)(value); print(f"robot.{setter}({value})")

print(f"Paddock area: {field.area()/1e4:.2f} ha   perimeter: {field_shp.length:.0f} m   gate: {CONFIG['gate']}")
fig, ax = plt.subplots(figsize=(8, 6))
draw_cells(ax, field_shp)
for i, (x, y) in enumerate(CONFIG["paddock"][:-1]):
    ax.plot(x, y, "o", color=COL["border"], ms=4); ax.text(x + 5, y + 5, f"V{i}", fontsize=8)
ax.plot(*CONFIG["gate"], marker="*", ms=18, mfc="gold", mec="k", zorder=10)
setup_ax(ax, "The paddock and its gate (★)", bounds=BOUNDS); add_scalebar(ax); add_north(ax); plt.show()
''')

md(r"""
### A.2 Headland width from the turn geometry
If $w_c<2r_{\min}$, a forward-only turn is an Ω-loop of depth $D_\Omega$ (§1.4). With `headland_factor="auto"` the notebook chooses
$k=\max(3,\lceil (D_\Omega+w/2)/w\rceil)$, so the loops stay inside the paddock.
""")

code(r'''
if WC < 2 * RMIN:
    D_OMEGA = RMIN + math.sqrt(max(4 * RMIN**2 - (RMIN + WC / 2)**2, 0.0))
else:
    D_OMEGA = RMIN
k = CONFIG["headland_factor"]
if k == "auto":
    k = max(3, math.ceil((D_OMEGA + W / 2) / W))
print(f"Omega-turn depth D = {D_OMEGA:.1f} m  ->  headland k = {k}  (width {k*W:.0f} m)")

const_hl = f2c.HG_Const_gen()
mid_hl = const_hl.generateHeadlands(field, 0.5 * k * W)   # headland track
no_hl  = const_hl.generateHeadlands(field, k * W)         # mainland
no_hl_shp = to_shapely(no_hl)

def headland_passes(cells):
    """Headland rounds (driven last), as working passes for the coverage analysis."""
    if not CONFIG["cover_headlands"]:
        return []
    n = max(1, int(round(k * W / WC)))
    try:
        rings = const_hl.generateHeadlandSwaths(cells, WC, n, False)
        out = []
        for i in range(len(rings)):
            for poly in _polys(to_shapely(rings[i])):
                out += [LineString(poly.exterior.coords)] + [LineString(r.coords) for r in poly.interiors]
        return out
    except Exception as e:
        print("⚠️ headland rounds unavailable:", e); return []

HL_A = headland_passes(field)
fig, ax = plt.subplots(figsize=(8, 6))
draw_cells(ax, field_shp, fc=COL["headland"]); draw_cells(ax, no_hl_shp, lw=1, ls="--")
for ln in HL_A:
    ax.plot(*ln.xy, color=COL["hl_track"], lw=0.8, ls=":")
setup_ax(ax, f"Headland {k}·w = {k*W:.0f} m with {len(HL_A)} headland rounds (dotted)", bounds=BOUNDS)
add_scalebar(ax); add_north(ax); plt.show()
''')

md(r"""
### A.3 Optimal swath angle $\theta^*$
""")

code(r'''
bf = f2c.SG_BruteForce()
mainland_cell = no_hl.getGeometry(0)
objectives = {"N swaths (NSwathModified)": f2c.OBJ_NSwathModified(),
              "Swath length (SwathLength)": f2c.OBJ_SwathLength()}
angles = np.deg2rad(np.arange(0, 180, 1.0))
costs = {n: np.array([bf.computeCostOfAngle(o, a, WC, mainland_cell) for a in angles]) for n, o in objectives.items()}
obj_nsw = objectives["N swaths (NSwathModified)"]
best_angle = bf.computeBestAngle(obj_nsw, WC, mainland_cell)
print(f"Optimal driving direction θ* = {np.rad2deg(best_angle):.1f}°")

fig = plt.figure(figsize=(12, 4.6))
ax1 = fig.add_subplot(1, 2, 1)
for (n, c), col in zip(costs.items(), ["#2f6db5", "#e07b39"]):
    ax1.plot(np.rad2deg(angles), (c - c.min()) / (np.ptp(c) or 1), color=col, lw=2, label=n)
ax1.axvline(np.rad2deg(best_angle), color="k", ls=":", lw=1)
ax1.set(xlabel=r"swath angle $\theta$ [°]", ylabel=r"normalised $J(\theta)$", xlim=(0, 180), title="Objective vs. swath angle")
ax1.legend(); ax1.grid(alpha=.4)
ax2 = fig.add_subplot(1, 2, 2, projection="polar")
c = costs["N swaths (NSwathModified)"]
th = np.concatenate([angles, angles + np.pi, angles[:1]]); rr = np.concatenate([c, c, c[:1]])
ax2.plot(th, rr, color="#2f6db5", lw=2); ax2.fill(th, rr, color="#2f6db5", alpha=.15)
ax2.set_title("Number of swaths (polar)", pad=15)
plt.tight_layout(); plt.show()

swaths_A = bf.generateBestSwaths(obj_nsw, WC, no_hl)
print(f"{swaths_A.sizeTotal()} swaths")
''')

md(r"""
### A.4 Route from the gate
The gate is projected onto the headland track and passed to `setStartAndEndPoint`, so the route starts and ends at the field entrance.
The classic patterns are shown for comparison; they cannot take a start point.
""")

code(r'''
planner = f2c.PP_PathPlanning()

def gate_point(mid_cells):
    poly = max(_polys(to_shapely(mid_cells)), key=lambda p: p.area)
    p = poly.exterior.interpolate(poly.exterior.project(Point(CONFIG["gate"])))
    return f2c.Point(p.x, p.y)

def route_with_gate(mid_cells, swaths, label):
    rp = f2c.RP_RoutePlannerBase()
    rp.setStartAndEndPoint(gate_point(mid_cells))
    t0 = time.time()
    route = rp.genRoute(mid_cells, swaths, False, 1e-4, True, int(CONFIG["rp_time"]), bool(CONFIG["rp_optimum"]))
    print(f"{label}: route optimised in {time.time()-t0:.0f} s")
    return route

def best_path(route, label, verbose=True):
    """Plan the route with Dubins-CC (and Reeds-Shepp-HC if allowed) and keep the faster valid one."""
    models = {"Dubins CC": f2c.PP_DubinsCurvesCC()}
    if CONFIG["allow_reverse"] and WC < 2 * RMIN:
        models["Reeds-Shepp HC"] = f2c.PP_ReedsSheppCurvesHC()
    rows, best = [], None
    for n, tm in models.items():
        path = planner.planPath(robot, route, tm)
        tr = track_from_path(path)
        out = length_outside(tr, field_shp)
        score = tr.length("turn") / CONFIG["turn_vel"] + CONFIG["reverse_penalty_s"] * tr.n_rev + 6.0 * out
        rows.append({"turn model": n, "turn length [m]": round(tr.length("turn")), "reversals": tr.n_rev,
                     "outside paddock [m]": round(out), "score [s]": round(score)})
        if best is None or score < best[0]:
            best = (score, n, path, tr)
    if verbose:
        print(f"{label}: chose {best[1]}"); display(pd.DataFrame(rows).set_index("turn model"))
    return best[2], best[3], best[1]

swaths_single = bf.generateSwaths(best_angle, WC, mainland_cell)
dubins = f2c.PP_DubinsCurves()
paths = {n: planner.planPath(robot, s.genSortedSwaths(swaths_single.clone()), dubins)
         for n, s in {"Boustrophedon": f2c.RP_Boustrophedon(), "Snake": f2c.RP_Snake(),
                      "Spiral (6)": f2c.RP_Spiral(6)}.items()}
route_A = route_with_gate(mid_hl, swaths_A, "A")
paths["Optimised from gate"] = planner.planPath(robot, route_A, dubins)

fig, axs = plt.subplots(2, 2, figsize=(13, 10))
rows = []
for ax, (name, p) in zip(axs.ravel(), paths.items()):
    draw_cells(ax, field_shp, fc=COL["headland"]); draw_cells(ax, no_hl_shp, lw=0.8, ls="--")
    draw_path(ax, p, lw=1.0, arrows=False)
    m = scenario_metrics(name, p, field_shp=field_shp, crop=no_hl_shp, coverage=False); rows.append(m)
    setup_ax(ax, f"{name}\nL = {m['total [m]']:,} m · over crop {m['over crop [m]']} m · outside {m['outside paddock [m]']} m", bounds=BOUNDS)
axs[0, 0].legend(handles=LEGEND_PATH, loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()
pd.DataFrame(rows).set_index("scenario")
''')

md(r"""
### A.5 Feasible turns
When $w_c<2r_{\min}$, forward-only turns are Ω-loops. Reeds–Shepp HC turns may reverse, giving a three-point turn instead. `best_path` keeps
whichever turn model is faster once reversals are penalised and any distance driven outside the paddock is heavily penalised.
""")

code(r'''
turn_models = {"Dubins": f2c.PP_DubinsCurves(), "Dubins CC": f2c.PP_DubinsCurvesCC(),
               "Reeds-Shepp": f2c.PP_ReedsSheppCurves(), "Reeds-Shepp HC": f2c.PP_ReedsSheppCurvesHC()}
p0 = swath_list(route_A)[0].startPoint(); zx, zy = p0.getX(), p0.getY()
fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
for ax, (n, tm) in zip(axs, turn_models.items()):
    tr = track_from_path(planner.planPath(robot, route_A, tm))
    draw_cells(ax, field_shp, fc=COL["headland"]); draw_cells(ax, no_hl_shp, lw=0.8, ls="--")
    draw_path(ax, tr, lw=1.4, arrows=False)
    ax.set_xlim(zx - 45, zx + 45); ax.set_ylim(zy - 45, zy + 45); ax.set_aspect("equal")
    ax.set_title(f"{n}\nL = {tr.line.length:,.0f} m · rev {tr.n_rev}"); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Turn models (zoom on the first swath)", fontweight="bold"); plt.tight_layout(); plt.show()

path_A, track_A, model_A = best_path(route_A, "Part A")
metrics_A = scenario_metrics("A · open paddock", track_A, field_shp=field_shp, crop=no_hl_shp, extra_passes=HL_A)
pd.DataFrame([metrics_A]).set_index("scenario").T
''')

code(r'''
fig, ax = plt.subplots(figsize=(11, 8))
draw_cells(ax, field_shp, fc=COL["headland"], lw=2)
draw_coverage(ax, track_A.pieces("swath"), WC)
draw_path(ax, track_A, lw=0.9)
setup_ax(ax, f"Part A · {model_A} from the gate — {metrics_A['total [m]']:,} m, ≈{metrics_A['time [min]']} min", bounds=BOUNDS)
ax.legend(handles=LEGEND_PATH, loc="upper left", framealpha=.9); add_scalebar(ax); add_north(ax)
fig.savefig("partA_path.png", bbox_inches="tight"); plt.show()
''')

# =============================================================================
# Part B
# =============================================================================
md(r"""
---
## Part B · The same paddock with pinpointed trees

### Why a naïve tree workflow performs poorly
The first version of this notebook produced a 17 km, ≈2.4 h path for 14 ha. The causes are instructive:
1. **Ω-loops everywhere.** $w_c=6$ m $<2r_{\min}=12$ m, so every forward U-turn becomes a bulb, and some bulbs left the paddock (§1.4).
2. **Island headlands.** The headland ring around each tree is *not connected* to the outer headland. Travel between them produced **straight hops across the crop**.
3. **Fragmented swaths.** Every tree cuts 3–5 swaths in two, and each fragment needs its own turns.
4. **Short, unconstrained search.** OR-Tools stopped at its first solution after 10 s, with no start or end point.

The three strategies below handle this differently. The comparison in Part C makes the differences measurable: *over crop [m]*, *outside paddock [m]*, and coverage with section control.

### B.1 Pinpointing the trees
""")

code(r'''
def random_trees(n, field_shp, margin, rng):
    inner = field_shp.buffer(-(k * W + 15))
    minx, miny, maxx, maxy = inner.bounds; out = []
    while len(out) < n:
        x, y, r = rng.uniform(minx, maxx), rng.uniform(miny, maxy), rng.uniform(4, 9)
        if inner.contains(Point(x, y)) and all(np.hypot(x - a, y - b) > r + c + 2 * margin + 3 * W for a, b, c in out):
            out.append((round(x, 1), round(y, 1), round(r, 1)))
    return out

TREES = random_trees(CONFIG["random_trees"], field_shp, CONFIG["tree_margin"], rng) if CONFIG["random_trees"] else CONFIG["trees"]
M = CONFIG["tree_margin"]
assert all(field_shp.buffer(-W).contains(Point(x, y).buffer(r + M)) for x, y, r in TREES), \
       "a tree buffer touches the paddock edge – move it or shrink it"
TREE_ZONE = unary_union([Point(x, y).buffer(r + M) for x, y, r in TREES])

fig, ax = plt.subplots(figsize=(9, 6.5))
draw_cells(ax, field_shp); draw_trees(ax, TREES, M)
for i, (x, y, r) in enumerate(TREES):
    ax.annotate(f"T{i}", (x, y), xytext=(x + r + M + 3, y + r + M + 3), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
ax.plot(*CONFIG["gate"], marker="*", ms=18, mfc="gold", mec="k", zorder=10)
setup_ax(ax, f"{len(TREES)} pinpointed trees (dashed = canopy + {M:.0f} m margin)", bounds=BOUNDS)
ax.legend(handles=LEGEND_TREE, loc="upper left"); add_scalebar(ax); add_north(ax); plt.show()
pd.DataFrame(TREES, columns=["x [m]", "y [m]", "canopy r [m]"]).rename_axis("tree")
''')

md(r"""
### B.2 Strategy B1: trees as holes with headland rings (F2C)
Each disc $B_i$ is added as an interior ring. The same headland generator then grows a ring of width $kw$ around every tree.
Swaths stop at those rings, and the route uses all the headland tracks.
""")

code(r'''
field_B = f2c.Cells(f2c.Cell(make_ring(CONFIG["paddock"])))
for (x, y, r) in TREES:
    field_B.addRing(0, make_ring(circle_coords(x, y, r + M)))
field_B_shp = to_shapely(field_B)
mid_hl_B = const_hl.generateHeadlands(field_B, 0.5 * k * W)
no_hl_B  = const_hl.generateHeadlands(field_B, k * W)
no_hl_B_shp = to_shapely(no_hl_B)
HL_B = headland_passes(field_B)

swaths_B1 = bf.generateBestSwaths(obj_nsw, WC, no_hl_B)
route_B1  = route_with_gate(mid_hl_B, swaths_B1, "B1")
path_B1, track_B1, model_B1 = best_path(route_B1, "B1")
metrics_B1 = scenario_metrics("B1 · holes", track_B1, field_shp=field_shp, crop=no_hl_B_shp,
                              obstacles=TREE_ZONE, trees=TREES, extra_passes=HL_B)

fig, ax = plt.subplots(figsize=(9, 6.5))
draw_cells(ax, field_B_shp, fc=COL["headland"]); draw_cells(ax, no_hl_B_shp, lw=1, ls="--")
draw_path(ax, track_B1, lw=0.8, arrows=False); draw_trees(ax, TREES, M)
setup_ax(ax, f"B1 · holes — {metrics_B1['total [m]']:,} m, over crop {metrics_B1['over crop [m]']} m", bounds=BOUNDS)
add_scalebar(ax); plt.show()
''')

md(r"""
### B.3 Strategy B2: Boustrophedon cell decomposition (F2C)
The area inside the headland track is cut into simpler cells at the tree *critical points* (Choset, 2000). Each cell gets its own swaths.
`HG_Corridor_gen` keeps neighbouring cells connected.
""")

code(r'''
try:
    decomp = f2c.DECOMP_Boustrophedon()
    decomp.setSplitAngle(best_angle)
    decomp_cells = decomp.decompose(mid_hl_B)
    no_hl_B2  = f2c.HG_Corridor_gen().generateHeadlands(decomp_cells, 0.5 * k * W)
    swaths_B2 = bf.generateBestSwaths(obj_nsw, WC, no_hl_B2)
    route_B2  = route_with_gate(mid_hl_B, swaths_B2, "B2")
    path_B2, track_B2, model_B2 = best_path(route_B2, "B2")
    metrics_B2 = scenario_metrics("B2 · decomposition", track_B2, field_shp=field_shp, crop=no_hl_B_shp,
                                  obstacles=TREE_ZONE, trees=TREES, extra_passes=HL_B)
    print(f"{decomp_cells.size()} cells")
except Exception as e:
    print("⚠️ Decomposition failed:", e); decomp_cells = None; track_B2 = None; metrics_B2 = None
''')

md(r"""
### B.4 Strategy B3: drive around the trees, with section control
F2C cannot do this, so it is implemented here. **Keep the open-paddock (Part A) swaths and route**, and where a swath crosses a tree, drive a smooth detour around it.
Sections switch off over the tree buffer and over ground that is already covered. This is how growers with section control usually treat isolated paddock trees.

**Detour geometry.** Work in a frame with the tree at the origin and the swath along $y=a$, where $|a|<R$:
- The clearance radius is $R=\max(r_i+m+\tfrac w2,\ r_{\min})$, so the implement edge stays $m$ away from the canopy and the curvature stays feasible.
- The fillet radius is $\rho=r_{\min}$.
- The fillet centres are $\left(\mp x_f,\ a+\rho\right)$, with $x_f=\sqrt{(R+\rho)^2-(a+\rho)^2}$.
- The tangent points with the tree circle are $\frac{R}{R+\rho}(\mp x_f,\ a+\rho)$.

$$\text{straight}\ \to\ \text{arc}(\rho,\circlearrowleft)\ \to\ \text{arc}(R,\circlearrowright)\ \to\ \text{arc}(\rho,\circlearrowleft)\ \to\ \text{straight},\qquad |\kappa|\le\tfrac1{r_{\min}}\ \text{everywhere.}$$

The short side is tried first, then the long side. If two trees are too close along a swath, they are merged into their enclosing circle.
""")

code(r'''
track_B3, issues_B3 = make_drive_around(track_A, TREES, M, W, RMIN, field_shp.buffer(-W / 2))
crop_B3 = no_hl_shp.difference(TREE_ZONE)
metrics_B3 = scenario_metrics("B3 · drive-around + SC", track_B3, field_shp=field_shp, crop=crop_B3,
                              obstacles=TREE_ZONE, trees=TREES, extra_passes=HL_A)
kmax = max(max_curvature(g) for g in track_B3.pieces("swath"))
print(f"detours added {track_B3.length('swath') - track_A.length('swath'):.0f} m; "
      f"max curvature {kmax:.3f} 1/m (limit {1/RMIN:.3f}); unresolved trees: {issues_B3 or 'none'}")

# zoom on the largest tree: detours + coverage with section control
tx, ty, tr_ = max(TREES, key=lambda t: t[2]); win = 4 * (tr_ + M + W)
fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, sc in zip(axs, (False, True)):
    draw_coverage_map(ax, SIMS[("B3 · drive-around + SC", sc)], "section control ON" if sc else "no section control")
    draw_path(ax, track_B3, lw=1.0, arrows=False); draw_trees(ax, TREES, M)
    ax.set_xlim(tx - win, tx + win); ax.set_ylim(ty - win, ty + win)
plt.suptitle("B3 detours around one tree · red = missed, green = once, orange = overlap", fontweight="bold")
plt.tight_layout(); plt.show()
''')

md(r"""
### B.5 Visual comparison of the three tree strategies
""")

code(r'''
TRACKS = {"B1 · holes": track_B1, "B3 · drive-around + SC": track_B3}
if track_B2 is not None:
    TRACKS = {"B1 · holes": track_B1, "B2 · decomposition": track_B2, "B3 · drive-around + SC": track_B3}
fig, axs = plt.subplots(1, len(TRACKS), figsize=(7 * len(TRACKS), 6.5))
for ax, (name, tr) in zip(np.atleast_1d(axs), TRACKS.items()):
    draw_cells(ax, field_shp, fc=COL["headland"]); draw_path(ax, tr, lw=0.7, arrows=False); draw_trees(ax, TREES, M)
    mm = {"B1 · holes": metrics_B1, "B2 · decomposition": metrics_B2, "B3 · drive-around + SC": metrics_B3}[name]
    setup_ax(ax, f"{name}\n{mm['total [m]']:,} m · {mm['time [min]']} min · over crop {mm['over crop [m]']} m", bounds=BOUNDS)
plt.tight_layout(); fig.savefig("partB_strategies.png", bbox_inches="tight"); plt.show()
''')

# =============================================================================
# Part C
# =============================================================================
md(r"""
---
## Part C · Compare, validate, export

### C.1 Scoreboard
The tree strategy is chosen by an **SC-aware cost** that weighs machine time, the product actually applied with section control, and the area left untreated:

$$C = c_t\,T + c_p\!\int_{F} c_{\text{SC}}\,dA + c_m\,M\,|F'|$$

The weights `c_t`, `c_p` and `c_m` are in `CONFIG`. Only strategies that clear every canopy are eligible.
""")

code(r'''
ALL = [m for m in [metrics_A, metrics_B1, metrics_B2, metrics_B3] if m]

def criterion_C(m):
    """C = c_t*T + c_p*applied_with_SC + c_m*missed_area (SC on); infeasible if a canopy is touched."""
    sim = SIMS[(m["scenario"], True)]
    missed_ha = sim["missed_pct"] / 100 * sim["target_ha"]
    C = CONFIG["c_t"] * m["time [min]"] + CONFIG["c_p"] * sim["applied_ha"] + CONFIG["c_m"] * missed_ha
    return C + (1e9 if m.get("min tree clearance [m]", 1) < 0 else 0)
for m in ALL:
    m["criterion C"] = round(criterion_C(m), 1)
cmp = pd.DataFrame(ALL).set_index("scenario")
BEST = min([m for m in ALL if m is not metrics_A], key=criterion_C)["scenario"]
display(cmp.T)
TR_BEST = {"B1 · holes": track_B1, "B2 · decomposition": track_B2, "B3 · drive-around + SC": track_B3}[BEST]
print("Recommended tree strategy:", BEST)

keys = ["criterion C", "time [min]", "over crop [m]", "missed [%] (SC)", "overlap [%] (no SC)", "product saved by SC [%]"]
fig, axs = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
colors = ["#7f8c8d", "#2f6db5", "#8e5cb8", "#e07b39"][:len(cmp)]
for ax, key in zip(axs, keys):
    bars = ax.bar(range(len(cmp)), cmp[key], color=colors)
    ax.set_xticks(range(len(cmp)), [s.split("·")[0].strip() for s in cmp.index])
    ax.set_title(key, fontsize=10); ax.bar_label(bars, fmt="%.1f", fontsize=8); ax.grid(axis="y", alpha=.3)
plt.suptitle("A = open paddock · B1 holes · B2 decomposition · B3 drive-around + section control", fontweight="bold")
plt.tight_layout(); plt.show()
''')

md(r"""
### C.2 Coverage maps, with and without section control
Each cell shows how many passes applied product to it: **red** = missed, **green** = once, **orange** = overlap. Tree buffers and ground outside the paddock are white.
The headland rounds are included when `cover_headlands=True`. Their corners follow the ring polylines, which is an approximation.
""")

code(r'''
names = [m["scenario"] for m in ALL]
fig, axs = plt.subplots(2, len(names), figsize=(4.6 * len(names), 8))
for j, n in enumerate(names):
    for i, sc in enumerate((False, True)):
        r = SIMS[(n, sc)]
        draw_coverage_map(axs[i, j], r, f"{n}\n{'SC' if sc else 'no SC'} · missed {r['missed_pct']:.1f}% · overlap {r['overlap_pct']:.1f}%")
        draw_trees(axs[i, j], TREES if n != "A · open paddock" else [], M)
plt.tight_layout(); fig.savefig("coverage_maps.png", bbox_inches="tight"); plt.show()
''')

md(r"""
### C.3 Best path: hero map, tree clearance, animation
""")

code(r'''
mb = cmp.loc[BEST]
fig, ax = plt.subplots(figsize=(12, 8.5))
draw_cells(ax, field_shp, fc=COL["headland"], lw=2)
draw_coverage(ax, TR_BEST.pieces("swath"), WC)
draw_path(ax, TR_BEST, lw=0.9); draw_trees(ax, TREES, M)
setup_ax(ax, f"{BEST} — {mb['total [m]']:,} m, ≈{mb['time [min]']} min, missed {mb['missed [%] (SC)']}% with SC", bounds=BOUNDS)
ax.legend(handles=LEGEND_PATH + LEGEND_TREE, loc="upper left", framealpha=.9); add_scalebar(ax); add_north(ax)
fig.savefig("partB_path.png", bbox_inches="tight"); plt.show()

clear = pd.DataFrame({"tree": [f"T{i}" for i in range(len(TREES))], "canopy r": [t[2] for t in TREES],
                      "edge clearance [m]": np.round(tree_clearance(TR_BEST, TREES, W), 2)}).set_index("tree")
display(clear)
assert (clear["edge clearance [m]"] > 0).all(), "⚠️ the implement would touch a canopy!"
print("✅ The implement never touches a tree canopy.")
''')

code(r'''
from matplotlib import animation
from IPython.display import HTML

line = TR_BEST.line
N_FRAMES = 240
dist = np.linspace(0, line.length, N_FRAMES)
pts = np.array([line.interpolate(s).coords[0][:2] for s in dist])
hdg = np.arctan2(*np.gradient(pts, axis=0)[:, ::-1].T)

fig, ax = plt.subplots(figsize=(9, 6.5))
draw_cells(ax, field_shp, fc=COL["headland"]); draw_trees(ax, TREES, M); setup_ax(ax, "", bounds=BOUNDS)
trail, = ax.plot([], [], color=COL["swath"], lw=WC * 0.55, alpha=0.35, solid_capstyle="butt")
track_ln, = ax.plot([], [], color=COL["swath"], lw=0.8)
bar = MplPolygon(np.zeros((4, 2)), closed=True, fc="#c0392b", ec="black", zorder=10); ax.add_patch(bar)

def frame(i):
    trail.set_data(pts[:i + 1, 0], pts[:i + 1, 1]); track_ln.set_data(pts[:i + 1, 0], pts[:i + 1, 1])
    h = hdg[i]; nvec = np.array([-np.sin(h), np.cos(h)]) * W / 2; tvec = np.array([np.cos(h), np.sin(h)]) * 2
    bar.set_xy([pts[i] + nvec + tvec, pts[i] - nvec + tvec, pts[i] - nvec - tvec, pts[i] + nvec - tvec])
    ax.set_title(f"{BEST}: {dist[i]:,.0f} / {line.length:,.0f} m")
    return trail, track_ln, bar

anim = animation.FuncAnimation(fig, frame, frames=N_FRAMES, interval=40, blit=True)
plt.close(fig); HTML(anim.to_jshtml())
''')

md(r"""
### C.4 Export
The path is written as a CSV (segment, kind, x, y) and as a GeoJSON (paddock, gate, trees, path pieces). Both open directly in QGIS.

**Real paddocks.** Load the boundary with `f2c.Parser.importFieldGml(...)` or build rings from GeoJSON or KML.
Convert to metres with `f2c.Transform.transformToUTM(field)`, transform the tree points the same way (e.g. with `pyproj`), plan as above, then convert back.
""")

code(r'''
rows = [(i, kind, x, y) for i, (g, kind) in enumerate(TR_BEST.segs) for x, y in g.coords]
pd.DataFrame(rows, columns=["segment", "kind", "x", "y"]).to_csv("coverage_path_trees.csv", index=False)
feats = [{"type": "Feature", "properties": {"layer": "paddock"}, "geometry": mapping(field_shp)},
         {"type": "Feature", "properties": {"layer": "gate"}, "geometry": mapping(Point(CONFIG["gate"]))}]
feats += [{"type": "Feature", "properties": {"layer": "tree", "id": f"T{i}", "canopy_r": r, "buffer_r": r + M},
           "geometry": mapping(Point(x, y))} for i, (x, y, r) in enumerate(TREES)]
feats += [{"type": "Feature", "properties": {"layer": "path", "kind": kind, "strategy": BEST}, "geometry": mapping(g)}
          for g, kind in TR_BEST.segs]
with open("coverage_path_trees.geojson", "w") as fh:
    json.dump({"type": "FeatureCollection", "features": feats}, fh)
cmp.to_csv("scenario_comparison.csv")
print("Saved: coverage_path_trees.csv/.geojson, scenario_comparison.csv, *.png")
try:
    from google.colab import files
    for fn in ["coverage_path_trees.csv", "coverage_path_trees.geojson", "scenario_comparison.csv", "partB_path.png"]:
        files.download(fn)
except ImportError:
    pass
''')

md(r"""
---
## Other algorithms worth trying
- **F2C decomposition variants:** `DECOMP_TrapezoidalDecomp` vs `DECOMP_Boustrophedon`, a different `setSplitAngle`, or a separate best angle per cell.
- **Custom routing with turn costs:** build your own OR-Tools routing model whose edge costs are the *actual* turn lengths (e.g. `OBJ_CompleteTurnPathObj_ReedsSheppHC`)
  rather than distances along the headland.
- **Skip-row (alternating) patterns:** `RP_Snake` visits every other swath, so each turn spans $2w_c\ge2r_{\min}$ and plain U-turns replace the Ω-loops.
- **Headland-first or spiral patterns:** these suit machines that must not drive over finished crop.
- **B3 extensions:** merge trees whose detours interact into a single obstacle, or switch to B1 just for large tree clumps.

## Exercises
1. **Gate.** Move `gate` to another vertex. How much do the distance and time change?
2. **Skip rows.** Set `width = cov_width = 12` (so $w_c\ge2r_{\min}$). Do the Ω-loops disappear, and how does $k$ change?
3. **Sections.** Compare `n_sections = 1, 3, 6, 12`. Plot the product saved against $n_s$.
4. **Search effort.** Try `rp_time = 5, 30, 120` with `rp_optimum = True`. Is the improvement worth the wait?
5. **Tree density.** Set `random_trees = 15`. When do B1 and B2 beat B3?

## Troubleshooting
| Symptom | Fix |
|---|---|
| Kernel restarts, with no Python error | A C++ crash. Check that you did not keep `.at(i)` from a temporary container (see the helpers) |
| `ImportError: libortools.so` | Run the Part 0 cell again (it recreates `/opt/or-tools`), or `!ldconfig` and restart |
| B2 decomposition error | Change `setSplitAngle` or move a tree. B1 and B3 still run |
| "unresolved trees" in B3 | The detour does not fit (tree too close to the end of a swath or to the boundary). Move the tree or use B1 for it |
| Section-control cell is slow | Increase `raster_res` (e.g. 2.0) |

*Fields2Cover © Wageningen University, BSD-3. Tutorial notebook CC-BY 4.0.*
""")

# =============================================================================
def to_lines(s):
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": [], "toc_visible": True, "name": "Fields2Cover_Paddock_Trees.ipynb"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": []}
for i, (kind, src) in enumerate(cells):
    c = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": to_lines(src)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)

out = sys.argv[1] if len(sys.argv) > 1 else "Fields2Cover_Paddock_Trees.ipynb"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print(f"wrote {out} with {len(cells)} cells")
