"""Map model: building footprints rasterised to a 2 m grid, line of sight, and a coarse flow field.

Standard library only. Everything is approximate (2D, ignores terrain height and trees); it is
used for cover-seeking and not walking into walls, not for exact physics.
"""
import json
import math
import os
import sys
from array import array
from collections import deque

CELL = 2.0          # fine grid, meters
COARSE = 2          # coarse cell = COARSE fine cells (4 m) for the flow field
INF = 65535


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class World:
    def __init__(self, map_dir):
        self.minx, self.miny, self.maxx, self.maxy = -540.0, -360.0, 580.0, 410.0
        self.ok = False
        prisms = []
        try:
            with open(os.path.join(map_dir, "v2", "world.json")) as f:
                w = json.load(f)
            b = w["bounds"]
            self.minx, self.miny = b["minX"] - 8, b["minY"] - 8
            self.maxx, self.maxy = b["maxX"] + 8, b["maxY"] + 8
            prisms = w["prisms"]
            self.ok = True
        except Exception as e:  # no map: bot still works, just without cover/walls
            log("world load failed:", repr(e))
        self.nx = int((self.maxx - self.minx) / CELL) + 1
        self.ny = int((self.maxy - self.miny) / CELL) + 1
        self.solid = bytearray(self.nx * self.ny)   # blocks walking
        self.sight = bytearray(self.nx * self.ny)   # blocks line of sight
        self.top = array("f", [-1e9]) * (self.nx * self.ny)   # absolute top height of the solid in a cell
        for p in prisms:
            try:
                if p.get("kind") not in ("building", "stadium"):
                    continue
                h = p["top"] - p["base"]
                if h < 2.0:
                    continue
                self._raster(p["ring"], p.get("holes") or [], h >= 7.0, p["top"])
            except Exception as e:
                log("prism skipped:", repr(e))
        self._build_coarse()

    # ---- rasterisation -------------------------------------------------
    @staticmethod
    def _edges(flat):
        pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
        es = []
        n = len(pts)
        for k in range(n):
            x0, y0 = pts[k]
            x1, y1 = pts[(k + 1) % n]
            if y0 == y1:
                continue
            if y0 > y1:
                x0, y0, x1, y1 = x1, y1, x0, y0
            es.append((x0, y0, x1, y1))
        return pts, es

    def _spans(self, es, j):
        y = self.miny + (j + 0.5) * CELL
        xs = sorted(x0 + (y - y0) * (x1 - x0) / (y1 - y0) for (x0, y0, x1, y1) in es if y0 <= y < y1)
        out = []
        for a in range(0, len(xs) - 1, 2):
            i0 = int(math.ceil((xs[a] - self.minx) / CELL - 0.5))
            i1 = int(math.floor((xs[a + 1] - self.minx) / CELL - 0.5))
            if i1 >= i0:
                out.append((max(i0, 0), min(i1, self.nx - 1)))
        return out

    def _raster(self, ring, holes, blocks_sight, top):
        pts, es = self._edges(ring)
        if not es:
            return
        ys = [p[1] for p in pts]
        j0 = max(0, int((min(ys) - self.miny) / CELL) - 1)
        j1 = min(self.ny - 1, int((max(ys) - self.miny) / CELL) + 1)
        hole_edges = []
        for hflat in holes:
            _, he = self._edges(hflat)
            hole_edges.append(he)
        for j in range(j0, j1 + 1):
            spans = self._spans(es, j)
            if not spans:
                continue
            row = {}
            for (a, b) in spans:
                for i in range(a, b + 1):
                    row[i] = 1
            for he in hole_edges:
                for (a, b) in self._spans(he, j):
                    for i in range(a, b + 1):
                        row.pop(i, None)
            base = j * self.nx
            for i in row:
                self.solid[base + i] = 1
                if top > self.top[base + i]:
                    self.top[base + i] = top
                if blocks_sight:
                    self.sight[base + i] = 1

    def _label_components(self):
        cnx, cny, cb = self.cnx, self.cny, self.cblock
        comp = array("H", [0]) * (cnx * cny)
        sizes = {}
        cid = 0
        for s0 in range(cnx * cny):
            if cb[s0] or comp[s0]:
                continue
            cid += 1
            comp[s0] = cid
            q = deque([s0])
            n = 0
            while q:
                c = q.popleft()
                n += 1
                cx, cy = c % cnx, c // cnx
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x_, y_ = cx + dx, cy + dy
                    if 0 <= x_ < cnx and 0 <= y_ < cny:
                        k = y_ * cnx + x_
                        if not cb[k] and not comp[k]:
                            comp[k] = cid
                            q.append(k)
            sizes[cid] = n
        self.comp = comp
        self.main_comp = max(sizes, key=sizes.get) if sizes else 0

    def enclosed(self, x, y):
        """True if the walkable region around (x,y) is cut off from the main map by walls."""
        if not self.main_comp:
            return False
        cs = CELL * COARSE
        ci, cj = int((x - self.minx) / cs), int((y - self.miny) / cs)
        seen = set()
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                i, j = ci + di, cj + dj
                if 0 <= i < self.cnx and 0 <= j < self.cny:
                    c = self.comp[j * self.cnx + i]
                    if c:
                        seen.add(c)
        if not seen:
            return False
        return self.main_comp not in seen

    def max_top_along(self, x, y, ux, uy, dist):
        m = -1e9
        s = 2.0
        while s <= dist:
            i = int((x + ux * s - self.minx) / CELL)
            j = int((y + uy * s - self.miny) / CELL)
            if 0 <= i < self.nx and 0 <= j < self.ny:
                v = self.top[j * self.nx + i]
                if v > m:
                    m = v
            s += 3.0
        return m

    def _build_coarse(self):
        self.cnx = (self.nx + COARSE - 1) // COARSE
        self.cny = (self.ny + COARSE - 1) // COARSE
        self.cblock = bytearray(self.cnx * self.cny)
        nx, ny, solid = self.nx, self.ny, self.solid
        for j in range(ny):
            row = j * nx
            cb = (j // COARSE) * self.cnx
            for i in range(nx):
                if solid[row + i]:
                    self.cblock[cb + i // COARSE] = 1
        self._label_components()

    # ---- queries ------------------------------------------------------
    def cell(self, x, y):
        return int((x - self.minx) / CELL), int((y - self.miny) / CELL)

    def is_solid(self, x, y, z=None):
        i = int((x - self.minx) / CELL)
        j = int((y - self.miny) / CELL)
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return True
        k = j * self.nx + i
        if not self.solid[k]:
            return False
        if z is None:
            return True
        return self.top[k] > z + 1.2   # a roof below our feet is not a wall

    def blocked(self, x0, y0, x1, y1, skip=3.0, step=2.5):
        """True if a building lies between the two points (line of sight blocked)."""
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        if d <= 2 * skip:
            return False
        ux, uy = dx / d, dy / d
        sight, nx, ny, minx, miny = self.sight, self.nx, self.ny, self.minx, self.miny
        s = skip
        end = d - skip
        while s < end:
            i = int((x0 + ux * s - minx) / CELL)
            j = int((y0 + uy * s - miny) / CELL)
            if 0 <= i < nx and 0 <= j < ny and sight[j * nx + i]:
                return True
            s += step
        return False

    def path_clear(self, x, y, ux, uy, length, z=None):
        """Can a mech (radius 2.5) walk `length` meters along (ux,uy) without hitting a wall?"""
        s = 4.0
        while s <= length + 0.01:
            cx, cy = x + ux * s, y + uy * s
            if self.is_solid(cx, cy, z):
                return False
            if self.is_solid(cx - uy * 2.2, cy + ux * 2.2, z) or self.is_solid(cx + uy * 2.2, cy - ux * 2.2, z):
                return False
            s += 4.0
        return True

    def cover_score(self, x, y, rays=12, reach=70.0):
        """How many of `rays` directions are blocked by a building within `reach` m (0..rays)."""
        n = 0
        for k in range(rays):
            a = k * math.tau / rays
            if self.blocked(x, y, x + math.cos(a) * reach, y + math.sin(a) * reach, skip=2.0, step=3.0):
                n += 1
        return n


class Flow:
    """Breadth-first distance field on the coarse grid, computed a slice at a time."""

    NB = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

    def __init__(self, world, gx, gy):
        self.w = world
        self.gx, self.gy = gx, gy
        cs = CELL * COARSE
        ci = int((gx - world.minx) / cs)
        cj = int((gy - world.miny) / cs)
        ci = min(max(ci, 0), world.cnx - 1)
        cj = min(max(cj, 0), world.cny - 1)
        start = self._nearest_free(ci, cj)
        self.dist = array("H", [INF]) * (world.cnx * world.cny)
        self.q = deque()
        self.done = start is None
        if start is not None:
            self.dist[start] = 0
            self.q.append(start)

    def _nearest_free(self, ci, cj):
        w = self.w
        for r in range(0, 40):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = ci + di, cj + dj
                    if 0 <= i < w.cnx and 0 <= j < w.cny and not w.cblock[j * w.cnx + i] \
                            and (not w.main_comp or w.comp[j * w.cnx + i] == w.main_comp):
                        return j * w.cnx + i
        return None

    def step(self, budget=1500):
        w, q, dist, cb, cnx, cny = self.w, self.q, self.dist, self.w.cblock, self.w.cnx, self.w.cny
        nb = self.NB
        while q and budget > 0:
            budget -= 1
            c = q.popleft()
            d = dist[c] + 1
            cx, cy = c % cnx, c // cnx
            for dx, dy in nb:
                nx_, ny_ = cx + dx, cy + dy
                if nx_ < 0 or ny_ < 0 or nx_ >= cnx or ny_ >= cny:
                    continue
                k = ny_ * cnx + nx_
                if cb[k] or dist[k] != INF:
                    continue
                if dx and dy and (cb[cy * cnx + nx_] or cb[ny_ * cnx + cx]):
                    continue  # no corner cutting
                dist[k] = d
                q.append(k)
        if not q:
            self.done = True

    def direction(self, x, y):
        """Unit vector toward the goal along the field, or None if this cell isn't reached yet."""
        w = self.w
        cs = CELL * COARSE
        ci = int((x - w.minx) / cs)
        cj = int((y - w.miny) / cs)
        if ci < 0 or cj < 0 or ci >= w.cnx or cj >= w.cny:
            return None
        here = self.dist[cj * w.cnx + ci]
        best, bd = None, here
        for dx, dy in self.NB:
            i, j = ci + dx, cj + dy
            if 0 <= i < w.cnx and 0 <= j < w.cny:
                dd = self.dist[j * w.cnx + i]
                if dd < bd:
                    if dx and dy and (w.cblock[cj * w.cnx + i] or w.cblock[j * w.cnx + ci]):
                        continue
                    bd, best = dd, (dx, dy)
        if best is None:
            return None
        n = math.hypot(*best)
        return best[0] / n, best[1] / n
