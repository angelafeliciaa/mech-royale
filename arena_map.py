"""Local queries over the static Mech Royale arena."""

from array import array
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys


CELL = 32


def cells(left, bottom, right, top):
    for x in range(math.floor(left / CELL), math.floor(right / CELL) + 1):
        for y in range(math.floor(bottom / CELL), math.floor(top / CELL) + 1):
            yield x, y


def edges(ring):
    points = tuple(zip(ring[::2], ring[1::2]))
    return tuple(zip(points, points[1:] + points[:1]))


def inside(x, y, ring):
    hit = False
    for (ax, ay), (bx, by) in ring:
        if (ay > y) != (by > y) and x < ax + (bx - ax) * (y - ay) / (by - ay):
            hit = not hit
    return hit


def edge_distance_sq(x, y, edge):
    (ax, ay), (bx, by) = edge
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    t = max(0, min(1, ((x - ax) * dx + (y - ay) * dy) / length_sq)) if length_sq else 0
    return (x - ax - t * dx) ** 2 + (y - ay - t * dy) ** 2


@dataclass(frozen=True)
class Solid:
    base: float
    top: float
    box: tuple
    rings: tuple

    def touches(self, x, y, radius=0):
        left, bottom, right, top = self.box
        if not (left - radius <= x <= right + radius and bottom - radius <= y <= top + radius):
            return False
        if inside(x, y, self.rings[0]) and not any(inside(x, y, hole) for hole in self.rings[1:]):
            return True
        if radius <= 0:
            return False  # only an exact edge hit (measure zero) is lost
        r2 = radius * radius
        return any(edge_distance_sq(x, y, edge) <= r2
                   for ring in self.rings for edge in ring)

    def intersects(self, start, end):
        x, y, z = start
        dx = end[0] - x
        dy = end[1] - y
        dz = end[2] - z
        lo, hi = 0.0, 1.0
        left, bottom, right, top = self.box
        for p, d, lower, upper in ((x, dx, left, right), (y, dy, bottom, top), (z, dz, self.base, self.top)):
            if -1e-9 < d < 1e-9:
                if p < lower or p > upper:
                    return False
                continue
            a = (lower - p) / d
            b = (upper - p) / d
            if a > b:
                a, b = b, a
            if a > lo:
                lo = a
            if b < hi:
                hi = b
            if lo > hi:
                return False
        if self.touches(x + lo * dx, y + lo * dy):
            return True
        for ring in self.rings:
            for (ax, ay), (bx, by) in ring:
                ex, ey = bx - ax, by - ay
                denom = dx * ey - dy * ex
                if abs(denom) < 1e-9:
                    continue
                t = ((ax - x) * ey - (ay - y) * ex) / denom
                u = ((ax - x) * dy - (ay - y) * dx) / denom
                if lo <= t <= hi and 0 <= u <= 1:
                    return True
        return False


class ArenaMap:
    def __init__(self, bounds, terrain, heights, solids):
        self.bounds, self.terrain, self.heights = bounds, terrain, heights
        self.solids = solids
        self.grid = defaultdict(list)
        for index, solid in enumerate(solids):
            for cell in cells(*solid.box):
                self.grid[cell].append(index)

    @classmethod
    def load(cls, map_info):
        root = Path(map_info["dir"])
        world_path = root / next(name for name in map_info["files"] if name.endswith("world.json"))
        world = json.loads(world_path.read_text(encoding="utf-8"))
        terrain = world["terrain"]
        heights = array("h")
        heights.frombytes((world_path.parent / terrain["file"]).read_bytes())
        if sys.byteorder != "little":
            heights.byteswap()
        if len(heights) != terrain["nx"] * terrain["ny"]:
            raise ValueError("Arena terrain grid size does not match its metadata")
        solids = []
        for prism in world["prisms"]:
            ring = prism["ring"]
            xs, ys = ring[::2], ring[1::2]
            solids.append(Solid(prism["base"], prism["top"], (min(xs), min(ys), max(xs), max(ys)),
                                tuple(edges(r) for r in [ring, *prism["holes"]])))
        return cls(world["bounds"], terrain, heights, solids)

    def nearby(self, start, end, pad=0):
        x, y = start[:2]
        u, v = end[:2]
        ids = set()
        for cell in cells(min(x, u) - pad, min(y, v) - pad, max(x, u) + pad, max(y, v) + pad):
            ids.update(self.grid.get(cell, ()))
        return (self.solids[i] for i in sorted(ids))

    def floor(self, x, y):
        terrain = self.terrain
        if not terrain:
            return 0.0
        nx, ny = terrain["nx"], terrain["ny"]
        u = max(0, min(nx - 1, (x - terrain["minX"]) / terrain["step"]))
        v = max(0, min(ny - 1, (y - terrain["minY"]) / terrain["step"]))
        i, j = min(int(u), nx - 2), min(int(v), ny - 2)
        fx, fy = u - i, v - j
        k = j * nx + i
        h = self.heights
        a = h[k] + fx * (h[k + 1] - h[k])
        b = h[k + nx] + fx * (h[k + nx + 1] - h[k + nx])
        return (a + fy * (b - a)) * terrain["scale"]

    def shot_clear(self, start, end):
        if any(solid.intersects(start, end) for solid in self.nearby(start, end)):
            return False
        distance = math.dist(start[:2], end[:2])
        steps = max(1, math.ceil(distance / 3))
        x, y, z = start
        dx, dy, dz = (b - a for a, b in zip(start, end))
        for i in range(1, steps):
            t = i / steps
            if z + t * dz <= self.floor(x + t * dx, y + t * dy):
                return False
        return True

    def walk_clear(self, start, goal, stats):
        return self._walk_blocker(start, goal, stats, solids_only=False) is None

    def wall_ahead(self, start, goal, stats):
        """Return the first solid in a walk corridor, ignoring terrain-only failures."""
        return self._walk_blocker(start, goal, stats, solids_only=True)

    def _walk_blocker(self, start, goal, stats, *, solids_only):
        x, y, z = start
        u, v = goal[:2]
        radius, height = stats["radius"], stats["height"]
        bounds = self.bounds
        if not solids_only and not (bounds["minX"] + radius <= u <= bounds["maxX"] - radius
                                    and bounds["minY"] + radius <= v <= bounds["maxY"] - radius):
            return True
        solids = tuple(self.nearby(start, goal, radius))
        if solids_only and not solids:
            return None
        distance = math.hypot(u - x, v - y)
        steps = max(1, math.ceil(distance / 1.5))
        rise = max(stats["step_up"], math.tan(math.radians(stats["max_slope_deg"])) * distance / steps)
        elevated = z > self.floor(x, y) + stats["step_up"]
        for i in range(1, steps + 1):
            t = i / steps
            px, py = x + (u - x) * t, y + (v - y) * t
            ground = self.floor(px, py)
            if not solids_only and ground > z + rise:
                return True
            # An elevated start can cross its current surface. After stepping off,
            # project the landing to ground before checking subsequent clearance.
            if elevated and not any(abs(s.top - z) <= 0.15 and s.touches(px, py, radius) for s in solids):
                elevated = False
            z = max(z, ground) if elevated else ground
            for solid in solids:
                if solid.base < z + height and solid.top > z + 0.05 and solid.touches(px, py, radius):
                    return solid
        return None
