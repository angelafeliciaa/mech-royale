"""Rooftop camping: pick a flat roof inside the zone, climb it, hold the edge above the hunter.

Everything here is defensive: any failure returns None so dex_bot falls back to ground logic.
"""

import math
import sys

from arena_map import edge_distance_sq, inside


class RoofCandidate:
    __slots__ = ("idx", "top", "h", "center", "clear", "outer", "holes", "launches", "solid")

    def __init__(self, idx, solid, top, h, center, clear, outer, holes, launches):
        self.idx = idx
        self.solid = solid
        self.top = top
        self.h = h
        self.center = center
        self.clear = clear
        self.outer = outer
        self.holes = holes
        self.launches = launches


class RoofPlanner:
    def __init__(self, arena, nav, mech_stats, min_h=12.0, max_h=55.0, min_clear=8.0, max_cands=120):
        self.arena = arena
        self.nav = nav
        self.radius = float(mech_stats.get("radius", 2.5))
        self.min_h = min_h
        self.max_h = max_h
        self.min_clear = min_clear
        self.cands = []
        try:
            self._build(max_cands)
        except Exception as exc:  # pragma: no cover
            print("roof: build failed:", repr(exc), file=sys.stderr, flush=True)
            self.cands = []

    # ------------------------------------------------------------------ build
    def _taller_blocks(self, solid, px, py, pad):
        for s in self.arena.nearby((px, py), (px, py), pad):
            if s is solid:
                continue
            if s.top > solid.top + 0.3 and s.base < solid.top + 9.0 and s.touches(px, py, pad):
                return True
        return False

    def _build(self, max_cands):
        arena, nav = self.arena, self.nav
        out = []
        for idx, solid in enumerate(arena.solids):
            left, bottom, right, top = solid.box
            w, d = right - left, top - bottom
            if w < 18.0 or d < 18.0:
                continue
            outer = solid.rings[0]
            holes = solid.rings[1:]
            all_edges = [e for r in solid.rings for e in r]
            gx, gy = (left + right) * 0.5, (bottom + top) * 0.5
            h = solid.top - arena.floor(gx, gy)
            if h < self.min_h - 3.0 or h > self.max_h + 5.0:
                continue
            # Interior sample: max clearance point (coarse grid).
            step = 3.0 if max(w, d) < 80.0 else 5.0
            best_c, best_pt = 0.0, None
            y = bottom + step
            while y < top:
                x = left + step
                while x < right:
                    if inside(x, y, outer) and not any(inside(x, y, hh) for hh in holes):
                        c = math.sqrt(min(edge_distance_sq(x, y, e) for e in all_edges))
                        if c > best_c and not self._taller_blocks(solid, x, y, 3.0):
                            best_c, best_pt = c, (x, y)
                    x += step
                y += step
            if best_pt is None or best_c < self.min_clear:
                continue
            cx, cy = best_pt
            h = solid.top - arena.floor(cx, cy)
            if h < self.min_h or h > self.max_h:
                continue
            launches = self._launches(solid, outer, holes, all_edges)
            if not launches:
                continue
            out.append(RoofCandidate(idx, solid, solid.top, h, (cx, cy), best_c, outer, holes, launches))
        out.sort(key=lambda c: -c.clear)
        self.cands = out[:max_cands]

    def _launches(self, solid, outer, holes, all_edges):
        arena, nav = self.arena, self.nav
        res = []
        for (ax, ay), (bx, by) in outer:
            ex, ey = bx - ax, by - ay
            ln = math.hypot(ex, ey)
            if ln < 7.0:
                continue
            n_seg = max(1, int(ln // 30.0))
            for k in range(n_seg):
                t = (k + 0.5) / n_seg
                mx, my = ax + ex * t, ay + ey * t
                nx, ny = ey / ln, -ex / ln
                if inside(mx + nx * 1.5, my + ny * 1.5, outer):
                    nx, ny = -nx, -ny
                lx, ly = mx + nx * (self.radius + 2.0), my + ny * (self.radius + 2.0)
                dx, dy = mx - nx * 5.5, my - ny * 5.5
                if not nav.is_free(lx, ly) or not nav.reachable(lx, ly):
                    continue
                g = arena.floor(lx, ly)
                climb = solid.top - g
                if climb < 8.0 or climb > self.max_h:
                    continue
                # No overhang/eave above the launch/climb corridor, no taller solid at landing.
                ok = True
                for qx, qy in ((lx, ly), (mx + nx * 1.0, my + ny * 1.0), (mx + nx * 2.5, my + ny * 2.5)):
                    if nav.top_at(qx, qy) > arena.floor(qx, qy) + 0.6:
                        ok = False
                        break
                if not ok:
                    continue
                if not inside(dx, dy, outer) or any(inside(dx, dy, hh) for hh in holes):
                    continue
                if math.sqrt(min(edge_distance_sq(dx, dy, e) for e in all_edges)) < 3.0:
                    continue
                if self._taller_blocks(solid, dx, dy, 3.5) or abs(nav.top_at(dx, dy) - solid.top) > 0.3:
                    continue
                res.append(((lx, ly), (dx, dy), climb))
                if len(res) >= 12:
                    return res
        return res

    # ------------------------------------------------------------------ queries
    def on_roof_ok(self, cand, px, py, margin):
        """True when (px, py) is on cand's roof at least `margin` m from any edge/hole/taller block."""
        if not inside(px, py, cand.outer) or any(inside(px, py, hh) for hh in cand.holes):
            return False
        m2 = margin * margin
        for ring in cand.solid.rings:
            for e in ring:
                if edge_distance_sq(px, py, e) < m2:
                    return False
        return not self._taller_blocks(cand.solid, px, py, 3.0)

    def is_on(self, cand, x, y, z):
        return z >= cand.top - 0.8 and cand.solid.touches(x, y) and not any(inside(x, y, hh) for hh in cand.holes)

    def edge_point(self, cand, tx, ty, back):
        """Walk from the roof center toward (tx, ty); return the last point still `back` m from the edge."""
        cx, cy = cand.center
        dx, dy = tx - cx, ty - cy
        d = math.hypot(dx, dy)
        if d < 1.0:
            return cand.center
        ux, uy = dx / d, dy / d
        last = cand.center
        steps = int(min(d, cand.clear + 60.0))
        for k in range(1, steps + 1):
            px, py = cx + ux * k, cy + uy * k
            if not self.on_roof_ok(cand, px, py, back):
                break
            last = (px, py)
        return last

    def pick(self, x, y, zone_tracker, zone, now, bad, energy_hint=100.0):
        """Best roof inside current and next circle, or None."""
        nx, ny = zone["next_center"]
        nr = zone["next_radius"]
        ccx, ccy, ccr = zone_tracker.current(zone, now)
        best, best_cost = None, math.inf
        for c in self.cands:
            if bad.get(c.idx, -1.0) > now:
                continue
            cx, cy = c.center
            if math.hypot(cx - nx, cy - ny) > nr - 6.0:
                continue
            if math.hypot(cx - ccx, cy - ccy) > ccr - 6.0:
                continue
            # nearest launch
            ld, lp = math.inf, None
            for lp_ in c.launches:
                dd = math.hypot(lp_[0][0] - x, lp_[0][1] - y)
                if dd < ld:
                    ld, lp = dd, lp_
            if lp is None:
                continue
            if zone_tracker.deadline(zone, lp[0][0], lp[0][1], 6.0) < now + ld * 1.4 / 11.0 + 8.0:
                continue
            cost = ld + 2.5 * lp[2] + 0.35 * math.hypot(cx - nx, cy - ny) - 1.5 * min(c.clear, 25.0)
            if cost < best_cost:
                best_cost, best = cost, (c, lp)
        return best
