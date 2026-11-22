"""Precomputed grid navigation, smooth clearance, escape planning, fast LOS, and cover search."""

from heapq import heappop, heappush
import math
import time

from arena_map import ArenaMap

_CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142),
)


def _dilate_rows(rows, nx, radius):
    """Dilate a list of 1-bit row masks by a Euclidean disk of given radius."""
    ny = len(rows)
    full = (1 << nx) - 1
    r_int = int(math.floor(radius))
    horiz_cache = {}
    for dy in range(-r_int, r_int + 1):
        hw = int(math.floor(math.sqrt(max(0.0, radius * radius - dy * dy))))
        if hw in horiz_cache:
            continue
        shifted = []
        for mask in rows:
            acc = mask
            for d in range(1, hw + 1):
                acc |= (mask << d) | (mask >> d)
            shifted.append(acc & full)
        horiz_cache[hw] = shifted

    out = [0] * ny
    for dy in range(-r_int, r_int + 1):
        hw = int(math.floor(math.sqrt(max(0.0, radius * radius - dy * dy))))
        src = horiz_cache[hw]
        y_lo = max(0, -dy)
        y_hi = min(ny, ny - dy)
        for j in range(y_lo, y_hi):
            out[j] |= src[j + dy]
    return out


class NavGrid:
    """2 m navigation grid and 1 m walkability bitmask built from ArenaMap."""

    def __init__(self, arena: ArenaMap, mech_stats: dict):
        self.arena = arena
        self.radius = float(mech_stats.get("radius", 2.5))
        self.height = float(mech_stats.get("height", 9.0))
        self.step_up = float(mech_stats.get("step_up", 1.2))
        self.max_slope_deg = float(mech_stats.get("max_slope_deg", 50.0))

        terrain = arena.terrain
        if not terrain:
            self._init_flat()
            return

        self.nx = int(terrain["nx"])
        self.ny = int(terrain["ny"])
        self.min_x = float(terrain["minX"])
        self.min_y = float(terrain["minY"])
        self.step = float(terrain["step"])
        self.scale = float(terrain["scale"])

        raw_rows = self._rasterize_obstacles()
        self.hard_rows = _dilate_rows(raw_rows, self.nx, self.radius + 0.5)

        self.cnx = (self.nx + 1) // 2
        self.cny = (self.ny + 1) // 2
        self.cn = self.cnx * self.cny
        self.blocked = bytearray(self.cn)

        cnx = self.cnx
        for cj in range(self.cny):
            hrow = self.hard_rows[2 * cj]
            base = cj * cnx
            for ci in range(cnx):
                if hrow & (1 << (2 * ci)):
                    self.blocked[base + ci] = 1

        self.clearance = self._build_clearance()
        self.comp, self.main_comp = self._label_components()
        self._g = [0.0] * self.cn
        self._stamp = [0] * self.cn
        self._closed = [0] * self.cn
        self._parent = [-1] * self.cn
        self._gen = 0

        self._path = []
        self._path_goal = None
        self._path_at = -99.0
        self._path_idx = 0
        self.flat = False

    def _init_flat(self):
        self.flat = True
        self.nx = self.ny = 0
        self.cnx = self.cny = self.cn = 0
        self.min_x = self.min_y = 0.0
        self.step = 1.0
        self.scale = 0.01
        self.hard_rows = []
        self.blocked = bytearray()
        self.clearance = bytearray()
        self.comp = []
        self.main_comp = 1
        self._path = []
        self._path_goal = None
        self._path_at = -99.0
        self._path_idx = 0

    def _rasterize_obstacles(self):
        nx, ny = self.nx, self.ny
        min_x, min_y, step, scale = self.min_x, self.min_y, self.step, self.scale
        heights = self.arena.heights
        height, step_up = self.height, self.step_up
        rows = [0] * ny

        # 1. Scanline-fill solid prisms that obstruct ground-level walking.
        for solid in self.arena.solids:
            left, bottom, right, top = solid.box
            j0 = max(0, math.ceil((bottom - min_y) / step))
            j1 = min(ny - 1, math.floor((top - min_y) / step))
            all_edges = [edge for ring in solid.rings for edge in ring]
            for j in range(j0, j1 + 1):
                y = min_y + j * step
                xs = []
                for (ax, ay), (bx, by) in all_edges:
                    if (ay <= y) != (by <= y):
                        xs.append(ax + (y - ay) * (bx - ax) / (by - ay))
                if len(xs) < 2:
                    continue
                xs.sort()
                row_mask = rows[j]
                off = j * nx
                for k in range(0, len(xs) - 1, 2):
                    i0 = max(0, math.ceil((xs[k] - min_x) / step))
                    i1 = min(nx - 1, math.floor((xs[k + 1] - min_x) / step))
                    for i in range(i0, i1 + 1):
                        g = heights[off + i] * scale
                        if solid.base < g + height and solid.top > g + step_up:
                            row_mask |= 1 << i
                rows[j] = row_mask

        # 2. Mark terrain edges steeper than max_slope_deg.
        slope_thr = math.tan(math.radians(self.max_slope_deg)) * step / scale
        prev = None
        for j in range(ny):
            off = j * nx
            cur = heights[off:off + nx]
            row_mask = rows[j]
            for i in range(nx - 1):
                if abs(cur[i] - cur[i + 1]) > slope_thr:
                    row_mask |= 3 << i
            if prev is not None:
                prev_mask = rows[j - 1]
                for i in range(nx):
                    if abs(prev[i] - cur[i]) > slope_thr:
                        bit = 1 << i
                        row_mask |= bit
                        prev_mask |= bit
                rows[j - 1] = prev_mask
            rows[j] = row_mask
            prev = cur

        # 3. Mask out-of-bounds margins.
        bounds = self.arena.bounds
        full = (1 << nx) - 1
        i_min = max(0, math.ceil((bounds["minX"] - min_x) / step))
        i_max = min(nx - 1, math.floor((bounds["maxX"] - min_x) / step))
        left_mask = (1 << i_min) - 1 if i_min > 0 else 0
        right_mask = (full ^ ((1 << (i_max + 1)) - 1)) if i_max < nx - 1 else 0
        edge_mask = left_mask | right_mask
        for j in range(ny):
            y = min_y + j * step
            if y < bounds["minY"] or y > bounds["maxY"]:
                rows[j] = full
            elif edge_mask:
                rows[j] |= edge_mask
        return rows

    def _build_clearance(self):
        """Compute smooth wall-clearance distance (0..4 in 2 m cells = 0..8 m)."""
        cnx, cny, blocked = self.cnx, self.cny, self.blocked
        cl = bytearray([4] * self.cn)
        frontier = []
        for idx, b in enumerate(blocked):
            if b:
                cl[idx] = 0
                frontier.append(idx)
        for dist in (1, 2, 3):
            nxt = []
            for c in frontier:
                ci = c % cnx
                cj = c // cnx
                if ci > 0 and cl[c - 1] > dist:
                    cl[c - 1] = dist
                    nxt.append(c - 1)
                if ci + 1 < cnx and cl[c + 1] > dist:
                    cl[c + 1] = dist
                    nxt.append(c + 1)
                if cj > 0 and cl[c - cnx] > dist:
                    cl[c - cnx] = dist
                    nxt.append(c - cnx)
                if cj + 1 < cny and cl[c + cnx] > dist:
                    cl[c + cnx] = dist
                    nxt.append(c + cnx)
            frontier = nxt
        return cl

    def _label_components(self):
        cnx, cny, blocked = self.cnx, self.cny, self.blocked
        hard_rows = self.hard_rows
        comp = [0] * self.cn
        comp_id = 0
        main_comp = 0
        main_size = 0
        for start in range(self.cn):
            if blocked[start] or comp[start]:
                continue
            comp_id += 1
            comp[start] = comp_id
            stack = [start]
            size = 0
            while stack:
                c = stack.pop()
                size += 1
                ci = c % cnx
                cj = c // cnx
                for di, dj in _CARDINAL:
                    ni = ci + di
                    nj = cj + dj
                    if 0 <= ni < cnx and 0 <= nj < cny:
                        nb = nj * cnx + ni
                        if blocked[nb] or comp[nb]:
                            continue
                        if hard_rows[2 * cj + dj] & (1 << (2 * ci + di)):
                            continue
                        comp[nb] = comp_id
                        stack.append(nb)
            if size > main_size:
                main_size = size
                main_comp = comp_id
        return comp, main_comp

    # ----------------------------------------------------------------- queries
    def in_bounds(self, x, y, pad=0.0):
        b = self.arena.bounds
        return (b["minX"] + pad <= x <= b["maxX"] - pad
                and b["minY"] + pad <= y <= b["maxY"] - pad)

    def cell(self, x, y):
        if self.flat:
            return 0
        ci = int((x - self.min_x) / (2.0 * self.step) + 0.5)
        cj = int((y - self.min_y) / (2.0 * self.step) + 0.5)
        if ci < 0:
            ci = 0
        elif ci >= self.cnx:
            ci = self.cnx - 1
        if cj < 0:
            cj = 0
        elif cj >= self.cny:
            cj = self.cny - 1
        return cj * self.cnx + ci

    def cell_xy(self, c):
        return (
            self.min_x + 2.0 * self.step * (c % self.cnx),
            self.min_y + 2.0 * self.step * (c // self.cnx),
        )

    def top_at(self, x, y):
        """Highest solid roof or terrain height at (x, y)."""
        z = self.arena.floor(x, y)
        for solid in self.arena.nearby((x, y), (x, y)):
            if solid.top > z and solid.touches(x, y):
                z = solid.top
        return z

    def is_free(self, x, y):
        if not self.in_bounds(x, y, self.radius):
            return False
        if self.flat:
            return True
        i = int((x - self.min_x) / self.step + 0.5)
        j = int((y - self.min_y) / self.step + 0.5)
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return False
        return not (self.hard_rows[j] & (1 << i))

    def clearance_m(self, x, y):
        """Approximate clearance in meters (0..8 m) from nearest obstacle."""
        if self.flat:
            return 8.0
        return float(self.clearance[self.cell(x, y)]) * 2.0

    def is_near_wall(self, x, y):
        return self.clearance_m(x, y) < 4.0

    def nearest_free(self, c, max_r=6, main_only=False):
        if self.flat:
            return 0
        if not self.blocked[c] and (not main_only or self.comp[c] == self.main_comp):
            return c
        cnx, cny = self.cnx, self.cny
        ci, cj = c % cnx, c // cnx
        for r in range(1, max_r + 1):
            best = -1
            best_d2 = 1 << 30
            for dj in range(-r, r + 1):
                nj = cj + dj
                if nj < 0 or nj >= cny:
                    continue
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    ni = ci + di
                    if 0 <= ni < cnx:
                        nb = nj * cnx + ni
                        if not self.blocked[nb] and (not main_only or self.comp[nb] == self.main_comp):
                            d2 = di * di + dj * dj
                            if d2 < best_d2:
                                best_d2 = d2
                                best = nb
            if best >= 0:
                return best
        return -1

    def reachable(self, x, y):
        if self.flat:
            return self.in_bounds(x, y, self.radius)
        c = self.nearest_free(self.cell(x, y), max_r=3)
        return c >= 0 and self.comp[c] == self.main_comp

    def off_main(self, pos):
        if self.flat:
            return False
        x, y, z = pos
        if z - self.arena.floor(x, y) > 3.2:
            # On an elevated roof/deck: only count as trapped if ground below isn't walkable main.
            return not self.is_free(x, y)
        c = self.nearest_free(self.cell(x, y), max_r=2)
        return c < 0 or self.comp[c] != self.main_comp

    def escape_plan(self, x, y):
        """Find (launch_xy, land_xy, max_top) to jump/jet out of a courtyard or rooftop into main_comp."""
        if self.flat:
            return None
        start = self.nearest_free(self.cell(x, y), max_r=4)
        if start < 0:
            land_c = self.nearest_free(self.cell(x, y), max_r=25, main_only=True)
            if land_c < 0:
                return None
            lx, ly = self.cell_xy(land_c)
            return ((x, y), (lx, ly), self._max_top_along(x, y, lx, ly))

        # BFS inside our disconnected component to find boundary cells adjacent to obstacles.
        cnx, cny = self.cnx, self.cny
        blocked, comp, main_comp = self.blocked, self.comp, self.main_comp
        my_comp = comp[start]
        if my_comp == main_comp:
            return None

        self._gen += 1
        gen = self._gen
        stamp = self._stamp
        stamp[start] = gen
        queue = [start]
        head = 0
        boundary = []
        while head < len(queue) and len(queue) < 4000:
            c = queue[head]
            head += 1
            ci = c % cnx
            cj = c // cnx
            is_edge = False
            for di, dj in _CARDINAL:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < cnx and 0 <= nj < cny:
                    nb = nj * cnx + ni
                    if blocked[nb]:
                        is_edge = True
                    elif stamp[nb] != gen and comp[nb] == my_comp:
                        stamp[nb] = gen
                        queue.append(nb)
            if is_edge:
                boundary.append(c)

        if len(boundary) > 120:
            step_b = max(1, len(boundary) // 120)
            boundary = boundary[::step_b]

        best_plan = None
        best_cost = math.inf
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1), (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707))
        for bc in boundary:
            bx, by = self.cell_xy(bc)
            walk_d = math.hypot(bx - x, by - y)
            if walk_d > best_cost:
                continue
            for ux, uy in dirs:
                for jump_d in (8.0, 14.0, 20.0, 28.0, 36.0):
                    lx, ly = bx + ux * jump_d, by + uy * jump_d
                    if not self.in_bounds(lx, ly, self.radius):
                        break
                    lc = self.cell(lx, ly)
                    if not blocked[lc] and comp[lc] == main_comp:
                        top_z = self._max_top_along(bx, by, lx, ly)
                        climb = max(0.0, top_z - self.arena.floor(bx, by))
                        if climb <= 30.0:
                            cost = walk_d + jump_d * 2.0 + climb * 3.0
                            if cost < best_cost:
                                best_cost = cost
                                best_plan = ((bx, by), (lx, ly), top_z)
                        break
        if best_plan is None:
            land_c = self.nearest_free(start, max_r=30, main_only=True)
            if land_c >= 0:
                lx, ly = self.cell_xy(land_c)
                best_plan = ((x, y), (lx, ly), self._max_top_along(x, y, lx, ly))
        return best_plan

    def _max_top_along(self, x0, y0, x1, y1):
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(2, int(dist / 2.0) + 1)
        mx = -1e9
        for k in range(steps + 1):
            t = k / steps
            z = self.top_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if z > mx:
                mx = z
        return mx

    def line_walkable(self, x0, y0, x1, y1):
        if not self.in_bounds(x1, y1, self.radius):
            return False
        if self.flat:
            return True
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        steps = max(1, int(dist) + 1)
        ux = dx / steps
        uy = dy / steps
        ox = (x0 - self.min_x) / self.step + 0.5
        oy = (y0 - self.min_y) / self.step + 0.5
        inv = 1.0 / self.step
        nx, ny, hard_rows = self.nx, self.ny, self.hard_rows
        start_k = 2 if steps > 3 else 1
        for k in range(start_k, steps + 1):
            i = int(ox + ux * k * inv)
            j = int(oy + uy * k * inv)
            if i < 0 or j < 0 or i >= nx or j >= ny or (hard_rows[j] & (1 << i)):
                return False
        return True

    def clear_path(self):
        self._path = []
        self._path_goal = None

    # ----------------------------------------------------------------- A*
    def find_path(self, start_xy, goal_xy, budget_s=0.014):
        if self.flat:
            return [goal_xy]
        s = self.nearest_free(self.cell(*start_xy), max_r=6)
        g = self.nearest_free(self.cell(*goal_xy), max_r=6)
        if s < 0 or g < 0 or self.comp[s] != self.comp[g]:
            return None

        cnx, cny = self.cnx, self.cny
        blocked, clearance, hard_rows = self.blocked, self.clearance, self.hard_rows
        self._gen += 1
        gen = self._gen
        g_arr, stamp, closed, parent = self._g, self._stamp, self._closed, self._parent

        gi, gj = g % cnx, g // cnx
        si, sj = s % cnx, s // cnx
        dx, dy = abs(si - gi), abs(sj - gj)
        h0 = (dx + dy) - 0.5858 * min(dx, dy)
        g_arr[s] = 0.0
        stamp[s] = gen
        parent[s] = -1
        heap = [(h0 * 1.45, s)]
        best_node, best_h = s, h0
        t0 = time.perf_counter()
        expanded = 0

        while heap:
            _, cur = heappop(heap)
            if closed[cur] == gen:
                continue
            closed[cur] = gen
            if cur == g:
                best_node = cur
                break
            expanded += 1
            if (expanded & 127) == 0 and (time.perf_counter() - t0) > budget_s:
                break

            ci = cur % cnx
            cj = cur // cnx
            g_cur = g_arr[cur]
            for di, dj, step_cost in _NEIGHBORS:
                ni = ci + di
                nj = cj + dj
                if ni < 0 or nj < 0 or ni >= cnx or nj >= cny:
                    continue
                nb = nj * cnx + ni
                if blocked[nb] or closed[nb] == gen:
                    continue
                if di and dj:
                    if blocked[cj * cnx + ni] or blocked[nj * cnx + ci]:
                        continue
                elif hard_rows[2 * cj + dj] & (1 << (2 * ci + di)):
                    continue

                cl = clearance[nb]
                mult = 1.0 + (0.25 * (3 - cl) if cl < 3 else 0.0)
                ng = g_cur + step_cost * mult
                if stamp[nb] != gen or ng < g_arr[nb]:
                    stamp[nb] = gen
                    g_arr[nb] = ng
                    parent[nb] = cur
                    ex = abs(ni - gi)
                    ey = abs(nj - gj)
                    h = (ex + ey) - 0.5858 * min(ex, ey)
                    if h < best_h:
                        best_h = h
                        best_node = nb
                    heappush(heap, (ng + h * 1.45, nb))

        nodes = []
        n = best_node
        while n != -1:
            nodes.append(n)
            n = parent[n]
        nodes.reverse()
        pts = [self.cell_xy(c) for c in nodes[1:]]
        pts.append(goal_xy)
        return pts

    def steer_to(self, pos, goal_xy, now):
        x, y = pos[0], pos[1]
        gx, gy = goal_xy[0], goal_xy[1]
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        if dist < 0.8:
            return 0.0, 0.0

        # If elevated on a prism surface, walk straight off toward goal.
        if not self.flat and pos[2] - self.arena.floor(x, y) > 3.0 and not self.is_free(x, y):
            return dx / dist, dy / dist

        if dist < 45.0 and self.line_walkable(x, y, gx, gy):
            return dx / dist, dy / dist

        if (not self._path or self._path_goal is None
                or math.hypot(gx - self._path_goal[0], gy - self._path_goal[1]) > 6.0
                or now - self._path_at > 2.2):
            pts = self.find_path((x, y), (gx, gy))
            self._path = pts or []
            self._path_goal = (gx, gy)
            self._path_at = now
            self._path_idx = 0

        if not self._path:
            return self._local_avoid(x, y, dx / dist, dy / dist)

        lo = self._path_idx
        hi = min(len(self._path), lo + 12)
        bi = lo
        bd = math.inf
        for i in range(lo, hi):
            px, py = self._path[i]
            dd = math.hypot(px - x, py - y)
            if dd < bd:
                bd, bi = dd, i
        self._path_idx = bi
        if bd > 12.0:
            self._path = []
            return self._local_avoid(x, y, dx / dist, dy / dist)

        for i in range(min(len(self._path) - 1, bi + 14), bi, -1):
            px, py = self._path[i]
            if self.line_walkable(x, y, px, py):
                td = math.hypot(px - x, py - y)
                if td > 0.2:
                    return (px - x) / td, (py - y) / td
        nxt = self._path[min(bi + 1, len(self._path) - 1)]
        td = math.hypot(nxt[0] - x, nxt[1] - y)
        return ((nxt[0] - x) / td, (nxt[1] - y) / td) if td > 0.2 else (dx / dist, dy / dist)

    def _local_avoid(self, x, y, ux, uy):
        base = math.atan2(uy, ux)
        for off in (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.2, -2.2):
            a = base + off
            c, s = math.cos(a), math.sin(a)
            if self.line_walkable(x, y, x + c * 6.0, y + s * 6.0):
                return c, s
        return ux, uy

    # ----------------------------------------------------------------- fast LOS
    def los(self, start, end):
        """Fast line-of-sight query against terrain heightfield and solid prisms."""
        sx, sy, sz = start
        ex, ey, ez = end
        dx = ex - sx
        dy = ey - sy
        dz = ez - sz
        dist = math.hypot(dx, dy)

        if not self.flat and dist > 2.0:
            steps = max(2, int(dist * 0.25) + 1)
            nx, ny = self.nx, self.ny
            nx1, ny1 = nx - 1, ny - 1
            min_x, min_y = self.min_x, self.min_y
            inv_step = 1.0 / self.step
            scale = self.scale
            heights = self.arena.heights
            ox = (sx - min_x) * inv_step
            oy = (sy - min_y) * inv_step
            ddx = dx * inv_step / steps
            ddy = dy * inv_step / steps
            ddz = dz / steps
            for k in range(1, steps):
                ix = int(ox + ddx * k)
                iy = int(oy + ddy * k)
                if 0 <= ix <= nx1 and 0 <= iy <= ny1:
                    if sz + ddz * k <= heights[iy * nx + ix] * scale:
                        return False

        grid = self.arena.grid
        solids = self.arena.solids
        seen = set()
        cell_steps = max(1, int(dist / 16.0) + 1)
        for k in range(cell_steps + 1):
            t = k / cell_steps
            cx = int(math.floor((sx + dx * t) / 32.0))
            cy = int(math.floor((sy + dy * t) / 32.0))
            bucket = grid.get((cx, cy))
            if bucket:
                seen.update(bucket)
        for idx in seen:
            if solids[idx].intersects(start, end):
                return False
        return True

    # ----------------------------------------------------------------- BFS cover
    def find_cover(self, pos, threats, stats, zone_penalty_fn, budget_s=0.009):
        """Find a nearby walkable cover cell hiding body & head from threats (4 m sample stride)."""
        if not threats:
            return None
        x, y = pos[0], pos[1]
        height = float(stats.get("height", self.height))
        arena = self.arena
        los_fn = self.los

        def hidden(cx, cy):
            cz = arena.floor(cx, cy)
            p_mid = (cx, cy, cz + height * 0.5)
            p_top = (cx, cy, cz + height)
            for eye in threats:
                if los_fn(eye, p_mid) or los_fn(eye, p_top):
                    return False
            return True

        if hidden(x, y) and zone_penalty_fn(x, y) == 0.0:
            return (x, y)
        if self.flat:
            return None

        start = self.nearest_free(self.cell(x, y), max_r=4)
        if start < 0:
            return None

        cnx, cny = self.cnx, self.cny
        blocked, hard_rows, clearance = self.blocked, self.hard_rows, self.clearance
        self._gen += 1
        gen = self._gen
        stamp, dist_arr = self._stamp, self._g
        stamp[start] = gen
        dist_arr[start] = 0.0
        queue = [start]
        head = 0
        checked = 0
        best_xy = None
        best_score = math.inf
        t0 = time.perf_counter()

        while head < len(queue) and checked < 160:
            if (checked & 7) == 0 and (time.perf_counter() - t0) > budget_s:
                break
            c = queue[head]
            head += 1
            steps = dist_arr[c]
            if steps > 48 or (best_xy is not None and steps * 2.0 > best_score + 24.0):
                break
            ci = c % cnx
            cj = c // cnx
            # Check wall-adjacent cells densely (4 m stride) and open terrain sparsely (8 m stride for hill cover).
            near_wall = clearance[c] <= 2
            sample = ((ci & 1) == 0 and (cj & 1) == 0) if near_wall else ((ci & 3) == 0 and (cj & 3) == 0)
            if steps > 0 and sample:
                cx, cy = self.cell_xy(c)
                checked += 1
                if hidden(cx, cy):
                    near_t = min(math.hypot(eye[0] - cx, eye[1] - cy) for eye in threats)
                    sc = steps * 2.0 + zone_penalty_fn(cx, cy) - min(near_t, 80.0) * 0.25
                    if sc < best_score:
                        best_score = sc
                        best_xy = (cx, cy)
            for di, dj in _CARDINAL:
                ni = ci + di
                nj = cj + dj
                if 0 <= ni < cnx and 0 <= nj < cny:
                    nb = nj * cnx + ni
                    if blocked[nb] or stamp[nb] == gen:
                        continue
                    if hard_rows[2 * cj + dj] & (1 << (2 * ci + di)):
                        continue
                    stamp[nb] = gen
                    dist_arr[nb] = steps + 1.0
                    queue.append(nb)
        return best_xy
