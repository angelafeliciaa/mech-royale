"""Zone timing, multi-candidate combat positioning, and projectile/laser dodging."""

import math

from dex_combat import actor_center, actor_eye, angle_diff, clamp, weapon_slot

_ANGLES_16 = tuple((math.cos(k * math.pi / 8.0), math.sin(k * math.pi / 8.0)) for k in range(16))


class ZoneTracker:
    """Tracks shrink anchor across phases and evaluates zone deadlines/penalties."""

    def __init__(self):
        self.hold_start = -1.0
        self.hold_circle = (0.0, 0.0, 1e9)

    def update(self, zone, now):
        ss = zone["shrink_start"]
        if ss != self.hold_start or now <= ss:
            self.hold_start = ss
            self.hold_circle = (zone["center"][0], zone["center"][1], zone["radius"])

    def at(self, zone, t):
        ss, se = zone["shrink_start"], zone["shrink_end"]
        if t <= ss or se <= ss:
            return self.hold_circle
        f = clamp((t - ss) / (se - ss), 0.0, 1.0)
        hx, hy, hr = self.hold_circle
        nx, ny = zone["next_center"]
        nr = zone["next_radius"]
        return (hx + (nx - hx) * f, hy + (ny - hy) * f, hr + (nr - hr) * f)

    def current(self, zone, now):
        live = (zone["center"][0], zone["center"][1], zone["radius"])
        if now <= zone["shrink_start"]:
            return live
        hx, hy, hr = self.hold_circle
        if abs(live[2] - hr) > 0.01 or abs(live[0] - hx) > 0.01:
            return live
        return self.at(zone, now)

    def deadline(self, zone, x, y, pad=8.0):
        nx, ny = zone["next_center"]
        nr = zone["next_radius"]
        if math.hypot(x - nx, y - ny) <= nr - pad:
            return math.inf
        ss, se = zone["shrink_start"], zone["shrink_end"]
        span = se - ss
        for k in range(25):
            t = ss + span * (k / 24.0)
            cx, cy, cr = self.at(zone, t)
            if math.hypot(x - cx, y - cy) > cr - pad:
                return t
        return se

    def penalty(self, zone, now, x, y):
        cx, cy, cr = self.current(zone, now)
        pen = 0.0
        d_cur = math.hypot(x - cx, y - cy) - cr
        if d_cur > -5.0:
            pen += 150.0 + 5.0 * (d_cur + 5.0)
        nx, ny = zone["next_center"]
        nr = zone["next_radius"]
        d_next = math.hypot(x - nx, y - ny) - nr
        if d_next > -5.0:
            soon = clamp(1.0 - (zone["shrink_start"] - now) / 25.0, 0.1, 1.0)
            pen += soon * (25.0 + 0.8 * (d_next + 5.0))
        return pen

    @staticmethod
    def goal(zone, x, y):
        nx, ny = zone["next_center"]
        nr = zone["next_radius"]
        if nr <= 30.0:
            return nx, ny
        margin = clamp(0.3 * nr, 3.0, 30.0)
        dx, dy = x - nx, y - ny
        d = math.hypot(dx, dy)
        if d < 1.0:
            return x, y
        r = max(0.0, nr - margin - 5.0)
        if d <= r:
            return x, y
        return nx + dx / d * r, ny + dy / d * r


def preferred_range(me, track, params, state=None):
    _, light = weapon_slot(me["weapons"], "lightning")
    _, laser = weapon_slot(me["weapons"], "laser")
    r = params["gun_range"]
    light_ok = light is not None
    if light_ok and state is not None:
        # Hysteresis: once the arc runs dry, kite at gun range until energy is back, instead of oscillating.
        low = state.get("light_low", False)
        if low and me["energy"] > 40.0:
            low = False
        elif not low and me["energy"] < 9.0:
            low = True
        state["light_low"] = low
        light_ok = not low
    elif light_ok:
        light_ok = me["energy"] > 12.0
    if light_ok:
        r = min(params["light_range"], light["stats"].get("range", 45.0) * 0.62)
    elif laser is not None:
        r = params["laser_range"]
    if track["kind"] == "npc":
        return 58.0
    if light is None:
        ew = track.get("weapons", ())
        if "lightning" in ew:
            r = max(r, 64.0)
        if "mortar" in ew:
            r = max(r, 80.0)
    return r


def fight_move(cfg, arena, nav, zone_tracker, obs, target, seen, cubes, want_los, params, state):
    """Score current position + tactical ring around self for combat movement."""
    me = obs["self"]
    zone = obs["zone"]
    now = obs["t"]
    x, y, zf = me["pos"]
    rp = preferred_range(me, target, params, state)

    tx, ty = target["pos"][0], target["pos"][1]
    bx, by = tx - x, ty - y
    bd = math.hypot(bx, by) + 1e-9
    ux, uy = bx / bd, by / bd

    # Fast anti-lead strafe flips matching 80-120 m bullet flight time (~0.32-0.48 s).
    if now >= state["strafe_until"]:
        if state["rng"].random() < 0.6:
            state["strafe"] = -state["strafe"]
        state["strafe_until"] = now + 0.3 + state["rng"].random() * 0.7
    sx, sy = -uy * state["strafe"], ux * state["strafe"]

    tc = actor_center(cfg, target)
    mh = me["stats"]["muzzle_height"]
    body_h = me["stats"]["height"] * 0.6

    # Nearby cube on same elevation for opportunistic mid-strafe pickup.
    near_cube = None
    near_cube_d = 40.0
    for c in cubes.values():
        cp = c["pos"]
        if abs(cp[2] - zf) < 4.0:
            cd = math.hypot(cp[0] - x, cp[1] - y)
            if cd < near_cube_d:
                near_cube_d = cd
                near_cube = cp

    # Limit crossfire checks to closest <= 4 other shooters within 170 m.
    others = []
    for e in seen:
        if e["id"] == target["id"]:
            continue
        ed = math.hypot(e["pos"][0] - x, e["pos"][1] - y)
        if ed <= 182.0:
            others.append((ed, actor_eye(cfg, e), e["pos"]))
    if len(others) > 4:
        others.sort(key=lambda item: item[0])
        others = others[:4]

    rings = (6.0, 12.0) if (params["peek"] or params.get("dual_ring", False)) else (6.0,)
    best_score = -math.inf
    best_mx = best_my = 0.0
    blocked_side = False

    # Evaluate stay-put (index -1) and 16 directions per ring.
    candidates = [(x, y, 0.0, 0.0, 0.0, True, -1)]
    for ri, r in enumerate(rings):
        for dx, dy in _ANGLES_16:
            cx, cy = x + dx * r, y + dy * r
            if not nav.line_walkable(x, y, cx, cy) or not nav.is_free(cx, cy):
                if ri == 0 and dx * sx + dy * sy > 0.9:
                    blocked_side = True
                continue
            candidates.append((cx, cy, dx, dy, r, False, ri))

    want_corner = params["peek"] or params.get("dual_ring", False)
    for cx, cy, dx, dy, r, is_stay, ri in candidates:
        cz = arena.floor(cx, cy)
        d = math.hypot(tx - cx, ty - cy)
        score = -1.1 * abs(d - rp) - r * 0.4
        los = nav.los((cx, cy, cz + mh), tc)
        if los:
            score += 45.0 * want_los
        if want_corner and ri == 0 and (los == (want_los > 0.0)):
            for q in range(0, 16, 4):
                qdx, qdy = _ANGLES_16[q]
                qx, qy = cx + qdx * 7.0, cy + qdy * 7.0
                if nav.is_free(qx, qy) and nav.los((qx, qy, arena.floor(qx, qy) + mh), tc) != los:
                    score += 8.0
                    break

        body = (cx, cy, cz + body_h)
        for _, eye_e, ep in others:
            if math.hypot(ep[0] - cx, ep[1] - cy) <= 170.0:
                if nav.los(eye_e, body):
                    score -= 30.0

        score -= zone_tracker.penalty(zone, now, cx, cy)
        if near_cube is not None:
            score += 1.5 * (near_cube_d - math.hypot(near_cube[0] - cx, near_cube[1] - cy))
        cl = nav.clearance_m(cx, cy)
        if cl < 4.0:
            score -= (4.0 - cl) * 2.5
        if want_los > 0.0:
            score += 9.0 * (dx * sx + dy * sy)
        if is_stay:
            score -= 8.0
            if want_los < 0.0 and not los:
                score += 14.0
        if score > best_score:
            best_score = score
            best_mx, best_my = dx, dy

    if blocked_side:
        state["strafe"] = -state["strafe"]
        state["strafe_until"] = now + 0.5
    return best_mx, best_my


def dodge_mortar(cfg, arena, nav, obs):
    """Return (mx, my) if an incoming mortar shell will land within blast radius."""
    me = obs["self"]
    x, y, _ = me["pos"]
    mcfg = cfg["weapons"]["mortar"]
    g = mcfg["gravity"] if mcfg["gravity"] > 0 else 22.0
    rad = mcfg["splash_radius"] + me["stats"]["radius"] + 2.0

    for p in obs.get("projectiles", ()):
        if p["kind"] != "mortar":
            continue
        px, py, pz = p["pos"]
        vx, vy, vz = p["vel"]
        t = 0.0
        while t < 3.5:
            px += vx * 0.05
            py += vy * 0.05
            pz += vz * 0.05
            vz -= g * 0.05
            t += 0.05
            hit = pz <= arena.floor(px, py)
            if not hit:
                for solid in arena.nearby((px, py), (px, py)):
                    if solid.base <= pz <= solid.top and solid.touches(px, py):
                        hit = True
                        break
            if hit:
                d = math.hypot(px - x, py - y)
                if d < rad and t < 2.5:
                    if d < 0.5:
                        dx, dy = -vy, vx
                        d = math.hypot(dx, dy) or 1.0
                    else:
                        dx, dy = x - px, y - py
                    ux, uy = dx / d, dy / d
                    for rot in (0.0, 0.7, -0.7, 1.4, -1.4):
                        c, s = math.cos(rot), math.sin(rot)
                        ex, ey = ux * c - uy * s, ux * s + uy * c
                        if nav.line_walkable(x, y, x + ex * 6.0, y + ey * 6.0):
                            return ex, ey
                break
    return None


def dodge_threats(cfg, arena, nav, zone_tracker, obs, strafe_sign, wall_aware=False):
    """Return (dx, dy, laser_threat) for bullets and aimed charging lasers."""
    me = obs["self"]
    now = obs["t"]
    zone = obs["zone"]
    x, y, zf = me["pos"]
    cz = zf + me["stats"]["height"] * 0.56
    reach = me["stats"]["radius"] + 1.1
    dx = dy = 0.0
    laser_threat = False

    mvx, mvy, mvz = me["vel"]
    half_h = me["stats"]["height"] * 0.5 + 0.6
    for p in obs.get("projectiles", ()):
        if p["kind"] == "mortar":
            continue
        px, py, pz = p["pos"]
        vx, vy, vz = p["vel"]
        # Work in our own frame: enemies lead our velocity, so a bullet that will
        # hit us looks like a miss relative to where we stand right now.
        rvx, rvy, rvz = vx - mvx, vy - mvy, vz - mvz
        v2 = rvx * rvx + rvy * rvy + rvz * rvz
        if v2 < 1.0:
            continue
        tc = ((x - px) * rvx + (y - py) * rvy + (cz - pz) * rvz) / v2
        if tc <= 0.0 or tc > 0.7:
            continue
        mxx = px + rvx * tc - x
        myy = py + rvy * tc - y
        mzz = pz + rvz * tc - cz
        if abs(mzz) > half_h:
            continue
        miss = math.hypot(mxx, myy)
        if miss < reach:
            vh = math.hypot(vx, vy)
            if vh < 1.0:
                continue
            perp_x, perp_y = -vy / vh, vx / vh
            if miss > 0.6:
                # Step away from the side the bullet passes on.
                if perp_x * mxx + perp_y * myy > 0.0:
                    perp_x, perp_y = -perp_x, -perp_y
            else:
                # Dead-on: the shooter led our current motion, so reverse it.
                lat = perp_x * mvx + perp_y * mvy
                if lat > 1.5 or (lat >= -1.5 and strafe_sign > 0.0):
                    # (fight_move strafes along -perp*strafe_sign; agree with it when idle)
                    perp_x, perp_y = -perp_x, -perp_y
            if wall_aware and not nav.line_walkable(x, y, x + perp_x * 5.0, y + perp_y * 5.0):
                if nav.line_walkable(x, y, x - perp_x * 5.0, y - perp_y * 5.0):
                    perp_x, perp_y = -perp_x, -perp_y
            dx += perp_x
            dy += perp_y

    for m in obs.get("mechs", ()):
        if m["kind"] != "mech" or not m.get("charging"):
            continue
        mx_, my_, mz_ = m["pos"]
        to_me = math.atan2(y - my_, x - mx_)
        if abs(angle_diff(m["aim"][0], to_me)) > 0.16:
            continue
        laser_threat = True
        if wall_aware:
            # Prefer the perpendicular sidestep that is walkable, in-zone, and breaks laser LOS.
            eye_m = (mx_, my_, mz_ + cfg["mech"]["eye_height"] * (1.0 + cfg["cubes"]["size_per_cube"] * m.get("cubes", 0)))
            best_side = None
            best_rank = (-1, -1, -1e9)
            for sgn in (strafe_sign, -strafe_sign):
                ang = to_me + (math.pi * 0.5) * sgn
                sx, sy = math.cos(ang), math.sin(ang)
                tx, ty = x + sx * 6.0, y + sy * 6.0
                walkable = 1 if nav.line_walkable(x, y, tx, ty) else 0
                zpen = zone_tracker.penalty(zone, now, tx, ty)
                breaks_los = 1 if (walkable and not nav.los(eye_m, (tx, ty, arena.floor(tx, ty) + me["stats"]["height"] * 0.55))) else 0
                rank = (walkable, breaks_los, -zpen)
                if rank > best_rank:
                    best_rank = rank
                    best_side = (sx, sy)
            dx += best_side[0] * 1.5
            dy += best_side[1] * 1.5
        else:
            ang = to_me + (math.pi * 0.5) * strafe_sign
            dx += math.cos(ang) * 1.5
            dy += math.sin(ang) * 1.5

    length = math.hypot(dx, dy)
    if length > 0.01:
        return dx / length, dy / length, laser_threat
    return 0.0, 0.0, False
