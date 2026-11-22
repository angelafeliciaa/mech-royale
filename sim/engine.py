"""Local lockstep duel simulator for mechfight/1 (two mechs, no loot/NPC/zone shrink).

Faithful where it matters for duels: movement accel/speeds/energy, torso slew, gun
projectiles with spread + building/mech collision, laser charge/hold/auto-fire hitscan,
lightning cone/LOS, rear armor, HP regen, vision wedge/near-range/memory, sounds, beams,
events. Per-bot obs dicts are built in the exact protocol shape (kit/protocol.md section 6).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_map import ArenaMap  # noqa: E402

TAU = math.tau


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def r2(v):
    return round(v, 2)


def r3(v):
    return round(v, 3)


def seg_cylinder(p0, p1, cx, cy, cz, radius, height):
    """Smallest t in [0,1] where segment p0->p1 enters the vertical cylinder, else None."""
    dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    fx, fy = p0[0] - cx, p0[1] - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    if a < 1e-12:
        if c > 0:
            return None
        t0, t1 = 0.0, 1.0
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        s = math.sqrt(disc)
        t0, t1 = (-b - s) / (2 * a), (-b + s) / (2 * a)
    lo, hi = max(0.0, t0), min(1.0, t1)
    if lo > hi:
        return None
    # vertical slab
    if abs(dz) < 1e-9:
        if not (cz <= p0[2] <= cz + height):
            return None
    else:
        za, zb = sorted(((cz - p0[2]) / dz, (cz + height - p0[2]) / dz))
        lo, hi = max(lo, za), min(hi, zb)
        if lo > hi:
            return None
    return lo


class Mech:
    def __init__(self, mid, pos, yaw, cfg, weapons, name):
        m = cfg["mech"]
        self.id = mid
        self.name = name
        self.pos = list(pos)
        self.vel = [0.0, 0.0, 0.0]
        self.aim = [yaw, 0.0]
        self.cmd_aim = [yaw, 0.0]
        self.hp = float(m["hp"])
        self.max_hp = float(m["hp"])
        self.energy = float(m["energy"])
        self.max_energy = float(m["energy"])
        self.grounded = True
        self.cubes = 0
        self.alive = True
        self.last_spend = -99.0
        self.last_combat = -99.0
        self.weapons = []
        for kind in weapons:
            if kind is None:
                self.weapons.append(None)
                continue
            w = {"kind": kind, "cooldown": 0.0, "stats": dict(cfg["weapons"][kind])}
            if kind == "gun":
                w["ammo"] = w["stats"]["mag"]
                w["reload"] = 0.0
            if kind == "laser":
                w["charge"] = 0.0
                w["hold"] = 0.0
                w["release"] = False
            self.weapons.append(w)
        while len(self.weapons) < 2:
            self.weapons.append(None)
        self.action = {"move": [0.0, 0.0], "boost": False, "jump": False, "fire": [False, False], "reload": False}
        self.events = []
        self.sounds = []
        self.beams = []
        self.seen_at = {}  # other id -> last time seen
        self.stats = {"damage_dealt": 0.0, "damage_taken": 0.0,
                      "shots": {"gun": 0, "laser": 0, "lightning": 0},
                      "hits": {"gun": 0, "laser": 0, "lightning": 0},
                      "rear_hits": 0}
        self.step_t = 0.0

    @property
    def charging(self):
        return any(w and w["kind"] == "laser" and w["charge"] > 0 for w in self.weapons)

    def muzzle(self, cfg):
        yaw, pitch = self.aim
        d = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
        r = cfg["mech"]["radius"] + 0.5
        return [self.pos[0] + d[0] * r, self.pos[1] + d[1] * r, self.pos[2] + cfg["mech"]["muzzle_height"] + d[2] * r], d


class World:
    def __init__(self, cfg, arena: ArenaMap, seed=0, max_t=90.0, alive=16):
        self.cfg = cfg
        self.arena = arena
        self.rng = random.Random(seed)
        self.tick = 0
        self.dt = 1.0 / cfg["match"]["tick_hz"]
        self.t = 0.0
        self.max_t = max_t
        self.alive_shown = alive
        self.mechs = []
        self.bullets = []
        self.next_bullet = 1
        z = cfg["zone"]
        self.zone = {"center": list(z["start_center"]), "radius": z["start_radius"],
                     "next_center": list(z["start_center"]), "next_radius": z["phases"][0]["radius"],
                     "shrink_start": 600, "shrink_end": 650, "dps": z["phases"][0]["dps"]}
        self.log = []

    # ---------------------------------------------------------------- setup
    def add_mech(self, pos, yaw, weapons, name):
        m = Mech(len(self.mechs), pos, yaw, self.cfg, weapons, name)
        self.mechs.append(m)
        return m

    # ---------------------------------------------------------------- geometry
    def floor_at(self, x, y, z):
        """Ground height under (x,y): terrain or the roof of a solid we stand on."""
        f = self.arena.floor(x, y)
        for s in self.arena.nearby((x, y), (x, y), 0.1):
            if s.top <= z + self.cfg["mech"]["step_up"] + 0.01 and s.top > f and s.touches(x, y, 0.0):
                f = s.top
        return f

    def blocked(self, x, y, z, radius, height):
        m = self.cfg["mech"]
        b = self.arena.bounds
        if not (b["minX"] + radius <= x <= b["maxX"] - radius and b["minY"] + radius <= y <= b["maxY"] - radius):
            return True
        for s in self.arena.nearby((x, y), (x, y), radius):
            if s.top > z + m["step_up"] and s.base < z + height and s.touches(x, y, radius):
                return True
        return False

    def los(self, a, b):
        return self.arena.shot_clear(a, b)

    def ray_wall_t(self, p0, p1):
        """Fraction along p0->p1 where it first hits a solid or terrain (1.0 = clear). Coarse."""
        if self.los(p0, p1):
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            q = [p0[i] + (p1[i] - p0[i]) * mid for i in range(3)]
            if self.los(p0, q):
                lo = mid
            else:
                hi = mid
        return lo

    def center(self, m):
        return (m.pos[0], m.pos[1], m.pos[2] + self.cfg["mech"]["height"] * 0.5)

    def eye(self, m):
        return (m.pos[0], m.pos[1], m.pos[2] + self.cfg["mech"]["eye_height"])

    # ---------------------------------------------------------------- damage
    def damage(self, src, dst, amount, weapon, from_dir):
        """from_dir: unit direction the shot travelled (toward dst)."""
        if not dst.alive:
            return 0.0
        m = self.cfg["mech"]
        bearing = math.atan2(-from_dir[1], -from_dir[0])  # toward source
        rear = abs(wrap(bearing - dst.aim[0])) > math.radians(m["front_arc_deg"]) * 0.5
        amt = amount * (1.0 - m.get("armor", 0.0))
        if rear:
            amt *= m["rear_mult"]
        amt = round(amt, 2)
        dst.hp -= amt
        dst.last_combat = self.t
        src.last_combat = self.t
        src.stats["damage_dealt"] += amt
        dst.stats["damage_taken"] += amt
        src.stats["hits"][weapon] += 1
        if rear:
            src.stats["rear_hits"] += 1
        ev = {"type": "damaged", "amount": amt, "weapon": weapon, "bearing": r3(bearing), "by": src.id}
        if rear:
            ev["rear"] = True
        dst.events.append(ev)
        hv = {"type": "hit", "target": dst.id, "amount": amt, "weapon": weapon}
        if rear:
            hv["rear"] = True
        src.events.append(hv)
        if dst.hp <= 0:
            dst.hp = 0.0
            dst.alive = False
            src.events.append({"type": "kill", "target": dst.id})
            self.sound(dst, "death")
        return amt

    def sound(self, by, kind):
        rng = self.cfg["sound"]["range"].get(kind, 0)
        for m in self.mechs:
            if m is by or not m.alive:
                continue
            d = math.dist(m.pos[:2], by.pos[:2])
            if d <= rng:
                b = math.atan2(by.pos[1] - m.pos[1], by.pos[0] - m.pos[0])
                b += math.radians(self.rng.gauss(0.0, self.cfg["sound"]["bearing_noise_deg"]))
                dist = "near" if d < self.cfg["sound"]["near"] else "mid" if d < self.cfg["sound"]["mid"] else "far"
                m.sounds.append({"kind": kind, "bearing": r3(wrap(b)), "dist": dist, "t": self.t})

    # ---------------------------------------------------------------- weapons
    def fire_gun(self, m, w):
        ws = w["stats"]
        w["ammo"] -= 1
        w["cooldown"] += ws["cooldown"]
        if w["ammo"] <= 0:
            w["reload"] = ws["reload"]
        origin, d = m.muzzle(self.cfg)
        yaw = m.aim[0] + math.radians(self.rng.gauss(0.0, ws["spread_deg"]))
        pitch = m.aim[1] + math.radians(self.rng.gauss(0.0, ws["spread_deg"]))
        d = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
        self.bullets.append({"id": self.next_bullet, "by": m.id, "pos": origin,
                             "vel": [d[0] * ws["speed"], d[1] * ws["speed"], d[2] * ws["speed"]],
                             "left": ws["range"], "damage": ws["damage"]})
        self.next_bullet += 1
        m.stats["shots"]["gun"] += 1
        self.sound(m, "gun")

    def hitscan(self, m, max_range):
        """Returns (end_point, target_mech or None, unit dir)."""
        origin, d = m.muzzle(self.cfg)
        end = [origin[i] + d[i] * max_range for i in range(3)]
        t_wall = self.ray_wall_t(origin, end)
        best, target = t_wall, None
        R, H = self.cfg["mech"]["radius"], self.cfg["mech"]["height"]
        for o in self.mechs:
            if o is m or not o.alive:
                continue
            t = seg_cylinder(origin, end, o.pos[0], o.pos[1], o.pos[2], R, H)
            if t is not None and t < best:
                best, target = t, o
        hit = [origin[i] + (end[i] - origin[i]) * best for i in range(3)]
        return origin, hit, target, d

    def fire_laser(self, m, w):
        ws = w["stats"]
        frac = ws["min_frac"] + (1.0 - ws["min_frac"]) * clamp(w["charge"] / ws["charge"], 0.0, 1.0)
        dmg = ws["damage"] * frac
        w["charge"] = 0.0
        w["hold"] = 0.0
        w["release"] = False
        w["cooldown"] = ws["cooldown"]
        origin, hit, target, d = self.hitscan(m, ws["range"])
        m.stats["shots"]["laser"] += 1
        self.beam(m, "laser", [origin, hit])
        self.sound(m, "laser")
        if target is not None:
            self.damage(m, target, dmg, "laser", d)

    def fire_lightning(self, m, w):
        ws = w["stats"]
        m.energy -= ws["energy"]
        m.last_spend = self.t
        w["cooldown"] = ws["cooldown"]
        m.stats["shots"]["lightning"] += 1
        origin, d = m.muzzle(self.cfg)
        half = math.radians(ws["cone_deg"]) * 0.5
        best, target = ws["range"], None
        for o in self.mechs:
            if o is m or not o.alive:
                continue
            c = self.center(o)
            dist = math.dist(origin, c)
            if dist > best:
                continue
            if abs(wrap(math.atan2(c[1] - origin[1], c[0] - origin[0]) - m.aim[0])) > half:
                continue
            if not self.los(origin, c):
                continue
            best, target = dist, o
        self.sound(m, "lightning")
        if target is None:
            end = [origin[i] + d[i] * ws["range"] for i in range(3)]
            t = self.ray_wall_t(origin, end)
            self.beam(m, "lightning", [origin, [origin[i] + (end[i] - origin[i]) * t for i in range(3)]])
            return
        c = self.center(target)
        self.beam(m, "lightning", [origin, list(c)])
        v = [c[i] - origin[i] for i in range(3)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        self.damage(m, target, ws["damage"], "lightning", [x / n for x in v])

    def beam(self, by, kind, path):
        for o in self.mechs:
            if o is by or not o.alive:
                continue
            eye = self.eye(o)
            if any(self.visible_point(o, eye, p) for p in path):
                o.beams.append({"by": by.id, "kind": kind, "path": [[r2(c) for c in p] for p in path], "t": self.t})

    def weapons_tick(self, m):
        dt = self.dt
        fire = m.action.get("fire") or [False, False]
        for i, w in enumerate(m.weapons):
            if w is None:
                continue
            w["cooldown"] = max(0.0, w["cooldown"] - dt)
            if w["cooldown"] < 1e-6:
                w["cooldown"] = 0.0
            held = bool(fire[i]) if i < len(fire) else False
            kind = w["kind"]
            if kind == "gun":
                ws = w["stats"]
                if w["reload"] > 0:
                    w["reload"] = max(0.0, w["reload"] - dt)
                    if w["reload"] < 1e-6:
                        w["reload"] = 0.0
                        w["ammo"] = ws["mag"]
                    continue
                if m.action.get("reload") and w["ammo"] < ws["mag"]:
                    w["reload"] = ws["reload"]
                    self.sound(m, "reload")
                    continue
                if held and w["cooldown"] <= 0 and w["ammo"] > 0:
                    self.fire_gun(m, w)
            elif kind == "laser":
                ws = w["stats"]
                if w["charge"] > 0:
                    was_full = w["charge"] >= ws["charge"]
                    w["charge"] = min(ws["charge"], w["charge"] + dt)
                    if was_full:
                        w["hold"] += dt
                    if not held:
                        w["release"] = True
                    if w["release"] and w["charge"] >= ws["min_charge"]:
                        self.fire_laser(m, w)
                    elif w["hold"] >= ws["hold_s"]:
                        self.fire_laser(m, w)
                elif held and w["cooldown"] <= 0:
                    w["charge"] = dt
                    w["release"] = False
                    self.sound(m, "laser_charge")
            elif kind == "lightning":
                ws = w["stats"]
                if held and w["cooldown"] <= 0 and m.energy >= ws["energy"]:
                    self.fire_lightning(m, w)

    def bullets_tick(self):
        dt = self.dt
        R, H = self.cfg["mech"]["radius"] + self.cfg["projectile"]["gun_radius"], self.cfg["mech"]["height"]
        keep = []
        for b in self.bullets:
            p0 = b["pos"]
            step = [b["vel"][i] * dt for i in range(3)]
            p1 = [p0[i] + step[i] for i in range(3)]
            seg_len = math.sqrt(sum(s * s for s in step))
            if seg_len > b["left"]:
                f = b["left"] / seg_len
                p1 = [p0[i] + step[i] * f for i in range(3)]
            # mechs
            best, target = 1.0, None
            for o in self.mechs:
                if o.id == b["by"] or not o.alive:
                    continue
                t = seg_cylinder(p0, p1, o.pos[0], o.pos[1], o.pos[2], R, H)
                if t is not None and t < best:
                    best, target = t, o
            if target is not None:
                # make sure no wall before the mech
                hitp = [p0[i] + (p1[i] - p0[i]) * best for i in range(3)]
                if self.los(p0, hitp):
                    n = math.sqrt(sum(v * v for v in b["vel"])) or 1.0
                    self.damage(self.mechs[b["by"]], target, b["damage"], "gun", [v / n for v in b["vel"]])
                    continue
            if not self.los(p0, p1):
                continue
            b["left"] -= seg_len
            if b["left"] <= 0:
                continue
            b["pos"] = p1
            keep.append(b)
        self.bullets = keep

    # ---------------------------------------------------------------- movement
    def move_tick(self, m):
        cfg = self.cfg["mech"]
        dt = self.dt
        a = m.action
        mv = a.get("move") or [0.0, 0.0]
        mx, my = float(mv[0]), float(mv[1])
        n = math.hypot(mx, my)
        if n > 1.0:
            mx, my = mx / n, my / n
            n = 1.0
        boost = bool(a.get("boost")) and m.energy > 0
        speed = cfg["walk_speed"]
        if boost and m.grounded and n > 1e-6:
            speed = cfg["boost_speed"]
            m.energy = max(0.0, m.energy - cfg["boost_drain"] * dt)
            m.last_spend = self.t
        if m.charging:
            speed *= cfg["charge_move_mult"]
        tvx, tvy = mx * speed, my * speed
        accel = cfg["ground_accel"] if m.grounded else cfg["air_accel"]
        dvx, dvy = tvx - m.vel[0], tvy - m.vel[1]
        dn = math.hypot(dvx, dvy)
        if dn > accel * dt:
            dvx, dvy = dvx / dn * accel * dt, dvy / dn * accel * dt
        m.vel[0] += dvx
        m.vel[1] += dvy

        # vertical
        if m.grounded and a.get("jump"):
            m.vel[2] = cfg["jump_speed"]
            m.grounded = False
            self.sound(m, "jet")
        if not m.grounded:
            m.vel[2] -= cfg["gravity"] * dt
            if boost:
                m.vel[2] = min(cfg["max_rise_speed"], m.vel[2] + cfg["jet_accel"] * dt)
                m.energy = max(0.0, m.energy - cfg["jet_drain"] * dt)
                m.last_spend = self.t
        # horizontal with sliding
        x, y, z = m.pos
        nx, ny = x + m.vel[0] * dt, y + m.vel[1] * dt
        R, H = cfg["radius"], cfg["height"]
        if self.blocked(nx, ny, z, R, H):
            if not self.blocked(nx, y, z, R, H):
                ny = y
                m.vel[1] = 0.0
            elif not self.blocked(x, ny, z, R, H):
                nx = x
                m.vel[0] = 0.0
            else:
                nx, ny = x, y
                m.vel[0] = m.vel[1] = 0.0
        # vertical resolve
        nz = z + m.vel[2] * dt if not m.grounded else z
        f = self.floor_at(nx, ny, z)
        if m.grounded:
            if f < z - cfg["step_up"]:  # walked off an edge
                m.grounded = False
                nz = z
            else:
                nz = f
        if not m.grounded and nz <= f:
            nz = f
            m.vel[2] = 0.0
            m.grounded = True
        m.pos = [nx, ny, nz]
        # step sounds
        hs = math.hypot(m.vel[0], m.vel[1])
        if m.grounded and hs > self.cfg["sound"]["step_speed"]:
            if self.t - m.step_t >= self.cfg["sound"]["step_every"]:
                m.step_t = self.t
                self.sound(m, "boost" if boost else "step")

    def push_apart(self):
        R = self.cfg["mech"]["radius"]
        live = [m for m in self.mechs if m.alive]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                dx, dy = b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]
                d = math.hypot(dx, dy)
                if d < 2 * R and abs(a.pos[2] - b.pos[2]) < self.cfg["mech"]["height"]:
                    if d < 1e-6:
                        dx, dy, d = 1.0, 0.0, 1.0
                    push = (2 * R - d) / 2
                    a.pos[0] -= dx / d * push
                    a.pos[1] -= dy / d * push
                    b.pos[0] += dx / d * push
                    b.pos[1] += dy / d * push

    def aim_tick(self, m):
        cfg = self.cfg["mech"]
        dt = self.dt
        ty, tp = m.cmd_aim
        tp = clamp(tp, cfg["min_pitch"], cfg["max_pitch"])
        dy = wrap(ty - m.aim[0])
        step = cfg["turn_rate"] * dt
        m.aim[0] = wrap(m.aim[0] + clamp(dy, -step, step))
        dp = tp - m.aim[1]
        step = cfg["pitch_rate"] * dt
        m.aim[1] += clamp(dp, -step, step)

    def regen_tick(self, m):
        cfg = self.cfg["mech"]
        dt = self.dt
        if self.t - m.last_spend >= cfg["energy_regen_delay"]:
            m.energy = min(m.max_energy, m.energy + cfg["energy_regen"] * dt)
        zc, zr = self.zone["center"], self.zone["radius"]
        in_zone = math.dist(m.pos[:2], zc) <= zr
        if not in_zone:
            m.hp -= self.zone["dps"] * dt
            m.last_combat = self.t
            m.events.append({"type": "damaged", "amount": r2(self.zone["dps"] * dt), "weapon": "zone", "bearing": None, "by": None})
            if m.hp <= 0:
                m.hp = 0.0
                m.alive = False
        elif self.t - m.last_combat >= cfg["regen_delay"]:
            m.hp = min(m.max_hp, m.hp + cfg["regen_per_s"] * dt)

    # ---------------------------------------------------------------- vision
    def visible_point(self, viewer, eye, p):
        v = self.cfg["vision"]
        d = math.dist(eye[:2], p[:2])
        if d > v["range"]:
            return False
        if d > v["near_range"]:
            b = math.atan2(p[1] - eye[1], p[0] - eye[0])
            if abs(wrap(b - viewer.aim[0])) > math.radians(v["fov_deg"]) * 0.5:
                return False
        return self.los(eye, p)

    def can_see(self, viewer, other):
        eye = self.eye(viewer)
        c = self.center(other)
        top = (other.pos[0], other.pos[1], other.pos[2] + self.cfg["mech"]["height"])
        return self.visible_point(viewer, eye, c) or self.visible_point(viewer, eye, top)

    # ---------------------------------------------------------------- obs
    def apply_action(self, m, action):
        if action is None:
            action = {}
        a = {"move": action.get("move") or [0.0, 0.0],
             "boost": bool(action.get("boost", False)),
             "jump": bool(action.get("jump", False)),
             "fire": list(action.get("fire") or [False, False]),
             "reload": bool(action.get("reload", False))}
        if action.get("aim") is not None:
            m.cmd_aim = [float(action["aim"][0]), float(action["aim"][1])]
        m.action = a

    def build_obs(self, m):
        cfg = self.cfg
        v = cfg["vision"]
        weapons = []
        for w in m.weapons:
            if w is None:
                weapons.append(None)
                continue
            o = {"kind": w["kind"], "cooldown": r2(w["cooldown"]), "stats": dict(w["stats"])}
            if w["kind"] == "gun":
                o["ammo"] = w["ammo"]
                o["reload"] = r2(w["reload"])
            elif w["kind"] == "laser":
                o["charge"] = r2(w["charge"])
            weapons.append(o)
        in_zone = math.dist(m.pos[:2], self.zone["center"]) <= self.zone["radius"]
        me = {"id": m.id, "pos": [r2(c) for c in m.pos], "vel": [r2(c) for c in m.vel],
              "aim": [r3(m.aim[0]), r3(m.aim[1])], "hp": r2(m.hp), "max_hp": m.max_hp,
              "energy": r2(m.energy), "max_energy": m.max_energy, "grounded": m.grounded,
              "cubes": 0, "upgrades": [], "weapons": weapons, "in_zone": in_zone,
              "stats": dict(cfg["mech"])}
        mechs = []
        for o in self.mechs:
            if o is m or not o.alive:
                continue
            if self.can_see(m, o):
                m.seen_at[o.id] = self.t
            if self.t - m.seen_at.get(o.id, -99.0) <= v["memory_s"]:
                mechs.append({"id": o.id, "kind": "mech", "pos": [r2(c) for c in o.pos], "vel": [r2(c) for c in o.vel],
                              "aim": [r3(o.aim[0]), r3(o.aim[1])], "hp": r2(o.hp), "max_hp": o.max_hp, "cubes": 0,
                              "weapons": [w["kind"] for w in o.weapons if w], "charging": o.charging})
        eye = self.eye(m)
        projectiles = [{"id": b["id"], "kind": "gun", "pos": [r2(c) for c in b["pos"]], "vel": [r2(c) for c in b["vel"]]}
                       for b in self.bullets if b["by"] != m.id and self.visible_point(m, eye, b["pos"])]
        sounds = [{"kind": s["kind"], "bearing": s["bearing"], "dist": s["dist"], "age": r2(self.t - s["t"])} for s in m.sounds]
        beams = [{"by": b["by"], "kind": b["kind"], "path": b["path"], "age": r2(self.t - b["t"])} for b in m.beams]
        obs = {"type": "obs", "tick": self.tick, "t": r2(self.t), "alive": self.alive_shown, "self": me,
               "mechs": mechs, "projectiles": projectiles, "crates": [], "cubes": [], "pickups": [],
               "sounds": sounds, "beams": beams, "events": list(m.events),
               "zone": dict(self.zone), "offer": None, "drops": []}
        m.events, m.sounds, m.beams = [], [], []
        return obs

    # ---------------------------------------------------------------- loop
    def sim_tick(self):
        for m in self.mechs:
            if not m.alive:
                continue
            self.aim_tick(m)
            self.weapons_tick(m)
        self.bullets_tick()
        for m in self.mechs:
            if m.alive:
                self.move_tick(m)
        self.push_apart()
        for m in self.mechs:
            if m.alive:
                self.regen_tick(m)
        self.tick += 1
        self.t = self.tick * self.dt

    def run(self, controllers, trace=None):
        """controllers: list of callables obs->action, one per mech. Returns result dict."""
        every = self.cfg["match"]["decide_every"]
        while self.t < self.max_t and sum(m.alive for m in self.mechs) > 1:
            if self.tick % every == 0:
                for m, ctl in zip(self.mechs, controllers):
                    if not m.alive:
                        continue
                    obs = self.build_obs(m)
                    try:
                        act = ctl(obs)
                    except Exception as exc:  # a crashing bot idles, like the server
                        act = {}
                        self.log.append(f"t={self.t:.1f} {m.name} raised {exc!r}")
                    self.apply_action(m, act)
                if trace is not None:
                    trace.append([(r2(m.pos[0]), r2(m.pos[1]), r2(m.pos[2]), r2(m.hp)) for m in self.mechs])
            for _ in range(every):
                self.sim_tick()
                if sum(m.alive for m in self.mechs) <= 1:
                    break
        return self.result()

    def result(self):
        alive = [m for m in self.mechs if m.alive]
        winner = alive[0].id if len(alive) == 1 else None
        return {"winner": winner, "t": r2(self.t), "mechs": [
            {"id": m.id, "name": m.name, "hp": r2(m.hp), "alive": m.alive, **m.stats} for m in self.mechs],
            "log": self.log}
