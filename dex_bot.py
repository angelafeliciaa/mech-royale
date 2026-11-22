"""Globex Leader clean-room Python bot (dex v2 architecture)."""

import math
import random
import sys

from arena_map import ArenaMap
from dex_combat import (
    actor_center,
    actor_eye,
    angle_diff,
    compute_fire_and_aim,
    mech_scale,
    select_targets,
    weapon_slot,
)
from dex_nav import NavGrid
from dex_tactics import ZoneTracker, dodge_mortar, dodge_threats, fight_move
import mechfight as mf

# Branch profile selector: "v2", "stack", "aggro", or "duel".
ACTIVE_PROFILE = "aggro"
# Experiment knobs applied on top of every profile (base and squad).
OVERRIDES = {'laser_seek':400.0,'loot_radius':200.0,'light_range':85.0,'laser_range':150.0}

_WEAPON_RANK = {"lightning": 3, "laser": 2, "mortar": 1}

# Upgrade priority (econ pass). Per-stack value, sustained:
#   plating   +15% EHP (unconditional; death is terminal, so first)
#   gun_mag   20-round mag + 1.92 s reload = +31% sustained gun DPS (12 rounds / 2.4 s
#             reload is the gun's real bottleneck; gun_rate only gives +7.7% once reloads count)
#   gun_damage +15% DPS;  armor 8.7% / 9.5% / 10.5% EHP per stack
#   nanites   regen 45->72 HP/s and 5->3.5 s delay: heals a 700 HP deficit in 13 s instead
#             of 20 s between sequential hunters
#   laser_damage +18% per laser cycle (> laser_charge +8.6% cycle time)
#   lightning_efficiency +10% dmg and -35% energy (> range)
_UPGRADE_ORDER_BASE = (
    "plating", "gun_mag", "gun_damage", "armor", "nanites", "gun_rate",
    "lightning_efficiency", "laser_damage", "servos", "lightning_range", "laser_charge",
    "gun_velocity", "gyros", "lightning_chain",
    "reactor", "capacitor", "mortar_reload", "mortar_splash", "mortar_velocity",
    "sensors", "jets", "laser_pierce",
)
_UPGRADE_ORDER_LASER = (
    "plating", "laser_damage", "gun_mag", "gun_damage", "armor", "nanites",
    "laser_charge", "gun_rate", "lightning_efficiency", "lightning_range", "servos",
    "gun_velocity", "gyros", "lightning_chain", "reactor", "capacitor",
    "mortar_reload", "mortar_splash", "mortar_velocity", "sensors", "jets", "laser_pierce",
)
_UPGRADE_ORDER_LIGHTNING = (
    "plating", "lightning_efficiency", "gun_mag", "gun_damage", "armor", "nanites",
    "lightning_range", "gun_rate", "servos", "lightning_chain", "laser_damage",
    "laser_charge", "gun_velocity", "gyros", "reactor", "capacitor",
    "mortar_reload", "mortar_splash", "mortar_velocity", "sensors", "jets", "laser_pierce",
)


def make_params(profile: str):
    base = {
        "retreat_hp": 0.38,
        "recover_hp": 0.80,
        "gun_range": 85.0,
        "light_range": 28.0,
        "laser_range": 85.0,
        "engage_dist": 60.0,
        "farm_npc": True,
        "ghost": True,
        "loot_radius": 90.0,
        "laser_seek": 0.0,
        "crate_until": 150.0,
        "use_drops": False,
        "dodge": True,
        "no_mortar": True,
        "peek": True,
        "laser_first": True,
        "crowd_hide": True,
        "max_target": 215.0,
        "hpw": 60.0 / 1400.0,
        "stick": 0.0,
        "low_bonus": 0.0,
        "lose_rule": False,
        "fight_loot": 0.0,
        "farm_hp": 0.70,
        "finish": 0.0,
        "finish_hp": 0.38,
        "finish_rush": False,
        "lethal": False,
        "lethal_k": 0.75,
        "shy": 0.0,
        "opening": False,
        "travel_boost": False,
        "fight_boost": 0.0,
        "pursue": 0.0,
        "backstab": 0.0,
        "suppress_urgent_laser": False,
        "wall_dodge": False,
        "sound_scan": False,
        "explore": False,
        "shoot_crates": False,
        "dual_ring": False,
        "light_first": False,
        "light_seek": 0.0,
        # Survival knobs (0 / False disables each).
        "heal_hp": 0.0,        # enter heal-cycle (break LOS, regen) below this HP fraction
        "heal_to": 0.78,       # leave heal-cycle at this HP fraction
        "heal_max_s": 25.0,    # give up a heal-cycle after this long
        "final_center": True,  # final collapse: hold the exact zone center, shoot from there
        "face_hidden": True,   # keep torso toward the last known hunter while repositioning
    }
    if profile in ("stack", "aggro", "duel"):
        base["opening"] = True
        base["travel_boost"] = True
        base["suppress_urgent_laser"] = True
        base["wall_dodge"] = True
        base["use_drops"] = True
        base["sound_scan"] = True
    base.update(OVERRIDES)
    return base


def squad_params(base: dict, profile: str):
    p = dict(base)
    p.update({
        "ghost": False,
        "peek": False,
        "laser_first": True,
        "laser_range": 120.0,
        "loot_radius": 150.0,
        "laser_seek": 240.0,
        "crate_until": 1e9,
        "stick": 200.0,
        "hpw": 0.18,
        "low_bonus": 200.0,
        "max_target": 240.0,
        "lose_rule": True,
        "fight_loot": 70.0,
        "farm_hp": 0.50,
        "lethal": True,
        "lethal_k": 0.75,
        "finish": 20.0,
        "heal_hp": 0.40,
    })
    if profile in ("stack", "aggro", "duel"):
        p["pursue"] = 0.38
        p["backstab"] = 45.0
    if profile == "aggro":
        p["fight_boost"] = 55.0
        p["finish_rush"] = True
        p["explore"] = True
        p["shoot_crates"] = True
    elif profile == "duel":
        p["dual_ring"] = True
        p["laser_range"] = 145.0
        p["farm_hp"] = 0.40
    p.update(OVERRIDES)
    return p


class Bot:
    name = f"globex-dex-{ACTIVE_PROFILE}"

    def __init__(self, hello: mf.Hello, arena=None, profile: str = ACTIVE_PROFILE):
        self.profile = profile
        self.cfg = hello["config"]
        self.slot = int(hello.get("slot", 0))
        seed = int(hello.get("seed", 0)) or (self.slot + 1)
        self.rng = random.Random(seed)
        self.arena = ArenaMap.load(hello["map"]) if arena is None else arena
        self.nav = NavGrid(self.arena, self.cfg["mech"])
        self.zone = ZoneTracker()

        self.base_p = make_params(profile)
        self.p = dict(self.base_p)

        self.tracks = {}
        self.npcs = {}
        self.crates = {}
        self.cubes = {}
        self.pickups = {}
        self.bad = {}

        self.start_alive = None
        self.kills = 0
        self.other_dead = 0
        self.not_practice = False
        self.prac_on = False
        self.open_on = False

        self.stick_id = None
        self.stick_at = -99.0
        self.retreat = False
        self.healing = False
        self.heal_since = -99.0
        self.lose_at = -99.0
        self.cover_goal = None
        self.cover_at = -99.0
        self.last_hurt = -99.0
        self.hurt_bearing = 0.0
        self.npc_hurt = -99.0
        self.npc_bearing = 0.0
        self.sound_t = -99.0
        self.sound_bearing = 0.0
        self.last_threat_at = -99.0

        self.loot_id = None
        self.loot_since = 0.0
        # Where our last victim died (its cubes + 2 drop there), for pickup after the fight.
        self.kill_xy = None
        self.kill_at = -99.0
        self.kill_cubes = 0
        self.home_xy = None
        self.home_at = -99.0
        self.explore_xy = None
        self.explore_until = -99.0
        self.travel_boost_on = False

        self.stuck_since = -1.0
        self.unstuck_until = -1.0
        self.unstuck_dir = (1.0, 0.0)
        self.esc_launch = None
        self.esc_land = None
        self.esc_top = 0.0
        self.esc_at = -99.0
        self.esc_flying = False

        self.fight_state = {
            "strafe": 1.0 if self.slot % 2 else -1.0,
            "strafe_until": 0.0,
            "rng": self.rng,
        }
        self.last_aim = [0.0, 0.0]
        self.mode = "init"
        import gc
        gc.collect()
        gc.freeze()
        gc.set_threshold(50000, 20, 100)

    # -------------------------------------------------------------- public act
    def act(self, obs: mf.Observation) -> mf.Action:
        self.nav.begin_tick()
        try:
            action = self._step(obs)
            self.last_aim = list(action.get("aim", self.last_aim))
            return action
        except Exception as exc:
            print("dex_bot error:", repr(exc), file=sys.stderr, flush=True)
            return {"tick": obs["tick"], "move": [0.0, 0.0], "aim": list(self.last_aim)}

    # -------------------------------------------------------------- perception
    def _observe(self, obs):
        me = obs["self"]
        now = obs["t"]
        x, y, zf = me["pos"]
        st = me["stats"]
        eye = (x, y, zf + st["eye_height"])
        muzzle = (x, y, zf + st["muzzle_height"])

        if self.start_alive is None:
            self.start_alive = obs["alive"]

        for ev in obs.get("events", ()):
            et = ev.get("type")
            if et == "damaged":
                if ev.get("weapon") == "npc" and ev.get("bearing") is not None:
                    self.npc_hurt = now
                    self.npc_bearing = ev["bearing"]
                by = ev.get("by")
                if by is not None:
                    self.last_hurt = now
                    if ev.get("bearing") is not None:
                        self.hurt_bearing = ev["bearing"]
                    if by in self.tracks:
                        self.tracks[by]["hit_me"] = now
                    if by in self.npcs:
                        self.npcs[by]["hit_me"] = now
            elif et == "kill":
                self.kills += 1
                victim = self.tracks.pop(ev.get("target"), None)
                if victim is not None and now - victim["seen"] < 6.0:
                    vp = victim["pos"]
                    self.kill_xy = (vp[0], vp[1], vp[2])
                    self.kill_at = now
                    self.kill_cubes = int(victim.get("cubes", 0) or 0) + self.cfg["cubes"]["death_drop_extra"]
            elif et == "take_failed":
                self.bad[ev.get("id")] = now + 10.0
            elif et == "dropped":
                self.bad[ev.get("id")] = now + 30.0

        self.bad = {k: v for k, v in self.bad.items() if v > now}

        # Track loudest recent combat sound bearing for idle scanning.
        best_sw = 0.0
        for s in obs.get("sounds", ()):
            if s.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "explosion"):
                w = 2.0 if s.get("dist") == "near" else (1.0 if s.get("dist") == "mid" else 0.0)
                if w > best_sw and s.get("bearing") is not None:
                    best_sw = w
                    self.sound_t = now
                    self.sound_bearing = s["bearing"]

        # Update mechs & NPCs.
        for m in obs.get("mechs", ()):
            tab = self.npcs if m["kind"] == "npc" else self.tracks
            mid = m["id"]
            prev = tab.get(mid)
            pos = m["pos"]
            vel = m.get("vel", (0.0, 0.0, 0.0))
            dx, dy = pos[0] - x, pos[1] - y
            dist = math.hypot(dx, dy)
            closing = -(vel[0] * dx + vel[1] * dy) / max(dist, 1.0)
            acc = (0.0, 0.0, 0.0)
            if prev is not None and 0.0 < now - prev["seen"] < 0.15:
                idt = 1.0 / (now - prev["seen"])
                pv = prev["vel"]
                acc = ((vel[0] - pv[0]) * idt, (vel[1] - pv[1]) * idt, 0.0)
            entry = {
                "id": mid,
                "kind": m["kind"],
                "pos": pos,
                "vel": vel,
                "acc": acc,
                "aim": m.get("aim", (0.0, 0.0)),
                "hp": m.get("hp", m.get("max_hp", 1400.0)),
                "max_hp": m.get("max_hp", 1400.0),
                "cubes": m.get("cubes", 0),
                "weapons": m.get("weapons", ()),
                "charging": bool(m.get("charging", False)),
                "seen": now,
                "dist": dist,
                "closing": closing,
                "hit_me": prev["hit_me"] if prev else -100.0,
                "engage": False,
            }
            c = actor_center(self.cfg, entry)
            los = self.nav.los(muzzle, c)
            if not los and m["kind"] == "mech":
                h = self.cfg["mech"]["height"] * mech_scale(self.cfg, entry["cubes"])
                los = self.nav.los(muzzle, (pos[0], pos[1], pos[2] + h * 0.85))
            entry["los"] = los
            tab[mid] = entry

        for tab, ttl in ((self.tracks, 12.0), (self.npcs, 15.0)):
            for mid in [k for k, v in tab.items() if now - v["seen"] > ttl]:
                del tab[mid]

        vis_crates = set()
        for c in obs.get("crates", ()):
            vis_crates.add(c["id"])
            self.crates[c["id"]] = {"id": c["id"], "pos": c["pos"], "hp": c.get("hp", 300.0), "seen": now}
        vis_cubes = set()
        for c in obs.get("cubes", ()):
            vis_cubes.add(c["id"])
            self.cubes[c["id"]] = {"id": c["id"], "pos": c["pos"], "seen": now}
        vis_picks = set()
        for p in obs.get("pickups", ()):
            vis_picks.add(p["id"])
            self.pickups[p["id"]] = {"id": p["id"], "kind": p["kind"], "pos": p["pos"], "seen": now}

        # Only erase unseen remembered loot if we have clear LOS inside near_range (confirmed gone) or reached it.
        near_r = self.cfg["vision"]["near_range"] * 0.85
        for store, vis, ttl in (
            (self.crates, vis_crates, 60.0),
            (self.cubes, vis_cubes, 50.0),
            (self.pickups, vis_picks, 90.0),
        ):
            for iid in list(store):
                if iid in vis:
                    continue
                item = store[iid]
                ip = item["pos"]
                d = math.hypot(ip[0] - x, ip[1] - y)
                if now - item["seen"] > ttl or d < 4.5:
                    del store[iid]
                elif d < near_r and self.nav.los(eye, (ip[0], ip[1], ip[2] + 1.0)):
                    del store[iid]

        # Detect practice squad (nobody kills each other) vs FFA.
        if self.start_alive - obs["alive"] - self.kills >= 3:
            self.other_dead += 1
        else:
            self.other_dead = 0
        if self.other_dead >= 5:
            self.not_practice = True

        want_prac = (not self.not_practice) and now > 20.0
        if want_prac != self.prac_on:
            self.prac_on = want_prac
            self.p = squad_params(self.base_p, self.profile) if want_prac else dict(self.base_p)

        want_open = (
            self.base_p["opening"]
            and not self.not_practice
            and not self.prac_on
            and (self.start_alive - obs["alive"] - self.kills <= 0)
        )
        if want_open != self.open_on:
            self.open_on = want_open
            if want_open:
                self.p = dict(self.base_p)
                self.p["ghost"] = False
                self.p["laser_first"] = True
                self.p["laser_seek"] = 240.0
                self.p["loot_radius"] = 150.0
                self.p["explore"] = True
                self.p["shoot_crates"] = True
                self.p.update(OVERRIDES)
            elif self.prac_on:
                self.p = squad_params(self.base_p, self.profile)
            else:
                self.p = dict(self.base_p)

        self.zone.update(obs["zone"], now)

    # -------------------------------------------------------------- upgrades
    def _pick_upgrade(self, obs):
        """Loadout-aware combat priority: plating/armor > primary weapon DPS > sustain/mobility."""
        offer = obs.get("offer")
        if not offer or not offer.get("choices"):
            return None
        me = obs["self"]
        held = {w["kind"] for w in me["weapons"] if w is not None}
        if "laser" in held:
            order = _UPGRADE_ORDER_LASER
        elif "lightning" in held:
            order = _UPGRADE_ORDER_LIGHTNING
        else:
            order = _UPGRADE_ORDER_LASER if (self.prac_on or self.open_on) else _UPGRADE_ORDER_BASE

        rank_map = {uid: idx for idx, uid in enumerate(order)}
        choices = offer["choices"]
        return min(range(len(choices)), key=lambda i: rank_map.get(choices[i]["id"], len(order)))

    # -------------------------------------------------------------- weapons/loot
    def _want_pickup(self, me, kind):
        if kind == "gun":
            return len(me["weapons"]) > 0 and me["weapons"][0] is None
        if weapon_slot(me["weapons"], kind)[0] >= 0:
            return False
        if kind == "mortar" and self.p["no_mortar"]:
            return False

        def rank(k):
            if self.p["light_first"] and k == "lightning":
                return 5
            if self.p["laser_first"] and k == "laser":
                return 4
            return _WEAPON_RANK.get(k, 0)

        cur = rank(me["weapons"][1]["kind"]) if len(me["weapons"]) > 1 and me["weapons"][1] is not None else 0
        return rank(kind) > cur

    def _choose_loot(self, obs, hp_frac, danger):
        me = obs["self"]
        zone = obs["zone"]
        now = obs["t"]
        x, y, _ = me["pos"]
        cx, cy, cr = self.zone.current(zone, now)
        walk = max(1.0, me["stats"]["walk_speed"])
        cube_cap = self.cfg["cubes"]["stat_cap"]

        best = None
        best_u = 0.0

        def consider(kind, item_id, pos, value):
            nonlocal best, best_u
            if item_id in self.bad:
                return
            d = math.hypot(pos[0] - x, pos[1] - y)
            if d > 260.0:
                return
            if math.hypot(pos[0] - cx, pos[1] - cy) > cr - 6.0:
                return
            if now + d * 1.4 / walk + 5.0 > self.zone.deadline(zone, pos[0], pos[1], 6.0):
                return
            if pos[2] - self.arena.floor(pos[0], pos[1]) > 3.2:
                return
            if not self.nav.reachable(pos[0], pos[1]):
                return
            u = value / (d + 18.0)
            if item_id == self.loot_id:
                u *= 1.35
            if u > best_u:
                best_u = u
                best = (kind, item_id, pos, d)

        loot_r = self.p["loot_radius"]
        laser_seek = self.p["laser_seek"]
        for p in self.pickups.values():
            seek = p["kind"] == "laser" and laser_seek > loot_r
            rad = laser_seek if seek else loot_r
            if p["kind"] == "lightning" and self.p["light_seek"] > rad:
                seek, rad = True, self.p["light_seek"]
            if self._want_pickup(me, p["kind"]) and math.hypot(p["pos"][0] - x, p["pos"][1] - y) < rad:
                v = 130.0 if (len(me["weapons"]) > 1 and me["weapons"][1] is None) else 60.0
                if seek:
                    v = max(v, 150.0)
                consider("pickup", p["id"], p["pos"], v)

        cube_v = 25.0 if me["cubes"] >= cube_cap else 75.0
        cube_r = max(60.0, loot_r * 0.7)
        for c in self.cubes.values():
            if math.hypot(c["pos"][0] - x, c["pos"][1] - y) < cube_r:
                consider("cube", c["id"], c["pos"], cube_v)

        # Cubes from our last kill: head to the death spot even if the cubes are not in view yet.
        if self.kill_xy is not None:
            kd = math.hypot(self.kill_xy[0] - x, self.kill_xy[1] - y)
            if now - self.kill_at > 25.0 or kd < 5.0:
                self.kill_xy = None
            elif me["cubes"] < cube_cap + 2 and kd < 120.0:
                consider("ksite", -1, self.kill_xy, 55.0 + 22.0 * min(self.kill_cubes, 8))

        # Early-game hunger: the squad sends one weak hunter at first, so power up before the rest arrive.
        early = now < 100.0 and me["cubes"] < 6
        npc_v = 85.0 * (1.5 if early else 1.0)
        crate_v = cube_v * 0.6 * (1.3 if early else 1.0)
        if hp_frac > self.p["farm_hp"] and not danger and now < self.p["crate_until"]:
            for c in self.crates.values():
                if math.hypot(c["pos"][0] - x, c["pos"][1] - y) < loot_r:
                    consider("crate", c["id"], c["pos"], crate_v)
            if self.p["farm_npc"] and me["cubes"] < cube_cap:
                for n in self.npcs.values():
                    if now - n["seen"] < 12.0 and math.hypot(n["pos"][0] - x, n["pos"][1] - y) < loot_r:
                        consider("npc", n["id"], n["pos"], npc_v)

        if self.p["use_drops"] and not danger:
            my_id = me.get("id")
            # Contest from farther away when healthy and few hunters have been seen recently.
            hunters = sum(1 for t in self.tracks.values() if now - t["seen"] < 6.0)
            drop_r = 175.0 if (hp_frac > 0.55 and hunters <= 1) else 115.0
            for d in obs.get("drops", ()):
                dp = d["pos"]
                dist = math.hypot(dp[0] - x, dp[1] - y)
                lands = d.get("lands_in", 0.0) or 0.0
                if dist > drop_r or dist / walk > lands + 15.0:
                    continue
                holder = d.get("holder")
                if holder is not None and holder != my_id:
                    continue
                if any(
                    now - t["seen"] < 3.0 and math.hypot(t["pos"][0] - dp[0], t["pos"][1] - dp[1]) < d["radius"] + 35.0
                    for t in self.tracks.values()
                ):
                    continue
                # 3 cubes + full repair: worth more the more HP we are missing.
                consider("drop", d["id"], dp, 240.0 + 120.0 * (1.0 - hp_frac))

        if best is None:
            self.loot_id = None
            return None
        if best[1] != self.loot_id:
            self.loot_id = best[1]
            self.loot_since = now
        else:
            # A drop can legitimately take walk + fall (25 s) + capture (4 s); don't give up on it early.
            limit = 30.0
            if best[0] == "drop":
                lands = 0.0
                for d in obs.get("drops", ()):
                    if d["id"] == best[1]:
                        lands = d.get("lands_in", 0.0) or 0.0
                limit = 40.0 + lands
            if now - self.loot_since > limit:
                if best[1] >= 0:
                    self.bad[best[1]] = now + 60.0
                else:
                    self.kill_xy = None
                self.loot_id = None
                return None
        return best

    # -------------------------------------------------------------- courtyard/rooftop escape
    def _escape_rooftop(self, me, now):
        pos = me["pos"]
        x, y, zf = pos
        if me["grounded"]:
            if not self.nav.off_main(pos):
                self.esc_launch = self.esc_land = None
                self.esc_flying = False
                return None
            self.esc_flying = False
            if self.esc_launch is None or now - self.esc_at > 6.0:
                plan = self.nav.escape_plan(x, y)
                self.esc_at = now
                if plan is None:
                    self.esc_launch = self.esc_land = None
                    return None
                self.esc_launch, self.esc_land, self.esc_top = plan
            if math.hypot(self.esc_launch[0] - x, self.esc_launch[1] - y) > 2.5:
                mx, my = self.nav.steer_to(pos, self.esc_launch, now)
                return (mx, my, False, False)
            if me["energy"] < min(70.0, me["max_energy"] * 0.7):
                return (0.0, 0.0, False, False)
            self.esc_flying = True
            dx, dy = self.esc_land[0] - x, self.esc_land[1] - y
            d = math.hypot(dx, dy) or 1.0
            return (dx / d, dy / d, False, True)

        if not self.esc_flying or self.esc_land is None:
            return None
        dx, dy = self.esc_land[0] - x, self.esc_land[1] - y
        d = math.hypot(dx, dy)
        if d < 1.5:
            return (0.0, 0.0, False, False)
        ux, uy = dx / d, dy / d
        # Hold jetpack until clearing the wall top between us and landing.
        need_z = self.esc_top + 2.5
        return (ux, uy, zf < need_z and me["energy"] > 1.0, False)

    # -------------------------------------------------------------- home / explore
    def _home_goal(self, obs):
        zone = obs["zone"]
        now = obs["t"]
        x, y, _ = obs["self"]["pos"]
        nx, ny = zone["next_center"]
        nr = zone["next_radius"]
        out = self.home_xy is None or math.hypot(self.home_xy[0] - nx, self.home_xy[1] - ny) > nr * 0.8
        if out or now - self.home_at > 12.0:
            self.home_at = now
            best_pt = None
            best_c = math.inf
            for _ in range(30):
                a = self.rng.random() * math.tau
                r = math.sqrt(self.rng.random()) * nr * 0.7
                px, py = nx + math.cos(a) * r, ny + math.sin(a) * r
                if not self.nav.reachable(px, py):
                    continue
                crowd = sum(
                    max(0.0, 80.0 - math.hypot(t["pos"][0] - px, t["pos"][1] - py)) / 80.0
                    for t in self.tracks.values() if now - t["seen"] < 20.0
                )
                c = 0.25 * math.hypot(px - x, py - y) + 40.0 * crowd + (10.0 if not self.nav.is_near_wall(px, py) else 0.0)
                if c < best_c:
                    best_c = c
                    best_pt = (px, py)
            if best_pt is not None:
                self.home_xy = best_pt
        return self.home_xy

    def _explore_goal(self, obs):
        zone = obs["zone"]
        now = obs["t"]
        x, y, _ = obs["self"]["pos"]
        nx, ny = zone["next_center"]
        nr = zone["next_radius"]
        if (self.explore_xy is None or now > self.explore_until
                or math.hypot(self.explore_xy[0] - x, self.explore_xy[1] - y) < 8.0
                or math.hypot(self.explore_xy[0] - nx, self.explore_xy[1] - ny) > nr * 0.85):
            for _ in range(20):
                a = self.rng.random() * math.tau
                r = math.sqrt(self.rng.random()) * nr * 0.75
                px, py = nx + math.cos(a) * r, ny + math.sin(a) * r
                if self.nav.reachable(px, py):
                    self.explore_xy = (px, py)
                    self.explore_until = now + 8.0
                    break
        return self.explore_xy

    # -------------------------------------------------------------- main tick
    def _step(self, obs):
        self._observe(obs)
        me = obs["self"]
        zone = obs["zone"]
        now = obs["t"]
        x, y, zf = me["pos"]
        st = me["stats"]
        hp_frac = me["hp"] / max(1.0, me["max_hp"])

        act: mf.Action = {"tick": obs["tick"], "move": [0.0, 0.0], "boost": False, "jump": False}
        pick = self._pick_upgrade(obs)
        if pick is not None:
            act["pick"] = pick

        _, gun = weapon_slot(me["weapons"], "gun")
        _, light = weapon_slot(me["weapons"], "lightning")
        _, laser = weapon_slot(me["weapons"], "laser")
        late = zone["next_radius"] <= 70.0 or obs["alive"] <= 3

        seen, target, hidden, shy = select_targets(
            self.cfg, me, self.tracks, now, self.p, self.stick_id, self.stick_at, obs["alive"], late,
        )
        if target is not None:
            self.stick_id = target["id"]
            self.stick_at = now

        foe = target
        if foe is None and self.p["peek"]:
            foe = hidden
        if foe is None and self.p["pursue"] > 0.0 and hidden is not None and hidden["hp"] < self.p["pursue"] * max(1.0, hidden["max_hp"]):
            foe = hidden

        danger = False
        crowd = 0
        for t in seen:
            if t["engage"] or now - t["hit_me"] < 4.0:
                danger = True
            if t["seen"] == now and (t["engage"] or (t["los"] and t["dist"] < 150.0)):
                crowd += 1
        if danger or now - self.last_hurt < 3.0:
            self.last_threat_at = now

        finishing = target is not None and target["dist"] < 120.0 and (target["hp"] < 200.0 or target["hp"] < me["hp"] * 0.5)
        outgunned = target is not None and "laser" in target.get("weapons", ()) and laser is None and target["dist"] > 55.0 and hp_frac < 0.8
        swamped = self.p["crowd_hide"] and (crowd >= 3 or (crowd == 2 and hp_frac < 0.65) or outgunned)

        if not self.retreat and ((hp_frac < self.p["retreat_hp"] and now - self.last_threat_at < 2.0) or swamped):
            self.retreat = True
            self.cover_at = -99.0
        if self.retreat and not swamped and (hp_frac > self.p["recover_hp"] or (now - self.last_threat_at > 7.0 and hp_frac > 0.55)):
            self.retreat = False

        if self.p["lose_rule"]:
            near = sum(1 for t in seen if t["seen"] == now and t["dist"] < 160.0)
            losing = (near >= 2 and hp_frac < 0.55) or (near >= 3 and hp_frac < 0.80)
            if target is not None:
                tf = target["hp"] / max(1.0, target["max_hp"])
                if hp_frac < 0.35 and tf > hp_frac + 0.10:
                    losing = True
                if tf < 0.25 and hp_frac > 0.20 and near <= 2:
                    losing = False
            if losing:
                self.lose_at = now
            self.retreat = self.lose_at > 0.0 and (now - self.lose_at < 1.5)

        # Heal cycle: low on HP and not winning the trade -> break LOS and regen
        # (45 HP/s after 5 s quiet) instead of trading down; stay hidden until healed.
        heal_hp = self.p.get("heal_hp", 0.0)
        if heal_hp > 0.0 and not late:
            if not self.healing:
                tf = target["hp"] / max(1.0, target["max_hp"]) if target is not None else None
                threatened = danger or now - self.last_hurt < 4.0
                if (hp_frac < heal_hp and threatened and not finishing
                        and (tf is None or tf > hp_frac + 0.10 or crowd >= 2)):
                    self.healing = True
                    self.heal_since = now
                    self.cover_at = -99.0
            elif (hp_frac >= self.p.get("heal_to", 0.78) or finishing
                    or now - self.heal_since > self.p.get("heal_max_s", 25.0)):
                self.healing = False
            if self.healing:
                self.retreat = True
        else:
            self.healing = False

        zgx, zgy = self.zone.goal(zone, x, y)
        zdist = math.hypot(zgx - x, zgy - y)
        walk = max(1.0, st["walk_speed"])
        t_need = zdist * 1.45 / walk + 4.0
        deadline = self.zone.deadline(zone, x, y, 8.0)
        zone_must = (not me["in_zone"]) or (now + t_need + 3.0 >= deadline)
        zone_should = now + t_need + 14.0 >= deadline
        # Final collapse (next radius 0, 260 HP/s): every metre from the centre costs ~1-2 s of
        # 260 dps, so hold the exact centre and shoot from there instead of fighting at the edge.
        final_rush = (
            self.p.get("final_center", False)
            and zone["next_radius"] <= 1.0
            and now >= zone["shrink_start"] - 8.0
        )
        if final_rush:
            zgx, zgy = zone["next_center"][0], zone["next_center"][1]
            zdist = math.hypot(zgx - x, zgy - y)

        mx = my = 0.0
        boost = False
        jump = False
        mode = "idle"
        farm = None
        farm_pos = None
        travel_dist = 0.0

        esc = self._escape_rooftop(me, now)
        mortar_mv = dodge_mortar(self.cfg, self.arena, self.nav, obs) if esc is None else None

        if esc is not None:
            mx, my, boost, jump = esc
            mode = "escape"
        elif now < self.unstuck_until:
            mx, my = self.unstuck_dir
            mode = "unstuck"
        elif mortar_mv is not None:
            mx, my = mortar_mv
            boost = me["energy"] > 15.0
            mode = "dodge"
        elif zone_must and zdist > 2.0:
            mx, my = self.nav.steer_to(me["pos"], (zgx, zgy), now)
            boost = (not me["in_zone"] or now + t_need >= deadline) and me["energy"] > 5.0
            mode = "zone"
        elif final_rush and zdist > 0.8:
            mx, my = self.nav.steer_to(me["pos"], (zgx, zgy), now)
            boost = me["energy"] > 20.0 and zdist > 6.0
            # Far out: "zone" mode suppresses half-speed laser charging; near: shoot freely.
            mode = "zone" if zdist > 6.0 else "center"
        elif final_rush:
            mode = "center"
        elif self.retreat and not finishing and not late:
            threats = [actor_eye(self.cfg, t) for t in sorted(seen, key=lambda t: t["dist"])[:4] if t["dist"] < 230.0]
            if threats and now - self.cover_at >= 0.4:
                self.cover_at = now
                self.cover_goal = self.nav.find_cover(
                    me["pos"], threats, st, lambda cx, cy: self.zone.penalty(zone, now, cx, cy),
                )
            # Not under fire right now: a cube within a short walk heals +4% max HP and is permanent power.
            heal_cube = None
            if hp_frac > 0.22 and (not threats or now - self.last_hurt > 2.5):
                heal_d = 40.0
                for c in self.cubes.values():
                    cp = c["pos"]
                    cd = math.hypot(cp[0] - x, cp[1] - y)
                    if (cd < heal_d and abs(cp[2] - zf) < 4.0
                            and self.nav.reachable(cp[0], cp[1])
                            and self.zone.penalty(zone, now, cp[0], cp[1]) == 0.0):
                        heal_d = cd
                        heal_cube = cp
            if heal_cube is not None:
                mx, my = self.nav.steer_to(me["pos"], (heal_cube[0], heal_cube[1]), now)
                boost = me["energy"] > 10.0 and (mx != 0.0 or my != 0.0)
                mode = "cover:cube"
            elif self.cover_goal is not None:
                if math.hypot(self.cover_goal[0] - x, self.cover_goal[1] - y) >= 1.5:
                    mx, my = self.nav.steer_to(me["pos"], self.cover_goal, now)
                boost = me["energy"] > 10.0 and (mx != 0.0 or my != 0.0)
                mode = "cover"
            elif target is not None:
                mx, my = fight_move(
                    self.cfg, self.arena, self.nav, self.zone, obs, target, seen, self.cubes, 1.0, self.p, self.fight_state,
                )
                mode = "cornered"
            else:
                mx, my = self.nav.steer_to(me["pos"], (zgx, zgy), now)
                mode = "flee"
        elif foe is not None:
            ready = gun is not None and gun.get("ammo", 0) > 0 and gun.get("reload", 0.0) <= 0.25 and foe["dist"] <= gun["stats"]["range"] - 10.0
            hot = laser is not None and laser.get("charge", 0.0) >= laser["stats"]["charge"] - 0.3 and foe["dist"] <= laser["stats"]["range"] - 5.0
            if hot or (light is not None and foe["dist"] <= light["stats"]["range"] + 6.0 and me["energy"] >= light["stats"]["energy"] * 3.0):
                ready = True
            if not self.p["peek"] and (light is not None or laser is not None):
                ready = True
            want_los = 1.0 if ready else -1.0
            if self.p["peek"] and not hot:
                for t in seen:
                    if t["seen"] == now and t["charging"] and abs(angle_diff(t["aim"][0], math.atan2(y - t["pos"][1], x - t["pos"][0]))) < 0.3:
                        want_los = -1.6

            move_p = self.p
            closing_finish = (
                self.p["finish"] > 0.0
                and foe["hp"] < self.p["finish_hp"] * max(1.0, foe["max_hp"])
                and hp_frac > 0.42
                and crowd <= 2
            )
            if closing_finish:
                move_p = dict(self.p)
                move_p["gun_range"] = move_p["laser_range"] = move_p["light_range"] = self.p["finish"]
            mx, my = fight_move(
                self.cfg, self.arena, self.nav, self.zone, obs, foe, seen, self.cubes, want_los, move_p, self.fight_state,
            )
            if light is not None and foe["dist"] > 50.0 and me["energy"] > 55.0 and want_los > 0.0:
                boost = True
            if self.p["finish_rush"] and closing_finish and foe["dist"] > 40.0 and me["energy"] > 35.0 and (laser is None or laser.get("charge", 0.0) <= 0.0):
                boost = True
            if self.p["fight_boost"] > 0.0 and me["energy"] > self.p["fight_boost"] and (laser is None or laser.get("charge", 0.0) <= 0.0) and (light is None or me["energy"] > 60.0):
                boost = True
            mode = "fight"
            if self.p["fight_loot"] > 0.0:
                r_lim = self.p["fight_loot"] * (0.57 if hp_frac <= 0.5 else 1.0)
                best_cube = None
                for c in self.cubes.values():
                    cp = c["pos"]
                    cd = math.hypot(cp[0] - x, cp[1] - y)
                    if (cd < r_lim and abs(cp[2] - zf) < 4.0
                            and self.nav.reachable(cp[0], cp[1])
                            and self.zone.penalty(zone, now, cp[0], cp[1]) == 0.0):
                        r_lim = cd
                        best_cube = cp
                if best_cube is not None:
                    mx, my = self.nav.steer_to(me["pos"], (best_cube[0], best_cube[1]), now)
                    mode = "fight:cube"
            # Keep banking a supply-drop capture (3 cubes + full HP) while trading at range with one foe.
            if self.p["use_drops"] and hp_frac > 0.45 and crowd <= 1 and foe["dist"] > 35.0:
                my_id = me.get("id")
                for d in obs.get("drops", ()):
                    if (d.get("lands_in", 1.0) or 0.0) > 0.0 or d.get("holder") not in (None, my_id):
                        continue
                    dp = d["pos"]
                    if math.hypot(dp[0] - x, dp[1] - y) < d["radius"] - 2.0 and abs(dp[2] - zf) < 4.0:
                        mx, my = 0.0, 0.0
                        boost = False
                        mode = "fight:drop"
                        break
        elif zone_should and zdist > 2.0:
            mx, my = self.nav.steer_to(me["pos"], (zgx, zgy), now)
            travel_dist = zdist
            mode = "zone"
        else:
            loot = self._choose_loot(obs, hp_frac, danger)
            if loot is not None:
                lkind, lid, lpos, ldist = loot
                mode = f"loot:{lkind}"
                travel_dist = ldist
                if lkind == "npc":
                    npc = self.npcs.get(lid)
                    if npc is not None and now - npc["seen"] < 1.0 and npc["los"] and npc["dist"] < 75.0:
                        farm = npc
                        mx, my = fight_move(
                            self.cfg, self.arena, self.nav, self.zone, obs, npc, seen, self.cubes, 1.0, self.p, self.fight_state,
                        )
                    else:
                        mx, my = self.nav.steer_to(me["pos"], (lpos[0], lpos[1]), now)
                elif lkind == "crate":
                    cp = (lpos[0], lpos[1], lpos[2] + self.cfg["crates"]["height"] * 0.5)
                    if ldist < 50.0 and self.nav.los((x, y, zf + st["muzzle_height"]), cp):
                        farm_pos = cp
                        if ldist < 12.0:
                            d = max(ldist, 0.1)
                            mx, my = (x - lpos[0]) / d, (y - lpos[1]) / d
                    else:
                        mx, my = self.nav.steer_to(me["pos"], (lpos[0], lpos[1]), now)
                else:
                    if lkind == "drop" and ldist < 2.5:
                        mx, my = 0.0, 0.0
                    else:
                        mx, my = self.nav.steer_to(me["pos"], (lpos[0], lpos[1]), now)
            elif zdist > 3.0 and (not self.p["ghost"] or zone["next_radius"] <= 30.0):
                mx, my = self.nav.steer_to(me["pos"], (zgx, zgy), now)
                travel_dist = zdist
                mode = "zone-walk"
            elif self.p["ghost"]:
                home = self._home_goal(obs)
                if home is not None and math.hypot(home[0] - x, home[1] - y) > 3.0:
                    mx, my = self.nav.steer_to(me["pos"], home, now)
                    if not danger and now - self.last_hurt > 6.0 and not self.open_on:
                        q = min(1.0, 4.6 / walk)
                        mx, my = mx * q, my * q
                    mode = "home"
            elif self.p["explore"] and not late and me["cubes"] < self.cfg["cubes"]["stat_cap"] and hp_frac > 0.65 and not danger:
                exp = self._explore_goal(obs)
                if exp is not None:
                    mx, my = self.nav.steer_to(me["pos"], exp, now)
                    travel_dist = math.hypot(exp[0] - x, exp[1] - y)
                    mode = "explore"

        # Safe travel boost with energy hysteresis.
        if self.p["travel_boost"] and not danger and travel_dist > 40.0 and (laser is None or laser.get("charge", 0.0) <= 0.0):
            if self.travel_boost_on and me["energy"] < 40.0:
                self.travel_boost_on = False
            elif not self.travel_boost_on and me["energy"] > 60.0:
                self.travel_boost_on = True
            if self.travel_boost_on:
                boost = True
        else:
            if me["energy"] < 40.0:
                self.travel_boost_on = False

        # Bullet & charging-laser dodging.
        if self.p["dodge"] and mode not in ("unstuck", "escape", "center"):
            ddx, ddy, lz = dodge_threats(
                self.cfg, self.arena, self.nav, self.zone, obs,
                self.fight_state["strafe"], wall_aware=self.p["wall_dodge"],
            )
            if ddx != 0.0 or ddy != 0.0:
                mx = mx * 0.4 + ddx * 0.9
                my = my * 0.4 + ddy * 0.9
                ml = math.hypot(mx, my)
                if ml > 1.0:
                    mx, my = mx / ml, my / ml
                if lz and me["energy"] > 25.0:
                    boost = True

        # Stuck detection & jump recovery.
        if mode == "escape":
            self.stuck_since = -1.0
        elif (mx != 0.0 or my != 0.0) and math.hypot(me["vel"][0], me["vel"][1]) < 1.2 and me["grounded"]:
            if self.stuck_since < 0.0:
                self.stuck_since = now
            elif now - self.stuck_since > 0.7:
                a = self.rng.random() * math.tau
                self.unstuck_dir = (math.cos(a), math.sin(a))
                self.unstuck_until = now + 0.5
                self.stuck_since = -1.0
                self.nav.clear_path()
                jump = True
        else:
            self.stuck_since = -1.0

        # Self-defense against attacking NPC drones when no mech target.
        defend_npc = False
        if target is None and self.npc_hurt > 0.0 and now - self.npc_hurt < 3.0:
            worst = None
            best_off = 0.7
            for n in self.npcs.values():
                if now - n["seen"] > 0.5 or not n["los"] or n["dist"] > 150.0:
                    continue
                off = 0.0 if now - n["hit_me"] < 3.0 else abs(angle_diff(math.atan2(n["pos"][1] - y, n["pos"][0] - x), self.npc_bearing))
                if off < best_off:
                    best_off = off
                    worst = n
            if worst is not None:
                farm = worst
                farm_pos = None
                defend_npc = True

        # Opportunistic crate shooting while walking past (dex-v2-aggro).
        if self.p["shoot_crates"] and target is None and farm is None and farm_pos is None and gun is not None and not self.retreat and not danger:
            muzzle = (x, y, zf + st["muzzle_height"])
            best_cd = 65.0
            for c in self.crates.values():
                if c["id"] in self.bad:
                    continue
                cp = (c["pos"][0], c["pos"][1], c["pos"][2] + self.cfg["crates"]["height"] * 0.5)
                cd = math.hypot(cp[0] - x, cp[1] - y)
                if 6.0 < cd < best_cd and self.nav.los(muzzle, cp):
                    best_cd = cd
                    farm_pos = cp

        if target is None and (self.retreat or hp_frac < 0.5) and not defend_npc:
            farm = None
            farm_pos = None

        aim, fire, want_reload = compute_fire_and_aim(
            self.cfg, self.arena, me, target, foe, farm, farm_pos, seen,
            (mx, my), now, mode, self.p, self.last_hurt, self.hurt_bearing,
            self.npc_hurt, self.npc_bearing, self.last_threat_at,
            self.sound_t, self.sound_bearing,
            face=(hidden if self.p.get("face_hidden", False) else None),
        )

        # Weapon pickups.
        pickup_r = self.cfg["loot"]["pickup_radius"] - 0.3
        slot1 = me["weapons"][1] if len(me["weapons"]) > 1 else None
        for p in obs.get("pickups", ()):
            pp = p["pos"]
            if math.hypot(pp[0] - x, pp[1] - y) > pickup_r or abs(pp[2] - zf) > 3.6:
                continue
            if not self._want_pickup(me, p["kind"]):
                continue
            if slot1 is not None and me["weapons"][0] is not None:
                if slot1["kind"] == "laser" and slot1.get("charge", 0.0) > 0.0:
                    break
                act["drop"] = 1
            act["take"] = p["id"]
            break

        act["move"] = [mx, my]
        act["boost"] = bool(boost and (mx != 0.0 or my != 0.0))
        act["jump"] = bool(jump)
        act["aim"] = aim
        act["fire"] = fire
        if want_reload:
            act["reload"] = True
        self.mode = mode
        return act


if __name__ == "__main__":
    mf.run(Bot)
