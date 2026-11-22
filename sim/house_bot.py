"""Scripted duel opponent with tunable skill. Uses only its obs (same information as a real bot)."""
from __future__ import annotations

import math
import random
from collections import deque

SKILLS = {
    # aim_noise: deg sigma added to every commanded aim; delay: seconds of reaction lag;
    # lead: fraction of the correct lead applied; dodge: strafe-flip rate.
    "easy": {"aim_noise": 4.0, "delay": 0.4, "lead": 0.5, "range": 90.0, "flip": 0.4},
    "mid": {"aim_noise": 1.5, "delay": 0.2, "lead": 0.9, "range": 75.0, "flip": 0.7},
    "hard": {"aim_noise": 0.4, "delay": 0.1, "lead": 1.0, "range": 60.0, "flip": 1.0},
    "expert": {"aim_noise": 0.15, "delay": 0.0, "lead": 1.0, "range": 50.0, "flip": 1.2},
}


def wrap(a):
    return (a + math.pi) % math.tau - math.pi


class HouseBot:
    name = "house"

    def __init__(self, hello, skill="mid", seed=0):
        self.cfg = hello["config"]
        self.s = dict(SKILLS[skill])
        self.rng = random.Random(seed)
        self.buf = deque()
        self.strafe = 1.0
        self.flip_at = 0.0
        self.last_seen = None
        self.last_seen_t = -99.0
        self.hurt_bearing = None
        self.hurt_t = -99.0
        self.sound_bearing = 0.0
        self.sound_t = -99.0
        self.aim = [0.0, 0.0]
        self.prog_pos = None
        self.prog_t = 0.0
        self.detour = None
        self.detour_until = -1.0

    def act(self, obs):
        act = self._act(obs)
        return self._unstick(obs, act)

    def _unstick(self, obs, act):
        """If we commanded movement but didn't progress for 1 s, detour sideways for 1.5 s (walls)."""
        now = obs["t"]
        p = obs["self"]["pos"]
        mv = act.get("move") or [0.0, 0.0]
        if math.hypot(*mv) < 0.1:
            self.prog_pos, self.prog_t = p, now
            return act
        if self.detour is not None and now < self.detour_until:
            act["move"] = self.detour
            act["jump"] = True
            act["boost"] = True
            return act
        if self.prog_pos is None or math.dist(p[:2], self.prog_pos[:2]) > 3.0:
            self.prog_pos, self.prog_t = p, now
        elif now - self.prog_t > 1.0:
            side = 1.0 if self.rng.random() < 0.5 else -1.0
            self.detour = [-mv[1] * side, mv[0] * side]
            self.detour_until = now + 1.5
            self.prog_pos, self.prog_t = p, now
            act["move"] = self.detour
        return act

    def _act(self, obs):
        now = obs["t"]
        # reaction delay: act on a delayed view of the enemy
        self.buf.append((now, obs))
        delayed = obs
        while self.buf and now - self.buf[0][0] > self.s["delay"]:
            delayed = self.buf.popleft()[1]
        me = obs["self"]
        x, y, z = me["pos"]
        st = me["stats"]
        act = {"tick": obs["tick"], "move": [0.0, 0.0], "boost": False, "fire": [False, False]}

        for ev in obs["events"]:
            if ev["type"] == "damaged" and ev.get("bearing") is not None:
                self.hurt_bearing, self.hurt_t = ev["bearing"], now
        foe = None
        for mch in delayed["mechs"]:
            if mch["kind"] == "mech":
                foe = mch
                break
        if foe is not None:
            self.last_seen, self.last_seen_t = foe, now

        if foe is None:
            # search: face damage source / sound / last known and walk there, else drift to zone center
            snd = [s for s in obs["sounds"] if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "step", "boost")]
            if snd:
                self.sound_bearing, self.sound_t = snd[-1]["bearing"], now
            if self.hurt_bearing is not None and now - self.hurt_t < 3.0:
                b = self.hurt_bearing
                self.aim = [b, 0.0]
                act["move"] = [math.cos(b), math.sin(b)]
            elif self.last_seen is not None and now - self.last_seen_t < 8.0:
                p = self.last_seen["pos"]
                b = math.atan2(p[1] - y, p[0] - x)
                self.aim = [b, 0.0]
                d = math.hypot(p[0] - x, p[1] - y)
                act["move"] = [(p[0] - x) / d, (p[1] - y) / d] if d > 8 else [0.0, 0.0]
            elif self.sound_t > now - 4.0:
                b = self.sound_bearing
                self.aim = [b, 0.0]
                act["move"] = [math.cos(b), math.sin(b)]
            else:
                zc = obs["zone"]["center"]
                d = math.hypot(zc[0] - x, zc[1] - y)
                if d > 15:
                    act["move"] = [(zc[0] - x) / d, (zc[1] - y) / d]
                self.aim = [self.aim[0] + 0.25, 0.0]
            act["aim"] = self.aim
            return act

        fx, fy, fz = foe["pos"]
        vx, vy, vz = foe["vel"]
        dx, dy = fx - x, fy - y
        dist = math.hypot(dx, dy)
        ux, uy = (dx / dist, dy / dist) if dist > 1e-6 else (1.0, 0.0)

        # movement: close to preferred range, strafe perpendicular, flip occasionally
        if now >= self.flip_at:
            self.strafe = -self.strafe if self.rng.random() < self.s["flip"] else self.strafe
            self.flip_at = now + self.rng.uniform(0.8, 2.0)
        px, py = -uy * self.strafe, ux * self.strafe
        want = self.s["range"]
        radial = 0.0
        if dist > want + 15:
            radial = 1.0
        elif dist < want - 15:
            radial = -0.6
        mx, my = ux * radial + px * 0.9, uy * radial + py * 0.9
        n = math.hypot(mx, my) or 1.0
        act["move"] = [mx / n, my / n]
        act["boost"] = dist > want + 40 and me["energy"] > 30

        # gun lead with true velocity (as observed), solving flight time
        gun = me["weapons"][0]
        muzzle = (x, y, z + st["muzzle_height"])
        speed = gun["stats"]["speed"] if gun else 252.0
        tgt = [fx, fy, fz + 4.5]
        lead = self.s["lead"]
        tof = dist / speed
        for _ in range(3):
            tgt = [fx + vx * tof * lead, fy + vy * tof * lead, fz + 4.5 + vz * tof * lead]
            tof = math.dist(muzzle, tgt) / speed
        ay = math.atan2(tgt[1] - muzzle[1], tgt[0] - muzzle[0])
        ap = math.atan2(tgt[2] - muzzle[2], math.hypot(tgt[0] - muzzle[0], tgt[1] - muzzle[1]))
        noise = math.radians(self.s["aim_noise"])
        self.aim = [wrap(ay + self.rng.gauss(0, noise)), ap + self.rng.gauss(0, noise)]
        act["aim"] = self.aim
        aim_err = abs(wrap(me["aim"][0] - ay))
        # angular size of the target: fire when the torso is close to on-target
        ang_tol = math.atan2(st["radius"] * 1.5, max(dist, 1.0))
        if gun and gun["kind"] == "gun":
            if gun.get("ammo", 0) > 0 and gun.get("reload", 0) <= 0 and aim_err < ang_tol + 0.02 and dist < gun["stats"]["range"]:
                act["fire"][0] = True
            elif gun.get("ammo", 0) == 0 and gun.get("reload", 0) <= 0:
                act["reload"] = True
        w1 = me["weapons"][1]
        if w1 and w1["kind"] == "laser":
            ws = w1["stats"]
            charge = w1.get("charge", 0.0)
            if charge > 0:
                # keep charging until full and on target, then release
                full = charge >= ws["charge"] - 0.01
                act["fire"][1] = not (full and aim_err < ang_tol)
            elif w1["cooldown"] <= 0 and dist < ws["range"] * 0.9 and aim_err < 0.3:
                act["fire"][1] = True
        elif w1 and w1["kind"] == "lightning":
            ws = w1["stats"]
            if dist < ws["range"] - 2 and aim_err < math.radians(ws["cone_deg"]) * 0.45 and me["energy"] >= ws["energy"]:
                act["fire"][1] = True
            if dist > 30:
                # lightning wants to close
                act["move"] = [ux * 0.8 + px * 0.5, uy * 0.8 + py * 0.5]
                act["boost"] = me["energy"] > 40
        return act
