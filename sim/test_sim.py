#!/usr/bin/env python3
"""Sanity checks for the duel engine. Run: python3 sim/test_sim.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import duel  # noqa: E402
from engine import World  # noqa: E402

HELLO = duel.load_hello()
CFG = HELLO["config"]


def world(seed=0):
    return World(CFG, duel.arena(), seed=seed, max_t=60.0)


def flat_spot(w, x, y):
    return [x, y, w.arena.floor(x, y)]


def open_pair(dist, w):
    """Two spots on flat open ground with LOS (search near the stadium's open field)."""
    import random
    rng = random.Random(7)
    for _ in range(2000):
        a = duel.open_spot(w, rng, (41.3, -72.8), 300)
        ang = rng.uniform(0, math.tau)
        b = [a[0] + dist * math.cos(ang), a[1] + dist * math.sin(ang)]
        b = [b[0], b[1], w.arena.floor(b[0], b[1])]
        if abs(a[2] - b[2]) > 0.3:
            continue
        if w.blocked(b[0], b[1], b[2], 3.5, 9):
            continue
        if any(s.touches(b[0], b[1], 2.5) for s in w.arena.nearby(b, b, 2.5)):
            continue
        if w.los((a[0], a[1], a[2] + 8), (b[0], b[1], b[2] + 4.5)) and w.los((a[0], a[1], a[2] + 2), (b[0], b[1], b[2] + 2)):
            return a, b
    raise RuntimeError("no flat open pair")


def step(w, ticks, actions):
    for _ in range(ticks):
        for m, a in zip(w.mechs, actions):
            w.apply_action(m, a)
        w.sim_tick()


def aim_at(w, shooter, target):
    mz = [shooter.pos[0], shooter.pos[1], shooter.pos[2] + CFG["mech"]["muzzle_height"]]
    c = w.center(target)
    yaw = math.atan2(c[1] - mz[1], c[0] - mz[0])
    pitch = math.atan2(c[2] - mz[2], math.dist(mz[:2], c[:2]))
    shooter.aim = [yaw, pitch]
    shooter.cmd_aim = [yaw, pitch]
    return [yaw, pitch]


def test_gun_hits_stationary():
    w = world()
    a, b = open_pair(100, w)
    s = w.add_mech(a, 0, ["gun", None], "S")
    t = w.add_mech(b, 0, ["gun", None], "T")
    aim = aim_at(w, s, t)
    aim_at(w, t, s)
    step(w, 80, [{"fire": [True, False], "aim": aim}, {}])  # 4 s: one mag + part of reload
    hits, shots = s.stats["hits"]["gun"], s.stats["shots"]["gun"]
    assert shots == 12, shots
    assert hits >= 10, (hits, shots)
    assert t.stats["damage_taken"] == hits * 45, t.stats
    assert not any(e.get("rear") for e in t.events), "front hit should not be rear"
    assert s.weapons[0]["reload"] > 0 or s.weapons[0]["ammo"] == 12
    print("gun ok", hits, "/", shots)


def test_rear_multiplier():
    w = world()
    a, b = open_pair(60, w)
    s = w.add_mech(a, 0, ["gun", None], "S")
    t = w.add_mech(b, 0, ["gun", None], "T")
    aim = aim_at(w, s, t)
    t.aim = [aim[0], 0.0]  # target faces away from shooter
    t.cmd_aim = list(t.aim)
    step(w, 20, [{"fire": [True, False], "aim": aim}, {}])
    hits = s.stats["hits"]["gun"]
    assert hits >= 3, hits
    assert abs(t.stats["damage_taken"] - hits * 67.5) < 1e-6, t.stats
    assert s.stats["rear_hits"] == hits
    print("rear ok", t.stats["damage_taken"])


def test_laser_timing():
    w = world()
    a, b = open_pair(150, w)
    s = w.add_mech(a, 0, ["gun", "laser"], "S")
    t = w.add_mech(b, 0, ["gun", None], "T")
    aim = aim_at(w, s, t)
    aim_at(w, t, s)
    # hold 0.4 s then release
    step(w, 8, [{"fire": [False, True], "aim": aim}, {}])
    assert abs(s.weapons[1]["charge"] - 0.4) < 1e-6, s.weapons[1]
    step(w, 1, [{"fire": [False, False], "aim": aim}, {}])
    assert s.stats["shots"]["laser"] == 1
    expected = 315 * (0.35 + 0.65 * 0.45 / 0.75)  # charge keeps building during the release tick
    assert abs(t.stats["damage_taken"] - expected) < 0.05, (t.stats["damage_taken"], expected)
    assert abs(s.weapons[1]["cooldown"] - 1.65) < 1e-6
    # can't recharge during cooldown
    step(w, 10, [{"fire": [False, True], "aim": aim}, {}])
    assert s.weapons[1]["charge"] == 0.0
    # full charge + hold auto-fires at 0.75 + 2.0 s
    step(w, 24, [{"fire": [False, False], "aim": aim}, {}])  # finish cooldown
    before = t.stats["damage_taken"]
    step(w, 54, [{"fire": [False, True], "aim": aim}, {}])  # 2.7 s
    assert s.stats["shots"]["laser"] == 1, "should not auto-fire before 2.75 s"
    step(w, 2, [{"fire": [False, True], "aim": aim}, {}])
    assert s.stats["shots"]["laser"] == 2, s.weapons[1]
    assert abs(t.stats["damage_taken"] - before - 315) < 1e-6
    # quick tap: released before min_charge still fires once min_charge reached
    step(w, 40, [{"fire": [False, False], "aim": aim}, {}])
    step(w, 1, [{"fire": [False, True], "aim": aim}, {}])
    step(w, 1, [{"fire": [False, False], "aim": aim}, {}])
    assert s.stats["shots"]["laser"] == 2
    step(w, 2, [{"fire": [False, False], "aim": aim}, {}])
    assert s.stats["shots"]["laser"] == 3, s.weapons[1]
    # half speed while charging
    step(w, 40, [{"fire": [False, False], "aim": aim}, {}])
    step(w, 40, [{"move": [1, 0], "fire": [False, True], "aim": aim}, {}])
    assert math.hypot(*s.vel[:2]) < 0.5 * 11.7 + 0.01, s.vel
    print("laser ok")


def test_lightning():
    w = world()
    a, b = open_pair(30, w)
    s = w.add_mech(a, 0, ["gun", "lightning"], "S")
    t = w.add_mech(b, 0, ["gun", None], "T")
    aim = aim_at(w, s, t)
    aim_at(w, t, s)
    e0 = s.energy
    step(w, 1, [{"fire": [False, True], "aim": aim}, {}])
    assert s.stats["hits"]["lightning"] == 1 and t.stats["damage_taken"] == 60
    assert abs(s.energy - (e0 - 3)) < 1e-6
    step(w, 4, [{"fire": [False, True], "aim": aim}, {}])
    assert s.stats["shots"]["lightning"] == 1, "cooldown 0.25 s"
    step(w, 1, [{"fire": [False, True], "aim": aim}, {}])
    assert s.stats["shots"]["lightning"] == 2
    # out of cone: fires but misses
    s.aim = [aim[0] + math.radians(40), 0]
    s.cmd_aim = list(s.aim)
    step(w, 5, [{"fire": [False, True], "aim": s.aim}, {}])
    assert s.stats["shots"]["lightning"] == 3 and s.stats["hits"]["lightning"] == 2
    print("lightning ok")


def test_cover_blocks():
    import random
    w = world()
    rng = random.Random(3)
    # find a pair with no LOS at eye level
    for _ in range(500):
        a, b = duel.pick_spawns(w, rng, "urban")
        if math.dist(a[:2], b[:2]) < 90:
            break
    s = w.add_mech(a, 0, ["gun", "laser"], "S")
    t = w.add_mech(b, 0, ["gun", None], "T")
    aim = aim_at(w, s, t)
    step(w, 60, [{"fire": [True, True], "aim": aim}, {}])
    assert s.stats["shots"]["gun"] >= 10
    assert t.stats["damage_taken"] == 0, t.stats
    assert not w.can_see(s, t)
    print("cover ok")


def test_movement_energy():
    w = world()
    a, b = open_pair(120, w)
    m = w.add_mech(a, 0, ["gun", None], "M")
    w.add_mech(b, 0, ["gun", None], "T")
    d = [b[0] - a[0], b[1] - a[1]]
    n = math.hypot(*d)
    d = [d[0] / n, d[1] / n]
    step(w, 2, [{"move": d}, {}])
    assert math.hypot(*m.vel[:2]) < 11.7 - 1, "accel takes time"
    step(w, 20, [{"move": d}, {}])
    assert abs(math.hypot(*m.vel[:2]) - 11.7) < 0.01
    e0 = m.energy
    step(w, 20, [{"move": d, "boost": True}, {}])
    assert abs(math.hypot(*m.vel[:2]) - 19.5) < 0.01
    assert abs(m.energy - (e0 - 17.0)) < 1e-6, m.energy
    step(w, 11, [{"move": [0, 0]}, {}])  # 0.55 s: still in regen delay
    assert abs(m.energy - (e0 - 17.0)) < 1e-6
    step(w, 20, [{"move": [0, 0]}, {}])
    assert m.energy > e0 - 17.0 + 10
    # jump
    step(w, 1, [{"jump": True}, {}])
    assert not m.grounded and m.vel[2] > 0
    step(w, 40, [{}, {}])
    assert m.grounded
    print("movement ok")


def test_vision_rules():
    w = world()
    a, b = open_pair(100, w)
    v = w.add_mech(a, 0, ["gun", None], "V")
    o = w.add_mech(b, 0, ["gun", None], "O")
    yaw = aim_at(w, v, o)[0]
    assert w.can_see(v, o)
    v.aim = [yaw + math.pi, 0]
    assert not w.can_see(v, o), "100 m behind: not visible"
    v.aim = [yaw + math.radians(55), 0]
    assert w.can_see(v, o), "inside the 120 deg wedge"
    v.aim = [yaw + math.radians(65), 0]
    assert not w.can_see(v, o), "outside the wedge"
    # memory: seen once, then turned away -> still listed for 5 s
    v.aim = [yaw, 0]
    obs = w.build_obs(v)
    assert len(obs["mechs"]) == 1
    v.aim = [yaw + math.pi, 0]
    v.cmd_aim = list(v.aim)
    step(w, 90, [{"aim": v.aim}, {}])
    assert len(w.build_obs(v)["mechs"]) == 1
    step(w, 20, [{"aim": v.aim}, {}])
    assert len(w.build_obs(v)["mechs"]) == 0
    print("vision ok")


def test_obs_shape():
    w = world()
    a, b = open_pair(80, w)
    m = w.add_mech(a, 0, ["gun", "laser"], "M")
    w.add_mech(b, 0, ["gun", None], "T")
    obs = w.build_obs(m)
    for k in ("type", "tick", "t", "alive", "self", "mechs", "projectiles", "crates", "cubes", "pickups",
              "sounds", "beams", "events", "zone", "offer", "drops"):
        assert k in obs, k
    s = obs["self"]
    for k in ("id", "pos", "vel", "aim", "hp", "max_hp", "energy", "max_energy", "grounded", "cubes",
              "upgrades", "weapons", "in_zone", "stats"):
        assert k in s, k
    assert s["weapons"][0]["kind"] == "gun" and "ammo" in s["weapons"][0] and "reload" in s["weapons"][0]
    assert s["weapons"][1]["kind"] == "laser" and "charge" in s["weapons"][1]
    for k in ("center", "radius", "next_center", "next_radius", "shrink_start", "shrink_end", "dps"):
        assert k in obs["zone"], k
    print("obs shape ok")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ALL OK")
