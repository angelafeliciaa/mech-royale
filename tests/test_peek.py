"""Synthetic tests for the laser peek-shot cycle (dex_tactics.laser_peek_phase + Bot wiring).

Run: python3 -m unittest tests.test_peek
"""

import copy
import gzip
import json
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dex_bot  # noqa: E402
from dex_tactics import PEEK_POP_FRAC, laser_peek_phase  # noqa: E402

FULL = 0.75
HOLD = 2.0
CD = 1.65
DT = 0.05


def step(prev, since, now, charge=0.0, cooldown=0.0, visible=True, dist=80.0, in_range=True,
         shorted=False, cover_ok=True, skip=False):
    return laser_peek_phase(prev, since, now, charge, FULL, cooldown, visible, dist, in_range, shorted, cover_ok, skip)


class PhaseMachine(unittest.TestCase):
    def test_full_cycle(self):
        ph, since = step("none", -99.0, 10.0)
        self.assertEqual(ph, "hide")
        self.assertEqual(since, 10.0)
        # Reached cover: target no longer visible, charge starts.
        ph, since = step(ph, since, 10.4, visible=False)
        self.assertEqual(ph, "charge")
        ph, since = step(ph, since, 10.8, charge=0.5, visible=False)
        self.assertEqual(ph, "charge")
        # 85% charged -> pop out even while still hidden.
        ph, since = step(ph, since, 11.1, charge=FULL * PEEK_POP_FRAC + 0.01, visible=False)
        self.assertEqual(ph, "pop")
        ph, since = step(ph, since, 11.5, charge=FULL + 0.3, visible=True)
        self.assertEqual(ph, "pop")
        # Released: charge 0, cooldown running -> back to cover.
        ph, since = step(ph, since, 11.6, charge=0.0, cooldown=CD, visible=True)
        self.assertEqual(ph, "cool")
        ph, since = step(ph, since, 12.5, charge=0.0, cooldown=0.7, visible=False)
        self.assertEqual(ph, "cool")
        # Cooldown over while hidden -> charge straight away.
        ph, since = step(ph, since, 13.3, charge=0.0, cooldown=0.0, visible=False)
        self.assertEqual(ph, "charge")

    def test_no_cover_falls_back_to_open_until_shot_is_out(self):
        ph, since = step("none", -99.0, 5.0)
        self.assertEqual(ph, "hide")
        ph, since = step(ph, since, 5.05, cover_ok=False)
        self.assertEqual(ph, "open")
        ph, since = step(ph, since, 5.5, charge=0.3)
        self.assertEqual(ph, "open")
        ph, since = step(ph, since, 5.6, charge=0.3, visible=False)
        self.assertEqual(ph, "open", "open sticks even if the foe ducks")
        ph, since = step(ph, since, 6.0, charge=FULL)
        self.assertEqual(ph, "pop")
        ph, since = step(ph, since, 6.1, charge=0.0, cooldown=CD)
        self.assertEqual(ph, "cool")

    def test_hide_times_out(self):
        ph, since = step("none", -99.0, 5.0)
        ph, since = step(ph, since, 6.4)
        self.assertEqual(ph, "hide")
        ph, since = step(ph, since, 6.6)
        self.assertEqual(ph, "open")

    def test_already_charging_in_the_open_finishes(self):
        ph, _ = step("none", -99.0, 5.0, charge=0.2)
        self.assertEqual(ph, "open")

    def test_guards(self):
        for kw in (dict(shorted=True), dict(dist=20.0), dict(in_range=False), dict(skip=True)):
            ph, _ = step("charge", 4.0, 5.0, charge=0.4, visible=False, **kw)
            self.assertEqual(ph, "none", kw)


def _load_hello_and_obs():
    with gzip.open(os.path.join(ROOT, "kit", "check", "stream.jsonl.gz"), "rt") as f:
        hello = obs = None
        for line in f:
            m = json.loads(line)
            if hello is None and "config" in m:
                hello = m
            elif "self" in m and m.get("mechs") == [] and m["self"]["weapons"][0] is not None:
                obs = m
                break
    hello["map"]["dir"] = os.path.join(ROOT, "kit", "arena")
    return hello, obs


class _Sim:
    """Minimal world: our mech moves along the bot's move vector (half speed while charging),
    an enemy stands still; the laser's charge/cooldown follow the trigger per protocol section 7."""

    def __init__(self, bot, obs, enemy_xy, laser_stats):
        self.bot = bot
        self.obs = obs
        self.enemy_xy = enemy_xy
        self.ls = laser_stats
        self.charge = 0.0
        self.cooldown = 0.0
        self.fired = 0
        self.phases = []
        self.modes = []

    def tick(self, now, tick, events=()):
        obs = copy.deepcopy(self.obs)
        me = obs["self"]
        obs["t"] = now
        obs["tick"] = tick
        obs["events"] = list(events)
        obs["alive"] = 16
        me["weapons"][1] = {
            "kind": "laser", "cooldown": round(self.cooldown, 3), "charge": round(self.charge, 3), "stats": self.ls,
        }
        ex, ey = self.enemy_xy
        ez = self.bot.arena.floor(ex, ey)
        obs["mechs"] = [{
            "id": 9, "kind": "mech", "pos": [ex, ey, ez], "vel": [0.0, 0.0, 0.0],
            "aim": [math.atan2(me["pos"][1] - ey, me["pos"][0] - ex), 0.0],
            "hp": 1400, "max_hp": 1400, "cubes": 0, "weapons": ["gun"], "charging": False,
        }]
        act = self.bot._step(obs)  # let exceptions propagate (Bot.act would swallow them)
        self.phases.append(self.bot.peek_phase)
        self.modes.append(self.bot.mode)
        fire = act.get("fire", [False, False])
        # Laser model.
        if self.cooldown > 0.0:
            self.cooldown = max(0.0, self.cooldown - DT)
            self.charge = 0.0
        elif fire[1]:
            self.charge += DT
            if self.charge >= self.ls["charge"] + self.ls["hold_s"]:
                self.charge = 0.0
                self.cooldown = self.ls["cooldown"]
                self.fired += 1
        elif self.charge > 0.0:
            self.charge = 0.0
            self.cooldown = self.ls["cooldown"]
            self.fired += 1
        # Movement (no collision; candidate positions were already walkable).
        mx, my = act["move"]
        speed = me["stats"]["walk_speed"] * (0.5 if self.charge > 0.0 else 1.0)
        nx, ny = me["pos"][0] + mx * speed * DT, me["pos"][1] + my * speed * DT
        if self.bot.nav.is_free(nx, ny):
            me["pos"][0], me["pos"][1] = nx, ny
            me["pos"][2] = self.bot.arena.floor(nx, ny)
            me["vel"] = [mx * speed, my * speed, 0.0]
        else:
            me["vel"] = [0.0, 0.0, 0.0]
        # Torso slews toward the requested aim at turn_rate.
        ay, ap = act["aim"]
        d = math.remainder(ay - me["aim"][0], math.tau)
        lim = me["stats"]["turn_rate"] * DT
        me["aim"][0] += max(-lim, min(lim, d))
        me["aim"][1] = ap
        self.obs["self"] = me
        return act


class BotCycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hello, cls.obs0 = _load_hello_and_obs()
        cls.laser_stats = dict(cls.hello["config"]["weapons"]["laser"])

    def _bot(self, overrides=None):
        dex_bot.OVERRIDES.clear()
        if overrides:
            dex_bot.OVERRIDES.update(overrides)
        bot = dex_bot.Bot(self.hello, profile="aggro")
        bot.p = dex_bot.squad_params(bot.base_p, "aggro")
        bot.prac_on = True
        return bot

    def _enemy_with_los(self, bot, me_pos, dist):
        x, y, z = me_pos
        muzzle = (x, y, z + bot.cfg["mech"]["muzzle_height"])
        for k in range(32):
            a = k * math.tau / 32.0
            ex, ey = x + math.cos(a) * dist, y + math.sin(a) * dist
            if not bot.nav.is_free(ex, ey):
                continue
            ez = bot.arena.floor(ex, ey)
            if abs(ez - z) < 3.0 and bot.nav.los(muzzle, (ex, ey, ez + 4.5)):
                return ex, ey
        self.skipTest("no visible enemy position found around the stream's spawn")

    def _run(self, bot, seconds, events_at=None):
        obs = copy.deepcopy(self.obs0)
        obs["t"] = 30.0
        me = obs["self"]
        enemy = self._enemy_with_los(bot, me["pos"], 80.0)
        sim = _Sim(bot, obs, enemy, self.laser_stats)
        now, tick = 30.0, 1000
        n = int(seconds / DT)
        for i in range(n):
            ev = events_at.get(i, ()) if events_at else ()
            sim.tick(now, tick, ev)
            now += DT
            tick += 1
        return sim

    def test_cycle_runs_without_errors_and_fires(self):
        bot = self._bot()
        self.assertTrue(bot.p["laser_peek"])
        sim = self._run(bot, 20.0)
        visited = set(sim.phases)
        self.assertIn("hide", visited, visited)
        self.assertTrue({"open", "charge"} & visited, visited)
        self.assertIn("pop", visited, visited)
        self.assertIn("cool", visited, visited)
        self.assertGreaterEqual(sim.fired, 2, (sim.fired, visited))
        # Never released before the pop threshold except via auto-fire guard.
        self.assertTrue(any(m.startswith("fight") for m in sim.modes), set(sim.modes))

    def test_short_blocks_charging(self):
        bot = self._bot()
        # Short lands at tick 20: charge must be 0 and the trigger up for short_s afterwards.
        ev = {20: ({"type": "effect", "effect": "short", "by": 9},)}
        obs = copy.deepcopy(self.obs0)
        obs["t"] = 30.0
        enemy = self._enemy_with_los(bot, obs["self"]["pos"], 80.0)
        sim = _Sim(bot, obs, enemy, self.laser_stats)
        now, tick = 30.0, 1000
        fired_during_short = 0
        for i in range(120):
            act = sim.tick(now, tick, ev.get(i, ()))
            if i == 20:
                sim.charge = 0.0  # the short wiped the charge
            if 21 <= i < 20 + int(2.0 / DT):
                self.assertEqual(bot.peek_phase, "none", i)
                if act["fire"][1]:
                    fired_during_short += 1
            now += DT
            tick += 1
        self.assertEqual(fired_during_short, 0)

    def test_override_disables_peek(self):
        bot = self._bot({"laser_peek": False})
        self.assertFalse(bot.p["laser_peek"])
        sim = self._run(bot, 8.0)
        self.assertEqual(set(sim.phases), {"none"})
        self.assertGreaterEqual(sim.fired, 1)
        dex_bot.OVERRIDES.clear()

    def test_close_target_just_fights(self):
        bot = self._bot()
        obs = copy.deepcopy(self.obs0)
        obs["t"] = 30.0
        enemy = self._enemy_with_los(bot, obs["self"]["pos"], 18.0)
        sim = _Sim(bot, obs, enemy, self.laser_stats)
        now, tick = 30.0, 1000
        checked = 0
        for _ in range(60):
            p = obs["self"]["pos"]
            close = math.hypot(enemy[0] - p[0], enemy[1] - p[1]) < 25.0
            sim.tick(now, tick)
            if close:
                self.assertEqual(bot.peek_phase, "none")
                checked += 1
            now += DT
            tick += 1
        self.assertGreater(checked, 5)


if __name__ == "__main__":
    unittest.main()
