#!/usr/bin/env python3
"""Practice-format matches: our Bot vs 15 house bots acting as one squad, full zone schedule, 480 s.

    python3 sim/practice.py --a working --n 96 --seed 1
    python3 sim/practice.py --a origin/main --n 96 --seed 1 --hunters-base 1 --hunters-per-kill 1 --all-at 300

Score per match = 16 - place; place = 1 + mechs still alive when we die (survivors are ranked by HP).
The squad: shared vision (if any house bot sees us, all know where we are for memory_s), never fights
itself, hunts us with `hunters` bots (nearest first) while the rest hold posts inside the next zone
circle and fight only if we come within --engage-range. --zone-deaths house bots idle at a random
phase and die to the zone (built-in bots do lose a few that way).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import duel  # noqa: E402
from engine import World, wrap  # noqa: E402
from house_bot import HouseBot  # noqa: E402

TIERS = ["easy"] * 4 + ["mid"] * 4 + ["hard"] * 4 + ["expert"] * 3
HOUSE_LOADOUTS = ["gun", "gun+laser", "gun+lightning"]


def spawn_points(world, rng, n, min_sep, frac):
    zc, zr = world.cfg["zone"]["start_center"], world.cfg["zone"]["start_radius"] * frac
    pts = []
    tries = 0
    while len(pts) < n and tries < 5000:
        tries += 1
        p = duel.open_spot(world, rng, zc, zr)
        if all(math.dist(p[:2], q[:2]) >= min_sep for q in pts):
            pts.append(p)
    if len(pts) < n:
        raise RuntimeError("could not place spawns")
    return pts


class Squad:
    def __init__(self, world, us, bots, hello, opts, rng):
        self.w = world
        self.us = us
        self.bots = bots  # list of (mech, HouseBot, tier)
        self.o = opts
        self.rng = rng
        self.known_pos = None
        self.known_vel = [0.0, 0.0, 0.0]
        self.known_t = -99.0
        self.posts = {}
        self.phase_seen = -2
        doomed = rng.sample([m.id for m, _, _ in bots], min(len(bots), opts["zone_deaths"]))
        self.doom = {mid: rng.randint(0, 3) for mid in doomed}  # phase index at which that bot stops moving

    def lite_obs(self, m, foe_known, sounds_ok=True):
        z = self.w.zone
        weapons = []
        for w in m.weapons:
            if w is None:
                weapons.append(None)
                continue
            o = {"kind": w["kind"], "cooldown": w["cooldown"], "stats": w["stats"]}
            if w["kind"] == "gun":
                o["ammo"], o["reload"] = w["ammo"], w["reload"]
            elif w["kind"] == "laser":
                o["charge"] = w["charge"]
            weapons.append(o)
        mechs = []
        if foe_known:
            u = self.us
            mechs.append({"id": u.id, "kind": "mech", "pos": self.known_pos, "vel": self.known_vel, "aim": u.aim,
                          "hp": u.hp, "max_hp": u.max_hp, "cubes": u.cubes,
                          "weapons": [w["kind"] for w in u.weapons if w], "charging": u.charging})
        obs = {"tick": self.w.tick, "t": self.w.t, "self": {"pos": m.pos, "vel": m.vel, "aim": m.aim, "energy": m.energy,
                                                             "stats": self.w.cfg["mech"], "weapons": weapons, "hp": m.hp},
               "mechs": mechs, "events": m.events, "sounds": m.sounds if sounds_ok else [],
               "zone": {"center": z["center"], "radius": z["radius"]}}
        m.events, m.sounds, m.beams = [], [], []
        return obs

    def post_for(self, m):
        """Hold position if it's inside the next circle (with margin); else the nearest point inside it."""
        z = self.w.zone
        key = (m.id, self.w.phase_i)
        p = self.posts.get(key)
        if p is None:
            nc, nr = z["next_center"], z["next_radius"] * self.o["post_frac"]
            d = math.dist(m.pos[:2], nc)
            if d <= nr:
                p = [m.pos[0], m.pos[1]]
            else:
                a = math.atan2(m.pos[1] - nc[1], m.pos[0] - nc[0]) + self.rng.uniform(-0.3, 0.3)
                r = nr * self.rng.uniform(0.6, 1.0)
                p = [nc[0] + r * math.cos(a), nc[1] + r * math.sin(a)]
            self.posts[key] = p
        return p

    def decide(self, kills):
        w = self.w
        us = self.us
        t = w.t
        # shared vision
        seen = False
        for m, _, _ in self.bots:
            if m.alive and math.dist(m.pos[:2], us.pos[:2]) <= w.cfg["vision"]["range"] and w.can_see(m, us):
                seen = True
                break
        if not seen:
            for m, _, _ in self.bots:
                if any(e["type"] == "damaged" and e.get("by") == us.id for e in m.events):
                    seen = True
                    break
        if seen:
            self.known_pos, self.known_vel, self.known_t = list(us.pos), list(us.vel), t
        known = t - self.known_t <= w.cfg["vision"]["memory_s"]
        stale = self.known_pos is not None and t - self.known_t <= self.o["stale_s"]
        # hunters: nearest-first to our (last known) position
        alive = [b for b in self.bots if b[0].alive]
        k = self.o["hunters_base"] + self.o["hunters_per_kill"] * kills
        if t >= self.o["all_at"] or w.zone["radius"] <= self.o["all_radius"]:
            k = len(alive)
        ref = self.known_pos or w.zone["next_center"]
        alive.sort(key=lambda b: math.dist(b[0].pos[:2], ref[:2]))
        hunters = {b[0].id for b in alive[:max(0, k)]}
        z = w.zone
        for m, bot, tier in alive:
            dist = math.dist(m.pos[:2], us.pos[:2])
            if m.id in self.doom and w.phase_i >= self.doom[m.id]:
                w.apply_action(m, {"move": [0, 0], "aim": m.aim})
                continue
            is_hunter = m.id in hunters
            engaged = (known or stale) and dist <= self.o["engage_range"]
            obs = self.lite_obs(m, (known or stale) and (is_hunter or engaged))
            act = bot.act(obs)
            # zone discipline: get inside the next circle before it closes
            nc, nr = z["next_center"], z["next_radius"]
            d_next = math.dist(m.pos[:2], nc)
            margin = 12.0 if is_hunter or engaged else 0.0
            shrinking = t >= z["shrink_start"] - 15.0
            if (shrinking and d_next > max(0.0, nr - margin)) or math.dist(m.pos[:2], z["center"]) > z["radius"] - 3:
                tgt = nc if nr > 0 else z["center"]
                dd = math.dist(m.pos[:2], tgt) or 1.0
                act["move"] = [(tgt[0] - m.pos[0]) / dd, (tgt[1] - m.pos[1]) / dd]
                act["boost"] = m.energy > 25
            elif not is_hunter and not engaged:
                p = self.post_for(m)
                dd = math.dist(m.pos[:2], p)
                act["move"] = [(p[0] - m.pos[0]) / dd, (p[1] - m.pos[1]) / dd] if dd > 6 else [0.0, 0.0]
                act["fire"] = [False, False]
                act["boost"] = False
                if not known:
                    act["aim"] = [m.aim[0] + 0.2, 0.0]
            w.apply_action(m, act)


def run_match(job):
    spec, seed, opts, our_loadout = job
    hello = duel.load_hello()
    cfg = hello["config"]
    rng = random.Random(seed)
    world = World(cfg, duel.arena(), seed=seed, max_t=cfg["match"]["duration_s"], zone_shrink=True)
    pts = spawn_points(world, rng, 16, cfg["match"]["spawn_min_separation"], cfg["match"]["spawn_zone_fraction"])
    kinds = [k for k in ("laser", "lightning") for _ in range(opts["pickups"] // 2)]
    for k in kinds:
        world.add_pickup(k, duel.open_spot(world, rng, cfg["zone"]["start_center"], cfg["zone"]["start_radius"] * 0.85))
    us = world.add_mech(pts[0], rng.uniform(-math.pi, math.pi), duel.LOADOUTS[our_loadout], "us", team=0)
    tiers = list(TIERS)
    rng.shuffle(tiers)
    bots = []
    h = dict(hello)
    for i, (p, tier) in enumerate(zip(pts[1:], tiers), start=1):
        lo = duel.LOADOUTS[rng.choice(HOUSE_LOADOUTS)]
        m = world.add_mech(p, rng.uniform(-math.pi, math.pi), lo, f"house-{tier}", team=1)
        bots.append((m, HouseBot(h, skill=tier, seed=seed * 100 + i), tier))
    ctl = duel.make_controller(spec, hello, "mid", seed, world, 0)
    squad = Squad(world, us, bots, hello, opts, rng)
    kills = 0
    t0 = time.time()
    every = cfg["match"]["decide_every"]
    hp_trace = []
    while world.t < world.max_t and us.alive:
        if world.tick % every == 0:
            obs = world.build_obs(us)
            kills = sum(1 for m, _, _ in bots if not m.alive and m.killed_by and m.killed_by[0] == us.id)
            try:
                act = ctl(obs)
            except Exception as exc:
                act = {}
                world.log.append(f"t={world.t:.1f} bot raised {exc!r}")
            world.apply_action(us, act)
            squad.decide(kills)
            if world.tick % 200 == 0:
                hp_trace.append((round(world.t), round(us.hp)))
        for _ in range(every):
            world.sim_tick()
            if not us.alive:
                break
    others_alive = [m for m, _, _ in bots if m.alive]
    if us.alive:
        place = 1 + sum(1 for m in others_alive if m.hp > us.hp)
        cause = "survived"
    else:
        place = 1 + len(others_alive)
        by, weapon = us.killed_by
        cause = "zone" if by is None else f"{weapon}:{world.mechs[by].name.split('-')[1]}"
    kills = sum(1 for m, _, _ in bots if not m.alive and m.killed_by and m.killed_by[0] == us.id)
    attackers = sum(1 for tt in us.hit_by.values() if world.t - tt < 10.0)
    zone_deaths = sum(1 for m, _, _ in bots if not m.alive and m.killed_by and m.killed_by[0] is None)
    return {"seed": seed, "points": 16 - place, "place": place, "kills": kills, "t": round(world.t, 1),
            "cause": cause, "zone_deaths": zone_deaths, "attackers": attackers, "dealt": round(us.stats["damage_dealt"]),
            "taken": round(us.stats["damage_taken"]), "cubes": us.cubes,
            "weapons": [w["kind"] for w in us.weapons if w], "hp_trace": hp_trace,
            "wall": round(time.time() - t0, 1), "log": world.log[:3]}


def summarize(results, label):
    n = len(results)
    pts = [r["points"] for r in results]
    mean = statistics.mean(pts)
    se = statistics.pstdev(pts) / math.sqrt(n)
    lines = [f"{label}: n={n}  points {mean:.2f} +- {se:.2f}  kills {statistics.mean(r['kills'] for r in results):.2f}"
             f"  survival {statistics.mean(r['t'] for r in results):.0f}s  dealt {statistics.mean(r['dealt'] for r in results):.0f}"
             f"  house zone deaths {statistics.mean(r['zone_deaths'] for r in results):.1f}"
             f"  attackers@death {statistics.mean(r['attackers'] for r in results):.1f}"
             f"  wall {statistics.mean(r['wall'] for r in results):.1f}s/match"]
    causes = Counter(r["cause"] for r in results)
    lines.append("  death causes: " + ", ".join(f"{k} {v}" for k, v in causes.most_common()))
    loot = Counter(k for r in results for k in r["weapons"] if k != "gun")
    lines.append("  our loot at end: " + ", ".join(f"{k} {v}" for k, v in loot.most_common()))
    places = Counter(r["place"] for r in results)
    lines.append("  places: " + " ".join(f"{p}:{places[p]}" for p in sorted(places)))
    errs = [l for r in results for l in r["log"]]
    if errs:
        lines.append(f"  bot exceptions: {len(errs)} (first: {errs[0]})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="working")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--our-loadout", default="gun", choices=sorted(duel.LOADOUTS))
    ap.add_argument("--hunters-base", type=int, default=0)
    ap.add_argument("--hunters-per-kill", type=int, default=1)
    ap.add_argument("--all-at", type=float, default=380.0, help="time after which every house bot hunts")
    ap.add_argument("--all-radius", type=float, default=70.0, help="zone radius at/below which every house bot hunts")
    ap.add_argument("--engage-range", type=float, default=70.0, help="non-hunters fight if we come this close")
    ap.add_argument("--stale-s", type=float, default=20.0, help="hunters keep walking to our last known position this long")
    ap.add_argument("--post-frac", type=float, default=0.85, help="non-hunters post within this fraction of the next circle")
    ap.add_argument("--zone-deaths", type=int, default=4, help="house bots that idle and die to the zone")
    ap.add_argument("--pickups", type=int, default=27, help="floor weapon pickups (half laser, half lightning; no mortars)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    opts = {"hunters_base": args.hunters_base, "hunters_per_kill": args.hunters_per_kill, "all_at": args.all_at,
            "all_radius": args.all_radius, "engage_range": args.engage_range, "stale_s": args.stale_s,
            "post_frac": args.post_frac, "zone_deaths": args.zone_deaths, "pickups": args.pickups}
    spec = args.a if args.a in ("house", "dummy") else duel.load_bot_class(args.a)._sim_source
    jobs = [(spec, args.seed * 100000 + i, opts, args.our_loadout) for i in range(args.n)]
    t0 = time.time()
    if args.jobs > 1 and args.n > 1:
        with mp.Pool(args.jobs) as pool:
            results = pool.map(run_match, jobs, chunksize=1)
    else:
        results = [run_match(j) for j in jobs]
    print(summarize(results, f"{args.a}[{args.our_loadout}]"))
    print(f"  opts {opts}  seed={args.seed}  elapsed {time.time() - t0:.0f}s")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
