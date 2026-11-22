#!/usr/bin/env python3
"""Run N local duels between two bots and report win rate / TTK / damage / hit rates.

    python3 sim/duel.py --a working --b house --n 50 --loadout gun+laser --seed 1
    python3 sim/duel.py --a origin/feat/peek --b origin/main --n 30 --loadout gun
    python3 sim/duel.py --a working --b house --skill hard --scenario urban --n 40

--a / --b: `working` (checked-out root), `house` (scripted bot, see --skill), a git ref, or a dir.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from engine import World  # noqa: E402
from house_bot import HouseBot  # noqa: E402
from loader import load_bot_class  # noqa: E402
from arena_map import ArenaMap  # noqa: E402

LOADOUTS = {
    "gun": ["gun", None],
    "gun+laser": ["gun", "laser"],
    "gun+lightning": ["gun", "lightning"],
    "laser": [None, "laser"],
}


def load_hello():
    with gzip.open(ROOT / "kit/check/stream.jsonl.gz", "rt") as f:
        hello = json.loads(f.readline())
    hello["map"] = {"dir": str(ROOT / "kit/arena"), "files": ["v2/world.json", "v2/terrain.i16"]}
    return hello


_ARENA = None


def arena():
    global _ARENA
    if _ARENA is None:
        _ARENA = ArenaMap.load(load_hello()["map"])
    return _ARENA


def open_spot(world: World, rng, center, radius):
    """Random point on walkable ground (not inside/on a solid), within radius of center."""
    cfg = world.cfg["mech"]
    for _ in range(500):
        a = rng.uniform(0, math.tau)
        r = radius * math.sqrt(rng.random())
        x, y = center[0] + r * math.cos(a), center[1] + r * math.sin(a)
        z = world.arena.floor(x, y)
        if world.blocked(x, y, z, cfg["radius"] + 1.0, cfg["height"]):
            continue
        # not on a roof: reject if any solid's top is near z and touches
        if any(s.touches(x, y, cfg["radius"]) for s in world.arena.nearby((x, y), (x, y), cfg["radius"])):
            continue
        return [x, y, z]
    raise RuntimeError("no spawn found")


def pick_spawns(world: World, rng, scenario):
    """Returns (posA, posB). open: mutual LOS at spawn; urban: no LOS; mixed: either."""
    zc = world.cfg["zone"]["start_center"]
    lo, hi = {"open": (90, 160), "urban": (50, 120), "mixed": (60, 160), "close": (35, 70)}[scenario]
    for _ in range(300):
        a = open_spot(world, rng, zc, 330)
        ang = rng.uniform(0, math.tau)
        d = rng.uniform(lo, hi)
        bx, by = a[0] + d * math.cos(ang), a[1] + d * math.sin(ang)
        try:
            b = open_spot(world, rng, (bx, by), 10)
        except RuntimeError:
            continue
        dd = math.dist(a[:2], b[:2])
        if not (lo <= dd <= hi + 12):
            continue
        eye = world.cfg["mech"]["eye_height"]
        los = world.los((a[0], a[1], a[2] + eye), (b[0], b[1], b[2] + 4.5))
        if scenario == "open" and not los:
            continue
        if scenario == "urban" and los:
            continue
        return a, b
    raise RuntimeError("no spawn pair found")


def make_controller(spec, hello, skill, seed, world, slot):
    h = dict(hello)
    h["slot"] = slot
    h["seed"] = seed
    if spec == "house":
        bot = HouseBot(h, skill=skill, seed=seed)
    elif spec == "dummy":  # stands still, faces the foe: a target for accuracy checks
        def dummy_act(obs):
            a = {"tick": obs["tick"], "move": [0.0, 0.0]}
            if obs["mechs"]:
                p, q = obs["mechs"][0]["pos"], obs["self"]["pos"]
                a["aim"] = [math.atan2(p[1] - q[1], p[0] - q[0]), 0.0]
            return a
        return dummy_act
    else:
        cls = _BOT_CACHE.get(spec)
        if cls is None:
            cls = _BOT_CACHE[spec] = load_bot_class(spec)
        try:
            bot = cls(h, arena=world.arena)
        except TypeError:
            bot = cls(h)
    return bot.act


_BOT_CACHE = {}


def run_duel(job):
    (a, b, skill_a, skill_b, loadout_a, loadout_b, scenario, seed, max_t, alive, facing, trace_path, swap) = job
    if swap:  # mirror: same seed/spawns, sides exchanged; results are mapped back so index 0 is always `a`
        a, b, skill_a, skill_b, loadout_a, loadout_b = b, a, skill_b, skill_a, loadout_b, loadout_a
    hello = load_hello()
    rng = random.Random(seed)
    world = World(hello["config"], arena(), seed=seed, max_t=max_t, alive=alive)
    pa, pb = pick_spawns(world, rng, scenario)
    ya = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
    yb = math.atan2(pa[1] - pb[1], pa[0] - pb[0])
    if facing == "toward":
        ya += rng.uniform(-0.5, 0.5)
        yb += rng.uniform(-0.5, 0.5)
    else:
        ya, yb = rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi)
    ma = world.add_mech(pa, ya, LOADOUTS[loadout_a], "A")
    mb = world.add_mech(pb, yb, LOADOUTS[loadout_b], "B")
    t0 = time.time()
    ctls = [make_controller(a, hello, skill_a, seed * 2 + 1, world, 0),
            make_controller(b, hello, skill_b, seed * 2 + 2, world, 1)]
    trace = [] if trace_path else None
    res = world.run(ctls, trace=trace)
    if swap:
        res["mechs"] = [dict(res["mechs"][1], id=0, name="A"), dict(res["mechs"][0], id=1, name="B")]
        res["winner"] = None if res["winner"] is None else 1 - res["winner"]
    res["seed"] = seed
    res["swap"] = swap
    res["wall"] = round(time.time() - t0, 2)
    res["spawn_dist"] = round(math.dist(pa[:2], pb[:2]), 1)
    if trace_path:
        Path(trace_path).write_text(json.dumps({"spawns": [pa, pb], "trace": trace, "result": res}))
    return res


def summarize(results, label_a, label_b):
    n = len(results)
    wins_a = sum(1 for r in results if r["winner"] == 0)
    wins_b = sum(1 for r in results if r["winner"] == 1)
    draws = n - wins_a - wins_b
    no_contact = sum(1 for r in results if all(m["damage_dealt"] == 0 for m in r["mechs"]))
    ttk = [r["t"] for r in results if r["winner"] is not None]

    def side(i):
        ms = [r["mechs"][i] for r in results]
        hp_left = statistics.mean(m["hp"] for m in ms)
        dealt = statistics.mean(m["damage_dealt"] for m in ms)
        taken = statistics.mean(m["damage_taken"] for m in ms)
        rates = {}
        for k in ("gun", "laser", "lightning"):
            shots = sum(m["shots"][k] for m in ms)
            hits = sum(m["hits"][k] for m in ms)
            if shots:
                rates[k] = (hits, shots, hits / shots)
        rear = sum(m["rear_hits"] for m in ms)
        return hp_left, dealt, taken, rates, rear

    lines = []
    se = math.sqrt(max(1e-9, (wins_a / n) * (1 - wins_a / n) / n)) if n else 0
    lines.append(f"{label_a} vs {label_b}: n={n}  A wins {wins_a} ({100*wins_a/n:.0f}% +-{100*se:.0f})  B wins {wins_b}  draws {draws} (no contact {no_contact})"
                 f"  mean TTK {statistics.mean(ttk):.1f}s" if ttk else f"{label_a} vs {label_b}: n={n} no decisive duels")
    for i, lab in ((0, "A"), (1, "B")):
        hp, dealt, taken, rates, rear = side(i)
        rs = "  ".join(f"{k} {h}/{s} ({100*p:.0f}%)" for k, (h, s, p) in rates.items())
        lines.append(f"  {lab}: hp_left {hp:.0f}  dealt {dealt:.0f}  taken {taken:.0f}  rear_hits {rear}  hit rates: {rs}")
    wall = sum(r["wall"] for r in results)
    lines.append(f"  sim wall {wall:.1f}s total, {wall/n:.2f}s/duel")
    errs = [l for r in results for l in r["log"]]
    if errs:
        lines.append(f"  bot exceptions: {len(errs)} (first: {errs[0]})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="working")
    ap.add_argument("--b", default="house")
    ap.add_argument("--n", type=int, default=20, help="duels (mirrored pairs share spawns: n/2 distinct seeds)")
    ap.add_argument("--loadout", default="gun", choices=sorted(LOADOUTS))
    ap.add_argument("--loadout-b", default=None, choices=sorted(LOADOUTS))
    ap.add_argument("--skill", default="mid", choices=("easy", "mid", "hard", "expert"), help="house bot skill (B, or A if A is house)")
    ap.add_argument("--skill-a", default=None, choices=("easy", "mid", "hard", "expert"))
    ap.add_argument("--scenario", default="mixed", choices=("open", "urban", "mixed", "close"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-t", type=float, default=90.0)
    ap.add_argument("--alive", type=int, default=16, help="`alive` shown to bots (16 mimics early match)")
    ap.add_argument("--facing", default="toward", choices=("toward", "random"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-mirror", action="store_true", help="don't play each seed from both sides (halves cost, adds side bias)")
    ap.add_argument("--json", default=None, help="write per-duel results here")
    ap.add_argument("--trace", default=None, help="(n=1) write a position/hp trace json here")
    args = ap.parse_args(argv)

    # resolve refs once in the parent so workers share the exported dir
    specs = {}
    for s in (args.a, args.b):
        if s not in ("house", "dummy") and s not in specs:
            cls = load_bot_class(s)
            specs[s] = cls._sim_source
    a_spec, b_spec = specs.get(args.a, args.a), specs.get(args.b, args.b)
    jobs = [(a_spec, b_spec, args.skill_a or args.skill, args.skill, args.loadout, args.loadout_b or args.loadout,
             args.scenario, args.seed * 100000 + i // (1 if args.no_mirror else 2), args.max_t, args.alive, args.facing,
             args.trace if args.n == 1 else None, (not args.no_mirror) and i % 2 == 1) for i in range(args.n)]
    t0 = time.time()
    if args.jobs > 1 and args.n > 1:
        with mp.Pool(args.jobs) as pool:
            results = pool.map(run_duel, jobs, chunksize=1)
    else:
        results = [run_duel(j) for j in jobs]
    la = f"{args.a}[{args.loadout}]" + (f"({args.skill_a or args.skill})" if args.a == "house" else "")
    lb = f"{args.b}[{args.loadout_b or args.loadout}]" + (f"({args.skill})" if args.b == "house" else "")
    print(summarize(results, la, lb))
    print(f"  scenario={args.scenario} seed={args.seed} elapsed {time.time()-t0:.1f}s")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
