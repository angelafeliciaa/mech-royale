"""Offline gun hit-rate harness: strafing target, 2-tick lockstep latency, spread.

Usage: python3 tools/aim_sim.py
Compares full constant-velocity lead against the damped lead in dex_combat.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dex_combat as dc  # noqa: E402

SPEED, WALK, ACCEL, RADIUS, BULLET_R, SPREAD = 252.0, 11.7, 58.5, 2.5, 0.3, math.radians(0.6)
MUZZLE_H, HEIGHT = 7.0, 9.0


def simulate(dist, flip_rate, decay, acc_gain, shots=4000, rng=None):
    rng = rng or random.Random(1)
    dc.LEAD_DECAY, dc.ACC_GAIN = decay, acc_gain
    hits = 0
    for _ in range(shots):
        # Target strafes along y at distance `dist` along x; commanded velocity flips as a Poisson process.
        cmd = WALK * rng.choice((-1.0, 1.0))
        vy = cmd * rng.uniform(-1.0, 1.0)
        y = 0.0
        hist = []
        dt = 0.05
        # Warm-up so the state is stationary, recording the last two decision-tick velocities.
        for k in range(40):
            if rng.random() < flip_rate * dt:
                cmd = -cmd
            dv = max(-ACCEL * dt, min(ACCEL * dt, cmd - vy))
            vy += dv
            y += vy * dt
            if k % 2 == 1:
                hist.append((y, vy))
        (y0, v0), (y1, v1) = hist[-2], hist[-1]
        acc = ((v1 - v0) / 0.1, 0.0, 0.0)
        muzzle = (0.0, 0.0, MUZZLE_H)
        yaw, pitch = dc.exact_gun_aim(muzzle, (dist, y1, HEIGHT * 0.5), (0.0, v1, 0.0), SPEED, 3.0, True, (0.0, acc[0], 0.0))
        yaw += rng.gauss(0.0, SPREAD)
        # Shot leaves 1 or 2 ticks after the observation, then flies.
        delay = rng.choice((1, 2))
        t_fly = (dist - 3.0) / SPEED
        total = delay * dt + t_fly
        ty, tv = y1, v1
        t = 0.0
        while t < total:
            step = min(dt, total - t)
            if rng.random() < flip_rate * step:
                cmd = -cmd
            dv = max(-ACCEL * step, min(ACCEL * step, cmd - tv))
            tv += dv
            ty += tv * step
            t += step
        # Bullet lateral position at x = dist.
        by = math.tan(yaw) * dist
        if abs(by - ty) < RADIUS + BULLET_R:
            hits += 1
    return hits / shots


def main():
    print("hit% by distance; rows = lead model, cols = target behaviour")
    models = (("full lead", 1e-6, 0.0), ("damped k=1.5", 1.5, 0.0), ("damped+acc", 1.5, 0.04))
    for dist in (60.0, 100.0, 150.0, 200.0):
        print(f"\ndist {dist:.0f} m (flight {(dist-3)/SPEED:.2f} s)")
        for name, decay, ag in models:
            res = [simulate(dist, fr, decay, ag) for fr in (0.0, 0.75, 1.5)]
            print(f"  {name:14s} straight {res[0]*100:5.1f}  flip0.75/s {res[1]*100:5.1f}  flip1.5/s {res[2]*100:5.1f}")


if __name__ == "__main__":
    main()
