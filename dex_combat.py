"""Exact gun intercept math, weapon control, and target selection."""

import math

LATENCY = 0.075
EFFECTIVE_RANGE = 175.0


def angle_diff(a, b):
    return math.remainder(a - b, math.tau)


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def gun_flight_time(rx, ry, rz, vx, vy, vz, speed, launch_offset):
    """Solve |r + v*t| = launch_offset + speed*t for the earliest forward intercept."""
    dist = math.sqrt(rx * rx + ry * ry + rz * rz)
    if dist <= launch_offset:
        return 0.0
    a = vx * vx + vy * vy + vz * vz - speed * speed
    b = 2.0 * (rx * vx + ry * vy + rz * vz - speed * launch_offset)
    c = (dist - launch_offset) * (dist + launch_offset)
    disc = b * b - 4.0 * a * c
    if disc >= 0.0:
        denom = math.sqrt(disc) - b
        if denom > 0.0:
            return 2.0 * c / denom
    return dist / speed


# Enemy velocity autocorrelation measured from the kit's recorded match decays
# ~exp(-LEAD_DECAY*tau): strafing bots flip often, so expected displacement over
# tau is v*(1-exp(-k*tau))/k, not v*tau. Current acceleration adds a small term.
LEAD_DECAY = 1.0
ACC_GAIN = 0.04
MAX_ACC = 60.0


def lead_displacement(vx, vy, vz, ax, ay, tau):
    if tau <= 0.0:
        return 0.0, 0.0, 0.0
    g = (1.0 - math.exp(-LEAD_DECAY * tau)) / LEAD_DECAY
    ga = ACC_GAIN * tau
    return vx * g + ax * ga, vy * g + ay * ga, vz * g


def exact_gun_aim(muzzle, target_pos, target_vel, speed, launch_offset, grounded, target_acc=None):
    """Compute (yaw, pitch) intercept with damped lead, decision latency and launch offset."""
    speed = speed if speed > 0 else 252.0
    launch_offset = launch_offset if launch_offset > 0 else 3.0
    vx, vy = target_vel[0], target_vel[1]
    vz = 0.0 if grounded else target_vel[2] * 0.5
    ax = ay = 0.0
    if target_acc is not None:
        ax, ay = target_acc[0], target_acc[1]
        am = math.hypot(ax, ay)
        if am > MAX_ACC:
            ax, ay = ax * MAX_ACC / am, ay * MAX_ACC / am
    px, py, pz = target_pos[0], target_pos[1], target_pos[2]
    for _ in range(3):
        d = math.sqrt((px - muzzle[0]) ** 2 + (py - muzzle[1]) ** 2 + (pz - muzzle[2]) ** 2)
        tau = LATENCY + max(0.0, d - launch_offset) / speed
        dx, dy, dz = lead_displacement(vx, vy, vz, ax, ay, tau)
        px = target_pos[0] + dx
        py = target_pos[1] + dy
        pz = target_pos[2] + dz
    return math.atan2(py - muzzle[1], px - muzzle[0]), math.atan2(pz - muzzle[2], math.hypot(px - muzzle[0], py - muzzle[1]))


def mortar_aim(arena, tube, foot, target_vel, speed, gravity, stats):
    """Low-arc mortar launch solution with trajectory obstruction check."""
    if speed <= 0 or gravity <= 0:
        return None
    t = 0.0
    yaw = pitch = 0.0
    for _ in range(3):
        px = foot[0] + target_vel[0] * (LATENCY + t)
        py = foot[1] + target_vel[1] * (LATENCY + t)
        dx = math.hypot(px - tube[0], py - tube[1])
        dz = foot[2] - tube[2]
        disc = speed ** 4 - gravity * (gravity * dx * dx + 2.0 * dz * speed * speed)
        if disc < 0.0 or dx < 1.0:
            return None
        pitch = math.atan((speed * speed - math.sqrt(disc)) / (gravity * dx))
        yaw = math.atan2(py - tube[1], px - tube[0])
        t = dx / max(speed * math.cos(pitch), 1e-6)
    if pitch < stats["min_pitch"] or pitch > stats["max_pitch"]:
        return None
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    tt = 0.12
    while tt < t * 0.9:
        horiz = speed * cp * tt
        sx = tube[0] + cy * horiz
        sy_ = tube[1] + sy * horiz
        sz = tube[2] + speed * sp * tt - 0.5 * gravity * tt * tt
        if sz <= arena.floor(sx, sy_):
            return None
        for solid in arena.nearby((sx, sy_), (sx, sy_)):
            if solid.base <= sz <= solid.top and solid.touches(sx, sy_):
                return None
        tt += 0.12
    return yaw, pitch


def weapon_slot(weapons, kind):
    for i, w in enumerate(weapons):
        if w is not None and w["kind"] == kind:
            return i, w
    return -1, None


def mech_scale(cfg, cubes):
    return 1.0 + cfg["cubes"]["size_per_cube"] * cubes


def actor_center(cfg, track):
    if track["kind"] == "npc":
        h = cfg["npc"]["height"]
    else:
        h = cfg["mech"]["height"] * mech_scale(cfg, track.get("cubes", 0))
    p = track["pos"]
    return (p[0], p[1], p[2] + h * 0.5)


def actor_eye(cfg, track):
    p = track["pos"]
    h = cfg["mech"]["eye_height"] * mech_scale(cfg, track.get("cubes", 0))
    return (p[0], p[1], p[2] + h)


def shows_back(cfg, track, x, y):
    arc = cfg["mech"].get("front_arc_deg", 180.0)
    if arc <= 0:
        arc = 180.0
    bearing_to_me = math.atan2(y - track["pos"][1], x - track["pos"][0])
    return abs(angle_diff(track["aim"][0], bearing_to_me)) > math.radians(arc * 0.5) + 0.1


def select_targets(cfg, me, tracks, now, params, stick_id, stick_at, alive, late):
    """Score visible enemies, returning (seen, target, hidden, shy)."""
    x, y, _ = me["pos"]
    hp_frac = me["hp"] / max(1.0, me["max_hp"])
    max_t = params["max_target"]
    hpw = params["hpw"]
    seen = []
    target = None
    hidden = None
    shy = None
    best_score = math.inf

    for t in tracks.values():
        if now - t["seen"] > 2.5:
            continue
        t["engage"] = False
        seen.append(t)
        if t["seen"] != now:
            t["los"] = False
            continue

        to_me = math.atan2(y - t["pos"][1], x - t["pos"][0])
        off = abs(angle_diff(t["aim"][0], to_me))
        aimed = off < 0.22
        dist = t["dist"]

        if params["ghost"]:
            engage = (
                dist < params["engage_dist"]
                or now - t["hit_me"] < 4.0
                or (aimed and t.get("charging") and dist < 120.0)
                or (t["hp"] < 350.0 and dist < 120.0)
                or alive <= 2
            )
            if not engage and params["shy"] > 0 and t["los"] and params["engage_dist"] <= dist < params["shy"] and (t["closing"] > 3.0 or aimed):
                if shy is None or dist < shy["dist"]:
                    shy = t
        else:
            engage = (
                dist < params["engage_dist"]
                or now - t["hit_me"] < 6.0
                or late
                or (t["closing"] > 3.0 and dist < 180.0)
                or (t["hp"] < 0.4 * max(1.0, t["max_hp"]) and dist < 160.0)
                or (aimed and dist < 190.0)
            )
        t["engage"] = engage

        if engage and not t["los"] and dist < max_t:
            if hidden is None or dist < hidden["dist"]:
                hidden = t
        if not engage or not t["los"] or dist > max_t:
            continue

        score = dist + hpw * t["hp"]
        if params["low_bonus"] > 0 and t["hp"] <= 0.3 * max(1.0, t["max_hp"]):
            score -= params["low_bonus"]
        if params["stick"] > 0 and t["id"] == stick_id and now - stick_at < 2.0:
            score -= params["stick"]
        if params["backstab"] > 0 and shows_back(cfg, t, x, y):
            score -= params["backstab"]
        if now - t["hit_me"] < 4.0:
            score -= 40.0
        if t["closing"] > 3.0:
            score -= 15.0
        if score < best_score:
            best_score = score
            target = t

    return seen, target, hidden, shy


def compute_fire_and_aim(
    cfg, arena, me, target, foe, farm, farm_pos, seen,
    move_xy, now, mode, params, last_hurt, hurt_bearing,
    npc_hurt, npc_bearing, last_threat_at,
    sound_t=-99.0, sound_bearing=0.0, face=None, laser_mode=None,
):
    """Return (aim, fire, want_reload). `face`: last-known hunter to keep the torso on (rear hits x1.5).

    `laser_mode`: None = normal; "hold" = don't START a charge (peek cycle walking to cover);
    "off" = never pull the laser trigger (shorted by lightning)."""
    x, y, zf = me["pos"]
    vx, vy, _ = me["vel"]
    st = me["stats"]
    muzzle = (x + vx * LATENCY, y + vy * LATENCY, zf + st["muzzle_height"])
    weapons = me["weapons"]
    _, gun = weapon_slot(weapons, "gun")

    shootable = False
    is_mech = False
    tp = (0.0, 0.0, 0.0)
    tv = (0.0, 0.0, 0.0)
    tr = th = 0.0
    tacc = None

    if target is not None:
        sc = mech_scale(cfg, target.get("cubes", 0))
        tr = cfg["mech"]["radius"] * sc
        th = cfg["mech"]["height"] * sc
        p = target["pos"]
        tp = (p[0], p[1], p[2] + th * 0.5)
        if not arena.shot_clear(muzzle, tp):
            tp = (p[0], p[1], p[2] + th * 0.85)
        tv = tuple(target.get("vel", (0.0, 0.0, 0.0)))
        tacc = target.get("acc")
        shootable = is_mech = True
    elif farm is not None:
        tr = cfg["npc"]["radius"]
        th = cfg["npc"]["height"]
        p = farm["pos"]
        tp = (p[0], p[1], p[2] + th * 0.5)
        tv = tuple(farm.get("vel", (0.0, 0.0, 0.0)))
        shootable = True
    elif farm_pos is not None:
        tr = cfg["crates"]["radius"]
        th = cfg["crates"]["height"]
        tp = farm_pos
        shootable = True

    fire = [False, False]
    aim = [me["aim"][0], 0.0]
    have_aim = False
    urgent_move = params["suppress_urgent_laser"] and mode in ("zone", "escape", "dodge", "cover")

    if shootable:
        dist = math.hypot(tp[0] - x, tp[1] - y)
        grounded = abs(tv[2]) < 0.5
        gun_ok = False
        gy = gp = 0.0
        if gun is not None:
            launch_offset = st["radius"] + 0.5 if st["radius"] > 0 else 3.0
            gy, gp = exact_gun_aim(muzzle, tp, tv, gun["stats"]["speed"], launch_offset, grounded, tacc)
            gun_ok = dist <= gun["stats"]["range"] * 0.96
            if gun_ok and is_mech and dist > EFFECTIVE_RANGE:
                # Beyond ~175 m a strafing mech's position is too uncertain over the
                # bullet flight time; hold the mag unless it moves little across the
                # line of sight or is already trading shots with us.
                lat = abs(tv[0] * (tp[1] - y) - tv[1] * (tp[0] - x)) / max(dist, 1.0)
                if lat > 4.0 and now - target["hit_me"] > 2.0:
                    gun_ok = False

        dp = (tp[0] + tv[0] * (LATENCY + 0.05), tp[1] + tv[1] * (LATENCY + 0.05), tp[2])
        hy = math.atan2(dp[1] - muzzle[1], dp[0] - muzzle[0])
        hp_ang = math.atan2(dp[2] - muzzle[2], math.hypot(dp[0] - muzzle[0], dp[1] - muzzle[1]))
        aim = [gy, gp] if gun is not None else [hy, hp_ang]
        have_aim = True

        tol_y = math.atan2(tr * 0.75, max(dist, 1.0))
        tol_p = math.atan2(th * 0.4, max(dist, 1.0))

        def slew_ok(want_y, want_p):
            ey = max(0.0, abs(angle_diff(want_y, me["aim"][0])) - st["turn_rate"] * 0.05)
            ep = max(0.0, abs(want_p - me["aim"][1]) - st["pitch_rate"] * 0.05)
            return ey < tol_y and ep < tol_p

        rear_mult = max(1.0, cfg["mech"].get("rear_mult", 1.5))

        for i, w in enumerate(weapons[:2]):
            if w is None:
                continue
            kind = w["kind"]
            ws = w["stats"]
            if kind == "gun":
                if gun_ok and w.get("ammo", 0) > 0 and w.get("reload", 0.0) <= 0.0 and slew_ok(gy, gp):
                    fire[i] = True
            elif kind == "lightning":
                cone = math.radians(ws["cone_deg"])
                if dist < ws["range"] - 1.5 and abs(angle_diff(hy, me["aim"][0])) < cone * 0.5 - 0.05 and me["energy"] >= ws["energy"]:
                    fire[i] = True
            elif kind == "laser":
                if not is_mech or dist > ws["range"] * 0.97:
                    continue
                if laser_mode == "off" or (laser_mode == "hold" and w.get("charge", 0.0) <= 0.0):
                    fire[i] = False
                    continue
                if urgent_move:
                    # Don't slow down urgent escapes with a laser charge; release immediately if already charged.
                    fire[i] = False
                    if w.get("charge", 0.0) > 0:
                        aim = [hy, hp_ang]
                    continue
                full = ws["charge"]
                charge = w.get("charge", 0.0)
                if params["lethal"] and ws["min_charge"] <= charge < full - 0.051:
                    dmg = (
                        ws["damage"]
                        * (ws["min_frac"] + (1.0 - ws["min_frac"]) * charge / full)
                        * (1.0 + cfg["cubes"]["damage_per_cube"] * min(me["cubes"], cfg["cubes"]["stat_cap"]))
                    )
                    if "mortar" in target.get("weapons", ()):
                        dmg *= 1.0 + ws["cookoff"]
                    if params["backstab"] > 0 and shows_back(cfg, target, x, y):
                        dmg *= rear_mult
                    if dmg * params["lethal_k"] >= target["hp"]:
                        if abs(angle_diff(hy, me["aim"][0])) < tol_y * 0.9 and abs(hp_ang - me["aim"][1]) < tol_p:
                            aim = [hy, hp_ang]
                            fire[i] = False
                            continue
                if charge < full - 0.051:
                    fire[i] = True
                    if charge > full * 0.45:
                        aim = [hy, hp_ang]
                else:
                    aim = [hy, hp_ang]
                    if abs(angle_diff(hy, me["aim"][0])) < tol_y * 0.9 and abs(hp_ang - me["aim"][1]) < tol_p:
                        fire[i] = False
                    else:
                        fire[i] = True
            elif kind == "mortar":
                if not is_mech or w.get("cooldown", 0.0) > 0.25 or dist < 16.0 or dist > 200.0:
                    continue
                tube = (muzzle[0], muzzle[1], zf + st["height"] + 0.5)
                foot = (tp[0], tp[1], tp[2] - th * 0.5)
                sol = mortar_aim(arena, tube, foot, tv, ws["speed"], ws["gravity"], st)
                if sol is not None:
                    my_, mp_ = sol
                    if gun is None or gun.get("reload", 0.0) > 0 or gun.get("ammo", 0) == 0 or not gun_ok:
                        aim = [my_, mp_]
                    ey = max(0.0, abs(angle_diff(my_, me["aim"][0])) - st["turn_rate"] * 0.05)
                    ep = max(0.0, abs(mp_ - me["aim"][1]) - st["pitch_rate"] * 0.05)
                    if ey < 0.04 and ep < 0.03:
                        fire[i] = True

    # Pre-charge laser when moving onto an engaged hidden target (unless suppressing urgent slowdown).
    laser_charging_hidden = False
    if target is None and foe is not None and not urgent_move and laser_mode != "off":
        for i, w in enumerate(weapons[:2]):
            if w is not None and w["kind"] == "laser":
                ws = w["stats"]
                charge = w.get("charge", 0.0)
                if laser_mode == "hold" and charge <= 0.0:
                    continue
                if w.get("cooldown", 0.0) <= 0.05 and foe["dist"] < ws["range"] and charge < ws["charge"] + ws["hold_s"] - 0.35:
                    fire[i] = True
                if charge > 0.0:
                    laser_charging_hidden = True

    if not have_aim and laser_charging_hidden:
        # Pre-aim the torso at the hidden foe's predicted center while still covered, so the
        # beam needs no slew once we step out (aim turns without LOS).
        fp = foe["pos"]
        fv = foe.get("vel", (0.0, 0.0, 0.0))
        fh = cfg["mech"]["height"] * mech_scale(cfg, foe.get("cubes", 0))
        px = fp[0] + fv[0] * (LATENCY + 0.1)
        py = fp[1] + fv[1] * (LATENCY + 0.1)
        pz = fp[2] + fh * 0.5
        aim = [math.atan2(py - muzzle[1], px - muzzle[0]), math.atan2(pz - muzzle[2], max(1.0, math.hypot(px - muzzle[0], py - muzzle[1])))]
        have_aim = True

    if not have_aim:
        watch = None
        for t in seen:
            if t["engage"] or now - t["hit_me"] < 6.0:
                if watch is None or t["dist"] < watch["dist"]:
                    watch = t
        mx, my = move_xy
        if watch is not None:
            aim = [math.atan2(watch["pos"][1] - y, watch["pos"][0] - x), 0.0]
        elif face is not None and now - face.get("seen", -99.0) < 8.0:
            aim = [math.atan2(face["pos"][1] - y, face["pos"][0] - x), 0.0]
        elif npc_hurt > 0 and now - npc_hurt < 2.5:
            aim = [npc_bearing, 0.0]
        elif now - last_hurt < 2.5:
            aim = [hurt_bearing, 0.0]
        elif params.get("sound_scan", False) and now - sound_t < 2.0:
            aim = [sound_bearing, 0.0]
        elif mx != 0.0 or my != 0.0:
            aim = [math.atan2(my, mx) + 1.15 * math.sin(now * 1.3), 0.0]
        else:
            aim = [me["aim"][0] + 0.35, 0.0]

    aim[1] = clamp(aim[1], st["min_pitch"], st["max_pitch"])
    want_reload = False
    if gun is not None and gun.get("reload", 0.0) <= 0.0 and gun.get("ammo", 0) < gun["stats"]["mag"]:
        if target is None and farm is None and farm_pos is None:
            if gun.get("ammo", 0) <= gun["stats"]["mag"] * 0.6 or now - last_threat_at > 3.0:
                want_reload = True

    return aim, fire, want_reload
