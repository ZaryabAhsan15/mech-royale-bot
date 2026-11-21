"""Rough local simulator (NOT the real game) to catch behaviour bugs: getting stuck in walls,
leaving the zone, dying to a hunting squad. It uses the real map and zone timeline, but
combat is crude. Compare versions of the bot relative to each other, not to leaderboard scores.

  python3 tools/sim.py --arena <kit>/arena --matches 10
"""
import argparse
import copy
import gzip
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bot as botmod  # noqa: E402
from world import World  # noqa: E402

DT = 0.1


class Mech:
    def __init__(self, i, x, y, hp=1400.0):
        self.id, self.x, self.y, self.z = i, x, y, 0.0
        self.vx = self.vy = 0.0
        self.hp, self.maxhp = hp, hp
        self.aim = [0.0, 0.0]
        self.alive = True
        self.last_hit = -99.0
        self.energy = 100.0
        self.ammo, self.reload_t, self.cd = 12, 0.0, 0.0
        self.vz = 0.0
        self.grounded = True
        self.known = {}  # id -> (last seen t)


def zone_at(cfg, t, centers):
    z = cfg["zone"]
    c = list(z["start_center"]); r = z["start_radius"]
    tt = 0.0
    out = None
    for k, ph in enumerate(z["phases"]):
        nc, nr = centers[k], ph["radius"]
        s, e = tt + ph["hold_s"], tt + ph["hold_s"] + ph["shrink_s"]
        if t < e:
            f = 0.0 if t < s else (t - s) / ph["shrink_s"]
            live = (c[0] + (nc[0] - c[0]) * f, c[1] + (nc[1] - c[1]) * f, r + (nr - r) * f)
            return {"center": list(c), "radius": r, "next_center": list(nc), "next_radius": nr,
                    "shrink_start": s, "shrink_end": e, "dps": ph["dps"]}, live
        c, r, tt = list(nc), nr, e
    return {"center": c, "radius": r, "next_center": c, "next_radius": r, "shrink_start": tt, "shrink_end": tt,
            "dps": z["phases"][-1]["dps"]}, (c[0], c[1], r)


def run_match(seed, hello_cfg, world, verbose=False, n_house=15, hunters="squad"):
    rng = random.Random(seed)
    cfg = hello_cfg
    z0 = cfg["zone"]
    # random zone centers, each inside the previous circle
    centers, c, r = [], z0["start_center"], z0["start_radius"]
    for ph in z0["phases"]:
        a, rr = rng.uniform(0, math.tau), rng.uniform(0, max(0.0, r - ph["radius"]) * 0.9)
        nc = [c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr]
        centers.append(nc); c, r = nc, ph["radius"]
    mechs = []
    while len(mechs) < 16:
        a, rr = rng.uniform(0, math.tau), z0["start_radius"] * 0.8 * math.sqrt(rng.random())
        x, y = z0["start_center"][0] + math.cos(a) * rr, z0["start_center"][1] + math.sin(a) * rr
        if world.is_solid(x, y) or any(math.hypot(x - m.x, y - m.y) < 130 for m in mechs):
            continue
        mechs.append(Mech(len(mechs), x, y))
    me = mechs[0]
    hello = {"type": "hello", "protocol": "mechfight/1", "slot": 0, "seed": seed,
             "map": {"dir": ARENA, "files": []}, "config": cfg}
    b = botmod.Bot(hello)
    t = 0.0; tick = 0; place = 16; alive = 16
    events = []
    max_ms = 0.0
    import time
    while t < 480 and me.alive:
        zone, live = zone_at(cfg, t, centers)
        # ---- bot decision
        obs_mechs = []
        for m in mechs:
            if m is me or not m.alive:
                continue
            d = math.hypot(m.x - me.x, m.y - me.y)
            seen = False
            if not world.blocked(me.x, me.y, m.x, m.y):
                if d < 45 + 0:
                    seen = True
                elif d < 200:
                    ang = abs(botmod.wrap(math.atan2(m.y - me.y, m.x - me.x) - me.aim[0]))
                    seen = ang < math.radians(60)
            if seen:
                me.known[m.id] = t
            if m.id in me.known and t - me.known[m.id] <= 5.0:
                obs_mechs.append({"id": m.id, "kind": "mech", "pos": [m.x, m.y, 0.0], "vel": [m.vx, m.vy, 0.0],
                                  "aim": m.aim, "hp": m.hp, "max_hp": m.maxhp, "weapons": ["gun"], "charging": False})
        inz = math.hypot(me.x - live[0], me.y - live[1]) <= live[2]
        obs = {"type": "obs", "tick": tick, "t": round(t, 2), "alive": sum(1 for m in mechs if m.alive),
               "self": {"id": 0, "pos": [me.x, me.y, me.z], "vel": [me.vx, me.vy, me.vz], "aim": me.aim, "hp": me.hp,
                        "max_hp": me.maxhp, "energy": me.energy, "max_energy": 100, "grounded": me.grounded, "cubes": 0,
                        "upgrades": [], "in_zone": inz,
                        "weapons": [{"kind": "gun", "cooldown": max(0, me.cd), "ammo": me.ammo, "reload": me.reload_t,
                                     "stats": cfg["weapons"]["gun"]}, None],
                        "stats": cfg["mech"]},
               "mechs": obs_mechs, "projectiles": [], "crates": [], "cubes": [], "pickups": [], "sounds": [],
               "beams": [], "events": events, "zone": zone, "offer": None, "drops": []}
        events = []
        t0 = time.perf_counter()
        act = b.act(obs)
        max_ms = max(max_ms, (time.perf_counter() - t0) * 1000)
        # ---- apply my action
        mv = act.get("move", [0, 0]); n = math.hypot(*mv)
        if n > 1: mv = [mv[0] / n, mv[1] / n]
        spd = cfg["mech"]["boost_speed"] if act.get("boost") and me.energy > 0 else cfg["mech"]["walk_speed"]
        me_cmd = (mv[0] * spd, mv[1] * spd)
        aim = act.get("aim", me.aim)
        tr = cfg["mech"]["turn_rate"] * DT
        me.aim = [me.aim[0] + max(-tr, min(tr, botmod.wrap(aim[0] - me.aim[0]))), aim[1]]
        shots = [(me, act["fire"][0])] if act.get("fire") else []
        # ---- house bots (hunters): chase me weakest-first, share vision -> they always know where I am
        # when any of them has LOS within 200m; otherwise head to zone center and sweep.
        shared_see = any((not world.blocked(m.x, m.y, me.x, me.y)) and math.hypot(m.x - me.x, m.y - me.y) < 200
                         for m in mechs if m is not me and m.alive)
        cmds = {0: me_cmd}
        for m in mechs[1:]:
            if not m.alive: continue
            if m.id <= n_house:
                target = (me.x, me.y) if (shared_see or t > 150) and hunters == "squad" else (live[0], live[1])
            else:
                target = (live[0], live[1])
            dx, dy = target[0] - m.x, target[1] - m.y; d = math.hypot(dx, dy) or 1
            sp = 11.7 if d > 40 else 0.0
            cmds[m.id] = (dx / d * sp, dy / d * sp)
            m.aim = [math.atan2(me.y - m.y, me.x - m.x), 0.0]
            dm = math.hypot(me.x - m.x, me.y - m.y)
            if dm < 150 and not world.blocked(m.x, m.y, me.x, me.y) and m.id <= n_house:
                # hunter squad fires: ~25% of gun shots hit on average at range (they lead imperfectly)
                if rng.random() < DT / 0.24 * (0.5 if dm < 60 else 0.25):
                    front_ok = abs(botmod.wrap(me.aim[0] - (m.aim[0] + math.pi))) < math.pi / 2
                    dmg = 45 * (1.0 if front_ok else 1.5)
                    me.hp -= dmg; me.last_hit = t
                    events.append({"type": "damaged", "amount": dmg, "weapon": "gun", "bearing": m.aim[0] + math.pi,
                                   "by": m.id, "rear": not front_ok})
                    m.known[0] = t
        # my shots: hit nearest visible mech if aligned
        for (s, f) in shots:
            if f and me.ammo > 0 and me.cd <= 0:
                me.ammo -= 1; me.cd = 0.24
                for m in sorted((m for m in mechs[1:] if m.alive), key=lambda m: math.hypot(m.x - me.x, m.y - me.y)):
                    d = math.hypot(m.x - me.x, m.y - me.y)
                    if d < 190 and not world.blocked(me.x, me.y, m.x, m.y):
                        if abs(botmod.wrap(math.atan2(m.y - me.y, m.x - me.x) - me.aim[0])) < 0.04 + 1.5 / d:
                            m.hp -= 45
                            if m.hp <= 0: m.alive = False
                        break
        me.cd -= DT
        if me.ammo == 0 and me.reload_t <= 0: me.reload_t = 2.4
        if me.reload_t > 0:
            me.reload_t -= DT
            if me.reload_t <= 0: me.ammo = 12
        if act.get("reload") and me.reload_t <= 0 and me.ammo < 12: me.reload_t = 2.4
        # ---- movement with wall sliding (+ simple vertical model so jets/roofs work for the bot)
        for m in mechs:
            if not m.alive: continue
            vx, vy = cmds.get(m.id, (0, 0))
            zz = m.z if m is me else None
            if m is me and not me.grounded:
                vx, vy = vx * 0.9, vy * 0.9
            nx, ny = m.x + vx * DT, m.y + vy * DT
            if world.is_solid(nx, ny, zz):
                if not world.is_solid(nx, m.y, zz): ny = m.y
                elif not world.is_solid(m.x, ny, zz): nx = m.x
                else: nx, ny = m.x, m.y
            m.vx, m.vy = (nx - m.x) / DT, (ny - m.y) / DT
            m.x, m.y = nx, ny
            if m is me:
                if act.get("jump") and me.grounded:
                    me.vz = cfg["mech"]["jump_speed"]; me.grounded = False
                if act.get("boost") and not me.grounded and me.energy > 0:
                    me.vz = min(me.vz + cfg["mech"]["jet_accel"] * DT, cfg["mech"]["max_rise_speed"])
                    me.energy = max(0, me.energy - cfg["mech"]["jet_drain"] * DT)
                elif act.get("boost") and me.grounded:
                    me.energy = max(0, me.energy - 17 * DT)
                else:
                    me.energy = min(100, me.energy + 18 * DT)
                me.vz -= cfg["mech"]["gravity"] * DT
                me.z += me.vz * DT
                g = 0.0
                if world.is_solid(me.x, me.y):
                    k = int((me.y - world.miny) / 2.0) * world.nx + int((me.x - world.minx) / 2.0)
                    if world.top[k] <= me.z + 1.2: g = world.top[k]
                if me.z <= g:
                    me.z, me.vz, me.grounded = g, 0.0, True
                else:
                    me.grounded = False
            # zone damage + regen
            if math.hypot(m.x - live[0], m.y - live[1]) > live[2]:
                m.hp -= zone["dps"] * DT; m.last_hit = t
                if m is me: events.append({"type": "damaged", "amount": zone["dps"] * DT, "weapon": "zone", "bearing": None, "by": None})
            elif t - m.last_hit > 5.0 and m.alive:
                m.hp = min(m.maxhp, m.hp + 45 * DT)
            if m.hp <= 0: m.alive = False
        # house bots also die to each other crudely: random attrition so late game thins out
        if t > 60 and rng.random() < DT / 14:
            vict = [m for m in mechs[1:] if m.alive]
            if vict: rng.choice(vict).alive = False
        t += DT; tick += 2
    alive_n = sum(1 for m in mechs if m.alive)
    place = 1 if (me.alive and alive_n == 1) else (alive_n + 1 if not me.alive else max(1, alive_n))
    return {"seed": seed, "t": round(t, 1), "alive": me.alive, "place": place, "hp": round(me.hp), "max_ms": round(max_ms, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--matches", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--hunters", default="squad")
    args = ap.parse_args()
    ARENA = args.arena
    with gzip.open(args.stream, "rt") as f:
        cfg = json.loads(f.readline())["config"]
    world = World(ARENA)
    res = []
    for k in range(args.matches):
        r = run_match(args.seed0 + k, cfg, world, hunters=args.hunters)
        res.append(r); print(r, flush=True)
    pts = [16 - r["place"] for r in res]
    print(f"mean points {sum(pts) / len(pts):.2f}  mean survival {sum(r['t'] for r in res) / len(res):.0f}s  worst tick ms {max(r['max_ms'] for r in res)}")
