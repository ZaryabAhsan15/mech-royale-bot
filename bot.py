"""hunter-max: survival, weapon-aware combat and objective bot for mechfight/1 (Mech Royale).

Based on the supplied hunter-v9 bot. This version keeps its map navigation,
cover seeking, loot handling, crate support, aiming, and protocol interface,
while retaining the original weakest-first kill policy and improving defensive
and objective decisions.

Implemented additions and corrections:
- Correct weapon-triangle decisions, gated by actual effective range.
- Preserve weakest-first targeting, except for an immediate close-range laser
  charge that lightning can interrupt.
- Give a charging laser lightning-interrupt priority when it is in range.
- Lead laser aim through the remaining charge time, not just action latency.
- Predict near-term visible projectile collisions and bias movement away from
  likely gun-bullet or mortar-shell impacts.
- Prefer hard cover against a visible charging laser whose aim ray threatens us.
- Use heard weapon sounds to turn the torso toward likely sources when no
  visible target is available.
- Select safe, reachable supply drops; approach before landing when feasible,
  and stand in the ring to capture when it is safe and uncontested.
- Keep the existing crate/weapon/zone/navigation foundation intact.

This file requires the supplied mechfight.py and world.py modules and the map directory
specified by hello. It has not been evaluated by the official scorer.
"""
import math
import random
import time

import mechfight as mf
from world import World, Flow, log

PI = math.pi
TAU = 2 * math.pi


def wrap(a):
    return (a + PI) % TAU - PI


UPGRADE_PREF = [
    "gun_damage", "laser_damage", "gun_rate", "laser_charge",
    "lightning_efficiency", "armor", "plating", "nanites", "servos",
    "gyros", "sensors", "reactor", "capacitor", "lightning_chain",
    "lightning_range", "gun_mag", "gun_velocity", "jets", "mortar_splash",
    "mortar_reload", "mortar_velocity", "laser_pierce"
]

# Existing behaviour switches. The strategic changes below do not require
# changing the navigation architecture.
WEAPON_FIGHT = True
FIGHT_MOVE = False
STRAFE_W = 0.8
LOW_HP_PICK = 0.0
UPGRADE_SURVIVE_FIRST = [
    "nanites", "plating", "armor", "gun_damage", "laser_damage", "gun_rate",
    "laser_charge", "lightning_efficiency", "servos", "gyros", "sensors",
    "reactor", "capacitor", "lightning_chain", "lightning_range", "gun_mag",
    "gun_velocity", "jets", "mortar_splash", "mortar_reload", "mortar_velocity",
    "laser_pierce"
]

# Crate farming remains available for compatibility with the old bot.
FARM_CRATES = True
FARM_RANGE = 70.0
FARM_STANDOFF = 18.0
FARM_FIRE_RANGE = 40.0
FARM_SPEED = 1.0
FARM_MIN_HP = 0.85
FARM_MAX_T = 300.0
FARM_GIVEUP_S = 30.0

WEAPON_RUSH = True
RUSH_COMMIT_DIST = 30.0
RUSH_COMMIT_ENEMY = 60.0

# Laser is the default long-range pickup preference. Lightning remains useful
# as a close-range laser-charge interrupt, not as a universal weapon upgrade.
LOOT_VALUE = {"laser": 3.5, "lightning": 2.6, "mortar": 1.5}

AIM_LATENCY = 0.075
SWAP_UP = False
SWAP_RANGE = 35.0
WEAPON_RANK = {"mortar": 1, "laser": 2, "lightning": 3}
OPENING_SPIN = False
OPENING_S = 8.0
LOOT_RANGE = {"laser": 120.0, "lightning": 100.0, "mortar": 70.0}

# Cube benefits to HP and damage stop at this count in the default rules.
# Beyond it, size still increases. Only take an extra cube from very nearby
# when it is exactly one cube away from another upgrade threshold.
DEFAULT_CUBE_STAT_CAP = 10
POST_CAP_CUBE_PICKUP_DIST = 6.0


class Bot:
    name = "hunter-max"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.rng = random.Random(hello["seed"])
        self.world = World(hello["map"]["dir"])
        self.flow = None
        self.pending = None
        self.flow_goal = None
        self.hx, self.hy = 0.0, 0.0
        self.last_dmg_t = -99.0
        self.last_attacker = None
        self.last_dmg_bearing = None
        self.strafe_sign = 1
        self.strafe_until = 0.0
        self.cover_target = None
        self.cover_checked = -99.0
        self.hist = []
        self.escape_until = -1.0
        self.escape_dir = (0.0, 0.0)
        self.scan = 0.0
        self.loot_id = None
        self.dead_loot = set()
        self.ticks = 0
        self.hop_until = -1.0
        self.hop_start = -1.0
        self.stuck_n = 0
        self.fighting = False
        self.crates = {}
        self.crate_id = None
        self.crate_since = 0.0
        self.crate_bad = set()
        self.farm = None
        self.farm_firing = False
        self.beam_attackers = set()
        self.projectiles = []
        self.walk_speed = 11.7
        self.sound_bearing = None
        self.drop_plan = None
        self._target_player_pos = None
        self._target_player_weapons = set()

    # ------------------------------------------------------------------ helpers
    def _pick_upgrade(self, offer, hpf=1.0):
        """Hurt -> survival upgrade first; healthy -> damage upgrade first."""
        pref = UPGRADE_SURVIVE_FIRST if hpf < LOW_HP_PICK else UPGRADE_PREF
        best, bi = 999, 0
        for i, c in enumerate(offer["choices"]):
            try:
                r = pref.index(c["id"])
            except ValueError:
                r = 500
            if r < best:
                best, bi = r, i
        return bi

    def _zone(self, obs, x, y):
        z = obs["zone"]
        t = obs["t"]
        c0, r0, c1, r1 = z["center"], z["radius"], z["next_center"], z["next_radius"]
        s, e = z["shrink_start"], z["shrink_end"]

        def live(tt):
            f = 0.0 if tt <= s else 1.0 if tt >= e else (tt - s) / max(1e-6, e - s)
            return (c0[0] + (c1[0] - c0[0]) * f,
                    c0[1] + (c1[1] - c0[1]) * f,
                    r0 + (r1 - r0) * f)

        lx, ly, lr = live(t + 5.0)
        d_live = math.hypot(x - lx, y - ly)
        d_next = math.hypot(x - c1[0], y - c1[1])
        urgent = (not obs["self"]["in_zone"]) or d_live > lr - 10.0
        go = (t >= s - 25.0 or t >= e) and d_next > r1 - 22.0 and r1 < r0 + 1
        if d_next > 0.6 * r1:
            k = 0.6 * r1 / max(d_next, 1e-6)
            goal = (c1[0] + (x - c1[0]) * k, c1[1] + (y - c1[1]) * k)
        else:
            goal = None
        late = r1 <= 90.0
        return {
            "urgent": urgent, "go": go, "goal": goal, "c": (c1[0], c1[1]),
            "r": r1, "c0": (c0[0], c0[1]), "r0": r0, "late": late,
            "s": s, "e": e, "d_next": d_next, "live": (lx, ly, lr)
        }

    def _update_flow(self, goal, t0):
        if goal is None:
            return None
        if (self.flow_goal is None or
                math.hypot(goal[0] - self.flow_goal[0], goal[1] - self.flow_goal[1]) > 25.0):
            if (self.pending is None or
                    math.hypot(goal[0] - self.pending.gx, goal[1] - self.pending.gy) > 25.0):
                self.pending = Flow(self.world, goal[0], goal[1])
        if self.pending is not None and time.perf_counter() - t0 < 0.010:
            while not self.pending.done and time.perf_counter() - t0 < 0.010:
                self.pending.step(150)
            if self.pending.done:
                self.flow, self.pending = self.pending, None
                self.flow_goal = (self.flow.gx, self.flow.gy)
        return self.flow if self.flow is not None else self.pending

    def _goal_dir(self, flow, x, y, goal):
        gx, gy = goal[0] - x, goal[1] - y
        d = math.hypot(gx, gy)
        if d < 1e-6:
            return None
        if flow is not None and d > 12.0:
            fd = flow.direction(x, y)
            if fd is not None:
                return fd
        return gx / d, gy / d

    def _weapon_map(self, me):
        return {w["kind"]: w for w in (me.get("weapons") or []) if w}

    def _cube_stat_cap(self):
        return int((self.cfg.get("cubes") or {}).get("stat_cap", DEFAULT_CUBE_STAT_CAP))

    def _next_cube_threshold(self, current):
        thresholds = (self.cfg.get("cubes") or {}).get(
            "upgrade_thresholds", [1, 3, 6, 10, 15, 21]
        )
        for value in thresholds:
            if int(value) > current:
                return int(value)
        return None

    def _projectile_danger(self, me, projectiles):
        """Return True when a currently visible projectile is on a near-term collision path."""
        x, y, z = me["pos"]
        vx, vy, vz = me.get("vel", (0.0, 0.0, 0.0))
        for p in projectiles:
            pp = p.get("pos") or (0.0, 0.0, 0.0)
            pv = p.get("vel") or (0.0, 0.0, 0.0)
            radius = 8.5 if p.get("kind") == "mortar" else 3.8
            for dt in (0.15, 0.30, 0.45, 0.60):
                qx = pp[0] + pv[0] * dt
                qy = pp[1] + pv[1] * dt
                qz = pp[2] + pv[2] * dt
                mx = x + vx * dt
                my = y + vy * dt
                mz = z + 4.5 + vz * dt
                if math.hypot(qx - mx, qy - my) < radius and abs(qz - mz) < (7.0 if p.get("kind") == "mortar" else 4.5):
                    return True
        return False

    def _drop_plan(self, obs, me, x, y, thr, zi, t):
        """Select a safely capturable supply drop, including one that has not landed yet."""
        drops = obs.get("drops") or []
        if not drops or t > 390.0:
            return None
        my_id = me.get("id")
        hp_frac = me.get("hp", 0.0) / max(1.0, me.get("max_hp", 1400.0))
        walk = max(6.0, me.get("stats", {}).get("walk_speed", self.walk_speed))
        candidates = []
        mechs = [m for m in (obs.get("mechs") or []) if m.get("kind") == "mech" and m.get("id") != my_id]
        for d in drops:
            holder = d.get("holder")
            if holder is not None and holder != my_id:
                continue
            px, py, pz = d["pos"]
            dist = math.hypot(px - x, py - y)
            radius = float(d.get("radius", 8.0))
            lands = max(0.0, float(d.get("lands_in", 0.0)))
            capture_s = float(d.get("capture_s", 4.0))
            travel_s = max(0.0, dist - max(0.0, radius - 2.0)) / walk * 1.35
            if dist > 230.0 or (lands > 0.0 and travel_s > lands + 2.0):
                continue
            # Avoid drops that will be outside the next safe circle.
            if math.hypot(px - zi["c"][0], py - zi["c"][1]) > max(0.0, zi["r"] - 12.0):
                continue
            # Do not contest a ring with another mech, and avoid starting a long
            # capture while an enemy is already close enough to interrupt it.
            ring_contested = False
            nearest_enemy = 1e9
            for m in mechs:
                mx, my, _ = m["pos"]
                ed = math.hypot(mx - px, my - py)
                nearest_enemy = min(nearest_enemy, ed)
                if ed <= radius + 2.0:
                    ring_contested = True
            if ring_contested:
                continue
            safety_radius = 92.0 if hp_frac > 0.55 else 68.0
            if nearest_enemy < safety_radius:
                continue
            # If already capturing, preserve the progress. Otherwise prefer drops
            # that can be reached before landing and are worth the detour.
            progress = float(d.get("progress", 0.0)) if holder == my_id else 0.0
            if holder == my_id:
                priority = 1000.0 + progress
            else:
                reward = 1.0 + (1.0 - hp_frac) * 1.7
                priority = reward * 120.0 / (18.0 + dist + travel_s * 2.0)
            candidates.append((priority, {
                "id": d.get("id"), "pos": (px, py, pz), "radius": radius,
                "capture_s": capture_s, "lands_in": lands, "holder": holder,
                "progress": progress, "inside": dist <= max(1.0, radius - 1.5),
                "nearest_enemy": nearest_enemy
            }))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t0 = time.perf_counter()
        self.ticks += 1
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs["t"]
        act = {"tick": obs["tick"]}
        W = self.world
        cfgm = self.cfg["mech"]
        stats = me.get("stats") or cfgm
        walk = stats.get("walk_speed", 11.7)
        self.walk_speed = walk
        self.projectiles = obs.get("projectiles") or []
        self.sound_bearing = None
        sound_priority = {"laser_charge": 8, "laser": 7, "mortar": 6,
                          "explosion": 5, "lightning": 4, "gun": 3}
        heard = [q for q in (obs.get("sounds") or [])
                 if q.get("kind") in sound_priority and q.get("age", 99.0) < 0.8]
        if heard:
            heard.sort(key=lambda q: (sound_priority[q["kind"]] - q.get("age", 0.0) * 3.0,
                                      1 if q.get("dist") == "near" else 0), reverse=True)
            self.sound_bearing = heard[0].get("bearing")
        self._target_player_pos = (x, y)
        self._target_player_weapons = set(self._weapon_map(me))

        # Track visible beam shooters. Their actual positions still come from
        # obs['mechs']; this set only influences which visible target we prioritize.
        self.beam_attackers = {
            b.get("by") for b in (obs.get("beams") or [])
            if b.get("by") is not None
        }

        if obs.get("offer"):
            act["pick"] = self._pick_upgrade(obs["offer"], me["hp"] / max(1.0, me["max_hp"]))

        self.farm = None
        self.farm_firing = False
        if FARM_CRATES:
            self._update_crates(obs, x, y)

        for ev in obs.get("events") or []:
            if ev.get("type") == "damaged" and ev.get("by") is not None:
                self.last_dmg_t = t
                self.last_attacker = ev["by"]
                self.last_dmg_bearing = ev.get("bearing")
                if FARM_CRATES and ev["by"] >= 100:
                    for cid, ce in self.crates.items():
                        if math.hypot(ce[0] - x, ce[1] - y) < 70.0:
                            self.crate_bad.add(cid)
        hot = (t - self.last_dmg_t) < 2.5

        zi = self._zone(obs, x, y)
        urgent, go, late = zi["urgent"], zi["go"], zi["late"]

        # ---- threats ----
        thr = []
        for m in obs.get("mechs") or []:
            if m.get("kind") != "mech":
                continue
            mx, my, mz = m["pos"]
            d = math.hypot(mx - x, my - y)
            los = not W.blocked(x, y, mx, my)
            w = 1.0 if los else 0.35
            if d < 45:
                w *= 1.3
            if m.get("id") == self.last_attacker and hot:
                w *= 1.5
            if m.get("charging"):
                w *= 1.4
            hp = m.get("hp", 1400.0)
            enemy_weapons = set(m.get("weapons") or [])
            thr.append({
                "id": m["id"], "x": mx, "y": my, "z": mz,
                "vx": m["vel"][0], "vy": m["vel"][1], "vz": m["vel"][2],
                "d": d, "los": los, "w": w, "m": m, "hp": hp,
                "pct": hp / max(1.0, m.get("max_hp", 1400.0)),
                "weapons": enemy_weapons, "charging": bool(m.get("charging"))
            })

        # Heard sounds remain soft directional threats, not targetable mechs.
        for s in obs.get("sounds") or []:
            if s.get("kind") in (
                "gun", "mortar", "explosion", "laser", "lightning", "laser_charge"
            ) and s.get("age", 99.0) < 1.5:
                dd = {"near": 40.0, "mid": 110.0, "far": 240.0}.get(s.get("dist"), 240.0)
                thr.append({
                    "id": None,
                    "x": x + math.cos(s["bearing"]) * dd,
                    "y": y + math.sin(s["bearing"]) * dd,
                    "z": z, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                    "d": dd, "los": False,
                    "w": 0.25 if dd > 100 else 0.5,
                    "m": None, "hp": 1400.0, "pct": 1.0,
                    "sound": True, "weapons": set(), "charging": False
                })

        # Keep both real mechs and sound bearings nearest-first. Several later
        # movement/aim fallbacks inspect only the first few threats.
        thr.sort(key=lambda e: e["d"])
        real = [e for e in thr if not e.get("sound")]
        vis = [e for e in real if e["los"] and e["d"] < 190.0]
        danger = bool(vis) or hot
        caution = danger or any(e["d"] < 130 for e in thr)
        nearest_vis = min(vis, key=lambda e: e["d"]) if vis else None
        projectile_hot = self._projectile_danger(me, self.projectiles)
        if projectile_hot:
            danger = True
            caution = True
        self.drop_plan = self._drop_plan(obs, me, x, y, thr, zi, t)
        # Do not start a drop objective in an active close-range fight. A low-HP
        # mech may still approach if enemies are safely distant from the ring.
        if any(e["d"] < 75.0 for e in real):
            self.drop_plan = None

        # ---- fight or flee ----
        hpf = me["hp"] / max(1.0, me["max_hp"])
        ftgt = self._pick_target(vis, hot)
        n_close = sum(1 for e in vis if e["d"] < 110.0)
        fight = (
            ftgt is not None and
            (FIGHT_MOVE or (WEAPON_FIGHT and self._weapon_edge(me, ftgt))) and
            ftgt["d"] < 170.0 and not urgent and not go and
            hpf > (0.40 if self.fighting is False else 0.28) and
            (n_close <= 2 or hpf > 0.75)
        )
        self.fighting = fight

        # ---- loot ----
        loot = None
        rush_try = WEAPON_RUSH and danger and bool(vis) and not urgent and not go
        if (not danger or rush_try) and not urgent and not (late and t > 300) and self.drop_plan is None:
            farm_ok = (
                FARM_CRATES and not go and hpf >= FARM_MIN_HP and t < FARM_MAX_T and
                not rush_try and any(w and w["kind"] == "gun" for w in me["weapons"]) and
                me.get("cubes", 0) < self._cube_stat_cap()
            )
            if rush_try:
                loot = self._rush_pickup(obs, me, x, y, z, thr)
            else:
                loot = self._find_loot(obs, me, x, y, z, thr, zi, farm_ok)
        if loot is not None and loot.get("crate"):
            self.farm = loot

        # ---- movement decision ----
        goal = zi["goal"] if (urgent or go) else None
        if self.drop_plan is not None and not (urgent or go):
            goal_pt = (self.drop_plan["pos"][0], self.drop_plan["pos"][1])
        elif loot is not None:
            goal_pt = (loot["pos"][0], loot["pos"][1]) if not (urgent or go) else goal
        else:
            goal_pt = goal
        flow = self._update_flow(goal_pt, t0) if goal_pt is not None else None
        gdir = self._goal_dir(flow, x, y, goal_pt) if goal_pt is not None else None

        sneak = min(1.0, 0.92 * self.cfg["sound"]["step_speed"] / max(1.0, walk))
        speed = sneak
        w_goal = 0.0
        if urgent:
            speed, w_goal = 1.0, 6.0
        elif go:
            need = max(0.0, zi["d_next"] - (zi["r"] - 22.0))
            t_left = max(5.0, (zi["s"] + 0.7 * (zi["e"] - zi["s"])) - t)
            speed = 1.0 if need / (sneak * walk) > t_left * 0.8 else sneak
            w_goal = 3.0
        elif self.drop_plan is not None:
            w_goal = 8.0
        elif loot is not None:
            w_goal = 7.0 if rush_try else 2.5
        if danger:
            speed = 1.0
        elif caution and speed < 1.0:
            speed = sneak
        if late:
            speed = 1.0
        if FARM_CRATES and loot is not None and not loot.get("is_weapon") and not caution and not (urgent or go):
            speed = FARM_SPEED
        if WEAPON_RUSH and loot is not None and loot.get("is_weapon") and not (urgent or go):
            speed = 1.0
        if self.drop_plan is not None and not (urgent or go):
            speed = 1.0

        # ---- hop: jump + jet over walls when walled in or stuck and must leave ----
        hop = False
        need_leave = (urgent or go) and goal_pt is not None and gdir is not None
        if need_leave:
            if t < self.hop_until:
                hop = True
                if (me["grounded"] and t - self.hop_start > 1.5 and
                        not W.enclosed(x, y) and
                        W.path_clear(x, y, gdir[0], gdir[1], 12.0, z)):
                    self.hop_until = -1.0
                    hop = False
            else:
                unreachable = (
                    flow is not None and gdir is not None and
                    math.hypot(goal_pt[0] - x, goal_pt[1] - y) > 30 and
                    flow.direction(x, y) is None and flow.done
                )
                if (W.enclosed(x, y) or unreachable or self.stuck_n >= 1) and me["energy"] >= 45 and me["grounded"]:
                    hop = True
                    self.hop_start, self.hop_until = t, t + 5.0
                    self.stuck_n = 0

        move = (0.0, 0.0)
        if hop and gdir is not None:
            move = (gdir[0], gdir[1])
            self.hx, self.hy = gdir
            speed = 1.0
        elif fight:
            ux, uy = self._fight_heading(x, y, z, ftgt, thr, zi, t, self._pref_range(me, ftgt))
            move = (ux, uy)
            self.hx, self.hy = ux, uy
            speed = 1.0
        elif danger or urgent or go or loot is not None or caution:
            ux, uy = self._choose_heading(x, y, z, thr, zi, gdir, w_goal, urgent, t)
            move = (ux * speed, uy * speed)
            self.hx, self.hy = ux, uy
        else:
            if time.perf_counter() - t0 < 0.020:
                tgt = self._cover_move(x, y, t, zi, thr)
                if tgt is not None:
                    ux, uy = tgt
                    move = (ux * sneak, uy * sneak)
                    self.hx, self.hy = ux, uy

        if self.drop_plan is not None and self.drop_plan.get("inside") and not danger and not (urgent or go):
            move = (0.0, 0.0)
        if self.farm is not None and not hop and self.drop_plan is None:
            fx, fy = self.farm["pos"][0], self.farm["pos"][1]
            if math.hypot(fx - x, fy - y) <= FARM_STANDOFF and not W.blocked(x, y, fx, fy):
                move = (0.0, 0.0)

        # Stuck handling
        self.hist.append((t, x, y))
        while self.hist and t - self.hist[0][0] > 2.5:
            self.hist.pop(0)
        if hop:
            pass
        elif t < self.escape_until:
            move = (self.escape_dir[0] * max(speed, 0.8), self.escape_dir[1] * max(speed, 0.8))
        elif math.hypot(*move) > 0.6 and len(self.hist) > 15:
            if math.hypot(x - self.hist[0][1], y - self.hist[0][2]) < 3.0:
                a = self.rng.uniform(0, TAU)
                self.escape_dir = (math.cos(a), math.sin(a))
                self.escape_until = t + 1.2
                self.stuck_n += 1
        act["move"] = [round(move[0], 4), round(move[1], 4)]

        boost = False
        jump = False
        if hop:
            ceiling = W.max_top_along(x, y, move[0], move[1], 45.0)
            if me["grounded"]:
                jump = True
            elif me["energy"] > 4 and z < ceiling + 4.0:
                boost = True
        elif not me["in_zone"] and me["energy"] > 20:
            boost = True
        elif danger and nearest_vis and me["hp"] < 0.45 * me["max_hp"] and me["energy"] > 45 and nearest_vis["d"] < 90:
            boost = True
        act["boost"] = boost
        act["jump"] = jump

        # ---- weapons / aim ----
        self._combat(act, obs, me, x, y, z, t, thr, vis, hot, late, zi, move)

        # ---- pick up ----
        if loot is not None:
            lx, ly = loot["pos"][0], loot["pos"][1]
            if math.hypot(lx - x, ly - y) < self.cfg["loot"]["pickup_radius"] - 0.6 and loot.get("is_weapon"):
                act["take"] = loot["id"]
                if loot.get("swap") is not None:
                    act["drop"] = loot["swap"]
                self.dead_loot.add(loot["id"])

        mf.draw.label(f"{'URG ' if urgent else ''}{'GO ' if go else ''}{'DNG ' if danger else ''}hp{me['hp']:.0f} v{len(vis)}")
        return act

    # ------------------------------------------------------------------ loot
    def _rush_pickup(self, obs, me, x, y, z, thr):
        """Get a useful special weapon before a distant hunter makes contact."""
        # Normally do not detour for loot once a hunter is within 60 m. One
        # exception: a lightning pickup almost underfoot can be worth taking
        # when a visible laser holder is already charging inside interrupt range.
        close_laser_threat = any(
            e["d"] <= 45.0 and e["los"] and e.get("charging") and
            "laser" in e.get("weapons", set())
            for e in thr if not e.get("sound")
        )
        if any(e["d"] < RUSH_COMMIT_ENEMY for e in thr if not e.get("sound")) and not close_laser_threat:
            return None
        slots = me["weapons"]
        if len(slots) < 2 or slots[1] is not None:
            return None
        have = {w["kind"] for w in slots if w}

        best, bs = None, -1.0
        for p in obs.get("pickups") or []:
            k = p["kind"]
            if k not in ("laser", "lightning") or k in have or p["id"] in self.dead_loot:
                continue
            d = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
            if d > RUSH_COMMIT_DIST or abs(p["pos"][2] - z) > 3.0:
                continue
            if close_laser_threat and (k != "lightning" or d > 6.0):
                continue
            value = 3.8 if (k == "lightning" and close_laser_threat) else (3.5 if k == "laser" else 2.5)
            score = value / (10.0 + d)
            if score > bs:
                bs, best = score, {"id": p["id"], "pos": p["pos"], "is_weapon": True}
        return best

    def _find_loot(self, obs, me, x, y, z, thr, zi=None, farm_ok=False):
        threats_near = any(e["d"] < 100 and not e.get("sound") for e in thr)
        if threats_near:
            return None
        slots = me["weapons"]
        best, bs = None, 0.0

        if len(slots) > 1 and slots[1] is None:
            have = {w["kind"] for w in slots if w}
            for p in obs.get("pickups") or []:
                k = p["kind"]
                if k == "gun" or k in have or p["id"] in self.dead_loot or k not in LOOT_VALUE:
                    continue
                d = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
                if d > LOOT_RANGE[k] or abs(p["pos"][2] - z) > 3.0:
                    continue
                value = LOOT_VALUE[k]
                # A nearby charging/laser-equipped enemy makes lightning more
                # valuable, but only inside the range where it can actually help.
                if (k == "lightning" and d <= 45.0 and any(
                    e["d"] <= 45.0 and e["los"] and "laser" in e.get("weapons", set())
                    for e in thr if not e.get("sound")
                )):
                    value += 1.8
                score = value / (10.0 + d)
                if score > bs:
                    bs, best = score, {"id": p["id"], "pos": p["pos"], "is_weapon": True}
        elif (SWAP_UP and len(slots) > 1 and slots[1] is not None and
              slots[1]["kind"] in WEAPON_RANK and slots[1].get("charge", 0.0) <= 0.0):
            mine = WEAPON_RANK[slots[1]["kind"]]
            have = {w["kind"] for w in slots if w}
            for p in obs.get("pickups") or []:
                k = p["kind"]
                if WEAPON_RANK.get(k, 0) <= mine or k in have or p["id"] in self.dead_loot:
                    continue
                d = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
                if d > SWAP_RANGE or abs(p["pos"][2] - z) > 3.0:
                    continue
                score = WEAPON_RANK[k] / (10.0 + d)
                if score > bs:
                    bs, best = score, {"id": p["id"], "pos": p["pos"], "is_weapon": True, "swap": 1}

        if best is None and obs["t"] < 330:
            cubes_now = int(me.get("cubes", 0))
            cap = self._cube_stat_cap()
            next_threshold = self._next_cube_threshold(cubes_now)
            if cubes_now < cap:
                cube_dist_limit = 40.0
            elif next_threshold is not None and next_threshold - cubes_now == 1:
                cube_dist_limit = POST_CAP_CUBE_PICKUP_DIST
            else:
                cube_dist_limit = 0.0

            if cube_dist_limit > 0.0:
                for c in obs.get("cubes") or []:
                    d = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
                    if d <= cube_dist_limit and abs(c["pos"][2] - z) <= 3.0:
                        score = 1.0 / (10.0 + d)
                        if score > bs:
                            bs, best = score, {"id": c["id"], "pos": c["pos"], "is_weapon": False}

        if best is None and farm_ok and zi is not None:
            best = self._pick_crate(obs, x, y, z, obs["t"], thr, zi)
        return best

    # ------------------------------------------------------------------ crate farming
    def _update_crates(self, obs, x, y):
        """Remember crates (obs lists them only while visible); forget ones that are gone."""
        seen = set()
        for c in obs.get("crates") or []:
            p = c["pos"]
            self.crates[c["id"]] = [p[0], p[1], p[2], 0]
            seen.add(c["id"])
        W = self.world
        for cid in list(self.crates):
            if cid in seen:
                continue
            e = self.crates[cid]
            if math.hypot(e[0] - x, e[1] - y) < 25.0 and not W.blocked(x, y, e[0], e[1]):
                e[3] += 1
                if e[3] >= 3:
                    del self.crates[cid]
            else:
                e[3] = 0

    def _pick_crate(self, obs, x, y, z, t, thr, zi):
        """Nearest worthwhile crate, or None. Sticks to the current crate."""
        if not self.crates:
            return None
        for e in thr:
            if e["d"] <= (110.0 if e.get("sound") else 150.0):
                return None
        c, r = (zi["c0"], zi["r0"]) if t < zi["s"] - 25.0 else (zi["c"], zi["r"])
        npcs = [m["pos"] for m in obs.get("mechs") or [] if m.get("kind") == "npc"]
        W = self.world
        best, bd = None, 1e9
        for cid, e in self.crates.items():
            if cid in self.crate_bad:
                continue
            cx, cy, cz = e[0], e[1], e[2]
            d = math.hypot(cx - x, cy - y)
            if d > FARM_RANGE or abs(cz - z) > 3.0:
                continue
            if math.hypot(cx - c[0], cy - c[1]) > r - 25.0:
                continue
            if W.is_solid(cx, cy):
                continue
            if any(math.hypot(cx - n[0], cy - n[1]) < 50.0 for n in npcs):
                continue
            if cid == self.crate_id:
                d -= 15.0
            if d < bd:
                best, bd = cid, d
        if best is None:
            self.crate_id = None
            return None
        if best != self.crate_id:
            self.crate_id, self.crate_since = best, t
        elif t - self.crate_since > FARM_GIVEUP_S:
            self.crate_bad.add(best)
            self.crate_id = None
            return None
        e = self.crates[best]
        return {"id": best, "pos": (e[0], e[1], e[2]), "is_weapon": False, "crate": True}

    # ------------------------------------------------------------------ movement
    def _choose_heading(self, x, y, z, thr, zi, gdir, w_goal, urgent, t):
        W = self.world
        sc_c, sr = zi["c"], zi["r"]
        if t < zi["s"] - 25.0:
            sc_c, sr = zi["c0"], zi["r0"]
        L = 16.0
        near_thr = [e for e in thr if e["d"] < 230][:4]
        cur = near_thr[:1]
        if t > self.strafe_until:
            self.strafe_sign = self.rng.choice((-1, 1))
            self.strafe_until = t + self.rng.uniform(0.7, 1.5)
        best, bsc = None, -1e9
        for k in range(16):
            a = k * TAU / 16
            ux, uy = math.cos(a), math.sin(a)
            if not W.path_clear(x, y, ux, uy, L, z):
                continue
            px, py = x + ux * L, y + uy * L
            sc = 0.0
            dd = math.hypot(px - sc_c[0], py - sc_c[1]) - (sr - 12.0)
            if dd > 0:
                sc -= dd * (1.5 if urgent else 0.5)
            if gdir is not None:
                sc += w_goal * (ux * gdir[0] + uy * gdir[1])
            sc += 0.6 * (ux * self.hx + uy * self.hy)
            for e in near_thr:
                d0 = e["d"]
                d1 = math.hypot(px - e["x"], py - e["y"])
                w = e["w"]
                sc += 3.0 * w * (d1 - d0) / L
                if d1 < 90:
                    sc -= w * (90.0 - d1) / 45.0
                blocked = (not e.get("sound") and d1 < 220 and W.blocked(px, py, e["x"], e["y"]))
                if blocked:
                    sc += 2.2 * w
                # A visible charging laser is the highest-priority line-of-sight
                # threat. Prefer a candidate that puts solid geometry between us
                # and its muzzle; avoid candidates still lying on its aim ray.
                if not e.get("sound") and e.get("charging") and "laser" in e.get("weapons", set()):
                    if blocked:
                        sc += 7.0
                    enemy_aim = (e.get("m") or {}).get("aim") or [0.0, 0.0]
                    ay, ap = enemy_aim[0], enemy_aim[1]
                    rx, ry, rz = math.cos(ap) * math.cos(ay), math.cos(ap) * math.sin(ay), math.sin(ap)
                    ox, oy, oz = e["x"] + 3.0 * rx, e["y"] + 3.0 * ry, e["z"] + 7.0 + 3.0 * rz
                    qx, qy, qz = px - ox, py - oy, (z + 4.5) - oz
                    along = qx * rx + qy * ry + qz * rz
                    if 0.0 < along < 280.0:
                        perp = math.sqrt(max(0.0, qx*qx + qy*qy + qz*qz - along*along))
                        if perp < 5.0 and not blocked:
                            sc -= (5.0 - perp) * 2.4
            # Predict visible projectiles against each candidate direction. This
            # is deliberately short-horizon: avoid imminent shells/bullets rather
            # than making large, noisy dodges from harmless distant shots.
            for p in self.projectiles:
                pp = p.get("pos") or (0.0, 0.0, 0.0)
                pv = p.get("vel") or (0.0, 0.0, 0.0)
                radius = 8.0 if p.get("kind") == "mortar" else 3.5
                closest = 1e9
                for dt in (0.20, 0.40, 0.60):
                    qx = pp[0] + pv[0] * dt
                    qy = pp[1] + pv[1] * dt
                    qz = pp[2] + pv[2] * dt
                    mx = x + ux * self.walk_speed * dt
                    my = y + uy * self.walk_speed * dt
                    mz = z + 4.5
                    if abs(qz - mz) < (7.0 if p.get("kind") == "mortar" else 4.5):
                        closest = min(closest, math.hypot(qx - mx, qy - my))
                if closest < radius:
                    sc -= 7.0 + (radius - closest) * 2.0
            if cur and cur[0]["d"] < 150 and cur[0]["los"]:
                e = cur[0]
                b = math.atan2(e["y"] - y, e["x"] - x)
                sc += 0.9 * self.strafe_sign * (-math.sin(b) * ux + math.cos(b) * uy)
            if sc > bsc:
                bsc, best = sc, (ux, uy)
        if best is None:
            if gdir is not None:
                return gdir
            if near_thr:
                e = near_thr[0]
                d = max(1e-6, e["d"])
                return (x - e["x"]) / d, (y - e["y"]) / d
            return self.hx, self.hy
        return best

    def _pick_target(self, vis, hot):
        """Keep the reliable weak-target snowball, with an emergency laser interrupt exception."""
        if not vis:
            return None
        # If a laser is actively charging inside lightning range, interrupting it
        # can prevent a 315-damage hit and a two-second follow-up charge.
        if "lightning" in self._target_player_weapons:
            close_charge = [e for e in vis if e.get("charging") and
                            "laser" in e.get("weapons", set()) and e["d"] <= 42.0]
            if close_charge:
                return min(close_charge, key=lambda e: e["d"])
        # Preserve the previous successful policy: answer a nearby attacker,
        # otherwise finish the weakest accessible enemy first.
        if hot:
            for e in vis:
                if e["id"] == self.last_attacker and e["d"] < 60.0:
                    return e
        return min(vis, key=lambda e: e["hp"] + 10.0 * e["d"])

    def _weapon_edge(self, me, tgt):
        """Return True only for a real counter matchup at a usable distance."""
        mine = self._weapon_map(me)
        theirs = tgt.get("weapons") or set()
        d = tgt["d"]
        los = tgt.get("los", False)

        # Lightning counters a laser only inside lightning range. Mortar is a
        # counter to our lightning, so don't call this a clean advantage if held.
        if ("lightning" in mine and "laser" in theirs and "mortar" not in theirs and
                los and me.get("energy", 0.0) >= 8.0):
            lr = mine["lightning"].get("stats", {}).get("range", 45.0)
            if d <= lr - 3.0:
                return True

        # Laser counters mortar, unless the opponent also has lightning.
        if "laser" in mine and "mortar" in theirs and "lightning" not in theirs:
            lr = mine["laser"].get("stats", {}).get("range", 280.0)
            if d <= lr - 20.0:
                return True

        # Mortar counters lightning, unless the opponent also has laser.
        if "mortar" in mine and "lightning" in theirs and "laser" not in theirs:
            if 35.0 <= d <= 190.0:
                return True

        # Gun-only matchups are neutral. Merely carrying a weapon that the
        # opponent lacks is not, by itself, a weapon advantage.
        return False

    def _pref_range(self, me, tgt):
        """Choose a range consistent with the actual counter and its reach."""
        mine = self._weapon_map(me)
        theirs = tgt.get("weapons") or set()
        if ("lightning" in mine and "laser" in theirs and "mortar" not in theirs and
                tgt["d"] <= mine["lightning"].get("stats", {}).get("range", 45.0) - 3.0):
            return 32.0
        if "laser" in mine and "mortar" in theirs and "lightning" not in theirs:
            return 100.0
        if "mortar" in mine and "lightning" in theirs and "laser" not in theirs:
            return 95.0
        return 85.0

    def _fight_heading(self, x, y, z, tgt, thr, zi, t, pref):
        """Strafe around the target at preferred range, inside the zone, away from friends."""
        W = self.world
        sc_c, sr = zi["c"], zi["r"]
        if t < zi["s"] - 25.0:
            sc_c, sr = zi["c0"], zi["r0"]
        if t > self.strafe_until:
            self.strafe_sign = self.rng.choice((-1, 1))
            self.strafe_until = t + self.rng.uniform(0.8, 1.8)
        b = math.atan2(tgt["y"] - y, tgt["x"] - x)
        others = [e for e in thr if e is not tgt and not e.get("sound") and e["d"] < 160.0][:3]
        L = 12.0
        best, bsc = None, -1e9
        for k in range(16):
            a = k * TAU / 16
            ux, uy = math.cos(a), math.sin(a)
            if not W.path_clear(x, y, ux, uy, L, z):
                continue
            px, py = x + ux * L, y + uy * L
            sc = 0.0
            dd = math.hypot(px - sc_c[0], py - sc_c[1]) - (sr - 12.0)
            if dd > 0:
                sc -= dd * 0.6
            sc -= abs(math.hypot(px - tgt["x"], py - tgt["y"]) - pref) / 12.0
            sc += STRAFE_W * self.strafe_sign * (-math.sin(b) * ux + math.cos(b) * uy)
            sc += 0.3 * (ux * self.hx + uy * self.hy)
            for e in others:
                d1 = math.hypot(px - e["x"], py - e["y"])
                if d1 < e["d"]:
                    sc -= 0.6 * (e["d"] - d1) / L
                if d1 < 80.0 and W.blocked(px, py, e["x"], e["y"]):
                    sc += 0.8
            if sc > bsc:
                bsc, best = sc, (ux, uy)
        if best is None:
            d = max(1e-6, tgt["d"])
            return -(tgt["y"] - y) / d, (tgt["x"] - x) / d
        return best

    def _cover_move(self, x, y, t, zi, thr):
        """When calm: if in the open, walk quietly to nearby cover inside the safe circle."""
        W = self.world
        if not W.ok:
            return None
        if t - self.cover_checked > 1.5:
            self.cover_checked = t
            here = W.cover_score(x, y)
            best, bs = None, here + 2.5
            c, r = zi["c"], zi["r"]
            if t < zi["s"] - 25.0:
                c, r = zi["c0"], zi["r0"]
            for rad in (10.0, 20.0, 32.0, 45.0):
                for k in range(12):
                    a = k * TAU / 12
                    px, py = x + math.cos(a) * rad, y + math.sin(a) * rad
                    if W.is_solid(px, py) or math.hypot(px - c[0], py - c[1]) > r - 20.0:
                        continue
                    if not W.path_clear(x, y, math.cos(a), math.sin(a), min(rad, 40.0)):
                        continue
                    s = W.cover_score(px, py) - 0.04 * rad
                    if s > bs:
                        bs, best = s, (px, py)
            self.cover_target = best
        if self.cover_target is None:
            return None
        dx, dy = self.cover_target[0] - x, self.cover_target[1] - y
        d = math.hypot(dx, dy)
        if d < 3.0:
            self.cover_target = None
            return None
        return dx / d, dy / d

    def _farm_aim(self, me, ws, x, y, z, mh, stats, cur_yaw, cur_pitch, fire):
        """Aim the gun at the crate being farmed. Returns [yaw, pitch] or None."""
        gi = None
        for i, w in enumerate(ws):
            if w and w["kind"] == "gun":
                gi = i
        if gi is None:
            return None
        cx, cy, cz = self.farm["pos"]
        d = math.hypot(cx - x, cy - y)
        if d > FARM_FIRE_RANGE or self.world.blocked(x, y, cx, cy):
            return None
        pitch = math.atan2(cz + 1.2 - (z + mh), max(d, 1.0))
        pitch = max(stats.get("min_pitch", -0.7), min(stats.get("max_pitch", 1.3), pitch))
        yaw = math.atan2(cy - y, cx - x)
        g = ws[gi]
        if g.get("ammo", 1) > 0:
            self.farm_firing = True
            if (g.get("reload", 0.0) <= 0 and abs(wrap(yaw - cur_yaw)) < 0.05 and
                    abs(pitch - cur_pitch) < 0.08):
                fire[gi] = True
        return [yaw, pitch]

    # ------------------------------------------------------------------ combat
    def _combat(self, act, obs, me, x, y, z, t, thr, vis, hot, late, zi, move):
        stats = me.get("stats") or {}
        mh = stats.get("muzzle_height", 7.0)
        cur_yaw, cur_pitch = me["aim"][0], me["aim"][1]
        ws = me["weapons"]
        fire = [False, False]
        reload = False

        tgt = self._pick_target(vis, hot)
        self._target_player_pos = (x, y)
        self._target_player_weapons = set(self._weapon_map(me))

        want = False
        if tgt is not None:
            hpf = me["hp"] / max(1.0, me["max_hp"])
            want = (
                hot or tgt["d"] <= 75.0 or late or obs["alive"] <= 5 or
                (tgt["pct"] <= 0.35 and tgt["d"] <= 120.0) or hpf > 0.40 or
                tgt.get("charging", False) or tgt.get("id") in self.beam_attackers
            )

        laser_slot = None
        for i, w in enumerate(ws):
            if w and w["kind"] == "laser":
                laser_slot = i

        aim = None
        if tgt is not None and want:
            dx, dy = tgt["x"] - x, tgt["y"] - y
            d = math.hypot(dx, dy)
            tz = tgt["z"] + 4.5
            my_z = z + mh
            ox, oy = x, y
            if AIM_LATENCY > 0.0:
                ox += me["vel"][0] * AIM_LATENCY
                oy += me["vel"][1] * AIM_LATENCY

            opts = []
            enemy_laser_charging = bool(tgt.get("charging"))
            for i, w in enumerate(ws):
                if not w:
                    continue
                k = w["kind"]
                cd = w.get("cooldown", 0.0)
                st = w.get("stats", {})
                if k == "laser":
                    charging = w.get("charge", 0.0) > 0
                    if charging or (cd <= 0.05 and d <= min(235.0, st.get("range", 280.0) - 20)):
                        opts.append((9 if charging else 3, i))
                elif k == "lightning":
                    lr = st.get("range", 45.0)
                    if d <= lr - 3.0 and me["energy"] >= 8:
                        # Interrupting an enemy's active laser charge is more
                        # valuable than continuing our own ordinary attack.
                        priority = 12 if enemy_laser_charging else 4
                        opts.append((priority, i))
                elif k == "mortar":
                    if cd <= 0.05 and 38.0 <= d <= 190.0:
                        opts.append((2, i))
                elif k == "gun":
                    if (w.get("ammo", 1) > 0 and w.get("reload", 0.0) <= 0 and
                            d <= min(190.0, st.get("range", 230.0) - 20)):
                        opts.append((1, i))

            if opts:
                opts.sort(reverse=True)
                prio, si = opts[0]
                w = ws[si]
                k = w["kind"]
                st = w.get("stats", {})
                if k == "gun" or k == "mortar":
                    spd = st.get("speed", 252.0 if k == "gun" else 72.0)
                    px, py, pz = tgt["x"], tgt["y"], tz
                    tt = d / spd + AIM_LATENCY
                    for _ in range(3):
                        px = tgt["x"] + tgt["vx"] * tt
                        py = tgt["y"] + tgt["vy"] * tt
                        pz = tz + tgt["vz"] * tt * 0.5
                        tt = math.hypot(px - ox, py - oy) / spd + AIM_LATENCY
                    yaw = math.atan2(py - oy, px - ox)
                    hd = math.hypot(px - ox, py - oy)
                    if k == "gun":
                        pitch = math.atan2(pz - my_z, hd)
                    else:
                        g = self.cfg["weapons"]["mortar"].get("gravity", 22.0)
                        r = g * hd / (spd * spd)
                        pitch = 0.5 * math.asin(r) if r < 0.98 else None
                else:
                    lead_t = AIM_LATENCY
                    if k == "laser":
                        # Hitscan occurs on release, not at trigger press. Lead the
                        # target through the remaining charge time instead of only
                        # compensating for the observation/action latency.
                        full_charge = st.get("charge", 0.75)
                        charged = w.get("charge", 0.0)
                        lead_t += max(0.0, full_charge - charged)
                        lead_t = min(1.0, lead_t)
                    lx = tgt["x"] + tgt["vx"] * lead_t
                    ly = tgt["y"] + tgt["vy"] * lead_t
                    lz = tz + tgt["vz"] * lead_t
                    yaw = math.atan2(ly - oy, lx - ox)
                    pitch = math.atan2(lz - my_z, math.hypot(lx - ox, ly - oy))

                if pitch is not None:
                    pitch = max(stats.get("min_pitch", -0.7), min(stats.get("max_pitch", 1.3), pitch))
                    aim = [yaw, pitch]
                    tol = max(0.02, min(0.09, 2.0 / max(d, 1.0)))
                    err_y = abs(wrap(yaw - cur_yaw))
                    ok = err_y < tol and abs(pitch - cur_pitch) < max(tol, 0.06)
                    if k == "lightning":
                        cone_deg = st.get("cone_deg", 30.0)
                        # Leave a small safety margin inside the actual half-cone.
                        ok = err_y < math.radians(max(2.0, cone_deg * 0.5 - 1.0))
                    if k == "laser":
                        ch = w.get("charge", 0.0)
                        full = st.get("charge", 0.75)
                        if ch > 0:
                            fire[si] = ch < full * 0.97 and err_y < 0.2
                        else:
                            fire[si] = ok
                    else:
                        fire[si] = ok

        if laser_slot is not None and ws[laser_slot].get("charge", 0.0) > 0 and not fire[laser_slot] and aim is None:
            fire[laser_slot] = False

        if FARM_CRATES and aim is None and tgt is None and self.farm is not None:
            aim = self._farm_aim(me, ws, x, y, z, mh, stats, cur_yaw, cur_pitch, fire)

        if aim is None:
            # Turn the torso toward the most recent damage bearing when we are
            # not actively aiming a shot. This helps keep the attacker in front.
            if hot and self.last_dmg_bearing is not None:
                aim = [self.last_dmg_bearing, 0.0]
            elif tgt is not None:
                aim = [math.atan2(tgt["y"] - y, tgt["x"] - x), 0.0]
            elif self.sound_bearing is not None:
                aim = [self.sound_bearing, 0.0]
            elif thr and not thr[0].get("sound") and thr[0]["d"] < 160:
                aim = [math.atan2(thr[0]["y"] - y, thr[0]["x"] - x), 0.0]
            elif math.hypot(*move) > 0.05:
                aim = [math.atan2(move[1], move[0]), 0.0]
            else:
                self.scan = wrap(self.scan + (0.38 if (OPENING_SPIN and t < OPENING_S) else 0.22))
                aim = [self.scan, 0.0]

        act["aim"] = [round(aim[0], 4), round(aim[1], 4)]
        act["fire"] = fire

        g = ws[0] if ws and ws[0] and ws[0]["kind"] == "gun" else None
        if g is not None:
            mag = g.get("stats", {}).get("mag", 12)
            if g.get("reload", 0.0) <= 0 and (
                g.get("ammo", mag) <= 0 or
                (not vis and g.get("ammo", mag) < mag and not self.farm_firing)
            ):
                reload = True
        act["reload"] = reload


if __name__ == "__main__":
    mf.run(Bot)
