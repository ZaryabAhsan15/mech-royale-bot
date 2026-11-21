"""Survival-first bot for mechfight/1 (Mech Royale).

Score is placement, so the plan is: stay inside the closing zone, stay quiet (moving slower than
`sound.step_speed` makes no noise, standing still makes none), keep buildings between us and
enemies, regenerate when nobody is shooting, and only shoot when attacked / cornered / late game.

Never raises: mechfight.run() also re-sends the last move if act() throws.
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


UPGRADE_PREF = ["armor", "plating", "nanites", "servos", "gyros", "sensors", "reactor", "capacitor",
                "gun_damage", "gun_rate", "laser_damage", "laser_charge", "lightning_efficiency",
                "lightning_chain", "lightning_range", "gun_mag", "gun_velocity", "jets",
                "mortar_splash", "mortar_reload", "mortar_velocity", "laser_pierce"]
LOOT_VALUE = {"laser": 3.0, "lightning": 2.5, "mortar": 1.5}
LOOT_RANGE = {"laser": 90.0, "lightning": 70.0, "mortar": 45.0}


class Bot:
    name = "survivor-v1"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.rng = random.Random(hello["seed"])
        self.world = World(hello["map"]["dir"])
        self.flow = None
        self.pending = None
        self.flow_goal = None
        self.hx, self.hy = 0.0, 0.0           # last chosen heading
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
        self.hop_until = -1.0      # while t < hop_until: jump + jet over walls toward the goal
        self.hop_start = -1.0
        self.stuck_n = 0

    # ------------------------------------------------------------------ helpers
    def _pick_upgrade(self, offer):
        best, bi = 999, 0
        for i, c in enumerate(offer["choices"]):
            try:
                r = UPGRADE_PREF.index(c["id"])
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
            return c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, r0 + (r1 - r0) * f

        lx, ly, lr = live(t + 5.0)
        d_live = math.hypot(x - lx, y - ly)
        d_next = math.hypot(x - c1[0], y - c1[1])
        urgent = (not obs["self"]["in_zone"]) or d_live > lr - 10.0
        go = (t >= s - 25.0 or t >= e) and d_next > r1 - 22.0 and r1 < r0 + 1
        # a point comfortably inside the next circle, on our side of it
        if d_next > 0.6 * r1:
            k = 0.6 * r1 / max(d_next, 1e-6)
            goal = (c1[0] + (x - c1[0]) * k, c1[1] + (y - c1[1]) * k)
        else:
            goal = None
        late = r1 <= 90.0
        return {"urgent": urgent, "go": go, "goal": goal, "c": (c1[0], c1[1]), "r": r1,
                "c0": (c0[0], c0[1]), "r0": r0, "late": late, "s": s, "e": e, "d_next": d_next,
                "live": (lx, ly, lr)}

    def _update_flow(self, goal, t0):
        if goal is None:
            return None
        if self.flow_goal is None or math.hypot(goal[0] - self.flow_goal[0], goal[1] - self.flow_goal[1]) > 25.0:
            if self.pending is None or math.hypot(goal[0] - self.pending.gx, goal[1] - self.pending.gy) > 25.0:
                self.pending = Flow(self.world, goal[0], goal[1])
        if self.pending is not None and time.perf_counter() - t0 < 0.02:
            self.pending.step(1500)
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

        if obs.get("offer"):
            act["pick"] = self._pick_upgrade(obs["offer"])

        for ev in obs["events"]:
            if ev["type"] == "damaged" and ev.get("by") is not None:
                self.last_dmg_t = t
                self.last_attacker = ev["by"]
                self.last_dmg_bearing = ev.get("bearing")
        hot = (t - self.last_dmg_t) < 2.5

        zi = self._zone(obs, x, y)
        urgent, go, late = zi["urgent"], zi["go"], zi["late"]

        # ---- threats ----
        thr = []
        for m in obs["mechs"]:
            if m["kind"] != "mech":
                continue
            mx, my, mz = m["pos"]
            d = math.hypot(mx - x, my - y)
            los = not W.blocked(x, y, mx, my)
            w = 1.0 if los else 0.35
            if d < 45:
                w *= 1.3
            if m["id"] == self.last_attacker and hot:
                w *= 1.5
            if m.get("charging"):
                w *= 1.4
            hp = m.get("hp", 1400.0)
            thr.append({"id": m["id"], "x": mx, "y": my, "z": mz, "vx": m["vel"][0], "vy": m["vel"][1],
                        "vz": m["vel"][2], "d": d, "los": los, "w": w, "m": m, "hp": hp,
                        "pct": hp / max(1.0, m.get("max_hp", 1400.0))})
        thr.sort(key=lambda e: e["d"])
        # heard gunfire/explosions: soft threats toward the bearing
        for s in obs["sounds"]:
            if s["kind"] in ("gun", "mortar", "explosion", "laser", "lightning", "laser_charge") and s["age"] < 1.5:
                dd = {"near": 40.0, "mid": 110.0, "far": 240.0}[s["dist"]]
                thr.append({"id": None, "x": x + math.cos(s["bearing"]) * dd, "y": y + math.sin(s["bearing"]) * dd,
                            "z": z, "vx": 0.0, "vy": 0.0, "vz": 0.0, "d": dd, "los": False, "w": 0.25 if dd > 100 else 0.5,
                            "m": None, "hp": 1400.0, "pct": 1.0, "sound": True})
        real = [e for e in thr if not e.get("sound")]
        vis = [e for e in real if e["los"] and e["d"] < 190.0]
        danger = bool(vis) or hot
        caution = danger or any(e["d"] < 130 for e in thr)
        nearest_vis = min(vis, key=lambda e: e["d"]) if vis else None

        # ---- loot ----
        loot = None
        if not danger and not urgent and not (late and t > 300):
            loot = self._find_loot(obs, me, x, y, z, thr)

        # ---- movement decision ----
        goal = zi["goal"] if (urgent or go) else None
        if loot is not None:
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
        elif loot is not None:
            w_goal = 2.5
        if danger:
            speed = 1.0
        elif caution and speed < 1.0:
            speed = sneak
        if late:
            speed = 1.0

        # ---- hop: jump + jet over walls when we are walled in or stuck and must leave ----
        hop = False
        need_leave = (urgent or go) and goal_pt is not None and gdir is not None
        if need_leave:
            if t < self.hop_until:
                hop = True
                if me["grounded"] and t - self.hop_start > 1.5 and not W.enclosed(x, y) and \
                        W.path_clear(x, y, gdir[0], gdir[1], 12.0, z):
                    self.hop_until = -1.0
                    hop = False
            else:
                unreachable = flow is not None and gdir is not None and math.hypot(goal_pt[0] - x, goal_pt[1] - y) > 30 \
                    and flow.direction(x, y) is None and flow.done
                if (W.enclosed(x, y) or unreachable or self.stuck_n >= 1) and me["energy"] >= 45 and me["grounded"]:
                    hop = True
                    self.hop_start, self.hop_until = t, t + 5.0
                    self.stuck_n = 0

        move = (0.0, 0.0)
        if hop and gdir is not None:
            move = (gdir[0], gdir[1])
            self.hx, self.hy = gdir
            speed = 1.0
        elif danger or urgent or go or loot is not None or caution:
            ux, uy = self._choose_heading(x, y, z, thr, zi, gdir, w_goal, urgent, t)
            move = (ux * speed, uy * speed)
            self.hx, self.hy = ux, uy
        else:
            # calm and safe: look for cover, otherwise hold still (silent)
            tgt = self._cover_move(x, y, t, zi, thr)
            if tgt is not None:
                ux, uy = tgt
                move = (ux * sneak, uy * sneak)
                self.hx, self.hy = ux, uy

        # stuck handling
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
                self.dead_loot.add(loot["id"])  # don't retry forever if it fails

        mf.draw.label(f"{'URG ' if urgent else ''}{'GO ' if go else ''}{'DNG ' if danger else ''}hp{me['hp']:.0f} v{len(vis)}")
        return act

    # ------------------------------------------------------------------ loot
    def _find_loot(self, obs, me, x, y, z, thr):
        threats_near = any(e["d"] < 100 and not e.get("sound") for e in thr)
        if threats_near:
            return None
        slots = me["weapons"]
        best, bs = None, 0.0
        if len(slots) > 1 and slots[1] is None:
            have = {w["kind"] for w in slots if w}
            for p in obs["pickups"]:
                k = p["kind"]
                if k == "gun" or k in have or p["id"] in self.dead_loot or k not in LOOT_VALUE:
                    continue
                d = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
                if d > LOOT_RANGE[k] or abs(p["pos"][2] - z) > 3.0:
                    continue
                sc = LOOT_VALUE[k] / (10.0 + d)
                if sc > bs:
                    bs, best = sc, {"id": p["id"], "pos": p["pos"], "is_weapon": True}
        if best is None and obs["t"] < 330:
            for c in obs["cubes"]:
                d = math.hypot(c["pos"][0] - x, c["pos"][1] - y)
                if d <= 28 and abs(c["pos"][2] - z) <= 3.0:
                    sc = 1.0 / (10.0 + d)
                    if sc > bs:
                        bs, best = sc, {"id": c["id"], "pos": c["pos"], "is_weapon": False}
        return best

    # ------------------------------------------------------------------ movement
    def _choose_heading(self, x, y, z, thr, zi, gdir, w_goal, urgent, t):
        W = self.world
        cx, cy = zi["live"][0], zi["live"][1]
        # keep inside the NEXT circle once it's the binding one, else the live one
        sc_c, sr = zi["c"], zi["r"]
        if t < zi["s"] - 25.0:
            sc_c, sr = zi["c0"], zi["r0"]
        L = 16.0
        near_thr = [e for e in thr if e["d"] < 230][:6]
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
                if not e.get("sound") and d1 < 200 and W.blocked(px, py, e["x"], e["y"]):
                    sc += 1.6 * w
            if cur and cur[0]["d"] < 150 and cur[0]["los"]:
                e = cur[0]
                b = math.atan2(e["y"] - y, e["x"] - x)
                sc += 0.9 * self.strafe_sign * (-math.sin(b) * ux + math.cos(b) * uy)
            if sc > bsc:
                bsc, best = sc, (ux, uy)
        if best is None:  # boxed in: try the goal direction, else away from the nearest threat
            if gdir is not None:
                return gdir
            if near_thr:
                e = near_thr[0]
                d = max(1e-6, e["d"])
                return (x - e["x"]) / d, (y - e["y"]) / d
            return self.hx, self.hy
        return best

    def _cover_move(self, x, y, t, zi, thr):
        """When calm: if we're out in the open, walk (quietly) to nearby cover inside the safe circle."""
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

    # ------------------------------------------------------------------ combat
    def _combat(self, act, obs, me, x, y, z, t, thr, vis, hot, late, zi, move):
        stats = me.get("stats") or {}
        mh = stats.get("muzzle_height", 7.0)
        eyeh = stats.get("height", 9.0)
        cur_yaw, cur_pitch = me["aim"][0], me["aim"][1]
        ws = me["weapons"]
        fire = [False, False]
        reload = False

        # choose a target
        tgt = None
        if vis:
            def key(e):
                return (0 if (hot and e["id"] == self.last_attacker) else 1, e["d"])
            tgt = min(vis, key=key)

        want = False
        if tgt is not None:
            want = (hot or tgt["d"] <= 75.0 or late or obs["alive"] <= 5 or
                    (tgt["pct"] <= 0.35 and tgt["d"] <= 120.0))

        # laser mid-charge: must keep deciding about it even without a target
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
            opts = []
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
                    if d <= st.get("range", 45.0) - 4 and me["energy"] >= 8:
                        opts.append((4, i))
                elif k == "mortar":
                    if cd <= 0.05 and 38.0 <= d <= 190.0:
                        opts.append((2, i))
                elif k == "gun":
                    if w.get("ammo", 1) > 0 and w.get("reload", 0.0) <= 0 and d <= min(190.0, st.get("range", 230.0) - 20):
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
                    tt = d / spd
                    for _ in range(3):
                        px = tgt["x"] + tgt["vx"] * tt
                        py = tgt["y"] + tgt["vy"] * tt
                        pz = tz + tgt["vz"] * tt * 0.5
                        tt = math.hypot(px - x, py - y) / spd
                    yaw = math.atan2(py - y, px - x)
                    hd = math.hypot(px - x, py - y)
                    if k == "gun":
                        pitch = math.atan2(pz - my_z, hd)
                    else:
                        g = self.cfg["weapons"]["mortar"].get("gravity", 22.0)
                        r = g * hd / (spd * spd)
                        pitch = 0.5 * math.asin(r) if r < 0.98 else None
                else:  # laser / lightning: hitscan, aim straight at the target
                    yaw = math.atan2(dy, dx)
                    pitch = math.atan2(tz - my_z, d)
                if pitch is not None:
                    pitch = max(stats.get("min_pitch", -0.7), min(stats.get("max_pitch", 1.3), pitch))
                    aim = [yaw, pitch]
                    tol = max(0.02, min(0.09, 2.0 / max(d, 1.0)))
                    err_y = abs(wrap(yaw - cur_yaw))
                    ok = err_y < tol and abs(pitch - cur_pitch) < max(tol, 0.06)
                    if k == "lightning":
                        ok = err_y < 0.35
                    if k == "laser":
                        ch = w.get("charge", 0.0)
                        full = st.get("charge", 0.75)
                        if ch > 0:
                            fire[si] = ch < full * 0.92 and err_y < 0.2  # release when full (or aim lost)
                        else:
                            fire[si] = ok
                    else:
                        fire[si] = ok

        # a charging laser with no target/permission: let go so it doesn't just sit
        if laser_slot is not None and ws[laser_slot].get("charge", 0.0) > 0 and not fire[laser_slot] and aim is None:
            fire[laser_slot] = False

        if aim is None:
            if tgt is not None:   # have a target but holding fire: still face it (front armor)
                aim = [math.atan2(tgt["y"] - y, tgt["x"] - x), 0.0]
            elif hot and self.last_dmg_bearing is not None:
                aim = [self.last_dmg_bearing, 0.0]
            elif thr and not thr[0].get("sound") and thr[0]["d"] < 160:
                aim = [math.atan2(thr[0]["y"] - y, thr[0]["x"] - x), 0.0]
            elif math.hypot(*move) > 0.05:
                aim = [math.atan2(move[1], move[0]), 0.0]
            else:
                self.scan = wrap(self.scan + 0.22)
                aim = [self.scan, 0.0]
        act["aim"] = [round(aim[0], 4), round(aim[1], 4)]
        act["fire"] = fire

        g = ws[0] if ws and ws[0] and ws[0]["kind"] == "gun" else None
        if g is not None:
            mag = g.get("stats", {}).get("mag", 12)
            if g.get("reload", 0.0) <= 0 and (g.get("ammo", mag) <= 0 or (not vis and g.get("ammo", mag) < mag)):
                reload = True
        act["reload"] = reload


if __name__ == "__main__":
    mf.run(Bot)
