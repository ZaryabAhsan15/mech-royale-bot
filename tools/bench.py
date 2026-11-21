"""Benchmark Bot.act() latency by replaying synthetic observations with the real map.

Usage (single run against current bot.py):
    python tools/bench.py --arena <kit>/arena

Usage (before/after comparison via git):
    python tools/bench.py --arena <kit>/arena --compare

The --arena argument must point to the directory that contains v2/world.json and
v2/terrain.i16.  If you also have a check/stream.jsonl.gz from the kit, pass it
with --stream and observations will be replayed from that file instead of the
built-in synthetic set.

Output per run:
  world.ok = True/False   <- tells you the map really loaded
  n=<N>  p50=X ms  p95=Y ms  max=Z ms  over25=A  over50=B
"""
import argparse
import gzip
import importlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Percentile (stdlib only)
# ---------------------------------------------------------------------------

def percentile(data, pct):
    if not data:
        return float("nan")
    s = sorted(data)
    idx = (len(s) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def report(timings, label=""):
    if not timings:
        print("  No timings collected.")
        return
    p50 = percentile(timings, 50)
    p95 = percentile(timings, 95)
    mx  = max(timings)
    over25 = sum(1 for t in timings if t > 25.0)
    over50 = sum(1 for t in timings if t > 50.0)
    tag = f"[{label}] " if label else ""
    print(f"  {tag}n={len(timings)}  p50={p50:.2f} ms  p95={p95:.2f} ms  "
          f"max={mx:.2f} ms  over25={over25}  over50={over50}")


# ---------------------------------------------------------------------------
# Minimal stub config (used when stream file is absent)
# ---------------------------------------------------------------------------

STUB_CFG = {
    "match": {"duration": 480},
    "bot": {},
    "mech": {"walk_speed": 11.7, "boost_speed": 23.4, "turn_rate": 4.0,
             "jump_speed": 14.0, "jet_accel": 20.0, "max_rise_speed": 18.0,
             "jet_drain": 25.0, "gravity": 20.0, "height": 9.0,
             "muzzle_height": 7.0, "min_pitch": -0.7, "max_pitch": 1.3,
             "rear_mult": 1.5},
    "vision": {"range": 200.0, "fov_deg": 120.0, "near_range": 45.0,
               "canopy_blocks": True, "show_enemy_hp": True},
    "sound": {"bearing_noise_deg": 15.0, "near": 50.0, "mid": 130.0, "far": 300.0,
              "step_speed": 8.5, "step_every": 1.5, "range": {}},
    "weapons": {
        "gun":       {"damage": 45.0, "range": 230.0, "speed": 252.0, "mag": 12,
                      "fire_rate": 4.0, "reload": 2.4},
        "mortar":    {"damage": 90.0, "range": 190.0, "speed": 72.0,
                      "splash": 12.0, "reload": 3.5, "gravity": 22.0},
        "laser":     {"damage": 120.0, "range": 280.0, "charge": 0.75,
                      "min_charge": 0.2, "pierce": 1},
        "lightning": {"damage": 60.0, "range": 45.0, "chain": 2,
                      "chain_range": 20.0, "efficiency": 0.5},
    },
    "projectile": {},
    "cubes": {"upgrade_thresholds": [3, 6, 10, 15, 21, 28, 36, 45]},
    "crates": {},
    "loot": {"pickup_radius": 5.0, "equip_s": 1.5, "floor_count": 20,
             "floor_roof_fraction": 0.4, "crate_weapon_chance": 0.3,
             "death_drop": 0.8,
             "weights": {"gun": 1.0, "mortar": 0.4, "laser": 0.3, "lightning": 0.3}},
    "zone": {
        "start_center": [0.0, 0.0], "start_radius": 450.0,
        "phases": [
            {"hold_s": 60, "shrink_s": 60, "radius": 300.0, "dps": 5.0},
            {"hold_s": 60, "shrink_s": 60, "radius": 180.0, "dps": 10.0},
            {"hold_s": 60, "shrink_s": 60, "radius": 90.0,  "dps": 15.0},
            {"hold_s": 60, "shrink_s": 60, "radius": 40.0,  "dps": 20.0},
            {"hold_s": 60, "shrink_s": 60, "radius": 10.0,  "dps": 30.0},
        ],
    },
    "replay": {},
    "debug": {"max_lines": 50, "max_points": 50, "max_text": 20,
              "max_text_chars": 200, "max_bytes": 4096,
              "max_match_bytes": 65536, "hold_s": 1.0},
    "upgrades": [],
}


# ---------------------------------------------------------------------------
# Zone helper (mirrors sim.py)
# ---------------------------------------------------------------------------

def _zone_at(cfg, t, centers):
    z = cfg["zone"]
    c = list(z["start_center"]); r = z["start_radius"]
    tt = 0.0
    for k, ph in enumerate(z["phases"]):
        nc, nr = centers[k], ph["radius"]
        s, e = tt + ph["hold_s"], tt + ph["hold_s"] + ph["shrink_s"]
        if t < e:
            f = 0.0 if t < s else (t - s) / ph["shrink_s"]
            live = (c[0] + (nc[0] - c[0]) * f,
                    c[1] + (nc[1] - c[1]) * f,
                    r   + (nr  - r)  * f)
            return {"center": list(c), "radius": r,
                    "next_center": list(nc), "next_radius": nr,
                    "shrink_start": s, "shrink_end": e, "dps": ph["dps"]}, live
        c, r, tt = list(nc), nr, e
    return {"center": c, "radius": r, "next_center": c, "next_radius": r,
            "shrink_start": tt, "shrink_end": tt,
            "dps": z["phases"][-1]["dps"]}, (c[0], c[1], r)


# ---------------------------------------------------------------------------
# Build synthetic observation sequence
# ---------------------------------------------------------------------------

def build_obs_sequence(cfg, arena, n_ticks=800, seed=42):
    """Return (world, [obs, ...]) using the real arena map."""
    from world import World
    world = World(arena)

    rng = random.Random(seed)
    z0 = cfg["zone"]
    centers = []
    c, r = z0["start_center"], z0["start_radius"]
    for ph in z0["phases"]:
        a   = rng.uniform(0, math.tau)
        rr  = rng.uniform(0, max(0.0, r - ph["radius"]) * 0.9)
        nc  = [c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr]
        centers.append(nc); c, r = nc, ph["radius"]

    # Spawn position: place the mech FAR from the zone's next center so the bot
    # immediately needs to navigate (flow field fires every tick).
    # Phase 1 starts shrinking at t=60 s; we set t_offset so we begin mid-shrink.
    cx0, cy0 = z0["start_center"]

    # Pick a spawn point near the edge of the start circle (not inside a wall)
    me_x, me_y = cx0 + z0["start_radius"] * 0.75, cy0
    for _ in range(200):
        a  = rng.uniform(0, math.tau)
        rr = z0["start_radius"] * rng.uniform(0.55, 0.80)
        tx, ty = cx0 + math.cos(a) * rr, cy0 + math.sin(a) * rr
        if not world.is_solid(tx, ty):
            me_x, me_y = tx, ty
            break

    # Enemies: mix of near (triggers danger/heading-scan) and far (triggers LOS)
    enemies = []
    for i, dist in enumerate([55, 110, 180, 42, 90, 140]):
        a = i * math.tau / 6
        ex, ey = me_x + math.cos(a) * dist, me_y + math.sin(a) * dist
        enemies.append({"id": float(i + 1), "x": ex, "y": ey})

    DT = 0.1
    # Start time offset so zone is already shrinking (phase 1 shrinks t=60..120)
    # We use three segments of n_ticks/3 each:
    #   segment A: t ~ 65..90  (zone shrinking, enemies visible -> _update_flow + _choose_heading)
    #   segment B: t ~ 125..150 (zone shrinking phase 2, enemies visible)
    #   segment C: t ~ 10..40  (early game, calm + no enemies -> _cover_move)
    seg = n_ticks // 3
    time_offsets = [65.0, 125.0, 10.0]
    obs_list = []

    for seg_idx, t_start in enumerate(time_offsets):
        seg_ticks = seg if seg_idx < 2 else (n_ticks - 2 * seg)
        ex_x, ex_y = me_x, me_y

        for tick in range(seg_ticks):
            t = round(t_start + tick * DT, 2)
            zone, live = _zone_at(cfg, t, centers)
            inz = math.hypot(ex_x - live[0], ex_y - live[1]) <= live[2]

            # Calm segment: no enemies visible, so _cover_move runs
            if seg_idx == 2:
                obs_mechs = []
                events = []
            else:
                obs_mechs = []
                for e in enemies:
                    d = math.hypot(e["x"] - ex_x, e["y"] - ex_y)
                    los = not world.blocked(ex_x, ex_y, e["x"], e["y"])
                    if los or d < 50:
                        obs_mechs.append({
                            "id": e["id"], "kind": "mech",
                            "pos": [e["x"], e["y"], 0.0],
                            "vel": [0.0, 0.0, 0.0],
                            "aim": [0.0, 0.0],
                            "hp": 1400.0, "max_hp": 1400.0,
                            "weapons": ["gun"], "charging": False,
                        })
                # Simulate being hit every 50 ticks so hot/strafe paths run
                events = []
                if tick % 50 == 10 and obs_mechs:
                    events.append({
                        "type": "damaged", "amount": 45.0, "weapon": "gun",
                        "bearing": math.atan2(enemies[0]["y"] - ex_y,
                                              enemies[0]["x"] - ex_x),
                        "by": enemies[0]["id"],
                    })

            obs = {
                "type": "obs", "tick": int(t * 10) * 2, "t": t,
                "alive": max(1, 16 - int(t) // 30),
                "self": {
                    "id": 0.0,
                    "pos": [ex_x, ex_y, 0.0],
                    "vel": [0.0, 0.0, 0.0],
                    "aim": [0.0, 0.0],
                    "hp": 1400.0, "max_hp": 1400.0,
                    "energy": 100.0, "max_energy": 100.0,
                    "grounded": True,
                    "cubes": float(int(t) // 30),
                    "upgrades": [],
                    "in_zone": inz,
                    "weapons": [
                        {"kind": "gun", "cooldown": 0.0, "ammo": 12, "reload": 0.0,
                         "stats": cfg["weapons"]["gun"]},
                        None,
                    ],
                    "stats": cfg["mech"],
                },
                "mechs": obs_mechs,
                "projectiles": [], "crates": [], "cubes": [],
                "pickups": [], "sounds": [], "beams": [],
                "events": events, "zone": zone, "offer": None, "drops": [],
            }
            obs_list.append(obs)

            # Advance position so the bot has to keep recalculating flow
            ex_x += math.cos(tick * 0.3) * cfg["mech"]["walk_speed"] * DT * 0.4
            ex_y += math.sin(tick * 0.3) * cfg["mech"]["walk_speed"] * DT * 0.4

    return world, obs_list


# ---------------------------------------------------------------------------
# Run benchmark for a given bot module loaded from a file path
# ---------------------------------------------------------------------------

def run_bench(bot_path, arena, stream_path, n_ticks, label, cfg_override=None):
    """
    Import bot from bot_path, build obs sequence, time act(), return timings.
    Prints world.ok status.
    """
    # Import the bot module from an explicit file path without polluting sys.modules
    import importlib.util
    spec = importlib.util.spec_from_file_location("_bench_bot", bot_path)
    botmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(botmod)

    from world import World

    cfg = cfg_override

    if stream_path and os.path.isfile(stream_path):
        print(f"  Loading config from stream: {stream_path}")
        with gzip.open(stream_path, "rt") as f:
            first = json.loads(f.readline())
        cfg = first.get("config", first)
    elif cfg is None:
        cfg = STUB_CFG

    print(f"  Building obs sequence ({n_ticks} ticks) with arena: {arena}")
    world, obs_list = build_obs_sequence(cfg, arena, n_ticks=n_ticks)
    print(f"  world.ok = {world.ok}")

    hello = {
        "type": "hello", "protocol": "mechfight/1",
        "slot": 0, "seed": 42,
        "map": {"dir": arena, "files": []},
        "config": cfg,
    }

    import mechfight as mf
    mf.draw.caps = cfg.get("debug", {})

    bot = botmod.Bot(hello)
    timings = []
    for obs in obs_list:
        mf.draw.clear()
        t0 = time.perf_counter()
        bot.act(obs)
        timings.append((time.perf_counter() - t0) * 1000.0)

    return timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark Bot.act() with real map.")
    ap.add_argument(
        "--arena", required=True,
        help="Path to the arena directory containing v2/world.json (from the mechfight kit).",
    )
    ap.add_argument(
        "--stream", default="",
        help="Optional path to check/stream.jsonl.gz; its config is used if present.",
    )
    ap.add_argument(
        "--ticks", type=int, default=800,
        help="Number of synthetic ticks to run (default 800).",
    )
    ap.add_argument(
        "--compare", action="store_true",
        help="Also benchmark the original bot.py from HEAD (git show HEAD:bot.py) "
             "and print both sets of numbers side-by-side.",
    )
    args = ap.parse_args()

    arena = os.path.abspath(args.arena)
    if not os.path.isdir(arena):
        sys.exit(f"ERROR: --arena path does not exist or is not a directory: {arena}")

    world_json = os.path.join(arena, "v2", "world.json")
    if not os.path.isfile(world_json):
        sys.exit(f"ERROR: v2/world.json not found inside {arena}\n"
                 f"       Make sure --arena points to the arena folder in the mechfight kit.")

    bot_path = os.path.join(ROOT, "bot.py")
    stream   = args.stream if args.stream else ""

    # Pre-load cfg once so both runs use the same config
    cfg = None
    if stream and os.path.isfile(stream):
        with gzip.open(stream, "rt") as f:
            first = json.loads(f.readline())
        cfg = first.get("config", first)
    else:
        cfg = STUB_CFG

    if args.compare:
        # --- original bot: use tools/_orig_bot.py (pre-edit snapshot) if present,
        #     otherwise fall back to git show HEAD:bot.py ---
        print("\n=== ORIGINAL (pre-edit bot.py) ===")
        orig_timings = None
        orig_path_tmp = None
        try:
            local_orig = os.path.join(HERE, "_orig_bot.py")
            if os.path.isfile(local_orig):
                orig_path = local_orig
                cleanup = False
            else:
                result = subprocess.run(
                    ["git", "show", "HEAD:bot.py"],
                    capture_output=True, text=True, cwd=ROOT, timeout=10,
                )
                if result.returncode != 0:
                    print(f"  git show failed: {result.stderr.strip()}")
                    raise RuntimeError("no original available")
                import tempfile as _tf
                with _tf.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False, dir=ROOT
                ) as tf:
                    tf.write(result.stdout)
                    orig_path = tf.name
                orig_path_tmp = orig_path
                cleanup = True
            try:
                orig_timings = run_bench(orig_path, arena, stream, args.ticks,
                                        label="original", cfg_override=cfg)
            finally:
                if cleanup and orig_path_tmp and os.path.exists(orig_path_tmp):
                    os.unlink(orig_path_tmp)
        except Exception as exc:
            print(f"  Could not benchmark original: {exc}")

        if orig_timings:
            report(orig_timings, label="original")

    # --- current bot.py ---
    print("\n=== CURRENT (working tree bot.py) ===")
    timings = run_bench(bot_path, arena, stream, args.ticks,
                        label="current", cfg_override=cfg)
    report(timings, label="current")

    if args.compare and orig_timings:
        print("\n=== SUMMARY ===")
        report(orig_timings, label="original")
        report(timings,      label="current ")


if __name__ == "__main__":
    main()
