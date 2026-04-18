"""
Entelect Grand Prix - Level 3 solver.

What's new vs Level 2:
- 70 laps on Spa (35 segments, 15050 m/lap).
- Weather cycles: Cold (1000s) → Light Rain (3000s) → Heavy Rain (2000s) → Dry (6000s) → repeat.
- Weather affects: tyre friction multiplier, acceleration multiplier, deceleration multiplier.
- Fuel soft cap: 370L (more generous than L2's 219L, since race is longer).
- Tyres still don't degrade (that's L4).

Strategy:
- For each weather phase, use the highest-friction tyre:
    dry         -> Soft         (friction 1.18)
    cold        -> Soft         (friction 1.00)
    light_rain  -> Intermediate (friction 1.08)
    heavy_rain  -> Wet          (friction 1.20)
- Pit for tyre change when weather transitions to a phase requiring a different tyre.
- Refuel at the same pits (opportunistic — no extra cost beyond refuel time).
- Use full-attack target speed 90 m/s; let reduced accel/brake in wet weather naturally slow us.

Approach:
1. Simulate race in laps, tracking cumulative time.
2. Determine weather for each lap from cumulative time at lap start.
3. When weather changes, pit at end of the preceding lap to swap tyres.
4. Plan fuel strategy separately (refuel enough to reach next pit / end of race).

Fuel model (empirically verified from L2 log):
- Fuel burnt during accel + cruise phases, NOT during braking.
- Formula per phase: F = (K_base + K_drag × v_avg²) × distance
  where K_base = 0.0005, K_drag = 1.5e-9.
"""

import json
import math
import os
from pathlib import Path

G = 9.8
SAFETY_MARGIN = 0.1
K_BASE = 0.0005
K_DRAG = 0.0000000015

WEATHER_KEYS = {
    "dry": "dry_friction_multiplier",
    "cold": "cold_friction_multiplier",
    "light_rain": "light_rain_friction_multiplier",
    "heavy_rain": "heavy_rain_friction_multiplier",
}


def load_config(path):
    with open(path, "r") as fh:
        return json.load(fh)


def corner_max_speed(friction, radius, crawl, v_max, margin=SAFETY_MARGIN):
    v = math.sqrt(friction * G * radius) + crawl
    v = min(v, v_max) - margin
    return max(v, crawl)


def group_corners(segments):
    groups, i, n = [], 0, len(segments)
    while i < n:
        if segments[i]["type"] == "corner":
            j = i
            while j < n and segments[j]["type"] == "corner":
                j += 1
            groups.append(list(range(i, j)))
            i = j
        else:
            i += 1
    return groups


def compute_corner_speeds(segments, friction, crawl, v_max):
    n = len(segments)
    cs = [None] * n
    for g in group_corners(segments):
        caps = [corner_max_speed(friction, segments[k]["radius_m"], crawl, v_max) for k in g]
        s = min(caps)
        for k in g:
            cs[k] = s
    return cs


def fuel_for_phase(v_in, v_out, distance):
    v_avg = (v_in + v_out) / 2
    return (K_BASE + K_DRAG * v_avg ** 2) * distance


def best_tyre_for_weather(tyres, weather_cond):
    key = WEATHER_KEYS[weather_cond]
    best_name, best_friction = None, -1
    for name, props in tyres.items():
        fr = props["life_span"] * props[key]
        if fr > best_friction:
            best_friction = fr
            best_name = name
    return best_name, best_friction


def tyre_set_id(available_sets, compound):
    for s in available_sets:
        if s["compound"].lower() == compound.lower():
            return s["ids"][0]
    raise ValueError(f"No tyre set available for compound {compound}")


def simulate_lap(segments, corner_speed, v_in, v_max, accel, brake):
    """
    Returns (time, fuel_used, exit_speed, lap_actions_list)
    with accel/brake already adjusted for current weather.
    """
    v = v_in
    total_t = 0.0
    total_f = 0.0
    lap_segs = []
    n = len(segments)

    for i, seg in enumerate(segments):
        if seg["type"] == "corner":
            # Constant speed through corner
            cs = corner_speed[i]
            total_t += seg["length_m"] / cs
            total_f += fuel_for_phase(cs, cs, seg["length_m"])
            v = cs
            lap_segs.append({"id": seg["id"], "type": "corner"})
            continue

        # Straight: find next corner's speed (within this lap, or wrap around)
        next_cs = None
        for j in range(i + 1, n):
            if segments[j]["type"] == "corner":
                next_cs = corner_speed[j]
                break
        if next_cs is None:
            # No more corners this lap — wrap around to start
            for j in range(0, i):
                if segments[j]["type"] == "corner":
                    next_cs = corner_speed[j]
                    break
        if next_cs is None:
            # Track has no corners at all (shouldn't happen)
            next_cs = v_max

        length = seg["length_m"]
        v0 = v
        target = v_max

        # Phase math:
        # If v0 >= target: just cruise at v0 then brake.
        if v0 >= target:
            d_brake = (v0 ** 2 - next_cs ** 2) / (2 * brake)
            d_cruise = length - d_brake
            if d_cruise < 0:
                # can't even fit the brake; panic-brake the whole length
                d_brake = length
                v_f = math.sqrt(max(v0 ** 2 - 2 * brake * length, next_cs ** 2))
                total_t += (v0 - v_f) / brake
                # no fuel during brake
                v = v_f
                chosen_target = v0
                actual_brake_start = length
            else:
                total_t += d_cruise / v0 + (v0 - next_cs) / brake
                total_f += fuel_for_phase(v0, v0, d_cruise)  # cruise fuel
                # no brake fuel
                v = next_cs
                chosen_target = v0  # spec: target<v0 means hold v0
                actual_brake_start = d_brake
        else:
            d_accel = (target ** 2 - v0 ** 2) / (2 * accel)
            d_brake = (target ** 2 - next_cs ** 2) / (2 * brake)
            if d_accel + d_brake <= length:
                d_cruise = length - d_accel - d_brake
                total_t += (target - v0) / accel + d_cruise / target + (target - next_cs) / brake
                total_f += fuel_for_phase(v0, target, d_accel)
                total_f += fuel_for_phase(target, target, d_cruise)
                # no brake fuel
                v = next_cs
                chosen_target = target
                actual_brake_start = d_brake
            else:
                # Peak below target
                lhs = 1 / (2 * accel) + 1 / (2 * brake)
                rhs = length + v0 ** 2 / (2 * accel) + next_cs ** 2 / (2 * brake)
                v_peak = math.sqrt(max(rhs / lhs, next_cs ** 2))
                v_peak = min(v_peak, target)
                d_a = (v_peak ** 2 - v0 ** 2) / (2 * accel)
                total_t += (v_peak - v0) / accel + (v_peak - next_cs) / brake
                total_f += fuel_for_phase(v0, v_peak, d_a)
                # no brake fuel
                v = next_cs
                chosen_target = v_peak
                actual_brake_start = (v_peak ** 2 - next_cs ** 2) / (2 * brake)

        lap_segs.append({
            "id": seg["id"],
            "type": "straight",
            "target_m/s": round(chosen_target, 2),
            "brake_start_m_before_next": math.ceil(actual_brake_start * 100) / 100,
        })

    return total_t, total_f, v, lap_segs


def weather_at_time(weather_conditions, starting_id, t, cycle_duration):
    """Return the weather_condition dict that is active at time t."""
    # Fast-forward to find where starting_id begins.
    # Build the ordered cycle starting at starting_id.
    n = len(weather_conditions)
    start_idx = None
    for i, w in enumerate(weather_conditions):
        if w["id"] == starting_id:
            start_idx = i
            break
    if start_idx is None:
        start_idx = 0

    ordered = weather_conditions[start_idx:] + weather_conditions[:start_idx]

    # Wrap t into the cycle
    t_mod = t % cycle_duration
    accum = 0.0
    for w in ordered:
        if accum + w["duration_s"] > t_mod:
            return w
        accum += w["duration_s"]
    return ordered[-1]  # fallback


def solve(config):
    car = config["car"]
    race = config["race"]
    track = config["track"]
    tyres = config["tyres"]["properties"]
    available_sets = config["available_sets"]

    accel_base = car["accel_m/se2"]
    brake_base = car["brake_m/se2"]
    v_max = car["max_speed_m/s"]
    crawl = car["crawl_constant_m/s"]
    tank = car["fuel_tank_capacity_l"]
    initial_fuel = car["initial_fuel_l"]

    total_laps = race["laps"]
    pit_exit_speed = race["pit_exit_speed_m/s"]
    refuel_rate = race["pit_refuel_rate_l/s"]
    base_pit_time = race["base_pit_stop_time_s"]
    pit_tyre_swap_time = race["pit_tyre_swap_time_s"]
    time_ref = race["time_reference_s"]
    soft_cap = race["fuel_soft_cap_limit_l"]

    weather_conditions = config["weather"]["conditions"]
    starting_weather_id = race["starting_weather_condition_id"]
    cycle_duration = sum(w["duration_s"] for w in weather_conditions)

    segments = track["segments"]

    # ============ PASS 1: FIND WEATHER-MANDATED PITS ============
    # Simulate the race with NO pit stops for fuel (just tyre changes at weather
    # transitions). This gives us the timing of weather changes in lap-space.

    def simulate_race(pit_plan):
        """
        pit_plan: dict mapping lap_num -> {'tyre_id': int or None, 'refuel': float}
                  (end-of-lap pit actions)
        Returns: (total_time, total_fuel_used, lap_records)
        Each lap_record: {lap, segments, pit_block, lap_time, lap_fuel, weather_cond}
        """
        cumulative_time = 0.0
        total_fuel = 0.0
        current_fuel = initial_fuel
        start_weather = next(w for w in weather_conditions if w["id"] == starting_weather_id)
        current_tyre_name = best_tyre_for_weather(tyres, start_weather["condition"])[0]
        initial_tyre_id = tyre_set_id(available_sets, current_tyre_name)

        v_lap_start = 0.0
        lap_records = []

        for lap_num in range(1, total_laps + 1):
            # Weather at lap start
            w_now = weather_at_time(weather_conditions, starting_weather_id,
                                    cumulative_time, cycle_duration)
            friction = tyres[current_tyre_name]["life_span"] * \
                       tyres[current_tyre_name][WEATHER_KEYS[w_now["condition"]]]
            accel = accel_base * w_now["acceleration_multiplier"]
            brake = brake_base * w_now["deceleration_multiplier"]

            corner_speed = compute_corner_speeds(segments, friction, crawl, v_max)
            lap_time, lap_fuel, v_exit, lap_segs = simulate_lap(
                segments, corner_speed, v_lap_start, v_max, accel, brake)

            cumulative_time += lap_time
            current_fuel -= lap_fuel
            total_fuel += lap_fuel
            ran_out = current_fuel < 0

            # Apply pit plan if any
            pit_info = pit_plan.get(lap_num)
            pit_block = {"enter": False}
            if pit_info is not None and lap_num < total_laps:
                pit_block = {"enter": True}
                pit_time = base_pit_time
                if pit_info.get("tyre_id"):
                    pit_block["tyre_change_set_id"] = pit_info["tyre_id"]
                    pit_time += pit_tyre_swap_time
                    # find tyre name from id
                    for s in available_sets:
                        if pit_info["tyre_id"] in s["ids"]:
                            current_tyre_name = s["compound"]
                            break
                refuel = pit_info.get("refuel", 0.0)
                # Cap at tank capacity
                refuel = min(refuel, tank - current_fuel)
                refuel = max(0.0, refuel)
                if refuel > 0:
                    pit_block["fuel_refuel_amount_l"] = round(refuel, 2)
                    pit_time += refuel / refuel_rate
                    current_fuel += refuel
                cumulative_time += pit_time
                v_lap_start = pit_exit_speed
            else:
                v_lap_start = v_exit

            lap_records.append({
                "lap": lap_num,
                "segments": lap_segs,
                "pit": pit_block,
                "lap_time": lap_time,
                "lap_fuel": lap_fuel,
                "weather": w_now["condition"],
                "current_fuel": current_fuel,
                "ran_out": ran_out,
                "tyre_at_start": current_tyre_name,  # after pit this is new tyre
            })

        return cumulative_time, total_fuel, lap_records, initial_tyre_id

    # Pass 1: iterate until weather-pit plan converges
    # Start with empty plan, find pits; re-sim with those pits; find pits; repeat.
    prev_weather_pits = None
    current_tyre_plan_name = None  # for tracking initial tyre
    for _ in range(10):  # max 10 iterations
        # Simulate with current weather-pit plan (iter 0: empty plan)
        plan_for_sim = {}
        if prev_weather_pits is not None:
            plan_for_sim = {
                lap: {"tyre_id": tyre_set_id(available_sets, t), "refuel": 0}
                for lap, t in prev_weather_pits.items()
            }
        _, _, lap_records_iter, initial_tyre_id = simulate_race(plan_for_sim)

        # Find the tyre-swap-needed points based on this iteration's weather timing
        new_weather_pits = {}
        # Track what tyre we WOULD be on, lap-by-lap, starting from optimal-for-start
        tyre = best_tyre_for_weather(tyres, lap_records_iter[0]["weather"])[0]
        for rec in lap_records_iter:
            optimal = best_tyre_for_weather(tyres, rec["weather"])[0]
            if optimal != tyre:
                prev_lap = rec["lap"] - 1
                if 1 <= prev_lap < total_laps:
                    new_weather_pits[prev_lap] = optimal
                    tyre = optimal

        if new_weather_pits == prev_weather_pits:
            break
        prev_weather_pits = new_weather_pits

    weather_pits = prev_weather_pits or {}

    # ============ PASS 2: FUEL PLANNING ============
    # Given mandatory pit laps, allocate refuel amounts.
    # Strategy: at each mandatory pit, refuel enough to reach the NEXT mandatory pit.
    # If running out of fuel before any mandatory pit, insert a fuel-only pit.

    mandatory_pits = sorted(weather_pits.keys())
    # Fuel planning with mandatory pits only:
    # Segments of the race: [start, pit1, pit2, ..., end]
    # For each segment, compute fuel needed. If > tank, problem — need an extra fuel pit.

    # Estimate fuel per lap under current weather/tyre
    # Use our base simulation's per-lap fuel data.
    lap_fuel_estimate = [r["lap_fuel"] for r in lap_records_iter]

    def fuel_for_range(start_lap, end_lap):
        """Fuel consumed in laps [start_lap..end_lap] inclusive."""
        return sum(lap_fuel_estimate[i - 1] for i in range(start_lap, end_lap + 1))

    # Build pit plan with refuel amounts
    pit_plan = {}
    all_pit_laps = list(mandatory_pits)

    # Check if we need additional fuel-only pits between mandatory ones
    # Starting fuel = initial_fuel (150L)
    # After each pit, tank = min(tank, current + refuel)
    # We want to add pits to ensure fuel never runs out.

    # Simulate fuel consumption given all_pit_laps as pit laps, filling tank each time
    def check_fuel_ok(pit_laps_sorted):
        fuel = initial_fuel
        last_pit = 0
        for p in pit_laps_sorted + [total_laps]:
            # Fuel consumed from last_pit+1 to p (inclusive), or for non-last segments: last_pit+1 to p inclusive
            span_end = p
            span_start = last_pit + 1
            if span_start > total_laps:
                break
            cons = fuel_for_range(span_start, min(span_end, total_laps))
            fuel -= cons
            if fuel < -0.5:
                return False, p, fuel
            if p < total_laps:  # there's a pit at end of lap p
                fuel = tank  # fill tank
            last_pit = p
        return True, None, fuel

    # Add fuel-only pits as needed
    current_pit_laps = sorted(mandatory_pits)

    def plan_fuel_sim(pit_laps_sorted):
        """
        Given a list of pit laps (sorted), simulate fuel consumption assuming
        tank is filled at each pit. Return (ok, fail_lap, fuel_history).
        fuel_history[i] = fuel remaining at END of lap i+1 (before any pit that lap).
        """
        fuel = initial_fuel
        pit_set = set(pit_laps_sorted)
        fuel_history = []
        for lap_num in range(1, total_laps + 1):
            fuel -= lap_fuel_estimate[lap_num - 1]
            fuel_history.append(fuel)
            if fuel < -0.5:
                return False, lap_num, fuel_history
            if lap_num in pit_set and lap_num < total_laps:
                fuel = tank  # fill tank
        return True, None, fuel_history

    # Iteratively add pits to prevent running dry (use BUFFER for safety)
    FUEL_BUFFER = 5.0  # L — safety margin against post-pit-lap fuel spike
    def plan_fuel_sim_buf(pit_laps_sorted):
        fuel = initial_fuel
        pit_set = set(pit_laps_sorted)
        fuel_history = []
        for lap_num in range(1, total_laps + 1):
            # If this lap is post-pit (pit was end of prev lap), assume ~10% more fuel
            extra = 0
            if (lap_num - 1) in pit_set and lap_num > 1:
                extra = 0.3  # post-pit lap uses slightly more fuel
            fuel -= (lap_fuel_estimate[lap_num - 1] + extra)
            fuel_history.append(fuel)
            if fuel < FUEL_BUFFER:
                return False, lap_num, fuel_history
            if lap_num in pit_set and lap_num < total_laps:
                fuel = tank
        return True, None, fuel_history

    while True:
        ok, fail_lap, history = plan_fuel_sim_buf(current_pit_laps)
        if ok:
            break
        # Find latest lap before fail_lap that works when inserted
        insert_at = None
        for candidate in range(fail_lap - 1, 0, -1):
            if candidate in current_pit_laps:
                continue
            trial = sorted(current_pit_laps + [candidate])
            ok2, _, _ = plan_fuel_sim_buf(trial)
            if ok2:
                insert_at = candidate
                break
        if insert_at is None:
            for candidate in range(fail_lap - 1, 0, -1):
                if candidate not in current_pit_laps:
                    insert_at = candidate
                    break
        if insert_at is None:
            break
        current_pit_laps = sorted(current_pit_laps + [insert_at])

    all_pit_laps = sorted(current_pit_laps)

    # Now assign refuel amounts. Fill tank at each pit except the last, where
    # we refuel only what's needed for remaining laps.
    fuel = initial_fuel
    pit_plan = {}
    for i, p in enumerate(all_pit_laps):
        # Fuel used during laps [prev+1 .. p]
        prev = all_pit_laps[i - 1] if i > 0 else 0
        cons = fuel_for_range(prev + 1, p)
        fuel -= cons
        # Don't pit on the last lap
        if p >= total_laps:
            continue
        # Is this the last pit before race end?
        is_last_pit = (i == len(all_pit_laps) - 1) or (all_pit_laps[-1] >= total_laps)
        if is_last_pit:
            future_cons = fuel_for_range(p + 1, total_laps)
            needed = future_cons - fuel + 3.0  # 3L buffer
            refuel = max(0.0, min(needed, tank - fuel))
        else:
            # Not last — fill tank
            refuel = tank - fuel
        refuel = max(0.0, min(refuel, tank - fuel))  # safety cap
        fuel += refuel

        tyre_id = None
        if p in weather_pits:
            tyre_id = tyre_set_id(available_sets, weather_pits[p])
        pit_plan[p] = {"tyre_id": tyre_id, "refuel": refuel}

    # ============ FINAL SIMULATION ============
    total_time, total_fuel, lap_records_final, _ = simulate_race(pit_plan)

    # Build output
    output = {
        "initial_tyre_id": initial_tyre_id,
        "laps": [
            {
                "lap": r["lap"],
                "segments": r["segments"],
                "pit": r["pit"],
            }
            for r in lap_records_final
        ],
    }

    base_score = 500000 * (time_ref / total_time) ** 3
    fuel_bonus = -500000 * (1 - total_fuel / soft_cap) ** 2 + 500000

    # Count pit types
    n_tyre_pits = sum(1 for r in lap_records_final if r["pit"].get("enter") and "tyre_change_set_id" in r["pit"])
    n_fuel_only = sum(1 for r in lap_records_final if r["pit"].get("enter") and "tyre_change_set_id" not in r["pit"])

    # Weather summary per lap
    weather_by_lap = [(r["lap"], r["weather"], r["tyre_at_start"]) for r in lap_records_final]

    diag = {
        "total_time": total_time,
        "total_fuel_used": total_fuel,
        "base_score": base_score,
        "fuel_bonus": fuel_bonus,
        "final_score": base_score + fuel_bonus,
        "n_tyre_pits": n_tyre_pits,
        "n_fuel_pits": n_fuel_only,
        "total_pits": n_tyre_pits + n_fuel_only,
        "weather_by_lap": weather_by_lap,
        "initial_tyre_id": initial_tyre_id,
    }
    return output, diag


def wcond_accel_mult(conds, cname):
    for c in conds:
        if c["condition"] == cname:
            return c["acceleration_multiplier"]
    return 1.0


def wcond_brake_mult(conds, cname):
    for c in conds:
        if c["condition"] == cname:
            return c["deceleration_multiplier"]
    return 1.0


if __name__ == "__main__":
    cfg = load_config(os.path.join(os.path.dirname(__file__), "3.txt"))
    out, diag = solve(cfg)

    print(f"Total time: {diag['total_time']:.1f}s  (ref: {cfg['race']['time_reference_s']}s)")
    print(f"Total fuel used: {diag['total_fuel_used']:.1f}L  (soft cap: {cfg['race']['fuel_soft_cap_limit_l']}L)")
    print(f"Pit stops: {diag['total_pits']} total ({diag['n_tyre_pits']} tyre, {diag['n_fuel_pits']} fuel-only)")
    print(f"Base score: {diag['base_score']:,.0f}")
    print(f"Fuel bonus: {diag['fuel_bonus']:,.0f}")
    print(f"Final score: {diag['final_score']:,.0f}")

    # Weather distribution
    from collections import Counter
    weather_counts = Counter(w for _, w, _ in diag['weather_by_lap'])
    print(f"\nLaps per weather: {dict(weather_counts)}")

    # Report the pit plan
    print("\nPit plan:")
    for lap in out["laps"]:
        if lap["pit"]["enter"]:
            p = lap["pit"]
            bits = []
            if "tyre_change_set_id" in p:
                # lookup compound
                for s in cfg["available_sets"]:
                    if p["tyre_change_set_id"] in s["ids"]:
                        bits.append(f"→{s['compound']} (id {p['tyre_change_set_id']})")
                        break
            if "fuel_refuel_amount_l" in p:
                bits.append(f"+{p['fuel_refuel_amount_l']:.1f}L")
            print(f"  Lap {lap['lap']}: {', '.join(bits)}")

    out_path = Path("../submission_level3.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote: {out_path}")