"""
Entelect Grand Prix - Level 4 solver.

What's new vs Level 3:
- Tyre DEGRADATION now matters. Life span and friction drop as tyres wear.
- Blowout occurs when accumulated degradation >= life_span -> limp mode.
- Scoring now includes a tyre bonus:
    tyre_bonus = 100_000 * sum(final_degradation_per_set_used) - 50_000 * blowouts
  i.e. the more worn each set ends up, the more points (up to blowout).
- 9 tyre sets available (Soft x2, Med x2, Hard x1, Int x2, Wet x2).
- Monaco, 80 laps, 50 segments/lap, fuel soft cap 611 L.
- Weather cycle: 8 phases totalling 22_000 s.

Degradation formulas (PDF page 5):
  straight: rate * length * K_STRAIGHT                       K_STRAIGHT = 1.66e-5
  brake:    ((v0/100)^2 - (vf/100)^2) * K_BRAKING * rate     K_BRAKING  = 0.0398
  corner:   (speed^2 / radius) * K_CORNER * rate             K_CORNER   = 2.65e-4
  friction: (base - total_deg) * weather_mult

Base friction coefficient comes from the PDF table (NOT in the JSON):
  Soft 1.8, Medium 1.7, Hard 1.6, Intermediate 1.2, Wet 1.1

The PDF's worked example uses weather_mult = 1 for dry (inconsistent with the
per-compound dry_friction_multiplier in JSON). Two friction modes are provided:
  FRICTION_MODE = 'pdf_literal'  -> (base - deg) * weather_mult
  FRICTION_MODE = 'lvl3_compat'  -> (1 - deg) * weather_mult  (what L3 used)
Start with lvl3_compat since that's the model our L3 submission scored under.

Strategy:
- Per-lap sim tracks tyre degradation across segments using the formulas.
- Choose tyres based on (a) weather friction and (b) degradation survival.
- Force a pit before a blowout (degradation crossing life_span).
- Tyre-bonus aware: we want each set to be USED (accumulate deg), so prefer
  to rotate tyres and leave each set near-worn when pitting.
"""

import json
import math
import os
from pathlib import Path
from collections import defaultdict

# ---------- Physics constants ----------
G = 9.8
SAFETY_MARGIN = 1.0          # m/s below max corner speed — stay clear of judge's limit
K_BASE = 0.0005              # L/m base fuel
K_DRAG = 0.0000000015        # L/m per (m/s)^2 drag-fuel
K_STRAIGHT = 0.0000166
K_BRAKING = 0.0398
K_CORNER = 0.000265

# ---------- Base friction coefficients from PDF page 4 ----------
BASE_FRICTION = {
    "Soft": 1.8,
    "Medium": 1.7,
    "Hard": 1.6,
    "Intermediate": 1.2,
    "Wet": 1.1,
}

# Which model to use. 'lvl3_compat' is our known-good model; 'pdf_literal' is
# the strict PDF reading. Flip this if the judge reports different corner speeds.
FRICTION_MODE = "lvl3_compat"

# How close to life_span to allow before forcing a pit.
BLOWOUT_SAFETY = 0.18        # keep at least 18% headroom before forcing a pit
WEAR_CEILING = 0.82          # max deg per set we aim to reach before swapping
WEAR_TARGET_LATE = 0.75      # in late-race finisher, pit earlier to leave margin

WEATHER_FRICTION_KEYS = {
    "dry": "dry_friction_multiplier",
    "cold": "cold_friction_multiplier",
    "light_rain": "light_rain_friction_multiplier",
    "heavy_rain": "heavy_rain_friction_multiplier",
}
WEATHER_DEG_KEYS = {
    "dry": "dry_degradation",
    "cold": "cold_degradation",
    "light_rain": "light_rain_degradation",
    "heavy_rain": "heavy_rain_degradation",
}


# ---------- Helpers ----------
def load_config(path):
    with open(path, "r") as fh:
        return json.load(fh)


def compute_friction(compound, deg, tyre_props, weather_cond):
    mult = tyre_props[WEATHER_FRICTION_KEYS[weather_cond]]
    if FRICTION_MODE == "pdf_literal":
        base = BASE_FRICTION[compound]
        return max((base - deg), 0.0) * mult
    else:  # lvl3_compat
        return max((1.0 - deg), 0.0) * mult


def corner_max_speed(friction, radius, crawl, v_max, margin=SAFETY_MARGIN):
    v = math.sqrt(max(friction, 0.0) * G * radius)
    v = min(v, v_max) - margin
    return max(v, crawl)


def fuel_for_phase(v_in, v_out, distance):
    v_avg = (v_in + v_out) / 2
    return (K_BASE + K_DRAG * v_avg ** 2) * distance


def deg_straight_cruise(length, rate):
    """Degradation over a pure-cruise straight section (not braking)."""
    return rate * length * K_STRAIGHT


def deg_braking(v0, vf, rate):
    return ((v0 / 100.0) ** 2 - (vf / 100.0) ** 2) * K_BRAKING * rate


def deg_corner(speed, radius, length, rate):
    # PDF formula: K_CORNER * speed^2 / radius * rate
    # Interpreting this as degradation for the corner as a whole (not per-metre).
    return K_CORNER * (speed ** 2) / radius * rate


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


def compute_corner_speeds(segments, compound, deg, tyre_props, weather_cond, crawl, v_max):
    """Corner speeds based on current degradation-adjusted friction.
    Consecutive corners take the min speed of the group (can't accelerate in a corner).
    """
    friction = compute_friction(compound, deg, tyre_props, weather_cond)
    n = len(segments)
    cs = [None] * n
    for g in group_corners(segments):
        caps = [corner_max_speed(friction, segments[k]["radius_m"], crawl, v_max) for k in g]
        s = min(caps)
        for k in g:
            cs[k] = s
    return cs


def weather_at_time(weather_conditions, starting_id, t):
    """Return active weather condition at race time t, cycling if needed."""
    start_idx = next((i for i, w in enumerate(weather_conditions) if w["id"] == starting_id), 0)
    ordered = weather_conditions[start_idx:] + weather_conditions[:start_idx]
    cycle = sum(w["duration_s"] for w in ordered)
    if cycle <= 0:
        return ordered[0]
    t_mod = t % cycle
    accum = 0.0
    for w in ordered:
        if accum + w["duration_s"] > t_mod:
            return w
        accum += w["duration_s"]
    return ordered[-1]


def compound_for_set_id(available_sets, set_id):
    for s in available_sets:
        if set_id in s["ids"]:
            return s["compound"]
    return None


def available_set_ids_for_compound(available_sets, compound):
    for s in available_sets:
        if s["compound"].lower() == compound.lower():
            return list(s["ids"])
    return []


def best_tyre_for_weather(tyre_props, weather_cond):
    key = WEATHER_FRICTION_KEYS[weather_cond]
    best_name, best_mult = None, -1.0
    for name, props in tyre_props.items():
        m = props[key]
        if m > best_mult:
            best_mult, best_name = m, name
    return best_name, best_mult


def build_corner_groups(segments):
    """
    Returns a dict mapping segment_index -> list of all segment indices in its
    consecutive corner group. Mirrors the lvl1 group_corners logic so that
    back-to-back corners are all constrained to the same (minimum) speed.
    """
    n = len(segments)
    corner_group_of = {}
    i = 0
    while i < n:
        if segments[i]["type"] == "corner":
            j = i
            while j < n and segments[j]["type"] == "corner":
                j += 1
            group = list(range(i, j))
            for k in group:
                corner_group_of[k] = group
            i = j
        else:
            i += 1
    return corner_group_of


def group_min_corner_speed(group_idxs, segments, friction, crawl, v_max):
    """Minimum safe speed across all corners in a consecutive group."""
    caps = [corner_max_speed(friction, segments[k]["radius_m"], crawl, v_max)
            for k in group_idxs]
    return min(caps)


def next_group_min_speed(from_idx, segments, corner_group_of, friction, crawl, v_max):
    """
    Starting after from_idx, find the next corner group and return its min speed.
    Wraps around the lap if necessary. Mirrors lvl1's look-ahead logic.
    """
    n = len(segments)
    for j in range(from_idx + 1, n):
        if segments[j]["type"] == "corner":
            return group_min_corner_speed(corner_group_of[j], segments, friction, crawl, v_max)
    # Wrap around
    for j in range(0, from_idx + 1):
        if segments[j]["type"] == "corner":
            return group_min_corner_speed(corner_group_of[j], segments, friction, crawl, v_max)
    return v_max  # no corners on track at all


def simulate_lap(segments, compound, start_deg, tyre_props, weather_cond,
                 v_in, v_max, accel, brake, crawl, life_span):
    """Simulate one lap with degradation accumulating segment by segment.
    Returns (time, fuel_used, exit_speed, end_deg, lap_segs, blew_out)
    where blew_out indicates a blowout during the lap.

    Consecutive corners are treated as a group (minimum speed of the group),
    exactly as in lvl1, so we never crash into a tighter corner after a
    looser one in the same sequence.
    """
    deg = start_deg
    v = v_in
    total_t = 0.0
    total_f = 0.0
    lap_segs = []
    n = len(segments)
    deg_rate = tyre_props[WEATHER_DEG_KEYS[weather_cond]]
    blew_out = False

    # Precompute which group each corner belongs to (static track structure).
    corner_group_of = build_corner_groups(segments)

    for i, seg in enumerate(segments):
        # Refresh friction each segment as degradation grows.
        friction = compute_friction(compound, deg, tyre_props, weather_cond)

        if seg["type"] == "corner":
            # Use group minimum speed — crucial for back-to-back corners.
            group = corner_group_of[i]
            cs = group_min_corner_speed(group, segments, friction, crawl, v_max)
            use_speed = min(v, cs)
            total_t += seg["length_m"] / use_speed
            total_f += fuel_for_phase(use_speed, use_speed, seg["length_m"])
            deg += deg_corner(use_speed, seg["radius_m"], seg["length_m"], deg_rate)
            v = use_speed
            lap_segs.append({"id": seg["id"], "type": "corner"})
            if deg >= life_span:
                blew_out = True
                break
            continue

        # Straight: look ahead to the NEXT CORNER GROUP and use its min speed
        # so we don't arrive too fast at the second (tighter) corner in a group.
        next_cs = next_group_min_speed(i, segments, corner_group_of, friction, crawl, v_max)

        length = seg["length_m"]
        v0 = v
        target = v_max

        # Compute accel/cruise/brake phases.
        if v0 >= target:
            d_brake = (v0 ** 2 - next_cs ** 2) / (2 * brake)
            d_cruise = length - d_brake
            if d_cruise < 0:
                d_brake = length
                v_f = math.sqrt(max(v0 ** 2 - 2 * brake * length, next_cs ** 2))
                total_t += (v0 - v_f) / brake
                deg += deg_braking(v0, v_f, deg_rate)
                v = v_f
                chosen_target = v0
                actual_brake_start = length
            else:
                total_t += d_cruise / v0 + (v0 - next_cs) / brake
                total_f += fuel_for_phase(v0, v0, d_cruise)
                deg += deg_straight_cruise(d_cruise, deg_rate)
                deg += deg_braking(v0, next_cs, deg_rate)
                v = next_cs
                chosen_target = v0
                actual_brake_start = d_brake
        else:
            d_accel = (target ** 2 - v0 ** 2) / (2 * accel)
            d_brake = (target ** 2 - next_cs ** 2) / (2 * brake)
            if d_accel + d_brake <= length:
                d_cruise = length - d_accel - d_brake
                total_t += (target - v0) / accel + d_cruise / target + (target - next_cs) / brake
                total_f += fuel_for_phase(v0, target, d_accel)
                total_f += fuel_for_phase(target, target, d_cruise)
                deg += deg_straight_cruise(d_accel + d_cruise, deg_rate)
                deg += deg_braking(target, next_cs, deg_rate)
                v = next_cs
                chosen_target = target
                actual_brake_start = d_brake
            else:
                lhs = 1 / (2 * accel) + 1 / (2 * brake)
                rhs = length + v0 ** 2 / (2 * accel) + next_cs ** 2 / (2 * brake)
                v_peak = math.sqrt(max(rhs / lhs, next_cs ** 2))
                v_peak = min(v_peak, target)
                d_a = (v_peak ** 2 - v0 ** 2) / (2 * accel)
                total_t += (v_peak - v0) / accel + (v_peak - next_cs) / brake
                total_f += fuel_for_phase(v0, v_peak, d_a)
                deg += deg_straight_cruise(d_a, deg_rate)
                deg += deg_braking(v_peak, next_cs, deg_rate)
                v = next_cs
                chosen_target = v_peak
                actual_brake_start = (v_peak ** 2 - next_cs ** 2) / (2 * brake)

        lap_segs.append({
            "id": seg["id"],
            "type": "straight",
            "target_m/s": round(chosen_target, 2),
            "brake_start_m_before_next": math.ceil(actual_brake_start * 100) / 100,
        })

        if deg >= life_span:
            blew_out = True
            break

    return total_t, total_f, v, deg, lap_segs, blew_out


# ---------- Main solver ----------
def solve(config):
    car = config["car"]
    race = config["race"]
    track = config["track"]
    tyre_props = config["tyres"]["properties"]
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

    segments = track["segments"]

    # Life span is 1 per PDF; we treat degradation as blowing out when deg >= life_span.
    life_span = 1.0

    # ---- Build pool of available tyre sets (each id usable once at a time) ----
    # We track which sets have been mounted and their accumulated degradation.
    # Sets can be re-mounted later with their existing degradation (not reset).
    # Initial tyre: best fresh choice for starting weather.
    start_weather = next(w for w in weather_conditions if w["id"] == starting_weather_id)
    start_compound, _ = best_tyre_for_weather(tyre_props, start_weather["condition"])
    start_ids = available_set_ids_for_compound(available_sets, start_compound)
    if not start_ids:
        # fallback to any set
        start_ids = [available_sets[0]["ids"][0]]
        start_compound = available_sets[0]["compound"]
    initial_tyre_id = start_ids[0]

    def simulate_race(pit_plan):
        """
        pit_plan: {lap: {'tyre_id': int or None, 'refuel': float}}
        Runs full race sim. Handles limp-mode persistence after a blowout.
        Blowouts are counted per SET (once per blown set), not per lap.
        Returns (time, fuel_used, lap_records, sets_used_state, blowouts)
        """
        cumulative_time = 0.0
        total_fuel_used = 0.0
        current_fuel = initial_fuel

        set_deg = defaultdict(float)
        mounted_ids = {initial_tyre_id}
        blown_sets = set()                     # set_ids that have blown out
        current_set_id = initial_tyre_id
        current_compound = start_compound
        limp_mode = False                      # persists until a pit

        total_lap_len = sum(s["length_m"] for s in segments)
        limp_speed = car["limp_constant_m/s"]

        v_lap_start = 0.0
        lap_records = []

        for lap_num in range(1, total_laps + 1):
            w_now = weather_at_time(weather_conditions, starting_weather_id, cumulative_time)
            cond = w_now["condition"]
            accel = accel_base * w_now["acceleration_multiplier"]
            brake = brake_base * w_now["deceleration_multiplier"]

            blew_this_lap = False
            if limp_mode:
                # Whole lap at limp speed; no further degradation (tyre already dead)
                lap_time = total_lap_len / limp_speed
                lap_fuel = K_BASE * total_lap_len
                v_exit = limp_speed
                end_deg = set_deg[current_set_id]
                # Build placeholder lap_segs for JSON — straight/corners with target=limp
                lap_segs = []
                for seg in segments:
                    if seg["type"] == "straight":
                        lap_segs.append({
                            "id": seg["id"], "type": "straight",
                            "target_m/s": limp_speed,
                            "brake_start_m_before_next": 0,
                        })
                    else:
                        lap_segs.append({"id": seg["id"], "type": "corner"})
            else:
                start_deg = set_deg[current_set_id]
                lap_time, lap_fuel, v_exit, end_deg, lap_segs, blew_this_lap = simulate_lap(
                    segments, current_compound, start_deg, tyre_props[current_compound],
                    cond, v_lap_start, v_max, accel, brake, crawl, life_span)

                if blew_this_lap:
                    # add remainder of lap in limp mode (rough estimate: half lap remaining)
                    remaining = total_lap_len * 0.5
                    lap_time += remaining / limp_speed
                    lap_fuel += K_BASE * remaining
                    v_exit = limp_speed
                    limp_mode = True
                    if current_set_id not in blown_sets:
                        blown_sets.add(current_set_id)

            cumulative_time += lap_time
            current_fuel -= lap_fuel
            total_fuel_used += lap_fuel
            set_deg[current_set_id] = min(end_deg, life_span)

            pit_info = pit_plan.get(lap_num)
            pit_block = {"enter": False}
            if pit_info is not None and lap_num < total_laps:
                pit_block = {"enter": True}
                pit_time = base_pit_time
                new_tyre_id = pit_info.get("tyre_id")
                if new_tyre_id:
                    pit_block["tyre_change_set_id"] = new_tyre_id
                    pit_time += pit_tyre_swap_time
                    current_set_id = new_tyre_id
                    current_compound = compound_for_set_id(available_sets, new_tyre_id)
                    mounted_ids.add(new_tyre_id)
                    limp_mode = False  # fresh tyre clears limp
                refuel = pit_info.get("refuel", 0.0)
                refuel = max(0.0, min(refuel, tank - current_fuel))
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
                "weather": cond,
                "current_fuel": current_fuel,
                "tyre_at_end": current_compound,
                "set_id": current_set_id,
                "set_deg": set_deg[current_set_id],
                "blew_out": blew_this_lap,
                "limp": limp_mode,
            })

        sets_used_state = {sid: set_deg[sid] for sid in mounted_ids}
        return cumulative_time, total_fuel_used, lap_records, sets_used_state, len(blown_sets)

    # ---- PASS 1: wear-budget-aware plan ----
    # Strategy: estimate per-lap wear for each compound*weather, then plan which
    # sets to use for which laps, ensuring:
    #  - Total wear across each set <= ~0.9 (safe) summing to near 9 * 0.9 = 8.1 ideally
    #  - Last set can physically cover the final laps
    #  - Weather-matched compounds preferred for friction where affordable

    def estimate_per_lap_wear(compound, weather_cond):
        """Simulate ONE lap with fresh tyre in that weather to get baseline wear."""
        accel = accel_base * next(w["acceleration_multiplier"] for w in weather_conditions if w["condition"] == weather_cond)
        brake = brake_base * next(w["deceleration_multiplier"] for w in weather_conditions if w["condition"] == weather_cond)
        _, _, _, end_deg, _, _ = simulate_lap(
            segments, compound, 0.0, tyre_props[compound],
            weather_cond, 60.0, v_max, accel, brake, crawl, life_span)
        return end_deg

    # Build: weather sequence per lap (using no-pit baseline)
    _, _, baseline_recs, _, _ = simulate_race({})
    weather_per_lap = [r["weather"] for r in baseline_recs]  # length = total_laps

    # Per-(compound, weather) wear lookup
    wear_table = {}
    for comp in tyre_props.keys():
        for cond in {"cold", "light_rain", "heavy_rain", "dry"}:
            wear_table[(comp, cond)] = estimate_per_lap_wear(comp, cond)

    def wear_cost(compound, lap_range):
        """Projected total wear if we put `compound` on for laps in lap_range (inclusive indices)."""
        return sum(wear_table[(compound, weather_per_lap[l - 1])] for l in lap_range)

    # Greedy planner: walk through laps, assign tyre set until it hits 0.85 budget.
    # Prefer the compound best-matched for the current weather, falling back to more
    # durable compounds if we're running low.
    WEAR_TARGET = 0.85  # aim to swap at this deg (leaves safety margin)

    def most_durable_compound_available(sets_remaining):
        """Compound with lowest average wear rate that still has a set available."""
        best_comp, best_rate = None, 1e9
        for comp, ids in sets_remaining.items():
            if not ids:
                continue
            # average wear rate across weather conditions as a heuristic
            avg = sum(wear_table[(comp, c)] for c in ("cold", "light_rain", "heavy_rain", "dry")) / 4
            if avg < best_rate:
                best_rate = avg
                best_comp = comp
        return best_comp

    def can_finish_race(compound, lap_range, budget=0.90):
        """Can `compound` physically finish `lap_range` without blowing?"""
        return wear_cost(compound, lap_range) <= budget

    def pick_safe_compound(sets_remaining, start_lap, must_finish=False):
        """Pick a compound. If must_finish, the compound must be able to cover start_lap..total_laps without blowing alone.
        Preference: best for the start_lap weather, else most durable."""
        cond = weather_per_lap[start_lap - 1]
        target = best_tyre_for_weather(tyre_props, cond)[0]
        prefs = [target, "Hard", "Medium", "Soft", "Intermediate", "Wet"]
        for p in prefs:
            if sets_remaining.get(p):
                if not must_finish:
                    return p
                lap_range = range(start_lap, total_laps + 1)
                if can_finish_race(p, lap_range, budget=0.92):
                    return p
        # fallback: most-durable even if won't finish
        return most_durable_compound_available(sets_remaining)

    def any_compound_can_finish(sets_remaining, from_lap, current_compound, current_wear):
        """True if there's a feasible way to finish the race without blowing any tyre.
        Checks both (a) stay on current and (b) swap to any remaining compound."""
        if from_lap > total_laps:
            return True
        lap_range = range(from_lap, total_laps + 1)
        # Option A: stay on current compound — needs (cur_wear + projected) <= 0.95
        cur_proj = wear_cost(current_compound, lap_range)
        if current_wear + cur_proj <= 0.95:
            return True
        # Option B: swap once now and that new set finishes
        for comp, ids in sets_remaining.items():
            if not ids:
                continue
            if wear_cost(comp, lap_range) <= 0.92:
                return True
        # Option C: swap now, use this to some mid-point, then swap again
        # (heuristic — try 1 or 2 more swaps in the remaining laps)
        # We don't recurse deeply; instead just test if SOMETHING durable is left.
        total_rem_budget = sum(0.92 for ids in sets_remaining.values() for _ in ids)
        # rough: if the sum of budgets exceeds total wear demand using most-durable compound
        durable = most_durable_compound_available(sets_remaining)
        if durable and wear_cost(durable, lap_range) <= total_rem_budget + (0.95 - current_wear):
            return True
        return False

    def build_wear_plan():
        plan = {}
        sets_remaining = {s["compound"]: list(s["ids"]) for s in available_sets}
        if start_compound in sets_remaining and initial_tyre_id in sets_remaining[start_compound]:
            sets_remaining[start_compound].remove(initial_tyre_id)

        current_compound = start_compound
        wear = 0.0
        lap = 1
        last_pit_lap = 0

        while lap <= total_laps:
            cond = weather_per_lap[lap - 1]
            step_wear = wear_table[(current_compound, cond)]
            laps_remaining = total_laps - lap + 1
            late_race = laps_remaining <= 18

            wear_budget = WEAR_TARGET_LATE if late_race else WEAR_CEILING

            # DURABILITY swap: wear would go over safe ceiling next
            if wear + step_wear > wear_budget and lap < total_laps:
                # Try to stay on current if it can finish safely
                if can_finish_race(current_compound, range(lap, total_laps + 1),
                                    budget=0.95 - wear):
                    wear += step_wear
                    lap += 1
                    continue

                pit_lap = lap - 1
                if pit_lap < 1:
                    pit_lap = 1
                if pit_lap - last_pit_lap < 2 and last_pit_lap > 0:
                    wear += step_wear
                    lap += 1
                    continue

                # Choose a compound that can feasibly finish the race from here.
                # Prefer weather-match first, but always require finishability.
                target = best_tyre_for_weather(tyre_props, cond)[0]
                # Enumerate preferences: weather-best, then durability order
                prefs = [target]
                # Add compounds sorted by durability (lowest avg wear rate)
                durability_order = sorted(
                    tyre_props.keys(),
                    key=lambda c: sum(wear_table[(c, cc)] for cc in ("cold","light_rain","heavy_rain","dry")) / 4
                )
                for c in durability_order:
                    if c not in prefs:
                        prefs.append(c)

                new_comp = None
                for p in prefs:
                    if sets_remaining.get(p):
                        # Will this compound by itself finish the race?
                        if can_finish_race(p, range(lap, total_laps + 1), budget=0.92):
                            new_comp = p
                            break
                # If no single compound can finish alone, accept the best compound we have
                # AND hope a later pit helps (but we need to verify any_compound_can_finish).
                if new_comp is None:
                    for p in prefs:
                        if sets_remaining.get(p):
                            new_comp = p
                            break

                if new_comp is None:
                    wear += step_wear
                    lap += 1
                    continue

                # Final feasibility: will we blow somewhere?  If current_compound can ALSO
                # not finish, we're in trouble — pit anyway.
                new_id = sets_remaining[new_comp].pop(0)
                if pit_lap >= total_laps:
                    break
                plan[pit_lap] = {"tyre_id": new_id, "refuel": 0.0}
                last_pit_lap = pit_lap
                current_compound = new_comp
                wear = 0.0
                continue

            # OPPORTUNISTIC weather swap
            best_now = best_tyre_for_weather(tyre_props, cond)[0]
            if (best_now != current_compound
                    and sets_remaining.get(best_now)
                    and (lap - last_pit_lap) >= 3
                    and not late_race):
                run = 1
                while lap + run - 1 < total_laps and weather_per_lap[lap + run - 1] == cond:
                    run += 1
                cur_mult = tyre_props[current_compound][WEATHER_FRICTION_KEYS[cond]]
                new_mult = tyre_props[best_now][WEATHER_FRICTION_KEYS[cond]]
                # Require substantial gain, long run, current set already decently worn
                if (new_mult > cur_mult * 1.22 and run >= 10 and wear >= 0.45):
                    pit_lap = lap - 1
                    if (pit_lap >= 1 and pit_lap < total_laps
                            and (pit_lap - last_pit_lap) >= 2):
                        # Feasibility: after this swap, can we still finish?
                        trial_sets = {c: list(ids) for c, ids in sets_remaining.items()}
                        trial_id = trial_sets[best_now].pop(0)
                        if any_compound_can_finish(trial_sets, lap, best_now, 0.0):
                            sets_remaining[best_now].remove(trial_id)
                            plan[pit_lap] = {"tyre_id": trial_id, "refuel": 0.0}
                            last_pit_lap = pit_lap
                            current_compound = best_now
                            wear = 0.0
                            continue

            wear += step_wear
            lap += 1

        return plan

    plan_v1 = build_wear_plan()

    # Iterate: after planning, re-simulate to refresh weather_per_lap (since pits shift times)
    for _iter in range(6):
        _, _, recs_iter, _, _ = simulate_race(plan_v1)
        new_weather = [r["weather"] for r in recs_iter]
        if new_weather == weather_per_lap:
            break
        weather_per_lap = new_weather
        # rebuild wear_table-free plan with updated weather
        plan_v1 = build_wear_plan()

    # ---- PASS 2: add safety pits for blowout protection ----
    # Walk the sim; whenever current set's deg would cross (life_span - safety),
    # insert a pit on the latest feasible lap to swap to a fresh set.
    def insert_safety_pits(plan):
        plan = dict(plan)
        for _ in range(40):
            _, _, recs, sets_state, blowouts = simulate_race(plan)

            # Find first lap where current set's deg is too high (near blow)
            threshold = life_span - BLOWOUT_SAFETY
            problem_lap = None
            for r in recs:
                if r["set_deg"] >= threshold and r["lap"] < total_laps:
                    problem_lap = r["lap"]
                    problem_set = r["set_id"]
                    break
            if problem_lap is None:
                break

            # Determine what sets are still available (not yet used in plan or as initial)
            used_ids = {initial_tyre_id}
            for p in plan.values():
                if p.get("tyre_id"):
                    used_ids.add(p["tyre_id"])
            available_now = {s["compound"]: [i for i in s["ids"] if i not in used_ids]
                             for s in available_sets}

            # Pick a fresh set: prefer best compound for weather at problem_lap
            w = recs[problem_lap - 1]["weather"] if problem_lap - 1 < len(recs) else "dry"
            pref = best_tyre_for_weather(tyre_props, w)[0]
            order = [pref, "Hard", "Medium", "Soft", "Intermediate", "Wet"]
            new_id = None
            for comp in order:
                if available_now.get(comp):
                    new_id = available_now[comp][0]
                    break
            if new_id is None:
                # No fresh sets left anywhere — we'll just have to eat the blowout
                break

            # Insert pit on problem_lap (swap before it would blow during that lap)
            # Actually pit on lap BEFORE the problem lap so the new tyre runs that lap.
            pit_on = max(1, problem_lap - 1)
            # If there's already a pit on pit_on, upgrade it with a tyre swap
            if pit_on in plan:
                plan[pit_on]["tyre_id"] = new_id
            else:
                plan[pit_on] = {"tyre_id": new_id, "refuel": 0.0}
        return plan

    plan_v2 = insert_safety_pits(plan_v1)

    # ---- PASS 3: fuel planning ----
    # Given tyre-pit laps, fill tank as needed. Add extra fuel-only pits if required.
    def add_fuel_pits(plan):
        _, _, recs, _, _ = simulate_race(plan)
        lap_fuel = [r["lap_fuel"] for r in recs]
        fuel_buffer = 5.0

        def check(pit_laps_sorted):
            fuel = initial_fuel
            pit_set = set(pit_laps_sorted)
            for lap_num in range(1, total_laps + 1):
                fuel -= lap_fuel[lap_num - 1]
                if fuel < fuel_buffer and lap_num < total_laps:
                    return False, lap_num
                if lap_num in pit_set and lap_num < total_laps:
                    fuel = tank
            return True, None

        plan = dict(plan)
        current_pit_laps = sorted(plan.keys())
        for _ in range(30):
            ok, fail_lap = check(current_pit_laps)
            if ok:
                break
            # Insert a fuel-only pit at the latest feasible lap before fail
            inserted = False
            for cand in range(fail_lap - 1, 0, -1):
                if cand in current_pit_laps:
                    continue
                ok2, _ = check(sorted(current_pit_laps + [cand]))
                if ok2:
                    plan[cand] = {"tyre_id": None, "refuel": 0.0}
                    current_pit_laps = sorted(current_pit_laps + [cand])
                    inserted = True
                    break
            if not inserted:
                # insert latest empty lap as last resort
                for cand in range(fail_lap - 1, 0, -1):
                    if cand not in current_pit_laps:
                        plan[cand] = {"tyre_id": None, "refuel": 0.0}
                        current_pit_laps = sorted(current_pit_laps + [cand])
                        break
                else:
                    break

        # Assign refuel amounts
        _, _, recs, _, _ = simulate_race(plan)
        lap_fuel = [r["lap_fuel"] for r in recs]
        fuel = initial_fuel
        all_pits = sorted(plan.keys())
        for i, p in enumerate(all_pits):
            prev = all_pits[i - 1] if i > 0 else 0
            cons = sum(lap_fuel[j - 1] for j in range(prev + 1, p + 1))
            fuel -= cons
            if p >= total_laps:
                continue
            is_last = (i == len(all_pits) - 1)
            if is_last:
                future = sum(lap_fuel[j - 1] for j in range(p + 1, total_laps + 1))
                needed = future - fuel + 3.0
                refuel = max(0.0, min(needed, tank - fuel))
            else:
                # Fill to aim for 611 L soft cap by race end (L4 fuel bonus)
                # For simplicity: fill tank at non-last pits.
                refuel = tank - fuel
            refuel = max(0.0, min(refuel, tank - fuel))
            fuel += refuel
            plan[p]["refuel"] = refuel
        return plan

    plan_final = add_fuel_pits(plan_v2)

    # ---- FINAL SIM ----
    total_time, total_fuel, lap_records_final, sets_used_state, blowouts = simulate_race(plan_final)

    # Build submission JSON
    output = {
        "initial_tyre_id": initial_tyre_id,
        "laps": [
            {"lap": r["lap"], "segments": r["segments"], "pit": r["pit"]}
            for r in lap_records_final
        ],
    }

    # Scoring
    base_score = 500000 * (time_ref / total_time) ** 3
    fuel_bonus = -500000 * (1 - total_fuel / soft_cap) ** 2 + 500000
    tyre_bonus = 100000 * sum(sets_used_state.values()) - 50000 * blowouts
    final_score = base_score + fuel_bonus + tyre_bonus

    # Diagnostics
    n_tyre_pits = sum(1 for r in lap_records_final if r["pit"].get("enter") and "tyre_change_set_id" in r["pit"])
    n_fuel_only = sum(1 for r in lap_records_final if r["pit"].get("enter") and "tyre_change_set_id" not in r["pit"])

    diag = {
        "total_time": total_time,
        "total_fuel_used": total_fuel,
        "base_score": base_score,
        "fuel_bonus": fuel_bonus,
        "tyre_bonus": tyre_bonus,
        "sum_tyre_deg": sum(sets_used_state.values()),
        "blowouts": blowouts,
        "sets_used": dict(sets_used_state),
        "final_score": final_score,
        "n_tyre_pits": n_tyre_pits,
        "n_fuel_pits": n_fuel_only,
        "total_pits": n_tyre_pits + n_fuel_only,
        "pit_plan": plan_final,
        "initial_tyre_id": initial_tyre_id,
    }
    return output, diag


if __name__ == "__main__":
    cfg = load_config(os.path.join(os.path.dirname(__file__), "4.txt"))
    out, diag = solve(cfg)

    print(f"Friction mode: {FRICTION_MODE}")
    print(f"Total time:       {diag['total_time']:.1f}s  (ref: {cfg['race']['time_reference_s']}s)")
    print(f"Total fuel used:  {diag['total_fuel_used']:.1f}L  (soft cap: {cfg['race']['fuel_soft_cap_limit_l']}L)")
    print(f"Tyre deg sum:     {diag['sum_tyre_deg']:.3f}")
    print(f"Blowouts:         {diag['blowouts']}")
    print(f"Pit stops:        {diag['total_pits']} total ({diag['n_tyre_pits']} tyre, {diag['n_fuel_pits']} fuel-only)")
    print(f"Base score:       {diag['base_score']:,.0f}")
    print(f"Fuel bonus:       {diag['fuel_bonus']:,.0f}")
    print(f"Tyre bonus:       {diag['tyre_bonus']:,.0f}")
    print(f"Final score:      {diag['final_score']:,.0f}")
    print(f"\nSets used (final deg):")
    for sid, d in sorted(diag["sets_used"].items()):
        comp = compound_for_set_id(cfg["available_sets"], sid)
        print(f"  set {sid} ({comp}): {d:.3f}")
    print(f"\nPit plan:")
    for lap_num in sorted(diag["pit_plan"].keys()):
        p = diag["pit_plan"][lap_num]
        bits = []
        if p.get("tyre_id"):
            comp = compound_for_set_id(cfg["available_sets"], p["tyre_id"])
            bits.append(f"->{comp} (id {p['tyre_id']})")
        if p.get("refuel", 0) > 0:
            bits.append(f"+{p['refuel']:.1f}L")
        print(f"  Lap {lap_num}: {', '.join(bits) if bits else '(empty)'}")

    out_path = Path(os.path.dirname(__file__)) / ".." / "submission_level4.txt"
    out_path = out_path.resolve()
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote: {out_path}")
