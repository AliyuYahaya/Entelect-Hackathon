"""
Entelect Grand Prix - Level 2 solver.

What's new vs Level 1:
- 60 laps on a longer track (Silverstone, 10370 m/lap) → 622 km race.
- Fuel matters: soft cap 219L, tank 150L, initial 150L.
- Pit stops available (20s base + 3s tyre swap + refuel_L/5).
- Still dry weather, still no tyre degradation (that's L3/L4).

Key finding from pre-analysis:
- Fuel consumption is dominated by K_base (linear with distance), not K_drag
  (speed²). Driving slower saves almost no fuel (0.05 L/lap between 90 and
  40 m/s). The soft cap will be blown through no matter what. Best to just
  drive flat-out and take the ~97k fuel-bonus hit.
- Full-attack needs ~315 L total. Tank is 150 L. So 2 pit stops minimum.
- Pit split 20/20/20 is optimal (pit-time is same regardless of when we pit
  since we refuel the same total amount).
- Each pit: refuel only (no tyre change, since tyres don't degrade in L2).
  This saves the 3s tyre swap.

Strategy:
- Start on Soft (id=1), target 90 m/s on every straight.
- Pit after lap 20 and after lap 40.
- Pit 1: refuel 105 L (bring tank from ~45 back to ~150).
- Pit 2: refuel just enough for the last 20 laps (~105 L).
- Post-pit laps start at 20 m/s (pit_exit_speed).
"""

import json
import math
import os
from pathlib import Path

G = 9.8
SAFETY_MARGIN = 0.1  # m/s below computed corner cap
K_BASE = 0.0005
K_DRAG = 0.0000000015


def load_config(path):
    with open(path, "r") as fh:
        return json.load(fh)


def corner_max_speed(friction, radius, crawl, v_max, margin=SAFETY_MARGIN):
    v = math.sqrt(friction * G * radius) + crawl
    v = min(v, v_max) - margin
    return max(v, crawl)


def group_corners(segments):
    groups = []
    i = 0
    n = len(segments)
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
        caps = [corner_max_speed(friction, segments[k]["radius_m"], crawl, v_max)
                for k in g]
        s = min(caps)
        for k in g:
            cs[k] = s
    return cs


def build_lap_segments(segments, corner_speed, v_in_start, v_max, accel, brake):
    """
    Build one lap's segment action list given the entry speed of segment 1.
    Returns (segments_list, exit_speed_after_lap).
    """
    lap_segs = []
    n = len(segments)
    v = v_in_start
    for i, seg in enumerate(segments):
        if seg["type"] == "corner":
            lap_segs.append({"id": seg["id"], "type": "corner"})
            v = corner_speed[i]
            continue

        # Next corner entry speed
        next_cs = None
        for j in range(i + 1, n):
            if segments[j]["type"] == "corner":
                next_cs = corner_speed[j]
                break
        if next_cs is None:
            # No more corners in this lap (shouldn't happen on this track, but
            # safety fallback): cruise at v_max and don't brake.
            lap_segs.append({
                "id": seg["id"],
                "type": "straight",
                "target_m/s": v_max,
                "brake_start_m_before_next": 0,
            })
            continue

        length = seg["length_m"]
        v0 = v
        target = v_max

        # If v0 already >= target, per spec: cruise at v0, no accel, then brake.
        if v0 >= target:
            d_brake = (v0 ** 2 - next_cs ** 2) / (2 * brake)
            actual_brake_start = d_brake
            chosen_target = target  # spec says target_m/s below entry just means hold entry speed
            v = next_cs
        else:
            # Can we reach target?
            d_accel = (target ** 2 - v0 ** 2) / (2 * accel)
            d_brake = (target ** 2 - next_cs ** 2) / (2 * brake)
            if d_accel + d_brake <= length:
                actual_brake_start = d_brake
                chosen_target = target
            else:
                # Peak below target
                lhs = 1 / (2 * accel) + 1 / (2 * brake)
                rhs = length + v0 ** 2 / (2 * accel) + next_cs ** 2 / (2 * brake)
                v_peak = math.sqrt(max(rhs / lhs, next_cs ** 2))
                v_peak = min(v_peak, target)
                actual_brake_start = (v_peak ** 2 - next_cs ** 2) / (2 * brake)
                chosen_target = v_peak
            v = next_cs

        lap_segs.append({
            "id": seg["id"],
            "type": "straight",
            "target_m/s": round(chosen_target, 2),
            "brake_start_m_before_next": math.ceil(actual_brake_start * 100) / 100,
        })
    return lap_segs, v


def fuel_for_phase(v_in, v_out, distance):
    v_avg = (v_in + v_out) / 2
    return (K_BASE + K_DRAG * v_avg ** 2) * distance


def simulate_lap_fuel(segments, corner_speed, v_in, v_max, accel, brake):
    """Return fuel consumed in one lap given entry speed."""
    v = v_in
    total_f = 0.0
    n = len(segments)
    for i, seg in enumerate(segments):
        if seg["type"] == "corner":
            total_f += fuel_for_phase(corner_speed[i], corner_speed[i], seg["length_m"])
            v = corner_speed[i]
        else:
            next_cs = None
            for j in range(i + 1, n):
                if segments[j]["type"] == "corner":
                    next_cs = corner_speed[j]
                    break
            if next_cs is None:
                next_cs = corner_speed[0]  # wrap-around

            length = seg["length_m"]
            v0 = v
            target = v_max

            if v0 >= target:
                d_brake = (v0 ** 2 - next_cs ** 2) / (2 * brake)
                d_cruise = length - d_brake
                if d_cruise > 0:
                    total_f += fuel_for_phase(v0, v0, d_cruise)
                total_f += fuel_for_phase(v0, next_cs, d_brake)
            else:
                d_accel = (target ** 2 - v0 ** 2) / (2 * accel)
                d_brake = (target ** 2 - next_cs ** 2) / (2 * brake)
                if d_accel + d_brake <= length:
                    d_cruise = length - d_accel - d_brake
                    total_f += fuel_for_phase(v0, target, d_accel)
                    total_f += fuel_for_phase(target, target, d_cruise)
                    total_f += fuel_for_phase(target, next_cs, d_brake)
                else:
                    lhs = 1 / (2 * accel) + 1 / (2 * brake)
                    rhs = length + v0 ** 2 / (2 * accel) + next_cs ** 2 / (2 * brake)
                    v_peak = math.sqrt(max(rhs / lhs, next_cs ** 2))
                    v_peak = min(v_peak, target)
                    d_a = (v_peak ** 2 - v0 ** 2) / (2 * accel)
                    d_b = (v_peak ** 2 - next_cs ** 2) / (2 * brake)
                    total_f += fuel_for_phase(v0, v_peak, d_a)
                    total_f += fuel_for_phase(v_peak, next_cs, d_b)
            v = next_cs
    return total_f, v


def simulate_lap_time(segments, corner_speed, v_in, v_max, accel, brake):
    """Return (time, exit_speed) for one lap given entry speed."""
    v = v_in
    total_t = 0.0
    n = len(segments)
    for i, seg in enumerate(segments):
        if seg["type"] == "corner":
            total_t += seg["length_m"] / corner_speed[i]
            v = corner_speed[i]
        else:
            next_cs = None
            for j in range(i + 1, n):
                if segments[j]["type"] == "corner":
                    next_cs = corner_speed[j]
                    break
            if next_cs is None:
                next_cs = corner_speed[0]
            length = seg["length_m"]
            v0 = v
            target = v_max
            if v0 >= target:
                d_brake = (v0 ** 2 - next_cs ** 2) / (2 * brake)
                d_cruise = length - d_brake
                total_t += d_cruise / v0 + (v0 - next_cs) / brake
            else:
                d_accel = (target ** 2 - v0 ** 2) / (2 * accel)
                d_brake = (target ** 2 - next_cs ** 2) / (2 * brake)
                if d_accel + d_brake <= length:
                    d_cruise = length - d_accel - d_brake
                    total_t += (target - v0) / accel + d_cruise / target + (target - next_cs) / brake
                else:
                    lhs = 1 / (2 * accel) + 1 / (2 * brake)
                    rhs = length + v0 ** 2 / (2 * accel) + next_cs ** 2 / (2 * brake)
                    v_peak = min(math.sqrt(max(rhs / lhs, next_cs ** 2)), target)
                    total_t += (v_peak - v0) / accel + (v_peak - next_cs) / brake
            v = next_cs
    return total_t, v


def solve(config, pit_laps=(20, 40)):
    car = config["car"]
    race = config["race"]
    track = config["track"]
    tyres = config["tyres"]["properties"]

    accel = car["accel_m/se2"]
    brake = car["brake_m/se2"]
    v_max = car["max_speed_m/s"]
    crawl = car["crawl_constant_m/s"]
    tank = car["fuel_tank_capacity_l"]
    initial_fuel = car["initial_fuel_l"]
    pit_exit_speed = race["pit_exit_speed_m/s"]
    total_laps = race["laps"]
    refuel_rate = race["pit_refuel_rate_l/s"]
    base_pit_time = race["base_pit_stop_time_s"]

    soft = tyres["Soft"]
    friction = soft["life_span"] * soft["dry_friction_multiplier"]

    segments = track["segments"]
    corner_speed = compute_corner_speeds(segments, friction, crawl, v_max)

    # Pre-compute fuel per lap for each possible entry speed we need.
    # Steady-state entry = corner_speed[-1] (exit of last corner).
    f_steady, _ = simulate_lap_fuel(segments, corner_speed, corner_speed[-1],
                                    v_max, accel, brake)
    f_lap1, _ = simulate_lap_fuel(segments, corner_speed, 0.0, v_max, accel, brake)
    f_post_pit, _ = simulate_lap_fuel(segments, corner_speed, pit_exit_speed,
                                      v_max, accel, brake)

    # Build the laps.
    pit_lap_set = set(pit_laps)

    laps_out = []
    fuel_in_tank = initial_fuel
    total_fuel_used = 0.0
    v_lap_start = 0.0  # lap 1 starts from rest

    for lap_num in range(1, total_laps + 1):
        lap_segs, v_exit = build_lap_segments(segments, corner_speed,
                                              v_lap_start, v_max, accel, brake)

        # Figure out fuel this lap (depends on entry speed)
        if lap_num == 1:
            lap_fuel = f_lap1
        elif v_lap_start == pit_exit_speed:
            lap_fuel = f_post_pit
        else:
            lap_fuel = f_steady

        fuel_in_tank -= lap_fuel
        total_fuel_used += lap_fuel

        # Decide pit
        pit_entry = False
        refuel = 0.0
        if lap_num in pit_lap_set:
            pit_entry = True
            # Figure out how much to refuel. If this is the last pit, only
            # refuel enough for remaining laps. Otherwise fill tank.
            remaining_laps = total_laps - lap_num
            future_pits = sum(1 for p in pit_laps if p > lap_num)
            if future_pits == 0:
                # Last pit. Refuel just enough for remaining laps.
                # First lap post-pit uses f_post_pit, rest use f_steady.
                needed = f_post_pit + max(0, remaining_laps - 1) * f_steady
                refuel = max(0.0, needed - fuel_in_tank)
            else:
                # Fill to tank capacity.
                refuel = tank - fuel_in_tank
                refuel = max(0.0, min(refuel, tank))
            fuel_in_tank += refuel

        lap_entry = {
            "lap": lap_num,
            "segments": lap_segs,
        }
        if pit_entry:
            pit_block = {
                "enter": True,
                "fuel_refuel_amount_l": round(refuel, 2),
            }
            lap_entry["pit"] = pit_block
        else:
            lap_entry["pit"] = {"enter": False}
        laps_out.append(lap_entry)

        # Next lap entry speed
        if pit_entry:
            v_lap_start = pit_exit_speed
        else:
            v_lap_start = v_exit

    output = {
        "initial_tyre_id": 1,
        "laps": laps_out,
    }

    # Estimate race time and score
    total_time = 0.0
    v_start = 0.0
    for lap_num in range(1, total_laps + 1):
        t, v_exit = simulate_lap_time(segments, corner_speed, v_start,
                                      v_max, accel, brake)
        total_time += t
        if lap_num in pit_lap_set:
            # Find the actual refuel amount from the output
            refuel_amt = laps_out[lap_num - 1]["pit"]["fuel_refuel_amount_l"]
            pit_time = base_pit_time + refuel_amt / refuel_rate
            total_time += pit_time
            v_start = pit_exit_speed
        else:
            v_start = v_exit

    time_ref = race["time_reference_s"]
    soft_cap = race["fuel_soft_cap_limit_l"]
    base_score = 500000 * (time_ref / total_time) ** 3
    fuel_bonus = -500000 * (1 - total_fuel_used / soft_cap) ** 2 + 500000

    diagnostics = {
        "friction": friction,
        "track_length_m": sum(s["length_m"] for s in segments),
        "f_steady_lap": f_steady,
        "f_lap1": f_lap1,
        "f_post_pit": f_post_pit,
        "total_fuel_used": total_fuel_used,
        "total_time": total_time,
        "base_score": base_score,
        "fuel_bonus": fuel_bonus,
        "final_score": base_score + fuel_bonus,
        "corner_speeds": {segments[i]["id"]: round(corner_speed[i], 3)
                          for i in range(len(segments))
                          if corner_speed[i] is not None},
    }
    return output, diagnostics


if __name__ == "__main__":
    cfg = load_config(os.path.join(os.path.dirname(__file__), "2.txt"))

    # Try a few pit-lap splits and pick the best
    best_score = -1
    best_out = None
    best_diag = None
    best_pits = None
    for p1, p2 in [(20, 40), (19, 40), (20, 41), (21, 41), (21, 42),
                   (19, 39), (22, 42), (18, 38), (20, 39)]:
        out, diag = solve(cfg, pit_laps=(p1, p2))
        if diag["final_score"] > best_score:
            best_score = diag["final_score"]
            best_out = out
            best_diag = diag
            best_pits = (p1, p2)

    print(f"Best pit laps: {best_pits}")
    print(f"Friction (Soft dry): {best_diag['friction']:.4f}")
    print(f"Track length: {best_diag['track_length_m']} m")
    print(f"Fuel per steady lap: {best_diag['f_steady_lap']:.3f} L")
    print(f"Fuel per lap 1: {best_diag['f_lap1']:.3f} L")
    print(f"Fuel per post-pit lap: {best_diag['f_post_pit']:.3f} L")
    print(f"Total fuel used: {best_diag['total_fuel_used']:.2f} L")
    print(f"Total time: {best_diag['total_time']:.2f} s")
    print(f"Base score: {best_diag['base_score']:,.0f}")
    print(f"Fuel bonus: {best_diag['fuel_bonus']:,.0f}")
    print(f"Final score: {best_diag['final_score']:,.0f}")

    out_path = Path("../Output2.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(best_out, fh, indent=2)
    print(f"\nWrote: {out_path}")