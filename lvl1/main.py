"""
Entelect Grand Prix - Level 1 solver.

Level 1 assumptions applied:
- Dry weather for the whole race (no weather changes).
- Tyres do not degrade (per spec). Friction stays constant.
- No fuel pressure (soft cap 9999, 150L tank is more than enough).
- No pit stops needed.

Strategy:
- Start on Soft tyres: friction = 1.8 * 1.18 = 2.124 (highest of all compounds in dry).
- For each corner, compute max safe entry speed from:
      v_corner_max = sqrt(friction * g * radius) + crawl_constant
  capped at car max speed.
- For back-to-back corners (no straight between), both must be taken at the
  minimum of the pair's max speeds (no acceleration possible between them).
- On each straight: target top speed (90 m/s), accelerate, cruise, brake so
  we enter the following corner at exactly its max safe entry speed.
  brake_distance = (v_cruise^2 - v_corner^2) / (2 * brake_m/s^2)
  brake_start_m_before_next = brake_distance
- Segment 1 starts at 0 m/s. There are no pit stops so pit_exit_speed is
  never used.
"""

import json
import math
from pathlib import Path

G = 9.8

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def corner_max_speed(friction, radius, crawl_constant, car_max, safety_margin=0.5):
    """
    Compute safe corner entry speed with a safety margin.

    The validator appears to use a strict comparison (v_entry >= v_max -> crash),
    so entering at exactly the computed max gets penalised. We back off by
    `safety_margin` m/s to sit comfortably below the threshold.
    """
    v = math.sqrt(friction * G * radius) + crawl_constant
    v = min(v, car_max) - safety_margin
    return max(v, crawl_constant)  # never go below crawl speed

def group_corners(segments):
    """
    Return a list of entry speeds indexed by segment position.
    For each segment, compute what speed we'd travel through it if it's a corner.
    For consecutive corner runs, the speed is the min of their individual caps.
    """
    # First pass: per-corner individual cap (we'll pass friction etc in later).
    # This helper only groups indices of consecutive corners.
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

def solve(config):
    car = config["car"]
    race = config["race"]
    track = config["track"]
    tyres = config["tyres"]["properties"]

    accel = car["accel_m/se2"]
    brake = car["brake_m/se2"]
    v_max = car["max_speed_m/s"]
    crawl = car["crawl_constant_m/s"]

    # Friction = life_span × dry_friction_multiplier (not the spec table's "base coefficient").
    # Empirically confirmed from the validator: with life_span=1 and dry_multi=1.18,
    # the corner caps match. The spec table's 1.8/1.7/... column appears misleading.
    soft = tyres["Soft"]
    base_friction = soft["life_span"]  # = 1.0
    friction = base_friction * soft["dry_friction_multiplier"]

    segments = track["segments"]
    n = len(segments)

    # Per-segment max speed if it's a corner, accounting for consecutive corner groups.
    corner_entry_speed = [None] * n
    for group in group_corners(segments):
        caps = []
        for idx in group:
            seg = segments[idx]
            caps.append(corner_max_speed(friction, seg["radius_m"], crawl, v_max))
        group_speed = min(caps)
        for idx in group:
            corner_entry_speed[idx] = group_speed

    # Build one lap's segment actions.
    # For straights we need: target_m/s, brake_start_m_before_next.
    # We aim to cruise at v_max and brake down to the next corner's entry speed.
    def build_lap_segments(initial_speed_for_first_segment=None):
        lap_segs = []
        for i, seg in enumerate(segments):
            if seg["type"] == "corner":
                lap_segs.append({"id": seg["id"], "type": "corner"})
                continue

            # Straight. Find next corner entry speed (the first corner after this straight).
            next_corner_speed = None
            for j in range(i + 1, n):
                if segments[j]["type"] == "corner":
                    next_corner_speed = corner_entry_speed[j]
                    break
            # If there's no corner after this straight in this lap, we still need
            # to wrap around to lap start — but every lap ends identically, and
            # segment 1 is a straight starting at 0 m/s. The last corner (15) ends
            # the lap and feeds into segment 1 of the next lap at its corner speed.
            # For the last straight before a pit decision we still aim at v_max
            # and brake for the next corner; since the track always has a next
            # corner within the lap for each straight, this branch shouldn't hit.
            if next_corner_speed is None:
                # Fall-through safety: no corner after — just cruise at v_max, no brake.
                lap_segs.append({
                    "id": seg["id"],
                    "type": "straight",
                    "target_m/s": v_max,
                    "brake_start_m_before_next": 0,
                })
                continue

            # Braking distance from v_max down to next_corner_speed.
            v_cruise = v_max
            brake_dist = (v_cruise ** 2 - next_corner_speed ** 2) / (2 * brake)
            # If straight is too short to reach v_max, we still set target to v_max
            # (speed is capped by what we actually achieve) and pick a brake point
            # that works. We need to figure out the actual peak speed reachable.
            # accel from initial to peak, then brake to corner_speed, total = length.
            # Easier: compute peak achievable given available distance.
            length = seg["length_m"]

            # Determine initial speed entering this straight.
            if i == 0:
                v0 = 0.0 if initial_speed_for_first_segment is None else initial_speed_for_first_segment
            else:
                # Previous segment: if corner, speed = that corner's group speed;
                # if straight, speed = the target we'd reach (corner speed at its end).
                prev = segments[i - 1]
                if prev["type"] == "corner":
                    v0 = corner_entry_speed[i - 1]
                else:
                    # Two straights in a row shouldn't occur on this track, but handle it.
                    v0 = v_max

            # Accel distance to reach v_cruise from v0.
            accel_dist = max(0.0, (v_cruise ** 2 - v0 ** 2) / (2 * accel))

            if accel_dist + brake_dist <= length:
                # We can reach v_max, cruise, then brake.
                actual_brake_start = brake_dist
                target = v_cruise
            else:
                # Straight too short to hit v_max. Find peak v_peak such that
                # accel_dist(v0 -> v_peak) + brake_dist(v_peak -> v_corner) = length.
                # (v_peak^2 - v0^2)/(2a) + (v_peak^2 - v_corner^2)/(2b) = length
                # v_peak^2 * (1/(2a) + 1/(2b)) = length + v0^2/(2a) + v_corner^2/(2b)
                lhs_coef = 1 / (2 * accel) + 1 / (2 * brake)
                rhs = length + v0 ** 2 / (2 * accel) + next_corner_speed ** 2 / (2 * brake)
                v_peak_sq = rhs / lhs_coef
                v_peak = math.sqrt(max(v_peak_sq, next_corner_speed ** 2))
                v_peak = min(v_peak, v_max)
                target = v_peak
                # Brake distance for the realised peak.
                actual_brake_start = (v_peak ** 2 - next_corner_speed ** 2) / (2 * brake)
                # If v_peak <= v0, the straight is so short we can't even hold v0;
                # per spec, specifying target < entry just means we stay at v0 and
                # then brake. We still need to brake enough to reach corner speed.
                if v_peak < v0:
                    actual_brake_start = (v0 ** 2 - next_corner_speed ** 2) / (2 * brake)
                    target = v0  # won't accelerate

            lap_segs.append({
                "id": seg["id"],
                "type": "straight",
                "target_m/s": round(target, 2),
                # Round brake start UP slightly so we definitely brake enough
                # to be below the corner cap (validator appears strict).
                "brake_start_m_before_next": math.ceil(actual_brake_start * 100) / 100,
            })
        return lap_segs

    laps_out = []
    total_laps = race["laps"]
    for lap_num in range(1, total_laps + 1):
        lap = {
            "lap": lap_num,
            "segments": build_lap_segments(),
            "pit": {"enter": False},
        }
        laps_out.append(lap)

    output = {
        "initial_tyre_id": 1,  # Soft
        "laps": laps_out,
    }
    return output, friction, corner_entry_speed

def simulate(config, output):
    """
    Simulate the race to estimate total time (for our own check).
    Level 1 simplifications: no degradation, dry weather, no pit stops.
    """
    car = config["car"]
    race = config["race"]
    segments = config["track"]["segments"]
    tyres = config["tyres"]["properties"]

    accel = car["accel_m/se2"]
    brake = car["brake_m/se2"]
    v_max = car["max_speed_m/s"]
    crawl = car["crawl_constant_m/s"]
    crash_penalty = race["corner_crash_penalty_s"]

    # Soft in dry: friction = life_span * dry_multiplier (empirically correct)
    friction = tyres["Soft"]["life_span"] * tyres["Soft"]["dry_friction_multiplier"]

    # Corner group speeds (same computation as solver)
    n = len(segments)
    corner_entry_speed = [None] * n
    groups = []
    i = 0
    while i < n:
        if segments[i]["type"] == "corner":
            j = i
            while j < n and segments[j]["type"] == "corner":
                j += 1
            groups.append(list(range(i, j)))
            i = j
        else:
            i += 1
    for group in groups:
        caps = [min(math.sqrt(friction * G * segments[idx]["radius_m"]) + crawl, v_max) for idx in group]
        gs = min(caps)
        for idx in group:
            corner_entry_speed[idx] = gs

    total_time = 0.0
    current_speed = 0.0

    for lap in output["laps"]:
        for seg_action in lap["segments"]:
            seg = next(s for s in segments if s["id"] == seg_action["id"])
            if seg["type"] == "corner":
                # Speed constant at corner_entry_speed. Time = length / speed.
                speed = current_speed  # we entered at this speed
                # Sanity: should equal corner_entry_speed[idx]
                total_time += seg["length_m"] / speed
                # exit speed unchanged
            else:
                target = seg_action["target_m/s"]
                brake_start = seg_action["brake_start_m_before_next"]
                length = seg["length_m"]

                # Next corner speed (to validate we arrive at right speed)
                idx = segments.index(seg)
                next_corner_speed = None
                for j in range(idx + 1, n):
                    if segments[j]["type"] == "corner":
                        next_corner_speed = corner_entry_speed[j]
                        break

                v0 = current_speed
                # Cap target by v_max
                target = min(target, v_max)

                # Phase A: accelerate from v0 to min(target, achievable) OR until brake point
                # brake starts at (length - brake_start) metres into the straight.
                brake_point_pos = length - brake_start

                # If target <= v0: per spec, follow through — no accel, no decel until brake point.
                if target <= v0:
                    peak = v0
                    cruise_end = brake_point_pos
                    # time to cruise to brake point at v0
                    t_cruise = cruise_end / v0 if v0 > 0 else float("inf")
                    # braking phase
                    # we have brake_start metres to decelerate
                    # final speed after braking: v_f^2 = v0^2 - 2*b*brake_start  (but clamp)
                    v_f_sq = v0 ** 2 - 2 * brake * brake_start
                    v_f = math.sqrt(max(v_f_sq, 0.0))
                    # time to brake: (v0 - v_f)/b  if we fully use brake_start distance
                    t_brake = (v0 - v_f) / brake if brake > 0 else 0
                    total_time += t_cruise + t_brake
                    current_speed = v_f
                else:
                    # Accelerate from v0 to target
                    t_accel = (target - v0) / accel
                    d_accel = v0 * t_accel + 0.5 * accel * t_accel ** 2
                    if d_accel > brake_point_pos:
                        # Can't reach target before brake point. Find peak reached at brake point.
                        # v_peak^2 = v0^2 + 2*a*brake_point_pos
                        v_peak_sq = v0 ** 2 + 2 * accel * brake_point_pos
                        v_peak = math.sqrt(v_peak_sq)
                        t_accel = (v_peak - v0) / accel
                        d_accel = brake_point_pos
                        t_cruise = 0
                    else:
                        v_peak = target
                        # Cruise from end of accel to brake_point_pos
                        d_cruise = brake_point_pos - d_accel
                        t_cruise = d_cruise / v_peak
                    # Brake phase over brake_start metres
                    v_f_sq = v_peak ** 2 - 2 * brake * brake_start
                    v_f = math.sqrt(max(v_f_sq, 0.0))
                    t_brake = (v_peak - v_f) / brake
                    total_time += t_accel + t_cruise + t_brake
                    current_speed = v_f

                # Check: did we arrive at/below next corner speed?
                if next_corner_speed is not None and current_speed > next_corner_speed + 1e-6:
                    # Crash penalty
                    total_time += crash_penalty
    return total_time

if __name__ == "__main__":
    config = load_config("1.txt")
    output, friction, corner_speeds = solve(config)

    # Write the submission
    with open("Output.txt", "w") as f:
        json.dump(output, f, indent=2)

    # Diagnostics
    print(f"Tyre: Soft (dry friction = {friction:.4f})")
    print("Corner entry speeds (m/s):")
    segs = config["track"]["segments"]
    for i, s in enumerate(segs):
        if s["type"] == "corner":
            print(f"  seg {s['id']} r={s['radius_m']}m -> {corner_speeds[i]:.3f}")

    est_time = simulate(config, output)
    print(f"\nEstimated total race time: {est_time:.3f} s")
    ref = config["race"]["time_reference_s"]
    if est_time > 0:
        score = 500000 * (ref / est_time) ** 3
        print(f"Reference time: {ref} s")
        print(f"Estimated base score: {score:,.0f}")

    print(f"\nWrote submission to output.txt")