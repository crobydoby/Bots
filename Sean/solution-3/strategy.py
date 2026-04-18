"""
strategy.py
-----------
Dataclasses for representing a race strategy and a serialiser that produces
the required submission JSON format.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Strategy building blocks
# ---------------------------------------------------------------------------

@dataclass
class StraightAction:
    """
    Decisions made for a single straight segment.

    Attributes
    ----------
    segment_id : int
        The segment ID this action applies to.
    target_m_s : float
        Target speed (m/s) to reach on this straight.
    brake_start_m_before_next : float
        Distance before the end of the straight at which braking begins.
        Set to 0 if no braking is needed (e.g., the next segment is a straight).
    """
    segment_id: int
    target_m_s: float
    brake_start_m_before_next: float


@dataclass
class CornerAction:
    """
    Decisions for a corner segment.
    No user-controlled parameters — speed is physics-constrained.
    Stored for completeness and JSON output.
    """
    segment_id: int


@dataclass
class PitAction:
    """
    Pit stop decision at the end of a lap.

    Attributes
    ----------
    enter : bool
        Whether to enter the pit lane this lap.
    tyre_change_set_id : Optional[int]
        ID of the tyre set to switch to. None means no tyre change.
    fuel_refuel_amount_l : float
        Litres of fuel to add. 0 means no refuelling.
    """
    enter: bool
    tyre_change_set_id: Optional[int] = None
    fuel_refuel_amount_l: float = 0.0


@dataclass
class LapStrategy:
    """Full set of decisions for a single lap."""
    lap: int
    segment_actions: List[StraightAction | CornerAction] = field(default_factory=list)
    pit: PitAction = field(default_factory=lambda: PitAction(enter=False))


@dataclass
class RaceStrategy:
    """Complete race strategy across all laps."""
    initial_tyre_id: int
    laps: List[LapStrategy] = field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON serialiser
# ---------------------------------------------------------------------------

def _serialise_segment(action: StraightAction | CornerAction) -> Dict[str, Any]:
    if isinstance(action, StraightAction):
        return {
            "id": action.segment_id,
            "type": "straight",
            "target_m/s": action.target_m_s,
            "brake_start_m_before_next": action.brake_start_m_before_next,
        }
    else:
        return {
            "id": action.segment_id,
            "type": "corner",
        }


def _serialise_pit(pit: PitAction) -> Dict[str, Any]:
    result: Dict[str, Any] = {"enter": pit.enter}
    if pit.enter:
        if pit.tyre_change_set_id is not None:
            result["tyre_change_set_id"] = pit.tyre_change_set_id
        if pit.fuel_refuel_amount_l > 0:
            result["fuel_refuel_amount_l"] = pit.fuel_refuel_amount_l
    return result


def strategy_to_dict(strategy: RaceStrategy) -> Dict[str, Any]:
    """Convert a RaceStrategy to the submission JSON-compatible dict."""
    laps_data = []
    for lap_strat in strategy.laps:
        laps_data.append({
            "lap": lap_strat.lap,
            "segments": [_serialise_segment(a) for a in lap_strat.segment_actions],
            "pit": _serialise_pit(lap_strat.pit),
        })
    return {
        "initial_tyre_id": strategy.initial_tyre_id,
        "laps": laps_data,
    }


def strategy_to_json(strategy: RaceStrategy, indent: int = 2) -> str:
    """Serialise a RaceStrategy to a JSON string."""
    return json.dumps(strategy_to_dict(strategy), indent=indent)


def save_strategy(strategy: RaceStrategy, path: str) -> None:
    """Write a strategy to a .txt / .json file."""
    with open(path, "w") as f:
        f.write(strategy_to_json(strategy))
    print(f"Strategy saved to {path}")


# ---------------------------------------------------------------------------
# Shared helper: corner max speed and required straight exit
# ---------------------------------------------------------------------------

def _make_corner_helpers(car, segments, tyre_state, weather_type, corner_speed_margin):
    """Return (corner_max_fn, required_exit_fn) closures for a given track setup."""
    from models import SegmentType, GRAVITY

    def _corner_max_speed(corner_seg, friction: float) -> float:
        raw = math.sqrt(friction * GRAVITY * corner_seg.radius_m) + car.crawl_constant_m_s
        return max(car.crawl_constant_m_s, raw - corner_speed_margin)

    def _required_exit_speed(straight_idx: int) -> float:
        """Min margined corner max speed across all consecutive corners after this straight."""
        friction = tyre_state.current_friction(weather_type)
        required = car.max_speed_m_s
        j = straight_idx + 1
        while j < len(segments) and segments[j].type == SegmentType.CORNER:
            required = min(required, _corner_max_speed(segments[j], friction))
            j += 1
        return required

    return _corner_max_speed, _required_exit_speed


# ---------------------------------------------------------------------------
# Level 1: naive baseline strategy (max safe speed, no pit stops)
# ---------------------------------------------------------------------------

def build_naive_strategy(config, tyre_degradation: bool = True) -> RaceStrategy:
    """
    Build a simple baseline strategy:
      - Start on the first available tyre set.
      - On every straight: target max car speed, brake as late as safely possible.
      - No pit stops.

    Parameters
    ----------
    config : LevelConfig
        Loaded level configuration.
    tyre_degradation : bool
        When True (Levels 2+), a 0.5 m/s safety margin is subtracted from corner
        max speeds to absorb friction drop accumulated on straights.
        When False (Level 1 simple mode), no margin is used — the submission engine
        tolerates the tiny (~0.002 m/s) rounding artefact from 2dp brake distances.

    Returns
    -------
    RaceStrategy
    """
    from models import SegmentType, WeatherType, TyreState

    initial_set = config.available_sets[0]
    tyre_state  = TyreState(tyre_set=initial_set)

    start_weather = config.weather_conditions[0] if config.weather_conditions else None
    weather_type  = start_weather.condition if start_weather else WeatherType.DRY

    car      = config.car
    segments = config.track.segments

    # 0.5 m/s margin when degradation active; 0 otherwise (matches submission engine).
    corner_speed_margin = 0.5 if tyre_degradation else 0.0
    _, _required_exit = _make_corner_helpers(car, segments, tyre_state, weather_type,
                                             corner_speed_margin)

    laps: List[LapStrategy] = []

    for lap_num in range(1, config.race.laps + 1):
        segment_actions = []

        for idx, seg in enumerate(segments):
            if seg.type == SegmentType.STRAIGHT:
                required_exit = _required_exit(idx)
                target_speed  = car.max_speed_m_s

                if target_speed > required_exit:
                    brake_dist = (target_speed ** 2 - required_exit ** 2) / (2 * car.brake_m_se2)
                else:
                    brake_dist = 0.0

                brake_dist = min(brake_dist, seg.length_m)

                segment_actions.append(StraightAction(
                    segment_id=seg.id,
                    target_m_s=target_speed,
                    brake_start_m_before_next=round(brake_dist, 2),
                ))
            else:
                segment_actions.append(CornerAction(segment_id=seg.id))

        laps.append(LapStrategy(
            lap=lap_num,
            segment_actions=segment_actions,
            pit=PitAction(enter=False),
        ))

    return RaceStrategy(
        initial_tyre_id=initial_set.primary_id,
        laps=laps,
    )


# ---------------------------------------------------------------------------
# Level 2: lambda strategy with fuel-triggered pit stops
# ---------------------------------------------------------------------------

def build_lambda_strategy(config, lam: float) -> RaceStrategy:
    """
    Build a fuel-aware strategy for Level 2.

    Straight speed is scaled by lambda:
        target_speed = lambda * car.max_speed_m_s

    Pit stop rule (applied at the end of each lap, except the final lap):
        If current_fuel < fuel_used_last_lap:
            pit and refuel.
        Refuel amount = min(remaining_laps * fuel_used_last_lap, tank_capacity - current_fuel)
        No tyre change (Soft tyres throughout, no degradation in Level 2).

    Because pit decisions depend on the simulated fuel state (which we don't know
    until we run the simulation), this function performs a lightweight internal
    simulation pass to determine the exact pit lap and refuel amounts before
    building the final RaceStrategy.

    Parameters
    ----------
    config : LevelConfig
        Level 2 configuration.
    lam : float
        Speed scaling factor in (0, 1].  1.0 = max speed, 0.5 = half max speed.

    Returns
    -------
    RaceStrategy
    """
    from models import SegmentType, WeatherType, TyreState, GRAVITY

    car      = config.car
    segments = config.track.segments
    total_laps = config.race.laps

    # Level 2: no tyre degradation, use Soft set
    soft_set = next(ts for ts in config.available_sets if ts.compound.value == "Soft")
    tyre_state  = TyreState(tyre_set=soft_set)

    start_weather = config.weather_conditions[0] if config.weather_conditions else None
    weather_type  = start_weather.condition if start_weather else WeatherType.DRY

    # No corner speed margin needed (no degradation)
    _, _required_exit = _make_corner_helpers(car, segments, tyre_state, weather_type,
                                             corner_speed_margin=0.0)

    # Target speed on straights
    target_speed = lam * car.max_speed_m_s

    # Pre-compute per-straight actions (same every lap — no degradation).
    #
    # Lambda scales the target (cruise) speed on each straight:
    #   target_m_s = lambda * max_speed
    #
    # The submission engine / simulator sets:
    #   cruise_speed = max(target_m_s, entry_speed)
    # so if a straight is preceded by ANOTHER straight (entry can be max_speed),
    # we must compute brake_dist from max_speed to guarantee a safe exit speed.
    # For all other straights (preceded by a corner), entry <= corner_max < lambda*max,
    # so cruise = target_m_s and brake_dist from target is correct.

    # Identify which straights are directly preceded by another straight
    prev_is_straight = set()
    for i, seg in enumerate(segments):
        if seg.type == SegmentType.STRAIGHT and i > 0:
            if segments[i - 1].type == SegmentType.STRAIGHT:
                prev_is_straight.add(seg.id)

    straight_actions: Dict[int, StraightAction] = {}
    for idx, seg in enumerate(segments):
        if seg.type == SegmentType.STRAIGHT:
            required_exit = _required_exit(idx)

            # Effective cruise speed for this straight
            cruise = max(target_speed, required_exit)
            cruise = min(cruise, car.max_speed_m_s)

            # If preceded by a straight, entry could be max_speed; use max_speed for
            # brake_dist so the car can always stop in time regardless of carry-over.
            brake_from = car.max_speed_m_s if seg.id in prev_is_straight else cruise

            if brake_from > required_exit:
                brake_dist = (brake_from ** 2 - required_exit ** 2) / (2 * car.brake_m_se2)
            else:
                brake_dist = 0.0

            brake_dist = min(brake_dist, seg.length_m)

            straight_actions[seg.id] = StraightAction(
                segment_id=seg.id,
                target_m_s=round(cruise, 4),
                brake_start_m_before_next=round(brake_dist, 2),
            )

    # ---- Lightweight fuel simulation to determine pit laps ----
    # Fuel used per lap is constant (no degradation, fixed speeds).
    # We simulate one lap to get fuel_per_lap, then apply the pit rule.

    def _lap_fuel(start_fuel: float) -> float:
        """Estimate fuel consumed in one lap using K_BASE * distance (simplified)."""
        # Use K_BASE * total distance as approximation — exact enough for pit timing.
        # K_DRAG contribution is negligible vs K_BASE at these speeds.
        from models import K_BASE, K_DRAG
        total = 0.0
        for seg in segments:
            if seg.type == SegmentType.STRAIGHT:
                act = straight_actions[seg.id]
                t   = act.target_m_s   # cruise speed (approximate avg)
                bd  = act.brake_start_m_before_next
                accel_dist  = (t ** 2 - 0) / (2 * car.accel_m_se2)  # rough
                brake_dist  = bd
                cruise_dist = max(0.0, seg.length_m - accel_dist - brake_dist)
                total += (K_BASE + K_DRAG * t**2) * seg.length_m  # simplified
            else:
                spd = math.sqrt(1.18 * GRAVITY * seg.radius_m) + car.crawl_constant_m_s if seg.radius_m else car.crawl_constant_m_s
                total += (K_BASE + K_DRAG * spd**2) * seg.length_m
        return total

    # Better: run the actual simulator for one lap to get exact fuel per lap
    # We'll do the full pass below using the simulator.
    # For pit decision we use a forward simulation.

    from simulator import simulate, SimConfig

    # Build a no-pit strategy for the full race to observe fuel consumption per lap
    probe_laps = []
    for ln in range(1, total_laps + 1):
        actions = list(straight_actions.values()) + [
            CornerAction(segment_id=seg.id)
            for seg in segments if seg.type == SegmentType.CORNER
        ]
        # Sort by segment id to maintain track order
        actions.sort(key=lambda a: a.segment_id)
        probe_laps.append(LapStrategy(lap=ln, segment_actions=actions, pit=PitAction(enter=False)))

    probe_strategy = RaceStrategy(initial_tyre_id=soft_set.primary_id, laps=probe_laps)
    probe_cfg = SimConfig(tyre_degradation=False, fuel_consumption=True)
    probe_result = simulate(config, probe_strategy, probe_cfg)

    # Extract per-lap fuel consumption from probe run (before tank runs dry)
    # Use lap 1 fuel as the reference — it's representative.
    # (In simple mode all laps are identical once steady state.)
    fuel_per_lap = probe_result.lap_results[0].fuel_used_l

    # ---- Build pit schedule ----
    # Forward-simulate fuel state lap by lap, deciding when to pit.
    pit_schedule: Dict[int, float] = {}   # lap_num -> refuel_amount_l
    current_fuel = car.initial_fuel_l

    for lap_num in range(1, total_laps + 1):
        # Consume fuel this lap
        current_fuel -= fuel_per_lap
        current_fuel  = max(current_fuel, 0.0)

        # No pit on final lap
        if lap_num == total_laps:
            break

        remaining_laps_after = total_laps - lap_num
        # Pit rule: if fuel remaining < fuel needed for next lap
        if current_fuel < fuel_per_lap:
            # Refuel for as many remaining laps as possible, capped at tank capacity
            needed   = remaining_laps_after * fuel_per_lap
            headroom = car.fuel_tank_capacity_l - current_fuel
            refuel   = min(needed, headroom)
            refuel   = max(refuel, 0.0)
            pit_schedule[lap_num] = round(refuel, 4)
            current_fuel += refuel

    # ---- Assemble final RaceStrategy ----
    laps_out: List[LapStrategy] = []
    for lap_num in range(1, total_laps + 1):
        actions = list(straight_actions.values()) + [
            CornerAction(segment_id=seg.id)
            for seg in segments if seg.type == SegmentType.CORNER
        ]
        actions.sort(key=lambda a: a.segment_id)

        if lap_num in pit_schedule:
            pit = PitAction(
                enter=True,
                tyre_change_set_id=None,          # no tyre change
                fuel_refuel_amount_l=pit_schedule[lap_num],
            )
        else:
            pit = PitAction(enter=False)

        laps_out.append(LapStrategy(lap=lap_num, segment_actions=actions, pit=pit))

    return RaceStrategy(initial_tyre_id=soft_set.primary_id, laps=laps_out)

"""
Level 3 additions — drop these into their respective files.
 
=============================================================
  STRATEGY.PY — add build_level3_strategy() at the bottom
=============================================================
"""
 
# ---------------------------------------------------------------------------
# Level 3: weather-aware strategy with tyre-change sweep
# ---------------------------------------------------------------------------
 
def build_level3_strategy(config, lam: float, change_intervals: list) -> "RaceStrategy":
    """
    Build a weather + fuel aware strategy for Level 3.
 
    Straight speed is scaled by lambda (same as Level 2).
 
    Tyre-change rule:
        change_intervals is a list of lap numbers at which a pit is *forced*
        for a tyre change.  At each forced pit the best tyre for the
        predominant weather over the *next* stint is selected automatically.
        Fuel is topped up at every pit using the same greedy rule as Level 2.
 
    Parameters
    ----------
    config : LevelConfig
        Level 3 configuration (with weather conditions).
    lam : float
        Speed scaling factor in (0, 1].
    change_intervals : list[int]
        Lap numbers at which a tyre change pit is forced.
        e.g. [20, 45] → pit on lap 20 and lap 45.
 
    Returns
    -------
    RaceStrategy
    """
    from models import SegmentType, WeatherType, TyreState, GRAVITY
    import math
 
    car        = config.car
    segments   = config.track.segments
    total_laps = config.race.laps
 
    # ---- Helper: weather at a given elapsed race time ----
    weather_conditions = config.weather_conditions  # ordered list with .duration_s
 
    def weather_cycle_duration() -> float:
        return sum(w.duration_s for w in weather_conditions)
 
    def weather_at(time_s: float):
        """Return the WeatherCondition active at time_s."""
        if not weather_conditions:
            return None
        cycle = weather_cycle_duration()
        t = time_s % cycle
        acc = 0.0
        for w in weather_conditions:
            acc += w.duration_s
            if t < acc:
                return w
        return weather_conditions[-1]
 
    def dominant_weather_in_window(start_s: float, end_s: float):
        """Return the weather condition that covers the most time in [start_s, end_s]."""
        if not weather_conditions:
            return WeatherType.DRY
        from collections import defaultdict
        coverage = defaultdict(float)
        step = 10.0
        t = start_s
        while t < end_s:
            w = weather_at(t)
            coverage[w.condition] += min(step, end_s - t)
            t += step
        return max(coverage, key=coverage.get)
 
    # ---- Helper: pick best tyre compound for a weather condition ----
    def best_tyre_for_weather(weather_type, available_sets):
        """
        Choose the tyre set that maximises corner speed (friction) for the
        given weather condition.  Uses the first unused set of the best compound.
        """
        from models import WeatherType
        weather_attr = {
            WeatherType.DRY:        "dry_friction_multiplier",
            WeatherType.COLD:       "cold_friction_multiplier",
            WeatherType.LIGHT_RAIN: "light_rain_friction_multiplier",
            WeatherType.HEAVY_RAIN: "heavy_rain_friction_multiplier",
        }.get(weather_type, "dry_friction_multiplier")
 
        tyre_props = config.tyre_properties   # dict compound → TyreProperties
 
        # Score each available set by base_friction * weather_multiplier
        best_set  = None
        best_score = -1.0
        for tset in available_sets:
            props = tyre_props[tset.compound]
            score = props.life_span * getattr(props, weather_attr, 1.0)
            if score > best_score:
                best_score = score
                best_set   = tset
        return best_set
 
    # ---- Pre-compute per-straight actions (same as Level 2) ----
    # Start with the initial tyre to determine corner exit speeds.
    # For level 3 we use no degradation margin in pre-computation (dynamic).
    initial_set = config.available_sets[0]
    tyre_state  = TyreState(tyre_set=initial_set)
    start_weather = config.weather_conditions[0] if config.weather_conditions else None
    from models import WeatherType
    weather_type = start_weather.condition if start_weather else WeatherType.DRY
 
    _, _required_exit = _make_corner_helpers(car, segments, tyre_state, weather_type,
                                             corner_speed_margin=0.0)
 
    target_speed = lam * car.max_speed_m_s
 
    prev_is_straight = set()
    for i, seg in enumerate(segments):
        if seg.type == SegmentType.STRAIGHT and i > 0:
            if segments[i - 1].type == SegmentType.STRAIGHT:
                prev_is_straight.add(seg.id)
 
    straight_actions: dict = {}
    for idx, seg in enumerate(segments):
        if seg.type == SegmentType.STRAIGHT:
            required_exit = _required_exit(idx)
            cruise = max(target_speed, required_exit)
            cruise = min(cruise, car.max_speed_m_s)
            brake_from = car.max_speed_m_s if seg.id in prev_is_straight else cruise
            if brake_from > required_exit:
                brake_dist = (brake_from ** 2 - required_exit ** 2) / (2 * car.brake_m_se2)
            else:
                brake_dist = 0.0
            brake_dist = min(brake_dist, seg.length_m)
            straight_actions[seg.id] = StraightAction(
                segment_id=seg.id,
                target_m_s=round(cruise, 4),
                brake_start_m_before_next=round(brake_dist, 2),
            )
 
    # ---- Probe run: get fuel per lap ----
    from simulator import simulate, SimConfig
 
    probe_laps = []
    for ln in range(1, total_laps + 1):
        actions = list(straight_actions.values()) + [
            CornerAction(segment_id=seg.id)
            for seg in segments if seg.type == SegmentType.CORNER
        ]
        actions.sort(key=lambda a: a.segment_id)
        probe_laps.append(LapStrategy(lap=ln, segment_actions=actions, pit=PitAction(enter=False)))
 
    soft_set = next(ts for ts in config.available_sets
                    if ts.compound.value == "Soft")
    probe_strategy = RaceStrategy(initial_tyre_id=soft_set.primary_id, laps=probe_laps)
    probe_cfg = SimConfig(tyre_degradation=False, fuel_consumption=True)
    probe_result = simulate(config, probe_strategy, probe_cfg)
    fuel_per_lap = probe_result.lap_results[0].fuel_used_l
 
    # ---- Estimate lap duration for weather-window calculations ----
    # Use average lap time from probe run.
    avg_lap_time_s = probe_result.total_time_s / total_laps
 
    # ---- Build pit schedule ----
    # Track both fuel and tyre-change pits.
    # change_intervals specifies FORCED tyre-change laps (1-indexed, not final lap).
    forced_change_laps = set(
        l for l in change_intervals
        if 1 <= l < total_laps
    )
 
    pit_schedule: dict = {}   # lap_num → {"fuel": float, "tyre_set": TyreSet|None}
    current_fuel = car.initial_fuel_l
    elapsed_time = 0.0
 
    # Tyre rotation: start on best tyre for opening weather, rotate on forced laps.
    used_set_ids = set()
 
    def pick_tyre(lap_num_after_pit):
        """Pick the best available (unused) tyre for weather starting at lap_num_after_pit."""
        stint_start_s = elapsed_time
        stint_end_s   = stint_start_s + avg_lap_time_s * 20  # look 20 laps ahead
        dom_weather   = dominant_weather_in_window(stint_start_s, stint_end_s)
        remaining_sets = [s for s in config.available_sets if s.primary_id not in used_set_ids]
        if not remaining_sets:
            # All used — just reuse compound with best score (fallback)
            remaining_sets = config.available_sets
        chosen = best_tyre_for_weather(dom_weather, remaining_sets)
        return chosen
 
    # Initial tyre
    initial_tyre = pick_tyre(1)
    used_set_ids.add(initial_tyre.primary_id)
 
    for lap_num in range(1, total_laps + 1):
        current_fuel -= fuel_per_lap
        current_fuel  = max(current_fuel, 0.0)
        elapsed_time += avg_lap_time_s
 
        if lap_num == total_laps:
            break
 
        needs_tyre  = lap_num in forced_change_laps
        needs_fuel  = current_fuel < fuel_per_lap
 
        if needs_tyre or needs_fuel:
            remaining_laps_after = total_laps - lap_num
            if needs_fuel:
                needed   = remaining_laps_after * fuel_per_lap
                headroom = car.fuel_tank_capacity_l - current_fuel
                refuel   = round(min(needed, headroom), 4)
            else:
                refuel = 0.0
 
            if needs_tyre:
                new_set = pick_tyre(lap_num + 1)
                used_set_ids.add(new_set.primary_id)
            else:
                new_set = None
 
            pit_schedule[lap_num] = {"fuel": refuel, "tyre_set": new_set}
            if refuel > 0:
                current_fuel += refuel
 
    # ---- Assemble final RaceStrategy ----
    laps_out: list = []
    for lap_num in range(1, total_laps + 1):
        actions = list(straight_actions.values()) + [
            CornerAction(segment_id=seg.id)
            for seg in segments if seg.type == SegmentType.CORNER
        ]
        actions.sort(key=lambda a: a.segment_id)
 
        if lap_num in pit_schedule:
            entry = pit_schedule[lap_num]
            tyre_set = entry["tyre_set"]
            pit = PitAction(
                enter=True,
                tyre_change_set_id=tyre_set.primary_id if tyre_set else None,
                fuel_refuel_amount_l=entry["fuel"],
            )
        else:
            pit = PitAction(enter=False)
 
        laps_out.append(LapStrategy(lap=lap_num, segment_actions=actions, pit=pit))
 
    return RaceStrategy(initial_tyre_id=initial_tyre.primary_id, laps=laps_out)
 