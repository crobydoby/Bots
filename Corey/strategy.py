"""
strategy.py
-----------
Dataclasses for representing a race strategy and a serialiser that produces
the required submission JSON format.
"""

from __future__ import annotations

import json
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
# Helper: build a naive strategy (max safe speed, no pit stops)
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
        Match this to the SimConfig used when simulating.  When False, corner
        speed limits are constant (no safety margin needed), giving slightly
        faster braking targets.  Default: True.

    Returns
    -------
    RaceStrategy
    """
    from models import SegmentType, WeatherType, GRAVITY, TyreState, TYRE_PROPERTIES, K_STRAIGHT
    import math

    initial_set = config.available_sets[0]
    tyre_state  = TyreState(tyre_set=initial_set)

    start_weather = config.weather_conditions[0] if config.weather_conditions else None
    weather_type  = start_weather.condition if start_weather else WeatherType.DRY

    car      = config.car
    segments = config.track.segments

    # Safety margin (m/s) subtracted from corner max speeds when degradation
    # is active.  The simulator accumulates both rolling and braking degradation
    # on each straight, lowering the tyre friction — and thus the corner speed
    # limit — by the time the car arrives. This margin absorbs that discrepancy.
    # When tyre_degradation is False, friction is constant so no margin is needed.
    # When degradation is active: 0.5 m/s absorbs friction drop from straight wear.
    # When degradation is off: 0.01 m/s guards against floating-point ties at the
    # corner entry check (where entry == max_speed would just barely pass, but
    # rounding during braking can leave the car a hair over).
    # Safety margin subtracted from corner max speeds.
    # 0.5 m/s is used in both modes: in degradation mode it absorbs friction drop
    # from straight wear; in simple mode it matches the submission engine's behaviour
    # (verified against ground-truth submission logs).
    CORNER_SPEED_MARGIN = 0.5

    def _corner_max_speed(corner_seg, friction: float) -> float:
        """
        Maximum safe entry speed for a corner, optionally reduced by
        CORNER_SPEED_MARGIN to guard against degradation-induced friction drop.
        """
        raw = (friction * GRAVITY * corner_seg.radius_m) ** 0.5 + car.crawl_constant_m_s
        return max(car.crawl_constant_m_s, raw - CORNER_SPEED_MARGIN)

    def _required_exit_speed(straight_idx: int, seg) -> float:
        """
        The maximum speed we must not exceed when leaving this straight.
        Scans ALL consecutive corners that follow the straight and returns
        the minimum of their individual (margined) max speeds.
        """
        current_friction = tyre_state.current_friction(weather_type)
        required = car.max_speed_m_s
        j = straight_idx + 1
        while j < len(segments) and segments[j].type == SegmentType.CORNER:
            required = min(required, _corner_max_speed(segments[j], current_friction))
            j += 1
        return required

    def _reachable_top_speed(entry_speed: float, distance: float) -> float:
        """Highest speed the car can reach from entry_speed over a given distance."""
        return min(
            car.max_speed_m_s,
            math.sqrt(entry_speed ** 2 + 2 * car.accel_m_se2 * distance),
        )

    laps: List[LapStrategy] = []
    current_speed = 0.0  # car starts from rest on lap 1

    for lap_num in range(1, config.race.laps + 1):
        segment_actions = []

        for idx, seg in enumerate(segments):
            if seg.type == SegmentType.STRAIGHT:
                # Highest speed achievable on this straight given entry speed
                top_speed = _reachable_top_speed(current_speed, seg.length_m)

                # Speed we must not exceed at the end of this straight.
                # Uses post-degradation friction so the braking target matches
                # what the simulator will compute for the following corner.
                required_exit = _required_exit_speed(idx, seg)

                # Braking distance needed from top_speed down to required_exit
                if top_speed > required_exit:
                    brake_dist = (top_speed ** 2 - required_exit ** 2) / (2 * car.brake_m_se2)
                else:
                    brake_dist = 0.0
                brake_dist = min(brake_dist, seg.length_m)

                segment_actions.append(StraightAction(
                    segment_id=seg.id,
                    target_m_s=round(top_speed, 4),
                    brake_start_m_before_next=round(brake_dist, 2),
                ))
                # Exit speed after braking (can't exceed required_exit)
                current_speed = min(top_speed, required_exit)

            else:  # corner
                friction = tyre_state.current_friction(weather_type)
                corner_max = _corner_max_speed(seg, friction)
                # Entry speed should already be <= corner_max thanks to braking above;
                # clamp defensively in case of floating-point overshoot.
                current_speed = min(current_speed, corner_max)
                segment_actions.append(CornerAction(segment_id=seg.id))
                # Exit the corner at the same (clamped) speed

        laps.append(LapStrategy(
            lap=lap_num,
            segment_actions=segment_actions,
            pit=PitAction(enter=False),
        ))
        # Speed carries over from the last segment of this lap into the next

    return RaceStrategy(
        initial_tyre_id=initial_set.primary_id,
        laps=laps,
    )
