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

def build_naive_strategy(config) -> RaceStrategy:
    """
    Build a simple baseline strategy:
      - Start on the first available tyre set.
      - On every straight: target max car speed, brake as late as safely possible.
      - No pit stops.

    Parameters
    ----------
    config : LevelConfig
        Loaded level configuration.

    Returns
    -------
    RaceStrategy
    """
    from models import SegmentType, WeatherType, GRAVITY, TyreState

    initial_set = config.available_sets[0]
    tyre_state  = TyreState(tyre_set=initial_set)

    # Use the starting weather to estimate corner entry speed limits
    start_weather = config.weather_conditions[0] if config.weather_conditions else None
    weather_type  = start_weather.condition if start_weather else WeatherType.DRY

    car      = config.car
    segments = config.track.segments

    laps: List[LapStrategy] = []

    for lap_num in range(1, config.race.laps + 1):
        segment_actions = []

        for idx, seg in enumerate(segments):
            if seg.type == SegmentType.STRAIGHT:
                # Determine required entry speed for the next segment (if a corner)
                next_seg = segments[idx + 1] if idx + 1 < len(segments) else None

                corners = []
                if next_seg and next_seg.type == SegmentType.CORNER:
                    check_idx = idx + 1
                    while check_idx < len(segments):
                        check_next = segments[check_idx]
                        if check_next.type != SegmentType.CORNER:
                            break
                        corners.append(check_next)
                        check_idx += 1

                if corners:
                    corner_radii = [c.radius_m for c in corners if c.radius_m is not None and c.radius_m > 0]
                    if corner_radii:
                        friction = tyre_state.current_friction(weather_type)
                        min_corner_radius = min(corner_radii)
                        max_corner = (
                            (friction * GRAVITY * min_corner_radius) ** 0.5
                            + car.crawl_constant_m_s
                        )
                        corner_entry_speed = min(max_corner, car.max_speed_m_s)

                        # Distance required to brake from max_speed to the limiting corner speed
                        target_speed = car.max_speed_m_s
                        v_i = target_speed
                        v_f = corner_entry_speed
                        if v_i > v_f:
                            # d = (v_i² - v_f²) / (2 * brake)
                            brake_dist = (v_i ** 2 - v_f ** 2) / (2 * car.brake_m_se2)
                        else:
                            brake_dist = 0.0

                        brake_dist = min(brake_dist, seg.length_m)
                    else:
                        target_speed = car.max_speed_m_s
                        brake_dist = 0.0
                else:
                    target_speed = car.max_speed_m_s
                    brake_dist   = 0.0

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
