"""
level_loader.py
---------------
Parses a level JSON file (e.g. 1.txt) into strongly-typed model objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from models import (
    Car, Race, Track, Segment, SegmentType,
    TyreProperties, TyreCompound, TyreSet,
    WeatherCondition, WeatherType,
    LevelConfig, TYRE_PROPERTIES,
)


def _parse_car(data: Dict[str, Any]) -> Car:
    return Car(
        max_speed_m_s=data["max_speed_m/s"],
        accel_m_se2=data["accel_m/se2"],
        brake_m_se2=data["brake_m/se2"],
        limp_constant_m_s=data["limp_constant_m/s"],
        crawl_constant_m_s=data["crawl_constant_m/s"],
        fuel_tank_capacity_l=data["fuel_tank_capacity_l"],
        initial_fuel_l=data["initial_fuel_l"],
    )


def _parse_race(data: Dict[str, Any]) -> Race:
    return Race(
        name=data["name"],
        laps=data["laps"],
        base_pit_stop_time_s=data["base_pit_stop_time_s"],
        pit_tyre_swap_time_s=data["pit_tyre_swap_time_s"],
        pit_refuel_rate_l_s=data["pit_refuel_rate_l/s"],
        corner_crash_penalty_s=data["corner_crash_penalty_s"],
        pit_exit_speed_m_s=data["pit_exit_speed_m/s"],
        fuel_soft_cap_limit_l=data["fuel_soft_cap_limit_l"],
        starting_weather_condition_id=data["starting_weather_condition_id"],
        time_reference_s=data["time_reference_s"],
    )


def _parse_track(data: Dict[str, Any]) -> Track:
    segments = []
    for s in data["segments"]:
        seg = Segment(
            id=s["id"],
            type=SegmentType(s["type"]),
            length_m=s["length_m"],
            radius_m=s.get("radius_m"),
        )
        segments.append(seg)
    return Track(name=data["name"], segments=segments)


def _parse_tyre_properties(data: Dict[str, Any]) -> Dict[TyreCompound, TyreProperties]:
    """
    Parse the tyres.properties section of the JSON.
    Falls back to the default TYRE_PROPERTIES table for any missing compound.
    """
    props: Dict[TyreCompound, TyreProperties] = dict(TYRE_PROPERTIES)  # start with defaults

    props_data = data.get("tyres", {}).get("properties", {})
    for compound_name, values in props_data.items():
        compound = TyreCompound(compound_name)
        props[compound] = TyreProperties(
            compound=compound,
            life_span=values["life_span"],
            dry_friction_multiplier=values["dry_friction_multiplier"],
            cold_friction_multiplier=values["cold_friction_multiplier"],
            light_rain_friction_multiplier=values["light_rain_friction_multiplier"],
            heavy_rain_friction_multiplier=values["heavy_rain_friction_multiplier"],
            dry_degradation=values["dry_degradation"],
            cold_degradation=values["cold_degradation"],
            light_rain_degradation=values["light_rain_degradation"],
            heavy_rain_degradation=values["heavy_rain_degradation"],
        )
    return props


def _parse_available_sets(data: Dict[str, Any]) -> list[TyreSet]:
    sets = []
    # The available_sets key may be at the top level or inside "tyres"
    raw = data.get("available_sets") or data.get("tyres", {}).get("available_sets", [])
    for entry in raw:
        sets.append(TyreSet(
            ids=list(entry["ids"]),
            compound=TyreCompound(entry["compound"]),
        ))
    return sets


def _parse_weather(data: Dict[str, Any]) -> list[WeatherCondition]:
    conditions = []
    weather_data = data.get("weather", {}).get("conditions", [])
    for w in weather_data:
        conditions.append(WeatherCondition(
            id=w["id"],
            condition=WeatherType(w["condition"]),
            duration_s=w["duration_s"],
            acceleration_multiplier=w["acceleration_multiplier"],
            deceleration_multiplier=w["deceleration_multiplier"],
        ))
    return conditions


def load_level(path: str | Path) -> LevelConfig:
    """
    Load a level JSON file and return a fully-populated LevelConfig.

    Parameters
    ----------
    path : str | Path
        Path to the level JSON file (e.g. "1.txt").

    Returns
    -------
    LevelConfig
    """
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)

    car        = _parse_car(data["car"])
    race       = _parse_race(data["race"])
    track      = _parse_track(data["track"])
    tyre_props = _parse_tyre_properties(data)
    avail_sets = _parse_available_sets(data)
    weather    = _parse_weather(data)

    return LevelConfig(
        car=car,
        race=race,
        track=track,
        tyre_properties=tyre_props,
        available_sets=avail_sets,
        weather_conditions=weather,
    )


if __name__ == "__main__":
    import sys
    level_path = sys.argv[1] if len(sys.argv) > 1 else "1.txt"
    config = load_level(level_path)
    print(f"Loaded: {config.race.name}")
    print(f"  Track : {config.track.name}  ({config.track.total_length_m:.0f} m/lap)")
    print(f"  Laps  : {config.race.laps}")
    print(f"  Segments: {len(config.track.segments)}")
    print(f"  Tyre sets available: {len(config.available_sets)}")
    print(f"  Weather conditions : {len(config.weather_conditions)}")
