"""
models.py
---------
Dataclasses representing all domain objects for the Entelect Grand Prix
race simulation. All values use SI units (metres, seconds, litres).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SegmentType(str, Enum):
    STRAIGHT = "straight"
    CORNER   = "corner"


class WeatherType(str, Enum):
    DRY         = "dry"
    COLD        = "cold"
    LIGHT_RAIN  = "light_rain"
    HEAVY_RAIN  = "heavy_rain"


class TyreCompound(str, Enum):
    SOFT         = "Soft"
    MEDIUM       = "Medium"
    HARD         = "Hard"
    INTERMEDIATE = "Intermediate"
    WET          = "Wet"


# ---------------------------------------------------------------------------
# Tyre degradation constants (from problem statement)
# ---------------------------------------------------------------------------

K_STRAIGHT = 0.0000166
K_BRAKING  = 0.0398
K_CORNER   = 0.000265

GRAVITY    = 9.8          # m/s²

# Fuel consumption constants
K_BASE = 0.0005           # l/m  (base consumption)
K_DRAG = 0.0000000015     # l/m  (speed-dependent drag consumption)


# ---------------------------------------------------------------------------
# Car
# ---------------------------------------------------------------------------

@dataclass
class Car:
    max_speed_m_s: float        # Maximum speed the car can reach (m/s)
    accel_m_se2: float          # Constant acceleration rate (m/s²)
    brake_m_se2: float          # Constant deceleration rate (m/s²)
    limp_constant_m_s: float    # Speed while in limp mode (m/s)
    crawl_constant_m_s: float   # Speed while in crawl mode (m/s)
    fuel_tank_capacity_l: float # Maximum fuel tank capacity (l)
    initial_fuel_l: float       # Fuel at race start (l)


# ---------------------------------------------------------------------------
# Track segments
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    id: int
    type: SegmentType
    length_m: float
    radius_m: Optional[float] = None   # Only for corners

    @property
    def is_straight(self) -> bool:
        return self.type == SegmentType.STRAIGHT

    @property
    def is_corner(self) -> bool:
        return self.type == SegmentType.CORNER


@dataclass
class Track:
    name: str
    segments: List[Segment]

    @property
    def straights(self) -> List[Segment]:
        return [s for s in self.segments if s.is_straight]

    @property
    def corners(self) -> List[Segment]:
        return [s for s in self.segments if s.is_corner]

    @property
    def total_length_m(self) -> float:
        return sum(s.length_m for s in self.segments)


# ---------------------------------------------------------------------------
# Tyres
# ---------------------------------------------------------------------------

@dataclass
class TyreProperties:
    """Weather-specific friction multipliers and degradation rates for one compound."""
    compound: TyreCompound
    life_span: float                        # Starting friction / life span value

    # Friction multipliers per weather condition
    dry_friction_multiplier: float
    cold_friction_multiplier: float
    light_rain_friction_multiplier: float
    heavy_rain_friction_multiplier: float

    # Degradation rates per weather condition
    dry_degradation: float
    cold_degradation: float
    light_rain_degradation: float
    heavy_rain_degradation: float

    def friction_multiplier(self, weather: WeatherType) -> float:
        return {
            WeatherType.DRY:        self.dry_friction_multiplier,
            WeatherType.COLD:       self.cold_friction_multiplier,
            WeatherType.LIGHT_RAIN: self.light_rain_friction_multiplier,
            WeatherType.HEAVY_RAIN: self.heavy_rain_friction_multiplier,
        }[weather]

    def degradation_rate(self, weather: WeatherType) -> float:
        return {
            WeatherType.DRY:        self.dry_degradation,
            WeatherType.COLD:       self.cold_degradation,
            WeatherType.LIGHT_RAIN: self.light_rain_degradation,
            WeatherType.HEAVY_RAIN: self.heavy_rain_degradation,
        }[weather]

    @property
    def base_friction_coefficient(self) -> float:
        """Compatibility alias used by older strategy code."""
        return self.life_span


@dataclass
class TyreSet:
    """A physical set of tyres identified by one or more IDs."""
    ids: List[int]
    compound: TyreCompound

    @property
    def primary_id(self) -> int:
        return self.ids[0]


@dataclass
class TyreState:
    """Runtime tyre state during a simulation."""
    tyre_set: TyreSet
    total_degradation: float = 0.0       # Accumulated degradation

    @property
    def life_remaining(self) -> float:
        """Remaining tyre life (blowout when <= 0)."""
        props = TYRE_PROPERTIES[self.tyre_set.compound]
        return props.life_span - self.total_degradation

    def is_blown(self) -> bool:
        return self.life_remaining <= 0.0

    def current_friction(self, weather: WeatherType) -> float:
        """tyre_friction = (base_friction_coeff - total_degradation) × weather_multiplier"""
        props = TYRE_PROPERTIES[self.tyre_set.compound]
        base = props.life_span  # life_span == base_friction_coefficient
        multiplier = props.friction_multiplier(weather)
        return (base - self.total_degradation) * multiplier


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

@dataclass
class WeatherCondition:
    id: int
    condition: WeatherType
    duration_s: float
    acceleration_multiplier: float
    deceleration_multiplier: float


# ---------------------------------------------------------------------------
# Race configuration
# ---------------------------------------------------------------------------

@dataclass
class Race:
    name: str
    laps: int
    base_pit_stop_time_s: float
    pit_tyre_swap_time_s: float
    pit_refuel_rate_l_s: float
    corner_crash_penalty_s: float
    pit_exit_speed_m_s: float
    fuel_soft_cap_limit_l: float
    starting_weather_condition_id: int
    time_reference_s: float


# ---------------------------------------------------------------------------
# Full level configuration (parsed from JSON)
# ---------------------------------------------------------------------------

@dataclass
class LevelConfig:
    car: Car
    race: Race
    track: Track
    tyre_properties: Dict[TyreCompound, TyreProperties]
    available_sets: List[TyreSet]
    weather_conditions: List[WeatherCondition]

    def get_weather_at_time(self, elapsed_s: float) -> WeatherCondition:
        """
        Return the active WeatherCondition at a given elapsed race time.
        Weather cycles back to the first condition once all have elapsed.
        """
        if not self.weather_conditions:
            # Default: dry with no multiplier effect
            return WeatherCondition(
                id=0, condition=WeatherType.DRY,
                duration_s=float('inf'),
                acceleration_multiplier=1.0,
                deceleration_multiplier=1.0
            )
        total_cycle = sum(w.duration_s for w in self.weather_conditions)
        t = elapsed_s % total_cycle
        for w in self.weather_conditions:
            if t < w.duration_s:
                return w
            t -= w.duration_s
        return self.weather_conditions[-1]

    def get_tyre_set_by_id(self, tyre_id: int) -> TyreSet:
        for ts in self.available_sets:
            if tyre_id in ts.ids:
                return ts
        raise ValueError(f"No tyre set with id {tyre_id}")

    def tyre_props(self, compound: TyreCompound) -> TyreProperties:
        return self.tyre_properties[compound]


# ---------------------------------------------------------------------------
# Default tyre properties table (from problem statement)
# These are the canonical values; the level JSON may override them.
# ---------------------------------------------------------------------------

TYRE_PROPERTIES: Dict[TyreCompound, TyreProperties] = {
    TyreCompound.SOFT: TyreProperties(
        compound=TyreCompound.SOFT,
        life_span=1.0,
        dry_friction_multiplier=1.18,
        cold_friction_multiplier=1.00,
        light_rain_friction_multiplier=0.92,
        heavy_rain_friction_multiplier=0.80,
        dry_degradation=0.11,
        cold_degradation=0.09,
        light_rain_degradation=0.12,
        heavy_rain_degradation=0.13,
    ),
    TyreCompound.MEDIUM: TyreProperties(
        compound=TyreCompound.MEDIUM,
        life_span=1.0,
        dry_friction_multiplier=1.08,
        cold_friction_multiplier=0.97,
        light_rain_friction_multiplier=0.88,
        heavy_rain_friction_multiplier=0.74,
        dry_degradation=0.10,
        cold_degradation=0.08,
        light_rain_degradation=0.09,
        heavy_rain_degradation=0.10,
    ),
    TyreCompound.HARD: TyreProperties(
        compound=TyreCompound.HARD,
        life_span=1.0,
        dry_friction_multiplier=0.98,
        cold_friction_multiplier=0.92,
        light_rain_friction_multiplier=0.82,
        heavy_rain_friction_multiplier=0.68,
        dry_degradation=0.07,
        cold_degradation=0.06,
        light_rain_degradation=0.07,
        heavy_rain_degradation=0.08,
    ),
    TyreCompound.INTERMEDIATE: TyreProperties(
        compound=TyreCompound.INTERMEDIATE,
        life_span=1.0,
        dry_friction_multiplier=0.90,
        cold_friction_multiplier=0.96,
        light_rain_friction_multiplier=1.08,
        heavy_rain_friction_multiplier=1.02,
        dry_degradation=0.14,
        cold_degradation=0.11,
        light_rain_degradation=0.08,
        heavy_rain_degradation=0.09,
    ),
    TyreCompound.WET: TyreProperties(
        compound=TyreCompound.WET,
        life_span=1.0,
        dry_friction_multiplier=0.72,
        cold_friction_multiplier=0.88,
        light_rain_friction_multiplier=1.02,
        heavy_rain_friction_multiplier=1.20,
        dry_degradation=0.16,
        cold_degradation=0.12,
        light_rain_degradation=0.09,
        heavy_rain_degradation=0.05,
    ),
}
