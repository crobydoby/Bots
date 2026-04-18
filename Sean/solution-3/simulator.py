"""
simulator.py
------------
Physics engine for the Entelect Grand Prix race simulation.

Given a LevelConfig and a RaceStrategy, simulates the full race and returns
a SimResult containing total race time, fuel used, tyre degradation,
blowout count, and a per-lap breakdown.

All physics formulas are taken verbatim from the problem statement.

Level modes
-----------
Pass a SimConfig to simulate() to control which mechanics are active:

    SimConfig(tyre_degradation=False, fuel_consumption=False)

This disables tyre degradation and fuel consumption entirely, leaving only
straight-speed / braking-point optimisation (Level 1 without degradation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from models import (
    Car, LevelConfig, Segment, SegmentType, TyreState,
    WeatherType, WeatherCondition,
    GRAVITY, K_STRAIGHT, K_BRAKING, K_CORNER, K_BASE, K_DRAG,
)
from strategy import RaceStrategy, StraightAction, CornerAction, PitAction, LapStrategy


# ---------------------------------------------------------------------------
# Simulation configuration flags
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    Controls which mechanics are active during simulation.

    Attributes
    ----------
    tyre_degradation : bool
        When False, tyres never degrade and blowouts cannot occur.
        Corner max speeds are computed from the tyre's full life_span friction,
        and remain constant for the entire race.  Default: True.
    fuel_consumption : bool
        When False, fuel is treated as unlimited — consumption is not tracked
        and the car never enters limp mode due to empty fuel.  Default: True.
    """
    tyre_degradation: bool = True
    fuel_consumption: bool = True

    @classmethod
    def simple(cls) -> "SimConfig":
        """Level 1 variant: no tyre degradation, unlimited fuel.
        Only straight-speed and braking-point decisions matter."""
        return cls(tyre_degradation=False, fuel_consumption=False)

    @classmethod
    def full(cls) -> "SimConfig":
        """Full simulation: all mechanics active (Levels 2–4)."""
        return cls(tyre_degradation=True, fuel_consumption=True)


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    segment_id: int
    segment_type: str
    time_s: float
    fuel_used_l: float
    tyre_degradation: float
    entry_speed_m_s: float
    exit_speed_m_s: float
    crashed: bool = False
    limp_mode: bool = False
    crawl_mode: bool = False


@dataclass
class LapResult:
    lap: int
    time_s: float                          # Driving time this lap (excl. pit stop)
    pit_time_s: float                      # Pit stop time (0 if no pit)
    total_time_s: float                    # time_s + pit_time_s
    fuel_used_l: float
    total_tyre_degradation: float
    blowouts: int
    crashes: int
    segment_results: List[SegmentResult] = field(default_factory=list)


@dataclass
class SimResult:
    total_time_s: float
    total_fuel_used_l: float
    total_tyre_degradation: float
    blowouts: int
    crashes: int
    lap_results: List[LapResult] = field(default_factory=list)

    # Scoring helpers
    def base_score(self, time_reference_s: float) -> float:
        return 500_000 * (time_reference_s / self.total_time_s) ** 3

    def fuel_bonus(self, fuel_soft_cap_limit_l: float) -> float:
        return -500_000 * (1 - self.total_fuel_used_l / fuel_soft_cap_limit_l) ** 2 + 500_000

    def tyre_bonus(self) -> float:
        return (100_000 * self.total_tyre_degradation) - (50_000 * self.blowouts)

    def score_level1(self, cfg: LevelConfig) -> float:
        return self.base_score(cfg.race.time_reference_s)

    def score_level2(self, cfg: LevelConfig) -> float:
        return self.base_score(cfg.race.time_reference_s) + self.fuel_bonus(cfg.race.fuel_soft_cap_limit_l)

    def score_level3(self, cfg: LevelConfig) -> float:
        return self.score_level2(cfg)

    def score_level4(self, cfg: LevelConfig) -> float:
        return self.score_level2(cfg) + self.tyre_bonus()


# ---------------------------------------------------------------------------
# Internal simulation state
# ---------------------------------------------------------------------------

@dataclass
class _RaceState:
    speed_m_s: float        # Current car speed
    fuel_l: float           # Remaining fuel
    elapsed_s: float        # Total elapsed race time (seconds)
    tyre: TyreState         # Active tyre + degradation
    limp_mode: bool = False
    crawl_mode: bool = False
    blowouts: int = 0
    crashes: int = 0
    total_fuel_used: float = 0.0
    total_degradation: float = 0.0


# ---------------------------------------------------------------------------
# Core physics helpers
# ---------------------------------------------------------------------------

def _accel_time(v_initial: float, v_final: float, accel: float) -> float:
    """Time to accelerate from v_initial to v_final at constant accel."""
    if accel <= 0 or v_final <= v_initial:
        return 0.0
    return (v_final - v_initial) / accel


def _accel_distance(v_initial: float, v_final: float, accel: float) -> float:
    """Distance covered while accelerating from v_initial to v_final."""
    if accel <= 0 or v_final <= v_initial:
        return 0.0
    return (v_final ** 2 - v_initial ** 2) / (2 * accel)


def _time_at_constant(distance: float, speed: float) -> float:
    """Time to cover a distance at constant speed."""
    if speed <= 0:
        return float('inf')
    return distance / speed


def _fuel_used(v_initial: float, v_final: float, distance: float) -> float:
    """
    Fuel used formula:
      F = (K_base + K_drag * ((v_i + v_f) / 2)^2) * distance
    """
    avg_speed = (v_initial + v_final) / 2.0
    return (K_BASE + K_DRAG * avg_speed ** 2) * distance


def _max_corner_speed(tyre_friction: float, radius_m: float, crawl_speed: float) -> float:
    """max_corner_speed = sqrt(tyre_friction * gravity * radius) + crawl_constant"""
    return math.sqrt(max(tyre_friction * GRAVITY * radius_m, 0.0)) + crawl_speed


# ---------------------------------------------------------------------------
# Straight segment simulation
# ---------------------------------------------------------------------------

def _simulate_straight(
    seg: Segment,
    action: StraightAction,
    state: _RaceState,
    car: Car,
    weather: WeatherCondition,
    sim_cfg: SimConfig,
) -> SegmentResult:
    """
    Simulate one straight segment.

    The submission engine model (verified against ground-truth logs):
      Phase 1: Accelerate from entry speed to target_m_s.
      Phase 2: Cruise at target_m_s.
      Phase 3: Brake over brake_start_m_before_next metres.
               exit_speed = sqrt(target_m_s² - 2 * brake * brake_dist)

    All three distances must sum to seg.length_m. If accel+brake > length
    (no room to reach target speed), the car peaks at the speed that can be
    reached and braked back to within the available distance.

    Tyre degradation is computed per-phase; fuel likewise.
    """
    seg_length  = seg.length_m
    entry_speed = state.speed_m_s
    limp        = state.limp_mode
    crawl       = state.crawl_mode

    accel = car.accel_m_se2 * weather.acceleration_multiplier
    brake = car.brake_m_se2 * weather.deceleration_multiplier

    # ---- Limp or crawl mode: constant speed, no accel/decel ----
    if limp:
        speed    = car.limp_constant_m_s
        seg_time = _time_at_constant(seg_length, speed)
        fuel     = _fuel_used(speed, speed, seg_length) if sim_cfg.fuel_consumption else 0.0
        deg      = _tyre_degrade_straight(seg_length, state.tyre, weather) if sim_cfg.tyre_degradation else 0.0

        _update_state(state, speed, fuel, seg_time, deg, sim_cfg)
        return SegmentResult(
            segment_id=seg.id, segment_type="straight",
            time_s=seg_time, fuel_used_l=fuel,
            tyre_degradation=deg,
            entry_speed_m_s=entry_speed, exit_speed_m_s=speed,
            limp_mode=True,
        )

    if crawl:
        # Crawl mode ends on encountering a straight; car resumes from crawl speed
        state.crawl_mode = False
        entry_speed = car.crawl_constant_m_s
        state.speed_m_s = entry_speed

    # ---- Normal straight ----
    target_speed = min(action.target_m_s, car.max_speed_m_s)
    # Speed follow-through: if entry already exceeds target, treat as cruise at entry
    cruise_speed = max(target_speed, entry_speed)
    brake_dist   = action.brake_start_m_before_next   # metres at end of segment

    # Distance to accelerate from entry to cruise_speed
    if entry_speed < cruise_speed and accel > 0:
        accel_dist = (cruise_speed ** 2 - entry_speed ** 2) / (2 * accel)
    else:
        accel_dist = 0.0

    cruise_dist = max(0.0, seg_length - accel_dist - brake_dist)

    # Exit speed from braking phase (submission engine formula)
    v_after_brake = math.sqrt(max(cruise_speed ** 2 - 2 * brake * brake_dist, 0.0))
    exit_speed    = v_after_brake

    # Times per phase
    t_accel  = (cruise_speed - entry_speed) / accel if accel > 0 and entry_speed < cruise_speed else 0.0
    t_cruise = cruise_dist / cruise_speed if cruise_speed > 0 else 0.0
    t_brake  = (cruise_speed - exit_speed) / brake if brake > 0 and cruise_speed > exit_speed else 0.0
    total_time = t_accel + t_cruise + t_brake

    # Fuel and degradation per phase
    total_fuel = 0.0
    total_deg  = 0.0

    if sim_cfg.fuel_consumption:
        total_fuel = (_fuel_used(entry_speed, cruise_speed, accel_dist)
                      + _fuel_used(cruise_speed, cruise_speed, cruise_dist)
                      + _fuel_used(cruise_speed, exit_speed, brake_dist))

    if sim_cfg.tyre_degradation:
        total_deg = (_tyre_degrade_straight(accel_dist + cruise_dist, state.tyre, weather)
                     + _tyre_degrade_braking(cruise_speed, exit_speed, state.tyre, weather))

    # Fuel check (only when fuel consumption is active)
    if sim_cfg.fuel_consumption and state.fuel_l - total_fuel <= 0:
        state.limp_mode = True
        total_fuel = state.fuel_l

    _update_state(state, exit_speed, total_fuel, total_time, total_deg, sim_cfg)

    return SegmentResult(
        segment_id=seg.id, segment_type="straight",
        time_s=total_time, fuel_used_l=total_fuel,
        tyre_degradation=total_deg,
        entry_speed_m_s=entry_speed, exit_speed_m_s=exit_speed,
    )


# ---------------------------------------------------------------------------
# Corner segment simulation
# ---------------------------------------------------------------------------

def _simulate_corner(
    seg: Segment,
    state: _RaceState,
    car: Car,
    weather: WeatherCondition,
    sim_cfg: SimConfig,
) -> SegmentResult:
    """
    Simulate one corner.
    Speed is constant throughout a corner (= entry speed).
    If entry speed > max_corner_speed → crash.

    When sim_cfg.tyre_degradation is False, the tyre's full life_span friction
    is used for the corner speed limit (no degradation ever occurred), and no
    degradation is accumulated.
    """
    entry_speed = state.speed_m_s
    crashed     = False

    # Use current (possibly degraded) friction, or full life_span friction when
    # degradation is disabled.
    if sim_cfg.tyre_degradation:
        tyre_friction = state.tyre.current_friction(weather.condition)
    else:
        from models import TYRE_PROPERTIES
        props = TYRE_PROPERTIES[state.tyre.tyre_set.compound]
        tyre_friction = props.life_span * props.friction_multiplier(weather.condition)

    max_speed = _max_corner_speed(tyre_friction, seg.radius_m, car.crawl_constant_m_s)

    # Crash tolerance: the submission engine rounds brake_dist to 2 decimal places,
    # which can produce an exit speed up to ~0.003 m/s above corner_max. The engine
    # does not penalise this rounding artifact — it only crashes on genuine overspeeds.
    CRASH_TOLERANCE = 0.005  # m/s — safely above rounding noise, below any real overspeed

    if state.limp_mode:
        corner_speed = car.limp_constant_m_s
    elif state.crawl_mode:
        corner_speed = car.crawl_constant_m_s
    elif entry_speed > max_speed + CRASH_TOLERANCE:
        crashed = True
        state.crashes += 1
        state.crawl_mode = True
        corner_speed = car.crawl_constant_m_s
        if sim_cfg.tyre_degradation:
            state.tyre.total_degradation += 0.1  # crash penalty
    else:
        # Use entry speed directly — including the tiny rounding-artefact overshoot.
        # The submission engine does not clamp entry speed to corner_max when the
        # excess is within the ~0.002 m/s rounding tolerance from 2dp brake_dist.
        corner_speed = entry_speed

    seg_time = _time_at_constant(seg.length_m, corner_speed) if corner_speed > 0 else float('inf')
    fuel     = _fuel_used(corner_speed, corner_speed, seg.length_m) if sim_cfg.fuel_consumption else 0.0
    deg      = _tyre_degrade_corner(corner_speed, seg.radius_m, state.tyre, weather) if sim_cfg.tyre_degradation else 0.0

    # Tyre blowout check (only when degradation is active)
    if sim_cfg.tyre_degradation and state.tyre.is_blown():
        state.limp_mode = True
        state.blowouts += 1

    # Fuel empty check (only when fuel consumption is active)
    if sim_cfg.fuel_consumption and state.fuel_l - fuel <= 0:
        state.limp_mode = True
        fuel = state.fuel_l

    _update_state(state, corner_speed, fuel, seg_time, deg, sim_cfg)

    return SegmentResult(
        segment_id=seg.id, segment_type="corner",
        time_s=seg_time, fuel_used_l=fuel,
        tyre_degradation=deg,
        entry_speed_m_s=entry_speed, exit_speed_m_s=corner_speed,
        crashed=crashed,
        limp_mode=state.limp_mode,
        crawl_mode=state.crawl_mode,
    )


# ---------------------------------------------------------------------------
# Tyre degradation helpers
# ---------------------------------------------------------------------------

def _tyre_degrade_straight(distance: float, tyre: TyreState, weather: WeatherCondition) -> float:
    """Total Straight Degradation = tyre_degradation_rate × length × K_STRAIGHT"""
    rate = tyre.tyre_set  # grab compound to get rate
    from models import TYRE_PROPERTIES
    props = TYRE_PROPERTIES[tyre.tyre_set.compound]
    deg_rate = props.degradation_rate(weather.condition)
    return deg_rate * distance * K_STRAIGHT


def _tyre_degrade_braking(v_initial: float, v_final: float, tyre: TyreState, weather: WeatherCondition) -> float:
    """
    Degradation while Braking =
      ((v_i/100)^2 - (v_f/100)^2) × K_BRAKING × tyre_degradation_rate
    """
    from models import TYRE_PROPERTIES
    props    = TYRE_PROPERTIES[tyre.tyre_set.compound]
    deg_rate = props.degradation_rate(weather.condition)
    return ((v_initial / 100) ** 2 - (v_final / 100) ** 2) * K_BRAKING * deg_rate


def _tyre_degrade_corner(speed: float, radius_m: float, tyre: TyreState, weather: WeatherCondition) -> float:
    """Total Corner Degradation = K_CORNER × (speed^2 / radius) × tyre_degradation_rate"""
    from models import TYRE_PROPERTIES
    props    = TYRE_PROPERTIES[tyre.tyre_set.compound]
    deg_rate = props.degradation_rate(weather.condition)
    if radius_m and radius_m > 0:
        return K_CORNER * (speed ** 2 / radius_m) * deg_rate
    return 0.0


# ---------------------------------------------------------------------------
# State updater
# ---------------------------------------------------------------------------

def _update_state(
    state: _RaceState,
    exit_speed: float,
    fuel_used: float,
    time_s: float,
    degradation: float,
    sim_cfg: SimConfig,
) -> None:
    state.speed_m_s  = exit_speed
    state.elapsed_s += time_s

    if sim_cfg.fuel_consumption:
        state.fuel_l           = max(0.0, state.fuel_l - fuel_used)
        state.total_fuel_used += fuel_used

    if sim_cfg.tyre_degradation:
        state.tyre.total_degradation += degradation
        state.total_degradation      += degradation
        # Check blowout after degradation update
        if state.tyre.is_blown() and not state.limp_mode:
            state.limp_mode = True
            state.blowouts += 1


# ---------------------------------------------------------------------------
# Pit stop
# ---------------------------------------------------------------------------

def _simulate_pit(
    pit: PitAction,
    state: _RaceState,
    car: Car,
    cfg: LevelConfig,
) -> float:
    """Apply pit stop effects and return pit stop time (seconds)."""
    if not pit.enter:
        return 0.0

    # Actual refuel is capped by available tank space (submission engine caps it too)
    actual_refuel = 0.0
    if pit.fuel_refuel_amount_l > 0:
        space_in_tank = car.fuel_tank_capacity_l - state.fuel_l
        actual_refuel = min(pit.fuel_refuel_amount_l, max(0.0, space_in_tank))

    refuel_time  = actual_refuel / cfg.race.pit_refuel_rate_l_s if actual_refuel > 0 else 0.0
    tyre_time    = cfg.race.pit_tyre_swap_time_s if pit.tyre_change_set_id is not None else 0.0
    base_time    = cfg.race.base_pit_stop_time_s
    pit_duration = refuel_time + tyre_time + base_time

    # Refuel — add actual amount to tank; do NOT add to total_fuel_used
    # (that counter tracks only fuel consumed during driving, not pit refuels)
    if actual_refuel > 0:
        state.fuel_l = min(car.fuel_tank_capacity_l, state.fuel_l + actual_refuel)

    # Tyre change
    if pit.tyre_change_set_id is not None:
        new_set = cfg.get_tyre_set_by_id(pit.tyre_change_set_id)
        state.tyre = TyreState(tyre_set=new_set)
        state.limp_mode  = False   # blowout resolved
        state.crawl_mode = False

    # Exit pit at pit_exit_speed
    state.speed_m_s = cfg.race.pit_exit_speed_m_s
    state.elapsed_s += pit_duration

    return pit_duration


# ---------------------------------------------------------------------------
# Main simulator entry point
# ---------------------------------------------------------------------------

def simulate(
    cfg: LevelConfig,
    strategy: RaceStrategy,
    sim_cfg: Optional[SimConfig] = None,
) -> SimResult:
    """
    Simulate the full race.

    Parameters
    ----------
    cfg     : LevelConfig
    strategy: RaceStrategy
    sim_cfg : SimConfig, optional
        Controls which mechanics are active.  Defaults to SimConfig.full()
        (all mechanics on).  Pass SimConfig.simple() for levels where tyre
        degradation and fuel consumption are not factors.

    Returns
    -------
    SimResult
    """
    if sim_cfg is None:
        sim_cfg = SimConfig.full()

    car = cfg.car

    # Initialise state
    initial_set = cfg.get_tyre_set_by_id(strategy.initial_tyre_id)
    state = _RaceState(
        speed_m_s=0.0,
        fuel_l=car.initial_fuel_l,
        elapsed_s=0.0,
        tyre=TyreState(tyre_set=initial_set),
    )

    # Build a lookup: (lap, segment_id) → action
    def _action_map(lap_strat: LapStrategy):
        return {a.segment_id: a for a in lap_strat.segment_actions}

    lap_results: List[LapResult] = []

    for lap_strat in strategy.laps:
        lap_num  = lap_strat.lap
        actions  = _action_map(lap_strat)
        seg_results: List[SegmentResult] = []
        lap_time   = 0.0
        lap_fuel   = 0.0
        lap_deg    = 0.0
        lap_blowouts = 0
        lap_crashes  = 0

        blowouts_before = state.blowouts
        crashes_before  = state.crashes

        for seg in cfg.track.segments:
            weather = cfg.get_weather_at_time(state.elapsed_s)

            if seg.type == SegmentType.STRAIGHT:
                action = actions.get(seg.id)
                if action is None or not isinstance(action, StraightAction):
                    # Fallback: cruise at max speed, no braking
                    action = StraightAction(
                        segment_id=seg.id,
                        target_m_s=car.max_speed_m_s,
                        brake_start_m_before_next=0.0,
                    )
                result = _simulate_straight(seg, action, state, car, weather, sim_cfg)
            else:
                result = _simulate_corner(seg, state, car, weather, sim_cfg)

            seg_results.append(result)
            lap_time += result.time_s
            lap_fuel += result.fuel_used_l
            lap_deg  += result.tyre_degradation

        pit_time = _simulate_pit(lap_strat.pit, state, car, cfg)

        lap_blowouts = state.blowouts - blowouts_before
        lap_crashes  = state.crashes  - crashes_before

        lap_results.append(LapResult(
            lap=lap_num,
            time_s=lap_time,
            pit_time_s=pit_time,
            total_time_s=lap_time + pit_time,
            fuel_used_l=lap_fuel,
            total_tyre_degradation=lap_deg,
            blowouts=lap_blowouts,
            crashes=lap_crashes,
            segment_results=seg_results,
        ))

    # Add crash-time penalties to total elapsed
    total_penalty = state.crashes * cfg.race.corner_crash_penalty_s
    total_time    = state.elapsed_s + total_penalty

    return SimResult(
        total_time_s=total_time,
        total_fuel_used_l=state.total_fuel_used,
        total_tyre_degradation=state.total_degradation,
        blowouts=state.blowouts,
        crashes=state.crashes,
        lap_results=lap_results,
    )
