"""
simulator.py
------------
Physics engine for the Entelect Grand Prix race simulation.

Given a LevelConfig and a RaceStrategy, simulates the full race and returns
a SimResult containing total race time, fuel used, tyre degradation,
blowout count, and a per-lap breakdown.

All physics formulas are taken verbatim from the problem statement.
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
) -> SegmentResult:
    """
    Simulate one straight segment.

    Phase 1: Accelerate from entry speed toward target_m_s (if not in limp/crawl).
    Phase 2: Cruise at target_m_s (or entry speed if speed-follow-through applies).
    Phase 3: Brake from braking_point to end of segment.
    """
    seg_length   = seg.length_m
    entry_speed  = state.speed_m_s
    crashed      = False
    limp         = state.limp_mode
    crawl        = state.crawl_mode

    accel = car.accel_m_se2 * weather.acceleration_multiplier
    brake = car.brake_m_se2 * weather.deceleration_multiplier

    # ---- Limp or crawl mode: constant speed, no accel/decel ----
    if limp:
        speed       = car.limp_constant_m_s
        seg_time    = _time_at_constant(seg_length, speed)
        fuel        = _fuel_used(speed, speed, seg_length)
        deg         = _tyre_degrade_straight(seg_length, state.tyre, weather)

        _update_state(state, speed, fuel, seg_time, deg)
        return SegmentResult(
            segment_id=seg.id, segment_type="straight",
            time_s=seg_time, fuel_used_l=fuel,
            tyre_degradation=deg,
            entry_speed_m_s=entry_speed, exit_speed_m_s=speed,
            limp_mode=True,
        )

    if crawl:
        # Crawl mode: constant speed until this straight, then can accelerate again
        # The problem states crawl mode ends on encountering a straight
        state.crawl_mode = False
        crawl = False
        entry_speed = car.crawl_constant_m_s
        state.speed_m_s = entry_speed

    # ---- Normal straight ----
    target_speed   = min(action.target_m_s, car.max_speed_m_s)
    brake_dist     = action.brake_start_m_before_next   # metres before segment end
    brake_start_at = seg_length - brake_dist             # position along segment

    # Speed follow-through: if entry speed > target speed, just continue at entry speed
    cruise_speed = max(target_speed, entry_speed)

    total_time = 0.0
    total_fuel = 0.0
    total_deg  = 0.0
    pos        = 0.0          # current position along segment (m)
    cur_speed  = entry_speed

    # -- Phase 1: acceleration --
    if cur_speed < cruise_speed and pos < brake_start_at:
        d_accel = min(
            _accel_distance(cur_speed, cruise_speed, accel),
            brake_start_at - pos,
        )
        t_accel     = _accel_time(cur_speed, cur_speed + (2 * accel * d_accel) ** 0.5 if d_accel > 0 else cur_speed, accel)
        # Recalculate actual speed reached given distance available
        v_end_accel = min(
            cruise_speed,
            math.sqrt(cur_speed ** 2 + 2 * accel * min(d_accel, brake_start_at - pos))
        )
        d_accel_actual = (v_end_accel ** 2 - cur_speed ** 2) / (2 * accel) if accel > 0 else 0.0
        t_accel        = _accel_time(cur_speed, v_end_accel, accel)

        total_fuel += _fuel_used(cur_speed, v_end_accel, d_accel_actual)
        total_deg  += _tyre_degrade_straight(d_accel_actual, state.tyre, weather)
        total_time += t_accel
        pos        += d_accel_actual
        cur_speed   = v_end_accel

    # -- Phase 2: cruise --
    cruise_dist = max(0.0, brake_start_at - pos)
    if cruise_dist > 0 and cur_speed > 0:
        total_time += _time_at_constant(cruise_dist, cur_speed)
        total_fuel += _fuel_used(cur_speed, cur_speed, cruise_dist)
        total_deg  += _tyre_degrade_straight(cruise_dist, state.tyre, weather)
        pos        += cruise_dist

    # -- Phase 3: braking --
    brake_seg_len = seg_length - pos
    if brake_seg_len > 0 and brake > 0 and cur_speed > 0:
        # We decelerate to whatever speed physics allows over brake_seg_len
        v_end_brake = max(
            math.sqrt(max(cur_speed ** 2 - 2 * brake * brake_seg_len, 0.0)),
            0.0
        )
        t_brake = (cur_speed - v_end_brake) / brake if brake > 0 else 0.0

        # Tyre braking degradation
        deg_brake = _tyre_degrade_braking(cur_speed, v_end_brake, state.tyre, weather)

        total_time += t_brake
        total_fuel += _fuel_used(cur_speed, v_end_brake, brake_seg_len)
        total_deg  += deg_brake
        cur_speed   = v_end_brake
        pos        += brake_seg_len

    exit_speed = cur_speed

    # Check fuel
    if state.fuel_l - total_fuel <= 0:
        state.limp_mode = True
        total_fuel = state.fuel_l  # can't use more than we have

    _update_state(state, exit_speed, total_fuel, total_time, total_deg)

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
) -> SegmentResult:
    """
    Simulate one corner.
    Speed is constant throughout a corner (= entry speed).
    If entry speed > max_corner_speed → crash.
    """
    entry_speed = state.speed_m_s
    crashed     = False

    tyre_friction = state.tyre.current_friction(weather.condition)
    max_speed     = _max_corner_speed(tyre_friction, seg.radius_m, car.crawl_constant_m_s)

    if state.limp_mode:
        corner_speed = car.limp_constant_m_s
    elif state.crawl_mode:
        corner_speed = car.crawl_constant_m_s
    elif entry_speed > max_speed:
        # Crash: crawl mode + penalty tyre degradation
        crashed = True
        state.crashes += 1
        state.crawl_mode = True
        corner_speed = car.crawl_constant_m_s
        # Flat 0.1 degradation penalty on crash
        state.tyre.total_degradation += 0.1
    else:
        corner_speed = entry_speed

    seg_time = _time_at_constant(seg.length_m, corner_speed) if corner_speed > 0 else float('inf')
    fuel     = _fuel_used(corner_speed, corner_speed, seg.length_m)
    deg      = _tyre_degrade_corner(corner_speed, seg.radius_m, state.tyre, weather)

    # Check tyre blowout
    if state.tyre.is_blown():
        state.limp_mode = True
        state.blowouts += 1

    # Check fuel
    if state.fuel_l - fuel <= 0:
        state.limp_mode = True
        fuel = state.fuel_l

    _update_state(state, corner_speed, fuel, seg_time, deg)

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
) -> None:
    state.speed_m_s         = exit_speed
    state.fuel_l            = max(0.0, state.fuel_l - fuel_used)
    state.elapsed_s        += time_s
    state.total_fuel_used  += fuel_used
    state.tyre.total_degradation += degradation
    state.total_degradation += degradation

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

    refuel_time  = pit.fuel_refuel_amount_l / cfg.race.pit_refuel_rate_l_s if pit.fuel_refuel_amount_l > 0 else 0.0
    tyre_time    = cfg.race.pit_tyre_swap_time_s if pit.tyre_change_set_id is not None else 0.0
    base_time    = cfg.race.base_pit_stop_time_s
    pit_duration = refuel_time + tyre_time + base_time

    # Refuel
    if pit.fuel_refuel_amount_l > 0:
        state.fuel_l = min(
            car.fuel_tank_capacity_l,
            state.fuel_l + pit.fuel_refuel_amount_l,
        )
        state.total_fuel_used += pit.fuel_refuel_amount_l

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

def simulate(cfg: LevelConfig, strategy: RaceStrategy) -> SimResult:
    """
    Simulate the full race.

    Parameters
    ----------
    cfg      : LevelConfig
    strategy : RaceStrategy

    Returns
    -------
    SimResult
    """
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
                result = _simulate_straight(seg, action, state, car, weather)
            else:
                result = _simulate_corner(seg, state, car, weather)

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
