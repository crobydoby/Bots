"""
main.py
-------
Entry point for the Entelect Grand Prix race strategy optimiser.

Usage
-----
    python main.py [level_file] [output_file]

    level_file  : path to the level JSON (default: 1.txt)
    output_file : path to write the submission JSON (default: submission.txt)

Currently runs the naive baseline strategy and prints a score summary.
Swap in your own optimiser in the `optimise()` function below.
"""

from __future__ import annotations

import sys
from pathlib import Path

from level_loader import load_level
from strategy    import build_naive_strategy, save_strategy
from simulator   import simulate


# ---------------------------------------------------------------------------
# Optimiser hook — replace this with your metaheuristic
# ---------------------------------------------------------------------------

def optimise(cfg, level_path: str):
    """
    Generate an optimised RaceStrategy for the given level config.

    At the moment this returns the naive greedy strategy as a placeholder.
    Replace the body of this function with your metaheuristic solver.
    """
    print("[optimise] Using naive baseline strategy (placeholder)")
    return build_naive_strategy(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    level_path  = sys.argv[1] if len(sys.argv) > 1 else "1.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "submission.txt"

    # 1. Load level
    print(f"Loading level: {level_path}")
    cfg = load_level(level_path)
    print(f"  Race  : {cfg.race.name}")
    print(f"  Track : {cfg.track.name}  ({cfg.track.total_length_m:.0f} m/lap × {cfg.race.laps} laps)")
    print(f"  Car   : max {cfg.car.max_speed_m_s} m/s  |  accel {cfg.car.accel_m_se2} m/s²  |  brake {cfg.car.brake_m_se2} m/s²")
    print()

    # 2. Optimise
    strategy = optimise(cfg, level_path)

    # 3. Simulate and score
    result = simulate(cfg, strategy)
    print(f"--- Simulation Result ---")
    print(f"  Total time  : {result.total_time_s:.2f} s  (ref: {cfg.race.time_reference_s:.2f} s)")
    print(f"  Fuel used   : {result.total_fuel_used_l:.4f} L  (soft cap: {cfg.race.fuel_soft_cap_limit_l} L)")
    print(f"  Tyre degrad : {result.total_tyre_degradation:.4f}")
    print(f"  Blowouts    : {result.blowouts}")
    print(f"  Crashes     : {result.crashes}")
    print()
    print(f"  Score (L1)  : {result.score_level1(cfg):,.0f}")
    print(f"  Score (L2)  : {result.score_level2(cfg):,.0f}")
    print(f"  Score (L4)  : {result.score_level4(cfg):,.0f}")
    print()

    # Per-lap summary
    print("  Lap  |  Time (s)  | Pit (s) | Fuel (L) | Crashes | Blowouts")
    print("  " + "-" * 60)
    for lr in result.lap_results:
        print(
            f"  {lr.lap:>3}  | {lr.time_s:>10.2f} | {lr.pit_time_s:>7.1f} "
            f"| {lr.fuel_used_l:>8.4f} | {lr.crashes:>7} | {lr.blowouts:>8}"
        )

    # 4. Save submission
    save_strategy(strategy, output_path)
    print(f"\nSubmission written to: {output_path}")


if __name__ == "__main__":
    main()
