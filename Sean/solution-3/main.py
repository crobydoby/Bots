"""
main.py
-------
Entry point for the Entelect Grand Prix race strategy optimiser.

Usage
-----
    # Level 1 (naive baseline, simple mode)
    python main.py 1.txt [output_file]

    # Level 2 (lambda sweep, fuel-aware pit stops)
    python main.py 2.txt [output_file]

    level_file  : path to the level JSON (default: 1.txt)
    output_file : path to write the best submission JSON (default: submission.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path

from level_loader import load_level
from strategy    import build_naive_strategy, build_lambda_strategy, save_strategy
from simulator   import simulate, SimConfig


# ---------------------------------------------------------------------------
# Level 1 — naive baseline
# ---------------------------------------------------------------------------

def run_level1(cfg, output_path: str) -> None:
    sim_cfg  = SimConfig.simple()
    strategy = build_naive_strategy(cfg, tyre_degradation=False)
    result   = simulate(cfg, strategy, sim_cfg)

    print("--- Level 1 Simulation Result ---")
    print(f"  Total time  : {result.total_time_s:.2f} s  (ref: {cfg.race.time_reference_s:.2f} s)")
    print(f"  Crashes     : {result.crashes}  |  Blowouts: {result.blowouts}")
    print(f"  Score (L1)  : {result.score_level1(cfg):,.0f}")
    print()

    save_strategy(strategy, output_path)
    print(f"Submission written to: {output_path}")


# ---------------------------------------------------------------------------
# Level 2 — lambda sweep with fuel-aware pit stops
# ---------------------------------------------------------------------------

def run_level2(cfg, output_path: str) -> None:
    """
    Sweep lambda from 0.50 to 1.00 in steps of 0.01.
    For each lambda:
      - Build a strategy with target_speed = lambda * max_speed on straights.
      - Pit rule: pit if fuel < last_lap_usage; refuel for remaining laps
        (capped at tank capacity).  No pit on final lap.
    Print a table sorted by Score (L2) and save the best submission.
    """
    sim_cfg = SimConfig(tyre_degradation=False, fuel_consumption=True)

    lam_values = [round(v * 0.01, 2) for v in range(50, 101)]  # 0.50 → 1.00

    print("--- Level 2 Lambda Sweep ---")
    print(f"  {'Lambda':>7}  {'Time (s)':>12}  {'Fuel (L)':>10}  {'Pits':>5}  {'Crashes':>8}  {'Score L2':>14}")
    print("  " + "-" * 70)

    best_score    = float('-inf')
    best_lam      = None
    best_strategy = None
    best_result   = None

    results_table = []

    for lam in lam_values:
        strategy = build_lambda_strategy(cfg, lam)
        result   = simulate(cfg, strategy, sim_cfg)
        score    = result.score_level2(cfg)

        # Count pit stops
        pit_count = sum(1 for ls in strategy.laps if ls.pit.enter)

        results_table.append((lam, result.total_time_s, result.total_fuel_used_l,
                               pit_count, result.crashes, score))

        if score > best_score:
            best_score    = score
            best_lam      = lam
            best_strategy = strategy
            best_result   = result

    # Print table sorted by score descending
    results_table.sort(key=lambda r: r[5], reverse=True)
    for lam, t, fuel, pits, crashes, score in results_table:
        marker = " ◄ best" if lam == best_lam else ""
        print(f"  {lam:>7.2f}  {t:>12.2f}  {fuel:>10.4f}  {pits:>5}  {crashes:>8}  {score:>14,.2f}{marker}")

    print()
    print(f"Best lambda : {best_lam:.2f}")
    print(f"Best score  : {best_score:,.2f}")
    print(f"Total time  : {best_result.total_time_s:.2f} s  (ref: {cfg.race.time_reference_s:.2f} s)")
    print(f"Fuel used   : {best_result.total_fuel_used_l:.4f} L  (soft cap: {cfg.race.fuel_soft_cap_limit_l} L)")
    print(f"Pit stops   : {sum(1 for ls in best_strategy.laps if ls.pit.enter)}")
    print(f"Crashes     : {best_result.crashes}")
    print()

    # Per-lap summary for best strategy
    print("  Lap  |  Time (s)  | Pit (s) | Fuel (L) | Crashes")
    print("  " + "-" * 52)
    for lr in best_result.lap_results:
        print(
            f"  {lr.lap:>3}  | {lr.time_s:>10.2f} | {lr.pit_time_s:>7.1f} "
            f"| {lr.fuel_used_l:>8.4f} | {lr.crashes:>7}"
        )

    save_strategy(best_strategy, output_path)
    print(f"\nBest submission written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    level_path  = sys.argv[1] if len(sys.argv) > 1 else "1.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "submission.txt"

    print(f"Loading level: {level_path}")
    cfg = load_level(level_path)
    print(f"  Race  : {cfg.race.name}")
    print(f"  Track : {cfg.track.name}  ({cfg.track.total_length_m:.0f} m/lap × {cfg.race.laps} laps)")
    print(f"  Car   : max {cfg.car.max_speed_m_s} m/s  |  accel {cfg.car.accel_m_se2} m/s²  |  brake {cfg.car.brake_m_se2} m/s²")
    print()

    level_name = Path(level_path).stem   # "1", "2", etc.

    if level_name == "3":
        run_level3(cfg, output_path)
    elif level_name == "2":
        run_level2(cfg, output_path)
    else:
        run_level1(cfg, output_path)


from strategy import build_level3_strategy, save_strategy
from simulator import simulate, SimConfig
 
 
def run_level3(cfg, output_path: str) -> None:
    """
    Two-phase sweep for Level 3:
 
    Phase 1 — Interval sweep at lambda=1.0
        Sweep all 1-stop and 2-stop tyre-change combinations (step 5 laps).
        Best compound per stint chosen automatically from weather window.
        Print table sorted by Score L3, identify best interval combo.
 
    Phase 2 — Lambda sweep at best interval combo
        Sweep lambda 0.80 → 1.00 (step 0.01) with the winning intervals.
        Print table, save best overall submission.
    """
    sim_cfg = SimConfig(tyre_degradation=False, fuel_consumption=True)
 
    candidates = list(range(10, cfg.race.laps - 5, 5))   # 10, 15, 20 … 60
 
    combos = [(n,) for n in candidates]
    for i, n1 in enumerate(candidates):
        for n2 in candidates[i + 1:]:
            combos.append((n1, n2))
 
    # ------------------------------------------------------------------
    # Phase 1: interval sweep at lam = 1.0
    # ------------------------------------------------------------------
    print("--- Level 3 Phase 1: Tyre-Change Interval Sweep (lambda=1.00) ---")
    print(f"  {len(combos)} combinations")
    print(f"  {'Intervals':>18}  {'Tyres':>28}  {'Time (s)':>10}  "
          f"{'Fuel (L)':>10}  {'Pits':>5}  {'Crashes':>8}  {'Score L3':>14}")
    print("  " + "-" * 105)
 
    best_score_p1 = float('-inf')
    best_combo    = None
    p1_table      = []
 
    for combo in combos:
        strategy = build_level3_strategy(cfg, 1.0, list(combo))
        result   = simulate(cfg, strategy, sim_cfg)
        score    = result.score_level3(cfg)
        pit_count = sum(1 for ls in strategy.laps if ls.pit.enter)
 
        tyre_labels = []
        for ls in strategy.laps:
            if ls.pit.enter and ls.pit.tyre_change_set_id is not None:
                tset = next((s for s in cfg.available_sets
                             if s.primary_id == ls.pit.tyre_change_set_id), None)
                if tset:
                    tyre_labels.append(f"L{ls.lap}:{tset.compound.value[:3]}")
        tyre_str = ",".join(tyre_labels) if tyre_labels else "no change"
 
        p1_table.append((combo, tyre_str, result.total_time_s,
                         result.total_fuel_used_l, pit_count, result.crashes, score))
 
        if score > best_score_p1:
            best_score_p1 = score
            best_combo    = combo
 
    p1_table.sort(key=lambda r: r[6], reverse=True)
    for combo, tyre_str, t, fuel, pits, crashes, score in p1_table[:20]:
        marker = " ◄ best" if combo == best_combo else ""
        print(f"  {str(combo):>18}  {tyre_str:>28}  {t:>10.2f}  "
              f"{fuel:>10.4f}  {pits:>5}  {crashes:>8}  {score:>14,.2f}{marker}")
 
    if len(p1_table) > 20:
        print(f"  ... ({len(p1_table) - 20} more not shown)")
 
    print(f"\n  Best interval combo: {best_combo}  |  Score: {best_score_p1:,.2f}\n")
 
    # ------------------------------------------------------------------
    # Phase 2: lambda sweep at best interval combo
    # ------------------------------------------------------------------
    lam_values = [round(v * 0.01, 2) for v in range(80, 101)]   # 0.80 → 1.00
 
    print(f"--- Level 3 Phase 2: Lambda Sweep at intervals={best_combo} ---")
    print(f"  {'Lambda':>7}  {'Time (s)':>12}  {'Fuel (L)':>10}  "
          f"{'Pits':>5}  {'Crashes':>8}  {'Score L3':>14}")
    print("  " + "-" * 72)
 
    best_score    = float('-inf')
    best_lam      = None
    best_strategy = None
    best_result   = None
    p2_table      = []
 
    for lam in lam_values:
        strategy = build_level3_strategy(cfg, lam, list(best_combo))
        result   = simulate(cfg, strategy, sim_cfg)
        score    = result.score_level3(cfg)
        pit_count = sum(1 for ls in strategy.laps if ls.pit.enter)
 
        p2_table.append((lam, result.total_time_s, result.total_fuel_used_l,
                         pit_count, result.crashes, score))
 
        if score > best_score:
            best_score    = score
            best_lam      = lam
            best_strategy = strategy
            best_result   = result
 
    p2_table.sort(key=lambda r: r[5], reverse=True)
    for lam, t, fuel, pits, crashes, score in p2_table:
        marker = " ◄ best" if lam == best_lam else ""
        print(f"  {lam:>7.2f}  {t:>12.2f}  {fuel:>10.4f}  "
              f"{pits:>5}  {crashes:>8}  {score:>14,.2f}{marker}")
 
    print()
    print(f"Best lambda    : {best_lam:.2f}")
    print(f"Best intervals : {best_combo}")
    print(f"Best score     : {best_score:,.2f}")
    print(f"Total time     : {best_result.total_time_s:.2f} s  "
          f"(ref: {cfg.race.time_reference_s:.2f} s)")
    print(f"Fuel used      : {best_result.total_fuel_used_l:.4f} L  "
          f"(soft cap: {cfg.race.fuel_soft_cap_limit_l} L)")
    print(f"Pit stops      : {sum(1 for ls in best_strategy.laps if ls.pit.enter)}")
    print(f"Crashes        : {best_result.crashes}")
    print()
 
    # Per-lap summary
    print("  Lap  |  Time (s)  | Pit (s) | Fuel (L) | Tyre change | Crashes")
    print("  " + "-" * 65)
    for lr in best_result.lap_results:
        lap_strat = best_strategy.laps[lr.lap - 1]
        tyre_note = ""
        if lap_strat.pit.enter and lap_strat.pit.tyre_change_set_id is not None:
            tset = next((s for s in cfg.available_sets
                         if s.primary_id == lap_strat.pit.tyre_change_set_id), None)
            tyre_note = f"→ {tset.compound.value}" if tset else "→ ?"
        print(
            f"  {lr.lap:>3}  | {lr.time_s:>10.2f} | {lr.pit_time_s:>7.1f} "
            f"| {lr.fuel_used_l:>8.4f} | {tyre_note:<11} | {lr.crashes:>7}"
        )
 
    save_strategy(best_strategy, output_path)
    print(f"\nBest submission written to: {output_path}")
 
 
 
# ---------------------------------------------------------------------------
# Wire into main() — replace the level dispatch block with:
# ---------------------------------------------------------------------------
#
#   if level_name == "3":
#       run_level3(cfg, output_path)
#   elif level_name == "2":
#       run_level2(cfg, output_path)
#   else:
#       run_level1(cfg, output_path)
#

if __name__ == "__main__":
    main()
