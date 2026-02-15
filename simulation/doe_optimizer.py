"""
DoE (Design of Experiments) parameter optimizer.

Uses L16 fractional factorial design to explore the parameter space
with only 16 runs instead of the full 576-combination grid.

Factors (4, each at 2 levels):
  κ  (kappa)          : [0.5,  2.5]   decay acceleration
  θ  (theta)          : [0.35, 0.60]  anomaly threshold
  β  (beta)           : [0.5,  2.0]   softmin sharpness
  bloom_interval (B)  : [7,   14]     days between Bloom events

Additional scan after L16:
  grace_period        : [14, 30, 60]
  deterministic_ratio : [0.70, 0.75, 0.80]
"""

import json
import time
import numpy as np
import pyDOE2
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.runner import SimulationRunner
from simulation.config import TestCase


# ------------------------------------------------------------------ #
# Parameter space                                                       #
# ------------------------------------------------------------------ #
FACTORS = {
    'kappa':          [0.5,  2.5],
    'theta':          [0.35, 0.60],
    'beta':           [0.5,  2.0],
    'bloom_interval': [7,   14],
}

FACTOR_NAMES = list(FACTORS.keys())

# Quick-run test cases (N=1K for speed; covers Sybil A and loop C)
QUICK_TEST_CASES = [
    TestCase('DOE_A10', 1000, 10, 'A', 'Simple Sybil k=10', ''),
    TestCase('DOE_C5',  1000,  5, 'C', 'Loop attack k=5',   ''),
]


def _decode_level(factor: str, level: float) -> float:
    """Map coded level (-1 or +1) → actual value."""
    lo, hi = FACTORS[factor]
    return lo + (level + 1) / 2 * (hi - lo)


def objective(metrics_list: List[Dict]) -> float:
    """
    Composite objective to MINIMISE.

    Lower is better:
      - Sybil ROI close to 0
      - FPR close to 0
      - TPR close to 1 (inverted: penalty for low TPR)
      - Gini ideally 0.2-0.3 (penalty for deviation from 0.25)

    Returns a weighted penalty score.
    """
    total = 0.0
    for m in metrics_list:
        roi  = m.get('sybil_roi', 1.0)
        fpr  = m.get('fpr', 0.1)
        tpr  = m.get('tpr', 0.0)
        gini = m.get('gini', 0.5)

        # Attacker scenarios only have meaningful ROI / TPR
        has_attackers = m.get('attacker_count', 0) > 0

        if has_attackers:
            total += 5.0 * min(roi, 5.0)          # ROI penalty (capped at 5)
            total += 20.0 * fpr                    # FPR penalty (strong)
            total += 5.0 * max(0, 0.95 - tpr)     # TPR penalty
        else:
            total += 20.0 * fpr                    # FPR applies everywhere

        total += 3.0 * abs(gini - 0.25)            # Gini target 0.25

    return total / len(metrics_list)


def run_doe_l16(output_dir: str = "/Users/kunimitsu/Projects/Meguri_pre3/results") -> List[Dict]:
    """
    Run L16 fractional factorial design and return ranked results.
    """
    runner = SimulationRunner(output_dir=output_dir)
    doe_dir = Path(output_dir) / "doe"
    doe_dir.mkdir(exist_ok=True, parents=True)

    # Generate L16 design matrix (4 factors, 2 levels each = 2^4 full factorial → 16 runs)
    design = pyDOE2.ff2n(4)   # shape (16, 4), levels ±1

    results = []
    print("=" * 65)
    print("DoE L16 Parameter Optimization")
    print(f"{'Run':>4}  {'κ':>5}  {'θ':>5}  {'β':>5}  {'Bloom':>5}  {'Score':>8}")
    print("=" * 65)

    for run_idx, row in enumerate(design):
        params = {name: _decode_level(name, row[i])
                  for i, name in enumerate(FACTOR_NAMES)}

        # Round bloom_interval to integer
        params['bloom_interval'] = int(round(params['bloom_interval']))

        run_metrics = []
        for tc in QUICK_TEST_CASES:
            tc_id = f"DOE_r{run_idx:02d}_{tc.test_id}"
            tc_run = TestCase(tc_id, tc.network_size, tc.attacker_count,
                              tc.scenario, tc.scenario_name, tc.description)
            m = runner.run_single_test_case(
                tc_run,
                kappa=params['kappa'],
                theta=params['theta'],
                beta=params['beta'],
                bloom_interval=params['bloom_interval'],
                verbose=False,
            )
            run_metrics.append(m)

        score = objective(run_metrics)
        result = {
            'run': run_idx,
            'params': params,
            'score': score,
            'metrics': run_metrics,
        }
        results.append(result)

        print(f"{run_idx:>4}  {params['kappa']:>5.2f}  {params['theta']:>5.2f}"
              f"  {params['beta']:>5.2f}  {params['bloom_interval']:>5d}  {score:>8.4f}")

    # Sort by score (lower = better)
    results.sort(key=lambda x: x['score'])

    # Save
    out_path = doe_dir / "doe_l16_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nTop 5 parameter combinations:")
    print(f"{'Rank':>4}  {'κ':>5}  {'θ':>5}  {'β':>5}  {'Bloom':>5}  {'Score':>8}")
    for rank, r in enumerate(results[:5], 1):
        p = r['params']
        print(f"{rank:>4}  {p['kappa']:>5.2f}  {p['theta']:>5.2f}"
              f"  {p['beta']:>5.2f}  {p['bloom_interval']:>5d}  {r['score']:>8.4f}")

    print(f"\nBest params: {results[0]['params']}")
    print(f"Saved to {out_path}")
    return results


def run_grace_period_scan(best_params: Dict,
                          output_dir: str = "/Users/kunimitsu/Projects/Meguri_pre3/results") -> Dict:
    """Scan grace_period {14, 30, 60} with the best params from L16."""
    runner = SimulationRunner(output_dir=output_dir)
    doe_dir = Path(output_dir) / "doe"
    doe_dir.mkdir(exist_ok=True, parents=True)

    tc = TestCase('GP_SCAN', 1000, 0, 'F', 'Grace Period scan', '')
    scan_results = []

    print("\n--- Grace Period Scan ---")
    print(f"{'GP days':>8}  {'FPR':>8}  {'Stabilization':>14}")
    for gp in [14, 30, 60]:
        m = runner.run_single_test_case(
            TestCase(f'GP_{gp}', 1000, 0, 'F', f'GP={gp}d', ''),
            grace_period=gp,
            verbose=False,
            **{k: v for k, v in best_params.items() if k != 'bloom_interval'},
            bloom_interval=best_params.get('bloom_interval', 7),
        )
        scan_results.append({'grace_period': gp, **m})
        print(f"{gp:>8}d  {m.get('fpr', 0):.4%}  {m.get('stabilization_median_days', '?'):>14}")

    out_path = doe_dir / "grace_period_scan.json"
    with open(out_path, 'w') as f:
        json.dump(scan_results, f, indent=2, default=str)

    return scan_results


if __name__ == '__main__':
    t0 = time.time()
    results = run_doe_l16()
    best = results[0]['params']

    print(f"\n[DoE complete in {time.time()-t0:.0f}s]")
    print("Running Grace Period scan with best params...")
    run_grace_period_scan(best)
    print(f"[Total: {time.time()-t0:.0f}s]")
