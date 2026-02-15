"""
T010: Scenario G — Distance axis value measurement.

Compares Sybil ROI under:
  - 3-layer design (Time + Entropy + neutral D)
  - 4-layer design (Time + Entropy + Network-distance D)

Runs k = [1, 5, 10, 20, 50] with N=1000 for speed.
"""

import json
from pathlib import Path

from .runner import SimulationRunner
from .config import TestCase

_DEFAULT_RESULTS_DIR = str(Path(__file__).parent.parent / "results")


def run_t010(output_dir: str = _DEFAULT_RESULTS_DIR,
             kappa: float = 1.5,
             theta: float = 0.45,
             beta: float = 1.0,
             verbose: bool = True) -> dict:
    """
    Run Scenario G: compare 3-layer vs 4-layer Sybil ROI across different k values.
    Uses N=1000 for speed.
    """
    runner = SimulationRunner(output_dir=output_dir)
    k_values = [1, 5, 10, 20, 50]
    results = {'three_layer': [], 'four_layer': []}

    if verbose:
        print("\n" + "="*65)
        print("T010: Scenario G — 3-layer vs 4-layer Distance Axis")
        print("="*65)
        print(f"{'k':>5}  {'3-layer ROI':>12}  {'4-layer ROI':>12}  {'Improvement':>12}")
        print("-"*65)

    for k in k_values:
        for use_dist, label, store_key in [
            (False, '3L', 'three_layer'),
            (True,  '4L', 'four_layer'),
        ]:
            tc = TestCase(
                test_id=f'T010_{label}_k{k}',
                network_size=1000,
                attacker_count=k,
                scenario='G',
                scenario_name=f'Distance axis ({label}), k={k}',
                description=f'use_network_distance={use_dist}',
            )
            m = runner.run_single_test_case(
                tc,
                kappa=kappa, theta=theta, beta=beta,
                use_network_distance=use_dist,
                verbose=False,
            )
            m['k'] = k
            m['layers'] = 4 if use_dist else 3
            results[store_key].append(m)

        roi_3 = results['three_layer'][-1]['sybil_roi']
        roi_4 = results['four_layer'][-1]['sybil_roi']
        improvement = (roi_3 - roi_4) / max(roi_3, 1e-9) * 100
        if verbose:
            print(f"{k:>5}  {roi_3:>12.4f}  {roi_4:>12.4f}  {improvement:>11.1f}%")

    # Save results
    doe_dir = Path(output_dir) / "doe"
    doe_dir.mkdir(exist_ok=True, parents=True)
    out_path = doe_dir / "t010_distance_axis.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\nSaved to {out_path}")

    return results


if __name__ == '__main__':
    run_t010()
