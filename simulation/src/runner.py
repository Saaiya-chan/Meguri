"""
Main simulation runner orchestrating the day-by-day loop.
Integrates all modules: network, transactions, scoring, Mana, community, metrics.
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .network_generator import NetworkGenerator, SimulationState
from .transaction_engine import TransactionEngine, AttackerStrategy
from .scoring import ScoringEngine
from .mana_distributor import ManaDistributor, DecayEngine
from .community_detector import CommunityDetector
from .metrics import MetricsCalculator, generate_summary_report
from .config import MeguriPhase0Config, TestCase


class SimulationRunner:
    """Orchestrates Phase 0 simulation with all 7 mechanisms integrated."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent / "results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.config = MeguriPhase0Config()

    def run_single_test_case(self, test_case: TestCase,
                            kappa: float = 1.0,
                            theta: float = 0.3,
                            beta: float = 1.0,
                            bloom_interval: int = 7,
                            deterministic_ratio: float = 0.75,
                            grace_period: int = 30,
                            community_cycle: int = 30,
                            enable_community: bool = True,
                            use_network_distance: bool = False,
                            verbose: bool = True) -> Dict:
        """
        Run a single test case with all mechanisms integrated.

        Args:
            test_case: TestCase configuration
            kappa: Decay acceleration coefficient
            theta: Anomaly threshold
            beta: Softmin scaling factor
            bloom_interval: Days between Bloom events
            deterministic_ratio: Fraction of Mana distributed deterministically
            grace_period: Days of Grace Period
            community_cycle: Days between community detection runs
            enable_community: Whether to run community detection
            verbose: Print progress

        Returns:
            Comprehensive metrics dictionary
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {test_case.test_id}: {test_case.scenario_name}")
            print(f"N={test_case.network_size}, k={test_case.attacker_count}, "
                  f"Scenario {test_case.scenario}")
            print(f"Params: κ={kappa}, θ={theta}, β={beta}, bloom={bloom_interval}d")
            print(f"{'='*60}")

        # --- Initialize all components ---
        attacker_count = test_case.attacker_count or 0

        gen = NetworkGenerator(
            network_size=test_case.network_size,
            attacker_count=attacker_count,
            random_seed=42
        )
        G, nests, honest_ids, attacker_ids = gen.generate()
        state = SimulationState(G, nests, honest_ids, attacker_ids)

        tx_engine = TransactionEngine(daily_tx_lambda=2.0, random_seed=42)
        scoring_engine = ScoringEngine()

        mana_dist = ManaDistributor(
            daily_mana_base=100.0,
            deterministic_ratio=deterministic_ratio,
            bloom_ratio=1.0 - deterministic_ratio,
            bloom_interval=bloom_interval,
        )

        decay_engine = DecayEngine(base_decay_rate=0.005)

        community_det = CommunityDetector(random_seed=42) if enable_community else None

        # --- Scenario-specific setup ---
        scenario_config = self._setup_scenario(test_case, state, honest_ids, attacker_ids)

        # --- Main simulation loop ---
        daily_scores = []
        num_days = self.config.network.simulation_days

        for day in range(num_days):
            if verbose and (day + 1) % 30 == 0:
                print(f"  Day {day + 1}/{num_days}")

            # 1. Execute transactions
            self._execute_transactions(
                state, G, day, test_case, tx_engine,
                honest_ids, attacker_ids, scenario_config
            )

            # 2. Community detection (monthly)
            if (community_det and enable_community and
                    day > 0 and day % community_cycle == 0):
                scores_for_community = scoring_engine.calculate_all_scores(
                    state, day, kappa=kappa, theta=theta, beta=beta
                )
                community_det.detect_communities(state, day, scores_for_community)

            # 3. Calculate all scores
            scores = scoring_engine.calculate_all_scores(
                state, day, kappa=kappa, theta=theta, beta=beta,
                use_network_distance=use_network_distance
            )

            # 4. Apply community normalization to scores
            if community_det and community_det.communities:
                for nest_id in scores:
                    raw_S = scores[nest_id]['S']
                    normalized_S = community_det.get_normalized_score(nest_id, raw_S)
                    scores[nest_id]['S'] = normalized_S
                    # Recalculate F and g_S with normalized score
                    scores[nest_id]['F'] = scoring_engine.calculate_anomaly_flag(
                        normalized_S, theta
                    )
                    scores[nest_id]['g_S'] = scoring_engine.calculate_decay_acceleration(
                        normalized_S, theta, kappa
                    )

            # 5. Grace Period: suppress decay acceleration for returning users
            self._apply_grace_period(state, scores, day, grace_period, scenario_config)

            # 6. Distribute Mana
            mana_dist.distribute_daily(state, day, scores)

            # 7. Apply decay
            decay_engine.apply_decay(state, scores)

            # 8. Record daily data
            for nest in state.nests:
                nid = nest.nest_id
                daily_scores.append({
                    'day': day,
                    'nest_id': nid,
                    'is_attacker': int(nest.is_attacker),
                    'T': scores[nid]['T'],
                    'E': scores[nid]['E'],
                    'D': scores[nid]['D'],
                    'S': scores[nid]['S'],
                    'F': scores[nid]['F'],
                    'g_S': scores[nid]['g_S'],
                    'balance': nest.balance,
                    'mana_received': nest.mana_received_today,
                    'total_transactions': nest.total_transactions,
                })

        # --- Calculate comprehensive metrics ---
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_all(
            state, daily_scores, mana_dist, community_det
        )

        # Add test case metadata
        metrics['test_id'] = test_case.test_id
        metrics['scenario'] = test_case.scenario
        metrics['scenario_name'] = test_case.scenario_name
        metrics['network_size'] = test_case.network_size
        metrics['params'] = {
            'kappa': kappa, 'theta': theta, 'beta': beta,
            'bloom_interval': bloom_interval,
            'deterministic_ratio': deterministic_ratio,
            'grace_period': grace_period,
            'community_cycle': community_cycle,
            'use_network_distance': use_network_distance,
        }

        # Save results
        self._save_results(test_case.test_id, daily_scores, metrics)

        if verbose:
            print(generate_summary_report(metrics, test_case.test_id))

        return metrics

    def _setup_scenario(self, test_case: TestCase, state: SimulationState,
                       honest_ids: List[int], attacker_ids: List[int]) -> Dict:
        """Configure scenario-specific behavior."""
        config = {
            'attacker_strategy': 'honest',
            'rural_nest_ids': set(),
            'new_participant_ids': set(),
            'returning_nest_ids': set(),
            'returning_start_day': 0,
            'returning_device_match': True,
        }

        if test_case.scenario == 'B':
            # Device-constrained Sybil: attackers need separate devices
            # Simulate higher cost via reduced transaction frequency
            config['attacker_strategy'] = 'device_constrained'

        elif test_case.scenario == 'C':
            # High-frequency loop attack
            config['attacker_strategy'] = 'loop'

        elif test_case.scenario == 'D':
            # Rural community: subset of honest users with low transaction rates
            n_rural = max(1, len(honest_ids) // 10)
            config['rural_nest_ids'] = set(honest_ids[:n_rural])

        elif test_case.scenario == 'E':
            # New participants: some honest users start later (day 30)
            n_new = max(1, len(honest_ids) // 5)
            config['new_participant_ids'] = set(honest_ids[-n_new:])

        elif test_case.scenario == 'F':
            # Returning users: some go inactive for 60+ days then return
            n_returning = max(1, len(honest_ids) // 10)
            config['returning_nest_ids'] = set(honest_ids[:n_returning])
            config['returning_start_day'] = 90  # Return on day 90

        return config

    def _execute_transactions(self, state: SimulationState, G,
                            day: int, test_case: TestCase,
                            tx_engine: TransactionEngine,
                            honest_ids: List[int], attacker_ids: List[int],
                            scenario_config: Dict):
        """Execute transactions based on scenario."""
        strategy = scenario_config['attacker_strategy']
        rural_ids = scenario_config.get('rural_nest_ids', set())
        new_ids = scenario_config.get('new_participant_ids', set())
        returning_ids = scenario_config.get('returning_nest_ids', set())
        returning_day = scenario_config.get('returning_start_day', 0)

        # Handle attacker transactions based on strategy
        if attacker_ids:
            if strategy == 'loop':
                AttackerStrategy.high_frequency_loop(state, attacker_ids, day)
            elif strategy == 'device_constrained':
                # Reduced frequency due to device management overhead
                for aid in attacker_ids:
                    if np.random.random() < 0.3:  # Only 30% chance to transact
                        AttackerStrategy.honest(state, [aid], day)
            else:
                AttackerStrategy.honest(state, attacker_ids, day)

        # Handle honest user transactions
        for hid in honest_ids:
            # Skip new participants before their start day
            if hid in new_ids and day < 30:
                continue

            # Skip returning users during absence (day 30-89)
            if hid in returning_ids and 30 <= day < returning_day:
                continue

            # Rural users: lower transaction rate
            if hid in rural_ids:
                if np.random.random() > 0.3:  # Only 30% chance to transact
                    continue

            nest = state.get_nest(hid)
            neighbors = list(G.neighbors(hid))
            if not neighbors:
                continue

            num_tx = np.random.poisson(2.0)
            for _ in range(num_tx):
                if np.random.random() < 0.7:
                    to_id = np.random.choice(neighbors)
                else:
                    to_id = np.random.randint(0, state.G.number_of_nodes())
                    if to_id == hid:
                        continue

                amount = np.random.randint(1, 101)
                if nest.balance >= amount:
                    nest.balance -= amount
                    state.nests_by_id[to_id].balance += amount
                    state.record_transaction(day, hid, to_id, float(amount))

    def _apply_grace_period(self, state: SimulationState, scores: Dict,
                           day: int, grace_period: int,
                           scenario_config: Dict):
        """
        Apply Grace Period: suppress decay acceleration for returning users.
        During Grace Period, g(S) is set to 1.0 (normal decay).
        """
        returning_ids = scenario_config.get('returning_nest_ids', set())
        returning_day = scenario_config.get('returning_start_day', 0)

        if not returning_ids:
            return

        # During Grace Period window after return
        if returning_day <= day < returning_day + grace_period:
            for nid in returning_ids:
                scores[nid]['g_S'] = 1.0  # Normal decay during grace

    def _save_results(self, test_id: str, daily_scores: List[Dict], metrics: Dict):
        """Save simulation results to CSV and JSON."""
        # Save scores CSV (sample every 7 days to reduce file size for large N)
        scores_dir = self.output_dir / "scores"
        scores_dir.mkdir(exist_ok=True, parents=True)

        csv_path = scores_dir / f"{test_id}_scores.csv"
        if daily_scores:
            # For large datasets, sample weekly
            if len(daily_scores) > 500_000:
                sampled = [s for s in daily_scores if s['day'] % 7 == 0]
            else:
                sampled = daily_scores

            keys = sampled[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(sampled)

        # Save metrics JSON
        summary_dir = self.output_dir / "summary"
        summary_dir.mkdir(exist_ok=True, parents=True)

        json_path = summary_dir / f"{test_id}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    def run_all_test_cases(self, kappa: float = 1.0, theta: float = 0.3,
                          beta: float = 1.0, bloom_interval: int = 7,
                          verbose: bool = True) -> List[Dict]:
        """
        Run all T001-T010 test cases.

        Returns:
            List of metrics dictionaries, one per test case.
        """
        results = []
        for tc in self.config.test_cases:
            # Skip T004 (N=100K) and T010 (variable k) for quick runs
            if tc.network_size > 10000:
                if verbose:
                    print(f"\n[SKIP] {tc.test_id}: N={tc.network_size} (too large for quick run)")
                continue
            if tc.attacker_count is None:
                if verbose:
                    print(f"\n[SKIP] {tc.test_id}: variable k (run separately)")
                continue

            metrics = self.run_single_test_case(
                tc, kappa=kappa, theta=theta, beta=beta,
                bloom_interval=bloom_interval, verbose=verbose
            )
            results.append(metrics)

        # Save combined results
        summary_dir = self.output_dir / "summary"
        summary_dir.mkdir(exist_ok=True, parents=True)
        combined_path = summary_dir / "all_test_cases.json"
        with open(combined_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if verbose:
            print(f"\n{'='*60}")
            print("ALL TEST CASES COMPLETE")
            print(f"{'='*60}")
            for r in results:
                roi = r.get('sybil_roi', 'N/A')
                tpr = r.get('tpr', 'N/A')
                fpr = r.get('fpr', 'N/A')
                tid = r.get('test_id', '???')
                scenario = r.get('scenario_name', '')
                roi_str = f"{roi:.4f}" if isinstance(roi, float) else str(roi)
                tpr_str = f"{tpr:.2%}" if isinstance(tpr, float) else str(tpr)
                fpr_str = f"{fpr:.2%}" if isinstance(fpr, float) else str(fpr)
                print(f"  {tid}: ROI={roi_str}, TPR={tpr_str}, FPR={fpr_str} | {scenario}")

        return results
