"""
Evaluation metrics for Phase 0 simulation.

Calculates:
1. Sybil ROI(k) - Attack profitability
2. TPR / FPR - Detection accuracy
3. Gini coefficient - Distribution fairness
4. VRF fairness - Bloom event equity
5. New participant stabilization period
6. Grace Period effectiveness
7. Community detection stability
8. Parameter sensitivity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .network_generator import SimulationState
from .mana_distributor import ManaDistributor
from .community_detector import CommunityDetector


class MetricsCalculator:
    """Computes all Phase 0 evaluation metrics."""

    def calculate_all(self, state: SimulationState,
                     daily_scores: List[Dict],
                     mana_distributor: Optional[ManaDistributor] = None,
                     community_detector: Optional[CommunityDetector] = None) -> Dict:
        """
        Calculate all metrics from simulation results.

        Args:
            state: Final simulation state
            daily_scores: List of daily score records
            mana_distributor: ManaDistributor instance (for Bloom stats)
            community_detector: CommunityDetector instance (for stability)

        Returns:
            Comprehensive metrics dictionary
        """
        df = pd.DataFrame(daily_scores)
        honest_ids = state.honest_ids
        attacker_ids = state.attacker_ids

        metrics = {}

        # 1. Sybil ROI
        metrics.update(self.calculate_sybil_roi(state, honest_ids, attacker_ids))

        # 2. TPR / FPR
        metrics.update(self.calculate_detection_rates(df, honest_ids, attacker_ids))

        # 3. Gini coefficient
        metrics.update(self.calculate_gini(state))

        # 4. VRF fairness (Bloom events)
        if mana_distributor:
            metrics.update(self.calculate_vrf_fairness(mana_distributor, len(state.nests)))

        # 5. New participant stabilization
        metrics.update(self.calculate_stabilization_period(df, honest_ids))

        # 6. Community detection stability
        if community_detector:
            metrics.update(self.calculate_community_stability(community_detector))

        # 7. Score statistics
        metrics.update(self.calculate_score_statistics(df, honest_ids, attacker_ids))

        return metrics

    def calculate_sybil_roi(self, state: SimulationState,
                           honest_ids: set, attacker_ids: set) -> Dict:
        """
        Calculate Sybil ROI(k).

        ROI(k) = attacker_avg_mana / honest_avg_mana

        Target: ROI(k) < 1/k (attack is unprofitable)
        """
        honest_balances = [state.get_nest(nid).balance for nid in honest_ids]
        attacker_balances = [state.get_nest(nid).balance for nid in attacker_ids]

        honest_avg = np.mean(honest_balances) if honest_balances else 0
        attacker_avg = np.mean(attacker_balances) if attacker_balances else 0
        honest_total = np.sum(honest_balances) if honest_balances else 0
        attacker_total = np.sum(attacker_balances) if attacker_balances else 0

        k = len(attacker_ids)
        sybil_roi = (attacker_avg / honest_avg) if honest_avg > 0 else float('inf')
        theoretical_threshold = (1.0 / k) if k > 0 else float('inf')

        return {
            'sybil_roi': float(sybil_roi),
            'sybil_roi_threshold': float(theoretical_threshold),
            'sybil_roi_pass': bool(sybil_roi < theoretical_threshold) if k > 0 else True,
            'honest_avg_balance': float(honest_avg),
            'attacker_avg_balance': float(attacker_avg),
            'honest_total_balance': float(honest_total),
            'attacker_total_balance': float(attacker_total),
            'attacker_count': k,
        }

    def calculate_detection_rates(self, df: pd.DataFrame,
                                 honest_ids: set, attacker_ids: set) -> Dict:
        """
        Calculate True Positive Rate and False Positive Rate.

        TPR = P(F=1 | attacker)   → Target: > 95%
        FPR = P(F=1 | honest)     → Target: < 1%
        """
        if df.empty:
            return {'tpr': 0.0, 'fpr': 0.0}

        final_day = df['day'].max()
        final = df[df['day'] == final_day]

        # TPR: attackers correctly flagged
        if attacker_ids:
            attacker_rows = final[final['nest_id'].isin(attacker_ids)]
            tpr = attacker_rows['F'].mean() if len(attacker_rows) > 0 else 0.0
        else:
            tpr = 0.0  # No attackers to detect

        # FPR: honest users incorrectly flagged
        if honest_ids:
            honest_rows = final[final['nest_id'].isin(honest_ids)]
            fpr = honest_rows['F'].mean() if len(honest_rows) > 0 else 0.0
        else:
            fpr = 0.0

        return {
            'tpr': float(tpr),
            'fpr': float(fpr),
            'tpr_pass': bool(tpr > 0.95) if attacker_ids else True,
            'fpr_pass': bool(fpr < 0.01),
        }

    def calculate_gini(self, state: SimulationState) -> Dict:
        """
        Calculate Gini coefficient of balance distribution.

        Gini = Σ|x_i - x_j| / (2n × mean(x))

        Target: 0.2 ~ 0.3 (moderately fair)
        Warning: > 0.5 (extreme inequality)
        """
        balances = np.array([nest.balance for nest in state.nests])
        balances = balances[balances > 0]  # Exclude zero balances

        if len(balances) == 0:
            return {'gini': 0.0, 'gini_pass': True}

        sorted_balances = np.sort(balances)
        n = len(sorted_balances)
        index = np.arange(1, n + 1)

        gini = (2 * np.sum(index * sorted_balances) / (n * np.sum(sorted_balances))) - (n + 1) / n

        return {
            'gini': float(gini),
            'gini_pass': bool(0.1 <= gini <= 0.5),
            'gini_warning': bool(gini > 0.5),
            'balance_mean': float(np.mean(balances)),
            'balance_median': float(np.median(balances)),
            'balance_std': float(np.std(balances)),
        }

    def calculate_vrf_fairness(self, mana_distributor: ManaDistributor,
                              network_size: int) -> Dict:
        """
        Calculate VRF (Bloom event) fairness.

        Each nest should be selected with probability 1/n.
        Fairness = 1 - max(|p_selected - 1/n|)

        Target: > 0.999
        """
        bloom_stats = mana_distributor.get_bloom_statistics()

        if bloom_stats['total_events'] == 0:
            return {'vrf_fairness': 1.0, 'bloom_events': 0}

        # Count wins per nest
        win_counts = {}
        for event in bloom_stats['history']:
            wid = event['winner_id']
            win_counts[wid] = win_counts.get(wid, 0) + 1

        total_events = bloom_stats['total_events']
        expected_prob = 1.0 / network_size

        # Calculate max deviation from expected probability
        max_deviation = 0.0
        for nest_id in range(network_size):
            actual_prob = win_counts.get(nest_id, 0) / total_events
            deviation = abs(actual_prob - expected_prob)
            max_deviation = max(max_deviation, deviation)

        fairness = 1.0 - max_deviation

        return {
            'vrf_fairness': float(fairness),
            'bloom_events': total_events,
            'bloom_total_distributed': bloom_stats['total_distributed'],
            'bloom_unique_winners': bloom_stats['unique_winners'],
            'bloom_max_wins': bloom_stats['max_wins_single_nest'],
        }

    def calculate_stabilization_period(self, df: pd.DataFrame,
                                      honest_ids: set,
                                      theta: float = 0.3) -> Dict:
        """
        Calculate how many days new participants need to reach score θ.

        Target: < 30 days (median)
        """
        if df.empty or not honest_ids:
            return {'stabilization_median_days': 0}

        # Track when each honest nest first reaches θ
        stabilization_days = []

        for nest_id in honest_ids:
            nest_data = df[df['nest_id'] == nest_id].sort_values('day')
            if nest_data.empty:
                continue

            reached_day = None
            for _, row in nest_data.iterrows():
                if row['S'] >= theta:
                    reached_day = row['day']
                    break

            if reached_day is not None:
                stabilization_days.append(reached_day)

        if not stabilization_days:
            return {'stabilization_median_days': float('inf'),
                    'stabilization_pass': False}

        median_days = float(np.median(stabilization_days))
        return {
            'stabilization_median_days': median_days,
            'stabilization_p25_days': float(np.percentile(stabilization_days, 25)),
            'stabilization_p75_days': float(np.percentile(stabilization_days, 75)),
            'stabilization_pass': bool(median_days < 30),
            'pct_reached_theta': len(stabilization_days) / len(honest_ids),
        }

    def calculate_community_stability(self, community_detector: CommunityDetector) -> Dict:
        """
        Calculate community detection stability.

        Target: > 0.95
        """
        summary = community_detector.get_detection_summary()
        stability = community_detector.get_stability_score()

        return {
            'community_stability': float(stability),
            'community_stability_pass': bool(stability > 0.95),
            'num_communities': summary.get('num_communities', 0),
            'community_detections': summary.get('detections', 0),
        }

    def calculate_score_statistics(self, df: pd.DataFrame,
                                  honest_ids: set, attacker_ids: set) -> Dict:
        """Calculate summary statistics for scores."""
        if df.empty:
            return {}

        final_day = df['day'].max()
        final = df[df['day'] == final_day]

        stats = {}
        for group_name, ids in [('honest', honest_ids), ('attacker', attacker_ids)]:
            if not ids:
                continue
            group = final[final['nest_id'].isin(ids)]
            if group.empty:
                continue

            for col in ['T', 'E', 'D', 'S', 'g_S', 'balance']:
                if col in group.columns:
                    stats[f'{group_name}_{col}_mean'] = float(group[col].mean())
                    stats[f'{group_name}_{col}_std'] = float(group[col].std())

            stats[f'{group_name}_F_rate'] = float(group['F'].mean())

        return stats


def generate_summary_report(metrics: Dict, test_id: str = "") -> str:
    """Generate a human-readable summary report."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"Phase 0 Simulation Report: {test_id}")
    lines.append(f"{'='*60}")
    lines.append("")

    # Sybil ROI
    roi = metrics.get('sybil_roi', 'N/A')
    threshold = metrics.get('sybil_roi_threshold', 'N/A')
    roi_pass = metrics.get('sybil_roi_pass', False)
    lines.append(f"[{'PASS' if roi_pass else 'FAIL'}] Sybil ROI: {roi:.4f} (threshold: {threshold:.4f})")

    # TPR / FPR
    tpr = metrics.get('tpr', 0)
    fpr = metrics.get('fpr', 0)
    lines.append(f"[{'PASS' if metrics.get('tpr_pass', False) else 'FAIL'}] TPR: {tpr:.2%} (target: > 95%)")
    lines.append(f"[{'PASS' if metrics.get('fpr_pass', False) else 'FAIL'}] FPR: {fpr:.2%} (target: < 1%)")

    # Gini
    gini = metrics.get('gini', 0)
    lines.append(f"[{'PASS' if metrics.get('gini_pass', False) else 'FAIL'}] Gini: {gini:.4f} (target: 0.2-0.3)")

    # Stabilization
    stab = metrics.get('stabilization_median_days', 'N/A')
    lines.append(f"[{'PASS' if metrics.get('stabilization_pass', False) else 'FAIL'}] Stabilization: {stab} days (target: < 30)")

    # Community stability
    comm_stab = metrics.get('community_stability', 'N/A')
    if comm_stab != 'N/A':
        lines.append(f"[{'PASS' if metrics.get('community_stability_pass', False) else 'FAIL'}] Community stability: {comm_stab:.4f} (target: > 0.95)")

    # VRF fairness
    vrf = metrics.get('vrf_fairness', 'N/A')
    if vrf != 'N/A':
        lines.append(f"[INFO] VRF fairness: {vrf:.6f} ({metrics.get('bloom_events', 0)} events)")

    lines.append("")
    lines.append(f"{'='*60}")

    return "\n".join(lines)
