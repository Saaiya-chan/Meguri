"""
Mana distribution engine for Meguri.

Hybrid 75/25 model:
- 75% deterministic: evenly distributed to all non-anomalous nests (F=0)
- 25% stochastic: Bloom events via VRF lottery at regular intervals
"""

import numpy as np
import secrets
from typing import Dict, List, Tuple
from .network_generator import SimulationState


class ManaDistributor:
    """
    Distributes newly generated Mana each round.

    Design principles:
    - Deterministic portion ensures baseline participation incentive
    - Bloom events provide "rare but possible good fortune"
    - VRF ensures cryptographic fairness (1/n probability per nest)
    - Only non-anomalous nests (F=0) receive distribution
    """

    def __init__(self, daily_mana_base: float = 100.0,
                 deterministic_ratio: float = 0.75,
                 bloom_ratio: float = 0.25,
                 bloom_interval: int = 7):
        """
        Initialize Mana distributor.

        Args:
            daily_mana_base: Base Mana generated per day (scales with network size)
            deterministic_ratio: Fraction for even distribution (default 75%)
            bloom_ratio: Fraction for Bloom event lottery (default 25%)
            bloom_interval: Days between Bloom events
        """
        self.daily_mana_base = daily_mana_base
        self.deterministic_ratio = deterministic_ratio
        self.bloom_ratio = bloom_ratio
        self.bloom_interval = bloom_interval

        # Accumulated Bloom pool (collects daily until event triggers)
        self.bloom_pool = 0.0

        # Track Bloom event history
        self.bloom_history: List[Dict] = []

    def distribute_daily(self, state: SimulationState, day: int,
                        scores: Dict[int, Dict]) -> Dict[int, float]:
        """
        Distribute Mana for one day.

        Args:
            state: Current simulation state
            day: Current day number
            scores: Dict of nest_id -> {'S', 'F', ...} from ScoringEngine

        Returns:
            Dict of nest_id -> mana_received for this day
        """
        # Scale daily Mana with network size
        network_size = len(state.nests)
        daily_total = self.daily_mana_base * (network_size / 1000.0)

        # Split into deterministic and bloom portions
        deterministic_pool = daily_total * self.deterministic_ratio
        bloom_daily = daily_total * self.bloom_ratio

        # Accumulate Bloom pool
        self.bloom_pool += bloom_daily

        # Find eligible nests (F=0, not flagged as anomaly)
        eligible_ids = [
            nest_id for nest_id, score in scores.items()
            if score['F'] == 0
        ]

        distribution = {}

        # --- Deterministic distribution (75%) ---
        if eligible_ids:
            per_nest = deterministic_pool / len(eligible_ids)
            for nest_id in eligible_ids:
                distribution[nest_id] = per_nest
        else:
            # No eligible nests: Mana is not distributed (burned)
            pass

        # --- Bloom event (25%, triggered every bloom_interval days) ---
        if day > 0 and day % self.bloom_interval == 0:
            bloom_distribution = self._execute_bloom_event(
                state, day, eligible_ids
            )
            for nest_id, amount in bloom_distribution.items():
                distribution[nest_id] = distribution.get(nest_id, 0.0) + amount

        # Apply distribution to nest balances
        for nest_id, amount in distribution.items():
            state.nests_by_id[nest_id].balance += amount
            state.nests_by_id[nest_id].mana_received_today = amount

        # Reset mana_received_today for nests that didn't receive
        for nest in state.nests:
            if nest.nest_id not in distribution:
                nest.mana_received_today = 0.0

        return distribution

    def _execute_bloom_event(self, state: SimulationState, day: int,
                            eligible_ids: List[int]) -> Dict[int, float]:
        """
        Execute a Bloom event: distribute accumulated pool via VRF lottery.

        Uses cryptographic randomness (secrets module) to simulate VRF.
        Each eligible nest has exactly 1/n probability of winning.

        Returns:
            Dict of nest_id -> bloom_amount
        """
        if not eligible_ids or self.bloom_pool <= 0:
            return {}

        bloom_amount = self.bloom_pool
        self.bloom_pool = 0.0  # Reset pool

        # VRF-simulated selection: each eligible nest has equal probability
        # Use secrets for cryptographic randomness
        winner_index = secrets.randbelow(len(eligible_ids))
        winner_id = eligible_ids[winner_index]

        # Record Bloom event
        self.bloom_history.append({
            'day': day,
            'bloom_amount': bloom_amount,
            'winner_id': winner_id,
            'eligible_count': len(eligible_ids),
            'probability': 1.0 / len(eligible_ids)
        })

        return {winner_id: bloom_amount}

    def get_bloom_statistics(self) -> Dict:
        """
        Return statistics about Bloom events for fairness analysis.

        Returns:
            Dict with bloom event statistics
        """
        if not self.bloom_history:
            return {'total_events': 0}

        winners = [e['winner_id'] for e in self.bloom_history]
        unique_winners = set(winners)
        total_distributed = sum(e['bloom_amount'] for e in self.bloom_history)

        # Count wins per nest
        win_counts = {}
        for w in winners:
            win_counts[w] = win_counts.get(w, 0) + 1

        return {
            'total_events': len(self.bloom_history),
            'total_distributed': total_distributed,
            'unique_winners': len(unique_winners),
            'max_wins_single_nest': max(win_counts.values()) if win_counts else 0,
            'avg_bloom_amount': total_distributed / len(self.bloom_history),
            'history': self.bloom_history
        }


class DecayEngine:
    """
    Applies daily balance decay to all nests.

    Decay model:
    - Base decay rate applied to all nests (natural decay)
    - Anomalous nests (F=1) receive accelerated decay via g(S)
    - Progressive: larger balances decay faster
    """

    def __init__(self, base_decay_rate: float = 0.005,
                 progressive_threshold: float = 200.0,
                 progressive_multiplier: float = 1.5):
        """
        Initialize decay engine.

        Args:
            base_decay_rate: Daily decay rate (0.5% per day)
            progressive_threshold: Balance above which progressive decay kicks in
            progressive_multiplier: Multiplier for progressive decay
        """
        self.base_decay_rate = base_decay_rate
        self.progressive_threshold = progressive_threshold
        self.progressive_multiplier = progressive_multiplier

    def apply_decay(self, state: SimulationState, scores: Dict[int, Dict]) -> Dict[int, float]:
        """
        Apply decay to all nest balances.

        Args:
            state: Current simulation state
            scores: Dict of nest_id -> {'g_S', ...} from ScoringEngine

        Returns:
            Dict of nest_id -> decay_amount (positive value = amount decayed)
        """
        decay_amounts = {}

        for nest in state.nests:
            nest_id = nest.nest_id
            g_S = scores[nest_id]['g_S']

            # Base decay rate, accelerated by g(S)
            effective_rate = self.base_decay_rate * g_S

            # Progressive decay for large holders
            if nest.balance > self.progressive_threshold:
                excess = nest.balance - self.progressive_threshold
                progressive_rate = effective_rate * self.progressive_multiplier
                decay_amount = (self.progressive_threshold * effective_rate +
                               excess * progressive_rate)
            else:
                decay_amount = nest.balance * effective_rate

            # Apply decay
            decay_amount = min(decay_amount, nest.balance)  # Can't go negative
            nest.balance -= decay_amount
            decay_amounts[nest_id] = decay_amount

        return decay_amounts
