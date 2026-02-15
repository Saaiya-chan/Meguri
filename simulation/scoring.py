"""
Scoring system for Proof of Being (PoB).
Implements 3-axis evaluation: Time, Entropy, Distance.
"""

import numpy as np
from typing import Dict, Tuple, List
from .network_generator import SimulationState


class ScoringEngine:
    """
    Calculates PoB scores for each nest.

    Three-axis evaluation:
    - T(v): Time axis - activity rhythm and freshness
    - E(v): Entropy axis - behavioral randomness (unpredictability)
    - D(v): Distance axis - geographic/network consistency (optional for Phase 0)

    Integrated score: S(v) = softmin([T, E, D], β)
    Binary flag: F(v) = 1 if S(v) < θ else 0
    Decay acceleration: g(S) = 1 + κ × max(0, θ - S)
    """

    def __init__(self, time_half_life: float = 14.0,
                 activity_window: int = 14,
                 baseline_window: int = 7,
                 entropy_time_slots: int = 24):
        """
        Initialize scoring engine.

        Args:
            time_half_life: Half-life for exponential decay (days)
            activity_window: Window for recent activity (days)
            baseline_window: Window for baseline calculation (days)
            entropy_time_slots: Number of time slots for entropy (e.g., 24 for hourly)
        """
        self.time_half_life = time_half_life
        self.activity_window = activity_window
        self.baseline_window = baseline_window
        self.entropy_time_slots = entropy_time_slots

    def calculate_time_score(self, state: SimulationState, nest_id: int, day: int) -> float:
        """
        Calculate time axis score T(v).

        T(v) = activity_rate × freshness
        - activity_rate: recent transaction count vs baseline
        - freshness: exponential decay from last transaction
        """
        # Recent transaction count
        recent_tx = state.get_transaction_count(nest_id, day, self.activity_window)

        # Baseline transaction count (expected activity)
        baseline_tx = state.get_transaction_count(nest_id, day, self.baseline_window)
        baseline_tx = max(baseline_tx / self.baseline_window * self.baseline_window, 1.0)

        activity_rate = min(recent_tx / max(baseline_tx, 1.0), 2.0)  # Cap at 2.0

        # Freshness: exponential decay from last transaction
        days_since_tx = day - state.nests_by_id[nest_id].last_tx_day
        freshness = np.exp(-days_since_tx / self.time_half_life)

        T = activity_rate * freshness
        return min(T, 1.0)  # Normalize to [0, 1]

    def calculate_entropy_score(self, state: SimulationState, nest_id: int, day: int) -> float:
        """
        Calculate entropy axis score E(v).

        E(v) = -Σ p_i × log(p_i)  (Shannon entropy)
        where p_i = relative frequency of transactions in time slot i

        High entropy (E close to 1.0) = human-like random behavior
        Low entropy (E close to 0) = bot-like regular patterns
        """
        amounts = state.get_transaction_history(nest_id, day, self.activity_window)

        if not amounts:
            return 0.5  # Default for inactive users

        # Assign transactions to time slots based on day
        # For simplicity: use transaction order modulo entropy_time_slots
        slot_counts = np.zeros(self.entropy_time_slots)

        for i, amount in enumerate(amounts):
            slot = (i % self.entropy_time_slots)
            slot_counts[slot] += 1

        # Calculate Shannon entropy
        total = np.sum(slot_counts)
        if total == 0:
            return 0.5

        probabilities = slot_counts / total
        probabilities = probabilities[probabilities > 0]  # Exclude zero entries

        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

        # Normalize to [0, 1]
        # Maximum entropy for 24 slots: log(24) ≈ 3.178
        max_entropy = np.log(self.entropy_time_slots)
        normalized_entropy = min(entropy / max_entropy, 1.0)

        return normalized_entropy

    def calculate_distance_score(self, state: SimulationState, nest_id: int, day: int) -> float:
        """
        Calculate distance axis score D(v).

        D(v) = consistency_score - movement_cost

        For Phase 0 MVP, return constant. Implement full version in Stage 2.
        """
        # Phase 0: Distance axis is optional
        # Return neutral score
        return 0.5

    def calculate_softmin(self, scores: List[float], beta: float = 1.0) -> float:
        """
        Calculate softmin of scores.
        Softmin emphasizes the weakest axis.

        softmin(x) = -log(Σ exp(-β × x_i)) / β
        """
        if not scores:
            return 0.5

        scores = np.array(scores, dtype=float)
        scores = np.clip(scores, 0.0, 1.0)  # Ensure [0, 1]

        # softmin
        exp_terms = np.exp(-beta * scores)
        softmin_val = -np.log(np.sum(exp_terms) / len(scores) + 1e-10) / beta

        # Normalize to [0, 1]
        normalized = min(max(softmin_val, 0.0), 1.0)

        return float(normalized)

    def calculate_integrated_score(self, state: SimulationState, nest_id: int, day: int,
                                  beta: float = 1.0) -> float:
        """
        Calculate integrated score S(v).

        S(v) = softmin([T(v), E(v), D(v)], β)
        """
        T = self.calculate_time_score(state, nest_id, day)
        E = self.calculate_entropy_score(state, nest_id, day)
        D = self.calculate_distance_score(state, nest_id, day)

        S = self.calculate_softmin([T, E, D], beta)

        return S

    def calculate_anomaly_flag(self, S: float, theta: float) -> int:
        """
        Calculate binary anomaly flag F(v).

        F(v) = 1 if S(v) < θ (anomaly detected)
               0 if S(v) >= θ (normal behavior)
        """
        return 1 if S < theta else 0

    def calculate_decay_acceleration(self, S: float, theta: float, kappa: float) -> float:
        """
        Calculate decay acceleration factor g(S).

        g(S) = 1 + κ × max(0, θ - S)

        If S >= θ: g(S) = 1 (normal decay)
        If S < θ: g(S) = 1 + κ × (θ - S) (accelerated decay)
        """
        return 1.0 + kappa * max(0.0, theta - S)

    def calculate_all_scores(self, state: SimulationState, day: int,
                            kappa: float = 1.0, theta: float = 0.3,
                            beta: float = 1.0) -> Dict[int, Dict]:
        """
        Calculate all scores for all nests.

        Returns:
            Dict mapping nest_id → {
                'T': T(v),
                'E': E(v),
                'D': D(v),
                'S': S(v),
                'F': F(v),
                'g_S': g(S)
            }
        """
        scores = {}

        for nest in state.get_all_nests():
            nest_id = nest.nest_id

            T = self.calculate_time_score(state, nest_id, day)
            E = self.calculate_entropy_score(state, nest_id, day)
            D = self.calculate_distance_score(state, nest_id, day)
            S = self.calculate_softmin([T, E, D], beta)
            F = self.calculate_anomaly_flag(S, theta)
            g_S = self.calculate_decay_acceleration(S, theta, kappa)

            scores[nest_id] = {
                'T': float(T),
                'E': float(E),
                'D': float(D),
                'S': float(S),
                'F': int(F),
                'g_S': float(g_S)
            }

        return scores
