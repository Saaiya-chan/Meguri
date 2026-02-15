"""
Scoring system for Proof of Being (PoB).
Implements 3-axis evaluation: Time, Entropy, Distance.

Stage 3 improvements:
- T(v): Bell-curve shaped activity score (both under-active and hyper-active penalized)
- E(v): Combined time-distribution + partner-diversity entropy (detects loop attacks)
- D(v): Network-distance consistency score (optional, for T010 comparison)
"""

import numpy as np
from typing import Dict, List
from .network_generator import SimulationState


class ScoringEngine:
    """
    Calculates PoB scores for each nest.

    Three-axis evaluation:
    - T(v): Time axis — natural activity rhythm (bell curve around expected rate)
    - E(v): Entropy axis — combined time diversity + partner diversity
    - D(v): Distance axis — network-distance consistency (Phase 0: optional)

    Integrated score: S(v) = softmin([T, E, D], β)
    Binary flag:      F(v) = 1 if S(v) < θ
    Decay accel:      g(S) = 1 + κ × max(0, θ - S)
    """

    def __init__(self, time_half_life: float = 14.0,
                 activity_window: int = 14,
                 baseline_window: int = 7,
                 entropy_time_slots: int = 24,
                 expected_daily_tx: float = 2.0,
                 partner_entropy_weight: float = 0.5):
        """
        Args:
            time_half_life:        Half-life for freshness decay (days)
            activity_window:       Window for recent-activity analysis (days)
            baseline_window:       Window for rolling-average baseline (days)
            entropy_time_slots:    Slots for time-distribution entropy (24 = hourly)
            expected_daily_tx:     Natural transaction rate per day (Poisson λ)
            partner_entropy_weight: 0.0 = time-only, 1.0 = partner-only, 0.5 = equal
        """
        self.time_half_life = time_half_life
        self.activity_window = activity_window
        self.baseline_window = baseline_window
        self.entropy_time_slots = entropy_time_slots
        self.expected_daily_tx = expected_daily_tx
        self.partner_entropy_weight = partner_entropy_weight

    # ------------------------------------------------------------------ #
    # T(v) — Time Axis                                                     #
    # ------------------------------------------------------------------ #
    def calculate_time_score(self, state: SimulationState, nest_id: int, day: int) -> float:
        """
        T(v) = bell_activity × freshness

        bell_activity: Gaussian around expected_daily_tx.
          - Too inactive  → low score
          - Just right    → high score  (peak at λ = 2 tx/day)
          - Hyper-active  → also lowered  (loop attackers do 5-10×)

        freshness: exponential decay from last transaction date.
        """
        # Recent transaction count over activity_window
        recent_tx = state.get_transaction_count(nest_id, day, self.activity_window)
        recent_daily_rate = recent_tx / max(self.activity_window, 1)

        # Gaussian bell centered at expected_daily_tx
        # σ chosen so that 3× the expected rate scores ~0.1
        sigma = self.expected_daily_tx * 1.2
        deviation = (recent_daily_rate - self.expected_daily_tx) / sigma
        bell_activity = float(np.exp(-0.5 * deviation ** 2))

        # Freshness
        days_since_tx = day - state.nests_by_id[nest_id].last_tx_day
        freshness = float(np.exp(-days_since_tx / self.time_half_life))

        T = bell_activity * freshness
        return min(max(T, 0.0), 1.0)

    # ------------------------------------------------------------------ #
    # E(v) — Entropy Axis                                                  #
    # ------------------------------------------------------------------ #
    def calculate_entropy_score(self, state: SimulationState, nest_id: int, day: int) -> float:
        """
        E(v) = (1 - w) × time_entropy  +  w × partner_entropy

        time_entropy:    Shannon entropy of *which day-of-week* transactions occur.
                         Loop bots cluster on specific days → lower entropy.
        partner_entropy: Shannon entropy of counterparty distribution.
                         Loop bots talk to only a few partners → lower entropy.

        w = partner_entropy_weight (default 0.5)
        """
        time_ent = self._time_entropy(state, nest_id, day)
        partner_ent = self._partner_entropy(state, nest_id, day)

        w = self.partner_entropy_weight
        E = (1.0 - w) * time_ent + w * partner_ent
        return float(np.clip(E, 0.0, 1.0))

    def _time_entropy(self, state: SimulationState, nest_id: int, day: int) -> float:
        """Shannon entropy over day-of-week slots (7 slots) within activity_window."""
        n_slots = 7  # day-of-week
        slot_counts = np.zeros(n_slots)

        for d in range(max(0, day - self.activity_window), day + 1):
            count = state.daily_nest_tx_count.get((d, nest_id), 0)
            if count > 0:
                slot_counts[d % n_slots] += count

        total = slot_counts.sum()
        if total == 0:
            return 0.5  # no activity → neutral

        p = slot_counts / total
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        return min(entropy / np.log(n_slots), 1.0)

    def _partner_entropy(self, state: SimulationState, nest_id: int, day: int) -> float:
        """
        Shannon entropy of counterparty distribution.

        Counts distinct partners and their transaction frequencies.
        Loop attackers (few partners, many tx each) → low entropy.
        Honest users (many partners, varied frequency) → high entropy.
        """
        partner_counts: Dict[int, int] = {}

        for (from_id, to_id), days_list in state.daily_transactions.items():
            if from_id != nest_id:
                continue
            recent = sum(1 for d in days_list if day - self.activity_window <= d <= day)
            if recent > 0:
                partner_counts[to_id] = partner_counts.get(to_id, 0) + recent

        if not partner_counts:
            return 0.5  # no outgoing tx → neutral

        counts = np.array(list(partner_counts.values()), dtype=float)
        p = counts / counts.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-12)))

        # Normalize: max entropy = log(N unique partners).
        # Cap normalizer at log(20) to avoid instability for very social nodes.
        n_partners = len(partner_counts)
        max_ent = np.log(max(n_partners, 2))
        return min(entropy / max_ent, 1.0)

    # ------------------------------------------------------------------ #
    # D(v) — Distance Axis                                                 #
    # ------------------------------------------------------------------ #
    def calculate_distance_score(self, state: SimulationState, nest_id: int, day: int,
                                 use_network_distance: bool = False) -> float:
        """
        D(v) — network-distance consistency score.

        use_network_distance=False (default): neutral 0.5 placeholder.
        use_network_distance=True  (T010):    penalizes nodes whose transaction
            partners are consistently far away in the network graph.

        Intuition: real users transact mostly with nearby neighbours;
        attackers operating multiple nests may violate this locality.
        """
        if not use_network_distance:
            return 0.5

        # --- Network-distance implementation (used in T010) ---
        G = state.G
        partner_counts: Dict[int, int] = {}
        for (from_id, to_id), days_list in state.daily_transactions.items():
            if from_id != nest_id:
                continue
            recent = sum(1 for d in days_list if day - self.activity_window <= d <= day)
            if recent > 0:
                partner_counts[to_id] = partner_counts.get(to_id, 0) + recent

        if not partner_counts:
            return 0.5

        try:
            import networkx as nx
            distances = []
            for partner_id, cnt in partner_counts.items():
                try:
                    d = nx.shortest_path_length(G, source=nest_id, target=partner_id)
                    distances.extend([d] * cnt)
                except nx.NetworkXNoPath:
                    distances.append(10)  # disconnected → far

            if not distances:
                return 0.5

            avg_dist = np.mean(distances)
            # Locality score: peak at distance 1-2, decays for larger distances
            # score = exp(-avg_dist / 3.0)
            locality = float(np.exp(-avg_dist / 3.0))
            return min(max(locality, 0.0), 1.0)

        except Exception:
            return 0.5

    # ------------------------------------------------------------------ #
    # Softmin / Aggregate                                                  #
    # ------------------------------------------------------------------ #
    def calculate_softmin(self, scores: List[float], beta: float = 1.0) -> float:
        """
        softmin emphasizes the weakest axis.
        softmin(x) = -log( mean(exp(-β × x_i)) ) / β
        """
        if not scores:
            return 0.5

        arr = np.clip(np.array(scores, dtype=float), 0.0, 1.0)
        val = float(-np.log(np.mean(np.exp(-beta * arr)) + 1e-12) / beta)
        return float(np.clip(val, 0.0, 1.0))

    def calculate_anomaly_flag(self, S: float, theta: float) -> int:
        return 1 if S < theta else 0

    def calculate_decay_acceleration(self, S: float, theta: float, kappa: float) -> float:
        return 1.0 + kappa * max(0.0, theta - S)

    # ------------------------------------------------------------------ #
    # Batch calculation                                                    #
    # ------------------------------------------------------------------ #
    def calculate_all_scores(self, state: SimulationState, day: int,
                             kappa: float = 1.0, theta: float = 0.3,
                             beta: float = 1.0,
                             use_network_distance: bool = False) -> Dict[int, Dict]:
        scores = {}
        for nest in state.get_all_nests():
            nid = nest.nest_id
            T = self.calculate_time_score(state, nid, day)
            E = self.calculate_entropy_score(state, nid, day)
            D = self.calculate_distance_score(state, nid, day, use_network_distance)
            S = self.calculate_softmin([T, E, D], beta)
            F = self.calculate_anomaly_flag(S, theta)
            g_S = self.calculate_decay_acceleration(S, theta, kappa)

            scores[nid] = {
                'T': float(T),
                'E': float(E),
                'D': float(D),
                'S': float(S),
                'F': int(F),
                'g_S': float(g_S),
            }
        return scores
