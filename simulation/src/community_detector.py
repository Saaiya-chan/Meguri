"""
Community detection and normalization for Meguri.

Monthly offline analysis:
- Detects communities using Louvain algorithm
- Calculates normalization factor α(c) per community
- Ensures rural/small communities aren't penalized
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

from .network_generator import SimulationState


class CommunityDetector:
    """
    Performs community detection on the transaction graph.

    Uses Louvain algorithm for clustering.
    Calculates normalization coefficient α(c) = global_avg_q / community_avg_q
    so that communities with lower average scores aren't structurally disadvantaged.
    """

    def __init__(self, resolution: float = 1.0, random_seed: int = 42):
        """
        Initialize community detector.

        Args:
            resolution: Louvain resolution parameter (higher = more communities)
            random_seed: Random seed for reproducibility
        """
        self.resolution = resolution
        self.random_seed = random_seed

        # Cached community assignments
        self.communities: Optional[Dict[int, int]] = None  # nest_id -> community_id
        self.alpha: Optional[Dict[int, float]] = None  # community_id -> α(c)
        self.last_detection_day: int = -1
        self.detection_history: List[Dict] = []

    def detect_communities(self, state: SimulationState, day: int,
                          scores: Dict[int, Dict]) -> Dict[int, int]:
        """
        Run Louvain community detection on the transaction graph.

        Args:
            state: Current simulation state
            day: Current day number
            scores: Dict of nest_id -> {'S', ...}

        Returns:
            Dict of nest_id -> community_id
        """
        # Build weighted transaction graph from recent history
        tx_graph = self._build_transaction_graph(state, day)

        if community_louvain is not None:
            # Use python-louvain library
            partition = community_louvain.best_partition(
                tx_graph,
                resolution=self.resolution,
                random_state=self.random_seed
            )
        else:
            # Fallback: use NetworkX's built-in community detection
            from networkx.algorithms.community import greedy_modularity_communities
            communities_list = greedy_modularity_communities(tx_graph)
            partition = {}
            for comm_id, comm_nodes in enumerate(communities_list):
                for node in comm_nodes:
                    partition[node] = comm_id

        # Ensure all nests have a community assignment
        for nest in state.nests:
            if nest.nest_id not in partition:
                partition[nest.nest_id] = -1  # Unassigned / isolated

        self.communities = partition
        self.last_detection_day = day

        # Calculate α(c) normalization coefficients
        self.alpha = self._calculate_normalization(partition, scores)

        # Record detection results
        community_ids = set(partition.values())
        self.detection_history.append({
            'day': day,
            'num_communities': len(community_ids),
            'community_sizes': self._get_community_sizes(partition),
            'alpha_values': dict(self.alpha) if self.alpha else {},
        })

        return partition

    def _build_transaction_graph(self, state: SimulationState, day: int,
                                window_days: int = 30) -> nx.Graph:
        """
        Build a weighted graph from recent transactions.

        Edge weight = number of transactions between two nests in the window.
        """
        G = nx.Graph()

        # Add all nest nodes
        for nest in state.nests:
            G.add_node(nest.nest_id)

        # Add edges from transaction history
        edge_weights = {}
        for (from_id, to_id), days in state.daily_transactions.items():
            # Count recent transactions
            recent_count = sum(1 for d in days if d >= max(0, day - window_days))
            if recent_count > 0:
                edge = tuple(sorted([from_id, to_id]))
                edge_weights[edge] = edge_weights.get(edge, 0) + recent_count

        for (u, v), weight in edge_weights.items():
            G.add_edge(u, v, weight=weight)

        return G

    def _calculate_normalization(self, partition: Dict[int, int],
                                scores: Dict[int, Dict]) -> Dict[int, float]:
        """
        Calculate normalization coefficient α(c) for each community.

        α(c) = global_avg_q / community_avg_q

        This ensures communities with naturally lower activity (e.g., rural)
        aren't structurally penalized.
        """
        if not scores:
            return {}

        # Global average score
        all_scores = [s['S'] for s in scores.values()]
        global_avg = np.mean(all_scores) if all_scores else 0.5

        # Per-community average scores
        community_scores = {}
        for nest_id, comm_id in partition.items():
            if nest_id in scores:
                if comm_id not in community_scores:
                    community_scores[comm_id] = []
                community_scores[comm_id].append(scores[nest_id]['S'])

        # Calculate α(c)
        alpha = {}
        for comm_id, comm_scores in community_scores.items():
            comm_avg = np.mean(comm_scores) if comm_scores else global_avg
            # Avoid division by zero; cap normalization factor
            alpha[comm_id] = min(max(global_avg / max(comm_avg, 0.01), 0.5), 2.0)

        return alpha

    def get_normalized_score(self, nest_id: int, raw_score: float) -> float:
        """
        Apply community normalization to a raw score.

        Args:
            nest_id: Nest ID
            raw_score: Raw S(v) score

        Returns:
            Normalized score: S(v) × α(c)
        """
        if self.communities is None or self.alpha is None:
            return raw_score

        comm_id = self.communities.get(nest_id, -1)
        alpha_c = self.alpha.get(comm_id, 1.0)

        return min(raw_score * alpha_c, 1.0)

    def _get_community_sizes(self, partition: Dict[int, int]) -> Dict[int, int]:
        """Get number of nests per community."""
        sizes = {}
        for comm_id in partition.values():
            sizes[comm_id] = sizes.get(comm_id, 0) + 1
        return sizes

    def get_stability_score(self) -> float:
        """
        Calculate stability of community detection across multiple runs.

        Returns value between 0 (unstable) and 1 (perfectly stable).
        Requires at least 2 detection runs.
        """
        if len(self.detection_history) < 2:
            return 1.0  # Not enough data

        # Compare consecutive detections
        prev = self.detection_history[-2]
        curr = self.detection_history[-1]

        # Simple stability metric: how similar are community counts
        prev_n = prev['num_communities']
        curr_n = curr['num_communities']

        if max(prev_n, curr_n) == 0:
            return 1.0

        similarity = min(prev_n, curr_n) / max(prev_n, curr_n)
        return similarity

    def get_detection_summary(self) -> Dict:
        """Return summary of community detection results."""
        if not self.detection_history:
            return {'detections': 0}

        latest = self.detection_history[-1]
        return {
            'detections': len(self.detection_history),
            'latest_day': latest['day'],
            'num_communities': latest['num_communities'],
            'community_sizes': latest['community_sizes'],
            'alpha_values': latest['alpha_values'],
            'stability': self.get_stability_score(),
        }
