"""
Network generation for Meguri simulation.
Creates Watts-Strogatz small-world networks with marked attacker nests.
"""

import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Set


@dataclass
class Nest:
    """Represents a single nest (wallet) in the network"""

    nest_id: int
    is_attacker: bool = False
    device_id: int = 0  # Device binding
    balance: float = 0.0
    mana_received_today: float = 0.0
    mana_decay_rate: float = 1.0
    last_tx_day: int = 0
    total_transactions: int = 0

    def __hash__(self):
        return hash(self.nest_id)

    def __eq__(self, other):
        if isinstance(other, Nest):
            return self.nest_id == other.nest_id
        return False


class NetworkGenerator:
    """Generate Watts-Strogatz small-world networks for Meguri simulation"""

    def __init__(self, network_size: int, attacker_count: int, k_nearest: int = 4,
                 rewire_prob: float = 0.3, random_seed: int = 42):
        """
        Initialize network generator.

        Args:
            network_size: Total number of nests (N)
            attacker_count: Number of attacker-controlled nests (k)
            k_nearest: Number of nearest neighbors in ring topology
            rewire_prob: Probability of rewiring in Watts-Strogatz
            random_seed: Random seed for reproducibility
        """
        self.network_size = network_size
        self.attacker_count = min(attacker_count, network_size)  # Ensure k <= N
        self.k_nearest = k_nearest
        self.rewire_prob = rewire_prob
        self.random_seed = random_seed

        np.random.seed(random_seed)
        # Use numpy's random state for networkx operations

    def generate(self) -> Tuple[nx.Graph, List[Nest], List[int], List[int]]:
        """
        Generate network and initialize nests.

        Returns:
            - G: NetworkX graph (nodes are nest_ids)
            - nests: List of Nest objects
            - honest_ids: List of honest nest IDs
            - attacker_ids: List of attacker nest IDs
        """
        # Generate Watts-Strogatz small-world network
        G = nx.watts_strogatz_graph(
            n=self.network_size,
            k=self.k_nearest,
            p=self.rewire_prob,
            seed=self.random_seed
        )

        # Create Nest objects
        nests = []
        attacker_ids = set()
        honest_ids = []

        # Randomly select attacker IDs
        if self.attacker_count > 0:
            attacker_ids = set(np.random.choice(
                self.network_size,
                size=self.attacker_count,
                replace=False
            ))

        for nest_id in range(self.network_size):
            is_attacker = nest_id in attacker_ids

            # Device binding: each nest has a device ID
            # For attackers in device-constrained scenario: 1 attacker = 1 device
            # For this basic setup, attackers can have same device (unrestricted)
            device_id = nest_id if is_attacker else nest_id

            nest = Nest(
                nest_id=nest_id,
                is_attacker=is_attacker,
                device_id=device_id,
                balance=100.0  # Initial balance
            )
            nests.append(nest)

            if not is_attacker:
                honest_ids.append(nest_id)

        return G, nests, honest_ids, list(attacker_ids)

    def get_neighbors(self, G: nx.Graph, nest_id: int) -> List[int]:
        """Get list of neighboring nest IDs"""
        return list(G.neighbors(nest_id))


class SimulationState:
    """Maintains state of the entire simulation"""

    def __init__(self, G: nx.Graph, nests: List[Nest], honest_ids: List[int],
                 attacker_ids: List[int]):
        """
        Initialize simulation state.

        Args:
            G: Network graph
            nests: List of Nest objects
            honest_ids: List of honest nest IDs
            attacker_ids: List of attacker nest IDs
        """
        self.G = G
        self.nests_by_id = {nest.nest_id: nest for nest in nests}
        self.nests = nests
        self.honest_ids = set(honest_ids)
        self.attacker_ids = set(attacker_ids)

        # Historical data
        self.daily_transactions = {}  # {(from_id, to_id): [day1, day2, ...]}
        self.daily_amounts = {}  # {(from_id, to_id): [amount1, amount2, ...]}
        self.daily_nest_tx_count = {}  # {(day, nest_id): count}
        self.daily_nest_amounts = {}  # {(day, nest_id): [amounts]}

    def get_nest(self, nest_id: int) -> Nest:
        """Get nest by ID"""
        return self.nests_by_id[nest_id]

    def get_all_nests(self) -> List[Nest]:
        """Get all nests"""
        return self.nests

    def get_honest_nests(self) -> List[Nest]:
        """Get honest nests only"""
        return [self.nests_by_id[nid] for nid in self.honest_ids]

    def get_attacker_nests(self) -> List[Nest]:
        """Get attacker nests only"""
        return [self.nests_by_id[nid] for nid in self.attacker_ids]

    def record_transaction(self, day: int, from_id: int, to_id: int, amount: float):
        """Record a transaction for historical analysis"""
        key = (from_id, to_id)
        if key not in self.daily_transactions:
            self.daily_transactions[key] = []
            self.daily_amounts[key] = []

        self.daily_transactions[key].append(day)
        self.daily_amounts[key].append(amount)

        # Record nest-level transaction count
        nest_tx_key = (day, from_id)
        if nest_tx_key not in self.daily_nest_tx_count:
            self.daily_nest_tx_count[nest_tx_key] = 0
            self.daily_nest_amounts[nest_tx_key] = []

        self.daily_nest_tx_count[nest_tx_key] += 1
        self.daily_nest_amounts[nest_tx_key].append(amount)

        # Update last transaction day for sender
        self.nests_by_id[from_id].last_tx_day = day
        self.nests_by_id[from_id].total_transactions += 1

    def get_transaction_history(self, nest_id: int, day: int,
                               window_days: int) -> List[float]:
        """Get transaction amounts for a nest within recent days"""
        amounts = []
        for d in range(max(0, day - window_days), day + 1):
            key = (d, nest_id)
            if key in self.daily_nest_amounts:
                amounts.extend(self.daily_nest_amounts[key])
        return amounts

    def get_transaction_count(self, nest_id: int, day: int,
                             window_days: int) -> int:
        """Get transaction count for a nest within recent days"""
        count = 0
        for d in range(max(0, day - window_days), day + 1):
            key = (d, nest_id)
            if key in self.daily_nest_tx_count:
                count += self.daily_nest_tx_count[key]
        return count
