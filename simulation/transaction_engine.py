"""
Transaction simulation engine for Meguri.
Handles daily Poisson-distributed transactions and attacker behavior.
"""

import numpy as np
from typing import List, Tuple
from .network_generator import SimulationState


class TransactionEngine:
    """Simulates daily transactions in the network"""

    def __init__(self, daily_tx_lambda: float = 2.0,
                 tx_amount_min: int = 1,
                 tx_amount_max: int = 100,
                 local_transaction_prob: float = 0.7,
                 random_seed: int = 42):
        """
        Initialize transaction engine.

        Args:
            daily_tx_lambda: Poisson λ for daily transactions per nest
            tx_amount_min: Minimum transaction amount
            tx_amount_max: Maximum transaction amount
            local_transaction_prob: Probability of transacting with neighbors
            random_seed: Random seed
        """
        self.daily_tx_lambda = daily_tx_lambda
        self.tx_amount_min = tx_amount_min
        self.tx_amount_max = tx_amount_max
        self.local_transaction_prob = local_transaction_prob

        np.random.seed(random_seed)

    def simulate_day(self, state: SimulationState, day: int,
                     attacker_strategy: str = 'honest'):
        """
        Simulate one day of transactions.

        Args:
            state: Current simulation state
            day: Current day number (0-indexed)
            attacker_strategy: Strategy for attacker nests
                - 'honest': Behave like honest users
                - 'loop': High-frequency small-amount loops
                - 'random': Maximum randomness to evade entropy detection
        """
        G = state.G

        # For each nest, decide how many transactions to make today
        for nest in state.get_all_nests():
            nest_id = nest.nest_id

            # Poisson-distributed transactions
            num_transactions = np.random.poisson(self.daily_tx_lambda)

            for _ in range(num_transactions):
                # Decide transaction type: local vs global
                is_local = np.random.random() < self.local_transaction_prob

                if is_local:
                    # Local: pick a neighbor
                    neighbors = list(G.neighbors(nest_id))
                    if neighbors:
                        to_id = np.random.choice(neighbors)
                    else:
                        continue
                else:
                    # Global: pick any other nest
                    other_nests = [n for n in range(state.G.number_of_nodes())
                                  if n != nest_id]
                    to_id = np.random.choice(other_nests)

                # Determine transaction amount
                if nest.is_attacker and attacker_strategy == 'loop':
                    # Attacker: small amounts in loops
                    amount = np.random.randint(1, 5)
                elif nest.is_attacker and attacker_strategy == 'random':
                    # Attacker: maximize randomness
                    amount = np.random.randint(self.tx_amount_min, self.tx_amount_max + 1)
                else:
                    # Honest user or honest attacker strategy
                    amount = np.random.randint(self.tx_amount_min, self.tx_amount_max + 1)

                # Check if sender has sufficient balance
                if nest.balance >= amount:
                    # Execute transaction
                    nest.balance -= amount
                    state.nests_by_id[to_id].balance += amount

                    # Record transaction for history
                    state.record_transaction(day, nest_id, to_id, float(amount))


class AttackerStrategy:
    """Defines behavior strategies for attacker-controlled nests"""

    @staticmethod
    def honest(state: SimulationState, attacker_ids: List[int], day: int):
        """Attacker behaves like honest users (baseline)"""
        engine = TransactionEngine()
        for attacker_id in attacker_ids:
            nest = state.get_nest(attacker_id)
            # Simulate honest-like transaction pattern
            neighbors = list(state.G.neighbors(attacker_id))
            if not neighbors:
                continue

            num_tx = np.random.poisson(2.0)
            for _ in range(num_tx):
                if np.random.random() < 0.7:
                    to_id = np.random.choice(neighbors)
                else:
                    to_id = np.random.randint(0, state.G.number_of_nodes())
                    if to_id == attacker_id:
                        continue

                amount = np.random.randint(1, 101)
                if nest.balance >= amount:
                    nest.balance -= amount
                    state.nests_by_id[to_id].balance += amount
                    state.record_transaction(day, attacker_id, to_id, float(amount))

    @staticmethod
    def high_frequency_loop(state: SimulationState, attacker_ids: List[int], day: int):
        """Attacker uses high-frequency small-amount loops between controlled nests"""
        for attacker_id in attacker_ids:
            # Attacker creates loops with other attacker nests
            other_attackers = [a for a in attacker_ids if a != attacker_id]
            if not other_attackers:
                continue

            # High frequency
            num_tx = np.random.poisson(5.0)
            for _ in range(num_tx):
                to_id = np.random.choice(other_attackers)
                amount = np.random.randint(1, 5)  # Small amounts

                nest = state.get_nest(attacker_id)
                if nest.balance >= amount:
                    nest.balance -= amount
                    state.nests_by_id[to_id].balance += amount
                    state.record_transaction(day, attacker_id, to_id, float(amount))

    @staticmethod
    def entropy_spoofing(state: SimulationState, attacker_ids: List[int], day: int):
        """
        Attacker maximizes randomness to evade entropy-based detection.
        Randomizes timing, amounts, and counterparties.
        """
        for attacker_id in attacker_ids:
            # Random number of transactions
            num_tx = np.random.poisson(2.0)
            if num_tx == 0:
                num_tx = 1  # At least one to maintain activity

            for _ in range(num_tx):
                # Random counterparty
                to_id = np.random.randint(0, state.G.number_of_nodes())
                if to_id == attacker_id:
                    continue

                # Random amount
                amount = np.random.randint(1, 101)

                nest = state.get_nest(attacker_id)
                if nest.balance >= amount:
                    nest.balance -= amount
                    state.nests_by_id[to_id].balance += amount
                    state.record_transaction(day, attacker_id, to_id, float(amount))
