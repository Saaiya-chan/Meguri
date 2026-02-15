"""
Configuration and parameter definitions for Phase 0 simulation
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================================
# DECAY MECHANISM PARAMETERS
# ============================================================================

@dataclass
class DecayConfig:
    """Parameters for decay mechanism"""

    # κ (Kappa) - decay acceleration coefficient
    # When S(v) < θ, increases decay rate: g(S) = 1 + κ × max(0, θ - S)
    kappa_range: List[float] = None  # Default: [0.5, 1.0, 2.0, 3.0]

    # θ (Theta) - anomaly threshold
    # If S(v) < θ, set F(v) = 1 (anomaly flag)
    theta_range: List[float] = None  # Default: [0.2, 0.3, 0.5, 0.7]

    def __post_init__(self):
        if self.kappa_range is None:
            self.kappa_range = [0.5, 1.0, 2.0, 3.0]
        if self.theta_range is None:
            self.theta_range = [0.2, 0.3, 0.5, 0.7]


# ============================================================================
# SCORING (PROOF OF BEING) PARAMETERS
# ============================================================================

@dataclass
class ScoringConfig:
    """Parameters for Proof of Being scoring (3-axis)"""

    # β - softmin scaling factor (lower = sharper, higher = smoother)
    beta_range: List[float] = None  # Default: [0.5, 1.0, 2.0]

    # Time axis decay rate (exponential: exp(-days / time_half_life))
    time_half_life: float = 14.0  # days

    # Recent activity window for T(v)
    activity_window: int = 14  # days

    # Rolling average window for baseline activity
    baseline_window: int = 7  # days

    # Number of time slots for entropy calculation (24 hours = hourly)
    entropy_time_slots: int = 24

    def __post_init__(self):
        if self.beta_range is None:
            self.beta_range = [0.5, 1.0, 2.0]


# ============================================================================
# MANA DISTRIBUTION PARAMETERS
# ============================================================================

@dataclass
class ManaConfig:
    """Parameters for Mana distribution"""

    # Hybrid ratio: 75% deterministic / 25% Bloom events
    deterministic_ratio: float = 0.75
    bloom_ratio: float = 0.25

    # Bloom event parameters
    bloom_interval_range: List[int] = None  # Days between Bloom events
    bloom_allocation_range: List[float] = None  # % of total Mana

    # Daily Mana generation (arbitrary base unit)
    daily_mana_base: float = 100.0

    def __post_init__(self):
        if self.bloom_interval_range is None:
            self.bloom_interval_range = [7, 10, 14]
        if self.bloom_allocation_range is None:
            self.bloom_allocation_range = [0.20, 0.25, 0.30]


# ============================================================================
# GRACE PERIOD PARAMETERS
# ============================================================================

@dataclass
class GracePeriodConfig:
    """Parameters for Grace Period (long-absence recovery)"""

    # Grace Period duration options
    grace_period_range: List[int] = None  # Days of reduced penalty

    # Device change grace period
    device_change_grace: int = 30  # days to complete device change

    # Inactivity threshold (to trigger Grace Period)
    inactivity_threshold: int = 60  # days

    def __post_init__(self):
        if self.grace_period_range is None:
            self.grace_period_range = [14, 30, 45, 60]


# ============================================================================
# COMMUNITY DETECTION PARAMETERS
# ============================================================================

@dataclass
class CommunityConfig:
    """Parameters for community detection and normalization"""

    # Analysis cycle (how often to recompute communities)
    analysis_cycle_options: List[str] = None  # ['weekly', 'monthly', 'quarterly']

    # Louvain resolution parameter
    louvain_resolution: float = 1.0

    # Random seed for Louvain (for reproducibility)
    random_seed: int = 42

    def __post_init__(self):
        if self.analysis_cycle_options is None:
            self.analysis_cycle_options = ['weekly', 'monthly', 'quarterly']


# ============================================================================
# NETWORK SIMULATION PARAMETERS
# ============================================================================

@dataclass
class NetworkConfig:
    """Parameters for network generation and transaction simulation"""

    # Network sizes for different test scenarios
    network_sizes: List[int] = None  # [1000, 10000, 100000]

    # Attacker nest counts for different scenarios
    attacker_counts: List[int] = None  # [1, 2, 5, 10, 20, 50, 100]

    # Watts-Strogatz parameters
    k_nearest: int = 4  # neighbors in ring topology
    rewire_prob: float = 0.3  # rewiring probability

    # Transaction parameters
    daily_tx_lambda: float = 2.0  # Poisson parameter for daily transactions
    tx_amount_min: int = 1
    tx_amount_max: int = 100
    local_transaction_prob: float = 0.7  # probability of local vs global transaction

    # Simulation duration
    simulation_days: int = 180

    def __post_init__(self):
        if self.network_sizes is None:
            self.network_sizes = [1000, 10000, 100000]
        if self.attacker_counts is None:
            self.attacker_counts = [1, 2, 5, 10, 20, 50, 100]


# ============================================================================
# TEST CASE CONFIGURATIONS
# ============================================================================

@dataclass
class TestCase:
    """Definition of a single test case"""

    test_id: str
    network_size: int
    attacker_count: int
    scenario: str  # A, B, C, D, E, F, G
    scenario_name: str
    description: str


def get_test_cases() -> List[TestCase]:
    """Return all test cases T001-T010"""
    return [
        TestCase(
            test_id="T001",
            network_size=1000,
            attacker_count=1,
            scenario="A",
            scenario_name="Simple Sybil (baseline)",
            description="ROI(1) ≈ 0.8~1.2"
        ),
        TestCase(
            test_id="T002",
            network_size=1000,
            attacker_count=10,
            scenario="A",
            scenario_name="Simple Sybil",
            description="ROI(10) < 1/10 = 0.1"
        ),
        TestCase(
            test_id="T003",
            network_size=10000,
            attacker_count=50,
            scenario="A",
            scenario_name="Simple Sybil",
            description="ROI(50) < 1/50 = 0.02"
        ),
        TestCase(
            test_id="T004",
            network_size=100000,
            attacker_count=100,
            scenario="A",
            scenario_name="Simple Sybil",
            description="ROI(100) < 1/150 ≈ 0.0067"
        ),
        TestCase(
            test_id="T005",
            network_size=10000,
            attacker_count=10,
            scenario="B",
            scenario_name="Device-constrained Sybil",
            description="ROI(10) << 0.1"
        ),
        TestCase(
            test_id="T006",
            network_size=10000,
            attacker_count=5,
            scenario="C",
            scenario_name="High-frequency loop (entropy spoofing)",
            description="TPR > 90%"
        ),
        TestCase(
            test_id="T007",
            network_size=10000,
            attacker_count=0,
            scenario="D",
            scenario_name="Rural community false positive",
            description="FPR < 1%"
        ),
        TestCase(
            test_id="T008",
            network_size=10000,
            attacker_count=0,
            scenario="E",
            scenario_name="New participant onboarding",
            description="Reach θ within 30 days"
        ),
        TestCase(
            test_id="T009",
            network_size=10000,
            attacker_count=0,
            scenario="F",
            scenario_name="Long-absence return (Grace Period)",
            description="Success rate > 99%"
        ),
        TestCase(
            test_id="T010",
            network_size=10000,
            attacker_count=None,  # 0-100 range
            scenario="G",
            scenario_name="Distance axis value measurement",
            description="3-layer vs 4-layer Sybil ROI comparison"
        ),
    ]


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

class MeguriPhase0Config:
    """Master configuration for Phase 0 simulation"""

    def __init__(self):
        self.decay = DecayConfig()
        self.scoring = ScoringConfig()
        self.mana = ManaConfig()
        self.grace_period = GracePeriodConfig()
        self.community = CommunityConfig()
        self.network = NetworkConfig()
        self.test_cases = get_test_cases()

    def get_parameter_ranges(self) -> Dict[str, List]:
        """Return all parameter ranges for DoE exploration"""
        return {
            'kappa': self.decay.kappa_range,
            'theta': self.decay.theta_range,
            'beta': self.scoring.beta_range,
            'bloom_interval': self.mana.bloom_interval_range,
            'bloom_allocation': self.mana.bloom_allocation_range,
            'grace_period': self.grace_period.grace_period_range,
            'analysis_cycle': self.community.analysis_cycle_options,
        }
