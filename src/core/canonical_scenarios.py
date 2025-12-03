"""
Canonical Taiko Fee Mechanism Scenario Evaluation

This module provides the SINGLE SOURCE OF TRUTH for scenario-based evaluation
of fee mechanism parameters. It integrates historical Ethereum L1 data for
constraint evaluation and optimization.

Key Features:
- Historical dataset loading and processing from CSV files
- Scenario replay for CRR and ruin probability evaluation
- Stress testing scenario generation
- Integration with optimization framework
- Comprehensive scenario validation and preprocessing

Scenario Types:
1. Historical: Real Ethereum L1 fee data (July 2022 spike, Luna crash, etc.)
2. Stress: Synthetic extreme scenarios for robustness testing
3. Monte Carlo: Statistical scenario generation for comprehensive evaluation

Usage:
    loader = CanonicalScenarioLoader()
    scenarios = loader.load_all_scenarios()
    evaluator = ScenarioEvaluator(scenarios)
    results = evaluator.evaluate_parameter_set(mu=0.0, nu=0.27, H=492)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import warnings
import csv
import os

from .canonical_fee_mechanism import CanonicalTaikoFeeCalculator, FeeParameters, VaultInitMode


class ScenarioType(Enum):
    """Types of evaluation scenarios."""
    HISTORICAL = "historical"       # Real Ethereum L1 data
    STRESS = "stress"              # Synthetic extreme scenarios
    MONTE_CARLO = "monte_carlo"    # Statistical scenario generation


@dataclass
class ScenarioData:
    """Container for scenario data with metadata."""

    name: str                           # Scenario identifier
    scenario_type: ScenarioType         # Type classification
    l1_basefee_wei: List[float]        # L1 basefee data in wei
    description: str = ""               # Human-readable description

    # Metadata
    source_file: Optional[str] = None   # Original data file
    block_range: Optional[Tuple[int, int]] = None  # Ethereum block range
    timestamp_range: Optional[Tuple[str, str]] = None  # Time period

    # Statistical properties (calculated automatically)
    duration_steps: int = 0             # Length in simulation steps
    mean_basefee_gwei: float = 0.0      # Average basefee in gwei
    max_basefee_gwei: float = 0.0       # Peak basefee in gwei
    volatility: float = 0.0             # Coefficient of variation

    def __post_init__(self):
        """Calculate statistical properties after initialization."""
        self.duration_steps = len(self.l1_basefee_wei)

        if len(self.l1_basefee_wei) > 0:
            basefee_gwei = np.array(self.l1_basefee_wei) / 1e9
            self.mean_basefee_gwei = float(np.mean(basefee_gwei))
            self.max_basefee_gwei = float(np.max(basefee_gwei))

            # Calculate volatility (coefficient of variation)
            if self.mean_basefee_gwei > 0:
                self.volatility = float(np.std(basefee_gwei) / self.mean_basefee_gwei)


@dataclass
class ScenarioEvaluationResults:
    """Results from evaluating parameter set against scenarios."""

    parameter_set: Dict[str, float]     # Evaluated parameters
    scenario_results: Dict[str, Dict]   # Results per scenario

    # Aggregate metrics
    average_crr: float = 0.0            # Average cost recovery ratio
    worst_case_crr: float = 0.0         # Worst CRR across scenarios
    average_ruin_prob: float = 0.0      # Average ruin probability
    max_ruin_prob: float = 0.0          # Maximum ruin probability

    # Constraint satisfaction
    crr_constraint_satisfied: bool = False      # CRR within bounds
    ruin_constraint_satisfied: bool = False     # Ruin prob within bounds
    all_constraints_satisfied: bool = False     # All constraints satisfied


class CanonicalScenarioLoader:
    """
    SINGLE SOURCE OF TRUTH for loading and managing evaluation scenarios.

    This class handles all scenario data loading, preprocessing, and validation
    to ensure consistency across the optimization framework.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize scenario loader.

        Args:
            data_dir: Path to data directory (defaults to project data/data_cache/)
        """
        if data_dir is None:
            # Default to project data directory
            current_dir = Path(__file__).parent.parent.parent
            self.data_dir = current_dir / "data" / "data_cache"
        else:
            self.data_dir = Path(data_dir)

        # Scenario registry
        self.scenarios: Dict[str, ScenarioData] = {}
        self.historical_files = {
            "july_2022_spike": "real_july_2022_spike_data.csv",
            "luna_crash": "luna_crash_true_peak_contiguous.csv",
            "recent_low_fees": "recent_low_fees_3hours.csv",
            "real_1hour": "real_1hour_contiguous.csv",
            "real_3hour": "real_3hour_contiguous.csv"
        }

    def load_all_scenarios(self) -> Dict[str, ScenarioData]:
        """
        Load all available scenarios from data directory.

        Returns:
            Dictionary of scenario name -> ScenarioData
        """
        self.scenarios.clear()

        # Load historical scenarios
        for name, filename in self.historical_files.items():
            try:
                scenario = self._load_historical_scenario(name, filename)
                if scenario is not None:
                    self.scenarios[name] = scenario
            except Exception as e:
                warnings.warn(f"Failed to load scenario {name}: {e}")

        # Generate stress test scenarios
        stress_scenarios = self._generate_stress_scenarios()
        self.scenarios.update(stress_scenarios)

        return self.scenarios

    def _load_historical_scenario(self, name: str, filename: str) -> Optional[ScenarioData]:
        """Load historical scenario from CSV file."""
        file_path = self.data_dir / filename

        if not file_path.exists():
            warnings.warn(f"Historical data file not found: {file_path}")
            return None

        try:
            # Read CSV with expected format: timestamp,basefee_wei,basefee_gwei,block_number
            df = pd.read_csv(file_path)

            # Validate required columns
            if 'basefee_wei' not in df.columns:
                warnings.warn(f"Missing 'basefee_wei' column in {filename}")
                return None

            # Extract basefee data
            basefee_wei = df['basefee_wei'].tolist()

            # Remove any invalid values
            basefee_wei = [float(x) for x in basefee_wei if pd.notna(x) and x > 0]

            if len(basefee_wei) == 0:
                warnings.warn(f"No valid basefee data in {filename}")
                return None

            # Extract metadata
            block_range = None
            timestamp_range = None

            if 'block_number' in df.columns:
                blocks = df['block_number'].dropna()
                if len(blocks) > 0:
                    # Handle hex block numbers
                    try:
                        if isinstance(blocks.iloc[0], str) and blocks.iloc[0].startswith('0x'):
                            first_block = int(blocks.iloc[0], 16)
                            last_block = int(blocks.iloc[-1], 16)
                        else:
                            first_block = int(blocks.iloc[0])
                            last_block = int(blocks.iloc[-1])
                        block_range = (first_block, last_block)
                    except (ValueError, IndexError):
                        pass

            if 'timestamp' in df.columns:
                timestamps = df['timestamp'].dropna()
                if len(timestamps) > 0:
                    try:
                        timestamp_range = (str(timestamps.iloc[0]), str(timestamps.iloc[-1]))
                    except IndexError:
                        pass

            return ScenarioData(
                name=name,
                scenario_type=ScenarioType.HISTORICAL,
                l1_basefee_wei=basefee_wei,
                description=f"Historical Ethereum L1 data: {name}",
                source_file=filename,
                block_range=block_range,
                timestamp_range=timestamp_range
            )

        except Exception as e:
            warnings.warn(f"Error loading {filename}: {e}")
            return None

    def _generate_stress_scenarios(self) -> Dict[str, ScenarioData]:
        """Generate synthetic stress test scenarios."""
        stress_scenarios = {}

        # Scenario 1: Extreme L1 fee spike
        spike_scenario = self._generate_fee_spike_scenario(
            name="extreme_spike",
            base_fee=15e9,          # 15 gwei baseline
            spike_peak=500e9,       # 500 gwei peak
            spike_duration=100,     # 100 steps spike
            total_duration=1000     # Total scenario length
        )
        stress_scenarios["extreme_spike"] = spike_scenario

        # Scenario 2: Sustained high fees
        sustained_high = self._generate_sustained_scenario(
            name="sustained_high_fees",
            fee_level=100e9,        # 100 gwei sustained
            duration=2000,          # Long duration
            volatility=0.3          # 30% volatility
        )
        stress_scenarios["sustained_high_fees"] = sustained_high

        # Scenario 3: Rapid oscillations
        oscillation_scenario = self._generate_oscillation_scenario(
            name="rapid_oscillations",
            base_fee=20e9,          # 20 gwei base
            amplitude=50e9,         # 50 gwei amplitude
            period=20,              # 20 step period
            total_duration=1000     # Total length
        )
        stress_scenarios["rapid_oscillations"] = oscillation_scenario

        # Scenario 4: Flash crash
        crash_scenario = self._generate_crash_scenario(
            name="flash_crash",
            initial_fee=50e9,       # Start at 50 gwei
            crash_fee=1e9,          # Crash to 1 gwei
            recovery_fee=30e9,      # Recover to 30 gwei
            crash_duration=50,      # 50 steps crash
            recovery_duration=200,  # 200 steps recovery
            total_duration=1000
        )
        stress_scenarios["flash_crash"] = crash_scenario

        return stress_scenarios

    def _generate_fee_spike_scenario(self, name: str, base_fee: float, spike_peak: float,
                                   spike_duration: int, total_duration: int) -> ScenarioData:
        """Generate a fee spike stress scenario."""
        basefee_data = []

        # Pre-spike: stable baseline
        pre_spike_length = (total_duration - spike_duration) // 2
        basefee_data.extend([base_fee] * pre_spike_length)

        # Spike: rapid rise and fall
        for i in range(spike_duration):
            progress = i / spike_duration
            if progress < 0.3:  # Rapid rise
                ratio = progress / 0.3
                fee = base_fee + (spike_peak - base_fee) * ratio
            elif progress < 0.7:  # Peak plateau
                fee = spike_peak
            else:  # Rapid fall
                ratio = (progress - 0.7) / 0.3
                fee = spike_peak - (spike_peak - base_fee) * ratio
            basefee_data.append(fee)

        # Post-spike: gradual recovery
        post_spike_length = total_duration - len(basefee_data)
        recovery_factor = 1.0
        for i in range(post_spike_length):
            recovery_factor *= 0.995  # Gradual decay
            fee = base_fee * (1.0 + 0.1 * recovery_factor)
            basefee_data.append(fee)

        return ScenarioData(
            name=name,
            scenario_type=ScenarioType.STRESS,
            l1_basefee_wei=basefee_data,
            description=f"Synthetic fee spike: {spike_peak/1e9:.1f} gwei peak"
        )

    def _generate_sustained_scenario(self, name: str, fee_level: float,
                                   duration: int, volatility: float) -> ScenarioData:
        """Generate sustained high fee scenario with volatility."""
        np.random.seed(42)  # Deterministic for testing

        # Generate GBM-like process
        basefee_data = []
        current_fee = fee_level

        for _ in range(duration):
            # Add random walk component
            shock = np.random.normal(0, volatility * 0.01)  # 1% per step volatility
            current_fee *= (1.0 + shock)

            # Mean reversion to fee_level
            reversion_strength = 0.05
            current_fee = current_fee * (1 - reversion_strength) + fee_level * reversion_strength

            # Floor at 1 gwei
            current_fee = max(current_fee, 1e9)

            basefee_data.append(current_fee)

        return ScenarioData(
            name=name,
            scenario_type=ScenarioType.STRESS,
            l1_basefee_wei=basefee_data,
            description=f"Sustained high fees: {fee_level/1e9:.1f} gwei ±{volatility*100:.0f}%"
        )

    def _generate_oscillation_scenario(self, name: str, base_fee: float, amplitude: float,
                                     period: int, total_duration: int) -> ScenarioData:
        """Generate oscillating fee scenario."""
        basefee_data = []

        for i in range(total_duration):
            # Sinusoidal oscillation
            phase = 2 * np.pi * i / period
            oscillation = amplitude * np.sin(phase)
            fee = base_fee + oscillation

            # Ensure positive fees
            fee = max(fee, 1e9)
            basefee_data.append(fee)

        return ScenarioData(
            name=name,
            scenario_type=ScenarioType.STRESS,
            l1_basefee_wei=basefee_data,
            description=f"Oscillating fees: {base_fee/1e9:.1f}±{amplitude/1e9:.1f} gwei"
        )

    def _generate_crash_scenario(self, name: str, initial_fee: float, crash_fee: float,
                               recovery_fee: float, crash_duration: int, recovery_duration: int,
                               total_duration: int) -> ScenarioData:
        """Generate flash crash scenario."""
        basefee_data = []

        # Pre-crash: stable at initial fee
        pre_crash_length = (total_duration - crash_duration - recovery_duration) // 2
        basefee_data.extend([initial_fee] * pre_crash_length)

        # Crash phase: rapid drop
        for i in range(crash_duration):
            progress = i / crash_duration
            fee = initial_fee - (initial_fee - crash_fee) * progress
            basefee_data.append(fee)

        # Recovery phase: gradual rise
        for i in range(recovery_duration):
            progress = i / recovery_duration
            fee = crash_fee + (recovery_fee - crash_fee) * progress
            basefee_data.append(fee)

        # Post-recovery: stable at recovery fee
        post_recovery_length = total_duration - len(basefee_data)
        basefee_data.extend([recovery_fee] * post_recovery_length)

        return ScenarioData(
            name=name,
            scenario_type=ScenarioType.STRESS,
            l1_basefee_wei=basefee_data,
            description=f"Flash crash: {initial_fee/1e9:.1f} → {crash_fee/1e9:.1f} → {recovery_fee/1e9:.1f} gwei"
        )


class ScenarioEvaluator:
    """
    Evaluates fee mechanism parameters against multiple scenarios.

    Provides constraint evaluation and performance metrics calculation
    across historical and stress test scenarios.
    """

    def __init__(self, scenarios: Optional[Dict[str, ScenarioData]] = None):
        """
        Initialize scenario evaluator.

        Args:
            scenarios: Dictionary of scenarios to evaluate against
        """
        if scenarios is None:
            # Auto-load all scenarios
            loader = CanonicalScenarioLoader()
            self.scenarios = loader.load_all_scenarios()
        else:
            self.scenarios = scenarios

        # Evaluation configuration
        self.crr_tolerance = 0.05       # ±5% CRR tolerance
        self.max_ruin_probability = 0.01  # 1% maximum ruin probability
        self.v_crit_ratio = 0.1         # 10% of target for ruin threshold

    def evaluate_parameter_set(self, **params) -> ScenarioEvaluationResults:
        """
        Evaluate parameter set against all scenarios.

        Args:
            **params: Fee mechanism parameters (mu, nu, H, etc.)

        Returns:
            ScenarioEvaluationResults with comprehensive evaluation
        """

        # Create parameter set with defaults
        param_dict = {
            'mu': params.get('mu', 0.0),
            'nu': params.get('nu', 0.27),
            'H': params.get('H', 492),
            'lambda_B': params.get('lambda_B', 0.1),
            'alpha_data': params.get('alpha_data', 20000.0),
            'Q_bar': params.get('Q_bar', 690000.0),
            'T': params.get('T', 1000.0)
        }

        scenario_results = {}
        crr_values = []
        ruin_probabilities = []

        for scenario_name, scenario in self.scenarios.items():
            try:
                result = self._evaluate_single_scenario(param_dict, scenario)
                scenario_results[scenario_name] = result

                crr_values.append(result['crr'])
                ruin_probabilities.append(result['ruin_probability'])

            except Exception as e:
                warnings.warn(f"Evaluation failed for scenario {scenario_name}: {e}")
                continue

        # Calculate aggregate metrics
        if crr_values:
            average_crr = float(np.mean(crr_values))
            worst_case_crr = float(np.min(crr_values))
        else:
            average_crr = 1.0
            worst_case_crr = 1.0

        if ruin_probabilities:
            average_ruin_prob = float(np.mean(ruin_probabilities))
            max_ruin_prob = float(np.max(ruin_probabilities))
        else:
            average_ruin_prob = 0.0
            max_ruin_prob = 0.0

        # Check constraint satisfaction
        crr_satisfied = abs(worst_case_crr - 1.0) <= self.crr_tolerance
        ruin_satisfied = max_ruin_prob <= self.max_ruin_probability
        all_satisfied = crr_satisfied and ruin_satisfied

        return ScenarioEvaluationResults(
            parameter_set=param_dict,
            scenario_results=scenario_results,
            average_crr=average_crr,
            worst_case_crr=worst_case_crr,
            average_ruin_prob=average_ruin_prob,
            max_ruin_prob=max_ruin_prob,
            crr_constraint_satisfied=crr_satisfied,
            ruin_constraint_satisfied=ruin_satisfied,
            all_constraints_satisfied=all_satisfied
        )

    def _evaluate_single_scenario(self, param_dict: Dict[str, float],
                                 scenario: ScenarioData) -> Dict[str, float]:
        """Evaluate parameters against single scenario."""

        # Create fee calculator with parameters
        fee_params = FeeParameters(**param_dict)
        calculator = CanonicalTaikoFeeCalculator(fee_params)

        # Initialize vault
        vault = calculator.create_vault(VaultInitMode.TARGET)

        # Run scenario simulation
        total_l2_revenue = 0.0
        total_l1_cost = 0.0
        vault_trajectory = [vault.balance]
        min_vault_balance = vault.balance

        # Limit simulation length for performance
        simulation_steps = min(len(scenario.l1_basefee_wei), 2000)

        for step in range(simulation_steps):
            l1_basefee_wei = scenario.l1_basefee_wei[step]

            # Calculate estimated fee
            estimated_fee_per_gas = calculator.calculate_estimated_fee_raw(
                l1_basefee_wei, vault.deficit
            )

            # Calculate transaction volume
            tx_volume = calculator.calculate_transaction_volume(
                estimated_fee_per_gas * fee_params.gas_per_tx
            )

            # L2 revenue calculation
            l2_gas_consumed = tx_volume * fee_params.gas_per_tx
            l2_revenue_step = estimated_fee_per_gas * l2_gas_consumed
            total_l2_revenue += l2_revenue_step

            # Collect fees
            vault.collect_fees(l2_revenue_step)

            # L1 costs (every batch interval)
            if step % fee_params.batch_interval_steps == 0:
                l1_cost_step = calculator.calculate_l1_batch_cost(l1_basefee_wei)
                total_l1_cost += l1_cost_step
                vault.pay_l1_costs(l1_cost_step)

            # Track vault trajectory
            vault_trajectory.append(vault.balance)
            min_vault_balance = min(min_vault_balance, vault.balance)

        # Calculate metrics
        crr = total_l2_revenue / total_l1_cost if total_l1_cost > 0 else 1.0

        # Ruin probability (binary: did it go below threshold?)
        v_crit = self.v_crit_ratio * fee_params.T
        ruin_occurred = min_vault_balance < v_crit
        ruin_probability = 1.0 if ruin_occurred else 0.0

        return {
            'crr': crr,
            'ruin_probability': ruin_probability,
            'min_vault_balance': min_vault_balance,
            'total_l2_revenue': total_l2_revenue,
            'total_l1_cost': total_l1_cost,
            'simulation_steps': simulation_steps
        }


# Convenience functions for integration with optimization framework

def load_default_scenarios() -> Dict[str, ScenarioData]:
    """Load default scenario set for optimization."""
    loader = CanonicalScenarioLoader()
    return loader.load_all_scenarios()


def evaluate_parameters_with_scenarios(mu: float = 0.0, nu: float = 0.27, H: int = 492,
                                      lambda_B: float = 0.1) -> ScenarioEvaluationResults:
    """Quick parameter evaluation with default scenarios."""
    evaluator = ScenarioEvaluator()
    return evaluator.evaluate_parameter_set(mu=mu, nu=nu, H=H, lambda_B=lambda_B)


def get_scenario_list_for_optimization() -> List[List[float]]:
    """Get scenario data in format expected by optimization framework."""
    scenarios = load_default_scenarios()

    # Convert to list of basefee lists
    scenario_list = []
    for scenario in scenarios.values():
        if len(scenario.l1_basefee_wei) > 100:  # Only use substantial scenarios
            scenario_list.append(scenario.l1_basefee_wei)

    return scenario_list


def validate_scenario_data(scenario_data: List[float], name: str = "unknown") -> bool:
    """Validate scenario data quality and format."""

    if not scenario_data or len(scenario_data) == 0:
        warnings.warn(f"Empty scenario data: {name}")
        return False

    if len(scenario_data) < 10:
        warnings.warn(f"Very short scenario ({len(scenario_data)} steps): {name}")
        return False

    # Check for invalid values
    invalid_count = sum(1 for x in scenario_data if not isinstance(x, (int, float)) or x <= 0)
    if invalid_count > 0:
        warnings.warn(f"Found {invalid_count} invalid basefee values in scenario: {name}")
        if invalid_count > len(scenario_data) * 0.1:  # > 10% invalid
            return False

    # Check for reasonable range
    basefee_gwei = [x / 1e9 for x in scenario_data if isinstance(x, (int, float)) and x > 0]
    if basefee_gwei:
        min_fee = min(basefee_gwei)
        max_fee = max(basefee_gwei)

        if min_fee < 0.001:  # Below 1 mwei
            warnings.warn(f"Very low basefee detected ({min_fee:.6f} gwei) in scenario: {name}")

        if max_fee > 10000:  # Above 10k gwei
            warnings.warn(f"Very high basefee detected ({max_fee:.1f} gwei) in scenario: {name}")

    return True