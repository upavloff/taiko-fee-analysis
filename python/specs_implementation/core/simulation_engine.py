"""
Simulation Engine Implementation

Integrates all components of the Taiko fee mechanism:
- L1 Cost Smoother (Section 2.3)
- Fee Controller (Section 3)
- Vault Dynamics (Section 4)

Provides a unified interface for running complete fee mechanism simulations
with realistic lumpy cash flows and exact mathematical implementations.
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd

from .l1_cost_smoother import L1CostSmoother
from .fee_controller import FeeController
from .vault_dynamics import VaultDynamics
from .validation import validate_simulation_inputs


class SimulationEngine:
    """
    Complete Taiko fee mechanism simulation engine.

    Integrates L1 cost smoothing, fee control, and vault dynamics to provide
    a unified simulation interface following SPECS.md exactly.
    """

    def __init__(
        self,
        # Core fee controller parameters
        mu: float,
        nu: float,
        horizon_h: int,
        target_vault_balance: float,
        # L1 cost smoothing
        lambda_c: float = 0.1,
        # Vault initialization
        initial_vault_balance: Optional[float] = None,
        # Gas calibration
        q_bar: float = 6.9e5,  # From Taiko Alethia data
        # Fee bounds (disabled for optimization)
        f_min: float = 0,   # No minimum bound
        f_max: float = float('inf'),  # No maximum bound
        # Rate limiting (disabled by default)
        kappa_up: float = 1.0,
        kappa_down: float = 1.0
    ):
        """
        Initialize complete simulation engine.

        Args:
            mu: L1 weight parameter μ ∈ [0,1]
            nu: Deficit weight parameter ν ∈ [0,1]
            horizon_h: Prediction horizon H > 0 (in batches)
            target_vault_balance: Target vault balance T > 0
            lambda_c: L1 cost smoothing parameter λ_C ∈ (0,1]
            initial_vault_balance: Initial vault balance. If None, starts at target.
            q_bar: Average gas per batch (default from Taiko data)
            f_min: Minimum basefee (wei per gas)
            f_max: Maximum basefee (wei per gas)
            kappa_up: Maximum relative up move per batch
            kappa_down: Maximum relative down move per batch
        """
        # Store core parameters
        self.mu = mu
        self.nu = nu
        self.horizon_h = horizon_h
        self.target_vault_balance = target_vault_balance
        self.q_bar = q_bar

        # Initialize components
        self.l1_smoother = L1CostSmoother(
            lambda_c=lambda_c,
            initial_cost=None  # Will initialize with first L1 cost
        )

        self.fee_controller = FeeController(
            mu=mu,
            nu=nu,
            horizon_h=horizon_h,
            q_bar=q_bar,
            f_min=f_min,
            f_max=f_max,
            kappa_up=kappa_up,
            kappa_down=kappa_down
        )

        self.vault_dynamics = VaultDynamics(
            target_balance=target_vault_balance,
            initial_balance=initial_vault_balance
        )

    def simulate_step(self, l1_cost_wei: Union[int, float]) -> Dict[str, Any]:
        """
        Simulate one complete step of the fee mechanism.

        Args:
            l1_cost_wei: Actual L1 cost C_L1(t) for this batch in WEI

        Returns:
            Dictionary with complete step results including all intermediate values
        """
        # Step 1: Update L1 cost smoother
        smoothed_l1_cost_wei = self.l1_smoother.update(l1_cost_wei)

        # Step 2: Get current deficit (in ETH, convert to wei)
        current_deficit_eth = self.vault_dynamics.get_deficit()
        current_deficit_wei = current_deficit_eth * 1e18

        # Step 3: Calculate L2 basefee (all inputs in wei)
        basefee_per_gas = self.fee_controller.calculate_fee(smoothed_l1_cost_wei, current_deficit_wei)

        # Step 4: Calculate revenue from basefee
        revenue = self.fee_controller.calculate_batch_revenue(basefee_per_gas)

        # Step 5: Update vault with revenue and L1 cost (convert L1 cost back to ETH for vault)
        l1_cost_eth = l1_cost_wei / 1e18
        vault_step = self.vault_dynamics.simulate_step(revenue, l1_cost_eth)

        # Step 6: Compile complete results
        return {
            # Input
            'l1_cost_actual': l1_cost_wei,

            # L1 smoothing
            'l1_cost_smoothed': smoothed_l1_cost_wei,

            # Fee calculation
            'deficit': current_deficit_eth,
            'basefee_per_gas': basefee_per_gas,
            'revenue': revenue,

            # Vault dynamics
            'subsidy_paid': vault_step['subsidy'],
            'vault_balance_before': vault_step['vault_balance_before'],
            'vault_balance_after': vault_step['vault_balance_after'],
            'deficit_after': vault_step['deficit_after'],
            'fully_subsidized': vault_step['fully_subsidized'],

            # Derived metrics
            'basefee_gwei': basefee_per_gas / 1e9,  # Convert to gwei
            'solvency_ratio': vault_step['vault_balance_after'] / self.target_vault_balance,

            # For debugging/analysis
            'raw_basefee': self.fee_controller.calculate_raw_basefee(smoothed_l1_cost_wei, current_deficit_wei),
            'clipped_basefee': self.fee_controller.apply_bounds(
                self.fee_controller.calculate_raw_basefee(smoothed_l1_cost_wei, current_deficit_wei)
            ),
        }

    @validate_simulation_inputs
    def simulate_series(self, l1_cost_series_wei: np.ndarray) -> pd.DataFrame:
        """
        Simulate the fee mechanism over a time series of L1 costs.

        Args:
            l1_cost_series_wei: Array of L1 costs [C_L1(0), ..., C_L1(T)] in WEI per batch

        Returns:
            DataFrame with complete simulation results
        """
        if len(l1_cost_series_wei) == 0:
            raise ValueError("L1 cost series cannot be empty")

        # Reset all components
        self.reset_state()

        # Run simulation
        results = []
        for i, l1_cost_wei in enumerate(l1_cost_series_wei):
            step_result = self.simulate_step(l1_cost_wei)
            step_result['step'] = i
            results.append(step_result)

        return pd.DataFrame(results)

    def reset_state(self) -> None:
        """
        Reset all component states for clean simulation.
        """
        self.l1_smoother.reset()
        self.fee_controller.reset_state()
        self.vault_dynamics.reset_state()

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get current state of all components.

        Returns:
            Dictionary with current state information
        """
        return {
            'vault_balance': self.vault_dynamics.get_vault_balance(),
            'vault_deficit': self.vault_dynamics.get_deficit(),
            'solvency_ratio': self.vault_dynamics.get_solvency_ratio(),
            'smoothed_l1_cost': self.l1_smoother.get_smoothed_cost(),
            'previous_fee': self.fee_controller.previous_fee,
            'target_balance': self.target_vault_balance,
            'parameters': {
                'mu': self.mu,
                'nu': self.nu,
                'horizon_h': self.horizon_h,
                'lambda_c': self.l1_smoother.lambda_c,
                'q_bar': self.q_bar,
            }
        }

    def calculate_metrics(self, simulation_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate key performance metrics from simulation results.

        Args:
            simulation_df: DataFrame from simulate_series()

        Returns:
            Dictionary of calculated metrics
        """
        if len(simulation_df) == 0:
            raise ValueError("Simulation DataFrame is empty")

        metrics = {}

        # Fee metrics
        fees_gwei = simulation_df['basefee_gwei']
        metrics['avg_fee_gwei'] = fees_gwei.mean()
        metrics['median_fee_gwei'] = fees_gwei.median()
        metrics['fee_std_gwei'] = fees_gwei.std()
        metrics['fee_cv'] = metrics['fee_std_gwei'] / metrics['avg_fee_gwei'] if metrics['avg_fee_gwei'] > 0 else 0
        metrics['fee_p95_gwei'] = fees_gwei.quantile(0.95)
        metrics['fee_p99_gwei'] = fees_gwei.quantile(0.99)

        # Vault metrics
        balances = simulation_df['vault_balance_after']
        deficits = simulation_df['deficit_after']
        metrics['avg_vault_balance'] = balances.mean()
        metrics['min_vault_balance'] = balances.min()
        metrics['avg_deficit'] = deficits.mean()
        metrics['max_deficit'] = deficits.max()
        metrics['insolvency_episodes'] = (balances < 0).sum()
        metrics['insolvency_rate'] = metrics['insolvency_episodes'] / len(simulation_df)

        # Solvency metrics
        solvency_ratios = simulation_df['solvency_ratio']
        metrics['avg_solvency_ratio'] = solvency_ratios.mean()
        metrics['min_solvency_ratio'] = solvency_ratios.min()

        # Revenue/cost recovery
        total_revenue = simulation_df['revenue'].sum()
        total_l1_costs = simulation_df['l1_cost_actual'].sum()
        metrics['cost_recovery_ratio'] = total_revenue / total_l1_costs if total_l1_costs > 0 else 0

        # Subsidy metrics
        subsidies = simulation_df['subsidy_paid']
        full_subsidies = simulation_df['fully_subsidized'].sum()
        metrics['avg_subsidy'] = subsidies.mean()
        metrics['full_subsidy_rate'] = full_subsidies / len(simulation_df)

        return metrics

    def validate_parameters(self) -> List[str]:
        """
        Validate all parameters are within acceptable ranges.

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []

        if not (0 <= self.mu <= 1):
            errors.append(f"mu must be in [0,1], got {self.mu}")
        if not (0 <= self.nu <= 1):
            errors.append(f"nu must be in [0,1], got {self.nu}")
        if self.horizon_h <= 0:
            errors.append(f"horizon_h must be positive, got {self.horizon_h}")
        if self.target_vault_balance <= 0:
            errors.append(f"target_vault_balance must be positive, got {self.target_vault_balance}")

        # Check for reasonable parameter combinations
        if self.nu / self.horizon_h >= 1:
            errors.append(f"ν/H = {self.nu}/{self.horizon_h} >= 1 may cause instability")

        return errors

    def __str__(self) -> str:
        """String representation of simulation engine."""
        return (f"SimulationEngine(μ={self.mu}, ν={self.nu}, H={self.horizon_h}, "
                f"T={self.target_vault_balance:.2e})")

    def __repr__(self) -> str:
        """Detailed representation of simulation engine."""
        return self.__str__()