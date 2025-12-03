"""
Canonical Taiko Fee Mechanism Implementation

This module provides the SINGLE SOURCE OF TRUTH for all fee mechanism calculations.
All other implementations (Python analysis scripts, JavaScript web interface)
must use this module to ensure consistency across the entire codebase.

Key Features:
- Authoritative fee calculation formula implementation
- Vault state management with proper initialization modes
- L1 cost estimation with smoothing and outlier rejection
- Transaction volume modeling with demand elasticity
- Comprehensive parameter validation and bounds checking
- Fully documented mathematical formulations

Formula Reference:
    F_estimated(t) = max(μ × C_L1(t) + ν × D(t)/H, F_min)

Where:
    - μ: L1 weight parameter [0.0, 1.0]
    - ν: Deficit weight parameter [0.0, 1.0]
    - H: Prediction horizon (time steps)
    - C_L1(t): L1 cost per transaction at time t
    - D(t): Vault deficit at time t
    - F_min: Minimum fee floor (1e-8 ETH)
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from enum import Enum

# Import unit safety system
try:
    from .units import (
        Wei, Gwei, ETH, gwei_to_wei, wei_to_gwei, safe_gwei_to_wei, safe_wei_to_gwei,
        validate_fee_range, validate_basefee_range, assert_reasonable_fee,
        UnitValidationError, UnitOverflowError
    )
    UNIT_SAFETY_AVAILABLE = True
except ImportError:
    # Fallback if units module not available
    UNIT_SAFETY_AVAILABLE = False
    warnings.warn("🚨 MOCK DATA WARNING: Unit safety system not available - unit validation disabled")


# Mock data detection and warnings
def warn_mock_data_usage(context: str, data_type: str, reason: str) -> None:
    """Issue clear warning when mock data is being used instead of real data."""
    warnings.warn(
        f"🚨 MOCK DATA WARNING: Using {data_type} in {context} - {reason}. "
        f"This violates scientific accuracy principles. Real data should be used in production.",
        UserWarning,
        stacklevel=3
    )


def validate_real_data_usage(data_source: str, value: any, context: str) -> None:
    """Validate that real data (not mock/hardcoded values) is being used."""
    # Check for common mock data patterns
    if isinstance(value, (int, float)):
        # Check for suspicious hardcoded values
        if value == 1.5 and "gwei" in context.lower():
            warn_mock_data_usage(context, "hardcoded 1.5 gwei base fee",
                                "artificial fee floor injection detected")
        elif value == 690000:
            warn_mock_data_usage(context, "arbitrary Q̄ = 690,000 constant",
                                "should use empirically measured gas consumption")
        elif value == 200 and "gas" in context.lower():
            warn_mock_data_usage(context, "hardcoded 200 gas",
                                "should be 20,000 gas per documentation")

    # Check data source provenance
    if data_source in ["mock", "placeholder", "fallback", "default", "hardcoded"]:
        warn_mock_data_usage(context, f"{data_source} data",
                           f"data source '{data_source}' indicates non-real data usage")


class VaultInitMode(Enum):
    """Vault initialization strategies for simulation scenarios."""
    TARGET = "target"           # Start at target balance
    DEFICIT = "deficit"         # Start below target (underfunded)
    SURPLUS = "surplus"         # Start above target (overfunded)
    CUSTOM = "custom"           # Use provided balance


@dataclass
class FeeParameters:
    """Canonical parameter set for fee mechanism configuration."""

    # Core mechanism parameters
    mu: float = 0.0                    # L1 weight [0.0, 1.0]
    nu: float = 0.27                   # Deficit weight [0.0, 1.0]
    H: int = 492                       # Prediction horizon (time steps)

    # Economic parameters
    target_balance: float = 1000.0     # Target vault balance (ETH)
    min_fee: float = 1e-8              # Minimum fee floor (ETH)

    # L1 cost calculation
    gas_per_tx: float = 20000.0        # Gas cost per L2 transaction (corrected from 2k bug)
    batch_interval_steps: int = 6      # Steps between L1 batch submissions

    # Transaction volume modeling
    base_tx_demand: float = 100.0      # Base transaction demand per step
    fee_elasticity: float = 2.0        # Demand elasticity to fee changes
    max_tx_demand: float = 1000.0      # Maximum transaction volume cap

    # Smoothing and bounds
    l1_smoothing_factor: float = 0.1   # EMA factor for L1 cost smoothing
    max_fee_change_ratio: float = 2.0  # Max fee change per step (ratio)
    outlier_threshold_sigma: float = 3.0  # Outlier rejection threshold

    # Guaranteed recovery (optional enhancement)
    guaranteed_recovery: bool = False   # Enable minimum deficit correction
    min_deficit_rate: float = 1e-3     # Minimum deficit correction per step

    def __post_init__(self):
        """Validate parameter ranges and consistency."""
        self._validate_parameters()

    def _validate_parameters(self):
        """Comprehensive parameter validation with detailed error messages."""
        errors = []

        # Core mechanism bounds
        if not (0.0 <= self.mu <= 1.0):
            errors.append(f"mu must be in [0.0, 1.0], got {self.mu}")
        if not (0.0 <= self.nu <= 1.0):
            errors.append(f"nu must be in [0.0, 1.0], got {self.nu}")
        if self.H <= 0:
            errors.append(f"H must be positive, got {self.H}")

        # Economic bounds
        if self.target_balance <= 0:
            errors.append(f"target_balance must be positive, got {self.target_balance}")
        if self.min_fee <= 0:
            errors.append(f"min_fee must be positive, got {self.min_fee}")

        # Gas and batching
        if self.gas_per_tx <= 0:
            errors.append(f"gas_per_tx must be positive, got {self.gas_per_tx}")
        if self.batch_interval_steps <= 0:
            errors.append(f"batch_interval_steps must be positive, got {self.batch_interval_steps}")

        # Transaction modeling
        if self.base_tx_demand <= 0:
            errors.append(f"base_tx_demand must be positive, got {self.base_tx_demand}")
        if self.fee_elasticity <= 0:
            errors.append(f"fee_elasticity must be positive, got {self.fee_elasticity}")
        if self.max_tx_demand <= 0:
            errors.append(f"max_tx_demand must be positive, got {self.max_tx_demand}")

        # Smoothing bounds
        if not (0.0 <= self.l1_smoothing_factor <= 1.0):
            errors.append(f"l1_smoothing_factor must be in [0.0, 1.0], got {self.l1_smoothing_factor}")
        if self.max_fee_change_ratio <= 1.0:
            errors.append(f"max_fee_change_ratio must be > 1.0, got {self.max_fee_change_ratio}")
        if self.outlier_threshold_sigma <= 0:
            errors.append(f"outlier_threshold_sigma must be positive, got {self.outlier_threshold_sigma}")

        # Guaranteed recovery
        if self.min_deficit_rate <= 0:
            errors.append(f"min_deficit_rate must be positive, got {self.min_deficit_rate}")

        if errors:
            raise ValueError("Parameter validation failed:\\n" + "\\n".join(errors))


@dataclass
class VaultState:
    """Canonical vault state representation."""

    balance: float                      # Current vault balance (ETH)
    target_balance: float               # Target vault balance (ETH)

    # History for analytics
    balance_history: List[float] = field(default_factory=list)
    fees_collected_history: List[float] = field(default_factory=list)
    l1_costs_paid_history: List[float] = field(default_factory=list)

    @property
    def deficit(self) -> float:
        """Current vault deficit (positive when underfunded)."""
        return max(0.0, self.target_balance - self.balance)

    @property
    def surplus(self) -> float:
        """Current vault surplus (positive when overfunded)."""
        return max(0.0, self.balance - self.target_balance)

    @property
    def deficit_ratio(self) -> float:
        """Deficit as fraction of target balance."""
        return self.deficit / self.target_balance if self.target_balance > 0 else 0.0

    def collect_fees(self, amount: float) -> None:
        """Collect fees into the vault."""
        if amount < 0:
            raise ValueError(f"Fee amount cannot be negative: {amount}")
        self.balance += amount
        self.fees_collected_history.append(amount)
        self.balance_history.append(self.balance)

    def pay_l1_costs(self, amount: float) -> None:
        """Pay L1 batch costs from the vault."""
        if amount < 0:
            raise ValueError(f"L1 cost amount cannot be negative: {amount}")
        self.balance -= amount
        self.l1_costs_paid_history.append(amount)
        self.balance_history.append(self.balance)


class CanonicalTaikoFeeCalculator:
    """
    SINGLE SOURCE OF TRUTH for Taiko fee mechanism calculations.

    This class implements the authoritative fee calculation logic that must be
    used by all components in the system to ensure consistency.
    """

    def __init__(self, params: FeeParameters):
        """Initialize calculator with validated parameters."""
        self.params = params

        # L1 cost smoothing state
        self._smoothed_l1_cost: Optional[float] = None
        self._l1_cost_history: List[float] = []

        # Fee change limiting
        self._previous_fee: Optional[float] = None

        # Validation
        self._validate_initialization()

    def _validate_initialization(self):
        """Validate calculator state after initialization."""
        if not isinstance(self.params, FeeParameters):
            raise TypeError(f"params must be FeeParameters, got {type(self.params)}")

    def calculate_l1_cost_per_tx(self, l1_basefee_wei: float, apply_smoothing: bool = True) -> float:
        """
        Calculate L1 cost per L2 transaction with optional smoothing.

        Args:
            l1_basefee_wei: L1 basefee in wei
            apply_smoothing: Whether to apply EMA smoothing for stability

        Returns:
            L1 cost per transaction in ETH

        Formula:
            C_L1 = (L1_basefee × gas_per_tx) / 10^18
        """
        if l1_basefee_wei < 0:
            raise ValueError(f"L1 basefee cannot be negative: {l1_basefee_wei}")

        # Mock data detection for gas_per_tx parameter
        validate_real_data_usage("parameter", self.params.gas_per_tx, "gas_per_tx configuration")

        # Unit safety validation if available
        if UNIT_SAFETY_AVAILABLE:
            try:
                # Validate L1 basefee is reasonable
                basefee_gwei = l1_basefee_wei / 1e9
                validate_basefee_range(basefee_gwei, "calculate_l1_cost_per_tx")

                # Validate gas amount is reasonable
                if self.params.gas_per_tx < 1000 or self.params.gas_per_tx > 1000000:
                    warnings.warn(
                        f"Unusual gas_per_tx: {self.params.gas_per_tx} "
                        f"(typical range: 20,000-200,000)"
                    )
            except (UnitValidationError, UnitOverflowError) as e:
                warnings.warn(f"Unit validation warning in L1 cost calculation: {e}")

        # Calculate raw L1 cost
        raw_cost = (l1_basefee_wei * self.params.gas_per_tx) / 1e18

        # Validate result is reasonable if unit safety available
        if UNIT_SAFETY_AVAILABLE and raw_cost > 0:
            if raw_cost < 1e-8 or raw_cost > 0.1:
                warnings.warn(
                    f"Unusual L1 cost result: {raw_cost:.8f} ETH "
                    f"(typical range: 0.00001-0.01 ETH) - check unit conversions"
                )

        # Apply smoothing if enabled
        if apply_smoothing:
            return self._apply_l1_cost_smoothing(raw_cost)
        else:
            return raw_cost

    def _apply_l1_cost_smoothing(self, raw_cost: float) -> float:
        """Apply EMA smoothing to L1 cost with outlier rejection."""
        # Store raw cost in history
        self._l1_cost_history.append(raw_cost)

        # Initialize smoothed cost on first call
        if self._smoothed_l1_cost is None:
            self._smoothed_l1_cost = raw_cost
            return raw_cost

        # Outlier detection using recent history
        if len(self._l1_cost_history) >= 10:
            recent_costs = self._l1_cost_history[-10:]
            mean_cost = np.mean(recent_costs)
            std_cost = np.std(recent_costs)

            # Reject outliers beyond threshold
            z_score = abs(raw_cost - mean_cost) / (std_cost + 1e-8)  # Prevent division by zero
            if z_score > self.params.outlier_threshold_sigma:
                # Use previous smoothed value for outliers
                raw_cost = self._smoothed_l1_cost

        # Apply EMA smoothing
        alpha = self.params.l1_smoothing_factor
        self._smoothed_l1_cost = alpha * raw_cost + (1 - alpha) * self._smoothed_l1_cost

        return self._smoothed_l1_cost

    def calculate_estimated_fee(self, l1_cost_per_tx: float, vault_deficit: float) -> float:
        """
        CANONICAL fee calculation - the authoritative implementation.

        Args:
            l1_cost_per_tx: L1 cost per transaction (ETH)
            vault_deficit: Current vault deficit (ETH, positive when underfunded)

        Returns:
            Estimated fee per transaction (ETH)

        Formula:
            F_estimated = max(μ × C_L1 + ν × D/H, F_min)
        """
        if l1_cost_per_tx < 0:
            raise ValueError(f"L1 cost cannot be negative: {l1_cost_per_tx}")
        if vault_deficit < 0:
            raise ValueError(f"Vault deficit cannot be negative: {vault_deficit}")

        # Mock data detection for fee mechanism parameters
        validate_real_data_usage("parameter", self.params.mu, "mu (L1 weight) parameter")
        validate_real_data_usage("parameter", self.params.nu, "nu (deficit weight) parameter")
        validate_real_data_usage("parameter", self.params.H, "H (prediction horizon) parameter")
        validate_real_data_usage("parameter", self.params.min_fee, "minimum fee floor")

        # Check for hardcoded minimum fee injection
        if self.params.min_fee > 1e-8:
            # Default min_fee is 1e-8 ETH, anything larger might be artificial
            fee_gwei_equiv = self.params.min_fee * 1e9
            if abs(fee_gwei_equiv - 1.5) < 0.1:  # Close to 1.5 gwei
                warn_mock_data_usage("fee calculation", "artificial minimum fee",
                                   f"min_fee={fee_gwei_equiv:.1f} gwei appears to be hardcoded injection")

        # Unit safety validation if available
        if UNIT_SAFETY_AVAILABLE:
            # Validate input values are reasonable for ETH units
            if l1_cost_per_tx > 0.1:  # L1 cost > 0.1 ETH is suspicious
                warnings.warn(
                    f"Very large L1 cost: {l1_cost_per_tx:.6f} ETH - verify units are correct"
                )
            if vault_deficit > 10000:  # Deficit > 10,000 ETH is suspicious
                warnings.warn(
                    f"Very large vault deficit: {vault_deficit:.2f} ETH - verify units are correct"
                )

        # L1 cost component
        l1_component = self.params.mu * l1_cost_per_tx

        # Deficit correction component
        deficit_component = self.params.nu * vault_deficit / self.params.H

        # Apply guaranteed recovery enhancement if enabled
        if self.params.guaranteed_recovery and vault_deficit > 0:
            # Warn about guaranteed recovery as it might be artificial
            warn_mock_data_usage("fee calculation", "guaranteed recovery mechanism",
                               "artificial minimum deficit correction rate applied")

            # Ensure minimum deficit correction rate
            standard_correction = deficit_component
            minimum_correction = self.params.min_deficit_rate
            deficit_component = max(standard_correction, minimum_correction)

        # Calculate raw fee
        raw_fee = l1_component + deficit_component

        # Apply minimum fee floor
        fee = max(raw_fee, self.params.min_fee)

        # Check if minimum fee floor was applied (indicates potential artificial inflation)
        if fee == self.params.min_fee and raw_fee < self.params.min_fee:
            fee_gwei = fee * 1e9
            if fee_gwei > 0.1:  # Minimum fee > 0.1 gwei might be artificial
                warn_mock_data_usage("fee calculation", "minimum fee floor applied",
                                   f"fee artificially increased from {raw_fee*1e9:.6f} to {fee_gwei:.6f} gwei")

        # Apply fee change limiting to prevent sudden jumps
        fee = self._apply_fee_change_limits(fee)

        # Final unit safety validation
        if UNIT_SAFETY_AVAILABLE and fee > 0:
            try:
                fee_gwei = fee * 1e9
                assert_reasonable_fee(fee_gwei, "calculate_estimated_fee result")
            except (UnitValidationError, UnitOverflowError) as e:
                # Don't fail calculation, but warn about suspicious result
                warnings.warn(f"Fee result validation warning: {e}")

        self._previous_fee = fee
        return fee

    def _apply_fee_change_limits(self, new_fee: float) -> float:
        """Limit fee changes to prevent sudden jumps that could harm UX."""
        if self._previous_fee is None:
            return new_fee

        # Calculate maximum allowed change
        max_ratio = self.params.max_fee_change_ratio
        max_fee = self._previous_fee * max_ratio
        min_fee = self._previous_fee / max_ratio

        # Clamp to allowed range
        return max(min_fee, min(new_fee, max_fee))

    def calculate_transaction_volume(self, estimated_fee: float) -> float:
        """
        Calculate transaction volume using demand elasticity model.

        Args:
            estimated_fee: Estimated fee per transaction (ETH)

        Returns:
            Transaction volume for this time step

        Model:
            volume = base_demand × (fee / reference_fee)^(-elasticity)
        """
        if estimated_fee <= 0:
            raise ValueError(f"Estimated fee must be positive: {estimated_fee}")

        # Use minimum fee as reference point
        reference_fee = self.params.min_fee
        fee_ratio = estimated_fee / reference_fee

        # Apply demand elasticity
        demand_multiplier = fee_ratio ** (-self.params.fee_elasticity)
        volume = self.params.base_tx_demand * demand_multiplier

        # Apply volume cap
        return min(volume, self.params.max_tx_demand)

    def calculate_l1_batch_cost(self, l1_basefee_wei: float) -> float:
        """
        Calculate cost of submitting an L1 batch.

        Args:
            l1_basefee_wei: L1 basefee in wei

        Returns:
            L1 batch submission cost (ETH)
        """
        if l1_basefee_wei < 0:
            raise ValueError(f"L1 basefee cannot be negative: {l1_basefee_wei}")

        # Unit safety validation if available
        if UNIT_SAFETY_AVAILABLE:
            try:
                # Validate L1 basefee is reasonable
                basefee_gwei = l1_basefee_wei / 1e9
                validate_basefee_range(basefee_gwei, "calculate_l1_batch_cost")
            except (UnitValidationError, UnitOverflowError) as e:
                warnings.warn(f"Unit validation warning in L1 batch cost calculation: {e}")

        # Batch cost is based on total gas for batch submission
        # This is separate from per-transaction cost calculation
        gas_per_batch = self.params.gas_per_tx * self.params.base_tx_demand
        batch_cost = (l1_basefee_wei * gas_per_batch) / 1e18

        # Validate result is reasonable
        if UNIT_SAFETY_AVAILABLE and batch_cost > 0:
            if batch_cost > 10.0:  # Batch cost > 10 ETH is very suspicious
                warnings.warn(
                    f"Very large batch cost: {batch_cost:.6f} ETH - verify unit conversions"
                )

        return batch_cost

    def create_vault(self, init_mode: VaultInitMode,
                    deficit_ratio: float = 0.0,
                    custom_balance: Optional[float] = None) -> VaultState:
        """
        Create vault with specified initialization mode.

        Args:
            init_mode: Vault initialization strategy
            deficit_ratio: Deficit as fraction of target (for DEFICIT/SURPLUS modes)
            custom_balance: Custom balance (for CUSTOM mode)

        Returns:
            Initialized VaultState
        """
        target = self.params.target_balance

        if init_mode == VaultInitMode.TARGET:
            balance = target
        elif init_mode == VaultInitMode.DEFICIT:
            balance = target * (1.0 - deficit_ratio)
        elif init_mode == VaultInitMode.SURPLUS:
            balance = target * (1.0 + deficit_ratio)
        elif init_mode == VaultInitMode.CUSTOM:
            if custom_balance is None:
                raise ValueError("custom_balance must be provided for CUSTOM init mode")
            balance = custom_balance
        else:
            raise ValueError(f"Unknown vault init mode: {init_mode}")

        return VaultState(balance=balance, target_balance=target)

    def reset_state(self) -> None:
        """Reset calculator state for new simulation."""
        self._smoothed_l1_cost = None
        self._l1_cost_history.clear()
        self._previous_fee = None


# Convenience functions for common operations
def create_default_calculator() -> CanonicalTaikoFeeCalculator:
    """Create calculator with default optimal parameters."""
    params = FeeParameters(
        mu=0.0,       # Optimal consensus value
        nu=0.27,      # Optimal consensus value
        H=492         # Optimal consensus value
    )
    return CanonicalTaikoFeeCalculator(params)


def create_balanced_calculator() -> CanonicalTaikoFeeCalculator:
    """Create calculator with balanced parameters."""
    params = FeeParameters(
        mu=0.0,       # Consensus optimal
        nu=0.27,      # Consensus optimal
        H=492         # Consensus optimal
    )
    return CanonicalTaikoFeeCalculator(params)


def create_crisis_calculator() -> CanonicalTaikoFeeCalculator:
    """Create calculator with crisis-resilient parameters."""
    params = FeeParameters(
        mu=0.0,       # Consensus optimal
        nu=0.88,      # Crisis-resilient value
        H=120         # Crisis-resilient value
    )
    return CanonicalTaikoFeeCalculator(params)


# Parameter validation utilities
def validate_fee_parameters(mu: float, nu: float, H: int) -> bool:
    """Quick validation of core fee mechanism parameters."""
    try:
        FeeParameters(mu=mu, nu=nu, H=H)
        return True
    except ValueError:
        return False


def get_optimal_parameters() -> Dict[str, Union[float, int]]:
    """Get the research-validated optimal parameter set."""
    return {
        "mu": 0.0,
        "nu": 0.27,
        "H": 492,
        "description": "Consensus optimal parameters from multi-scenario research"
    }