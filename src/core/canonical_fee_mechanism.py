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
    F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t)

Where:
    - μ: DA cost pass-through coefficient [0.0, 1.0]
    - ν: Vault-healing intensity coefficient [0.0, 1.0]
    - C_DA(t) = α_data × B̂_L1(t): smoothed marginal DA cost per L2 gas
    - C_vault(t) = D(t)/(H × Q̄): full-strength vault-healing surcharge per L2 gas
    - α_data: Expected L1 DA gas per 1 L2 gas
    - B̂_L1(t): Smoothed L1 basefee (ETH per L1 gas)
    - D(t): Vault deficit (ETH, positive when underfunded)
    - H: Recovery horizon (batches)
    - Q̄: Typical L2 gas per batch (governance constant)
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

    # Core mechanism parameters (2024 NSGA-II optimized)
    mu: float = 0.0                    # L1 weight [0.0, 1.0] - DA cost pass-through coefficient
    nu: float = 0.369                  # Deficit weight [0.0, 1.0] - enhanced vault-healing intensity coefficient
    H: int = 1794                      # Prediction horizon (time steps) - ~1 hour recovery horizon in batches

    # New Fee Mechanism Parameters (per specification, 2024 optimized)
    alpha_data: float = 0.5            # DA gas ratio: expected L1 DA gas per 1 L2 gas (FIXED: was 20000x too high)
    lambda_B: float = 0.365            # EMA smoothing factor for L1 basefee [0.0, 1.0] - enhanced stability
    Q_bar: float = 690000.0            # Average L2 gas per batch (governance constant)
    T: float = 1000.0                  # Target vault balance (ETH)

    # Economic parameters (legacy - kept for compatibility)
    target_balance: float = 1000.0     # Legacy target vault balance (ETH) - use T instead
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

        # New fee mechanism parameter bounds
        if self.alpha_data <= 0:
            errors.append(f"alpha_data must be positive, got {self.alpha_data}")
        if not (0.0 < self.lambda_B <= 1.0):
            errors.append(f"lambda_B must be in (0.0, 1.0], got {self.lambda_B}")
        if self.Q_bar <= 0:
            errors.append(f"Q_bar must be positive, got {self.Q_bar}")
        if self.T <= 0:
            errors.append(f"T must be positive, got {self.T}")

        # Economic bounds
        if self.target_balance <= 0:
            errors.append(f"target_balance must be positive, got {self.target_balance}")
        if self.min_fee <= 0:
            errors.append(f"min_fee must be positive, got {self.min_fee}")

        # Consistency checks
        if abs(self.T - self.target_balance) > 1e-6:
            warnings.warn(
                f"T ({self.T}) and target_balance ({self.target_balance}) differ. "
                f"Using T for new fee mechanism calculations."
            )

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

        # L1 basefee EMA smoothing state (new specification)
        self._smoothed_l1_basefee: Optional[float] = None  # B̂_L1(t) in ETH per L1 gas
        self._l1_basefee_history: List[float] = []

        # Legacy L1 cost smoothing state (kept for compatibility)
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

    def calculate_smoothed_l1_basefee(self, l1_basefee_wei: float) -> float:
        """
        Calculate smoothed L1 basefee using EMA (new specification).

        Args:
            l1_basefee_wei: Raw L1 basefee in wei

        Returns:
            Smoothed L1 basefee in ETH per L1 gas

        Formula:
            B̂_L1(t) = (1 - λ_B) × B̂_L1(t-1) + λ_B × B_L1(t)
        """
        if l1_basefee_wei < 0:
            raise ValueError(f"L1 basefee cannot be negative: {l1_basefee_wei}")

        # Convert to ETH per L1 gas
        l1_basefee_eth = l1_basefee_wei / 1e18

        # Mock data validation
        validate_real_data_usage("parameter", self.params.lambda_B, "lambda_B EMA smoothing factor")

        # Unit safety validation if available
        if UNIT_SAFETY_AVAILABLE:
            try:
                basefee_gwei = l1_basefee_wei / 1e9
                validate_basefee_range(basefee_gwei, "calculate_smoothed_l1_basefee")
            except (UnitValidationError, UnitOverflowError) as e:
                warnings.warn(f"Unit validation warning in L1 basefee smoothing: {e}")

        # Store in history
        self._l1_basefee_history.append(l1_basefee_eth)

        # Initialize on first call
        if self._smoothed_l1_basefee is None:
            self._smoothed_l1_basefee = l1_basefee_eth
            return self._smoothed_l1_basefee

        # Apply EMA smoothing: B̂_L1(t) = (1 - λ_B) × B̂_L1(t-1) + λ_B × B_L1(t)
        lambda_B = self.params.lambda_B
        self._smoothed_l1_basefee = (
            (1.0 - lambda_B) * self._smoothed_l1_basefee +
            lambda_B * l1_basefee_eth
        )

        return self._smoothed_l1_basefee

    def calculate_C_DA(self, l1_basefee_wei: float) -> float:
        """
        Calculate DA cost term C_DA(t) (new specification).

        Args:
            l1_basefee_wei: Raw L1 basefee in wei

        Returns:
            C_DA(t) = α_data × B̂_L1(t): smoothed marginal DA cost per L2 gas (ETH)

        Formula:
            C_DA(t) = α_data × B̂_L1(t)
        """
        # Get smoothed L1 basefee
        smoothed_basefee = self.calculate_smoothed_l1_basefee(l1_basefee_wei)

        # Mock data validation for alpha_data
        validate_real_data_usage("parameter", self.params.alpha_data, "alpha_data DA gas ratio")

        # Calculate DA cost: C_DA(t) = α_data × B̂_L1(t)
        C_DA = self.params.alpha_data * smoothed_basefee

        # Validate result
        if UNIT_SAFETY_AVAILABLE and C_DA > 0:
            if C_DA < 1e-12 or C_DA > 0.1:
                warnings.warn(
                    f"Unusual C_DA result: {C_DA:.12f} ETH per L2 gas "
                    f"(check alpha_data={self.params.alpha_data} and basefee units)"
                )

        return C_DA

    def calculate_C_vault(self, vault_deficit: float) -> float:
        """
        Calculate vault healing term C_vault(t) (new specification).

        Args:
            vault_deficit: Current vault deficit in ETH (positive when underfunded)

        Returns:
            C_vault(t) = D(t)/(H × Q̄): full-strength vault-healing surcharge per L2 gas (ETH)

        Formula:
            C_vault(t) = D(t) / (H × Q̄)
        """
        if vault_deficit < 0:
            raise ValueError(f"Vault deficit cannot be negative: {vault_deficit}")

        # Mock data validation
        validate_real_data_usage("parameter", self.params.Q_bar, "Q_bar average L2 gas per batch")
        validate_real_data_usage("parameter", self.params.H, "H recovery horizon")

        # Check for suspicious Q_bar values
        if abs(self.params.Q_bar - 690000) < 1000:
            warn_mock_data_usage("vault healing calculation", "Q_bar ≈ 690,000",
                               "should use empirically measured average gas per batch")

        # Calculate vault healing: C_vault(t) = D(t) / (H × Q̄)
        denominator = self.params.H * self.params.Q_bar
        C_vault = vault_deficit / denominator

        return C_vault

    def _apply_l1_cost_smoothing(self, raw_cost: float) -> float:
        """Legacy L1 cost smoothing - kept for backward compatibility."""
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

    def calculate_estimated_fee_raw(self, l1_basefee_wei: float, vault_deficit: float) -> float:
        """
        CANONICAL raw fee calculation - NEW SPECIFICATION implementation.

        Args:
            l1_basefee_wei: L1 basefee in wei
            vault_deficit: Current vault deficit (ETH, positive when underfunded)

        Returns:
            F_L2_raw(t): Raw estimated fee per L2 gas (ETH)

        Formula:
            F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t)
        """
        if l1_basefee_wei < 0:
            raise ValueError(f"L1 basefee cannot be negative: {l1_basefee_wei}")
        if vault_deficit < 0:
            raise ValueError(f"Vault deficit cannot be negative: {vault_deficit}")

        # Mock data detection for fee mechanism parameters
        validate_real_data_usage("parameter", self.params.mu, "mu (DA cost pass-through) parameter")
        validate_real_data_usage("parameter", self.params.nu, "nu (vault healing intensity) parameter")

        # Unit safety validation if available
        if UNIT_SAFETY_AVAILABLE:
            if vault_deficit > 10000:  # Deficit > 10,000 ETH is suspicious
                warnings.warn(
                    f"Very large vault deficit: {vault_deficit:.2f} ETH - verify units are correct"
                )

        # Calculate DA cost component: μ × C_DA(t)
        C_DA = self.calculate_C_DA(l1_basefee_wei)
        da_component = self.params.mu * C_DA

        # Calculate vault healing component: ν × C_vault(t)
        C_vault = self.calculate_C_vault(vault_deficit)
        vault_component = self.params.nu * C_vault

        # Calculate raw fee: F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t)
        raw_fee = da_component + vault_component

        # Final unit safety validation
        if UNIT_SAFETY_AVAILABLE and raw_fee > 0:
            try:
                fee_gwei = raw_fee * 1e9
                assert_reasonable_fee(fee_gwei, "calculate_estimated_fee_raw result")
            except (UnitValidationError, UnitOverflowError) as e:
                # Don't fail calculation, but warn about suspicious result
                warnings.warn(f"Fee result validation warning: {e}")

        return raw_fee

    def calculate_estimated_fee(self, l1_cost_per_tx: float, vault_deficit: float) -> float:
        """
        Legacy fee calculation - kept for backward compatibility.

        NOTE: This method is deprecated. Use calculate_estimated_fee_raw() with L1 basefee instead.

        Args:
            l1_cost_per_tx: L1 cost per transaction (ETH)
            vault_deficit: Current vault deficit (ETH, positive when underfunded)

        Returns:
            Estimated fee per transaction (ETH)

        Formula:
            F_estimated = max(μ × C_L1 + ν × D/H, F_min)
        """
        warnings.warn(
            "calculate_estimated_fee() with L1 cost is deprecated. "
            "Use calculate_estimated_fee_raw() with L1 basefee for new specification.",
            DeprecationWarning,
            stacklevel=2
        )

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
        # Use new T parameter for target balance
        target = self.params.T

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
        # Reset new specification state
        self._smoothed_l1_basefee = None
        self._l1_basefee_history.clear()

        # Reset legacy state
        self._smoothed_l1_cost = None
        self._l1_cost_history.clear()

        # Reset fee limiting
        self._previous_fee = None


# Convenience functions for common operations
def create_default_calculator() -> CanonicalTaikoFeeCalculator:
    """Create calculator with default optimal parameters (new specification)."""
    params = FeeParameters(
        # Core mechanism (2024 NSGA-II optimized - balanced strategy)
        mu=0.0,           # Confirmed optimal - pure deficit-based correction
        nu=0.369,         # Updated optimal - enhanced vault healing
        H=1794,           # Updated optimal - ~1 hour horizon for better stability

        # New specification parameters (2024 optimized)
        alpha_data=0.5,      # Realistic L1 DA gas per L2 gas (FIXED: was 20000x too high)
        lambda_B=0.365,      # Enhanced smoothing for L1 basefee stability
        Q_bar=690000.0,      # Average L2 gas per batch (Taiko Alethia estimate)
        T=1000.0             # Target vault balance
    )
    return CanonicalTaikoFeeCalculator(params)


def create_balanced_calculator() -> CanonicalTaikoFeeCalculator:
    """Create calculator with balanced parameters."""
    params = FeeParameters(
        mu=0.0,       # Consensus optimal
        nu=0.48,      # Conservative approach (higher than optimal for safety)
        H=1794,       # Updated to match optimized horizon

        # Standard new specification parameters
        alpha_data=0.5,       # Realistic L1 DA gas per L2 gas ratio
        lambda_B=0.2,        # Slightly faster smoothing for balanced approach
        Q_bar=690000.0,
        T=1000.0
    )
    return CanonicalTaikoFeeCalculator(params)


def create_crisis_calculator() -> CanonicalTaikoFeeCalculator:
    """Create calculator with crisis-resilient parameters."""
    params = FeeParameters(
        mu=0.0,       # Consensus optimal
        nu=0.88,      # Aggressive deficit recovery
        H=120,        # Shorter horizon for faster response

        # Crisis-tuned parameters
        alpha_data=0.5,       # Realistic L1 DA gas per L2 gas ratio
        lambda_B=0.5,        # Very fast L1 response for crisis
        Q_bar=690000.0,
        T=1000.0
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
        "nu": 0.369,
        "H": 1794,
        "lambda_B": 0.365,
        "description": "2024 NSGA-II optimized parameters from balanced multi-objective optimization"
    }