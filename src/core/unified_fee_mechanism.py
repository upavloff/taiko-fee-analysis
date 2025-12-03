"""
Unified Taiko Fee Mechanism Implementation

This is the SINGLE SOURCE OF TRUTH implementation of the L2 Sustainability Basefee
as defined in AUTHORITATIVE_SPECIFICATION.md.

All other implementations (JavaScript, legacy modules) must be consistent with this module.
"""

import warnings
import numpy as np
from typing import Dict, Union, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ParameterCalibrationStatus(Enum):
    """Parameter calibration status levels."""
    CALIBRATED = "calibrated"          # Based on real Taiko data
    THEORETICAL = "theoretical"        # Based on theoretical estimates
    UNCALIBRATED = "uncalibrated"     # No real data, using placeholder


@dataclass
class FeeParameters:
    """Unified parameter set for fee mechanism configuration."""

    # Core mechanism parameters
    mu: float = 0.0                    # DA cost pass-through coefficient [0.0, 1.0]
    nu: float = 0.5                    # Vault healing intensity coefficient [0.0, 1.0]
    H: int = 144                       # Recovery horizon (batches under typical load)
    lambda_B: float = 0.3              # EMA smoothing factor for L1 basefee [0.0, 1.0]

    # System constants (REQUIRE REAL DATA CALIBRATION)
    alpha_data: float = 0.022           # Expected L1 DA gas per 1 L2 gas [THEORETICAL ESTIMATE]
    Q_bar: float = 150000.0           # Typical L2 gas per batch [CONSERVATIVE ESTIMATE]
    T: float = 1000.0                  # Target vault balance (ETH)

    # UX wrapper parameters
    F_min: float = 1e-12              # Minimum sustainability basefee (ETH per L2 gas)
    F_max: float = 1e-6               # Maximum sustainability basefee (ETH per L2 gas)
    kappa_up: float = 0.1             # Max relative fee increase per batch [0.0, 1.0]
    kappa_down: float = 0.1           # Max relative fee decrease per batch [0.0, 1.0]

    # Parameter calibration status tracking
    alpha_data_status: ParameterCalibrationStatus = ParameterCalibrationStatus.THEORETICAL
    Q_bar_status: ParameterCalibrationStatus = ParameterCalibrationStatus.THEORETICAL
    optimization_status: ParameterCalibrationStatus = ParameterCalibrationStatus.UNCALIBRATED

    def __post_init__(self):
        """Validate parameters and emit warnings for uncalibrated values."""
        self._validate_ranges()
        self._emit_calibration_warnings()

    def _validate_ranges(self):
        """Validate parameter ranges."""
        if not (0.0 <= self.mu <= 1.0):
            raise ValueError(f"mu must be in [0.0, 1.0], got {self.mu}")
        if not (0.0 <= self.nu <= 1.0):
            raise ValueError(f"nu must be in [0.0, 1.0], got {self.nu}")
        if not (0.0 < self.lambda_B <= 1.0):
            raise ValueError(f"lambda_B must be in (0.0, 1.0], got {self.lambda_B}")
        if self.H <= 0:
            raise ValueError(f"H must be positive, got {self.H}")
        if self.alpha_data <= 0:
            raise ValueError(f"alpha_data must be positive, got {self.alpha_data}")
        if self.Q_bar <= 0:
            raise ValueError(f"Q_bar must be positive, got {self.Q_bar}")
        if not (0.0 <= self.kappa_up <= 1.0):
            raise ValueError(f"kappa_up must be in [0.0, 1.0], got {self.kappa_up}")
        if not (0.0 <= self.kappa_down <= 1.0):
            raise ValueError(f"kappa_down must be in [0.0, 1.0], got {self.kappa_down}")

    def _emit_calibration_warnings(self):
        """Emit warnings for uncalibrated parameters."""
        if self.alpha_data_status != ParameterCalibrationStatus.CALIBRATED:
            warnings.warn(
                f"🚨 PARAMETER WARNING: α_data = {self.alpha_data} is {self.alpha_data_status.value.upper()}. "
                f"This parameter requires calibration from real Taiko proposeBlock transaction data. "
                f"Current estimate may be off by orders of magnitude.",
                UserWarning,
                stacklevel=2
            )

        if self.Q_bar_status != ParameterCalibrationStatus.CALIBRATED:
            warnings.warn(
                f"🚨 PARAMETER WARNING: Q̄ = {self.Q_bar} is {self.Q_bar_status.value.upper()}. "
                f"This parameter requires measurement from real Taiko L2 batch sizes.",
                UserWarning,
                stacklevel=2
            )

        if self.optimization_status != ParameterCalibrationStatus.CALIBRATED:
            warnings.warn(
                f"🚨 OPTIMIZATION WARNING: Parameters (μ={self.mu}, ν={self.nu}, H={self.H}) are {self.optimization_status.value.upper()}. "
                f"Optimal values require re-optimization with calibrated α_data and Q̄.",
                UserWarning,
                stacklevel=2
            )


@dataclass
class VaultState:
    """Vault state tracking."""
    balance: float = 1000.0            # Current vault balance (ETH)
    target: float = 1000.0             # Target vault balance (ETH)

    @property
    def deficit(self) -> float:
        """Calculate vault deficit (positive when underfunded)."""
        return max(0.0, self.target - self.balance)

    @property
    def surplus(self) -> float:
        """Calculate vault surplus (positive when overfunded)."""
        return max(0.0, self.balance - self.target)


class UnifiedFeeCalculator:
    """
    Unified Fee Calculator implementing the authoritative L2 Sustainability Basefee specification.

    This is the canonical implementation of:
    F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t)

    With UX wrapper providing clipping and rate limiting.
    """

    def __init__(self, params: Optional[FeeParameters] = None):
        self.params = params or FeeParameters()
        self._smoothed_l1_basefee: Optional[float] = None
        self._last_final_fee: Optional[float] = None

        # Print parameter status on initialization
        self._print_initialization_status()

    def _print_initialization_status(self):
        """Print parameter calibration status."""
        print("🏗️ UNIFIED FEE CALCULATOR INITIALIZED")
        print(f"   Formula: F_L2_raw(t) = μ×C_DA(t) + ν×C_vault(t)")
        print(f"   Parameters: μ={self.params.mu}, ν={self.params.nu}, H={self.params.H}")
        print(f"   Constants: α_data={self.params.alpha_data} ({self.params.alpha_data_status.value})")
        print(f"             Q̄={self.params.Q_bar:,.0f} ({self.params.Q_bar_status.value})")
        if (self.params.alpha_data_status != ParameterCalibrationStatus.CALIBRATED or
            self.params.Q_bar_status != ParameterCalibrationStatus.CALIBRATED):
            print(f"   ⚠️  UNCALIBRATED parameters detected - see warnings above")
        print()

    def update_smoothed_l1_basefee(self, l1_basefee_wei: float) -> float:
        """Update smoothed L1 basefee using EMA."""
        l1_basefee_eth_per_gas = l1_basefee_wei / 1e18

        if self._smoothed_l1_basefee is None:
            # Initialize with first observation
            self._smoothed_l1_basefee = l1_basefee_eth_per_gas
        else:
            # EMA update: B̂_L1(t) = (1 - λ_B) × B̂_L1(t-1) + λ_B × B_L1(t)
            self._smoothed_l1_basefee = (
                (1 - self.params.lambda_B) * self._smoothed_l1_basefee +
                self.params.lambda_B * l1_basefee_eth_per_gas
            )

        return self._smoothed_l1_basefee

    def calculate_C_DA(self, l1_basefee_wei: float) -> float:
        """
        Calculate smoothed marginal DA cost per L2 gas.
        C_DA(t) = α_data × B̂_L1(t)
        """
        smoothed_basefee = self.update_smoothed_l1_basefee(l1_basefee_wei)
        return self.params.alpha_data * smoothed_basefee

    def calculate_C_vault(self, vault_deficit: float) -> float:
        """
        Calculate full-strength vault healing surcharge per L2 gas.
        C_vault(t) = D(t) / (H × Q̄)
        """
        return vault_deficit / (self.params.H * self.params.Q_bar)

    def calculate_raw_fee(self, l1_basefee_wei: float, vault_deficit: float) -> float:
        """
        Calculate raw L2 sustainability basefee.
        F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t)
        """
        C_DA = self.calculate_C_DA(l1_basefee_wei)
        C_vault = self.calculate_C_vault(vault_deficit)

        return self.params.mu * C_DA + self.params.nu * C_vault

    def apply_clipping(self, raw_fee: float) -> float:
        """
        Apply global clipping to raw fee.
        F_clip(t) = min(max(F_L2_raw(t), F_min), F_max)
        """
        return min(max(raw_fee, self.params.F_min), self.params.F_max)

    def apply_rate_limiting(self, clipped_fee: float) -> float:
        """
        Apply rate limiting to clipped fee.
        F_L2(t) = min(F_L2(t-1)×(1+κ_↑), max(F_L2(t-1)×(1-κ_↓), F_clip(t)))
        """
        if self._last_final_fee is None:
            # Initialize with first fee
            self._last_final_fee = clipped_fee
            return clipped_fee

        # Calculate rate-limited bounds
        max_increase = self._last_final_fee * (1 + self.params.kappa_up)
        max_decrease = self._last_final_fee * (1 - self.params.kappa_down)

        # Apply rate limiting
        final_fee = min(max_increase, max(max_decrease, clipped_fee))
        self._last_final_fee = final_fee

        return final_fee

    def calculate_final_fee(self, l1_basefee_wei: float, vault_deficit: float) -> Dict[str, float]:
        """
        Calculate final user-facing L2 sustainability basefee with full UX wrapper.

        Returns:
            Dict with fee calculation breakdown
        """
        # Step 1: Calculate raw fee
        raw_fee = self.calculate_raw_fee(l1_basefee_wei, vault_deficit)

        # Step 2: Apply clipping
        clipped_fee = self.apply_clipping(raw_fee)

        # Step 3: Apply rate limiting
        final_fee = self.apply_rate_limiting(clipped_fee)

        return {
            'raw_fee_eth_per_gas': raw_fee,
            'clipped_fee_eth_per_gas': clipped_fee,
            'final_fee_eth_per_gas': final_fee,
            'final_fee_gwei_per_gas': final_fee * 1e9,
            'C_DA': self.calculate_C_DA(l1_basefee_wei),
            'C_vault': self.calculate_C_vault(vault_deficit),
            'smoothed_l1_basefee': self._smoothed_l1_basefee
        }


def create_conservative_calculator() -> UnifiedFeeCalculator:
    """Create calculator with conservative parameter estimates."""
    params = FeeParameters(
        # Conservative mechanism parameters
        mu=0.0,                        # Pure deficit correction (no L1 pass-through)
        nu=0.3,                        # Moderate vault healing
        H=144,                         # ~4.8 minute horizon
        lambda_B=0.2,                  # Conservative smoothing

        # Conservative estimates (marked as theoretical)
        alpha_data=0.22,               # Midpoint of 0.15-0.28 theoretical range
        Q_bar=200_000.0,               # Conservative batch size estimate
        T=1000.0,                      # 1000 ETH target

        # Status tracking
        alpha_data_status=ParameterCalibrationStatus.THEORETICAL,
        Q_bar_status=ParameterCalibrationStatus.THEORETICAL,
        optimization_status=ParameterCalibrationStatus.UNCALIBRATED
    )

    return UnifiedFeeCalculator(params)


def create_experimental_calculator() -> UnifiedFeeCalculator:
    """Create calculator with experimental parameters for testing."""
    params = FeeParameters(
        # Experimental parameters
        mu=0.1,                        # Small L1 component
        nu=0.7,                        # Aggressive vault healing
        H=72,                          # ~2.4 minute horizon
        lambda_B=0.5,                  # Responsive smoothing

        # Same estimates (still theoretical)
        alpha_data=0.22,
        Q_bar=200_000.0,
        T=1000.0,

        # Status tracking
        alpha_data_status=ParameterCalibrationStatus.THEORETICAL,
        Q_bar_status=ParameterCalibrationStatus.THEORETICAL,
        optimization_status=ParameterCalibrationStatus.UNCALIBRATED
    )

    return UnifiedFeeCalculator(params)