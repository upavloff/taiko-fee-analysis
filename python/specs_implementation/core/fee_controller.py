"""
Fee Controller Implementation

Implements Section 3 of SPECS.md: Fee Rule (Per-Gas Basefee)

Mathematical Formulas:

3.1 Raw per-gas basefee:
f^raw(t) = μ * Ĉ_L1(t)/Q̄ + ν * D(t)/(H*Q̄)

3.2 Static min/max bounds:
f^clip(t) = min(F_max, max(F_min, f^raw(t)))

3.3 Rate-limiting fee jumps:
F_L2(t-1)(1-κ_↓) ≤ F_L2(t) ≤ F_L2(t-1)(1+κ_↑)

where:
- μ ∈ [0,1]: L1 weight parameter
- ν ∈ [0,1]: deficit weight parameter
- H > 0: amortization horizon (in batches)
- Q̄: average gas per batch (calibrated at 6.9e5)
- D(t): vault deficit at time t
- Ĉ_L1(t): smoothed L1 cost estimate
"""

from typing import Optional, Union
import numpy as np
from .units import Wei, WeiPerGas
from .validation import validate_fee_function, validate_fee_calculation_inputs


class FeeController:
    """
    Fee controller implementing raw basefee calculation, clipping, and rate limiting.

    Implements Section 3 of SPECS.md with exact mathematical formulas.
    """

    def __init__(
        self,
        mu: float,
        nu: float,
        horizon_h: int,
        q_bar: float = 6.9e5,  # Calibrated from Taiko Alethia data
        f_min: float = 0,   # No minimum bound
        f_max: float = float('inf'),  # No maximum bound
        kappa_up: float = 1.0,   # No rate limiting by default
        kappa_down: float = 1.0  # No rate limiting by default
    ):
        """
        Initialize the fee controller.

        Args:
            mu: L1 weight parameter μ ∈ [0,1]
            nu: Deficit weight parameter ν ∈ [0,1]
            horizon_h: Amortization horizon H > 0 (in batches)
            q_bar: Average gas per batch Q̄ (default: 6.9e5 from Taiko data)
            f_min: Minimum basefee F_min (default: 1e6 wei = 0.001 gwei)
            f_max: Maximum basefee F_max (default: 1e12 wei = 1000 gwei)
            kappa_up: Max relative up move per batch κ_↑ ∈ [0,1]
            kappa_down: Max relative down move per batch κ_↓ ∈ [0,1]

        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate core parameters
        if not (0 <= mu <= 1):
            raise ValueError(f"mu must be in [0,1], got {mu}")
        if not (0 <= nu <= 1):
            raise ValueError(f"nu must be in [0,1], got {nu}")
        if horizon_h <= 0:
            raise ValueError(f"horizon_h must be positive, got {horizon_h}")
        if q_bar <= 0:
            raise ValueError(f"q_bar must be positive, got {q_bar}")

        # Validate bounds
        if f_min < 0:
            raise ValueError(f"f_min must be non-negative, got {f_min}")
        if f_max <= f_min:
            raise ValueError(f"f_max must be greater than f_min, got f_max={f_max}, f_min={f_min}")

        # Validate rate limiting parameters
        if not (0 <= kappa_up <= 1):
            raise ValueError(f"kappa_up must be in [0,1], got {kappa_up}")
        if not (0 <= kappa_down <= 1):
            raise ValueError(f"kappa_down must be in [0,1], got {kappa_down}")

        # Store parameters
        self.mu = mu
        self.nu = nu
        self.horizon_h = horizon_h
        self.q_bar = q_bar
        self.f_min = f_min
        self.f_max = f_max
        self.kappa_up = kappa_up
        self.kappa_down = kappa_down

        # State for rate limiting
        self.previous_fee: Optional[float] = None

    @validate_fee_function
    def calculate_raw_basefee(self, smoothed_l1_cost_wei: Union[int, float], deficit_wei: Union[int, float]) -> int:
        """
        Calculate raw per-gas basefee before clipping and rate limiting.

        Implements: f^raw(t) = μ * Ĉ_L1(t)/Q̄ + ν * D(t)/(H*Q̄)

        Args:
            smoothed_l1_cost_wei: Smoothed L1 cost estimate Ĉ_L1(t) in WEI per batch
            deficit_wei: Vault deficit D(t) in WEI (positive = deficit, negative = surplus)

        Returns:
            Raw basefee f^raw(t) in wei per gas

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs (decorator also validates)
        validate_fee_calculation_inputs(smoothed_l1_cost_wei, deficit_wei, self.q_bar, "calculate_raw_basefee")

        if smoothed_l1_cost_wei < 0:
            raise ValueError(f"Smoothed L1 cost cannot be negative, got {smoothed_l1_cost_wei:,.0f} wei")

        # Apply formula: f^raw(t) = μ * Ĉ_L1(t)/Q̄ + ν * D(t)/(H*Q̄)
        # Note: All inputs in wei, output in wei per gas
        l1_component = self.mu * smoothed_l1_cost_wei / self.q_bar
        deficit_component = self.nu * deficit_wei / (self.horizon_h * self.q_bar)

        raw_fee_wei_per_gas = l1_component + deficit_component

        # Convert to integer wei
        return int(round(raw_fee_wei_per_gas))

    def apply_bounds(self, raw_fee_wei: int) -> int:
        """
        Apply static min/max bounds to raw basefee.

        Implements: f^clip(t) = min(F_max, max(F_min, f^raw(t)))

        Args:
            raw_fee_wei: Raw basefee f^raw(t) in wei per gas

        Returns:
            Clipped basefee f^clip(t) in wei per gas
        """
        return int(min(self.f_max, max(self.f_min, raw_fee_wei)))

    def apply_rate_limiting(self, clipped_fee_wei: int) -> int:
        """
        Apply rate limiting to clipped basefee.

        Implements: F_L2(t-1)(1-κ_↓) ≤ F_L2(t) ≤ F_L2(t-1)(1+κ_↑)

        Args:
            clipped_fee_wei: Clipped basefee f^clip(t) in wei per gas

        Returns:
            Rate-limited basefee F_L2(t) in wei per gas
        """
        if self.previous_fee is None:
            # No rate limiting for first fee
            final_fee = clipped_fee_wei
        else:
            # Apply rate limiting bounds
            max_up_move = self.previous_fee * (1 + self.kappa_up)
            max_down_move = self.previous_fee * (1 - self.kappa_down)

            final_fee = int(min(max_up_move, max(max_down_move, clipped_fee_wei)))

        # Update previous fee for next iteration
        self.previous_fee = final_fee
        return int(final_fee)

    @validate_fee_function
    def calculate_fee(self, smoothed_l1_cost_wei: Union[int, float], deficit_wei: Union[int, float]) -> int:
        """
        Calculate final L2 basefee with all stages: raw → clipped → rate-limited.

        Args:
            smoothed_l1_cost_wei: Smoothed L1 cost estimate Ĉ_L1(t) in WEI per batch
            deficit_wei: Vault deficit D(t) in WEI

        Returns:
            Final L2 basefee F_L2(t) in wei per gas
        """
        # Stage 1: Calculate raw basefee
        raw_fee = self.calculate_raw_basefee(smoothed_l1_cost_wei, deficit_wei)

        # Stage 2: Apply bounds
        clipped_fee = self.apply_bounds(raw_fee)

        # Stage 3: Apply rate limiting
        final_fee = self.apply_rate_limiting(clipped_fee)

        return final_fee

    def calculate_batch_revenue(self, basefee_per_gas: float, gas_used: Optional[float] = None) -> float:
        """
        Calculate total revenue collected from a batch.

        Implements: R(t) = F_L2(t) * Q(t) = F_L2(t) * Q̄

        Args:
            basefee_per_gas: L2 basefee F_L2(t) in wei per gas
            gas_used: Gas used in batch Q(t). If None, uses Q̄.

        Returns:
            Total revenue R(t) in wei
        """
        gas = gas_used if gas_used is not None else self.q_bar
        return basefee_per_gas * gas

    def reset_state(self) -> None:
        """
        Reset controller state (useful for new simulations).
        """
        self.previous_fee = None

    def get_deficit_decay_factor(self) -> float:
        """
        Get the deficit decay factor φ from the approximate recursion.

        From SPECS.md Section 4.3: φ := 1 - ν/H

        Returns:
            Deficit decay factor φ ∈ [0,1]
        """
        return 1 - (self.nu / self.horizon_h)

    def process_series(
        self,
        smoothed_l1_costs: np.ndarray,
        deficits: np.ndarray
    ) -> np.ndarray:
        """
        Process time series of L1 costs and deficits to compute basefee series.

        Args:
            smoothed_l1_costs: Array of smoothed L1 costs [Ĉ_L1(0), ..., Ĉ_L1(T)]
            deficits: Array of deficits [D(0), ..., D(T)]

        Returns:
            Array of final basefees [F_L2(0), ..., F_L2(T)]

        Raises:
            ValueError: If input arrays have different lengths
        """
        if len(smoothed_l1_costs) != len(deficits):
            raise ValueError("L1 costs and deficits must have same length")

        # Reset state for clean processing
        self.reset_state()

        basefees = np.zeros(len(smoothed_l1_costs))

        for i, (l1_cost, deficit) in enumerate(zip(smoothed_l1_costs, deficits)):
            basefees[i] = self.calculate_fee(l1_cost, deficit)

        return basefees

    def __str__(self) -> str:
        """String representation of the fee controller."""
        return (f"FeeController(μ={self.mu}, ν={self.nu}, H={self.horizon_h}, "
                f"Q̄={self.q_bar:.1e}, rate_limits=({self.kappa_down},{self.kappa_up}))")

    def __repr__(self) -> str:
        """Detailed representation of the fee controller."""
        return self.__str__()