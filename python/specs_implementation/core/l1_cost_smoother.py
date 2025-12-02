"""
L1 Cost Smoother Implementation

Implements Section 2.3 of SPECS.md:
Exponential Moving Average (EMA) smoothing for L1 posting costs.

Mathematical Formula:
Ĉ_L1(t) = (1-λ_C) * Ĉ_L1(t-1) + λ_C * C_L1(t)

where:
- λ_C ∈ (0,1]: smoothing parameter
- C_L1(t): actual L1 cost at time t
- Ĉ_L1(t): smoothed L1 cost estimate
"""

from typing import Optional
import numpy as np


class L1CostSmoother:
    """
    Exponential Moving Average smoother for L1 costs.

    Implements the EMA formula from SPECS.md Section 2.3:
    Ĉ_L1(t) = (1-λ_C) * Ĉ_L1(t-1) + λ_C * C_L1(t)

    Attributes:
        lambda_c: Smoothing parameter λ_C ∈ (0,1]
        smoothed_cost: Current smoothed cost estimate Ĉ_L1(t)
    """

    def __init__(self, lambda_c: float, initial_cost: Optional[float] = None):
        """
        Initialize the L1 cost smoother.

        Args:
            lambda_c: Smoothing parameter λ_C ∈ (0,1].
                     Higher values = faster adaptation to new costs.
            initial_cost: Initial smoothed cost estimate. If None, will be set
                         to the first observed cost.

        Raises:
            ValueError: If lambda_c is not in (0,1]
        """
        if not (0 < lambda_c <= 1):
            raise ValueError(f"lambda_c must be in (0,1], got {lambda_c}")

        self.lambda_c = lambda_c
        self.smoothed_cost = initial_cost
        self._initialized = initial_cost is not None

    def update(self, actual_cost: float) -> float:
        """
        Update the smoothed cost estimate with a new actual cost.

        Implements: Ĉ_L1(t) = (1-λ_C) * Ĉ_L1(t-1) + λ_C * C_L1(t)

        Args:
            actual_cost: Actual L1 cost C_L1(t) at current time

        Returns:
            Updated smoothed cost estimate Ĉ_L1(t)

        Raises:
            ValueError: If actual_cost is negative
        """
        if actual_cost < 0:
            raise ValueError(f"L1 cost cannot be negative, got {actual_cost}")

        if not self._initialized:
            # Initialize with first observation
            self.smoothed_cost = actual_cost
            self._initialized = True
        else:
            # Apply EMA formula
            self.smoothed_cost = (1 - self.lambda_c) * self.smoothed_cost + self.lambda_c * actual_cost

        return self.smoothed_cost

    def get_smoothed_cost(self) -> Optional[float]:
        """
        Get the current smoothed cost estimate.

        Returns:
            Current smoothed cost Ĉ_L1(t), or None if not initialized
        """
        return self.smoothed_cost

    def reset(self, initial_cost: Optional[float] = None) -> None:
        """
        Reset the smoother state.

        Args:
            initial_cost: New initial smoothed cost. If None, smoother
                         will reinitialize with next update.
        """
        self.smoothed_cost = initial_cost
        self._initialized = initial_cost is not None

    def process_series(self, cost_series: np.ndarray) -> np.ndarray:
        """
        Process a series of L1 costs and return smoothed estimates.

        Efficiently applies EMA to entire time series.

        Args:
            cost_series: Array of L1 costs [C_L1(0), C_L1(1), ..., C_L1(T)]

        Returns:
            Array of smoothed costs [Ĉ_L1(0), Ĉ_L1(1), ..., Ĉ_L1(T)]

        Raises:
            ValueError: If cost_series contains negative values or is empty
        """
        if len(cost_series) == 0:
            raise ValueError("Cost series cannot be empty")

        if np.any(cost_series < 0):
            raise ValueError("Cost series cannot contain negative values")

        # Initialize output array
        smoothed_series = np.zeros_like(cost_series)

        # Initialize with first cost if needed
        if not self._initialized:
            current_smoothed = cost_series[0]
            self._initialized = True
        else:
            current_smoothed = self.smoothed_cost

        # Apply EMA formula iteratively
        alpha = self.lambda_c
        for i, cost in enumerate(cost_series):
            if i == 0 and not self._initialized:
                current_smoothed = cost
            else:
                current_smoothed = (1 - alpha) * current_smoothed + alpha * cost

            smoothed_series[i] = current_smoothed

        # Update internal state to last smoothed value
        self.smoothed_cost = current_smoothed

        return smoothed_series

    def __str__(self) -> str:
        """String representation of the smoother."""
        status = "initialized" if self._initialized else "uninitialized"
        return f"L1CostSmoother(λ_C={self.lambda_c}, smoothed_cost={self.smoothed_cost}, {status})"

    def __repr__(self) -> str:
        """Detailed representation of the smoother."""
        return self.__str__()