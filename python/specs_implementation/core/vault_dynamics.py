"""
Vault Dynamics Implementation

Implements Section 4 of SPECS.md: Vault Dynamics

Mathematical Formulas:

4.1 Subsidy rule:
S(t) = min(C_L1(t), V(t))

4.2 Vault balance update:
V(t+1) = V(t) + R(t) - S(t)

4.2 Deficit calculation:
D(t) = T - V(t)

4.3 Deficit recursion (simplified, solvent regime):
D(t+1) = φ * D(t) + (1-μ) * C_L1(t)
where φ := 1 - ν/H

where:
- V(t): vault balance at time t
- T: target vault level
- D(t): vault deficit (positive = below target)
- S(t): subsidy paid to proposer
- R(t): revenue collected from L2 fees
- C_L1(t): actual L1 cost for batch t
"""

from typing import Optional, Tuple
import numpy as np


class VaultDynamics:
    """
    Vault dynamics implementing subsidy rules, balance updates, and deficit tracking.

    Implements Section 4 of SPECS.md with exact mathematical formulas.
    """

    def __init__(self, target_balance: float, initial_balance: Optional[float] = None):
        """
        Initialize vault dynamics.

        Args:
            target_balance: Target vault level T > 0
            initial_balance: Initial vault balance V(0). If None, starts at target.

        Raises:
            ValueError: If target_balance is not positive
        """
        if target_balance <= 0:
            raise ValueError(f"Target balance must be positive, got {target_balance}")

        self.target_balance = target_balance
        self.vault_balance = initial_balance if initial_balance is not None else target_balance

        # Track history for analysis
        self.balance_history = [self.vault_balance]
        self.deficit_history = [self.calculate_deficit()]

    def calculate_deficit(self) -> float:
        """
        Calculate current vault deficit.

        Implements: D(t) = T - V(t)

        Returns:
            Current deficit D(t). Positive = deficit, negative = surplus.
        """
        return self.target_balance - self.vault_balance

    def calculate_subsidy(self, l1_cost: float) -> float:
        """
        Calculate subsidy payment to proposer.

        Implements: S(t) = min(C_L1(t), V(t))

        Args:
            l1_cost: Actual L1 cost C_L1(t) for current batch

        Returns:
            Subsidy amount S(t)

        Raises:
            ValueError: If l1_cost is negative
        """
        if l1_cost < 0:
            raise ValueError(f"L1 cost cannot be negative, got {l1_cost}")

        return min(l1_cost, self.vault_balance)

    def update_vault(self, revenue: float, l1_cost: float) -> Tuple[float, float]:
        """
        Update vault balance with revenue and subsidy payment.

        Implements:
        - S(t) = min(C_L1(t), V(t))
        - V(t+1) = V(t) + R(t) - S(t)

        Args:
            revenue: Fee revenue collected R(t)
            l1_cost: Actual L1 cost C_L1(t) for current batch

        Returns:
            Tuple of (subsidy_paid, new_vault_balance)

        Raises:
            ValueError: If revenue or l1_cost are negative
        """
        if revenue < 0:
            raise ValueError(f"Revenue cannot be negative, got {revenue}")
        if l1_cost < 0:
            raise ValueError(f"L1 cost cannot be negative, got {l1_cost}")

        # Calculate subsidy according to rule
        subsidy = self.calculate_subsidy(l1_cost)

        # Update vault balance
        self.vault_balance = self.vault_balance + revenue - subsidy

        # Record history
        self.balance_history.append(self.vault_balance)
        self.deficit_history.append(self.calculate_deficit())

        return subsidy, self.vault_balance

    def get_vault_balance(self) -> float:
        """
        Get current vault balance.

        Returns:
            Current vault balance V(t)
        """
        return self.vault_balance

    def get_deficit(self) -> float:
        """
        Get current vault deficit.

        Returns:
            Current deficit D(t)
        """
        return self.calculate_deficit()

    def is_solvent(self, threshold: float = 0.0) -> bool:
        """
        Check if vault is solvent (above threshold).

        Args:
            threshold: Minimum balance threshold (default: 0)

        Returns:
            True if vault balance > threshold
        """
        return self.vault_balance > threshold

    def get_solvency_ratio(self) -> float:
        """
        Get vault balance as ratio of target.

        Returns:
            Ratio V(t)/T ∈ [0, ∞)
        """
        return self.vault_balance / self.target_balance

    def reset_state(self, initial_balance: Optional[float] = None) -> None:
        """
        Reset vault to initial state.

        Args:
            initial_balance: New initial balance. If None, uses target balance.
        """
        self.vault_balance = (
            initial_balance if initial_balance is not None else self.target_balance
        )
        self.balance_history = [self.vault_balance]
        self.deficit_history = [self.calculate_deficit()]

    def get_balance_history(self) -> np.ndarray:
        """
        Get history of vault balances.

        Returns:
            Array of vault balances [V(0), V(1), ..., V(t)]
        """
        return np.array(self.balance_history)

    def get_deficit_history(self) -> np.ndarray:
        """
        Get history of vault deficits.

        Returns:
            Array of deficits [D(0), D(1), ..., D(t)]
        """
        return np.array(self.deficit_history)

    def simulate_step(self, revenue: float, l1_cost: float) -> dict:
        """
        Simulate one vault dynamics step and return detailed results.

        Args:
            revenue: Fee revenue R(t)
            l1_cost: L1 cost C_L1(t)

        Returns:
            Dictionary with step details:
            - 'revenue': revenue collected
            - 'l1_cost': L1 cost incurred
            - 'subsidy': subsidy paid
            - 'vault_balance_before': balance before update
            - 'vault_balance_after': balance after update
            - 'deficit_before': deficit before update
            - 'deficit_after': deficit after update
            - 'fully_subsidized': whether proposer was fully reimbursed
        """
        # Record before state
        balance_before = self.vault_balance
        deficit_before = self.calculate_deficit()

        # Calculate subsidy
        subsidy = self.calculate_subsidy(l1_cost)
        fully_subsidized = (subsidy == l1_cost)

        # Update vault
        subsidy_paid, balance_after = self.update_vault(revenue, l1_cost)
        deficit_after = self.calculate_deficit()

        return {
            'revenue': revenue,
            'l1_cost': l1_cost,
            'subsidy': subsidy_paid,
            'vault_balance_before': balance_before,
            'vault_balance_after': balance_after,
            'deficit_before': deficit_before,
            'deficit_after': deficit_after,
            'fully_subsidized': fully_subsidized,
        }

    def process_series(
        self,
        revenue_series: np.ndarray,
        l1_cost_series: np.ndarray
    ) -> dict:
        """
        Process time series of revenues and L1 costs.

        Args:
            revenue_series: Array of revenues [R(0), R(1), ..., R(T)]
            l1_cost_series: Array of L1 costs [C_L1(0), C_L1(1), ..., C_L1(T)]

        Returns:
            Dictionary with simulation results:
            - 'vault_balances': vault balance trajectory
            - 'deficits': deficit trajectory
            - 'subsidies': subsidies paid
            - 'solvency_ratios': V(t)/T trajectory

        Raises:
            ValueError: If input arrays have different lengths
        """
        if len(revenue_series) != len(l1_cost_series):
            raise ValueError("Revenue and L1 cost series must have same length")

        # Reset state for clean processing
        self.reset_state()

        n_steps = len(revenue_series)
        subsidies = np.zeros(n_steps)
        balances = np.zeros(n_steps + 1)  # Include initial state
        deficits = np.zeros(n_steps + 1)

        # Record initial state
        balances[0] = self.vault_balance
        deficits[0] = self.calculate_deficit()

        # Process each step
        for i, (revenue, l1_cost) in enumerate(zip(revenue_series, l1_cost_series)):
            subsidy, _ = self.update_vault(revenue, l1_cost)
            subsidies[i] = subsidy
            balances[i + 1] = self.vault_balance
            deficits[i + 1] = self.calculate_deficit()

        return {
            'vault_balances': balances,
            'deficits': deficits,
            'subsidies': subsidies,
            'solvency_ratios': balances / self.target_balance,
        }

    def __str__(self) -> str:
        """String representation of vault dynamics."""
        return (f"VaultDynamics(T={self.target_balance:.2e}, "
                f"V(t)={self.vault_balance:.2e}, D(t)={self.calculate_deficit():.2e})")

    def __repr__(self) -> str:
        """Detailed representation of vault dynamics."""
        return self.__str__()