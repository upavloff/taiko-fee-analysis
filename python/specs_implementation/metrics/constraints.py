"""
Hard Constraints Implementation

Implements SPECS.md Section 6: Hard Constraint Evaluation

Mathematical Formulas:

6.1 Solvency/ruin probability:
p_insolv(θ) = Pr[∃t ≤ T_horizon: V(t) < V_min] ≤ ε

6.2 Long-run cost recovery (budget balance):
CRR(θ) = (Σt F_L2(t)*Q̄) / (Σt C_L1(t)) ∈ [1-δ_cr, 1+δ_cr]

6.3 Extreme fee bound (UX sanity):
F_0.99(θ) ≤ F_max_UX

6.4 Fairness guardrail (optional):
φ(k) = F_paid(k) / C_resp(k) ∈ [1-δ_fair, 1+δ_fair] for 95% of cohorts
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ConstraintThresholds:
    """Threshold parameters for constraint evaluation"""
    # Solvency constraint (Section 6.1)
    V_min: float = 0.0  # Critical vault threshold (ETH)
    epsilon_insolvency: float = 0.01  # Max insolvency risk (1%)

    # Cost recovery constraint (Section 6.2)
    delta_cr: float = 0.05  # Cost recovery tolerance (±5%)

    # UX constraint (Section 6.3)
    F_max_UX: float = 50e9  # Max acceptable fee (50 gwei in wei/gas)

    # Fairness constraint (Section 6.4) - optional
    delta_fair: float = 0.5  # Fairness tolerance (±50%)
    fairness_threshold: float = 0.95  # 95% of cohorts must satisfy
    cohort_size_batches: int = 300  # ~1 month of batches


@dataclass
class SimulationResults:
    """Container for simulation trajectory data"""
    timestamps: np.ndarray
    vault_balances: np.ndarray  # V(t) trajectory
    fees_per_gas: np.ndarray   # F_L2(t) trajectory
    l1_costs: np.ndarray       # C_L1(t) trajectory
    revenues: np.ndarray       # R(t) = F_L2(t) * Q̄
    subsidies: np.ndarray      # S(t) trajectory
    deficits: np.ndarray       # D(t) = T - V(t)
    Q_bar: float               # Average gas per batch


@dataclass
class ConstraintResults:
    """Results of constraint evaluation"""
    is_feasible: bool
    violations: Dict[str, str]

    # Detailed metrics
    insolvency_probability: float
    cost_recovery_ratio: float
    fee_p99_gwei: float
    fairness_violation_rate: Optional[float] = None


class ConstraintEvaluator:
    """SPECS.md Section 6: Hard Constraint Evaluation"""

    def __init__(self, thresholds: Optional[ConstraintThresholds] = None):
        """
        Initialize constraint evaluator

        Args:
            thresholds: Constraint threshold parameters
        """
        self.thresholds = thresholds or ConstraintThresholds()

    def evaluate_all_constraints(
        self,
        simulation_results: SimulationResults
    ) -> ConstraintResults:
        """
        Evaluate all hard constraints on simulation results

        Args:
            simulation_results: Complete simulation trajectory

        Returns:
            ConstraintResults with feasibility and detailed metrics
        """
        violations = {}

        # 6.1 Solvency constraint
        insolvency_prob = self.evaluate_solvency_constraint(simulation_results)
        if insolvency_prob > self.thresholds.epsilon_insolvency:
            violations['solvency'] = (
                f"Insolvency risk {insolvency_prob:.4f} > "
                f"threshold {self.thresholds.epsilon_insolvency:.4f}"
            )

        # 6.2 Cost recovery constraint
        cost_recovery_ratio = self.evaluate_cost_recovery_constraint(simulation_results)
        lower_bound = 1 - self.thresholds.delta_cr
        upper_bound = 1 + self.thresholds.delta_cr
        if not (lower_bound <= cost_recovery_ratio <= upper_bound):
            violations['cost_recovery'] = (
                f"Cost recovery ratio {cost_recovery_ratio:.4f} outside "
                f"bounds [{lower_bound:.4f}, {upper_bound:.4f}]"
            )

        # 6.3 UX constraint
        fee_p99_gwei = self.evaluate_ux_constraint(simulation_results)
        fee_p99_gwei_threshold = self.thresholds.F_max_UX / 1e9  # Convert to gwei
        if fee_p99_gwei > fee_p99_gwei_threshold:
            violations['ux_extreme_fees'] = (
                f"99th percentile fee {fee_p99_gwei:.2f} gwei > "
                f"threshold {fee_p99_gwei_threshold:.2f} gwei"
            )

        # 6.4 Fairness constraint (optional)
        fairness_violation_rate = None
        if len(simulation_results.timestamps) >= self.thresholds.cohort_size_batches:
            fairness_violation_rate = self.evaluate_fairness_constraint(simulation_results)
            if fairness_violation_rate > (1 - self.thresholds.fairness_threshold):
                violations['fairness'] = (
                    f"Fairness violation rate {fairness_violation_rate:.2%} > "
                    f"threshold {1 - self.thresholds.fairness_threshold:.2%}"
                )

        is_feasible = len(violations) == 0

        return ConstraintResults(
            is_feasible=is_feasible,
            violations=violations,
            insolvency_probability=insolvency_prob,
            cost_recovery_ratio=cost_recovery_ratio,
            fee_p99_gwei=fee_p99_gwei,
            fairness_violation_rate=fairness_violation_rate
        )

    def evaluate_solvency_constraint(
        self,
        simulation_results: SimulationResults
    ) -> float:
        """
        Evaluate solvency/ruin probability constraint

        SPECS.md 6.1: p_insolv(θ) = Pr[∃t ≤ T_horizon: V(t) < V_min] ≤ ε

        Args:
            simulation_results: Simulation trajectory data

        Returns:
            Insolvency probability (0.0 = never insolvent, 1.0 = always insolvent)
        """
        vault_balances = simulation_results.vault_balances

        # Check if vault ever goes below critical threshold
        insolvency_events = vault_balances < self.thresholds.V_min

        # For Monte Carlo simulations, this would be fraction of runs with insolvency
        # For single deterministic run, it's binary: did insolvency occur?
        if np.any(insolvency_events):
            # Could refine this to account for stochastic scenarios
            insolvency_probability = 1.0
        else:
            insolvency_probability = 0.0

        return insolvency_probability

    def evaluate_cost_recovery_constraint(
        self,
        simulation_results: SimulationResults
    ) -> float:
        """
        Evaluate long-run cost recovery (budget balance) constraint

        SPECS.md 6.2: CRR(θ) = (Σt F_L2(t)*Q̄) / (Σt C_L1(t))

        Args:
            simulation_results: Simulation trajectory data

        Returns:
            Cost recovery ratio (1.0 = perfect balance, >1.0 = surplus, <1.0 = deficit)
        """
        total_fees_collected = np.sum(simulation_results.revenues)
        total_l1_costs = np.sum(simulation_results.l1_costs)

        if total_l1_costs <= 0:
            return float('inf')  # Edge case: no L1 costs

        cost_recovery_ratio = total_fees_collected / total_l1_costs
        return cost_recovery_ratio

    def evaluate_ux_constraint(
        self,
        simulation_results: SimulationResults
    ) -> float:
        """
        Evaluate extreme fee bound (UX sanity) constraint

        SPECS.md 6.3: F_0.99(θ) ≤ F_max_UX

        Args:
            simulation_results: Simulation trajectory data

        Returns:
            99th percentile fee in gwei
        """
        fees_wei_per_gas = simulation_results.fees_per_gas

        # Calculate 99th percentile
        fee_p99_wei = np.percentile(fees_wei_per_gas, 99)

        # Convert to gwei for readability
        fee_p99_gwei = fee_p99_wei / 1e9

        return fee_p99_gwei

    def evaluate_fairness_constraint(
        self,
        simulation_results: SimulationResults
    ) -> float:
        """
        Evaluate fairness guardrail (optional) constraint

        SPECS.md 6.4: φ(k) = F_paid(k) / C_resp(k) for cohorts k

        Args:
            simulation_results: Simulation trajectory data

        Returns:
            Fraction of cohorts violating fairness bounds
        """
        cohort_size = self.thresholds.cohort_size_batches
        n_batches = len(simulation_results.timestamps)
        n_cohorts = n_batches // cohort_size

        if n_cohorts == 0:
            return 0.0  # Not enough data for fairness analysis

        violation_count = 0

        for k in range(n_cohorts):
            start_idx = k * cohort_size
            end_idx = (k + 1) * cohort_size

            # Calculate cohort metrics
            F_paid_k = np.sum(simulation_results.revenues[start_idx:end_idx])
            C_resp_k = np.sum(simulation_results.l1_costs[start_idx:end_idx])

            if C_resp_k <= 0:
                continue  # Skip cohorts with no L1 costs

            # Markup ratio for cohort k
            phi_k = F_paid_k / C_resp_k

            # Check fairness bounds
            lower_bound = 1 - self.thresholds.delta_fair
            upper_bound = 1 + self.thresholds.delta_fair

            if not (lower_bound <= phi_k <= upper_bound):
                violation_count += 1

        violation_rate = violation_count / n_cohorts if n_cohorts > 0 else 0.0
        return violation_rate