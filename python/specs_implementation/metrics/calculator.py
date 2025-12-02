"""
Metrics Calculation Pipeline

Provides high-level interface for computing all SPECS.md metrics from simulation data.
Integrates constraints and objectives evaluation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict

from .constraints import (
    ConstraintEvaluator, ConstraintThresholds, ConstraintResults, SimulationResults
)
from .objectives import (
    ObjectiveCalculator, ObjectiveWeights, ObjectiveResults, StakeholderProfile
)


@dataclass
class MetricsConfiguration:
    """Complete configuration for metrics calculation"""
    constraint_thresholds: ConstraintThresholds
    objective_weights: ObjectiveWeights


@dataclass
class ComprehensiveMetrics:
    """Complete metrics evaluation results"""
    # Core feasibility
    is_feasible: bool
    constraint_results: ConstraintResults
    objective_results: ObjectiveResults

    # Summary scores (for optimization)
    ux_score: float
    robustness_score: float
    capital_efficiency_score: float

    # Configuration used
    configuration: MetricsConfiguration


class MetricsCalculator:
    """
    High-level interface for SPECS.md metrics calculation

    Combines constraint evaluation (Section 6) and objective calculation (Section 7)
    into a unified pipeline for parameter optimization.

    Supports stakeholder-specific optimization profiles:
    - PROTOCOL_LAUNCH: Safety-first for new protocol deployment
    - USER_CENTRIC: Low fees and predictability for user adoption
    - OPERATOR_FOCUSED: Capital efficiency for business operations
    - STRESS_TESTED: Crisis-ready robustness optimization
    - BALANCED: Equal consideration of all objectives
    """

    def __init__(self, config: Optional[MetricsConfiguration] = None, stakeholder_profile: Optional[StakeholderProfile] = None):
        """
        Initialize metrics calculator with stakeholder-specific optimization

        Args:
            config: Complete metrics configuration. Uses defaults if not provided.
            stakeholder_profile: Predefined stakeholder profile for automatic weight selection
        """
        if config is None:
            # Use stakeholder profile to generate appropriate weights
            objective_weights = (
                ObjectiveCalculator.get_stakeholder_weights(stakeholder_profile)
                if stakeholder_profile is not None
                else ObjectiveWeights()
            )
            config = MetricsConfiguration(
                constraint_thresholds=ConstraintThresholds(),
                objective_weights=objective_weights
            )

        self.config = config
        self.stakeholder_profile = stakeholder_profile
        self.constraint_evaluator = ConstraintEvaluator(config.constraint_thresholds)
        self.objective_calculator = ObjectiveCalculator(config.objective_weights)

    def evaluate_parameter_set(
        self,
        simulation_results: SimulationResults
    ) -> ComprehensiveMetrics:
        """
        Complete evaluation of a parameter set

        Args:
            simulation_results: Full simulation trajectory

        Returns:
            ComprehensiveMetrics with feasibility, constraints, and objectives
        """
        # Evaluate hard constraints
        constraint_results = self.constraint_evaluator.evaluate_all_constraints(
            simulation_results
        )

        # Calculate soft objectives
        objective_results = self.objective_calculator.calculate_all_objectives(
            simulation_results,
            V_min=self.config.constraint_thresholds.V_min
        )

        # Create comprehensive metrics
        comprehensive_metrics = ComprehensiveMetrics(
            is_feasible=constraint_results.is_feasible,
            constraint_results=constraint_results,
            objective_results=objective_results,
            ux_score=objective_results.ux_objective,
            robustness_score=objective_results.robustness_objective,
            capital_efficiency_score=objective_results.capital_efficiency_objective,
            configuration=self.config
        )

        return comprehensive_metrics

    @classmethod
    def for_stakeholder(cls, profile: StakeholderProfile) -> 'MetricsCalculator':
        """
        Create a metrics calculator optimized for a specific stakeholder profile

        Args:
            profile: Stakeholder optimization profile

        Returns:
            MetricsCalculator configured for the stakeholder
        """
        return cls(stakeholder_profile=profile)

    def get_profile_description(self) -> str:
        """
        Get human-readable description of current optimization profile

        Returns:
            Description of the stakeholder profile and weight priorities
        """
        if self.stakeholder_profile is None:
            return "Custom objective weights (no predefined profile)"

        weights = self.config.objective_weights

        descriptions = {
            StakeholderProfile.PROTOCOL_LAUNCH: f"Protocol Launch Profile - Prioritizes safety (DWD weight: {weights.u1_dwd:.0e}) and stability (jump protection: {weights.w3_jump_p95:.1f}x) over fees",
            StakeholderProfile.USER_CENTRIC: f"User-Centric Profile - Prioritizes low fees (fee weight: {weights.w1_avg_fee:.0e}) and predictability (jump protection: {weights.w3_jump_p95:.1f}x)",
            StakeholderProfile.OPERATOR_FOCUSED: f"Operator-Focused Profile - Prioritizes capital efficiency and business metrics, moderate safety (DWD weight: {weights.u1_dwd:.0e})",
            StakeholderProfile.STRESS_TESTED: f"Stress-Tested Profile - Optimized for crisis scenarios with maximum safety (DWD weight: {weights.u1_dwd:.0e})",
            StakeholderProfile.BALANCED: f"Balanced Profile - Balanced optimization across UX, safety, and efficiency"
        }

        return descriptions.get(self.stakeholder_profile, "Unknown profile")

    def batch_evaluate_parameters(
        self,
        parameter_simulations: List[Tuple[Dict, SimulationResults]]
    ) -> List[Tuple[Dict, ComprehensiveMetrics]]:
        """
        Evaluate multiple parameter sets in batch

        Args:
            parameter_simulations: List of (parameters_dict, simulation_results) tuples

        Returns:
            List of (parameters_dict, comprehensive_metrics) tuples
        """
        results = []

        for params, sim_results in parameter_simulations:
            metrics = self.evaluate_parameter_set(sim_results)
            results.append((params, metrics))

        return results

    def filter_feasible_parameters(
        self,
        parameter_metrics: List[Tuple[Dict, ComprehensiveMetrics]]
    ) -> List[Tuple[Dict, ComprehensiveMetrics]]:
        """
        Filter to only feasible parameter sets

        Args:
            parameter_metrics: List of (parameters, metrics) tuples

        Returns:
            Filtered list containing only feasible parameter sets
        """
        feasible_results = [
            (params, metrics)
            for params, metrics in parameter_metrics
            if metrics.is_feasible
        ]
        return feasible_results

    def find_pareto_frontier(
        self,
        feasible_parameters: List[Tuple[Dict, ComprehensiveMetrics]],
        objectives: List[str] = None
    ) -> List[Tuple[Dict, ComprehensiveMetrics]]:
        """
        Find Pareto frontier among feasible parameter sets

        Args:
            feasible_parameters: List of feasible (parameters, metrics) tuples
            objectives: List of objective names to consider. Defaults to all three.

        Returns:
            List of Pareto-optimal (parameters, metrics) tuples
        """
        if not feasible_parameters:
            return []

        if objectives is None:
            objectives = ['ux_score', 'robustness_score', 'capital_efficiency_score']

        # Extract objective vectors
        objective_matrix = []
        for _, metrics in feasible_parameters:
            obj_vector = []
            for obj_name in objectives:
                if obj_name == 'ux_score':
                    obj_vector.append(metrics.ux_score)
                elif obj_name == 'robustness_score':
                    obj_vector.append(metrics.robustness_score)
                elif obj_name == 'capital_efficiency_score':
                    obj_vector.append(metrics.capital_efficiency_score)
                else:
                    raise ValueError(f"Unknown objective: {obj_name}")
            objective_matrix.append(obj_vector)

        objective_matrix = np.array(objective_matrix)

        # Find Pareto frontier (assuming minimization for all objectives)
        pareto_indices = self._find_pareto_indices(objective_matrix)

        pareto_frontier = [
            feasible_parameters[i] for i in pareto_indices
        ]

        return pareto_frontier

    def _find_pareto_indices(self, objective_matrix: np.ndarray) -> List[int]:
        """
        Find indices of Pareto-optimal solutions

        Args:
            objective_matrix: N x M matrix where N=solutions, M=objectives

        Returns:
            List of indices of Pareto-optimal solutions
        """
        n_solutions = objective_matrix.shape[0]
        is_pareto = np.ones(n_solutions, dtype=bool)

        for i in range(n_solutions):
            if is_pareto[i]:
                # Check if solution i is dominated by any other solution
                for j in range(n_solutions):
                    if i != j and is_pareto[j]:
                        # j dominates i if j is better or equal in all objectives
                        # and strictly better in at least one
                        dominates = np.all(objective_matrix[j] <= objective_matrix[i])
                        strictly_better = np.any(objective_matrix[j] < objective_matrix[i])

                        if dominates and strictly_better:
                            is_pareto[i] = False
                            break

        pareto_indices = np.where(is_pareto)[0].tolist()
        return pareto_indices

    def get_summary_report(
        self,
        comprehensive_metrics: ComprehensiveMetrics
    ) -> Dict:
        """
        Generate summary report for parameter evaluation

        Args:
            comprehensive_metrics: Complete metrics evaluation

        Returns:
            Dictionary with formatted summary information
        """
        metrics = comprehensive_metrics

        # Constraint summary
        constraint_summary = {
            'feasible': metrics.is_feasible,
            'insolvency_probability': metrics.constraint_results.insolvency_probability,
            'cost_recovery_ratio': metrics.constraint_results.cost_recovery_ratio,
            'fee_p99_gwei': metrics.constraint_results.fee_p99_gwei,
            'violations': metrics.constraint_results.violations
        }

        # Objective summary
        objective_summary = {
            'ux_metrics': {
                'avg_fee_gwei': metrics.objective_results.avg_fee_gwei,
                'cv_global': metrics.objective_results.cv_global,
                'jump_p95': metrics.objective_results.jump_p95,
                'cv_1h': metrics.objective_results.cv_1h,
                'cv_6h': metrics.objective_results.cv_6h,
                'total_score': metrics.ux_score
            },
            'robustness_metrics': {
                'deficit_weighted_duration': metrics.objective_results.deficit_weighted_duration,
                'max_underfunding_streak': metrics.objective_results.max_underfunding_streak,
                'total_score': metrics.robustness_score
            },
            'capital_efficiency': {
                'eth_per_gas_unit': metrics.objective_results.capital_efficiency_eth_per_gas,
                'total_score': metrics.capital_efficiency_score
            }
        }

        summary = {
            'feasible': metrics.is_feasible,
            'constraints': constraint_summary,
            'objectives': objective_summary
        }

        return summary

    @staticmethod
    def create_simulation_results_from_arrays(
        timestamps: np.ndarray,
        vault_balances: np.ndarray,
        fees_per_gas: np.ndarray,
        l1_costs: np.ndarray,
        Q_bar: float,
        target_vault_size: float = 1000.0
    ) -> SimulationResults:
        """
        Helper to create SimulationResults from raw arrays

        Args:
            timestamps: Time series timestamps
            vault_balances: V(t) trajectory
            fees_per_gas: F_L2(t) trajectory
            l1_costs: C_L1(t) trajectory
            Q_bar: Average gas per batch
            target_vault_size: Target vault balance T

        Returns:
            SimulationResults object
        """
        # Calculate derived quantities
        revenues = fees_per_gas * Q_bar
        subsidies = np.minimum(l1_costs, vault_balances)
        deficits = target_vault_size - vault_balances

        return SimulationResults(
            timestamps=timestamps,
            vault_balances=vault_balances,
            fees_per_gas=fees_per_gas,
            l1_costs=l1_costs,
            revenues=revenues,
            subsidies=subsidies,
            deficits=deficits,
            Q_bar=Q_bar
        )