"""
Formal Objective Function Combinations for Multi-Objective Optimization

This module defines mathematically rigorous objective functions for optimizing
the Taiko fee mechanism. Each objective function represents a different
stakeholder priority or deployment scenario.

Key Features:
1. Multiple optimization strategies (user-focused, protocol-focused, balanced)
2. Mathematically justified weight combinations
3. Pareto-efficient multi-objective formulations
4. Production-ready constraint integration
5. Scenario-specific objective customization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from enhanced_metrics import EnhancedMechanismMetrics


class OptimizationStrategy(Enum):
    """
    Predefined optimization strategies for different deployment scenarios.

    Each strategy represents a different stakeholder priority or protocol phase.
    """
    USER_CENTRIC = "user_centric"           # Maximize user experience
    PROTOCOL_CENTRIC = "protocol_centric"   # Maximize protocol robustness
    BALANCED = "balanced"                   # Balance all concerns
    LAUNCH_SAFE = "launch_safe"            # Conservative launch parameters
    CRISIS_READY = "crisis_ready"          # Prepare for extreme volatility
    CAPITAL_EFFICIENT = "capital_efficient" # Minimize capital requirements
    MEV_RESISTANT = "mev_resistant"        # Maximize attack resistance


@dataclass
class ObjectiveWeights:
    """
    Weight configuration for objective function components.

    All weights should sum to 1.0 for proper normalization.
    """

    # Primary metric weights (40% total budget)
    user_experience_weight: float = 0.40

    # Secondary metric weights (35% total budget)
    protocol_stability_weight: float = 0.35

    # Tertiary metric weights (25% total budget)
    economic_efficiency_weight: float = 0.15
    production_readiness_weight: float = 0.10

    # Advanced constraints (can be zero)
    adversarial_robustness_weight: float = 0.0

    # Individual metric fine-tuning within categories
    fee_affordability_emphasis: float = 1.0    # Within UX category
    vault_robustness_emphasis: float = 1.0     # Within stability category
    cost_recovery_emphasis: float = 1.0        # Within efficiency category

    def __post_init__(self):
        """Validate weight configuration."""
        main_weights_sum = (
            self.user_experience_weight +
            self.protocol_stability_weight +
            self.economic_efficiency_weight +
            self.production_readiness_weight +
            self.adversarial_robustness_weight
        )

        if abs(main_weights_sum - 1.0) > 0.01:
            warnings.warn(f"Objective weights sum to {main_weights_sum:.3f}, not 1.0")


class ObjectiveFunctionFactory:
    """
    Factory for creating objective functions tailored to different optimization goals.

    This class generates objective functions that can be used with multi-objective
    optimization algorithms like NSGA-II or single-objective optimizers.
    """

    @staticmethod
    def get_strategy_weights(strategy: OptimizationStrategy) -> ObjectiveWeights:
        """
        Get predefined weight configurations for different optimization strategies.

        Each strategy is designed for a specific deployment scenario or stakeholder focus.
        """

        if strategy == OptimizationStrategy.USER_CENTRIC:
            # Maximize user experience - for competitive L2s
            return ObjectiveWeights(
                user_experience_weight=0.60,      # Heavy focus on UX
                protocol_stability_weight=0.25,   # Basic safety
                economic_efficiency_weight=0.10,  # Less important
                production_readiness_weight=0.05, # Minimal
                fee_affordability_emphasis=1.5    # Extra focus on low fees
            )

        elif strategy == OptimizationStrategy.PROTOCOL_CENTRIC:
            # Maximize protocol robustness - for critical infrastructure
            return ObjectiveWeights(
                user_experience_weight=0.20,      # Basic UX
                protocol_stability_weight=0.50,   # Maximum safety
                economic_efficiency_weight=0.20,  # Important for sustainability
                production_readiness_weight=0.10, # Implementation matters
                vault_robustness_emphasis=1.5     # Extra safety focus
            )

        elif strategy == OptimizationStrategy.BALANCED:
            # Balanced optimization - for general deployment
            return ObjectiveWeights(
                user_experience_weight=0.40,
                protocol_stability_weight=0.35,
                economic_efficiency_weight=0.15,
                production_readiness_weight=0.10
            )

        elif strategy == OptimizationStrategy.LAUNCH_SAFE:
            # Conservative parameters for protocol launch
            return ObjectiveWeights(
                user_experience_weight=0.25,
                protocol_stability_weight=0.45,   # Extra safety
                economic_efficiency_weight=0.10,
                production_readiness_weight=0.20, # Implementation critical
                vault_robustness_emphasis=2.0     # Maximum safety
            )

        elif strategy == OptimizationStrategy.CRISIS_READY:
            # Optimize for extreme market conditions
            return ObjectiveWeights(
                user_experience_weight=0.30,      # UX matters but not primary
                protocol_stability_weight=0.40,   # Crisis resilience
                economic_efficiency_weight=0.15,
                production_readiness_weight=0.05,
                adversarial_robustness_weight=0.10 # Anti-attack measures
            )

        elif strategy == OptimizationStrategy.CAPITAL_EFFICIENT:
            # Minimize capital requirements while maintaining safety
            return ObjectiveWeights(
                user_experience_weight=0.35,
                protocol_stability_weight=0.30,
                economic_efficiency_weight=0.25,  # Higher efficiency focus
                production_readiness_weight=0.10,
                cost_recovery_emphasis=1.5        # Capital efficiency
            )

        elif strategy == OptimizationStrategy.MEV_RESISTANT:
            # Maximize resistance to manipulation and attacks
            return ObjectiveWeights(
                user_experience_weight=0.25,
                protocol_stability_weight=0.35,
                economic_efficiency_weight=0.15,
                production_readiness_weight=0.10,
                adversarial_robustness_weight=0.15 # Direct anti-MEV focus
            )

        else:
            # Default to balanced
            return ObjectiveWeights()


class MultiObjectiveFunction:
    """
    Multi-objective function implementation for Pareto optimization.

    This class provides both single-objective (weighted sum) and true multi-objective
    formulations for different optimization algorithms.
    """

    def __init__(self, weights: ObjectiveWeights,
                 constraints: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize objective function with weights and constraints.

        Args:
            weights: Objective weight configuration
            constraints: Optional constraints as {metric_name: (min_val, max_val)}
        """
        self.weights = weights
        self.constraints = constraints or {}

    def calculate_weighted_objective(self, metrics: EnhancedMechanismMetrics) -> float:
        """
        Calculate single weighted objective score for single-objective optimization.

        Returns:
            Single objective value (higher is better)
        """

        # Primary UX objective with internal emphasis weights
        ux_score = metrics.user_experience_composite
        if hasattr(self.weights, 'fee_affordability_emphasis'):
            # Adjust for fee affordability emphasis
            ux_adjustment = (metrics.fee_affordability_score *
                           (self.weights.fee_affordability_emphasis - 1.0) * 0.2)
            ux_score = min(1.0, ux_score + ux_adjustment)

        # Protocol stability with emphasis
        stability_score = metrics.protocol_stability_composite
        if hasattr(self.weights, 'vault_robustness_emphasis'):
            stability_adjustment = (metrics.vault_robustness_score *
                                  (self.weights.vault_robustness_emphasis - 1.0) * 0.2)
            stability_score = min(1.0, stability_score + stability_adjustment)

        # Economic efficiency with emphasis
        efficiency_score = metrics.economic_efficiency_composite
        if hasattr(self.weights, 'cost_recovery_emphasis'):
            efficiency_adjustment = (metrics.cost_recovery_ratio *
                                   (self.weights.cost_recovery_emphasis - 1.0) * 0.2)
            efficiency_score = min(1.0, efficiency_score + efficiency_adjustment)

        # Production readiness
        production_score = metrics.production_readiness_composite

        # Adversarial robustness (if weighted)
        robustness_score = 0
        if self.weights.adversarial_robustness_weight > 0:
            robustness_score = (
                metrics.mev_attack_resistance * 0.4 +
                metrics.extreme_volatility_survival * 0.3 +
                metrics.demand_shock_resilience * 0.3
            )

        # Weighted combination
        weighted_objective = (
            self.weights.user_experience_weight * ux_score +
            self.weights.protocol_stability_weight * stability_score +
            self.weights.economic_efficiency_weight * efficiency_score +
            self.weights.production_readiness_weight * production_score +
            self.weights.adversarial_robustness_weight * robustness_score
        )

        return weighted_objective

    def calculate_pareto_objectives(self, metrics: EnhancedMechanismMetrics) -> Tuple[float, float, float]:
        """
        Calculate multiple objectives for Pareto optimization.

        Returns:
            Tuple of (UX_objective, Stability_objective, Efficiency_objective)
            All values are negated for minimization algorithms.
        """
        return metrics.get_pareto_objectives()

    def evaluate_constraints(self, metrics: EnhancedMechanismMetrics) -> Dict[str, bool]:
        """
        Evaluate constraint satisfaction.

        Returns:
            Dictionary of {constraint_name: is_satisfied}
        """
        violations = {}

        for metric_name, (min_val, max_val) in self.constraints.items():
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)
                violations[metric_name] = min_val <= value <= max_val
            else:
                warnings.warn(f"Constraint metric '{metric_name}' not found in metrics")
                violations[metric_name] = False

        return violations

    def is_feasible(self, metrics: EnhancedMechanismMetrics) -> bool:
        """
        Check if a solution satisfies all constraints.

        Returns:
            True if all constraints are satisfied
        """
        if not self.constraints:
            return True

        violations = self.evaluate_constraints(metrics)
        return all(violations.values())


class ProductionObjectiveFunction(MultiObjectiveFunction):
    """
    Production-ready objective function with real-world constraints.

    This extends the base objective function with constraints relevant for
    actual protocol deployment.
    """

    def __init__(self, weights: ObjectiveWeights):
        """
        Initialize with production-appropriate constraints.
        """

        # Production constraints based on protocol requirements
        production_constraints = {
            # Critical safety constraints
            'vault_insolvency_risk': (0.0, 0.05),        # Max 5% insolvency risk
            'vault_robustness_score': (0.8, 1.0),        # Min 80% robustness
            'crisis_resilience_score': (0.7, 1.0),       # Min 70% crisis resilience

            # User experience constraints
            'fee_affordability_score': (0.3, 1.0),       # Reasonable affordability
            'fee_predictability_score': (0.6, 1.0),      # Min 60% predictability
            'fee_rate_of_change_p95': (0.0, 0.5),        # Max 50% rate of change

            # Economic viability constraints
            'cost_recovery_ratio': (0.8, 1.2),           # 80-120% cost recovery
            'user_cost_burden': (0.0, 0.1),              # Max 10% user cost burden

            # Implementation constraints
            'implementation_complexity': (0.5, 1.0),      # Reasonably simple
            'backwards_compatibility_score': (0.3, 1.0),  # Some compatibility required
        }

        super().__init__(weights, production_constraints)

    def calculate_deployment_risk_score(self, metrics: EnhancedMechanismMetrics) -> float:
        """
        Calculate overall deployment risk based on constraint violations.

        Returns:
            Risk score from 0.0 (no risk) to 1.0 (high risk)
        """
        violations = self.evaluate_constraints(metrics)
        violation_count = sum(1 for satisfied in violations.values() if not satisfied)
        total_constraints = len(violations)

        if total_constraints == 0:
            return 0.0

        # Risk increases with constraint violations
        violation_ratio = violation_count / total_constraints

        # Additional risk factors
        risk_factors = []

        # High fee volatility increases deployment risk
        if metrics.fee_rate_of_change_p95 > 0.3:
            risk_factors.append(0.2)

        # Low vault robustness increases deployment risk
        if metrics.vault_robustness_score < 0.9:
            risk_factors.append(0.15)

        # Poor crisis resilience increases deployment risk
        if metrics.crisis_resilience_score < 0.8:
            risk_factors.append(0.15)

        # High implementation complexity increases deployment risk
        if metrics.implementation_complexity < 0.7:
            risk_factors.append(0.1)

        total_risk = violation_ratio + sum(risk_factors)
        return min(1.0, total_risk)


class ObjectiveFunctionSuite:
    """
    Complete suite of objective functions for comprehensive analysis.

    This class provides all standard objective functions and allows for
    easy comparison across different optimization strategies.
    """

    def __init__(self):
        """Initialize the complete suite of objective functions."""
        self.strategies = {
            strategy: ObjectiveFunctionFactory.get_strategy_weights(strategy)
            for strategy in OptimizationStrategy
        }

        self.objective_functions = {
            name: MultiObjectiveFunction(weights)
            for name, weights in self.strategies.items()
        }

        # Add production-ready versions
        self.production_functions = {
            f"{name}_production": ProductionObjectiveFunction(weights)
            for name, weights in self.strategies.items()
        }

    def evaluate_all_strategies(self, metrics: EnhancedMechanismMetrics) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a solution across all optimization strategies.

        Returns:
            Dictionary of {strategy_name: {metric_type: value}}
        """
        results = {}

        # Standard strategies
        for name, obj_func in self.objective_functions.items():
            strategy_name = name.value if hasattr(name, 'value') else str(name)
            results[strategy_name] = {
                'weighted_objective': obj_func.calculate_weighted_objective(metrics),
                'is_feasible': obj_func.is_feasible(metrics)
            }

            pareto_objectives = obj_func.calculate_pareto_objectives(metrics)
            results[strategy_name].update({
                'ux_objective': pareto_objectives[0],
                'stability_objective': pareto_objectives[1],
                'efficiency_objective': pareto_objectives[2]
            })

        # Production strategies
        for name, obj_func in self.production_functions.items():
            results[name] = {
                'weighted_objective': obj_func.calculate_weighted_objective(metrics),
                'is_feasible': obj_func.is_feasible(metrics),
                'deployment_risk': obj_func.calculate_deployment_risk_score(metrics)
            }

        return results

    def recommend_strategy(self, deployment_phase: str = "mainnet") -> OptimizationStrategy:
        """
        Recommend optimization strategy based on deployment phase.

        Args:
            deployment_phase: "testnet", "mainnet_launch", "mainnet_mature"

        Returns:
            Recommended optimization strategy
        """

        if deployment_phase == "testnet":
            return OptimizationStrategy.USER_CENTRIC  # Optimize for testing
        elif deployment_phase == "mainnet_launch":
            return OptimizationStrategy.LAUNCH_SAFE   # Conservative launch
        elif deployment_phase == "mainnet_mature":
            return OptimizationStrategy.BALANCED      # Balanced optimization
        else:
            return OptimizationStrategy.BALANCED      # Default


def create_custom_objective_function(
    user_weight: float = 0.4,
    stability_weight: float = 0.35,
    efficiency_weight: float = 0.15,
    production_weight: float = 0.1,
    constraints: Optional[Dict[str, Tuple[float, float]]] = None
) -> MultiObjectiveFunction:
    """
    Create a custom objective function with user-specified weights.

    Args:
        user_weight: Weight for user experience metrics
        stability_weight: Weight for protocol stability metrics
        efficiency_weight: Weight for economic efficiency metrics
        production_weight: Weight for production readiness metrics
        constraints: Optional constraint dictionary

    Returns:
        Custom MultiObjectiveFunction instance
    """

    weights = ObjectiveWeights(
        user_experience_weight=user_weight,
        protocol_stability_weight=stability_weight,
        economic_efficiency_weight=efficiency_weight,
        production_readiness_weight=production_weight
    )

    return MultiObjectiveFunction(weights, constraints)


# Example usage and testing functions

def demonstrate_objective_functions():
    """
    Demonstrate different objective function configurations.
    """
    print("Taiko Fee Mechanism Objective Functions")
    print("=" * 50)

    suite = ObjectiveFunctionSuite()

    for strategy in OptimizationStrategy:
        weights = ObjectiveFunctionFactory.get_strategy_weights(strategy)

        print(f"\n{strategy.value.upper().replace('_', ' ')} STRATEGY:")
        print(f"  User Experience: {weights.user_experience_weight:.1%}")
        print(f"  Protocol Stability: {weights.protocol_stability_weight:.1%}")
        print(f"  Economic Efficiency: {weights.economic_efficiency_weight:.1%}")
        print(f"  Production Readiness: {weights.production_readiness_weight:.1%}")
        if weights.adversarial_robustness_weight > 0:
            print(f"  Adversarial Robustness: {weights.adversarial_robustness_weight:.1%}")

    print(f"\nTotal objective functions available: {len(suite.objective_functions) + len(suite.production_functions)}")


if __name__ == "__main__":
    demonstrate_objective_functions()