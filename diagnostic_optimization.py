"""
Diagnostic script to understand why optimization is not finding feasible solutions.
"""

import sys
sys.path.append('src')

from src.core.theoretical_optimization import TheoreticalOptimizer, StakeholderProfile, TheoreticalObjectiveWeights
from src.core.canonical_fee_mechanism import CanonicalTaikoFeeCalculator, FeeParameters, VaultInitMode
import numpy as np

def diagnose_constraints():
    """Diagnose constraint feasibility with known good parameters."""

    print("🔍 CONSTRAINT FEASIBILITY DIAGNOSIS")
    print("=" * 50)

    optimizer = TheoreticalOptimizer(epsilon_crr=0.10, epsilon_ruin=0.05)  # Relaxed constraints

    # Test with known good parameters from CLAUDE.md
    test_cases = [
        {"name": "Optimal", "mu": 0.0, "nu": 0.27, "H": 492, "lambda_B": 0.365, "T": 1000.0},
        {"name": "Balanced", "mu": 0.0, "nu": 0.48, "H": 492, "lambda_B": 0.365, "T": 1000.0},
        {"name": "Crisis", "mu": 0.0, "nu": 0.88, "H": 120, "lambda_B": 0.365, "T": 1000.0},
        {"name": "High μ", "mu": 0.5, "nu": 0.3, "H": 300, "lambda_B": 0.365, "T": 1000.0},
        {"name": "Conservative", "mu": 0.1, "nu": 0.1, "H": 600, "lambda_B": 0.365, "T": 1500.0},
    ]

    # Load scenarios for testing
    scenario_data = optimizer._load_historical_scenarios()

    for case in test_cases:
        print(f"\n📊 Testing {case['name']} parameters:")
        print(f"   μ={case['mu']}, ν={case['nu']}, H={case['H']}, T={case['T']}")

        # Create individual
        individual = {
            'mu': case['mu'],
            'nu': case['nu'],
            'H': case['H'],
            'lambda_B': case['lambda_B'],
            'T': case['T']
        }

        try:
            # Evaluate metrics
            metrics = optimizer._evaluate_individual(individual, scenario_data)

            # Check constraints
            constraint_violation = optimizer._calculate_constraint_violation(metrics)

            print(f"   CRR: {metrics.cost_recovery_ratio:.3f} (target: 0.90-1.10)")
            print(f"   Ruin Prob: {metrics.ruin_probability:.3f} (target: <0.05)")
            print(f"   Median Fee: {metrics.median_fee * 1e9:.2f} gwei")
            print(f"   Fee CV: {metrics.fee_cv:.3f}")
            print(f"   Constraint Violation: {constraint_violation:.3f}")
            print(f"   Feasible: {'✅' if constraint_violation == 0 else '❌'}")

        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")

def test_simplified_optimization():
    """Run a simplified optimization with relaxed constraints."""

    print("\n\n🎯 SIMPLIFIED OPTIMIZATION TEST")
    print("=" * 50)

    # Very relaxed constraints
    optimizer = TheoreticalOptimizer(epsilon_crr=0.15, epsilon_ruin=0.10)

    # Override bounds to known good regions
    optimizer.bounds = {
        'mu': (0.0, 0.5),      # Focus on low μ region
        'nu': (0.1, 0.8),      # Focus on moderate ν region
        'H': (120, 600),       # Focus on practical H range
        'lambda_B': (0.3, 0.4), # Keep λ_B close to optimal
        'T': (800, 1200)       # Reasonable target range
    }

    # Small test with balanced profile
    print("Running mini-optimization for balanced profile...")

    results = optimizer.optimize_for_profile(
        profile=StakeholderProfile.PROTOCOL_DAO,
        population_size=20,
        generations=10
    )

    print(f"Runtime: {results['optimization_time']:.1f}s")
    print(f"Best solutions found: {len(results['best_solutions'])}")

    if results['best_solutions']:
        best = results['best_solutions'][-1]
        params = best['parameters']
        metrics = best['metrics']

        print(f"\n🏆 Best solution found:")
        print(f"   Parameters: μ={params['mu']:.3f}, ν={params['nu']:.3f}, H={params['H']}")
        print(f"   CRR: {metrics.cost_recovery_ratio:.3f}")
        print(f"   Ruin Prob: {metrics.ruin_probability:.3f}")
        print(f"   Fee: {metrics.median_fee * 1e9:.2f} gwei (CV: {metrics.fee_cv:.3f})")
        print(f"   Objectives: {[f'{obj:.3f}' for obj in best['objectives']]}")

    return results

def analyze_objective_sensitivity():
    """Analyze how objective weights affect parameter selection."""

    print("\n\n⚖️ OBJECTIVE WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 50)

    # Test extreme weight profiles
    weight_profiles = [
        ("UX Focused", {"a1_stability": 5.0, "a2_jumpiness": 3.0, "a3_high_fees": 10.0}),
        ("Safety Focused", {"b1_deficit_duration": 5.0, "b2_max_deficit": 5.0, "b3_recovery_time": 3.0}),
        ("Efficiency Focused", {"c1_target_size": 5.0, "c2_vault_deviation": 3.0, "c3_capital_efficiency": 5.0}),
    ]

    optimizer = TheoreticalOptimizer(epsilon_crr=0.15, epsilon_ruin=0.10)
    scenario_data = optimizer._load_historical_scenarios()

    # Test parameter
    test_param = {'mu': 0.2, 'nu': 0.4, 'H': 300, 'lambda_B': 0.365, 'T': 1000.0}

    for profile_name, weight_overrides in weight_profiles:
        print(f"\n📊 {profile_name}:")

        # Create custom weights
        weights = TheoreticalObjectiveWeights()
        for param, value in weight_overrides.items():
            setattr(weights, param, value)

        # Evaluate with these weights
        metrics = optimizer._evaluate_individual(test_param, scenario_data)
        objectives = optimizer._calculate_objectives(metrics, weights)

        print(f"   Objectives: UX={objectives[0]:.3f}, Safety={objectives[1]:.3f}, Efficiency={objectives[2]:.3f}")
        print(f"   Total: {sum(objectives):.3f}")

if __name__ == "__main__":
    # Run diagnostics
    diagnose_constraints()
    results = test_simplified_optimization()
    analyze_objective_sensitivity()

    print("\n🎯 DIAGNOSIS COMPLETE")
    print("Key findings will guide constraint tuning for full optimization.")