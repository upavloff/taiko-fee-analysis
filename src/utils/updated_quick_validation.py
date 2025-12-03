#!/usr/bin/env python3
"""
Updated Quick Fee Mechanism Validation - Using Canonical Modules

Simple validation demonstrating the canonical fee mechanism modules.
This script provides a quick way to test the canonical implementations
without complex dependencies or long optimizations.

Key Features:
- Uses canonical_fee_mechanism.py for all calculations
- Demonstrates parameter validation
- Quick performance assessment
- Comparison with optimal research parameters
"""

import sys
import os
from typing import Dict, Any, List, Tuple

# Add project paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import canonical modules
from core.canonical_fee_mechanism import (
    CanonicalTaikoFeeCalculator,
    FeeParameters,
    VaultInitMode,
    create_default_calculator,
    get_optimal_parameters,
    validate_fee_parameters
)
from core.canonical_metrics import (
    calculate_basic_metrics,
    validate_metric_thresholds
)


def quick_fee_calculation_test():
    """
    Quick test of canonical fee calculation.
    """
    print("🧮 Quick Fee Calculation Test")
    print("-" * 40)

    # Test scenarios
    scenarios = [
        {"name": "Low L1 (5 gwei)", "l1_gwei": 5.0, "deficit": 0.0},
        {"name": "Medium L1 (20 gwei)", "l1_gwei": 20.0, "deficit": 50.0},
        {"name": "High L1 (100 gwei)", "l1_gwei": 100.0, "deficit": 100.0},
        {"name": "Crisis L1 (500 gwei)", "l1_gwei": 500.0, "deficit": 200.0},
    ]

    # Use optimal parameters from research
    calculator = create_default_calculator()

    print(f"Using optimal parameters: μ={calculator.params.mu}, ν={calculator.params.nu}, H={calculator.params.H}")
    print()
    print(f"{'Scenario':<20} {'L1 Cost':<12} {'Estimated Fee':<15} {'Fee (gwei)'}")
    print("-" * 60)

    for scenario in scenarios:
        l1_basefee_wei = int(scenario["l1_gwei"] * 1e9)
        vault_deficit = scenario["deficit"]

        # Calculate L1 cost and fee
        l1_cost = calculator.calculate_l1_cost_per_tx(l1_basefee_wei, apply_smoothing=False)
        estimated_fee = calculator.calculate_estimated_fee(l1_cost, vault_deficit)

        print(f"{scenario['name']:<20} {l1_cost*1e9:<12.3f} {estimated_fee:.6f} ETH  {estimated_fee*1e9:<15.3f}")

    print("-" * 60)
    print()


def quick_parameter_validation():
    """
    Quick validation of different parameter sets.
    """
    print("🔬 Quick Parameter Validation")
    print("-" * 40)

    # Test parameter sets
    test_parameters = [
        {"name": "Research Optimal", "mu": 0.0, "nu": 0.27, "H": 492},
        {"name": "Conservative", "mu": 0.0, "nu": 0.48, "H": 492},
        {"name": "Crisis Ready", "mu": 0.0, "nu": 0.88, "H": 120},
        {"name": "Invalid μ", "mu": 1.5, "nu": 0.3, "H": 144},
        {"name": "Invalid H", "mu": 0.2, "nu": 0.3, "H": 145},  # Not 6-aligned
    ]

    for params in test_parameters:
        is_valid = validate_fee_parameters(params["mu"], params["nu"], params["H"])
        status = "✅" if is_valid else "❌"
        print(f"{status} {params['name']:<20}: μ={params['mu']}, ν={params['nu']}, H={params['H']}")

    print()


def quick_simulation_test():
    """
    Quick simulation to test metrics calculation.
    """
    print("⚡ Quick Simulation Test")
    print("-" * 40)

    # Create calculator and vault
    calculator = create_default_calculator()
    vault = calculator.create_vault(VaultInitMode.DEFICIT, deficit_ratio=0.1)

    print(f"Initial vault: {vault.balance:.1f} ETH (target: {vault.target_balance:.1f} ETH)")

    # Generate simple L1 data (100 steps = ~3.3 minutes)
    simulation_steps = 100
    base_fee = 20e9  # 20 gwei
    l1_basefees = [base_fee * (1 + 0.1 * (i % 10 - 5) / 5) for i in range(simulation_steps)]

    # Run simple simulation
    results = {
        'timeStep': [],
        'l1Basefee': [],
        'estimatedFee': [],
        'transactionVolume': [],
        'vaultBalance': [],
        'vaultDeficit': [],
        'feesCollected': [],
        'l1CostsPaid': []
    }

    total_fees = 0.0
    total_costs = 0.0

    for step in range(simulation_steps):
        l1_basefee_wei = l1_basefees[step]

        # Calculate fee and volume
        l1_cost = calculator.calculate_l1_cost_per_tx(l1_basefee_wei)
        estimated_fee = calculator.calculate_estimated_fee(l1_cost, vault.deficit)
        tx_volume = calculator.calculate_transaction_volume(estimated_fee)

        # Vault operations
        fees_collected = estimated_fee * tx_volume
        vault.collect_fees(fees_collected)
        total_fees += fees_collected

        l1_costs_paid = 0.0
        if step % calculator.params.batch_interval_steps == 0:
            l1_costs_paid = calculator.calculate_l1_batch_cost(l1_basefee_wei)
            vault.pay_l1_costs(l1_costs_paid)
            total_costs += l1_costs_paid

        # Record results
        results['timeStep'].append(step)
        results['l1Basefee'].append(l1_basefee_wei / 1e9)
        results['estimatedFee'].append(estimated_fee)
        results['transactionVolume'].append(tx_volume)
        results['vaultBalance'].append(vault.balance)
        results['vaultDeficit'].append(vault.deficit)
        results['feesCollected'].append(fees_collected)
        results['l1CostsPaid'].append(l1_costs_paid)

    # Calculate basic metrics
    metrics = calculate_basic_metrics(results)

    print(f"Simulation results ({simulation_steps} steps):")
    print(f"   Average fee: {metrics['average_fee_gwei']:.3f} gwei")
    print(f"   Fee stability (CV): {metrics['fee_stability_cv']:.3f}")
    print(f"   Time underfunded: {metrics['time_underfunded_pct']:.1f}%")
    print(f"   L1 tracking error: {metrics['l1_tracking_error']:.3f}")
    print(f"   Overall score: {metrics['overall_score']:.3f}")

    # Financial summary
    net_revenue = total_fees - total_costs
    cost_coverage = total_fees / total_costs if total_costs > 0 else float('inf')

    print(f"Financial summary:")
    print(f"   Total fees collected: {total_fees:.3f} ETH")
    print(f"   Total L1 costs: {total_costs:.3f} ETH")
    print(f"   Net revenue: {net_revenue:.3f} ETH")
    print(f"   Cost coverage ratio: {cost_coverage:.2f}")

    # Validate against thresholds
    grades = validate_metric_thresholds(
        average_fee_gwei=metrics['average_fee_gwei'],
        fee_cv=metrics['fee_stability_cv'],
        underfunded_pct=metrics['time_underfunded_pct'],
        tracking_error=metrics['l1_tracking_error']
    )

    print(f"Performance grades:")
    for metric, grade in grades.items():
        emoji = {"excellent": "🟢", "good": "🟡", "poor": "🔴"}.get(grade, "⚪")
        print(f"   {emoji} {metric.replace('_', ' ').title()}: {grade}")

    print()


def compare_with_research_findings():
    """
    Compare current canonical implementation with research findings.
    """
    print("🔬 Research Findings Comparison")
    print("-" * 40)

    optimal_params = get_optimal_parameters()

    print(f"Research-validated optimal parameters:")
    print(f"   μ = {optimal_params['mu']} (L1 weight)")
    print(f"   ν = {optimal_params['nu']} (deficit weight)")
    print(f"   H = {optimal_params['H']} (prediction horizon)")
    print(f"   Description: {optimal_params['description']}")
    print()

    # Test with typical conditions
    l1_basefee_wei = int(20 * 1e9)  # 20 gwei
    vault_deficit = 50.0  # 50 ETH

    calculator = create_default_calculator()
    l1_cost = calculator.calculate_l1_cost_per_tx(l1_basefee_wei)
    estimated_fee = calculator.calculate_estimated_fee(l1_cost, vault_deficit)

    print(f"Fee calculation with optimal parameters:")
    print(f"   L1 basefee: 20.0 gwei")
    print(f"   Vault deficit: 50.0 ETH")
    print(f"   Calculated L1 cost: {l1_cost*1e9:.3f} gwei per tx")
    print(f"   Estimated fee: {estimated_fee*1e9:.3f} gwei")
    print()

    # Check key insights from research
    mu_zero_insight = optimal_params['mu'] == 0.0
    batch_alignment = optimal_params['H'] % 6 == 0
    reasonable_nu = 0.1 <= optimal_params['nu'] <= 0.9

    print(f"Research insights validation:")
    print(f"   {'✅' if mu_zero_insight else '❌'} μ=0.0 (optimal L1 correlation bias elimination)")
    print(f"   {'✅' if batch_alignment else '❌'} H is 6-step aligned (batch cycle resonance)")
    print(f"   {'✅' if reasonable_nu else '❌'} ν in reasonable range (0.1-0.9)")
    print()


def canonical_architecture_demo():
    """
    Demonstrate canonical architecture benefits.
    """
    print("🏗️  Canonical Architecture Benefits")
    print("-" * 40)

    print("✅ Single Source of Truth:")
    print("   - canonical_fee_mechanism.py: Authoritative fee calculations")
    print("   - canonical_metrics.py: Consistent performance metrics")
    print("   - canonical_optimization.py: NSGA-II optimization")
    print()

    print("✅ Guaranteed Consistency:")
    print("   - Python analysis scripts use canonical modules")
    print("   - JavaScript web interface calls Python API")
    print("   - All components reference same implementations")
    print()

    print("✅ Easy Maintenance:")
    print("   - Changes in one place propagate everywhere")
    print("   - No code duplication to maintain")
    print("   - Clear modular boundaries")
    print()

    print("✅ Scientific Rigor:")
    print("   - Research parameters embedded in code")
    print("   - Validation functions for parameter checking")
    print("   - Comprehensive metrics for analysis")
    print()


def main():
    """
    Run all quick validation tests.
    """
    print("🚀 Quick Canonical Fee Mechanism Validation")
    print("=" * 60)
    print()

    try:
        # 1. Quick fee calculation test
        quick_fee_calculation_test()

        # 2. Parameter validation
        quick_parameter_validation()

        # 3. Quick simulation
        quick_simulation_test()

        # 4. Research comparison
        compare_with_research_findings()

        # 5. Architecture demo
        canonical_architecture_demo()

        print("🎉 All quick validation tests completed successfully!")
        print()
        print("Next steps:")
        print("   - Run full optimization: python src/analysis/updated_comprehensive_optimization.py")
        print("   - Start API server: python src/api/server.py")
        print("   - Detailed analysis: python src/analysis/updated_alpha_validation_demo.py")

        return 0

    except Exception as e:
        print(f"❌ Quick validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)