#!/usr/bin/env python3
"""
Find Optimal SPECS Parameters

Quick optimization script to find the best parameters using SPECS.md formulas
and real historical data.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add python directory to path
sys.path.append('python')

from specs_implementation.core.simulation_engine import SimulationEngine
from specs_implementation.metrics.calculator import MetricsCalculator
from specs_implementation.metrics.constraints import SimulationResults

def load_crisis_data():
    """Load Luna crash data for optimization"""
    file_path = "data_cache/luna_crash_true_peak_contiguous.csv"

    if not os.path.exists(file_path):
        print(f"❌ Cannot find {file_path}")
        return None

    df = pd.read_csv(file_path)
    basefee_wei = df['basefee_wei'].values

    # Convert to L1 costs
    batch_gas = 200_000
    txs_per_batch = 100
    gas_per_tx = max(batch_gas / txs_per_batch, 200)
    l1_costs = (basefee_wei * gas_per_tx) / 1e18

    # Use first 50 points for fast optimization
    return l1_costs[:50]

def evaluate_parameters(mu, nu, H, l1_costs):
    """Evaluate a parameter set"""
    try:
        # Create simulation
        engine = SimulationEngine(
            mu=mu, nu=nu, horizon_h=H,
            target_vault_balance=1000.0,
            initial_vault_balance=1000.0
        )

        # Run simulation
        sim_df = engine.simulate_series(l1_costs)

        # Convert to metrics format
        sim_results = SimulationResults(
            timestamps=np.arange(len(sim_df)),
            vault_balances=sim_df['vault_balance_after'].values,
            fees_per_gas=sim_df['basefee_per_gas'].values,
            l1_costs=sim_df['l1_cost_actual'].values,
            revenues=sim_df['revenue'].values,
            subsidies=sim_df['subsidy_paid'].values,
            deficits=sim_df['deficit_after'].values,
            Q_bar=6.9e5
        )

        # Calculate comprehensive metrics
        metrics_calc = MetricsCalculator()
        comprehensive_metrics = metrics_calc.evaluate_parameter_set(sim_results)

        # Get basic simulation metrics
        basic_metrics = engine.calculate_metrics(sim_df)

        return {
            'feasible': comprehensive_metrics.is_feasible,
            'ux_score': comprehensive_metrics.ux_score,
            'robustness_score': comprehensive_metrics.robustness_score,
            'capital_efficiency': comprehensive_metrics.capital_efficiency_score,
            'avg_fee_gwei': basic_metrics['avg_fee_gwei'],
            'cost_recovery_ratio': comprehensive_metrics.constraint_results.cost_recovery_ratio,
            'min_vault': basic_metrics['min_vault_balance'],
            'violations': len(comprehensive_metrics.constraint_results.violations)
        }

    except Exception as e:
        return {'error': str(e)}

def main():
    """Find optimal parameters"""
    print("🔧 Finding Optimal SPECS Parameters")
    print("="*50)

    # Load data
    print("📊 Loading crisis data...")
    l1_costs = load_crisis_data()
    if l1_costs is None:
        return

    print(f"✅ Loaded {len(l1_costs)} data points")
    print(f"   L1 cost range: {np.min(l1_costs):.6f} - {np.max(l1_costs):.6f} ETH")

    # Test parameter ranges (more focused)
    print("\n🎯 Testing parameter combinations...")

    # Based on CLAUDE.md optimal parameters
    mu_values = [0.0, 0.1, 0.2]  # L1 weight
    nu_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]  # Deficit weight
    H_values = [36, 72, 144, 288]  # Horizon

    best_results = []

    total_combinations = len(mu_values) * len(nu_values) * len(H_values)
    combination = 0

    for mu in mu_values:
        for nu in nu_values:
            for H in H_values:
                combination += 1
                print(f"[{combination}/{total_combinations}] Testing μ={mu}, ν={nu}, H={H}")

                result = evaluate_parameters(mu, nu, H, l1_costs)

                if 'error' in result:
                    print(f"  ❌ Error: {result['error']}")
                    continue

                result.update({'mu': mu, 'nu': nu, 'H': H})

                # Show key metrics
                feasible = "✅" if result['feasible'] else "❌"
                print(f"  {feasible} Fee: {result['avg_fee_gwei']:.3f} gwei, "
                      f"CR: {result['cost_recovery_ratio']:.2f}, "
                      f"MinV: {result['min_vault']:.0f} ETH, "
                      f"Violations: {result['violations']}")

                if result['feasible']:
                    best_results.append(result)

    # Analyze results
    print(f"\n📊 Optimization Results:")
    print(f"   Total tested: {total_combinations}")
    print(f"   Feasible: {len(best_results)}")

    if best_results:
        # Sort by UX score (lower is better)
        best_ux = sorted(best_results, key=lambda x: x['ux_score'])[:3]

        print(f"\n🏆 Top 3 UX Parameters:")
        for i, result in enumerate(best_ux):
            print(f"   {i+1}. μ={result['mu']}, ν={result['nu']}, H={result['H']}")
            print(f"      UX Score: {result['ux_score']:.3f}, Fee: {result['avg_fee_gwei']:.3f} gwei")
            print(f"      Cost Recovery: {result['cost_recovery_ratio']:.3f}")

        # Check current optimal vs best found
        print(f"\n📋 Current vs Optimal Comparison:")
        current_optimal = {'mu': 0.0, 'nu': 0.1, 'H': 36}
        current_result = next((r for r in best_results if
                              r['mu'] == current_optimal['mu'] and
                              r['nu'] == current_optimal['nu'] and
                              r['H'] == current_optimal['H']), None)

        if current_result:
            print(f"   Current optimal (μ=0.0, ν=0.1, H=36):")
            print(f"     UX Score: {current_result['ux_score']:.3f}")
            print(f"     Avg Fee: {current_result['avg_fee_gwei']:.3f} gwei")
            print(f"     Rank: #{best_ux.index(current_result)+1 if current_result in best_ux else 'N/A'}")

        best_found = best_ux[0]
        print(f"\n   Best found (μ={best_found['mu']}, ν={best_found['nu']}, H={best_found['H']}):")
        print(f"     UX Score: {best_found['ux_score']:.3f}")
        print(f"     Avg Fee: {best_found['avg_fee_gwei']:.3f} gwei")

        if current_result and best_found != current_result:
            improvement = ((current_result['ux_score'] - best_found['ux_score']) / current_result['ux_score']) * 100
            print(f"     Improvement: {improvement:.1f}% better UX score")

    else:
        print("❌ No feasible parameters found!")
        print("💡 Consider adjusting constraint thresholds or target vault size")

if __name__ == "__main__":
    main()