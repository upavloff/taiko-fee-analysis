#!/usr/bin/env python3
"""
Find Optimal SPECS Parameters

Comprehensive optimization to find the best μ ∈ [0,1], ν ∈ [0,1], H parameters
using SPECS.md formulas with relaxed but realistic constraints.
"""

import sys
import os
import numpy as np
import pandas as pd
from itertools import product

# Add python directory to path
sys.path.append('python')

from specs_implementation.core.simulation_engine import SimulationEngine
from specs_implementation.metrics.constraints import ConstraintEvaluator, ConstraintThresholds, SimulationResults
from specs_implementation.metrics.objectives import ObjectiveCalculator, ObjectiveWeights, StakeholderProfile
from specs_implementation.metrics.calculator import MetricsCalculator

def load_optimization_datasets():
    """Load multiple datasets for robust optimization"""
    datasets = {
        'luna_crash': 'data_cache/luna_crash_true_peak_contiguous.csv',
        'july_spike': 'data_cache/real_july_2022_spike_data.csv',
        'pepe_crisis': 'data_cache/may_2023_pepe_crisis_data.csv',
        'low_fees': 'data_cache/recent_low_fees_3hours.csv'
    }

    loaded_data = {}

    for name, file_path in datasets.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                basefee_wei = df['basefee_wei'].values

                # Convert to L1 costs (ETH per transaction)
                batch_gas = 200_000
                txs_per_batch = 100
                gas_per_tx = max(batch_gas / txs_per_batch, 200)
                l1_costs = (basefee_wei * gas_per_tx) / 1e18

                # Use subset for faster optimization
                loaded_data[name] = l1_costs[:50]  # First 50 points
                print(f"✅ Loaded {name}: {len(loaded_data[name])} points, range {np.min(l1_costs):.6f}-{np.max(l1_costs):.6f} ETH")

            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")

    return loaded_data

def evaluate_parameter_set(mu, nu, H, datasets):
    """Evaluate parameter set across multiple datasets"""

    # Relaxed constraints for realistic optimization
    relaxed_thresholds = ConstraintThresholds(
        V_min=0.0,                  # No negative vault balance
        epsilon_insolvency=0.10,    # 10% insolvency risk tolerance
        delta_cr=0.30,              # ±30% cost recovery tolerance
        F_max_UX=200e9,             # 200 gwei max UX fee
    )

    constraint_evaluator = ConstraintEvaluator(relaxed_thresholds)
    objective_calculator = ObjectiveCalculator()

    results = []
    total_violations = 0

    for dataset_name, l1_costs in datasets.items():
        try:
            # Create simulation engine
            engine = SimulationEngine(
                mu=mu, nu=nu, horizon_h=H,
                target_vault_balance=1000.0,
                initial_vault_balance=1000.0
            )

            # Run simulation
            sim_df = engine.simulate_series(l1_costs)

            # Convert to SimulationResults format
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

            # Evaluate constraints
            constraint_results = constraint_evaluator.evaluate_all_constraints(sim_results)

            # Calculate objectives
            objective_results = objective_calculator.calculate_all_objectives(sim_results, V_min=0.0)

            # Get basic metrics
            basic_metrics = engine.calculate_metrics(sim_df)

            dataset_result = {
                'dataset': dataset_name,
                'feasible': constraint_results.is_feasible,
                'violations': len(constraint_results.violations),
                'avg_fee_gwei': basic_metrics['avg_fee_gwei'],
                'fee_cv': basic_metrics['fee_cv'],
                'min_vault': basic_metrics['min_vault_balance'],
                'cost_recovery_ratio': constraint_results.cost_recovery_ratio,
                'insolvency_prob': constraint_results.insolvency_probability,
                'ux_score': objective_results.ux_objective,
                'robustness_score': objective_results.robustness_objective,
                'capital_efficiency': objective_results.capital_efficiency_objective
            }

            results.append(dataset_result)
            total_violations += dataset_result['violations']

        except Exception as e:
            # Penalize failed simulations heavily
            results.append({
                'dataset': dataset_name,
                'feasible': False,
                'violations': 999,
                'error': str(e)[:50],
                'avg_fee_gwei': 999.0,
                'ux_score': 999.0,
                'robustness_score': 999.0,
                'capital_efficiency': 999.0
            })
            total_violations += 999

    # Aggregate results across datasets
    feasible_results = [r for r in results if r.get('feasible', False)]

    if len(feasible_results) == 0:
        # No feasible results
        return {
            'overall_feasible': False,
            'total_violations': total_violations,
            'avg_fee_gwei': 999.0,
            'avg_ux_score': 999.0,
            'avg_robustness_score': 999.0,
            'avg_capital_efficiency': 999.0,
            'feasible_datasets': 0,
            'results': results
        }

    # Calculate averages across feasible datasets
    avg_fee = np.mean([r['avg_fee_gwei'] for r in feasible_results])
    avg_ux_score = np.mean([r['ux_score'] for r in feasible_results])
    avg_robustness = np.mean([r['robustness_score'] for r in feasible_results])
    avg_capital_eff = np.mean([r['capital_efficiency'] for r in feasible_results])

    return {
        'overall_feasible': len(feasible_results) >= len(datasets) * 0.75,  # 75% datasets must be feasible
        'total_violations': total_violations,
        'avg_fee_gwei': avg_fee,
        'avg_ux_score': avg_ux_score,
        'avg_robustness_score': avg_robustness,
        'avg_capital_efficiency': avg_capital_eff,
        'feasible_datasets': len(feasible_results),
        'results': results
    }

def main():
    """Find optimal parameters with comprehensive search"""
    print("🎯 Finding Optimal SPECS Parameters")
    print("   μ ∈ [0,1], ν ∈ [0,1], H ∈ {36,72,144,288}")
    print("="*60)

    # Load datasets
    print("📊 Loading optimization datasets...")
    datasets = load_optimization_datasets()

    if not datasets:
        print("❌ No datasets loaded!")
        return

    print(f"✅ Loaded {len(datasets)} datasets for optimization")

    # Parameter grid - comprehensive search within bounds
    mu_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    nu_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    H_values = [36, 72, 144, 288]  # 6-step aligned

    total_combinations = len(mu_values) * len(nu_values) * len(H_values)
    print(f"\n🔧 Testing {total_combinations} parameter combinations...")

    all_results = []
    feasible_results = []

    combination_count = 0

    for mu, nu, H in product(mu_values, nu_values, H_values):
        combination_count += 1
        print(f"[{combination_count:3d}/{total_combinations}] μ={mu:.1f}, ν={nu:.2f}, H={H:3d}", end="")

        result = evaluate_parameter_set(mu, nu, H, datasets)
        result.update({'mu': mu, 'nu': nu, 'H': H})
        all_results.append(result)

        if result['overall_feasible']:
            feasible_results.append(result)
            status = "✅"
        else:
            status = "❌"

        print(f" {status} Fee:{result['avg_fee_gwei']:.1f}gwei, "
              f"Feasible:{result['feasible_datasets']}/{len(datasets)}, "
              f"UX:{result['avg_ux_score']:.2f}")

    # Analyze results
    print(f"\n📊 Optimization Results:")
    print(f"   Total combinations: {len(all_results)}")
    print(f"   Overall feasible: {len(feasible_results)}")
    print(f"   Success rate: {len(feasible_results)/len(all_results)*100:.1f}%")

    if feasible_results:
        # Sort by different criteria
        best_ux = sorted(feasible_results, key=lambda x: x['avg_ux_score'])[:5]
        best_robustness = sorted(feasible_results, key=lambda x: x['avg_robustness_score'])[:3]
        best_capital = sorted(feasible_results, key=lambda x: x['avg_capital_efficiency'])[:3]

        print(f"\n🏆 TOP 5 PARAMETERS BY UX SCORE:")
        print(f"   Rank | μ    ν    H  | UX Score | Avg Fee | Robust | CapEff | Datasets")
        print(f"   -----|------------|----------|---------|--------|--------|----------")
        for i, result in enumerate(best_ux):
            print(f"   {i+1:4d} | {result['mu']:.1f}  {result['nu']:.2f} {result['H']:3d} | "
                  f"{result['avg_ux_score']:8.3f} | {result['avg_fee_gwei']:7.2f} | "
                  f"{result['avg_robustness_score']:6.1f} | {result['avg_capital_efficiency']:6.1e} | "
                  f"{result['feasible_datasets']}/{len(datasets)}")

        print(f"\n🛡️ TOP 3 BY ROBUSTNESS:")
        for i, result in enumerate(best_robustness):
            print(f"   {i+1}. μ={result['mu']:.1f}, ν={result['nu']:.2f}, H={result['H']:3d} "
                  f"- Robustness: {result['avg_robustness_score']:.3f}")

        print(f"\n💰 TOP 3 BY CAPITAL EFFICIENCY:")
        for i, result in enumerate(best_capital):
            print(f"   {i+1}. μ={result['mu']:.1f}, ν={result['nu']:.2f}, H={result['H']:3d} "
                  f"- CapEff: {result['avg_capital_efficiency']:.2e}")

        # Recommended parameters
        recommended = best_ux[0]
        print(f"\n🎯 RECOMMENDED OPTIMAL PARAMETERS:")
        print(f"   μ = {recommended['mu']:.1f}  (L1 cost weight)")
        print(f"   ν = {recommended['nu']:.2f}  (Deficit weight)")
        print(f"   H = {recommended['H']:3d}  (Prediction horizon)")

        print(f"\n📈 Performance Metrics:")
        print(f"   Average Fee: {recommended['avg_fee_gwei']:.2f} gwei")
        print(f"   UX Score: {recommended['avg_ux_score']:.3f} (lower is better)")
        print(f"   Robustness Score: {recommended['avg_robustness_score']:.1f} (lower is better)")
        print(f"   Capital Efficiency: {recommended['avg_capital_efficiency']:.2e} ETH/gas")
        print(f"   Feasible Datasets: {recommended['feasible_datasets']}/{len(datasets)}")

        # Update CLAUDE.md recommendation
        print(f"\n📝 CLAUDE.md Update Recommendation:")
        print(f"   Replace current parameters with:")
        print(f"   - SPECS Optimal: μ={recommended['mu']:.1f}, ν={recommended['nu']:.2f}, H={recommended['H']} (validated across crisis scenarios)")

        return recommended

    else:
        print("❌ No feasible parameter combinations found!")
        print("💡 Consider further relaxing constraints or adjusting the model.")
        return None

if __name__ == "__main__":
    main()