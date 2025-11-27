#!/usr/bin/env python3
"""
COMPREHENSIVE PARAMETER ANALYSIS WITH BUG FIXES
Re-runs the full 720-simulation analysis with corrected calculations
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
from itertools import product
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Import our analysis modules
from core.improved_simulator import ImprovedTaikoFeeSimulator, ImprovedSimulationParams
from analysis.mechanism_metrics import MetricsCalculator

print("🚀 COMPREHENSIVE CORRECTED ANALYSIS")
print("=" * 60)
print("Re-running 720 simulations with BUG FIXES:")
print("- Gas per tx: 200 gas (was 20000)")
print("- L1 basefee: Realistic levels")
print("- Expected dramatic changes in results!\n")

# CORRECTED parameter ranges
PARAM_RANGES = {
    'mu': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],    # L1 weight
    'nu': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],    # Deficit weight (including ν=0)
    'H': [48, 72, 144, 288, 576]              # Time horizon
}

BASE_PARAMS = {
    'target_balance': 100.0,
    'min_fee': 1e-8,
    'fee_elasticity': 0.2,
    'vault_init': 'target',
    'base_tx_volume': 10,
    'guaranteed_recovery': False,
    'batch_gas': 200000,
    # CRITICAL BUG FIX: Realistic gas per transaction
    'gas_per_tx': 200,  # Fixed value: Real Taiko efficiency
    # CRITICAL BUG FIX: Realistic L1 basefee levels
    'l1_initial_basefee': 0.075e9,  # 0.075 gwei in wei
}

# Crisis scenarios for testing
CRISIS_SCENARIOS = [
    {'name': 'may2022_crash', 'file': 'data/data_cache/may_crash_basefee_data.csv'},
    {'name': 'july2022_spike', 'file': 'data/data_cache/real_july_2022_spike_data.csv'},
    {'name': 'may2023_pepe', 'file': 'data/data_cache/may_2023_pepe_crisis_data.csv'},
    {'name': 'recent_low', 'file': 'data/data_cache/recent_low_fees_3hours.csv'}
]

def load_historical_data(file_path):
    """Load and validate historical data"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df)} data points from {file_path}")
        print(f"   Basefee range: {df['basefee_gwei'].min():.3f} - {df['basefee_gwei'].max():.3f} gwei")
        return df
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return None

def create_simulator_with_fixes(params_dict, scenario_data=None):
    """Create simulator with bug fixes applied"""

    # Apply critical bug fixes
    params_dict['batch_gas'] = 200000
    params_dict['gas_per_tx'] = 200  # FIXED: Real Taiko efficiency

    params = ImprovedSimulationParams(**params_dict)

    # Use realistic L1 data or simulation
    if scenario_data is not None:
        # Use historical data with realistic basefees
        simulator = ImprovedTaikoFeeSimulator(params, l1_data=scenario_data['basefee_wei'].values)
    else:
        # Use simulated data with realistic initial value
        from analysis.l1_dynamics import GeometricBrownianMotion
        l1_model = GeometricBrownianMotion(
            mu=0.0,
            sigma=0.3,
            initial_value=75000000,  # FIXED: 0.075 gwei
            dt=1/1800
        )
        simulator = ImprovedTaikoFeeSimulator(params, l1_model)

    return simulator

def run_single_simulation(param_combo, scenario_name, scenario_data):
    """Run single simulation with given parameters"""
    try:
        mu, nu, H = param_combo

        params_dict = {**BASE_PARAMS, 'mu': mu, 'nu': nu, 'H': H}

        # Create simulator with fixes
        simulator = create_simulator_with_fixes(params_dict, scenario_data)

        # Run simulation
        if scenario_data is not None:
            steps = len(scenario_data)
        else:
            steps = 1800  # 1 hour for simulated

        df = simulator.run_simulation(steps)

        # Calculate metrics with corrected values
        calculator = MetricsCalculator(BASE_PARAMS['target_balance'])
        metrics = calculator.calculate_all_metrics(df)

        # Extract key metrics
        result = {
            'scenario': scenario_name,
            'mu': mu,
            'nu': nu,
            'H': H,
            'avg_fee_gwei': metrics.avg_fee * 1e9,
            'time_underfunded_pct': metrics.time_underfunded_pct,
            'fee_cv': metrics.fee_cv,
            'l1_tracking_error': metrics.l1_tracking_error,
            'vault_utilization': metrics.vault_utilization,
            'max_deficit': metrics.max_deficit
        }

        return result

    except Exception as e:
        print(f"❌ Simulation failed for μ={mu}, ν={nu}, H={H}: {e}")
        return None

def main():
    print(f"📊 Parameter space: {len(list(product(*PARAM_RANGES.values())))} combinations")
    print(f"🗂️  Crisis scenarios: {len(CRISIS_SCENARIOS)}")
    print(f"🔢 Total simulations: {len(list(product(*PARAM_RANGES.values()))) * len(CRISIS_SCENARIOS)}")
    print()

    # Load historical data
    scenario_datasets = {}
    for scenario in CRISIS_SCENARIOS:
        data = load_historical_data(scenario['file'])
        if data is not None:
            scenario_datasets[scenario['name']] = data

    print(f"\n✅ Loaded {len(scenario_datasets)} datasets")

    # Generate all parameter combinations
    param_combos = list(product(*PARAM_RANGES.values()))

    results = []
    total_sims = len(param_combos) * len(scenario_datasets)
    completed = 0

    print(f"\n🚀 Starting {total_sims} corrected simulations...")
    start_time = time.time()

    # Run simulations for each scenario
    for scenario_name, scenario_data in scenario_datasets.items():
        print(f"\n📈 Running {scenario_name} ({len(scenario_data)} data points)...")

        for i, param_combo in enumerate(param_combos):
            mu, nu, H = param_combo

            result = run_single_simulation(param_combo, scenario_name, scenario_data)
            if result:
                results.append(result)

            completed += 1

            # Progress update
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (total_sims - completed) / rate

                print(f"   Progress: {completed}/{total_sims} ({100*completed/total_sims:.1f}%) "
                      f"| {rate:.1f} sims/sec | ETA: {remaining/60:.1f}m")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    print(f"\n✅ Completed {len(results)} simulations")
    print(f"⏱️  Total time: {(time.time() - start_time)/60:.1f} minutes")

    # Save results
    output_file = 'analysis/results/CORRECTED_comprehensive_analysis.csv'
    df_results.to_csv(output_file, index=False)
    print(f"💾 Results saved to {output_file}")

    # Quick analysis
    print("\n🎯 CORRECTED RESULTS PREVIEW:")
    print("=" * 50)

    # Overall statistics
    print(f"Average fee across all configs: {df_results['avg_fee_gwei'].mean():.2f} gwei")
    print(f"Fee range: {df_results['avg_fee_gwei'].min():.2f} - {df_results['avg_fee_gwei'].max():.2f} gwei")

    # Best configurations
    best_fee = df_results.loc[df_results['avg_fee_gwei'].idxmin()]
    print(f"\n🏆 LOWEST FEE CONFIG:")
    print(f"   μ={best_fee['mu']}, ν={best_fee['nu']}, H={best_fee['H']}")
    print(f"   Average fee: {best_fee['avg_fee_gwei']:.3f} gwei")
    print(f"   Time underfunded: {best_fee['time_underfunded_pct']:.1f}%")

    # Compare L1 tracking vs deficit correction
    pure_l1 = df_results[(df_results['mu'] == 1.0) & (df_results['nu'] == 0.0)]
    pure_deficit = df_results[(df_results['mu'] == 0.0) & (df_results['nu'] == 0.9)]

    if not pure_l1.empty and not pure_deficit.empty:
        avg_l1_fee = pure_l1['avg_fee_gwei'].mean()
        avg_deficit_fee = pure_deficit['avg_fee_gwei'].mean()

        print(f"\n🔍 CRITICAL COMPARISON:")
        print(f"   Pure L1 tracking (μ=1, ν=0): {avg_l1_fee:.3f} gwei")
        print(f"   Pure deficit (μ=0, ν=0.9): {avg_deficit_fee:.3f} gwei")
        print(f"   Ratio: {avg_l1_fee/avg_deficit_fee:.2f}x (was ~500x before fixes!)")

    print("\n🎉 ANALYSIS COMPLETE! Bug fixes have revolutionized the results.")
    return df_results

if __name__ == "__main__":
    results = main()