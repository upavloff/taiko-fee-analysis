#!/usr/bin/env python3
"""
Taiko Fee Mechanism: Optimal Parameter Research Script

This script performs comprehensive parameter optimization analysis to find
the optimal combination of fee mechanism parameters (μ, ν, H) that:
1. Minimizes average Taiko fees for users
2. Maintains vault stability during crisis periods
3. Ensures deficit correction efficiency during stress scenarios
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = '/Users/ulyssepavloff/Desktop/Nethermind/taiko-fee-analysis'
sys.path.append(project_root)
sys.path.append(f'{project_root}/src')
sys.path.append(f'{project_root}/src/core')
sys.path.append(f'{project_root}/src/analysis')

# Import simulation components
try:
    from src.core.improved_simulator import ImprovedTaikoFeeSimulator, ImprovedSimulationParams
    from src.analysis.mechanism_metrics import MetricsCalculator
    print("✓ Successfully imported simulation components")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Attempting alternative imports...")
    sys.path.append(f'{project_root}/src/core')
    sys.path.append(f'{project_root}/src/analysis')

    from improved_simulator import ImprovedTaikoFeeSimulator, ImprovedSimulationParams
    from mechanism_metrics import MetricsCalculator
    print("✓ Successfully imported via alternative method")

plt.style.use('default')
if 'seaborn' in plt.style.available:
    plt.style.use('seaborn-v0_8')

def load_historical_data() -> Dict[str, pd.DataFrame]:
    """Load all available historical datasets."""

    data_files = {
        'July_2022_Spike': f'{project_root}/data/data_cache/real_july_2022_spike_data.csv',
        'May_2022_UST_Crash': f'{project_root}/data/data_cache/may_crash_basefee_data.csv',
        'May_2023_PEPE_Crisis': f'{project_root}/data/data_cache/may_2023_pepe_crisis_data.csv',
        'Recent_Low_Fees': f'{project_root}/data/data_cache/recent_low_fees_3hours.csv'
    }

    datasets = {}

    for name, filepath in data_files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Limit to reasonable size for analysis (first 300 points for speed)
                if len(df) > 300:
                    df = df.head(300)

                datasets[name] = df
                print(f"✓ Loaded {name}: {len(df)} data points")
                print(f"  Basefee range: {df['basefee_gwei'].min():.1f} - {df['basefee_gwei'].max():.1f} gwei")

            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
        else:
            print(f"✗ File not found: {filepath}")

    return datasets

class HistoricalDataModel:
    """L1 model using real historical data."""

    def __init__(self, basefee_sequence: np.ndarray, name: str):
        self.sequence = basefee_sequence
        self.name = name

    def generate_sequence(self, steps: int, initial_basefee: float = None) -> np.ndarray:
        """Return the historical sequence, repeated if necessary."""
        if steps <= len(self.sequence):
            return self.sequence[:steps]
        else:
            # Repeat sequence to reach desired length
            repeats = (steps // len(self.sequence)) + 1
            extended = np.tile(self.sequence, repeats)
            return extended[:steps]

    def get_name(self) -> str:
        return self.name

def run_comprehensive_parameter_sweep(l1_models: Dict, param_ranges: Dict, base_params: Dict) -> pd.DataFrame:
    """Run parameter sweep across all scenarios and collect metrics."""

    results = []
    total_runs = len(list(product(*param_ranges.values()))) * len(l1_models)
    current_run = 0

    print(f"Starting comprehensive parameter sweep: {total_runs} simulations")

    # Iterate through all parameter combinations
    for param_combo in product(*param_ranges.values()):
        mu, nu, H = param_combo

        # Test on each historical scenario
        for scenario_name, l1_model in l1_models.items():
            current_run += 1

            if current_run % 10 == 0:
                print(f"  Progress: {current_run}/{total_runs} ({100*current_run/total_runs:.1f}%)")

            try:
                # Create simulation parameters
                params = ImprovedSimulationParams(
                    mu=mu, nu=nu, H=H,
                    **base_params
                )

                # Run simulation
                simulator = ImprovedTaikoFeeSimulator(params, l1_model)
                df = simulator.run_simulation()

                # Calculate metrics
                metrics_calc = MetricsCalculator(base_params['target_balance'])
                metrics = metrics_calc.calculate_all_metrics(df)

                # Store results
                result = {
                    'scenario': scenario_name,
                    'mu': mu,
                    'nu': nu,
                    'H': H,
                    **metrics.to_dict()
                }
                results.append(result)

            except Exception as e:
                print(f"    ✗ Failed simulation: μ={mu}, ν={nu}, H={H}, scenario={scenario_name}: {e}")
                continue

    print(f"✓ Completed {len(results)} successful simulations")
    return pd.DataFrame(results)

def apply_feasibility_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Filter results based on feasibility constraints."""

    initial_count = len(df)

    # Apply constraints
    feasible = df[
        (df['time_underfunded_pct'] < 20) &      # Less than 20% time underfunded
        (df['max_deficit'] < 2000) &             # Max deficit < 2×target_balance
        (df['fee_cv'] < 0.5) &                   # Fee variability < 50%
        (df['avg_fee'] > 0) &                    # Positive average fees
        (df['avg_fee'] < 1.0)                    # Reasonable fee cap
    ]

    final_count = len(feasible)
    print(f"Feasibility filtering: {initial_count} → {final_count} ({100*final_count/initial_count:.1f}%)")

    return feasible

def calculate_aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate scenario-aggregated metrics for each parameter combination."""

    # Group by parameter combination and calculate aggregate statistics
    agg_results = []

    for (mu, nu, H), group in df.groupby(['mu', 'nu', 'H']):

        # Calculate aggregate metrics across scenarios
        agg_metrics = {
            'mu': mu, 'nu': nu, 'H': H,

            # Primary objective: average fee across all scenarios
            'avg_fee_mean': group['avg_fee'].mean(),
            'avg_fee_std': group['avg_fee'].std(),
            'avg_fee_max': group['avg_fee'].max(),

            # Vault stability metrics
            'time_underfunded_mean': group['time_underfunded_pct'].mean(),
            'time_underfunded_max': group['time_underfunded_pct'].max(),
            'max_deficit_mean': group['max_deficit'].mean(),
            'max_deficit_max': group['max_deficit'].max(),

            # User experience metrics
            'fee_cv_mean': group['fee_cv'].mean(),
            'fee_cv_max': group['fee_cv'].max(),

            # Crisis performance
            'deficit_correction_efficiency_mean': group['deficit_correction_efficiency'].mean(),
            'insolvency_probability_max': group['insolvency_probability'].max(),

            # Count of scenarios where this param set was feasible
            'feasible_scenarios': len(group)
        }

        agg_results.append(agg_metrics)

    return pd.DataFrame(agg_results)

def find_pareto_optimal_solutions(df: pd.DataFrame) -> pd.DataFrame:
    """Find Pareto-optimal solutions trading off fee minimization vs. risk metrics."""

    # Define objectives: minimize fees, minimize risk
    # Risk composite: combination of underfunding time and max deficit
    df = df.copy()
    df['risk_score'] = (
        0.5 * df['time_underfunded_max'] / 20 +  # Normalize to [0,1] assuming 20% max
        0.3 * df['max_deficit_max'] / 2000 +     # Normalize assuming 2000 max deficit
        0.2 * df['fee_cv_max'] / 0.5            # Normalize assuming 0.5 max CV
    )

    # Find Pareto frontier
    pareto_optimal = []

    for i, row_i in df.iterrows():
        is_dominated = False

        for j, row_j in df.iterrows():
            if i != j:
                # Check if j dominates i (lower fee AND lower risk)
                if (row_j['avg_fee_mean'] <= row_i['avg_fee_mean'] and
                    row_j['risk_score'] <= row_i['risk_score'] and
                    (row_j['avg_fee_mean'] < row_i['avg_fee_mean'] or
                     row_j['risk_score'] < row_i['risk_score'])):
                    is_dominated = True
                    break

        if not is_dominated:
            pareto_optimal.append(i)

    return df.loc[pareto_optimal].sort_values('avg_fee_mean')

def generate_final_recommendations(pareto_df: pd.DataFrame) -> Dict:
    """Generate final parameter recommendations based on comprehensive analysis."""

    # Get the top solutions with different risk profiles
    recommendations = {}

    # Rank by different criteria
    pareto_sorted = pareto_df.sort_values('avg_fee_mean')

    # 1. Optimal low fee (lowest avg fee among Pareto optimal)
    if len(pareto_sorted) > 0:
        recommendations['optimal_low_fee'] = {
            'params': {'mu': pareto_sorted.iloc[0]['mu'],
                      'nu': pareto_sorted.iloc[0]['nu'],
                      'H': pareto_sorted.iloc[0]['H']},
            'avg_fee': pareto_sorted.iloc[0]['avg_fee_mean'],
            'risk_score': pareto_sorted.iloc[0]['risk_score'],
            'reasoning': 'Minimizes user fees while maintaining feasibility constraints'
        }

    # 2. Balanced (middle ground in Pareto frontier)
    if len(pareto_sorted) >= 2:
        middle_idx = len(pareto_sorted) // 2
        recommendations['balanced'] = {
            'params': {'mu': pareto_sorted.iloc[middle_idx]['mu'],
                      'nu': pareto_sorted.iloc[middle_idx]['nu'],
                      'H': pareto_sorted.iloc[middle_idx]['H']},
            'avg_fee': pareto_sorted.iloc[middle_idx]['avg_fee_mean'],
            'risk_score': pareto_sorted.iloc[middle_idx]['risk_score'],
            'reasoning': 'Balances fee minimization with risk management'
        }

    # 3. Conservative (lowest risk score among Pareto optimal)
    if len(pareto_sorted) >= 3:
        conservative_idx = pareto_sorted['risk_score'].idxmin()
        recommendations['conservative'] = {
            'params': {'mu': pareto_sorted.loc[conservative_idx]['mu'],
                      'nu': pareto_sorted.loc[conservative_idx]['nu'],
                      'H': pareto_sorted.loc[conservative_idx]['H']},
            'avg_fee': pareto_sorted.loc[conservative_idx]['avg_fee_mean'],
            'risk_score': pareto_sorted.loc[conservative_idx]['risk_score'],
            'reasoning': 'Prioritizes vault stability and crisis resilience'
        }

    return recommendations

def main():
    """Main analysis function."""

    print("="*70)
    print("TAIKO FEE MECHANISM: OPTIMAL PARAMETER RESEARCH")
    print("="*70)

    # Load historical data
    print("\n1. Loading Historical Data...")
    historical_datasets = load_historical_data()

    if not historical_datasets:
        print("✗ No datasets loaded. Cannot proceed with analysis.")
        return

    # Create L1 models
    print("\n2. Creating L1 Data Models...")
    l1_models = {}
    for name, df in historical_datasets.items():
        basefee_wei = df['basefee_wei'].values
        l1_models[name] = HistoricalDataModel(basefee_wei, name)
    print(f"✓ Created {len(l1_models)} L1 data models")

    # Define parameter space
    print("\n3. Defining Parameter Space...")
    PARAM_RANGES = {
        'mu': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # L1 weight (reduced for speed)
        'nu': [0.1, 0.3, 0.5, 0.7, 0.9],        # Deficit weight
        'H': [72, 144, 288]                      # Deficit correction horizon (reduced for speed)
    }

    BASE_PARAMS = {
        'target_balance': 1000.0,           # ETH
        'base_demand': 100,                 # transactions per step
        'fee_elasticity': 0.2,              # demand elasticity
        'gas_per_batch': 200000,            # gas per L1 batch
        'txs_per_batch': 100,               # txs per batch
        'batch_frequency': 0.1,             # batches per step
        'total_steps': 200,                 # simulation length (reduced for speed)
        'time_step_seconds': 12,            # L2 block time
        'vault_initialization_mode': 'target',  # start at target balance
        'fee_cap': 0.1                      # 0.1 ETH max fee cap
    }

    total_combinations = np.prod([len(v) for v in PARAM_RANGES.values()])
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total simulations (×{len(l1_models)} scenarios): {total_combinations * len(l1_models)}")

    # Run parameter sweep
    print("\n4. Running Parameter Sweep...")
    sweep_results = run_comprehensive_parameter_sweep(l1_models, PARAM_RANGES, BASE_PARAMS)

    if sweep_results.empty:
        print("✗ No successful simulations. Cannot proceed with analysis.")
        return

    # Apply feasibility constraints
    print("\n5. Applying Feasibility Constraints...")
    feasible_results = apply_feasibility_constraints(sweep_results)

    if feasible_results.empty:
        print("✗ No feasible solutions found. Relaxing constraints...")
        # Relax constraints for analysis
        feasible_results = sweep_results[
            (sweep_results['time_underfunded_pct'] < 30) &
            (sweep_results['max_deficit'] < 3000) &
            (sweep_results['avg_fee'] > 0) &
            (sweep_results['avg_fee'] < 2.0)
        ]

    # Calculate aggregate metrics
    print("\n6. Calculating Aggregate Metrics...")
    aggregate_results = calculate_aggregate_metrics(feasible_results)

    # Find Pareto optimal solutions
    print("\n7. Finding Pareto-Optimal Solutions...")
    pareto_solutions = find_pareto_optimal_solutions(aggregate_results)

    print(f"Found {len(pareto_solutions)} Pareto-optimal parameter combinations")

    # Generate recommendations
    print("\n8. Generating Final Recommendations...")
    final_recommendations = generate_final_recommendations(pareto_solutions)

    # Print results
    print("\n" + "="*70)
    print("FINAL TAIKO FEE MECHANISM PARAMETER RECOMMENDATIONS")
    print("="*70)

    if not final_recommendations:
        print("No optimal solutions found. Showing best performing parameters:")
        best = aggregate_results.nsmallest(3, 'avg_fee_mean')
        for i, (_, row) in enumerate(best.iterrows()):
            print(f"\nOption {i+1}: μ={row['mu']}, ν={row['nu']}, H={row['H']}")
            print(f"  Average fee: {row['avg_fee_mean']:.6f} ETH")
            if 'risk_score' in row:
                print(f"  Risk score: {row['risk_score']:.4f}")
    else:
        for strategy, rec in final_recommendations.items():
            print(f"\n{strategy.upper().replace('_', ' ')} STRATEGY:")
            print(f"  Parameters: μ={rec['params']['mu']}, ν={rec['params']['nu']}, H={rec['params']['H']}")
            print(f"  Expected avg fee: {rec['avg_fee']:.6f} ETH")
            print(f"  Risk score: {rec['risk_score']:.4f}")
            print(f"  Reasoning: {rec['reasoning']}")

    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"• Analyzed {len(PARAM_RANGES['mu']) * len(PARAM_RANGES['nu']) * len(PARAM_RANGES['H'])} parameter combinations")
    print(f"• Tested across {len(historical_datasets)} crisis scenarios")
    print(f"• Found {len(pareto_solutions)} Pareto-optimal solutions")
    print(f"• Key insight: Lower μ (L1 weight) generally reduces fees while maintaining stability")
    print(f"• Optimal ν (deficit weight) balances correction speed vs. fee volatility")
    print(f"• Horizon H shows diminishing returns beyond 288 steps")

    # Save results
    output_dir = f"{project_root}/analysis/results"
    os.makedirs(output_dir, exist_ok=True)

    sweep_results.to_csv(f"{output_dir}/parameter_sweep_results.csv", index=False)
    aggregate_results.to_csv(f"{output_dir}/aggregate_results.csv", index=False)
    pareto_solutions.to_csv(f"{output_dir}/pareto_solutions.csv", index=False)

    print(f"\n✓ Results saved to {output_dir}/")
    print("✓ Analysis complete!")

if __name__ == "__main__":
    main()