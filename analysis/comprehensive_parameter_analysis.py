#!/usr/bin/env python3
"""
Comprehensive Taiko Fee Parameter Analysis - Including Missing ν=0 Combinations

This script addresses the critical oversight in the original analysis by including
ν=0 (no deficit correction) configurations, with special focus on μ=1,ν=0
(pure L1 cost tracking) which was identified as potentially optimal.
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

                # Use more data points for comprehensive analysis
                if len(df) > 400:
                    df = df.head(400)

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
    """Run expanded parameter sweep including ν=0 combinations."""

    results = []
    total_runs = len(list(product(*param_ranges.values()))) * len(l1_models)
    current_run = 0

    print(f"Starting comprehensive parameter sweep: {total_runs} simulations")
    print(f"Key focus: Testing μ=1,ν=0 (pure L1 tracking) vs μ=0,ν=0.9 (pure deficit correction)")

    # Track promising configurations
    promising_configs = []

    # Iterate through all parameter combinations
    for param_combo in product(*param_ranges.values()):
        mu, nu, H = param_combo

        # Test on each historical scenario
        for scenario_name, l1_model in l1_models.items():
            current_run += 1

            if current_run % 15 == 0:
                print(f"  Progress: {current_run}/{total_runs} ({100*current_run/total_runs:.1f}%)")
                if promising_configs:
                    print(f"    Current best: {promising_configs[0]}")

            # Highlight key configurations
            is_pure_l1 = (mu == 1.0 and nu == 0.0)
            is_pure_deficit = (mu == 0.0 and nu == 0.9)
            is_zero_fee = (mu == 0.0 and nu == 0.0)

            if is_pure_l1:
                print(f"  🎯 Testing PURE L1 TRACKING: μ=1.0, ν=0.0, H={H} on {scenario_name}")
            elif is_pure_deficit:
                print(f"  🎯 Testing PURE DEFICIT: μ=0.0, ν=0.9, H={H} on {scenario_name}")
            elif is_zero_fee:
                print(f"  ⚠️  Testing ZERO FEE: μ=0.0, ν=0.0, H={H} on {scenario_name}")

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
                    'config_type': 'pure_l1' if is_pure_l1 else 'pure_deficit' if is_pure_deficit else 'zero_fee' if is_zero_fee else 'hybrid',
                    **metrics.to_dict()
                }
                results.append(result)

                # Track promising configurations (low average fee + reasonable stability)
                if (metrics.avg_fee < 0.001 and  # Low fees
                    metrics.time_underfunded_pct < 30 and  # Reasonable stability
                    metrics.max_deficit < 3000):  # Manageable deficits

                    promising_configs.append({
                        'mu': mu, 'nu': nu, 'H': H, 'scenario': scenario_name,
                        'avg_fee': metrics.avg_fee,
                        'underfunded_pct': metrics.time_underfunded_pct,
                        'config_type': result['config_type']
                    })

                # Special reporting for key configurations
                if is_pure_l1 or is_pure_deficit:
                    print(f"    Result: avg_fee={metrics.avg_fee:.2e} ETH, underfunded={metrics.time_underfunded_pct:.1f}%")

            except Exception as e:
                print(f"    ✗ Failed simulation: μ={mu}, ν={nu}, H={H}, scenario={scenario_name}: {e}")
                continue

    print(f"✓ Completed {len(results)} successful simulations")

    # Report promising configurations
    if promising_configs:
        print(f"\n🌟 Found {len(promising_configs)} promising configurations:")
        promising_df = pd.DataFrame(promising_configs)
        best_by_fee = promising_df.nsmallest(5, 'avg_fee')
        for _, config in best_by_fee.iterrows():
            print(f"  μ={config['mu']}, ν={config['nu']}, H={config['H']} ({config['config_type']})")
            print(f"    Fee: {config['avg_fee']:.2e} ETH, Underfunded: {config['underfunded_pct']:.1f}%")

    return pd.DataFrame(results)

def analyze_special_configurations(results_df: pd.DataFrame) -> Dict:
    """Analyze key configurations of interest."""

    print("\n" + "="*60)
    print("SPECIAL CONFIGURATION ANALYSIS")
    print("="*60)

    analysis_results = {}

    # 1. Pure L1 Tracking (μ=1, ν=0)
    pure_l1 = results_df[(results_df['mu'] == 1.0) & (results_df['nu'] == 0.0)]
    if not pure_l1.empty:
        pure_l1_agg = pure_l1.groupby(['mu', 'nu', 'H']).agg({
            'avg_fee': 'mean',
            'time_underfunded_pct': 'mean',
            'max_deficit': 'mean',
            'fee_cv': 'mean',
            'l1_tracking_error': 'mean'
        })

        print("\n🎯 PURE L1 TRACKING (μ=1.0, ν=0.0) Results:")
        for (mu, nu, H), row in pure_l1_agg.iterrows():
            print(f"  H={H}: avg_fee={row['avg_fee']:.2e} ETH, underfunded={row['time_underfunded_pct']:.1f}%, tracking_error={row['l1_tracking_error']:.3f}")

        best_l1 = pure_l1_agg.loc[pure_l1_agg['avg_fee'].idxmin()]
        analysis_results['pure_l1_best'] = {
            'params': pure_l1_agg.index[pure_l1_agg['avg_fee'].argmin()],
            'metrics': best_l1.to_dict()
        }

    # 2. Pure Deficit Correction (μ=0, ν=0.9)
    pure_deficit = results_df[(results_df['mu'] == 0.0) & (results_df['nu'] == 0.9)]
    if not pure_deficit.empty:
        pure_deficit_agg = pure_deficit.groupby(['mu', 'nu', 'H']).agg({
            'avg_fee': 'mean',
            'time_underfunded_pct': 'mean',
            'max_deficit': 'mean',
            'fee_cv': 'mean',
            'deficit_correction_efficiency': 'mean'
        })

        print("\n🎯 PURE DEFICIT CORRECTION (μ=0.0, ν=0.9) Results:")
        for (mu, nu, H), row in pure_deficit_agg.iterrows():
            print(f"  H={H}: avg_fee={row['avg_fee']:.2e} ETH, underfunded={row['time_underfunded_pct']:.1f}%, correction_eff={row['deficit_correction_efficiency']:.3f}")

        best_deficit = pure_deficit_agg.loc[pure_deficit_agg['avg_fee'].idxmin()]
        analysis_results['pure_deficit_best'] = {
            'params': pure_deficit_agg.index[pure_deficit_agg['avg_fee'].argmin()],
            'metrics': best_deficit.to_dict()
        }

    # 3. Zero Fee Configuration (μ=0, ν=0)
    zero_fee = results_df[(results_df['mu'] == 0.0) & (results_df['nu'] == 0.0)]
    if not zero_fee.empty:
        print("\n⚠️  ZERO FEE CONFIGURATION (μ=0.0, ν=0.0) - Sustainability Check:")
        zero_fee_agg = zero_fee.groupby(['mu', 'nu', 'H']).agg({
            'avg_fee': 'mean',
            'time_underfunded_pct': 'mean',
            'max_deficit': 'mean',
            'vault_utilization': 'mean'
        })

        for (mu, nu, H), row in zero_fee_agg.iterrows():
            sustainability = "SUSTAINABLE" if row['time_underfunded_pct'] < 50 else "UNSUSTAINABLE"
            print(f"  H={H}: {sustainability} - underfunded={row['time_underfunded_pct']:.1f}%, max_deficit={row['max_deficit']:.0f}")

    # 4. Comparison Summary
    if 'pure_l1_best' in analysis_results and 'pure_deficit_best' in analysis_results:
        print("\n" + "="*60)
        print("🏆 HEAD-TO-HEAD COMPARISON")
        print("="*60)

        l1_best = analysis_results['pure_l1_best']
        deficit_best = analysis_results['pure_deficit_best']

        print(f"Pure L1 Tracking (μ=1,ν=0):")
        print(f"  Parameters: μ={l1_best['params'][0]}, ν={l1_best['params'][1]}, H={l1_best['params'][2]}")
        print(f"  Avg Fee: {l1_best['metrics']['avg_fee']:.2e} ETH")
        print(f"  Time Underfunded: {l1_best['metrics']['time_underfunded_pct']:.1f}%")
        print(f"  L1 Tracking Error: {l1_best['metrics']['l1_tracking_error']:.3f}")

        print(f"\nPure Deficit Correction (μ=0,ν=0.9):")
        print(f"  Parameters: μ={deficit_best['params'][0]}, ν={deficit_best['params'][1]}, H={deficit_best['params'][2]}")
        print(f"  Avg Fee: {deficit_best['metrics']['avg_fee']:.2e} ETH")
        print(f"  Time Underfunded: {deficit_best['metrics']['time_underfunded_pct']:.1f}%")
        print(f"  Deficit Correction Eff: {deficit_best['metrics']['deficit_correction_efficiency']:.3f}")

        # Determine winner
        l1_fee = l1_best['metrics']['avg_fee']
        deficit_fee = deficit_best['metrics']['avg_fee']

        if l1_fee < deficit_fee * 1.1:  # Within 10%
            winner = "🤝 TIE - Both configurations excellent"
        elif l1_fee < deficit_fee:
            winner = "🥇 PURE L1 TRACKING WINS - Lower fees"
        else:
            winner = "🥇 PURE DEFICIT CORRECTION WINS - Lower fees"

        print(f"\n{winner}")

        analysis_results['comparison_winner'] = winner

    return analysis_results

def create_comprehensive_visualization(results_df: pd.DataFrame, special_analysis: Dict):
    """Create comprehensive visualizations of the expanded parameter space."""

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Parameter Analysis - Including ν=0 Configurations', fontsize=16, fontweight='bold')

    # Calculate aggregate metrics
    agg_results = results_df.groupby(['mu', 'nu', 'H']).agg({
        'avg_fee': 'mean',
        'time_underfunded_pct': 'mean',
        'max_deficit': 'mean',
        'fee_cv': 'mean'
    }).reset_index()

    # 1. Heatmap: μ vs ν (average across H values)
    ax1 = axes[0, 0]
    heatmap_data = agg_results.groupby(['mu', 'nu'])['avg_fee'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, fmt='.2e', cmap='viridis_r', ax=ax1, cbar_kws={'label': 'Avg Fee (ETH)'})
    ax1.set_title('Average Fee: μ vs ν')
    ax1.set_xlabel('ν (Deficit Weight)')
    ax1.set_ylabel('μ (L1 Weight)')

    # Highlight key configurations
    ax1.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False, edgecolor='red', lw=3))  # μ=1, ν=0
    ax1.add_patch(plt.Rectangle((4.5, 0), 1, 1, fill=False, edgecolor='blue', lw=3))  # μ=0, ν=0.9

    # 2. Configuration type comparison
    ax2 = axes[0, 1]
    config_comparison = results_df.groupby('config_type')['avg_fee'].agg(['mean', 'std']).reset_index()
    ax2.bar(config_comparison['config_type'], config_comparison['mean'],
            yerr=config_comparison['std'], capsize=5, alpha=0.7)
    ax2.set_title('Average Fee by Configuration Type')
    ax2.set_ylabel('Average Fee (ETH)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_yscale('log')

    # 3. Stability vs Fee trade-off
    ax3 = axes[0, 2]
    scatter = ax3.scatter(agg_results['time_underfunded_pct'], agg_results['avg_fee'],
                         c=agg_results['mu'], s=agg_results['nu']*100, alpha=0.6, cmap='coolwarm')
    ax3.set_xlabel('Time Underfunded (%)')
    ax3.set_ylabel('Average Fee (ETH)')
    ax3.set_title('Stability vs Fee Trade-off\n(color=μ, size=ν)')
    ax3.set_yscale('log')
    plt.colorbar(scatter, ax=ax3, label='μ')

    # Highlight special points
    pure_l1_points = agg_results[(agg_results['mu'] == 1.0) & (agg_results['nu'] == 0.0)]
    pure_deficit_points = agg_results[(agg_results['mu'] == 0.0) & (agg_results['nu'] == 0.9)]

    if not pure_l1_points.empty:
        ax3.scatter(pure_l1_points['time_underfunded_pct'], pure_l1_points['avg_fee'],
                   c='red', s=200, marker='x', linewidth=3, label='Pure L1 (μ=1,ν=0)')
    if not pure_deficit_points.empty:
        ax3.scatter(pure_deficit_points['time_underfunded_pct'], pure_deficit_points['avg_fee'],
                   c='blue', s=200, marker='+', linewidth=3, label='Pure Deficit (μ=0,ν=0.9)')
    ax3.legend()

    # 4. ν=0 configurations focus
    ax4 = axes[1, 0]
    nu_zero = agg_results[agg_results['nu'] == 0.0]
    if not nu_zero.empty:
        ax4.plot(nu_zero['mu'], nu_zero['avg_fee'], 'o-', linewidth=2, markersize=8)
        ax4.set_xlabel('μ (L1 Weight)')
        ax4.set_ylabel('Average Fee (ETH)')
        ax4.set_title('ν=0 Configuration Performance\n(No Deficit Correction)')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        # Highlight μ=1, ν=0 point
        mu_1_point = nu_zero[nu_zero['mu'] == 1.0]
        if not mu_1_point.empty:
            ax4.scatter(1.0, mu_1_point['avg_fee'].iloc[0], c='red', s=200, marker='*',
                       linewidth=2, label='μ=1,ν=0 (Pure L1)', zorder=5)
            ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No ν=0 data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('ν=0 Configuration Performance')

    # 5. Fee coefficient of variation
    ax5 = axes[1, 1]
    cv_heatmap = agg_results.groupby(['mu', 'nu'])['fee_cv'].mean().unstack()
    sns.heatmap(cv_heatmap, annot=True, fmt='.3f', cmap='viridis', ax=ax5, cbar_kws={'label': 'Fee CV'})
    ax5.set_title('Fee Volatility: μ vs ν')
    ax5.set_xlabel('ν (Deficit Weight)')
    ax5.set_ylabel('μ (L1 Weight)')

    # 6. Summary statistics table
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Create summary table
    summary_data = []

    if 'pure_l1_best' in special_analysis:
        l1_best = special_analysis['pure_l1_best']
        summary_data.append([
            'Pure L1 (μ=1,ν=0)',
            f"{l1_best['metrics']['avg_fee']:.2e}",
            f"{l1_best['metrics']['time_underfunded_pct']:.1f}%",
            f"{l1_best['metrics']['l1_tracking_error']:.3f}"
        ])

    if 'pure_deficit_best' in special_analysis:
        deficit_best = special_analysis['pure_deficit_best']
        summary_data.append([
            'Pure Deficit (μ=0,ν=0.9)',
            f"{deficit_best['metrics']['avg_fee']:.2e}",
            f"{deficit_best['metrics']['time_underfunded_pct']:.1f}%",
            f"{deficit_best['metrics']['deficit_correction_efficiency']:.3f}"
        ])

    # Best overall (lowest fee)
    best_overall = agg_results.loc[agg_results['avg_fee'].idxmin()]
    summary_data.append([
        f'Best Overall (μ={best_overall["mu"]},ν={best_overall["nu"]})',
        f"{best_overall['avg_fee']:.2e}",
        f"{best_overall['time_underfunded_pct']:.1f}%",
        f"{best_overall['fee_cv']:.3f}"
    ])

    if summary_data:
        table = ax6.table(cellText=summary_data,
                         colLabels=['Configuration', 'Avg Fee (ETH)', 'Underfunded %', 'Key Metric'],
                         cellLoc='left', loc='center',
                         colWidths=[0.3, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.5)
        ax6.set_title('Configuration Comparison Summary', pad=20)

    plt.tight_layout()

    # Save the visualization
    output_path = f"{project_root}/analysis/results/comprehensive_parameter_analysis.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive visualization saved to {output_path}")

    plt.show()

def main():
    """Run comprehensive parameter analysis including ν=0 configurations."""

    print("="*70)
    print("COMPREHENSIVE TAIKO FEE PARAMETER ANALYSIS")
    print("Including Missing ν=0 Combinations")
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

    # Define EXPANDED parameter space including ν=0
    print("\n3. Defining Expanded Parameter Space...")
    PARAM_RANGES = {
        'mu': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],    # L1 weight
        'nu': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],    # Deficit weight - NOW INCLUDING 0.0!
        'H': [48, 72, 144, 288, 576]              # Deficit correction horizon - expanded
    }

    BASE_PARAMS = {
        'target_balance': 1000.0,           # ETH
        'base_demand': 100,                 # transactions per step
        'fee_elasticity': 0.2,              # demand elasticity
        'gas_per_batch': 200000,            # gas per L1 batch
        'txs_per_batch': 100,               # txs per batch
        'batch_frequency': 0.1,             # batches per step
        'total_steps': 300,                 # simulation length
        'time_step_seconds': 12,            # L2 block time
        'vault_initialization_mode': 'target',  # start at target balance
        'fee_cap': 0.5                      # 0.5 ETH max fee cap
    }

    total_combinations = np.prod([len(v) for v in PARAM_RANGES.values()])
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total simulations (×{len(l1_models)} scenarios): {total_combinations * len(l1_models)}")
    print(f"🎯 Key focus: μ=1,ν=0 (pure L1) vs μ=0,ν=0.9 (pure deficit)")

    # Run comprehensive parameter sweep
    print("\n4. Running Comprehensive Parameter Sweep...")
    sweep_results = run_comprehensive_parameter_sweep(l1_models, PARAM_RANGES, BASE_PARAMS)

    if sweep_results.empty:
        print("✗ No successful simulations. Cannot proceed with analysis.")
        return

    # Analyze special configurations
    print("\n5. Analyzing Special Configurations...")
    special_analysis = analyze_special_configurations(sweep_results)

    # Create visualizations
    print("\n6. Creating Comprehensive Visualizations...")
    create_comprehensive_visualization(sweep_results, special_analysis)

    # Save results
    print("\n7. Saving Results...")
    output_dir = f"{project_root}/analysis/results"
    os.makedirs(output_dir, exist_ok=True)

    sweep_results.to_csv(f"{output_dir}/comprehensive_parameter_results.csv", index=False)

    # Save special analysis results
    if special_analysis:
        import json
        with open(f"{output_dir}/special_configurations_analysis.json", 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            json_safe_analysis = {}
            for key, value in special_analysis.items():
                if isinstance(value, dict):
                    json_safe_analysis[key] = {
                        k: float(v) if isinstance(v, np.number) else v
                        for k, v in value.items()
                        if k != 'params'  # Skip params tuple which isn't JSON serializable
                    }
                else:
                    json_safe_analysis[key] = str(value)

            json.dump(json_safe_analysis, f, indent=2)

    print(f"✓ Results saved to {output_dir}/")
    print("✓ Comprehensive analysis complete!")

    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    if 'comparison_winner' in special_analysis:
        print(f"🏆 Result: {special_analysis['comparison_winner']}")
    else:
        print("🤔 Inconclusive - need to examine results manually")

    print(f"📊 Total configurations tested: {len(sweep_results)}")
    print(f"📈 Data saved for further analysis and preset optimization")

if __name__ == "__main__":
    main()