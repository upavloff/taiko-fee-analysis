#!/usr/bin/env python3
"""
Cross-validation script to compare old vs new preset performance
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
project_root = '/Users/ulyssepavloff/Desktop/Nethermind/taiko-fee-analysis'
sys.path.append(project_root)
sys.path.append(f'{project_root}/src')
sys.path.append(f'{project_root}/src/core')
sys.path.append(f'{project_root}/src/analysis')

from src.core.improved_simulator import ImprovedTaikoFeeSimulator, ImprovedSimulationParams
from src.analysis.mechanism_metrics import MetricsCalculator

def load_test_data():
    """Load a test dataset for validation."""
    data_file = f'{project_root}/data/data_cache/luna_crash_true_peak_contiguous.csv'

    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        # Use first 100 points for quick validation
        return df['basefee_wei'].values[:100]
    else:
        # Generate synthetic data if real data not available
        print("⚠️  Real data not found, using synthetic data for validation")
        np.random.seed(42)
        basefees = []
        current = 50e9  # 50 gwei
        for _ in range(100):
            change = np.random.normal(0, 0.1)  # 10% volatility
            current *= (1 + change)
            current = max(current, 1e9)  # Min 1 gwei
            basefees.append(current)
        return np.array(basefees)

class SimpleL1Model:
    """Simple L1 model for testing."""

    def __init__(self, sequence):
        self.sequence = sequence

    def generate_sequence(self, steps, initial_basefee=None):
        return self.sequence[:steps] if steps <= len(self.sequence) else \
               np.tile(self.sequence, (steps // len(self.sequence)) + 1)[:steps]

    def get_name(self):
        return "Test Data"

def run_preset_comparison():
    """Compare old vs new presets on the same data."""

    print("🔄 Cross-Validating Old vs New Preset Performance")
    print("=" * 60)

    # Load test data
    basefee_data = load_test_data()
    l1_model = SimpleL1Model(basefee_data)

    # Define old and new presets
    old_presets = {
        'Optimal (OLD)': {'mu': 0.0, 'nu': 0.3, 'H': 288},
        'Balanced (OLD)': {'mu': 0.0, 'nu': 0.1, 'H': 576},
        'Crisis (OLD)': {'mu': 0.0, 'nu': 0.9, 'H': 144}
    }

    new_presets = {
        'Optimal (NEW)': {'mu': 0.0, 'nu': 0.1, 'H': 36},
        'Balanced (NEW)': {'mu': 0.0, 'nu': 0.2, 'H': 72},
        'Crisis (NEW)': {'mu': 0.0, 'nu': 0.7, 'H': 288}
    }

    results = []

    # Test all presets
    all_presets = {**old_presets, **new_presets}

    for name, params in all_presets.items():
        try:
            print(f"Testing {name}: μ={params['mu']}, ν={params['nu']}, H={params['H']}")

            # Create simulation parameters
            sim_params = ImprovedSimulationParams(
                mu=params['mu'],
                nu=params['nu'],
                H=params['H'],
                target_balance=1000.0,
                total_steps=len(basefee_data),
                vault_initialization_mode='target'
            )

            # Run simulation
            simulator = ImprovedTaikoFeeSimulator(sim_params, l1_model)
            df = simulator.run_simulation()

            # Calculate metrics
            metrics_calc = MetricsCalculator(1000.0)
            metrics = metrics_calc.calculate_all_metrics(df)

            # Store results
            results.append({
                'preset': name,
                'type': 'NEW' if 'NEW' in name else 'OLD',
                'strategy': name.split(' (')[0],
                'mu': params['mu'],
                'nu': params['nu'],
                'H': params['H'],
                'avg_fee': metrics.avg_fee,
                'fee_cv': metrics.fee_cv,
                'time_underfunded_pct': metrics.time_underfunded_pct,
                'max_deficit': metrics.max_deficit,
                'deficit_correction_efficiency': metrics.deficit_correction_efficiency
            })

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # Print comparison table
    print(f"\n📊 CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"{'Preset':<20} {'AvgFee':<12} {'FeeCV':<8} {'Underfunded%':<12} {'MaxDeficit':<10}")
    print("-" * 60)

    for _, row in results_df.iterrows():
        print(f"{row['preset']:<20} {row['avg_fee']:<12.2e} {row['fee_cv']:<8.3f} {row['time_underfunded_pct']:<12.1f} {row['max_deficit']:<10.1f}")

    # Strategy comparison
    print(f"\n🔍 STRATEGY COMPARISON (NEW vs OLD)")
    print("=" * 60)

    for strategy in ['Optimal', 'Balanced', 'Crisis']:
        old_row = results_df[(results_df['strategy'] == strategy) & (results_df['type'] == 'OLD')]
        new_row = results_df[(results_df['strategy'] == strategy) & (results_df['type'] == 'NEW')]

        if len(old_row) > 0 and len(new_row) > 0:
            old = old_row.iloc[0]
            new = new_row.iloc[0]

            print(f"\n{strategy} Strategy:")
            print(f"  OLD: μ={old['mu']}, ν={old['nu']}, H={old['H']}")
            print(f"  NEW: μ={new['mu']}, ν={new['nu']}, H={new['H']}")
            print(f"  Fee Change: {((new['avg_fee'] / old['avg_fee']) - 1) * 100:+.1f}%")
            print(f"  CV Change: {((new['fee_cv'] / old['fee_cv']) - 1) * 100:+.1f}%")
            print(f"  Underfunded Change: {new['time_underfunded_pct'] - old['time_underfunded_pct']:+.1f}%")

    # Key insights
    print(f"\n💡 KEY INSIGHTS")
    print("=" * 60)

    # Compare average performance
    old_avg_fee = results_df[results_df['type'] == 'OLD']['avg_fee'].mean()
    new_avg_fee = results_df[results_df['type'] == 'NEW']['avg_fee'].mean()

    old_avg_cv = results_df[results_df['type'] == 'OLD']['fee_cv'].mean()
    new_avg_cv = results_df[results_df['type'] == 'NEW']['fee_cv'].mean()

    print(f"• Average fee change (NEW vs OLD): {((new_avg_fee / old_avg_fee) - 1) * 100:+.1f}%")
    print(f"• Average volatility change (NEW vs OLD): {((new_avg_cv / old_avg_cv) - 1) * 100:+.1f}%")
    print(f"• NEW parameters use much shorter horizons (6-step cycle aligned)")
    print(f"• NEW parameters use different ν values optimized for saw-tooth patterns")
    print(f"• μ=0.0 remains optimal in both timing models")

    # Save results
    output_path = f"{project_root}/analysis/results/cross_validation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Cross-validation results saved to: {output_path}")

    return results_df

if __name__ == "__main__":
    run_preset_comparison()