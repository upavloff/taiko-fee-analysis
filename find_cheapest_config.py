#!/usr/bin/env python3
"""
Find the absolute cheapest L2 fee configuration with corrected gas calculation.
Senior developer approach: focused, efficient analysis.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from src.core.improved_simulator import ImprovedTaikoFeeSimulator, ImprovedSimulationParams
    from src.analysis.mechanism_metrics import MetricsCalculator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Running from project root...")
    sys.exit(1)

print("🎯 FINDING CHEAPEST L2 FEE CONFIGURATION")
print("=" * 60)
print("Using corrected implementation: txs_per_batch = 100 → gasPerTx = 2000")

# Load realistic low fee data for baseline
data_file = project_root / 'data/data_cache/recent_low_fees_3hours.csv'
if not data_file.exists():
    print(f"❌ Data file not found: {data_file}")
    sys.exit(1)

df = pd.read_csv(data_file)
print(f"✅ Loaded {len(df)} data points")
print(f"   Basefee range: {df['basefee_gwei'].min():.3f} - {df['basefee_gwei'].max():.3f} gwei")

# Take subset for faster analysis
test_data = df.head(50)  # 50 points for speed
l1_basefees = test_data['basefee_wei'].values

# Focused parameter ranges for cheapest fee optimization
param_ranges = {
    'mu': [0.0, 0.1, 0.2],  # L1 weight (0.0 likely optimal)
    'nu': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],  # Deficit weight
    'H': [24, 48, 72, 144]  # Horizon steps
}

print(f"\n🔬 Testing {np.prod([len(v) for v in param_ranges.values()])} combinations...")

results = []

for mu in param_ranges['mu']:
    for nu in param_ranges['nu']:
        for H in param_ranges['H']:
            # Skip invalid combinations
            if mu == 0 and nu == 0:
                continue

            # Create parameters with corrected gas calculation
            params = ImprovedSimulationParams(
                mu=mu, nu=nu, H=H,
                gas_per_batch=200000,    # Standard L1 batch gas
                txs_per_batch=100,       # CORRECTED: 100 txs per batch
                target_balance=1000.0,   # ETH
                base_demand=100,
                fee_elasticity=0.2,
                total_steps=len(test_data),
                time_step_seconds=12
            )

            # Verify correct gas calculation
            expected_gas_per_tx = params.gas_per_batch / params.txs_per_batch
            assert expected_gas_per_tx == 2000, f"Gas calculation wrong: {expected_gas_per_tx}"

            try:
                # Run simulation
                simulator = ImprovedTaikoFeeSimulator(params, l1_data=l1_basefees)
                df_results = simulator.run_simulation()

                # Calculate key metrics
                avg_fee_eth = df_results['fee'].mean()
                avg_fee_gwei = avg_fee_eth * 1e9
                max_fee_gwei = df_results['fee'].max() * 1e9
                min_fee_gwei = df_results['fee'].min() * 1e9

                # Vault metrics
                avg_deficit = df_results['vault_deficit'].mean()
                max_deficit = df_results['vault_deficit'].max()
                underfunded_pct = (df_results['vault_deficit'] > 0).mean() * 100

                results.append({
                    'mu': mu, 'nu': nu, 'H': H,
                    'avg_fee_eth': avg_fee_eth,
                    'avg_fee_gwei': avg_fee_gwei,
                    'max_fee_gwei': max_fee_gwei,
                    'min_fee_gwei': min_fee_gwei,
                    'avg_deficit': avg_deficit,
                    'max_deficit': max_deficit,
                    'underfunded_pct': underfunded_pct,
                    'gas_per_tx': expected_gas_per_tx
                })

            except Exception as e:
                print(f"❌ Error with μ={mu}, ν={nu}, H={H}: {e}")
                continue

# Convert to DataFrame and analyze
results_df = pd.DataFrame(results)

if len(results_df) == 0:
    print("❌ No valid results generated")
    sys.exit(1)

# Sort by average fee (cheapest first)
results_df = results_df.sort_values('avg_fee_gwei')

print(f"\n🏆 ANALYSIS COMPLETE - {len(results_df)} valid configurations")
print("=" * 60)

# Display top 5 cheapest
print("\n🥇 TOP 5 CHEAPEST CONFIGURATIONS:")
print("-" * 50)

for i, (_, row) in enumerate(results_df.head(5).iterrows()):
    print(f"{i+1}. μ={row['mu']}, ν={row['nu']}, H={row['H']}")
    print(f"   Avg Fee: {row['avg_fee_gwei']:.6f} gwei")
    print(f"   Range: {row['min_fee_gwei']:.6f} - {row['max_fee_gwei']:.6f} gwei")
    print(f"   Underfunded: {row['underfunded_pct']:.1f}% of time")
    print()

# Get the absolute cheapest
cheapest = results_df.iloc[0]

print("🎯 ABSOLUTE CHEAPEST CONFIGURATION:")
print("=" * 50)
print(f"Parameters: μ={cheapest['mu']}, ν={cheapest['nu']}, H={cheapest['H']}")
print(f"Average Fee: {cheapest['avg_fee_gwei']:.6f} gwei ({cheapest['avg_fee_eth']:.2e} ETH)")
print(f"Fee Range: {cheapest['min_fee_gwei']:.6f} - {cheapest['max_fee_gwei']:.6f} gwei")
print(f"Gas per TX: {cheapest['gas_per_tx']} gas (corrected)")
print(f"Vault Underfunded: {cheapest['underfunded_pct']:.1f}% of time")

# Compare with current "optimal" (μ=0.0, ν=0.9, H=72)
current_optimal = results_df[(results_df['mu'] == 0.0) & (results_df['nu'] == 0.9) & (results_df['H'] == 72)]

if len(current_optimal) > 0:
    current = current_optimal.iloc[0]
    print(f"\n📊 COMPARISON WITH CURRENT 'OPTIMAL' PRESET:")
    print("-" * 50)
    print(f"Current 'optimal' (μ=0.0, ν=0.9, H=72): {current['avg_fee_gwei']:.6f} gwei")
    print(f"New cheapest: {cheapest['avg_fee_gwei']:.6f} gwei")
    improvement = ((current['avg_fee_gwei'] - cheapest['avg_fee_gwei']) / current['avg_fee_gwei'] * 100)
    print(f"Improvement: {improvement:.2f}% cheaper" if improvement > 0 else "No improvement")

# Generate preset configuration
print(f"\n🔧 RECOMMENDED PRESET CODE:")
print("=" * 50)
print("'cheapest-fees': {")
print(f"    mu: {cheapest['mu']},")
print(f"    nu: {cheapest['nu']},")
print(f"    H: {cheapest['H']},")
print(f"    description: '💰 CHEAPEST: Absolute minimum L2 fees',")
print(f"    useCase: 'Minimizes L2 fees to {cheapest['avg_fee_gwei']:.6f} gwei average. Vault underfunded {cheapest['underfunded_pct']:.1f}% of time with corrected gas calculation.'")
print("}")

print(f"\n✅ ANALYSIS COMPLETE")
print(f"📁 Results saved to: cheapest_config_results.csv")

# Save results
results_df.to_csv(project_root / 'cheapest_config_results.csv', index=False)

# Output the cheapest config for easy parsing
print(f"\nCHEAPEST_CONFIG: {cheapest['mu']},{cheapest['nu']},{cheapest['H']},{cheapest['avg_fee_gwei']:.6f}")