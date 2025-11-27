#!/usr/bin/env python3
"""
Simple analysis to find cheapest L2 fee configuration.
Uses minimal dependencies for reliable execution.
"""

import pandas as pd
import numpy as np

print("🎯 FINDING CHEAPEST L2 FEE CONFIGURATION")
print("=" * 60)

# Load test data
try:
    df = pd.read_csv('./data/data_cache/recent_low_fees_3hours.csv')
    print(f"✅ Loaded {len(df)} data points")
    print(f"   Basefee range: {df['basefee_gwei'].min():.3f} - {df['basefee_gwei'].max():.3f} gwei")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Take first 50 points for analysis
test_data = df.head(50)
basefees_wei = test_data['basefee_wei'].values

# Corrected parameters
GAS_PER_BATCH = 200000
TXS_PER_BATCH = 100
GAS_PER_TX = GAS_PER_BATCH / TXS_PER_BATCH  # 2000 gas
print(f"✅ Using corrected gas calculation: {GAS_PER_TX} gas per tx")

# Parameter ranges to test
mu_values = [0.0, 0.1, 0.2]
nu_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
H_values = [24, 48, 72, 144]

results = []

print(f"\n🔬 Testing {len(mu_values) * len(nu_values) * len(H_values)} parameter combinations...")

for mu in mu_values:
    for nu in nu_values:
        for H in H_values:
            # Skip invalid combinations
            if mu == 0 and nu == 0:
                continue

            # Simulate fee mechanism
            target_balance = 1.0  # ETH
            vault_balance = target_balance
            fees = []

            for basefee_wei in basefees_wei:
                # L1 cost calculation (corrected)
                l1_cost = (basefee_wei * GAS_PER_TX) / 1e18  # ETH
                l1_component = mu * l1_cost

                # Vault deficit calculation
                vault_deficit = target_balance - vault_balance
                deficit_component = nu * (vault_deficit / H)

                # Total fee
                fee = max(l1_component + deficit_component, 1e-8)  # Min fee
                fees.append(fee)

                # Update vault (simplified)
                vault_balance += fee - (l1_cost if mu > 0 else 0.001)  # Small operational cost

            # Calculate metrics
            avg_fee_eth = np.mean(fees)
            avg_fee_gwei = avg_fee_eth * 1e9
            max_fee_gwei = np.max(fees) * 1e9
            min_fee_gwei = np.min(fees) * 1e9

            results.append({
                'mu': mu, 'nu': nu, 'H': H,
                'avg_fee_gwei': avg_fee_gwei,
                'max_fee_gwei': max_fee_gwei,
                'min_fee_gwei': min_fee_gwei,
                'gas_per_tx': GAS_PER_TX
            })

# Sort by average fee (cheapest first)
results = sorted(results, key=lambda x: x['avg_fee_gwei'])

print(f"\n🏆 ANALYSIS COMPLETE - {len(results)} configurations tested")
print("=" * 60)

# Top 5 cheapest
print("\n🥇 TOP 5 CHEAPEST CONFIGURATIONS:")
print("-" * 50)

for i, config in enumerate(results[:5]):
    print(f"{i+1}. μ={config['mu']}, ν={config['nu']}, H={config['H']}")
    print(f"   Avg Fee: {config['avg_fee_gwei']:.6f} gwei")
    print(f"   Range: {config['min_fee_gwei']:.6f} - {config['max_fee_gwei']:.6f} gwei")
    print()

# Absolute cheapest
cheapest = results[0]

print("🎯 ABSOLUTE CHEAPEST CONFIGURATION:")
print("=" * 50)
print(f"Parameters: μ={cheapest['mu']}, ν={cheapest['nu']}, H={cheapest['H']}")
print(f"Average Fee: {cheapest['avg_fee_gwei']:.6f} gwei")
print(f"Fee Range: {cheapest['min_fee_gwei']:.6f} - {cheapest['max_fee_gwei']:.6f} gwei")
print(f"Gas per TX: {cheapest['gas_per_tx']} gas (corrected)")

# Compare with current "optimal"
current_optimal = next((r for r in results if r['mu'] == 0.0 and r['nu'] == 0.9 and r['H'] == 72), None)

if current_optimal:
    print(f"\n📊 COMPARISON WITH CURRENT 'OPTIMAL' PRESET:")
    print("-" * 50)
    print(f"Current 'optimal' (μ=0.0, ν=0.9, H=72): {current_optimal['avg_fee_gwei']:.6f} gwei")
    print(f"New cheapest: {cheapest['avg_fee_gwei']:.6f} gwei")
    improvement = ((current_optimal['avg_fee_gwei'] - cheapest['avg_fee_gwei']) / current_optimal['avg_fee_gwei'] * 100)
    print(f"Improvement: {improvement:.2f}% cheaper" if improvement > 0 else "Current optimal is already cheapest")

# Generate JavaScript preset code
print(f"\n🔧 JAVASCRIPT PRESET CODE:")
print("=" * 50)
print("'cheapest-fees': {")
print(f"    mu: {cheapest['mu']},")
print(f"    nu: {cheapest['nu']},")
print(f"    H: {cheapest['H']},")
print(f"    description: '💰 CHEAPEST: Absolute minimum L2 fees',")
print(f"    useCase: 'Minimizes L2 fees to {cheapest['avg_fee_gwei']:.6f} gwei average with corrected gas calculation (2000 gas/tx).'")
print("}")

print(f"\n✅ ANALYSIS COMPLETE")

# Output results for script parsing
print(f"\nCHEAPEST_RESULT: mu={cheapest['mu']},nu={cheapest['nu']},H={cheapest['H']},fee={cheapest['avg_fee_gwei']:.6f}")

# Save results
import json
with open('cheapest_analysis_results.json', 'w') as f:
    json.dump({
        'cheapest_config': cheapest,
        'all_results': results
    }, f, indent=2)

print("📁 Results saved to cheapest_analysis_results.json")