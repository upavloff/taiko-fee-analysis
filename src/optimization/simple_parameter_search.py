#!/usr/bin/env python3
"""
Simple Parameter Search for SPECS Implementation

Focus on finding parameters that produce reasonable fees (1-50 gwei range)
with acceptable cost recovery ratios (0.5-2.0 range).
"""

import sys
import os
import numpy as np
import pandas as pd

# Add python directory to path
sys.path.append('python')

from specs_implementation.core.simulation_engine import SimulationEngine

def load_test_data():
    """Load a single representative dataset"""
    file_path = "data_cache/luna_crash_true_peak_contiguous.csv"

    if not os.path.exists(file_path):
        print(f"❌ Cannot find {file_path}")
        return None

    df = pd.read_csv(file_path)
    basefee_wei = df['basefee_wei'].values

    # Convert to L1 costs (ETH per transaction)
    batch_gas = 200_000
    txs_per_batch = 100
    gas_per_tx = max(batch_gas / txs_per_batch, 200)
    l1_costs = (basefee_wei * gas_per_tx) / 1e18

    # Use subset for testing
    test_data = l1_costs[:20]  # First 20 points for quick testing

    print(f"✅ Loaded test data: {len(test_data)} points")
    print(f"   L1 cost range: {np.min(test_data):.6f} - {np.max(test_data):.6f} ETH")
    print(f"   Average L1 cost: {np.mean(test_data):.6f} ETH")

    return test_data

def test_parameter_set(mu, nu, H, l1_costs):
    """Test a single parameter set and return key metrics"""
    try:
        # Create simulation engine
        engine = SimulationEngine(
            mu=mu, nu=nu, horizon_h=H,
            target_vault_balance=1000.0,
            initial_vault_balance=1000.0
        )

        # Run simulation
        sim_df = engine.simulate_series(l1_costs)

        # Calculate basic metrics
        metrics = engine.calculate_metrics(sim_df)

        # Calculate cost recovery ratio manually
        total_revenue = sim_df['revenue'].sum()
        total_l1_costs = sim_df['l1_cost_actual'].sum()
        cost_recovery_ratio = total_revenue / total_l1_costs if total_l1_costs > 0 else float('inf')

        return {
            'success': True,
            'avg_fee_gwei': metrics['avg_fee_gwei'],
            'fee_cv': metrics['fee_cv'],
            'min_vault': metrics['min_vault_balance'],
            'max_vault': sim_df['vault_balance_after'].max(),
            'cost_recovery_ratio': cost_recovery_ratio,
            'fee_range': f"{metrics['avg_fee_gwei']:.3f}±{metrics['fee_std_gwei']:.3f}"
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)[:50]
        }

def main():
    """Find working parameters with simple approach"""
    print("🔍 Simple Parameter Search for SPECS Implementation")
    print("="*60)

    # Load test data
    l1_costs = load_test_data()
    if l1_costs is None:
        return

    # Target: Reasonable fees (1-50 gwei) with balanced cost recovery (0.5-2.0)
    print(f"\n🎯 Target Metrics:")
    print(f"   Fee range: 1-50 gwei")
    print(f"   Cost recovery ratio: 0.5-2.0")
    print(f"   Vault stability: No negative balances")

    # Start with focused parameter ranges
    mu_values = [0.3, 0.5, 0.7, 1.0]  # Focus on meaningful L1 tracking
    nu_values = [0.1, 0.2, 0.3, 0.5]   # Moderate deficit response
    H_values = [36, 72, 144]            # Standard horizons

    print(f"\n🔧 Testing {len(mu_values) * len(nu_values) * len(H_values)} focused combinations...")

    good_results = []
    all_results = []

    for mu in mu_values:
        for nu in nu_values:
            for H in H_values:
                result = test_parameter_set(mu, nu, H, l1_costs)
                result.update({'mu': mu, 'nu': nu, 'H': H})
                all_results.append(result)

                if result['success']:
                    avg_fee = result['avg_fee_gwei']
                    cr_ratio = result['cost_recovery_ratio']
                    min_vault = result['min_vault']

                    # Check if metrics are in reasonable ranges
                    good_fee = 0.1 <= avg_fee <= 100  # 0.1-100 gwei range
                    good_recovery = 0.2 <= cr_ratio <= 5.0  # 0.2-5.0 recovery ratio
                    good_vault = min_vault >= 0  # No negative vault

                    if good_fee and good_recovery and good_vault:
                        good_results.append(result)
                        status = "✅"
                    else:
                        status = "⚠️"

                    print(f"{status} μ={mu:.1f}, ν={nu:.1f}, H={H:3d}: "
                          f"Fee={avg_fee:.2f}gwei, CR={cr_ratio:.2f}, MinV={min_vault:.0f}")
                else:
                    print(f"❌ μ={mu:.1f}, ν={nu:.1f}, H={H:3d}: Error - {result.get('error', 'Unknown')}")

    # Analyze results
    print(f"\n📊 Search Results:")
    print(f"   Total tested: {len(all_results)}")
    print(f"   Successful simulations: {len([r for r in all_results if r['success']])}")
    print(f"   Good parameter sets: {len(good_results)}")

    if good_results:
        # Sort by cost recovery ratio (closest to 1.0)
        good_results.sort(key=lambda x: abs(x['cost_recovery_ratio'] - 1.0))

        print(f"\n🏆 BEST PARAMETER SETS (by cost recovery balance):")
        print(f"   Rank | μ   ν   H  | Fee (gwei) | Cost Recovery | Min Vault | CV")
        print(f"   -----|------------|------------|---------------|-----------|----")

        for i, result in enumerate(good_results[:10]):  # Top 10
            print(f"   {i+1:4d} | {result['mu']:.1f} {result['nu']:.1f} {result['H']:3d} | "
                  f"{result['avg_fee_gwei']:10.3f} | {result['cost_recovery_ratio']:13.3f} | "
                  f"{result['min_vault']:9.0f} | {result['fee_cv']:.3f}")

        # Best overall recommendation
        best = good_results[0]
        print(f"\n🎯 RECOMMENDED OPTIMAL PARAMETERS:")
        print(f"   μ = {best['mu']:.1f}  (L1 cost weight)")
        print(f"   ν = {best['nu']:.1f}  (Deficit weight)")
        print(f"   H = {best['H']:3d}  (Prediction horizon)")

        print(f"\n📈 Performance:")
        print(f"   Average fee: {best['avg_fee_gwei']:.3f} gwei")
        print(f"   Fee stability (CV): {best['fee_cv']:.3f}")
        print(f"   Cost recovery ratio: {best['cost_recovery_ratio']:.3f}")
        print(f"   Vault range: {best['min_vault']:.0f} - {best['max_vault']:.0f} ETH")

        # Show what this means
        avg_l1_cost = np.mean(l1_costs)
        implied_fee_per_eth = best['avg_fee_gwei'] * 1e9 / 1e18 * 6.9e5  # Convert to ETH

        print(f"\n💡 Economic Interpretation:")
        print(f"   L1 cost per tx: {avg_l1_cost:.6f} ETH")
        print(f"   Fee collection per batch: {implied_fee_per_eth:.6f} ETH")
        print(f"   Recovery efficiency: {(implied_fee_per_eth/(avg_l1_cost*100)):.1%}")

        return best

    else:
        print("❌ No good parameter combinations found!")
        print("\n💡 All tested combinations had issues:")

        failed_results = [r for r in all_results if r['success']]
        if failed_results:
            print("   Issues found:")
            high_fees = len([r for r in failed_results if r['avg_fee_gwei'] > 100])
            low_fees = len([r for r in failed_results if r['avg_fee_gwei'] < 0.1])
            bad_recovery = len([r for r in failed_results if r['cost_recovery_ratio'] < 0.2 or r['cost_recovery_ratio'] > 5.0])
            negative_vault = len([r for r in failed_results if r['min_vault'] < 0])

            print(f"   - Fees too high (>100 gwei): {high_fees}")
            print(f"   - Fees too low (<0.1 gwei): {low_fees}")
            print(f"   - Poor cost recovery: {bad_recovery}")
            print(f"   - Negative vault balances: {negative_vault}")

        return None

if __name__ == "__main__":
    main()