#!/usr/bin/env python3
"""
Diagnose SPECS Fee Calculation

Analyze why fees are stuck at minimum and find the correct scaling.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add python directory to path
sys.path.append('python')

from specs_implementation.core.fee_controller import FeeController

def diagnose_fee_scaling():
    """Diagnose the fee calculation scaling"""
    print("🔍 Diagnosing SPECS Fee Calculation")
    print("="*50)

    # Load some real L1 costs
    file_path = "data_cache/luna_crash_true_peak_contiguous.csv"
    df = pd.read_csv(file_path)
    basefee_wei = df['basefee_wei'].values[:10]  # First 10 values

    # Current conversion parameters
    batch_gas = 200_000
    txs_per_batch = 100
    gas_per_tx = max(batch_gas / txs_per_batch, 200)
    Q_bar_current = 6.9e5

    print(f"📊 Current Parameters:")
    print(f"   Batch gas: {batch_gas:,}")
    print(f"   Txs per batch: {txs_per_batch}")
    print(f"   Gas per tx: {gas_per_tx}")
    print(f"   Q̄ (avg gas per batch): {Q_bar_current:,.0f}")

    print(f"\n📈 Sample basefees:")
    for i, bf in enumerate(basefee_wei[:5]):
        print(f"   {i+1}. {bf:,.0f} wei ({bf/1e9:.3f} gwei)")

    # Convert to L1 costs
    l1_costs_eth = (basefee_wei * gas_per_tx) / 1e18

    print(f"\n💰 L1 Costs (ETH per tx):")
    for i, cost in enumerate(l1_costs_eth[:5]):
        print(f"   {i+1}. {cost:.6f} ETH")

    # Test fee calculation with different Q̄ values
    test_Q_bar_values = [6.9e5, 6.9e4, 6.9e3, 100, 1]  # Different scales

    deficit_scenarios = [0, 10, 100, 1000]  # Different deficit levels

    for Q_bar in test_Q_bar_values:
        print(f"\n🎯 Testing with Q̄ = {Q_bar:.0f}")

        controller = FeeController(
            mu=0.0,  # Focus on deficit component first
            nu=0.1,
            horizon_h=36,
            q_bar=Q_bar,
            f_min=1e6,  # 0.001 gwei minimum
            f_max=1e12  # 1000 gwei maximum
        )

        for deficit in deficit_scenarios:
            # Test with first L1 cost
            l1_cost = l1_costs_eth[0]  # Use first cost value

            raw_fee = controller.calculate_raw_basefee(l1_cost, deficit)
            final_fee = controller.calculate_fee(l1_cost, deficit)

            raw_gwei = raw_fee / 1e9
            final_gwei = final_fee / 1e9

            bounded = "📌" if final_fee == controller.f_min or final_fee == controller.f_max else "✅"

            print(f"   D={deficit:4.0f} ETH: raw={raw_gwei:.6f} gwei, final={final_gwei:.6f} gwei {bounded}")

        # Check what Q̄ would give reasonable fees
        target_fee_gwei = 1.0  # Target 1 gwei fee
        target_fee_wei = target_fee_gwei * 1e9

        # For deficit=0, only L1 component matters: μ * L1_cost / Q̄
        if controller.mu > 0:
            l1_cost = l1_costs_eth[0]
            required_Q_bar = controller.mu * l1_cost / (target_fee_wei / 1e18)
            print(f"   💡 For {target_fee_gwei} gwei fee with μ={controller.mu}: Q̄ should be ≈ {required_Q_bar:.0f}")

    # Try with μ=1.0 to see L1 component scaling
    print(f"\n🧪 Testing with μ=1.0 (pure L1 tracking):")
    controller_pure_l1 = FeeController(
        mu=1.0,  # Pure L1 tracking
        nu=0.0,  # No deficit component
        horizon_h=36,
        q_bar=Q_bar_current,
        f_min=1e6,
        f_max=1e12
    )

    for i in range(3):
        l1_cost = l1_costs_eth[i]
        raw_fee = controller_pure_l1.calculate_raw_basefee(l1_cost, 0)
        final_fee = controller_pure_l1.calculate_fee(l1_cost, 0)

        print(f"   L1={l1_cost:.6f} ETH → raw={raw_fee/1e9:.6f} gwei, final={final_fee/1e9:.6f} gwei")

    # Calculate what the fee should be for cost recovery
    print(f"\n💡 Cost Recovery Analysis:")
    avg_l1_cost = np.mean(l1_costs_eth[:10])
    print(f"   Average L1 cost: {avg_l1_cost:.6f} ETH per tx")

    # For perfect cost recovery: Fee per gas * Q̄ = L1 cost per tx * txs_per_batch
    # So: Fee per gas = (L1 cost per tx * txs_per_batch) / Q̄
    perfect_recovery_fee_per_gas = (avg_l1_cost * txs_per_batch) / Q_bar_current
    perfect_recovery_fee_gwei = perfect_recovery_fee_per_gas * 1e9

    print(f"   For perfect cost recovery:")
    print(f"     Fee per gas: {perfect_recovery_fee_per_gas:.2e} ETH/gas")
    print(f"     Fee per gas: {perfect_recovery_fee_gwei:.6f} gwei")

    if perfect_recovery_fee_gwei < 0.001:
        suggested_Q_bar = (avg_l1_cost * txs_per_batch) / (1e-12)  # For 0.001 gwei
        print(f"     💡 Suggested Q̄ for 0.001 gwei minimum: {suggested_Q_bar:.0f}")

def test_web_interface_integration():
    """Test if the web interface can load correctly"""
    print(f"\n🌐 Testing Web Interface Integration")
    print("="*50)

    # Check if key files exist and have SPECS components
    files_to_check = ['app.js', 'index.html']

    for file_name in files_to_check:
        if os.path.exists(file_name):
            print(f"✅ {file_name} exists")

            if file_name == 'app.js':
                # Check for SPECS components
                with open(file_name, 'r') as f:
                    content = f.read()

                specs_indicators = [
                    'SpecsSimulationEngine',
                    'SPECS.md',
                    'calculate_raw_fee',
                    'specs_implementation'
                ]

                for indicator in specs_indicators:
                    count = content.count(indicator)
                    if count > 0:
                        print(f"   ✅ Found '{indicator}': {count} occurrences")
                    else:
                        print(f"   ❌ Missing '{indicator}'")
        else:
            print(f"❌ {file_name} not found")

if __name__ == "__main__":
    diagnose_fee_scaling()
    test_web_interface_integration()