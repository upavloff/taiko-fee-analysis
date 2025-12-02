#!/usr/bin/env python3
"""Diagnose the critical unit mismatch in fee calculation"""

import sys
import numpy as np
from pathlib import Path

# Add the specs_implementation to path
sys.path.append(str(Path(__file__).parent / "python" / "specs_implementation"))

from core.fee_controller import FeeController

def diagnose_unit_mismatch():
    """Identify and demonstrate the unit mismatch bug"""

    print("🚨 CRITICAL UNIT MISMATCH DIAGNOSIS")
    print("=" * 50)

    # Example: 50 gwei L1 basefee
    basefee_gwei = 50
    basefee_wei = basefee_gwei * 1e9
    gas_per_tx = 2000
    l1_cost_wei = basefee_wei * gas_per_tx  # Wei per batch
    l1_cost_eth = l1_cost_wei / 1e18        # ETH per batch

    print(f"📊 Example: {basefee_gwei} gwei L1 basefee")
    print(f"   L1 cost per batch: {l1_cost_wei:,.0f} wei = {l1_cost_eth:.6f} ETH")
    print()

    # Test fee controller
    controller = FeeController(mu=0.5, nu=0.2, horizon_h=144, q_bar=6.9e5)

    print("🔍 Current (BROKEN) Calculation:")
    print("-" * 40)

    # Current broken calculation (ETH input)
    broken_l1_component_eth_per_gas = controller.mu * l1_cost_eth / controller.q_bar
    broken_l1_component_wei_per_gas = broken_l1_component_eth_per_gas * 1e18
    broken_l1_component_gwei_per_gas = broken_l1_component_wei_per_gas / 1e9

    print(f"   Input: {l1_cost_eth:.6f} ETH (WRONG UNIT)")
    print(f"   L1 component: {broken_l1_component_eth_per_gas:.12f} ETH/gas")
    print(f"   L1 component: {broken_l1_component_wei_per_gas:.6f} wei/gas")
    print(f"   L1 component: {broken_l1_component_gwei_per_gas:.6f} gwei/gas")
    print()

    print("✅ Correct Calculation (should be):")
    print("-" * 40)

    # Correct calculation (WEI input)
    correct_l1_component_wei_per_gas = controller.mu * l1_cost_wei / controller.q_bar
    correct_l1_component_gwei_per_gas = correct_l1_component_wei_per_gas / 1e9

    print(f"   Input: {l1_cost_wei:,.0f} wei (CORRECT UNIT)")
    print(f"   L1 component: {correct_l1_component_wei_per_gas:.6f} wei/gas")
    print(f"   L1 component: {correct_l1_component_gwei_per_gas:.6f} gwei/gas")
    print()

    print("🎯 DIAGNOSIS:")
    print("-" * 15)
    print("❌ Current code passes ETH to fee controller")
    print("❌ Fee controller expects WEI but gets ETH")
    print("❌ Result is 10^18 times too small")
    print("❌ All fees become ~0 wei = 0 gwei")
    print()
    print("✅ FIX: Convert ETH to WEI before passing to fee controller")
    print(f"   Correct factor: {correct_l1_component_gwei_per_gas / broken_l1_component_gwei_per_gas:.0f}x larger")

def test_different_scenarios():
    """Test the fix across different L1 scenarios"""

    print(f"\n📊 TESTING ACROSS SCENARIOS")
    print("=" * 50)

    scenarios = [
        ("Normal", 20),
        ("Moderate", 50),
        ("High", 100),
        ("Crisis", 200)
    ]

    controller = FeeController(mu=0.5, nu=0.2, horizon_h=144, q_bar=6.9e5)
    gas_per_tx = 2000

    print("Scenario   | L1 (gwei) | Broken Fee | Fixed Fee | Factor")
    print("-----------|-----------|------------|-----------|--------")

    for name, basefee_gwei in scenarios:
        basefee_wei = basefee_gwei * 1e9

        # Current broken approach
        l1_cost_eth = (basefee_wei * gas_per_tx) / 1e18
        broken_fee = controller.calculate_raw_basefee(l1_cost_eth, 0)

        # Fixed approach
        l1_cost_wei = basefee_wei * gas_per_tx
        fixed_fee = controller.calculate_raw_basefee(l1_cost_wei, 0)

        factor = fixed_fee / (broken_fee + 1e-20)  # Avoid div by zero

        print(f"{name:10} | {basefee_gwei:9d} | {broken_fee/1e9:10.6f} | {fixed_fee/1e9:9.3f} | {factor:.0f}x")

if __name__ == "__main__":
    diagnose_unit_mismatch()
    test_different_scenarios()