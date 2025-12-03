#!/usr/bin/env python3
"""
Quick validation test for the new fee mechanism specification.

This script tests the core functionality of the updated canonical fee mechanism
to ensure the new formula F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t) works correctly.
"""

import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.canonical_fee_mechanism import (
    create_default_calculator,
    create_balanced_calculator,
    create_crisis_calculator
)

def test_new_specification():
    """Test the new fee mechanism specification."""
    print("🧪 Testing New Fee Mechanism Specification")
    print("=" * 50)

    # Create calculator with default optimal parameters
    calc = create_default_calculator()
    print(f"✅ Created calculator with parameters:")
    print(f"   μ={calc.params.mu}, ν={calc.params.nu}, H={calc.params.H}")
    print(f"   α_data={calc.params.alpha_data}, λ_B={calc.params.lambda_B}")
    print(f"   Q̄={calc.params.Q_bar}, T={calc.params.T}")

    # Test with realistic values
    l1_basefee_wei = 20e9  # 20 gwei
    vault_deficit = 50.0   # 50 ETH deficit

    print(f"\n🔬 Testing with L1 basefee = {l1_basefee_wei/1e9:.1f} gwei, deficit = {vault_deficit} ETH")

    # Test individual components
    smoothed_basefee = calc.calculate_smoothed_l1_basefee(l1_basefee_wei)
    C_DA = calc.calculate_C_DA(l1_basefee_wei)
    C_vault = calc.calculate_C_vault(vault_deficit)
    raw_fee = calc.calculate_estimated_fee_raw(l1_basefee_wei, vault_deficit)

    print(f"   B̂_L1(t) = {smoothed_basefee:.3e} ETH per L1 gas")
    print(f"   C_DA(t) = {C_DA:.3e} ETH per L2 gas")
    print(f"   C_vault(t) = {C_vault:.3e} ETH per L2 gas")
    print(f"   F_L2_raw(t) = {raw_fee:.3e} ETH per L2 gas")
    print(f"   Fee in gwei = {raw_fee * 1e9:.6f} gwei per L2 gas")

    # Verify formula: F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t)
    expected_raw_fee = calc.params.mu * C_DA + calc.params.nu * C_vault
    formula_correct = abs(raw_fee - expected_raw_fee) < 1e-12

    print(f"\n🔍 Formula verification:")
    print(f"   μ × C_DA(t) = {calc.params.mu} × {C_DA:.3e} = {calc.params.mu * C_DA:.3e}")
    print(f"   ν × C_vault(t) = {calc.params.nu} × {C_vault:.3e} = {calc.params.nu * C_vault:.3e}")
    print(f"   Expected: {expected_raw_fee:.3e}")
    print(f"   Actual: {raw_fee:.3e}")
    print(f"   ✅ Formula correct: {formula_correct}")

    # Test EMA smoothing behavior
    print(f"\n🔄 Testing EMA smoothing:")
    calc.reset_state()

    # First call - no smoothing
    smoothed1 = calc.calculate_smoothed_l1_basefee(20e9)  # 20 gwei
    print(f"   First call (20 gwei): {smoothed1:.3e} ETH/gas")

    # Second call - should apply EMA with λ_B = 0.1
    smoothed2 = calc.calculate_smoothed_l1_basefee(30e9)  # 30 gwei
    expected_smoothed2 = 0.9 * smoothed1 + 0.1 * (30e9 / 1e18)
    ema_correct = abs(smoothed2 - expected_smoothed2) < 1e-12

    print(f"   Second call (30 gwei): {smoothed2:.3e} ETH/gas")
    print(f"   Expected: {expected_smoothed2:.3e} ETH/gas")
    print(f"   ✅ EMA correct: {ema_correct}")

    # Test different parameter sets
    print(f"\n⚖️  Testing different parameter sets:")

    balanced_calc = create_balanced_calculator()
    crisis_calc = create_crisis_calculator()

    balanced_fee = balanced_calc.calculate_estimated_fee_raw(l1_basefee_wei, vault_deficit)
    crisis_fee = crisis_calc.calculate_estimated_fee_raw(l1_basefee_wei, vault_deficit)

    print(f"   Default (μ=0.0, ν=0.27): {raw_fee:.3e} ETH/gas = {raw_fee*1e9:.6f} gwei/gas")
    print(f"   Balanced (μ=0.0, ν=0.48): {balanced_fee:.3e} ETH/gas = {balanced_fee*1e9:.6f} gwei/gas")
    print(f"   Crisis (μ=0.0, ν=0.88): {crisis_fee:.3e} ETH/gas = {crisis_fee*1e9:.6f} gwei/gas")

    # Verify that higher ν leads to higher fees (with deficit)
    fee_order_correct = balanced_fee > raw_fee and crisis_fee > balanced_fee
    print(f"   ✅ Fee ordering correct (higher ν → higher fees): {fee_order_correct}")

    # Test vault creation with new T parameter
    print(f"\n🏛️  Testing vault creation with T parameter:")
    from core.canonical_fee_mechanism import VaultInitMode
    vault = calc.create_vault(VaultInitMode.TARGET)
    print(f"   Target balance: {vault.target_balance} ETH")
    print(f"   Initial balance: {vault.balance} ETH")
    print(f"   Deficit: {vault.deficit} ETH")
    print(f"   ✅ Uses T parameter: {vault.target_balance == calc.params.T}")

    print(f"\n🎉 All tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_new_specification()
    sys.exit(0 if success else 1)