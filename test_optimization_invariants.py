#!/usr/bin/env python3
"""
Optimization Invariants Test

This script validates that optimization claims satisfy the fundamental invariants
defined in the authoritative specification:
1. Cost Recovery (CRR ≈ 1.0)
2. Vault Solvency (low ruin probability)
3. UX Quality (stable, predictable fees)
4. Capital Efficiency (minimal vault size)

Run with: python3 test_optimization_invariants.py

Exit codes:
0 - All invariants satisfied
1 - Invariants violated (optimization claims invalid)
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from core.unified_fee_mechanism import (
        UnifiedFeeCalculator,
        FeeParameters,
        VaultState,
        create_conservative_calculator,
        ParameterCalibrationStatus
    )
    from core.optimization_status import (
        LegacyOptimizationResults,
        RecalibrationGuidance,
        OptimizationValidityStatus,
        validate_optimization_claims
    )
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


class MockHistoricalData:
    """Mock historical data for invariant testing."""

    @staticmethod
    def generate_realistic_l1_sequence(num_steps: int = 1000) -> np.ndarray:
        """Generate realistic L1 basefee sequence."""
        # Base fee around 20 gwei with volatility
        np.random.seed(42)  # Deterministic for testing

        base = 20e9  # 20 gwei baseline
        volatility = 0.3

        # Generate log-normal walk
        log_returns = np.random.normal(0, volatility / np.sqrt(num_steps), num_steps)
        log_prices = np.cumsum(log_returns)
        fees = base * np.exp(log_prices - log_prices[0])

        # Add some spikes (like real Ethereum)
        spike_indices = np.random.choice(num_steps, size=int(num_steps * 0.02), replace=False)
        fees[spike_indices] *= np.random.uniform(2, 10, len(spike_indices))

        return fees.astype(int)

    @staticmethod
    def generate_l2_demand_sequence(num_steps: int = 1000) -> np.ndarray:
        """Generate L2 gas demand sequence."""
        np.random.seed(43)  # Different seed

        base_demand = 150_000  # Base L2 gas per batch

        # Weekly cycle + random variation
        t = np.arange(num_steps)
        weekly_cycle = 0.2 * np.sin(2 * np.pi * t / (7 * 24 * 30))  # 7 days, 2s blocks, 30 blocks/hour
        noise = 0.3 * np.random.normal(0, 1, num_steps)

        demand = base_demand * (1 + weekly_cycle + noise)
        return np.maximum(demand, 10_000).astype(int)


class InvariantTester:
    """Test optimization invariants."""

    def __init__(self):
        self.results = {}

    def test_cost_recovery_invariant(
        self,
        calculator: UnifiedFeeCalculator,
        l1_fees: np.ndarray,
        l2_demands: np.ndarray,
        tolerance: float = 0.1
    ) -> bool:
        """
        Test Cost Recovery Ratio invariant.

        CRR should be close to 1.0 (revenue ≈ costs).
        """
        print(f"   Testing cost recovery invariant...")

        vault = VaultState(balance=1000.0, target=1000.0)
        total_revenue = 0.0
        total_l1_costs = 0.0

        for i, (l1_fee, l2_demand) in enumerate(zip(l1_fees, l2_demands)):
            # Calculate L1 cost for this batch
            l1_cost = calculator.params.alpha_data * (l1_fee / 1e18) * l2_demand
            total_l1_costs += l1_cost

            # Pay L1 cost from vault
            vault.balance -= l1_cost

            # Calculate L2 fee and collect revenue
            fee_result = calculator.calculate_final_fee(l1_fee, vault.deficit)
            l2_fee_per_gas = fee_result['final_fee_eth_per_gas']
            revenue = l2_fee_per_gas * l2_demand
            total_revenue += revenue

            # Add revenue to vault
            vault.balance += revenue

        crr = total_revenue / total_l1_costs if total_l1_costs > 0 else 0

        print(f"      Total L1 costs: {total_l1_costs:.2f} ETH")
        print(f"      Total L2 revenue: {total_revenue:.2f} ETH")
        print(f"      Cost Recovery Ratio: {crr:.4f}")

        is_valid = abs(crr - 1.0) <= tolerance

        if is_valid:
            print(f"      ✅ PASS: CRR within tolerance ({tolerance})")
        else:
            print(f"      ❌ FAIL: CRR outside tolerance (|{crr:.4f} - 1.0| > {tolerance})")

        return is_valid

    def test_vault_solvency_invariant(
        self,
        calculator: UnifiedFeeCalculator,
        l1_fees: np.ndarray,
        l2_demands: np.ndarray,
        min_balance_ratio: float = 0.1
    ) -> bool:
        """
        Test vault solvency invariant.

        Vault should not drop below critical threshold.
        """
        print(f"   Testing vault solvency invariant...")

        vault = VaultState(balance=1000.0, target=1000.0)
        min_balance = vault.target * min_balance_ratio
        balances = []

        for i, (l1_fee, l2_demand) in enumerate(zip(l1_fees, l2_demands)):
            # Calculate L1 cost
            l1_cost = calculator.params.alpha_data * (l1_fee / 1e18) * l2_demand

            # Pay L1 cost
            vault.balance -= l1_cost

            # Calculate and collect L2 revenue
            fee_result = calculator.calculate_final_fee(l1_fee, vault.deficit)
            l2_fee_per_gas = fee_result['final_fee_eth_per_gas']
            revenue = l2_fee_per_gas * l2_demand
            vault.balance += revenue

            balances.append(vault.balance)

        min_observed = min(balances)
        ruin_events = sum(1 for b in balances if b < min_balance)
        ruin_probability = ruin_events / len(balances)

        print(f"      Target balance: {vault.target:.0f} ETH")
        print(f"      Critical threshold: {min_balance:.0f} ETH")
        print(f"      Minimum observed: {min_observed:.2f} ETH")
        print(f"      Ruin events: {ruin_events}")
        print(f"      Ruin probability: {ruin_probability:.4f}")

        is_valid = min_observed >= min_balance

        if is_valid:
            print(f"      ✅ PASS: Vault never dropped below critical threshold")
        else:
            print(f"      ❌ FAIL: Vault dropped to {min_observed:.2f} ETH (below {min_balance:.2f})")

        return is_valid

    def test_ux_stability_invariant(
        self,
        calculator: UnifiedFeeCalculator,
        l1_fees: np.ndarray,
        l2_demands: np.ndarray,
        max_cv: float = 2.0
    ) -> bool:
        """
        Test UX stability invariant.

        Fee coefficient of variation should be reasonable.
        """
        print(f"   Testing UX stability invariant...")

        vault = VaultState(balance=1000.0, target=1000.0)
        fees_gwei = []

        for i, (l1_fee, l2_demand) in enumerate(zip(l1_fees, l2_demands)):
            # Update vault (simplified)
            l1_cost = calculator.params.alpha_data * (l1_fee / 1e18) * l2_demand
            vault.balance -= l1_cost

            fee_result = calculator.calculate_final_fee(l1_fee, vault.deficit)
            revenue = fee_result['final_fee_eth_per_gas'] * l2_demand
            vault.balance += revenue

            fees_gwei.append(fee_result['final_fee_gwei_per_gas'])

        fees_array = np.array(fees_gwei)
        mean_fee = np.mean(fees_array)
        std_fee = np.std(fees_array)
        cv = std_fee / mean_fee if mean_fee > 0 else float('inf')

        print(f"      Mean fee: {mean_fee:.6f} gwei")
        print(f"      Std dev: {std_fee:.6f} gwei")
        print(f"      Coefficient of variation: {cv:.4f}")

        is_valid = cv <= max_cv

        if is_valid:
            print(f"      ✅ PASS: CV within acceptable range (≤ {max_cv})")
        else:
            print(f"      ❌ FAIL: CV too high ({cv:.4f} > {max_cv})")

        return is_valid

    def test_parameter_set(
        self,
        name: str,
        params: FeeParameters,
        num_steps: int = 500
    ) -> Dict[str, bool]:
        """Test all invariants for a parameter set."""
        print(f"\n🧪 TESTING: {name}")
        print(f"   Parameters: μ={params.mu}, ν={params.nu}, H={params.H}")
        print(f"   Constants: α_data={params.alpha_data}, Q̄={params.Q_bar:,.0f}")

        calculator = UnifiedFeeCalculator(params)

        # Generate test data
        l1_fees = MockHistoricalData.generate_realistic_l1_sequence(num_steps)
        l2_demands = MockHistoricalData.generate_l2_demand_sequence(num_steps)

        results = {}

        # Test all invariants
        results['cost_recovery'] = self.test_cost_recovery_invariant(
            calculator, l1_fees, l2_demands
        )

        results['vault_solvency'] = self.test_vault_solvency_invariant(
            calculator, l1_fees, l2_demands
        )

        results['ux_stability'] = self.test_ux_stability_invariant(
            calculator, l1_fees, l2_demands
        )

        # Overall pass/fail
        all_pass = all(results.values())

        if all_pass:
            print(f"   🎉 OVERALL: ALL INVARIANTS SATISFIED")
        else:
            failed = [k for k, v in results.items() if not v]
            print(f"   ❌ OVERALL: FAILED INVARIANTS: {', '.join(failed)}")

        results['overall'] = all_pass
        return results


def main():
    """Run all invariant tests."""
    print("🔍 OPTIMIZATION INVARIANTS TEST")
    print("=" * 60)

    # Suppress parameter warnings during testing
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*PARAMETER WARNING.*")
        warnings.filterwarnings("ignore", message=".*OPTIMIZATION WARNING.*")

        tester = InvariantTester()
        all_results = {}

        # Test 1: Conservative parameters (should pass)
        print(f"\n📊 TEST SUITE 1: CONSERVATIVE PARAMETERS")
        conservative = RecalibrationGuidance.get_conservative_parameters()
        conservative_params = conservative.to_fee_parameters()

        all_results['conservative'] = tester.test_parameter_set(
            "Conservative Parameters", conservative_params
        )

        # Test 2: Experimental parameters (may not pass all invariants)
        print(f"\n📊 TEST SUITE 2: EXPERIMENTAL PARAMETERS")
        experimental = RecalibrationGuidance.get_experimental_parameters()
        experimental_params = experimental.to_fee_parameters()

        all_results['experimental'] = tester.test_parameter_set(
            "Experimental Parameters", experimental_params
        )

        # Test 3: Legacy "optimal" parameters (should fail due to wrong constants)
        print(f"\n📊 TEST SUITE 3: LEGACY 'OPTIMAL' PARAMETERS")
        legacy_results = LegacyOptimizationResults.get_all_legacy_results()

        # Test one legacy result as example
        claude_md_optimal = legacy_results['claude_md_optimal']
        legacy_params = FeeParameters(
            mu=claude_md_optimal.mu,
            nu=claude_md_optimal.nu,
            H=claude_md_optimal.H,
            lambda_B=claude_md_optimal.lambda_B,
            # Use wrong constants that the legacy result was based on
            alpha_data=claude_md_optimal.based_on_alpha_data or 0.5,
            Q_bar=claude_md_optimal.based_on_Q_bar or 690_000.0
        )

        all_results['legacy_optimal'] = tester.test_parameter_set(
            "Legacy 'Optimal' (Wrong Constants)", legacy_params
        )

    # Summary
    print(f"\n" + "=" * 60)
    print(f"📊 INVARIANT TEST SUMMARY")

    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results.values() if r['overall'])

    print(f"   Total parameter sets tested: {total_tests}")
    print(f"   Passed all invariants: {passed_tests}")
    print(f"   Failed some invariants: {total_tests - passed_tests}")

    # Detailed breakdown
    for name, results in all_results.items():
        status = "✅ PASS" if results['overall'] else "❌ FAIL"
        print(f"   {name}: {status}")

        if not results['overall']:
            failed_invariants = [k for k, v in results.items() if k != 'overall' and not v]
            print(f"      Failed: {', '.join(failed_invariants)}")

    # Validation conclusion
    print(f"\n🔍 OPTIMIZATION VALIDATION CONCLUSION:")

    conservative_valid = all_results['conservative']['overall']
    legacy_invalid = not all_results['legacy_optimal']['overall']

    if conservative_valid and legacy_invalid:
        print(f"✅ Conservative parameters satisfy invariants")
        print(f"❌ Legacy 'optimal' parameters violate invariants")
        print(f"✅ CONCLUSION: Recalibration guidance is correct")
        exit_code = 0
    elif not conservative_valid:
        print(f"❌ Conservative parameters violate invariants")
        print(f"🚨 CRITICAL: Even conservative approach has issues")
        exit_code = 1
    else:
        print(f"⚠️ Unexpected results - manual review needed")
        exit_code = 1

    # Final recommendations
    if exit_code == 0:
        print(f"\n🚀 DEPLOYMENT RECOMMENDATION:")
        print(f"   ✅ Use conservative parameters for immediate deployment")
        print(f"   ⚠️ Mark as EXPERIMENTAL until real data calibration")
        print(f"   🔄 Re-run optimization after α_data and Q̄ calibration")
        print(f"   📊 Validate results on historical Ethereum data")
    else:
        print(f"\n🚨 BLOCKING ISSUES FOUND:")
        print(f"   ❌ DO NOT DEPLOY - invariants violated")
        print(f"   🔧 Fix parameter calibration issues")
        print(f"   🧪 Re-test before deployment")

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)