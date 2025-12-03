#!/usr/bin/env python3
"""
Unit Safety Debugging Utilities

This script provides comprehensive debugging tools for investigating
unit conversion issues and validating fee mechanism calculations.

Key Features:
- Interactive unit mismatch diagnosis
- Fee calculation validation with step-by-step breakdown
- Historical data analysis for unit safety
- Emergency unit check for rapid debugging
- Comprehensive system health checks

Usage:
    python src/utils/unit_debug.py --diagnose-value 0.0001 --alleged-unit gwei
    python src/utils/unit_debug.py --validate-calculation --l1-gwei 50 --deficit-eth 100
    python src/utils/unit_debug.py --system-check
    python src/utils/unit_debug.py --emergency-check
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.units import (
        Wei, Gwei, ETH, gwei_to_wei, wei_to_gwei,
        validate_fee_range, validate_basefee_range, assert_reasonable_fee,
        diagnose_unit_mismatch, create_unit_safety_report,
        UnitValidationError, UnitOverflowError
    )
    from core.canonical_fee_mechanism import (
        CanonicalTaikoFeeCalculator, FeeParameters,
        create_default_calculator, create_balanced_calculator, create_crisis_calculator
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import unit safety modules: {e}")
    MODULES_AVAILABLE = False


class UnitDebugger:
    """Interactive unit safety debugging utility."""

    def __init__(self):
        self.calculator = create_default_calculator() if MODULES_AVAILABLE else None

    def diagnose_value(self, value: float, alleged_unit: str) -> None:
        """Diagnose potential unit mismatches for a value."""
        print("🔍 UNIT MISMATCH DIAGNOSIS")
        print("=" * 50)

        if not MODULES_AVAILABLE:
            print("❌ Unit safety modules not available")
            return

        diagnosis = diagnose_unit_mismatch(value, alleged_unit)
        print(diagnosis)

        # Additional context-specific analysis
        if alleged_unit.lower() == "gwei":
            self._analyze_gwei_context(value)
        elif alleged_unit.lower() == "wei":
            self._analyze_wei_context(value)
        elif alleged_unit.lower() == "eth":
            self._analyze_eth_context(value)

    def _analyze_gwei_context(self, value: float) -> None:
        """Analyze value in gwei context."""
        print("\\nContext Analysis (Gwei):")
        print("-" * 25)

        if value < 0.001:
            print("🚨 CRITICAL: Extremely small for gwei")
            print("   → Likely ETH value passed where gwei expected")
            print(f"   → If this is {value} ETH, it equals {value * 1e9:.6f} gwei")

        elif 0.001 <= value < 0.1:
            print("⚠️  WARNING: Small for typical fee context")
            print("   → Could be correct for very low fee periods")
            print("   → Verify this represents realistic market conditions")

        elif 0.1 <= value <= 1000:
            print("✅ NORMAL: Reasonable fee range")
            print("   → Typical for normal to high fee environments")

        elif 1000 < value <= 10000:
            print("⚠️  WARNING: Very high for typical context")
            print("   → Could be correct for extreme fee spikes")
            print("   → Verify this represents realistic market conditions")

        else:
            print("🚨 CRITICAL: Extremely high for gwei")
            print("   → Likely wei value passed where gwei expected")
            print(f"   → If this is {value} wei, it equals {value / 1e9:.6f} gwei")

    def _analyze_wei_context(self, value: float) -> None:
        """Analyze value in wei context."""
        print("\\nContext Analysis (Wei):")
        print("-" * 25)

        gwei_equiv = value / 1e9
        eth_equiv = value / 1e18

        print(f"As gwei: {gwei_equiv:.6f}")
        print(f"As ETH: {eth_equiv:.6f}")

        if value < 1e15:  # Less than 0.001 ETH
            print("⚠️  WARNING: Very small for wei")
            print("   → Might be ETH or gwei value in wrong units")

        elif 1e15 <= value <= 1e20:  # 0.001 to 100 ETH
            print("✅ NORMAL: Reasonable range for wei amounts")

        else:
            print("⚠️  WARNING: Very large for typical fee context")

    def _analyze_eth_context(self, value: float) -> None:
        """Analyze value in ETH context."""
        print("\\nContext Analysis (ETH):")
        print("-" * 25)

        gwei_equiv = value * 1e9
        wei_equiv = value * 1e18

        print(f"As gwei: {gwei_equiv:.6f}")
        print(f"As wei: {wei_equiv:.0f}")

        if value < 1e-6:  # Less than 0.001 gwei
            print("🚨 CRITICAL: Extremely small for ETH")
            print("   → Likely wei value passed where ETH expected")

        elif 1e-6 <= value <= 0.01:  # 0.001 to 10 gwei
            print("✅ NORMAL: Reasonable fee range in ETH")

        elif 0.01 < value <= 1.0:
            print("⚠️  WARNING: Large for typical fee context")
            print("   → Could be correct for vault balances or extreme scenarios")

        else:
            print("⚠️  WARNING: Very large for fee context")
            print("   → Likely correct for vault balances, unusual for fees")

    def validate_fee_calculation(self, l1_basefee_gwei: float, vault_deficit_eth: float,
                               mu: float = 0.0, nu: float = 0.27, H: int = 492) -> None:
        """Validate fee calculation with step-by-step breakdown."""
        print("🧮 FEE CALCULATION VALIDATION")
        print("=" * 50)

        if not MODULES_AVAILABLE:
            print("❌ Unit safety modules not available")
            return

        try:
            # Step 1: Convert L1 basefee to wei and validate
            print("Step 1: L1 Basefee Conversion & Validation")
            print("-" * 45)
            l1_basefee_wei = l1_basefee_gwei * 1e9
            print(f"Input: {l1_basefee_gwei} gwei")
            print(f"Converted: {l1_basefee_wei:,.0f} wei")

            validate_basefee_range(l1_basefee_gwei, "validation check")
            print("✅ Basefee range check passed")

            # Step 2: Calculate L1 cost per transaction
            print("\\nStep 2: L1 Cost Calculation")
            print("-" * 30)
            l1_cost_eth = self.calculator.calculate_l1_cost_per_tx(l1_basefee_wei, apply_smoothing=False)
            print(f"Gas per tx: {self.calculator.params.gas_per_tx:,.0f}")
            print(f"L1 cost: {l1_cost_eth:.8f} ETH")
            print(f"L1 cost: {l1_cost_eth * 1e9:.6f} gwei equivalent")

            # Step 3: Validate deficit
            print("\\nStep 3: Vault Deficit Validation")
            print("-" * 33)
            print(f"Vault deficit: {vault_deficit_eth:.6f} ETH")
            if vault_deficit_eth > 10000:
                print("⚠️  WARNING: Very large deficit - verify units")
            else:
                print("✅ Deficit amount seems reasonable")

            # Step 4: Calculate fee components
            print("\\nStep 4: Fee Component Calculation")
            print("-" * 35)
            l1_component = mu * l1_cost_eth
            deficit_component = nu * vault_deficit_eth / H

            print(f"Parameters: μ={mu}, ν={nu}, H={H}")
            print(f"L1 component: μ × L1_cost = {mu} × {l1_cost_eth:.8f} = {l1_component:.8f} ETH")
            print(f"Deficit component: ν × deficit/H = {nu} × {vault_deficit_eth:.6f}/{H} = {deficit_component:.8f} ETH")

            # Step 5: Calculate final fee
            print("\\nStep 5: Final Fee Calculation")
            print("-" * 31)
            raw_fee = l1_component + deficit_component
            min_fee = 1e-8  # ETH
            final_fee = max(raw_fee, min_fee)

            print(f"Raw fee: {raw_fee:.8f} ETH")
            print(f"Min fee floor: {min_fee:.8f} ETH")
            print(f"Final fee: {final_fee:.8f} ETH")
            print(f"Final fee: {final_fee * 1e9:.6f} gwei")

            # Step 6: Validate result
            print("\\nStep 6: Result Validation")
            print("-" * 25)
            final_fee_gwei = final_fee * 1e9

            try:
                validate_fee_range(final_fee_gwei, "calculation result")
                print("✅ Fee range validation passed")
            except UnitValidationError as e:
                print(f"❌ Fee range validation failed: {e}")

            try:
                assert_reasonable_fee(final_fee_gwei, "calculation result")
                print("✅ Reasonable fee check passed")
            except UnitValidationError as e:
                print(f"⚠️  Reasonable fee warning: {e}")

            # Verify with canonical calculator
            print("\\nStep 7: Canonical Calculator Verification")
            print("-" * 42)
            canonical_fee = self.calculator.calculate_estimated_fee(l1_cost_eth, vault_deficit_eth)
            canonical_fee_gwei = canonical_fee * 1e9

            print(f"Manual calculation: {final_fee_gwei:.6f} gwei")
            print(f"Canonical result: {canonical_fee_gwei:.6f} gwei")
            print(f"Difference: {abs(final_fee_gwei - canonical_fee_gwei):.6f} gwei")

            if abs(final_fee_gwei - canonical_fee_gwei) < 1e-6:
                print("✅ Calculations match - validation successful")
            else:
                print("❌ Calculations differ - investigate discrepancy")

        except Exception as e:
            print(f"❌ Validation failed with error: {e}")
            import traceback
            traceback.print_exc()

    def run_emergency_check(self) -> None:
        """Run emergency unit check for rapid debugging."""
        print("🚨 EMERGENCY UNIT CHECK")
        print("=" * 50)

        if not MODULES_AVAILABLE:
            print("❌ Unit safety modules not available")
            return

        # Test known values that should work
        test_scenarios = [
            {
                "name": "Basic conversion test",
                "l1_basefee_gwei": 50.0,
                "expected_l1_cost_gwei": 1.0,  # 50 gwei * 20k gas / 1e9 ≈ 1.0 gwei per tx
                "vault_deficit_eth": 0.0,
                "expected_min_fee": True
            },
            {
                "name": "Deficit recovery test",
                "l1_basefee_gwei": 50.0,
                "vault_deficit_eth": 492.0,  # Exactly H ETH deficit
                "expected_fee_gwei": 0.27,  # ν * deficit / H = 0.27 * 492 / 492 = 0.27 gwei
            }
        ]

        all_passed = True

        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\\nTest {i}: {scenario['name']}")
            print("-" * (10 + len(scenario['name'])))

            try:
                # Calculate with canonical implementation
                l1_basefee_wei = scenario["l1_basefee_gwei"] * 1e9
                l1_cost = self.calculator.calculate_l1_cost_per_tx(l1_basefee_wei, apply_smoothing=False)
                fee = self.calculator.calculate_estimated_fee(l1_cost, scenario["vault_deficit_eth"])
                fee_gwei = fee * 1e9

                print(f"  L1 basefee: {scenario['l1_basefee_gwei']} gwei")
                print(f"  L1 cost: {l1_cost:.8f} ETH ({l1_cost * 1e9:.6f} gwei)")
                print(f"  Vault deficit: {scenario['vault_deficit_eth']} ETH")
                print(f"  Calculated fee: {fee_gwei:.6f} gwei")

                # Validate expectations
                if scenario.get("expected_min_fee"):
                    min_fee_gwei = 1e-8 * 1e9  # 0.01 gwei
                    if abs(fee_gwei - min_fee_gwei) < 1e-6:
                        print("  ✅ Fee equals minimum fee floor as expected")
                    else:
                        print(f"  ❌ Expected minimum fee ({min_fee_gwei:.6f} gwei), got {fee_gwei:.6f} gwei")
                        all_passed = False

                if "expected_fee_gwei" in scenario:
                    expected = scenario["expected_fee_gwei"]
                    if abs(fee_gwei - expected) < 0.01:  # 0.01 gwei tolerance
                        print(f"  ✅ Fee matches expected value ({expected:.6f} gwei)")
                    else:
                        print(f"  ❌ Expected {expected:.6f} gwei, got {fee_gwei:.6f} gwei")
                        all_passed = False

                # Check for suspicious values
                if fee_gwei < 0.0001:
                    print(f"  🚨 CRITICAL: Fee suspiciously small ({fee_gwei:.8f} gwei)")
                    all_passed = False

                if fee_gwei > 10000:
                    print(f"  ⚠️  WARNING: Fee very large ({fee_gwei:.6f} gwei)")

            except Exception as e:
                print(f"  ❌ Test failed with error: {e}")
                all_passed = False

        print("\\n" + "=" * 50)
        if all_passed:
            print("✅ ALL EMERGENCY CHECKS PASSED")
            print("   Unit safety system is working correctly")
        else:
            print("❌ EMERGENCY CHECKS FAILED")
            print("   Unit conversion issues detected - investigate immediately")

        return all_passed

    def run_system_health_check(self) -> None:
        """Run comprehensive system health check."""
        print("🏥 SYSTEM HEALTH CHECK")
        print("=" * 50)

        if not MODULES_AVAILABLE:
            print("❌ Unit safety modules not available")
            print("   Check imports and installation")
            return

        health_status = {
            "unit_classes": False,
            "conversions": False,
            "validation": False,
            "canonical_integration": False,
            "edge_cases": False
        }

        # Test 1: Unit classes
        print("\\n1. Testing Unit Classes")
        print("-" * 23)
        try:
            wei_test = Wei(1000000000)
            gwei_test = Gwei(1.0)
            eth_test = ETH(0.001)
            print("✅ Unit class creation successful")
            health_status["unit_classes"] = True
        except Exception as e:
            print(f"❌ Unit class test failed: {e}")

        # Test 2: Conversions
        print("\\n2. Testing Conversions")
        print("-" * 20)
        try:
            gwei_val = Gwei(50.0)
            wei_val = gwei_to_wei(gwei_val)
            back_gwei = wei_to_gwei(wei_val)
            if abs(back_gwei.value - gwei_val.value) < 1e-6:
                print("✅ Conversion accuracy verified")
                health_status["conversions"] = True
            else:
                print("❌ Conversion accuracy failed")
        except Exception as e:
            print(f"❌ Conversion test failed: {e}")

        # Test 3: Validation
        print("\\n3. Testing Validation")
        print("-" * 20)
        try:
            validate_fee_range(10.0, "health check")
            validate_basefee_range(50.0, "health check")
            print("✅ Validation functions working")
            health_status["validation"] = True
        except Exception as e:
            print(f"❌ Validation test failed: {e}")

        # Test 4: Canonical integration
        print("\\n4. Testing Canonical Integration")
        print("-" * 32)
        try:
            calculator = create_default_calculator()
            l1_cost = calculator.calculate_l1_cost_per_tx(50e9, apply_smoothing=False)
            fee = calculator.calculate_estimated_fee(l1_cost, 100.0)
            if 0 < fee < 1.0:  # Reasonable range
                print("✅ Canonical integration working")
                health_status["canonical_integration"] = True
            else:
                print(f"❌ Canonical integration - unreasonable fee: {fee}")
        except Exception as e:
            print(f"❌ Canonical integration test failed: {e}")

        # Test 5: Edge cases
        print("\\n5. Testing Edge Cases")
        print("-" * 20)
        try:
            # Test overflow protection
            try:
                gwei_to_wei(Gwei(1e20))
                print("❌ Overflow protection failed")
            except UnitOverflowError:
                print("✅ Overflow protection working")
                health_status["edge_cases"] = True
        except Exception as e:
            print(f"❌ Edge case test failed: {e}")

        # Overall health summary
        print("\\n" + "=" * 50)
        print("HEALTH SUMMARY")
        print("=" * 50)

        passed_tests = sum(health_status.values())
        total_tests = len(health_status)

        for test_name, status in health_status.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}")

        print(f"\\nOverall: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("🎉 SYSTEM HEALTHY - All unit safety measures operational")
        elif passed_tests >= total_tests * 0.8:
            print("⚠️  SYSTEM MOSTLY HEALTHY - Some issues detected")
        else:
            print("🚨 SYSTEM UNHEALTHY - Multiple critical issues")

        return health_status

    def analyze_historical_data(self, data_file: Optional[str] = None) -> None:
        """Analyze historical data for unit consistency."""
        print("📊 HISTORICAL DATA ANALYSIS")
        print("=" * 50)

        if not data_file:
            print("ℹ️  No data file specified - using sample analysis")
            self._sample_historical_analysis()
            return

        try:
            import pandas as pd
            df = pd.read_csv(data_file)
            print(f"Loaded {len(df)} data points from {data_file}")

            # Analyze basefee columns
            basefee_columns = [col for col in df.columns if 'basefee' in col.lower()]
            print(f"Found basefee columns: {basefee_columns}")

            for col in basefee_columns:
                self._analyze_basefee_column(df, col)

        except Exception as e:
            print(f"❌ Historical data analysis failed: {e}")

    def _sample_historical_analysis(self) -> None:
        """Run sample historical analysis with known data ranges."""
        print("\\nSample Analysis of Known Historical Ranges:")
        print("-" * 45)

        known_ranges = [
            {"period": "UST/Luna Crash (May 2022)", "min_gwei": 7, "max_gwei": 1352},
            {"period": "July 2022 Spike", "min_gwei": 60, "max_gwei": 184},
            {"period": "Recent Low Fees (Nov 2024)", "min_gwei": 0.055, "max_gwei": 0.092},
            {"period": "Normal Operation", "min_gwei": 10, "max_gwei": 50}
        ]

        for range_info in known_ranges:
            print(f"\\n{range_info['period']}:")
            print(f"  Range: {range_info['min_gwei']}-{range_info['max_gwei']} gwei")

            # Test if these ranges would pass validation
            try:
                validate_basefee_range(range_info['min_gwei'], range_info['period'])
                validate_basefee_range(range_info['max_gwei'], range_info['period'])
                print("  ✅ Range passes validation")
            except Exception as e:
                print(f"  ⚠️  Validation issue: {e}")

            # Test fee calculations at these levels
            if MODULES_AVAILABLE:
                try:
                    calculator = create_default_calculator()

                    for test_gwei in [range_info['min_gwei'], range_info['max_gwei']]:
                        l1_cost = calculator.calculate_l1_cost_per_tx(test_gwei * 1e9, apply_smoothing=False)
                        fee = calculator.calculate_estimated_fee(l1_cost, 100.0)  # 100 ETH deficit
                        fee_gwei = fee * 1e9
                        print(f"    {test_gwei} gwei L1 → {fee_gwei:.6f} gwei L2 fee")

                except Exception as e:
                    print(f"  ❌ Fee calculation failed: {e}")

    def _analyze_basefee_column(self, df, column: str) -> None:
        """Analyze a basefee column for unit consistency."""
        values = df[column].dropna()
        print(f"\\nAnalyzing column: {column}")
        print(f"Count: {len(values):,}")
        print(f"Range: {values.min():.6f} - {values.max():.6f}")
        print(f"Mean: {values.mean():.6f}")

        # Try to determine unit
        if values.min() > 1e15:  # Likely wei
            print("  Detected unit: Wei")
            gwei_values = values / 1e9
            print(f"  As gwei: {gwei_values.min():.6f} - {gwei_values.max():.6f}")
        elif values.max() < 1e-6:  # Likely ETH
            print("  Detected unit: ETH")
            gwei_values = values * 1e9
            print(f"  As gwei: {gwei_values.min():.6f} - {gwei_values.max():.6f}")
        else:  # Likely gwei
            print("  Detected unit: Gwei")
            gwei_values = values

        # Check for validation issues
        problematic_count = 0
        for val in gwei_values:
            try:
                validate_basefee_range(val, f"{column} validation")
            except:
                problematic_count += 1

        if problematic_count > 0:
            print(f"  ⚠️  {problematic_count}/{len(values)} values outside reasonable range")
        else:
            print("  ✅ All values pass validation")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Unit Safety Debugging Utilities")

    parser.add_argument("--diagnose-value", type=float,
                       help="Value to diagnose for unit mismatches")
    parser.add_argument("--alleged-unit", type=str, choices=["wei", "gwei", "eth"],
                       help="What unit the value is claimed to be")

    parser.add_argument("--validate-calculation", action="store_true",
                       help="Validate fee calculation with step-by-step breakdown")
    parser.add_argument("--l1-gwei", type=float, default=50.0,
                       help="L1 basefee in gwei for calculation validation")
    parser.add_argument("--deficit-eth", type=float, default=100.0,
                       help="Vault deficit in ETH for calculation validation")
    parser.add_argument("--mu", type=float, default=0.0,
                       help="L1 weight parameter")
    parser.add_argument("--nu", type=float, default=0.27,
                       help="Deficit weight parameter")
    parser.add_argument("--H", type=int, default=492,
                       help="Prediction horizon parameter")

    parser.add_argument("--emergency-check", action="store_true",
                       help="Run emergency unit check")
    parser.add_argument("--system-check", action="store_true",
                       help="Run comprehensive system health check")
    parser.add_argument("--analyze-data", type=str,
                       help="Path to historical data CSV file to analyze")

    args = parser.parse_args()

    debugger = UnitDebugger()

    if args.diagnose_value is not None:
        if not args.alleged_unit:
            print("❌ Error: --alleged-unit required when using --diagnose-value")
            return
        debugger.diagnose_value(args.diagnose_value, args.alleged_unit)

    elif args.validate_calculation:
        debugger.validate_fee_calculation(
            args.l1_gwei, args.deficit_eth, args.mu, args.nu, args.H
        )

    elif args.emergency_check:
        debugger.run_emergency_check()

    elif args.system_check:
        debugger.run_system_health_check()

    elif args.analyze_data:
        debugger.analyze_historical_data(args.analyze_data)

    else:
        # Default: run emergency check
        print("🚀 Running default emergency unit check...")
        debugger.run_emergency_check()

        if MODULES_AVAILABLE:
            print("\\n" + "=" * 50)
            print(create_unit_safety_report())


if __name__ == "__main__":
    main()