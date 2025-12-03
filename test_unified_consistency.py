#!/usr/bin/env python3
"""
Unified Fee Mechanism Consistency Test (Python)

This script validates that the unified fee mechanism implementation
matches the authoritative specification exactly.

Run with: python3 test_unified_consistency.py

Exit codes:
0 - All tests pass
1 - Tests fail (implementation inconsistent)
"""

import sys
import os
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from core.unified_fee_mechanism import (
        UnifiedFeeCalculator,
        FeeParameters,
        VaultState,
        create_conservative_calculator,
        create_experimental_calculator,
        ParameterCalibrationStatus
    )
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


class TestRunner:
    """Test runner for unified fee mechanism."""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0

    def assert_equal(self, actual, expected, message, tolerance=1e-10):
        """Assert two values are equal within tolerance."""
        self.tests_run += 1

        if isinstance(expected, float):
            if abs(actual - expected) <= tolerance:
                print(f"✅ PASS: {message}")
                self.tests_passed += 1
            else:
                print(f"❌ FAIL: {message}")
                print(f"   Expected: {expected}")
                print(f"   Actual: {actual}")
                print(f"   Error: {abs(actual - expected)}")
                self.tests_failed += 1
        else:
            if actual == expected:
                print(f"✅ PASS: {message}")
                self.tests_passed += 1
            else:
                print(f"❌ FAIL: {message}")
                print(f"   Expected: {expected}")
                print(f"   Actual: {actual}")
                self.tests_failed += 1

    def assert_true(self, condition, message):
        """Assert condition is true."""
        self.tests_run += 1

        if condition:
            print(f"✅ PASS: {message}")
            self.tests_passed += 1
        else:
            print(f"❌ FAIL: {message}")
            self.tests_failed += 1

    def summary(self):
        """Print test summary and return exit code."""
        print(f"\n📊 TEST SUMMARY")
        print(f"   Tests run: {self.tests_run}")
        print(f"   Passed: {self.tests_passed}")
        print(f"   Failed: {self.tests_failed}")

        if self.tests_failed == 0:
            print(f"🎉 ALL TESTS PASSED - Implementation is consistent")
            return 0
        else:
            print(f"❌ {self.tests_failed} TESTS FAILED - Implementation has issues")
            return 1


def test_parameter_validation():
    """Test parameter validation."""
    print("\n🔧 PARAMETER VALIDATION TESTS")

    runner = TestRunner()

    # Test valid parameters
    try:
        params = FeeParameters(mu=0.5, nu=0.7, H=144)
        runner.assert_true(True, "Valid parameter creation")
    except Exception as e:
        runner.assert_true(False, f"Valid parameter creation failed: {e}")

    # Test invalid mu
    try:
        params = FeeParameters(mu=1.5)  # Invalid
        runner.assert_true(False, "Invalid mu should raise error")
    except ValueError:
        runner.assert_true(True, "Invalid mu correctly raises ValueError")

    # Test invalid nu
    try:
        params = FeeParameters(nu=-0.1)  # Invalid
        runner.assert_true(False, "Invalid nu should raise error")
    except ValueError:
        runner.assert_true(True, "Invalid nu correctly raises ValueError")

    # Test invalid H
    try:
        params = FeeParameters(H=-10)  # Invalid
        runner.assert_true(False, "Invalid H should raise error")
    except ValueError:
        runner.assert_true(True, "Invalid H correctly raises ValueError")

    return runner


def test_component_calculations():
    """Test individual component calculations."""
    print("\n🧮 COMPONENT CALCULATION TESTS")

    runner = TestRunner()

    # Create calculator with known parameters
    params = FeeParameters(
        mu=0.1,
        nu=0.5,
        H=144,
        lambda_B=0.3,
        alpha_data=0.22,
        Q_bar=200_000.0
    )
    calculator = UnifiedFeeCalculator(params)

    # Test C_DA calculation
    l1_basefee_wei = 20e9  # 20 gwei
    l1_basefee_eth = l1_basefee_wei / 1e18

    C_DA = calculator.calculate_C_DA(l1_basefee_wei)
    expected_C_DA = params.alpha_data * l1_basefee_eth

    runner.assert_equal(C_DA, expected_C_DA,
                       f"C_DA calculation: {C_DA:.2e} ETH/gas")

    # Test C_vault calculation
    vault_deficit = 100.0  # 100 ETH
    C_vault = calculator.calculate_C_vault(vault_deficit)
    expected_C_vault = vault_deficit / (params.H * params.Q_bar)

    runner.assert_equal(C_vault, expected_C_vault,
                       f"C_vault calculation: {C_vault:.2e} ETH/gas")

    # Test raw fee calculation
    raw_fee = calculator.calculate_raw_fee(l1_basefee_wei, vault_deficit)
    expected_raw_fee = params.mu * C_DA + params.nu * C_vault

    runner.assert_equal(raw_fee, expected_raw_fee,
                       f"Raw fee calculation: {raw_fee:.2e} ETH/gas")

    return runner


def test_ema_smoothing():
    """Test EMA smoothing behavior."""
    print("\n📊 EMA SMOOTHING TESTS")

    runner = TestRunner()

    params = FeeParameters(lambda_B=0.3)
    calculator = UnifiedFeeCalculator(params)

    # Test initialization
    smoothed1 = calculator.update_smoothed_l1_basefee(10e9)  # 10 gwei
    expected1 = 10e9 / 1e18

    runner.assert_equal(smoothed1, expected1,
                       "EMA initialization with first value")

    # Test EMA update
    smoothed2 = calculator.update_smoothed_l1_basefee(30e9)  # 30 gwei
    expected2 = (1 - 0.3) * (10e9 / 1e18) + 0.3 * (30e9 / 1e18)

    runner.assert_equal(smoothed2, expected2,
                       "EMA update with second value")

    return runner


def test_ux_wrapper():
    """Test UX wrapper (clipping and rate limiting)."""
    print("\n🎨 UX WRAPPER TESTS")

    runner = TestRunner()

    params = FeeParameters(
        F_min=1e-12,
        F_max=1e-6,
        kappa_up=0.1,
        kappa_down=0.1
    )
    calculator = UnifiedFeeCalculator(params)

    # Test clipping - high fee
    high_fee = 1e-3  # Very high
    clipped_high = calculator.apply_clipping(high_fee)
    runner.assert_equal(clipped_high, params.F_max,
                       "Fee clipping (high fee)")

    # Test clipping - low fee
    low_fee = 1e-15  # Very low
    clipped_low = calculator.apply_clipping(low_fee)
    runner.assert_equal(clipped_low, params.F_min,
                       "Fee clipping (low fee)")

    # Test clipping - normal fee
    normal_fee = 1e-9
    clipped_normal = calculator.apply_clipping(normal_fee)
    runner.assert_equal(clipped_normal, normal_fee,
                       "Fee clipping (normal fee)")

    # Test rate limiting initialization
    first_limited = calculator.apply_rate_limiting(1e-9)
    runner.assert_equal(first_limited, 1e-9,
                       "Rate limiting initialization")

    # Test rate limiting - small increase
    small_increase = calculator.apply_rate_limiting(1.05e-9)
    runner.assert_equal(small_increase, 1.05e-9,
                       "Rate limiting (small increase allowed)")

    # Test rate limiting - large increase (should be capped)
    calculator._last_final_fee = 1e-9  # Reset to known value
    large_increase = calculator.apply_rate_limiting(2e-9)  # 100% increase
    expected_capped = 1e-9 * (1 + 0.1)  # Capped at 10% increase
    runner.assert_equal(large_increase, expected_capped,
                       "Rate limiting (large increase capped)")

    return runner


def test_full_pipeline():
    """Test complete fee calculation pipeline."""
    print("\n💰 FULL PIPELINE TESTS")

    runner = TestRunner()

    calculator = create_conservative_calculator()

    # Test with realistic values
    l1_basefee_wei = 20e9  # 20 gwei
    vault_deficit = 100.0  # 100 ETH

    result = calculator.calculate_final_fee(l1_basefee_wei, vault_deficit)

    # Basic sanity checks
    runner.assert_true(result['final_fee_eth_per_gas'] >= 0,
                      "Final fee is non-negative")

    runner.assert_true(result['final_fee_gwei_per_gas'] >= 0,
                      "Final fee in gwei is non-negative")

    runner.assert_true(not (result['final_fee_eth_per_gas'] != result['final_fee_eth_per_gas']),
                      "Final fee is not NaN")

    # Check breakdown components exist
    required_keys = ['raw_fee_eth_per_gas', 'clipped_fee_eth_per_gas',
                     'final_fee_eth_per_gas', 'final_fee_gwei_per_gas',
                     'C_DA', 'C_vault', 'smoothed_l1_basefee']

    for key in required_keys:
        runner.assert_true(key in result,
                          f"Result contains {key}")

    print(f"   Final fee: {result['final_fee_gwei_per_gas']:.6f} gwei/gas")
    print(f"   C_DA: {result['C_DA']:.2e} ETH/gas")
    print(f"   C_vault: {result['C_vault']:.2e} ETH/gas")

    return runner


def test_calibration_warnings():
    """Test that calibration warnings are properly emitted."""
    print("\n⚠️ CALIBRATION WARNING TESTS")

    runner = TestRunner()

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create calculator (should trigger warnings)
        calculator = create_conservative_calculator()

        # Check that warnings were emitted
        warning_messages = [str(warning.message) for warning in w]

        alpha_warning = any("α_data" in msg for msg in warning_messages)
        q_bar_warning = any("Q̄" in msg for msg in warning_messages)
        opt_warning = any("OPTIMIZATION WARNING" in msg for msg in warning_messages)

        runner.assert_true(alpha_warning, "α_data calibration warning emitted")
        runner.assert_true(q_bar_warning, "Q̄ calibration warning emitted")
        runner.assert_true(opt_warning, "Optimization warning emitted")

    return runner


def test_edge_cases():
    """Test edge cases and stress conditions."""
    print("\n⚡ EDGE CASE TESTS")

    runner = TestRunner()

    calculator = create_conservative_calculator()

    edge_cases = [
        {"l1_basefee": 1e6, "deficit": 0, "name": "Very low L1 fee, no deficit"},
        {"l1_basefee": 1000e9, "deficit": 10000, "name": "Very high L1 fee, large deficit"},
        {"l1_basefee": 1e9, "deficit": 0.001, "name": "Normal fee, tiny deficit"},
        {"l1_basefee": 50e9, "deficit": 5000, "name": "High fee, moderate deficit"}
    ]

    for case in edge_cases:
        try:
            result = calculator.calculate_final_fee(case["l1_basefee"], case["deficit"])

            # Basic sanity checks
            is_valid = (
                result['final_fee_eth_per_gas'] >= 0 and
                result['final_fee_gwei_per_gas'] >= 0 and
                not (result['final_fee_eth_per_gas'] != result['final_fee_eth_per_gas'])  # not NaN
            )

            runner.assert_true(is_valid,
                              f"{case['name']}: Valid result ({result['final_fee_gwei_per_gas']:.6f} gwei)")

        except Exception as e:
            runner.assert_true(False,
                              f"{case['name']}: Exception - {str(e)}")

    return runner


def main():
    """Run all tests."""
    print("🔬 UNIFIED FEE MECHANISM CONSISTENCY TEST")
    print("=" * 60)

    # Suppress warnings during testing (we test them separately)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*PARAMETER WARNING.*")
        warnings.filterwarnings("ignore", message=".*OPTIMIZATION WARNING.*")

        test_results = []

        # Run all test suites
        test_results.append(test_parameter_validation())
        test_results.append(test_component_calculations())
        test_results.append(test_ema_smoothing())
        test_results.append(test_ux_wrapper())
        test_results.append(test_full_pipeline())
        test_results.append(test_edge_cases())

    # Run calibration warning tests (with warnings enabled)
    test_results.append(test_calibration_warnings())

    # Aggregate results
    total_tests = sum(r.tests_run for r in test_results)
    total_passed = sum(r.tests_passed for r in test_results)
    total_failed = sum(r.tests_failed for r in test_results)

    print("\n" + "=" * 60)
    print(f"📊 OVERALL TEST SUMMARY")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {total_passed}")
    print(f"   Failed: {total_failed}")

    if total_failed == 0:
        print(f"\n🎉 ALL TESTS PASSED")
        print(f"✅ Unified fee mechanism implementation is CONSISTENT")
        print(f"✅ Ready for deployment (with calibration warnings)")
        return 0
    else:
        print(f"\n❌ {total_failed} TESTS FAILED")
        print(f"🚨 Implementation has INCONSISTENCIES")
        print(f"🚨 DO NOT DEPLOY until issues are resolved")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)