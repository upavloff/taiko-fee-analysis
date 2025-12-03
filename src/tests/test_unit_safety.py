"""
Comprehensive Unit Safety Test Suite

This test suite validates the unit safety system and ensures critical
unit conversion bugs cannot occur in the fee mechanism.

Key Test Areas:
1. Unit class validation and conversion accuracy
2. Range checking and bounds validation
3. Error detection for common unit mistakes
4. Integration with canonical fee mechanism
5. Golden value tests for regression detection
6. Edge cases and overflow protection

Critical Safety Goals:
- Prevent ETH/Wei confusion (10^18 factor errors)
- Catch unreasonable fee values immediately
- Ensure consistent behavior across modules
- Detect regressions in fee calculations
"""

import pytest
import numpy as np
import warnings
from decimal import Decimal
from unittest.mock import patch, MagicMock

# Import unit safety system
from src.core.units import (
    Wei, Gwei, ETH, gwei_to_wei, wei_to_gwei, safe_gwei_to_wei, safe_wei_to_gwei,
    validate_fee_range, validate_basefee_range, assert_reasonable_fee,
    assert_no_unit_mismatch, diagnose_unit_mismatch,
    UnitValidationError, UnitOverflowError
)

# Import canonical fee mechanism for integration tests
from src.core.canonical_fee_mechanism import (
    CanonicalTaikoFeeCalculator, FeeParameters,
    create_default_calculator, create_balanced_calculator
)


class TestUnitClasses:
    """Test unit class validation and conversion accuracy."""

    def test_wei_creation_and_validation(self):
        """Test Wei class creation and validation."""
        # Valid wei values
        wei1 = Wei(100000000000000000)  # 0.1 ETH
        assert wei1.value == 100000000000000000

        # Float conversion
        wei2 = Wei(1.5e18)
        assert wei2.value == int(1.5e18)

        # Negative values should fail
        with pytest.raises(UnitValidationError):
            Wei(-100)

        # Overflow protection
        with pytest.raises(UnitOverflowError):
            Wei(10**31)

    def test_gwei_creation_and_validation(self):
        """Test Gwei class creation and validation."""
        # Valid gwei values
        gwei1 = Gwei(50.0)
        assert gwei1.value == 50.0

        # Negative values should fail
        with pytest.raises(UnitValidationError):
            Gwei(-1.0)

        # Non-finite values should fail
        with pytest.raises(UnitValidationError):
            Gwei(float('inf'))

        with pytest.raises(UnitValidationError):
            Gwei(float('nan'))

    def test_eth_creation_and_validation(self):
        """Test ETH class creation and validation."""
        # Valid ETH values
        eth1 = ETH(0.001)
        assert eth1.value == 0.001

        # Negative values should fail
        with pytest.raises(UnitValidationError):
            ETH(-0.1)

        # Large values should warn but not fail
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eth_large = ETH(2.0)  # Large for fee context
            assert len(w) == 1
            assert "Unusually large ETH value" in str(w[0].message)

    def test_unit_conversions_accuracy(self):
        """Test accuracy of unit conversions."""
        # Test gwei <-> wei conversions
        original_gwei = 50.123456
        gwei_obj = Gwei(original_gwei)
        wei_obj = gwei_obj.to_wei()
        back_to_gwei = wei_obj.to_gwei()

        assert abs(back_to_gwei.value - original_gwei) < 1e-6

        # Test ETH <-> wei conversions
        original_eth = 0.001234567
        eth_obj = ETH(original_eth)
        wei_obj = eth_obj.to_wei()
        back_to_eth = wei_obj.to_eth()

        assert abs(back_to_eth.value - original_eth) < 1e-12

        # Test specific conversion values
        assert gwei_to_wei(Gwei(1.0)).value == 1000000000
        assert wei_to_gwei(Wei(1000000000)).value == 1.0

    def test_unit_arithmetic_operations(self):
        """Test arithmetic operations on unit classes."""
        wei1 = Wei(1000000000)  # 1 gwei
        wei2 = Wei(2000000000)  # 2 gwei

        # Addition
        result_add = wei1 + wei2
        assert result_add.value == 3000000000

        # Subtraction
        result_sub = wei2 - wei1
        assert result_sub.value == 1000000000

        # Multiplication
        result_mul = wei1 * 3
        assert result_mul.value == 3000000000

        # Division
        result_div = wei2 / 2
        assert result_div.value == 1000000000

        # Type safety - should fail with wrong types
        with pytest.raises(TypeError):
            wei1 + 1000000000  # Can't add int to Wei

        with pytest.raises(TypeError):
            wei1 + Gwei(1.0)   # Can't add Gwei to Wei

    def test_unit_string_representations(self):
        """Test human-readable string representations."""
        # Wei representations
        wei_small = Wei(100)
        assert "100 wei" in str(wei_small)

        wei_gwei = Wei(1500000000)  # 1.5 gwei
        assert "1.5" in str(wei_gwei) and "gwei" in str(wei_gwei)

        wei_eth = Wei(int(1.5e18))  # 1.5 ETH
        assert "1.5" in str(wei_eth) and "ETH" in str(wei_eth)

        # Gwei representations
        gwei_val = Gwei(42.123456)
        assert "42.123456 gwei" == str(gwei_val)

        # ETH representations
        eth_val = ETH(0.001234)
        assert "0.001234 ETH" == str(eth_val)


class TestRangeValidation:
    """Test range checking and bounds validation."""

    def test_fee_range_validation(self):
        """Test fee range validation catches unreasonable values."""
        # Valid fee ranges should pass
        validate_fee_range(1.0, "test")  # 1 gwei
        validate_fee_range(100.0, "test")  # 100 gwei

        # Too low fees should fail
        with pytest.raises(UnitValidationError) as exc_info:
            validate_fee_range(0.0001, "test")  # 0.0001 gwei
        assert "too low" in str(exc_info.value).lower()
        assert "unit conversion error" in str(exc_info.value)

        # Too high fees should fail
        with pytest.raises(UnitValidationError) as exc_info:
            validate_fee_range(50000.0, "test")  # 50,000 gwei
        assert "too high" in str(exc_info.value).lower()

    def test_basefee_range_validation(self):
        """Test basefee range validation with warnings."""
        # Valid basefee ranges should pass silently
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_basefee_range(50.0, "test")
            assert len(w) == 0

        # Very low basefees should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_basefee_range(0.0001, "test")
            assert len(w) == 1
            assert "very low" in str(w[0].message).lower()

        # Very high basefees should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_basefee_range(5000.0, "test")
            assert len(w) == 1
            assert "very high" in str(w[0].message).lower()

    def test_reasonable_fee_assertions(self):
        """Test reasonable fee assertion helpers."""
        # Normal fees should pass
        assert_reasonable_fee(1.0, "test")

        # Suspiciously small fees should fail
        with pytest.raises(UnitValidationError) as exc_info:
            assert_reasonable_fee(0.00001, "test")  # 0.00001 gwei
        assert "suspiciously small" in str(exc_info.value).lower()

        # Zero fees should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert_reasonable_fee(0.0, "test")
            assert len(w) == 1
            assert "exactly zero" in str(w[0].message).lower()

    def test_unit_mismatch_detection(self):
        """Test unit mismatch detection and assertions."""
        # Values in expected range should pass
        assert_no_unit_mismatch(50.0, (1.0, 100.0), "gwei", "test")

        # Values outside expected range should fail
        with pytest.raises(UnitValidationError) as exc_info:
            assert_no_unit_mismatch(0.001, (1.0, 100.0), "gwei", "test")
        assert "unit mismatch" in str(exc_info.value).lower()
        assert "ETH when wei expected" in str(exc_info.value)


class TestGoldenValueTests:
    """Golden value tests for regression detection."""

    def test_canonical_fee_calculation_golden_values(self):
        """Test fee calculation with pre-calculated expected values."""
        calculator = create_default_calculator()

        # Test scenario 1: 50 gwei L1 basefee, no deficit
        l1_basefee_wei = 50 * 1e9  # 50 gwei
        gas_per_tx = 20000
        l1_cost = calculator.calculate_l1_cost_per_tx(l1_basefee_wei, apply_smoothing=False)
        estimated_fee = calculator.calculate_estimated_fee(l1_cost, vault_deficit=0.0)

        # Expected values (manually calculated)
        expected_l1_cost_eth = (50e9 * 20000) / 1e18  # = 0.001 ETH
        expected_fee_eth = max(0.0 * expected_l1_cost_eth + 0.27 * 0.0 / 492, 1e-8)  # = 1e-8 ETH
        expected_fee_gwei = expected_fee_eth * 1e9  # = 0.01 gwei

        assert abs(l1_cost - expected_l1_cost_eth) < 1e-10
        assert abs(estimated_fee - expected_fee_eth) < 1e-10

        # Test scenario 2: With vault deficit
        vault_deficit_eth = 100.0  # 100 ETH deficit
        estimated_fee_deficit = calculator.calculate_estimated_fee(l1_cost, vault_deficit_eth)

        expected_deficit_component = 0.27 * 100.0 / 492  # ≈ 0.0549 ETH
        expected_fee_deficit_eth = 0.0 * expected_l1_cost_eth + expected_deficit_component

        assert abs(estimated_fee_deficit - expected_fee_deficit_eth) < 1e-6

    def test_unit_conversion_golden_values(self):
        """Test unit conversions with known exact values."""
        # Test exact conversion values
        assert gwei_to_wei(Gwei(1.0)).value == 1_000_000_000
        assert gwei_to_wei(Gwei(50.123)).value == 50_123_000_000

        assert wei_to_gwei(Wei(1_000_000_000)).value == 1.0
        assert abs(wei_to_gwei(Wei(50_123_000_000)).value - 50.123) < 1e-9

        # Test ETH conversions
        assert ETH(1.0).to_wei().value == 1_000_000_000_000_000_000
        assert Wei(1_000_000_000_000_000_000).to_eth().value == 1.0

    def test_regression_detection(self):
        """Test against known good calculation results to detect regressions."""
        calculator = create_default_calculator()

        # Historical test case: July 2022 spike data conditions
        test_scenarios = [
            {
                "name": "Low L1 fee",
                "l1_basefee_gwei": 10.0,
                "vault_deficit_eth": 0.0,
                "expected_l1_cost_eth": 0.0002,  # 10 gwei * 20k gas / 1e18
                "expected_fee_gwei_min": 0.01,   # Min fee floor
            },
            {
                "name": "High L1 fee",
                "l1_basefee_gwei": 200.0,
                "vault_deficit_eth": 0.0,
                "expected_l1_cost_eth": 0.004,   # 200 gwei * 20k gas / 1e18
                "expected_fee_gwei_min": 0.01,   # Still min fee due to mu=0
            },
            {
                "name": "Deficit recovery",
                "l1_basefee_gwei": 50.0,
                "vault_deficit_eth": 1000.0,     # Large deficit
                "expected_l1_cost_eth": 0.001,
                "expected_fee_gwei_approx": 548.8,  # 0.27 * 1000 / 492 * 1e9 ≈ 548.8 gwei
            }
        ]

        for scenario in test_scenarios:
            l1_basefee_wei = scenario["l1_basefee_gwei"] * 1e9
            l1_cost = calculator.calculate_l1_cost_per_tx(l1_basefee_wei, apply_smoothing=False)
            estimated_fee = calculator.calculate_estimated_fee(l1_cost, scenario["vault_deficit_eth"])

            # Validate L1 cost calculation
            assert abs(l1_cost - scenario["expected_l1_cost_eth"]) < 1e-6

            # Validate fee calculation
            fee_gwei = estimated_fee * 1e9
            if "expected_fee_gwei_min" in scenario:
                assert fee_gwei >= scenario["expected_fee_gwei_min"]
            if "expected_fee_gwei_approx" in scenario:
                assert abs(fee_gwei - scenario["expected_fee_gwei_approx"]) < 1.0


class TestIntegrationWithCanonicalModule:
    """Test integration with canonical fee mechanism."""

    def test_canonical_module_unit_safety_integration(self):
        """Test that canonical module properly uses unit safety system."""
        calculator = create_default_calculator()

        # Test with reasonable inputs - should not raise warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            l1_basefee_wei = 50 * 1e9  # 50 gwei
            l1_cost = calculator.calculate_l1_cost_per_tx(l1_basefee_wei)
            fee = calculator.calculate_estimated_fee(l1_cost, 100.0)

            # Should have minimal warnings for reasonable inputs
            unit_warnings = [warning for warning in w if "unit" in str(warning.message).lower()]
            assert len(unit_warnings) == 0

    def test_unit_safety_warnings_for_suspicious_values(self):
        """Test that unit safety system warns about suspicious values."""
        calculator = create_default_calculator()

        # Test with suspiciously high L1 basefee
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            l1_basefee_wei = 5000 * 1e9  # 5000 gwei (very high)
            calculator.calculate_l1_cost_per_tx(l1_basefee_wei)

            # Should warn about high basefee
            basefee_warnings = [w for w in w if "basefee" in str(w.message).lower()]
            assert len(basefee_warnings) > 0

        # Test with suspiciously large deficit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            l1_cost = 0.001  # 1 milliETH
            calculator.calculate_estimated_fee(l1_cost, 50000.0)  # 50,000 ETH deficit

            # Should warn about large deficit
            deficit_warnings = [w for w in w if "deficit" in str(w.message).lower()]
            assert len(deficit_warnings) > 0

    def test_parameter_validation_integration(self):
        """Test parameter validation integration."""
        # Test invalid mu parameter
        with pytest.raises(ValueError):
            FeeParameters(mu=1.5)  # mu > 1.0

        # Test invalid nu parameter
        with pytest.raises(ValueError):
            FeeParameters(nu=-0.1)  # nu < 0.0

        # Test invalid H parameter
        with pytest.raises(ValueError):
            FeeParameters(H=-100)  # H <= 0


class TestEdgeCasesAndOverflow:
    """Test edge cases and overflow protection."""

    def test_overflow_protection_in_conversions(self):
        """Test overflow protection in unit conversions."""
        # Very large gwei value should cause overflow error
        with pytest.raises(UnitOverflowError):
            gwei_to_wei(Gwei(1e20))

        # Very large ETH value should cause overflow error
        with pytest.raises(UnitOverflowError):
            ETH(1e15).to_wei()

    def test_precision_edge_cases(self):
        """Test precision handling in edge cases."""
        # Very small values
        tiny_gwei = Gwei(1e-9)  # 1 nano-gwei
        tiny_wei = tiny_gwei.to_wei()
        assert tiny_wei.value == 1

        # Very large precision
        precise_gwei = Gwei(12.123456789)
        wei_conversion = precise_gwei.to_wei()
        back_to_gwei = wei_conversion.to_gwei()
        assert abs(back_to_gwei.value - precise_gwei.value) < 1e-9

    def test_zero_value_handling(self):
        """Test zero value handling across the system."""
        # Zero values should be valid
        zero_wei = Wei(0)
        zero_gwei = Gwei(0)
        zero_eth = ETH(0)

        assert zero_wei.value == 0
        assert zero_gwei.value == 0.0
        assert zero_eth.value == 0.0

        # Conversions should work
        assert zero_gwei.to_wei().value == 0
        assert zero_wei.to_gwei().value == 0.0

    def test_boundary_value_testing(self):
        """Test boundary values at validation limits."""
        # Minimum reasonable fee
        min_fee_gwei = 0.001
        validate_fee_range(min_fee_gwei, "boundary test")

        # Maximum reasonable fee
        max_fee_gwei = 10000.0
        validate_fee_range(max_fee_gwei, "boundary test")

        # Just below minimum should fail
        with pytest.raises(UnitValidationError):
            validate_fee_range(0.0009, "boundary test")

        # Just above maximum should fail
        with pytest.raises(UnitValidationError):
            validate_fee_range(10001.0, "boundary test")


class TestDiagnosticTools:
    """Test diagnostic and debugging tools."""

    def test_unit_mismatch_diagnosis(self):
        """Test unit mismatch diagnostic tool."""
        # Test gwei diagnosis
        diagnosis_gwei = diagnose_unit_mismatch(0.0001, "gwei")
        assert "LIKELY ISSUE" in diagnosis_gwei
        assert "Too small for gwei" in diagnosis_gwei

        # Test wei diagnosis
        diagnosis_wei = diagnose_unit_mismatch(100, "wei")
        assert "POSSIBLE ISSUE" in diagnosis_wei
        assert "Very small for wei" in diagnosis_wei

        # Test ETH diagnosis
        diagnosis_eth = diagnose_unit_mismatch(1e-12, "eth")
        assert "LIKELY ISSUE" in diagnosis_eth
        assert "Too small for ETH" in diagnosis_eth

    def test_safety_conversion_helpers(self):
        """Test safe conversion helper functions."""
        # Safe conversions should work for valid values
        wei_result = safe_gwei_to_wei(50.0)
        assert isinstance(wei_result, Wei)
        assert wei_result.value == 50_000_000_000

        gwei_result = safe_wei_to_gwei(50_000_000_000)
        assert isinstance(gwei_result, Gwei)
        assert gwei_result.value == 50.0

        # Invalid values should raise appropriate errors
        with pytest.raises(UnitValidationError):
            safe_gwei_to_wei(-10.0)

        with pytest.raises(UnitValidationError):
            safe_wei_to_gwei(-1000)


class TestPerformanceAndMemory:
    """Test performance and memory characteristics."""

    def test_unit_operation_performance(self):
        """Test that unit operations don't introduce significant overhead."""
        import time

        # Baseline: raw arithmetic
        start_time = time.time()
        for i in range(10000):
            result = (50.0 * 1e9 * 20000) / 1e18
        baseline_time = time.time() - start_time

        # With unit safety
        start_time = time.time()
        for i in range(10000):
            basefee_gwei = Gwei(50.0)
            basefee_wei = basefee_gwei.to_wei()
            cost = (basefee_wei.value * 20000) / 1e18
        unit_time = time.time() - start_time

        # Unit operations should not be more than 10x slower
        assert unit_time < baseline_time * 10

    def test_memory_usage(self):
        """Test that unit classes don't use excessive memory."""
        import sys

        # Test Wei memory usage
        wei_obj = Wei(12345678901234567890)
        wei_size = sys.getsizeof(wei_obj)
        assert wei_size < 1000  # Should be small

        # Test Gwei memory usage
        gwei_obj = Gwei(123.456789)
        gwei_size = sys.getsizeof(gwei_obj)
        assert gwei_size < 1000  # Should be small


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])

    # Also run the quick unit tests from the validation module
    print("\\n" + "="*50)
    print("UNIT SAFETY SYSTEM VALIDATION")
    print("="*50)

    # Test basic functionality
    try:
        # Test conversions
        test_gwei = Gwei(50.0)
        test_wei = gwei_to_wei(test_gwei)
        back_to_gwei = wei_to_gwei(test_wei)

        print(f"✅ Conversion test: {test_gwei} -> {test_wei} -> {back_to_gwei}")

        # Test fee validation
        validate_fee_range(10.0, "test")
        print("✅ Fee range validation passed")

        # Test canonical integration
        calculator = create_default_calculator()
        l1_cost = calculator.calculate_l1_cost_per_tx(50e9, apply_smoothing=False)
        fee = calculator.calculate_estimated_fee(l1_cost, 100.0)
        print(f"✅ Canonical integration: L1 cost={l1_cost:.6f} ETH, Fee={fee*1e9:.3f} gwei")

        print("\\n🛡️ Unit safety system is working correctly!")

    except Exception as e:
        print(f"❌ Unit safety system test failed: {e}")
        raise