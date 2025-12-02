"""
Unit safety tests - prevent unit mismatch bugs

These tests verify that fee calculations use correct units and catch
conversion errors before they cause silent failures.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.units import (
    Wei, Gwei, EthAmount, GasAmount, WeiPerGas, WeiPerBatch,
    gwei_to_wei, wei_to_gwei, eth_to_wei, wei_to_eth,
    l1_basefee_to_batch_cost, validate_fee_range, validate_l1_cost
)
from core.fee_controller import FeeController


class TestUnitConversions:
    """Test all unit conversion functions"""

    def test_gwei_wei_conversions(self):
        """Test gwei ↔ wei conversions"""
        # 1 gwei = 1e9 wei
        assert gwei_to_wei(Gwei(1.0)) == Wei(1_000_000_000)
        assert wei_to_gwei(Wei(1_000_000_000)) == Gwei(1.0)

        # 50.5 gwei
        assert gwei_to_wei(Gwei(50.5)) == Wei(50_500_000_000)
        assert wei_to_gwei(Wei(50_500_000_000)) == Gwei(50.5)

    def test_eth_wei_conversions(self):
        """Test ETH ↔ wei conversions"""
        # 1 ETH = 1e18 wei
        assert eth_to_wei(EthAmount(1.0)) == Wei(1_000_000_000_000_000_000)
        assert wei_to_eth(Wei(1_000_000_000_000_000_000)) == EthAmount(1.0)

        # 0.001 ETH
        assert eth_to_wei(EthAmount(0.001)) == Wei(1_000_000_000_000_000)
        assert wei_to_eth(Wei(1_000_000_000_000_000)) == EthAmount(0.001)

    def test_l1_basefee_to_batch_cost(self):
        """Test L1 basefee to batch cost conversion"""
        # 50 gwei, 2000 gas/tx, 100 tx/batch
        basefee = gwei_to_wei(Gwei(50.0))  # 50 gwei
        batch_cost = l1_basefee_to_batch_cost(basefee, gas_per_tx=2000, txs_per_batch=100)

        # Expected: 50e9 * 2000 * 100 = 1e16 wei
        assert batch_cost.value == 10_000_000_000_000_000
        assert batch_cost.to_eth() == EthAmount(0.01)

    def test_wei_per_gas_operations(self):
        """Test WeiPerGas operations"""
        fee_rate = WeiPerGas(50_000_000_000)  # 50 gwei/gas

        # Conversions
        assert fee_rate.to_gwei_per_gas() == 50.0
        assert fee_rate.to_eth_per_gas() == 50e-9

        # Multiplication: fee_rate * gas = total_wei
        gas = GasAmount(1000)
        total_cost = fee_rate * gas
        assert total_cost == Wei(50_000_000_000_000)

    def test_wei_per_batch_operations(self):
        """Test WeiPerBatch operations"""
        batch_cost = WeiPerBatch(10_000_000_000_000_000)  # 0.01 ETH

        # Conversions
        assert batch_cost.to_eth() == EthAmount(0.01)

        # Convert to per-gas rate
        q_bar = GasAmount(690_000)
        fee_rate = batch_cost.to_wei_per_gas(q_bar)
        expected_wei_per_gas = 10_000_000_000_000_000 // 690_000
        assert fee_rate.value == expected_wei_per_gas


class TestFeeCalculationUnits:
    """Test fee calculations use correct units"""

    def test_fee_controller_expects_wei(self):
        """Verify fee controller expects wei, not ETH"""
        controller = FeeController(mu=0.5, nu=0.2, horizon_h=144, q_bar=690_000)

        # Test with wei input (correct)
        l1_cost_wei = 100_000_000_000_000  # 0.0001 ETH in wei
        fee_wei = controller.calculate_raw_basefee(l1_cost_wei, deficit=0)

        # Should produce reasonable fee (not tiny)
        fee_gwei = fee_wei / 1e9
        assert fee_gwei > 0.01, f"Fee too small: {fee_gwei} gwei"
        assert fee_gwei < 10, f"Fee too large: {fee_gwei} gwei"

    def test_fee_controller_with_eth_input_fails(self):
        """Demonstrate fee controller fails with ETH input"""
        controller = FeeController(mu=0.5, nu=0.2, horizon_h=144, q_bar=690_000)

        # Test with ETH input (incorrect)
        l1_cost_eth = 0.0001  # Same amount but in ETH
        fee_wei = controller.calculate_raw_basefee(l1_cost_eth, deficit=0)

        # Should produce unreasonably tiny fee
        fee_gwei = fee_wei / 1e9
        assert fee_gwei < 0.0001, f"Fee should be tiny with ETH input: {fee_gwei}"

    def test_realistic_fee_scenarios(self):
        """Test known realistic scenarios"""
        controller = FeeController(mu=0.5, nu=0.2, horizon_h=144, q_bar=690_000)

        scenarios = [
            # (L1 basefee gwei, expected fee range gwei)
            (20, (0.01, 0.1)),    # Normal: ~0.03 gwei
            (50, (0.03, 0.15)),   # Moderate: ~0.07 gwei
            (100, (0.07, 0.25)),  # High: ~0.14 gwei
            (200, (0.14, 0.5)),   # Crisis: ~0.29 gwei
        ]

        for basefee_gwei, (min_fee, max_fee) in scenarios:
            # Convert properly: basefee → batch cost → wei
            basefee_wei = gwei_to_wei(Gwei(basefee_gwei))
            batch_cost = l1_basefee_to_batch_cost(basefee_wei, gas_per_tx=2000)

            # Calculate fee (using wei input)
            fee_wei = controller.calculate_raw_basefee(batch_cost.value, deficit=0)
            fee_gwei = fee_wei / 1e9

            assert min_fee <= fee_gwei <= max_fee, \
                f"Fee {fee_gwei:.3f} gwei not in expected range [{min_fee}, {max_fee}] for {basefee_gwei} gwei L1"


class TestValidationFunctions:
    """Test validation functions catch unrealistic values"""

    def test_fee_range_validation(self):
        """Test fee range validation"""
        # Valid fees
        assert validate_fee_range(WeiPerGas(1_000_000))    # 0.001 gwei - minimum
        assert validate_fee_range(WeiPerGas(50_000_000_000))  # 50 gwei - normal
        assert validate_fee_range(WeiPerGas(1_000_000_000_000))  # 1000 gwei - maximum

        # Invalid fees
        assert not validate_fee_range(WeiPerGas(100))      # 0.0000001 gwei - too small
        assert not validate_fee_range(WeiPerGas(2_000_000_000_000))  # 2000 gwei - too large

    def test_l1_cost_validation(self):
        """Test L1 cost validation"""
        # Valid costs
        assert validate_l1_cost(WeiPerBatch(1_000_000_000_000_000))    # 0.001 ETH
        assert validate_l1_cost(WeiPerBatch(10_000_000_000_000_000))   # 0.01 ETH
        assert validate_l1_cost(WeiPerBatch(100_000_000_000_000_000))  # 0.1 ETH - maximum

        # Invalid costs
        assert not validate_l1_cost(WeiPerBatch(-1000))               # Negative
        assert not validate_l1_cost(WeiPerBatch(200_000_000_000_000_000))  # 0.2 ETH - too large


class TestRegressionPrevention:
    """Specific tests to prevent the original bug"""

    def test_original_bug_detection(self):
        """Test that catches the original ETH/wei mismatch"""
        controller = FeeController(mu=0.5, nu=0.2, horizon_h=144, q_bar=690_000)

        # Simulate the original broken flow
        basefee_wei = 50 * 1e9  # 50 gwei
        gas_per_tx = 2000

        # BROKEN: Convert to ETH (original bug)
        l1_cost_eth = (basefee_wei * gas_per_tx) / 1e18
        broken_fee = controller.calculate_raw_basefee(l1_cost_eth, deficit=0)

        # CORRECT: Keep in wei
        l1_cost_wei = basefee_wei * gas_per_tx
        correct_fee = controller.calculate_raw_basefee(l1_cost_wei, deficit=0)

        # The ratio should be ~1e18 (indicating the bug)
        ratio = correct_fee / (broken_fee + 1e-20)
        assert ratio > 1e15, f"Bug not detected! Ratio: {ratio:.0f} (should be ~1e18)"

        # Correct fee should be reasonable
        assert correct_fee / 1e9 > 0.01, "Correct fee should be > 0.01 gwei"

    def test_zero_fee_detection(self):
        """Test that detects when fees are suspiciously close to zero"""
        controller = FeeController(mu=0.5, nu=0.2, horizon_h=144, q_bar=690_000)

        # Any reasonable L1 cost should produce non-zero fee
        reasonable_l1_costs_wei = [
            20e9 * 2000,    # 20 gwei * 2000 gas
            100e9 * 2000,   # 100 gwei * 2000 gas
            500e9 * 2000,   # 500 gwei * 2000 gas
        ]

        for l1_cost_wei in reasonable_l1_costs_wei:
            fee_wei = controller.calculate_raw_basefee(l1_cost_wei, deficit=0)
            fee_gwei = fee_wei / 1e9

            # Any reasonable L1 cost should produce fee > 0.001 gwei
            assert fee_gwei > 0.001, \
                f"Suspiciously small fee {fee_gwei:.6f} gwei for L1 cost {l1_cost_wei:,.0f} wei"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])