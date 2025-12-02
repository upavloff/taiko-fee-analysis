"""
Golden integration tests with known expected values

These tests use carefully calculated expected values to catch any changes
in fee mechanism behavior, including subtle unit conversion bugs.
"""

import pytest
import numpy as np
from core.simulation_engine import SimulationEngine
from core.fee_controller import FeeController
from core.units import gwei_to_wei, l1_basefee_to_batch_cost


class TestGoldenValues:
    """Integration tests with pre-calculated golden values"""

    def test_single_step_fee_calculation(self):
        """Test single step with known inputs → known output"""

        # Known inputs
        mu, nu, H = 0.5, 0.2, 144
        q_bar = 690_000
        l1_basefee_gwei = 50.0
        deficit_eth = 0.0

        # Convert inputs properly
        basefee_wei = gwei_to_wei(l1_basefee_gwei)
        batch_cost = l1_basefee_to_batch_cost(basefee_wei, gas_per_tx=2000)

        # Create controller
        controller = FeeController(mu=mu, nu=nu, horizon_h=H, q_bar=q_bar)

        # Calculate fee
        fee_wei = controller.calculate_raw_basefee(batch_cost.value, deficit_eth)
        fee_gwei = fee_wei / 1e9

        # Golden value calculated manually:
        # L1 component = μ * L1_cost_wei / Q̄
        # L1_cost_wei = 50e9 * 2000 = 1e14 wei
        # L1 component = 0.5 * 1e14 / 690000 = 72463768.116 wei
        # Fee = 72463768.116 wei = 0.072463768 gwei
        expected_fee_gwei = 0.072463768
        tolerance = 0.000001

        assert abs(fee_gwei - expected_fee_gwei) < tolerance, \
            f"Expected {expected_fee_gwei:.6f} gwei, got {fee_gwei:.6f} gwei"

    def test_deficit_component_calculation(self):
        """Test deficit component with known values"""

        mu, nu, H = 0.0, 0.3, 72  # Only deficit component (μ=0)
        q_bar = 690_000
        deficit_eth = 1.0  # 1 ETH deficit

        controller = FeeController(mu=mu, nu=nu, horizon_h=H, q_bar=q_bar)

        fee_wei = controller.calculate_raw_basefee(0, deficit_eth)
        fee_gwei = fee_wei / 1e9

        # Golden value: ν * D / (H * Q̄)
        # = 0.3 * 1e18 / (72 * 690000)
        # = 0.3 * 1e18 / 49680000
        # = 6038647342.995 wei = 6.038647343 gwei
        expected_fee_gwei = 6.038647343
        tolerance = 0.000001

        assert abs(fee_gwei - expected_fee_gwei) < tolerance, \
            f"Expected {expected_fee_gwei:.6f} gwei, got {fee_gwei:.6f} gwei"

    def test_combined_components(self):
        """Test L1 + deficit components together"""

        mu, nu, H = 0.4, 0.25, 144
        q_bar = 690_000
        l1_basefee_gwei = 30.0
        deficit_eth = 0.5

        # Convert inputs
        basefee_wei = gwei_to_wei(l1_basefee_gwei)
        batch_cost = l1_basefee_to_batch_cost(basefee_wei, gas_per_tx=2000)

        controller = FeeController(mu=mu, nu=nu, horizon_h=H, q_bar=q_bar)

        fee_wei = controller.calculate_raw_basefee(batch_cost.value, deficit_eth)
        fee_gwei = fee_wei / 1e9

        # Golden calculation:
        # L1 component = 0.4 * (30e9 * 2000) / 690000 = 0.4 * 6e13 / 690000 = 34782608.696 wei
        # Deficit component = 0.25 * 0.5e18 / (144 * 690000) = 0.25 * 5e17 / 99360000 = 1259645061.728 wei
        # Total = 34782608.696 + 1259645061.728 = 1294427670.424 wei = 1.294427670 gwei
        expected_fee_gwei = 1.294427670
        tolerance = 0.000001

        assert abs(fee_gwei - expected_fee_gwei) < tolerance, \
            f"Expected {expected_fee_gwei:.6f} gwei, got {fee_gwei:.6f} gwei"

    def test_simulation_engine_integration(self):
        """Test full simulation engine with known scenario"""

        # Scenario: Constant 40 gwei L1 for 10 steps
        l1_basefees_gwei = np.full(10, 40.0)
        l1_basefees_wei = [gwei_to_wei(gwei) for gwei in l1_basefees_gwei]
        l1_costs_wei = [l1_basefee_to_batch_cost(wei, gas_per_tx=2000).value for wei in l1_basefees_wei]

        # Create engine
        engine = SimulationEngine(
            mu=0.6, nu=0.15, horizon_h=72, lambda_c=0.2,
            target_vault_balance=1.0, q_bar=690_000
        )

        # Run simulation
        results = engine.simulate_series(np.array(l1_costs_wei))

        # Check first few fees (before dynamics kick in)
        first_fee_gwei = results['basefee_per_gas'][0] / 1e9

        # Golden value for first step (no deficit yet):
        # L1 component = 0.6 * (40e9 * 2000) / 690000 = 0.6 * 8e13 / 690000 = 69565217.391 wei = 0.0696 gwei
        expected_first_fee = 0.069565217
        tolerance = 0.000001

        assert abs(first_fee_gwei - expected_first_fee) < tolerance, \
            f"Expected first fee {expected_first_fee:.6f} gwei, got {first_fee_gwei:.6f} gwei"

        # Sanity checks
        assert len(results['basefee_per_gas']) == 10
        assert all(fee > 0 for fee in results['basefee_per_gas']), "All fees should be positive"
        assert all(fee < 1e12 for fee in results['basefee_per_gas']), "Fees should be reasonable (< 1000 gwei)"


class TestRegressionGoldens:
    """Specific regression tests for the unit mismatch bug"""

    def test_original_bug_scenario(self):
        """Test the exact scenario that revealed the bug"""

        # Original scenario from optimization
        scenarios = {
            "Normal": (20, 0.028986),      # 20 gwei → 0.029 gwei fee
            "Moderate": (50, 0.072464),    # 50 gwei → 0.072 gwei fee
            "Crisis": (150, 0.217391),     # 150 gwei → 0.217 gwei fee
        }

        controller = FeeController(mu=0.5, nu=0.2, horizon_h=144, q_bar=690_000)

        for name, (basefee_gwei, expected_fee_gwei) in scenarios.items():
            # Proper conversion
            basefee_wei = gwei_to_wei(basefee_gwei)
            batch_cost = l1_basefee_to_batch_cost(basefee_wei, gas_per_tx=2000)

            # Calculate fee
            fee_wei = controller.calculate_raw_basefee(batch_cost.value, deficit=0)
            fee_gwei = fee_wei / 1e9

            tolerance = 0.000001
            assert abs(fee_gwei - expected_fee_gwei) < tolerance, \
                f"{name}: Expected {expected_fee_gwei:.6f} gwei, got {fee_gwei:.6f} gwei"

    def test_no_zero_fees_with_reasonable_inputs(self):
        """Ensure no fees are zero with reasonable L1 costs"""

        controller = FeeController(mu=0.3, nu=0.1, horizon_h=144, q_bar=690_000)

        # Test range of reasonable L1 basefees
        test_basefees_gwei = [5, 10, 20, 50, 100, 200, 500, 1000]

        for basefee_gwei in test_basefees_gwei:
            basefee_wei = gwei_to_wei(basefee_gwei)
            batch_cost = l1_basefee_to_batch_cost(basefee_wei, gas_per_tx=2000)

            fee_wei = controller.calculate_raw_basefee(batch_cost.value, deficit=0)
            fee_gwei = fee_wei / 1e9

            # Should never be zero or near-zero
            assert fee_gwei > 0.001, \
                f"Fee too small ({fee_gwei:.6f} gwei) for {basefee_gwei} gwei L1 basefee"

            # Should be proportional to L1 basefee
            expected_ratio = 0.3 * 2000 / 690_000  # μ * gas_per_tx / q_bar
            expected_fee_gwei = basefee_gwei * expected_ratio
            tolerance = expected_fee_gwei * 0.01  # 1% tolerance

            assert abs(fee_gwei - expected_fee_gwei) < tolerance, \
                f"Fee {fee_gwei:.6f} not proportional to L1 basefee {basefee_gwei} gwei"


class TestParameterSensitivity:
    """Test parameter sensitivity with golden values"""

    def test_mu_sensitivity(self):
        """Test how μ affects fees"""

        # Fixed scenario
        basefee_wei = gwei_to_wei(50.0)
        batch_cost = l1_basefee_to_batch_cost(basefee_wei, gas_per_tx=2000)

        mu_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        expected_fees = [0.0, 0.018116, 0.072464, 0.108696, 0.144928]  # Calculated manually

        for mu, expected_fee in zip(mu_values, expected_fees):
            controller = FeeController(mu=mu, nu=0.0, horizon_h=144, q_bar=690_000)
            fee_wei = controller.calculate_raw_basefee(batch_cost.value, deficit=0)
            fee_gwei = fee_wei / 1e9

            tolerance = 0.000001
            assert abs(fee_gwei - expected_fee) < tolerance, \
                f"μ={mu}: Expected {expected_fee:.6f} gwei, got {fee_gwei:.6f} gwei"

    def test_nu_sensitivity(self):
        """Test how ν affects fees with deficit"""

        deficit_eth = 2.0  # 2 ETH deficit
        nu_values = [0.0, 0.1, 0.2, 0.3, 0.5]
        expected_fees = [0.0, 2.415458, 4.830917, 7.246376, 12.077293]  # Calculated manually

        for nu, expected_fee in zip(nu_values, expected_fees):
            controller = FeeController(mu=0.0, nu=nu, horizon_h=144, q_bar=690_000)
            fee_wei = controller.calculate_raw_basefee(0, deficit_eth)
            fee_gwei = fee_wei / 1e9

            tolerance = 0.000001
            assert abs(fee_gwei - expected_fee) < tolerance, \
                f"ν={nu}: Expected {expected_fee:.6f} gwei, got {fee_gwei:.6f} gwei"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])