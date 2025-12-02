"""
Unit tests for FeeController

Tests the fee calculation implementation against mathematical specifications
from SPECS.md Section 3.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from ..core.fee_controller import FeeController


class TestFeeController:
    """Test suite for FeeController mathematical operations."""

    def test_initialization_valid_params(self):
        """Test controller initialization with valid parameters."""
        controller = FeeController(mu=0.5, nu=0.3, horizon_h=144)
        assert controller.mu == 0.5
        assert controller.nu == 0.3
        assert controller.horizon_h == 144
        assert controller.q_bar == 6.9e5  # Default
        assert controller.previous_fee is None

    def test_initialization_invalid_params(self):
        """Test controller initialization with invalid parameters."""
        # Invalid mu
        with pytest.raises(ValueError):
            FeeController(mu=-0.1, nu=0.3, horizon_h=144)
        with pytest.raises(ValueError):
            FeeController(mu=1.1, nu=0.3, horizon_h=144)

        # Invalid nu
        with pytest.raises(ValueError):
            FeeController(mu=0.5, nu=-0.1, horizon_h=144)
        with pytest.raises(ValueError):
            FeeController(mu=0.5, nu=1.1, horizon_h=144)

        # Invalid horizon
        with pytest.raises(ValueError):
            FeeController(mu=0.5, nu=0.3, horizon_h=0)
        with pytest.raises(ValueError):
            FeeController(mu=0.5, nu=0.3, horizon_h=-10)

        # Invalid bounds
        with pytest.raises(ValueError):
            FeeController(mu=0.5, nu=0.3, horizon_h=144, f_min=-1)
        with pytest.raises(ValueError):
            FeeController(mu=0.5, nu=0.3, horizon_h=144, f_min=100, f_max=50)

    def test_raw_basefee_formula_exact(self):
        """Test raw basefee formula: f^raw(t) = μ * Ĉ_L1(t)/Q̄ + ν * D(t)/(H*Q̄)"""
        mu = 0.6
        nu = 0.4
        horizon_h = 72
        q_bar = 6.9e5

        controller = FeeController(mu=mu, nu=nu, horizon_h=horizon_h, q_bar=q_bar)

        smoothed_l1_cost = 0.01  # 0.01 ETH = 1e16 wei
        deficit = 1000.0  # ETH

        # Manual calculation
        l1_component = mu * smoothed_l1_cost / q_bar
        deficit_component = nu * deficit / (horizon_h * q_bar)
        expected = l1_component + deficit_component

        result = controller.calculate_raw_basefee(smoothed_l1_cost, deficit)
        assert_allclose(result, expected, rtol=1e-15)

    def test_raw_basefee_negative_l1_cost(self):
        """Test error handling for negative L1 cost."""
        controller = FeeController(mu=0.5, nu=0.3, horizon_h=144)

        with pytest.raises(ValueError):
            controller.calculate_raw_basefee(-0.01, 100.0)

    def test_raw_basefee_components_isolation(self):
        """Test raw basefee components can be isolated."""
        q_bar = 6.9e5
        controller = FeeController(mu=1.0, nu=0.0, horizon_h=144, q_bar=q_bar)

        # Only L1 component
        l1_cost = 0.02
        result = controller.calculate_raw_basefee(l1_cost, 100.0)
        expected = l1_cost / q_bar
        assert_allclose(result, expected, rtol=1e-15)

        # Only deficit component
        controller = FeeController(mu=0.0, nu=0.5, horizon_h=72, q_bar=q_bar)
        deficit = 500.0
        result = controller.calculate_raw_basefee(0.01, deficit)
        expected = 0.5 * deficit / (72 * q_bar)
        assert_allclose(result, expected, rtol=1e-15)

    def test_apply_bounds_formula(self):
        """Test bounds application: f^clip(t) = min(F_max, max(F_min, f^raw(t)))"""
        f_min = 1e6  # 0.001 gwei
        f_max = 1e12  # 1000 gwei
        controller = FeeController(mu=0.5, nu=0.3, horizon_h=144, f_min=f_min, f_max=f_max)

        # Below minimum
        result = controller.apply_bounds(f_min / 2)
        assert result == f_min

        # Above maximum
        result = controller.apply_bounds(f_max * 2)
        assert result == f_max

        # Within bounds
        mid_value = (f_min + f_max) / 2
        result = controller.apply_bounds(mid_value)
        assert result == mid_value

    def test_rate_limiting_first_fee(self):
        """Test rate limiting with no previous fee."""
        controller = FeeController(mu=0.5, nu=0.3, horizon_h=144, kappa_up=0.1, kappa_down=0.1)

        fee = 1e9  # 1 gwei
        result = controller.apply_rate_limiting(fee)
        assert result == fee
        assert controller.previous_fee == fee

    def test_rate_limiting_formula(self):
        """Test rate limiting formula: F_L2(t-1)(1-κ_↓) ≤ F_L2(t) ≤ F_L2(t-1)(1+κ_↑)"""
        kappa_up = 0.2
        kappa_down = 0.15
        controller = FeeController(
            mu=0.5, nu=0.3, horizon_h=144,
            kappa_up=kappa_up, kappa_down=kappa_down
        )

        # Set previous fee
        previous_fee = 1e9  # 1 gwei
        controller.previous_fee = previous_fee

        # Test upward limit
        high_fee = previous_fee * 2  # Way above limit
        result = controller.apply_rate_limiting(high_fee)
        expected_max = previous_fee * (1 + kappa_up)
        assert_allclose(result, expected_max, rtol=1e-15)

        # Test downward limit
        controller.previous_fee = previous_fee  # Reset
        low_fee = previous_fee * 0.5  # Way below limit
        result = controller.apply_rate_limiting(low_fee)
        expected_min = previous_fee * (1 - kappa_down)
        assert_allclose(result, expected_min, rtol=1e-15)

        # Test within limits
        controller.previous_fee = previous_fee  # Reset
        normal_fee = previous_fee * 1.1  # Within upward limit
        result = controller.apply_rate_limiting(normal_fee)
        assert_allclose(result, normal_fee, rtol=1e-15)

    def test_no_rate_limiting_default(self):
        """Test default behavior with no rate limiting (κ=1.0)."""
        controller = FeeController(mu=0.5, nu=0.3, horizon_h=144)  # Default κ=1.0

        # Set previous fee
        controller.previous_fee = 1e9

        # Large jump should be allowed
        new_fee = 10e9  # 10x increase
        result = controller.apply_rate_limiting(new_fee)
        assert result == new_fee

        # Large drop should be allowed
        controller.previous_fee = 10e9
        new_fee = 1e9  # 90% decrease
        result = controller.apply_rate_limiting(new_fee)
        assert result == new_fee

    def test_calculate_fee_complete_pipeline(self):
        """Test complete fee calculation pipeline."""
        mu = 0.3
        nu = 0.7
        horizon_h = 36
        q_bar = 6.9e5
        f_min = 1e6
        f_max = 1e11

        controller = FeeController(
            mu=mu, nu=nu, horizon_h=horizon_h, q_bar=q_bar,
            f_min=f_min, f_max=f_max, kappa_up=0.1, kappa_down=0.1
        )

        smoothed_l1_cost = 0.005
        deficit = 200.0

        # Manual calculation of expected result
        raw_fee = mu * smoothed_l1_cost / q_bar + nu * deficit / (horizon_h * q_bar)
        clipped_fee = min(f_max, max(f_min, raw_fee))
        # Rate limiting: no previous fee, so should equal clipped_fee

        result = controller.calculate_fee(smoothed_l1_cost, deficit)

        assert_allclose(result, clipped_fee, rtol=1e-15)
        assert controller.previous_fee == result

    def test_batch_revenue_calculation(self):
        """Test batch revenue calculation: R(t) = F_L2(t) * Q(t)"""
        q_bar = 5e5
        controller = FeeController(mu=0.5, nu=0.3, horizon_h=144, q_bar=q_bar)

        basefee = 2e9  # 2 gwei
        expected_revenue = basefee * q_bar

        # Using default gas
        result = controller.calculate_batch_revenue(basefee)
        assert_allclose(result, expected_revenue, rtol=1e-15)

        # Using custom gas
        custom_gas = 3e5
        expected_custom = basefee * custom_gas
        result = controller.calculate_batch_revenue(basefee, custom_gas)
        assert_allclose(result, expected_custom, rtol=1e-15)

    def test_deficit_decay_factor(self):
        """Test deficit decay factor calculation: φ = 1 - ν/H"""
        nu = 0.6
        horizon_h = 120
        controller = FeeController(mu=0.0, nu=nu, horizon_h=horizon_h)

        expected_phi = 1 - nu / horizon_h
        result = controller.get_deficit_decay_factor()

        assert_allclose(result, expected_phi, rtol=1e-15)

    def test_process_series_consistency(self):
        """Test process_series produces consistent results with sequential calls."""
        controller = FeeController(mu=0.4, nu=0.6, horizon_h=72)

        smoothed_costs = np.array([0.001, 0.002, 0.0015, 0.003])
        deficits = np.array([50.0, 80.0, 120.0, 90.0])

        # Sequential processing
        controller.reset_state()
        sequential_results = []
        for l1_cost, deficit in zip(smoothed_costs, deficits):
            sequential_results.append(controller.calculate_fee(l1_cost, deficit))

        # Batch processing
        batch_results = controller.process_series(smoothed_costs, deficits)

        assert_allclose(batch_results, sequential_results, rtol=1e-15)

    def test_process_series_mismatched_lengths(self):
        """Test error handling for mismatched input array lengths."""
        controller = FeeController(mu=0.5, nu=0.3, horizon_h=144)

        with pytest.raises(ValueError):
            controller.process_series(np.array([0.001, 0.002]), np.array([50.0]))

    def test_reset_state_functionality(self):
        """Test state reset functionality."""
        controller = FeeController(mu=0.5, nu=0.3, horizon_h=144)

        # Set some state
        controller.calculate_fee(0.001, 100.0)
        assert controller.previous_fee is not None

        # Reset
        controller.reset_state()
        assert controller.previous_fee is None

    def test_extreme_parameter_combinations(self):
        """Test behavior with extreme but valid parameter combinations."""
        # Very aggressive deficit correction
        controller = FeeController(mu=0.0, nu=0.9, horizon_h=6)
        result = controller.calculate_raw_basefee(0.001, 1000.0)
        assert result > 0
        assert not np.isnan(result)

        # Pure L1 pass-through
        controller = FeeController(mu=1.0, nu=0.0, horizon_h=1000)
        result = controller.calculate_raw_basefee(0.001, 1000.0)
        expected = 0.001 / 6.9e5  # No deficit component
        assert_allclose(result, expected, rtol=1e-15)

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme input values."""
        controller = FeeController(mu=0.5, nu=0.5, horizon_h=100)

        # Very large L1 cost
        large_l1_cost = 10.0  # 10 ETH
        result = controller.calculate_raw_basefee(large_l1_cost, 0.0)
        assert not np.isnan(result)
        assert not np.isinf(result)

        # Very large deficit
        large_deficit = 1e6  # 1M ETH
        result = controller.calculate_raw_basefee(0.001, large_deficit)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_canonical_parameters(self):
        """Test with canonical optimal parameters from SPECS.md."""
        # Optimal parameters: μ=0.0, ν=0.1, H=36
        controller = FeeController(mu=0.0, nu=0.1, horizon_h=36)

        # Should work without errors
        result = controller.calculate_fee(0.002, 500.0)
        assert result >= controller.f_min
        assert result <= controller.f_max

        # Deficit decay factor check
        phi = controller.get_deficit_decay_factor()
        expected_phi = 1 - 0.1 / 36
        assert_allclose(phi, expected_phi, rtol=1e-15)

    def test_string_representations(self):
        """Test string representations."""
        controller = FeeController(mu=0.5, nu=0.3, horizon_h=144)

        str_repr = str(controller)
        assert "μ=0.5" in str_repr or "mu=0.5" in str_repr
        assert "ν=0.3" in str_repr or "nu=0.3" in str_repr
        assert "H=144" in str_repr