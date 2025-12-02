"""
Unit tests for L1CostSmoother

Tests the EMA smoothing implementation against mathematical specifications
from SPECS.md Section 2.3.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from ..core.l1_cost_smoother import L1CostSmoother


class TestL1CostSmoother:
    """Test suite for L1CostSmoother mathematical operations."""

    def test_initialization_valid_params(self):
        """Test smoother initialization with valid parameters."""
        smoother = L1CostSmoother(lambda_c=0.1)
        assert smoother.lambda_c == 0.1
        assert smoother.smoothed_cost is None
        assert not smoother._initialized

        # With initial cost
        smoother = L1CostSmoother(lambda_c=0.5, initial_cost=100.0)
        assert smoother.lambda_c == 0.5
        assert smoother.smoothed_cost == 100.0
        assert smoother._initialized

    def test_initialization_invalid_params(self):
        """Test smoother initialization with invalid parameters."""
        with pytest.raises(ValueError):
            L1CostSmoother(lambda_c=0.0)  # Not in (0,1]

        with pytest.raises(ValueError):
            L1CostSmoother(lambda_c=1.1)  # Not in (0,1]

        with pytest.raises(ValueError):
            L1CostSmoother(lambda_c=-0.1)  # Negative

    def test_single_update_initialization(self):
        """Test that first update initializes the smoother."""
        smoother = L1CostSmoother(lambda_c=0.1)

        # First update should initialize
        result = smoother.update(50.0)
        assert result == 50.0
        assert smoother.smoothed_cost == 50.0
        assert smoother._initialized

    def test_single_update_negative_cost(self):
        """Test error handling for negative costs."""
        smoother = L1CostSmoother(lambda_c=0.1)

        with pytest.raises(ValueError):
            smoother.update(-10.0)

    def test_ema_formula_exact(self):
        """Test EMA formula implementation: Ĉ_L1(t) = (1-λ_C) * Ĉ_L1(t-1) + λ_C * C_L1(t)"""
        lambda_c = 0.3
        smoother = L1CostSmoother(lambda_c=lambda_c, initial_cost=100.0)

        # Apply formula manually
        initial_smoothed = 100.0
        new_cost = 200.0
        expected = (1 - lambda_c) * initial_smoothed + lambda_c * new_cost

        result = smoother.update(new_cost)
        assert_allclose(result, expected, rtol=1e-15)
        assert_allclose(smoother.smoothed_cost, expected, rtol=1e-15)

    def test_ema_convergence(self):
        """Test EMA convergence to constant input."""
        lambda_c = 0.1
        smoother = L1CostSmoother(lambda_c=lambda_c)

        constant_cost = 150.0

        # Apply constant input many times
        for _ in range(100):
            result = smoother.update(constant_cost)

        # Should converge to constant value
        assert_allclose(result, constant_cost, rtol=1e-10)

    def test_ema_different_lambda_values(self):
        """Test EMA behavior with different lambda values."""
        costs = [100.0, 200.0, 150.0]

        # Higher lambda should adapt faster
        smoother_fast = L1CostSmoother(lambda_c=0.9)
        smoother_slow = L1CostSmoother(lambda_c=0.1)

        results_fast = []
        results_slow = []

        for cost in costs:
            results_fast.append(smoother_fast.update(cost))
            results_slow.append(smoother_slow.update(cost))

        # Fast smoother should be closer to latest values
        assert abs(results_fast[-1] - costs[-1]) < abs(results_slow[-1] - costs[-1])

    def test_process_series_empty(self):
        """Test error handling for empty series."""
        smoother = L1CostSmoother(lambda_c=0.1)

        with pytest.raises(ValueError):
            smoother.process_series(np.array([]))

    def test_process_series_negative_values(self):
        """Test error handling for negative values in series."""
        smoother = L1CostSmoother(lambda_c=0.1)

        with pytest.raises(ValueError):
            smoother.process_series(np.array([10.0, -5.0, 20.0]))

    def test_process_series_mathematical_correctness(self):
        """Test process_series produces same results as sequential updates."""
        lambda_c = 0.2
        costs = np.array([100.0, 150.0, 80.0, 120.0, 200.0])

        # Method 1: Sequential updates
        smoother1 = L1CostSmoother(lambda_c=lambda_c)
        sequential_results = []
        for cost in costs:
            sequential_results.append(smoother1.update(cost))

        # Method 2: Batch processing
        smoother2 = L1CostSmoother(lambda_c=lambda_c)
        batch_results = smoother2.process_series(costs)

        # Results should be identical
        assert_allclose(batch_results, sequential_results, rtol=1e-15)

    def test_process_series_manual_calculation(self):
        """Test process_series against manual EMA calculation."""
        lambda_c = 0.4
        costs = np.array([50.0, 100.0, 75.0])

        smoother = L1CostSmoother(lambda_c=lambda_c)
        results = smoother.process_series(costs)

        # Manual calculation
        expected = np.zeros_like(costs)
        expected[0] = costs[0]  # Initialize with first value
        expected[1] = (1 - lambda_c) * expected[0] + lambda_c * costs[1]
        expected[2] = (1 - lambda_c) * expected[1] + lambda_c * costs[2]

        assert_allclose(results, expected, rtol=1e-15)

    def test_process_series_updates_state(self):
        """Test that process_series updates internal state correctly."""
        lambda_c = 0.3
        costs = np.array([10.0, 20.0, 30.0])

        smoother = L1CostSmoother(lambda_c=lambda_c)
        results = smoother.process_series(costs)

        # Internal state should match last result
        assert_allclose(smoother.smoothed_cost, results[-1], rtol=1e-15)
        assert smoother._initialized

    def test_reset_functionality(self):
        """Test reset functionality."""
        smoother = L1CostSmoother(lambda_c=0.1)

        # Initialize and update
        smoother.update(100.0)
        smoother.update(200.0)
        assert smoother._initialized
        assert smoother.smoothed_cost != 100.0

        # Reset without initial cost
        smoother.reset()
        assert not smoother._initialized
        assert smoother.smoothed_cost is None

        # Reset with initial cost
        smoother.reset(initial_cost=50.0)
        assert smoother._initialized
        assert smoother.smoothed_cost == 50.0

    def test_boundary_lambda_values(self):
        """Test behavior at lambda boundary values."""
        # Lambda = 1.0 (maximum adaptation)
        smoother = L1CostSmoother(lambda_c=1.0, initial_cost=100.0)
        result = smoother.update(200.0)
        assert result == 200.0  # Should immediately jump to new value

        # Lambda very close to 0 (minimal adaptation)
        smoother = L1CostSmoother(lambda_c=0.001, initial_cost=100.0)
        result = smoother.update(200.0)
        expected = 0.999 * 100.0 + 0.001 * 200.0
        assert_allclose(result, expected, rtol=1e-15)

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values."""
        smoother = L1CostSmoother(lambda_c=0.1)

        # Very large values
        large_cost = 1e15
        result = smoother.update(large_cost)
        assert result == large_cost
        assert not np.isnan(result)
        assert not np.isinf(result)

        # Very small values
        small_cost = 1e-10
        result = smoother.update(small_cost)
        expected = 0.9 * large_cost + 0.1 * small_cost
        assert_allclose(result, expected, rtol=1e-15)

    def test_string_representations(self):
        """Test string representations."""
        smoother = L1CostSmoother(lambda_c=0.25, initial_cost=100.0)

        str_repr = str(smoother)
        assert "0.25" in str_repr
        assert "100.0" in str_repr
        assert "initialized" in str_repr

        # Uninitialized smoother
        smoother_uninit = L1CostSmoother(lambda_c=0.1)
        str_repr = str(smoother_uninit)
        assert "uninitialized" in str_repr

    def test_get_smoothed_cost(self):
        """Test get_smoothed_cost method."""
        smoother = L1CostSmoother(lambda_c=0.1)

        # Before initialization
        assert smoother.get_smoothed_cost() is None

        # After initialization
        smoother.update(50.0)
        assert smoother.get_smoothed_cost() == 50.0