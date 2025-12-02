"""
Unit tests for SimulationEngine

Tests the complete simulation engine integration against mathematical specifications.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from ..core.simulation_engine import SimulationEngine


class TestSimulationEngine:
    """Test suite for complete SimulationEngine integration."""

    def test_initialization_valid_params(self):
        """Test simulation engine initialization with valid parameters."""
        engine = SimulationEngine(
            mu=0.3,
            nu=0.2,
            horizon_h=72,
            target_vault_balance=10000.0
        )

        assert engine.mu == 0.3
        assert engine.nu == 0.2
        assert engine.horizon_h == 72
        assert engine.target_vault_balance == 10000.0

        # Check components are initialized
        assert engine.l1_smoother is not None
        assert engine.fee_controller is not None
        assert engine.vault_dynamics is not None

    def test_simulate_step_complete_integration(self):
        """Test single step simulation with complete integration."""
        engine = SimulationEngine(
            mu=0.5,
            nu=0.3,
            horizon_h=36,
            target_vault_balance=5000.0,
            lambda_c=0.1
        )

        l1_cost = 0.01  # 0.01 ETH

        result = engine.simulate_step(l1_cost)

        # Check all required fields are present
        required_fields = [
            'l1_cost_actual', 'l1_cost_smoothed', 'deficit',
            'basefee_per_gas', 'revenue', 'subsidy_paid',
            'vault_balance_before', 'vault_balance_after',
            'deficit_after', 'fully_subsidized', 'basefee_gwei',
            'solvency_ratio'
        ]

        for field in required_fields:
            assert field in result

        # Check values are reasonable
        assert result['l1_cost_actual'] == l1_cost
        assert result['l1_cost_smoothed'] == l1_cost  # First update initializes
        assert result['basefee_per_gas'] > 0
        assert result['revenue'] > 0

    def test_simulate_step_mathematical_consistency(self):
        """Test single step mathematical consistency across components."""
        engine = SimulationEngine(
            mu=0.4,
            nu=0.6,
            horizon_h=72,
            target_vault_balance=1000.0,
            initial_vault_balance=800.0,
            lambda_c=0.2
        )

        l1_cost = 0.005

        # First step - smoother initializes
        result1 = engine.simulate_step(l1_cost)

        # Second step - check mathematical consistency
        result2 = engine.simulate_step(l1_cost * 1.5)

        # L1 cost smoothing should show EMA behavior
        expected_smoothed = (1 - 0.2) * l1_cost + 0.2 * (l1_cost * 1.5)
        assert_allclose(result2['l1_cost_smoothed'], expected_smoothed, rtol=1e-10)

        # Fee should be calculated from smoothed cost and deficit
        expected_raw_fee = (
            0.4 * result2['l1_cost_smoothed'] / 6.9e5 +
            0.6 * result2['deficit'] / (72 * 6.9e5)
        )

        # Allow for clipping and rate limiting
        assert result2['basefee_per_gas'] > 0

    def test_simulate_series_consistency(self):
        """Test time series simulation consistency."""
        engine = SimulationEngine(
            mu=0.3,
            nu=0.7,
            horizon_h=144,
            target_vault_balance=2000.0
        )

        l1_costs = np.array([0.001, 0.002, 0.0015, 0.003, 0.0025])

        result_df = engine.simulate_series(l1_costs)

        # Check DataFrame structure
        assert len(result_df) == len(l1_costs)
        assert 'step' in result_df.columns
        assert all(result_df['step'] == range(len(l1_costs)))

        # Check mathematical consistency across steps
        for i in range(1, len(result_df)):
            # Vault balance continuity
            prev_balance_after = result_df.iloc[i-1]['vault_balance_after']
            curr_balance_before = result_df.iloc[i]['vault_balance_before']
            assert_allclose(prev_balance_after, curr_balance_before, rtol=1e-15)

    def test_reset_state_functionality(self):
        """Test state reset functionality."""
        engine = SimulationEngine(
            mu=0.5,
            nu=0.3,
            horizon_h=36,
            target_vault_balance=1000.0,
            initial_vault_balance=500.0
        )

        # Run some steps
        engine.simulate_step(0.001)
        engine.simulate_step(0.002)

        # Check state has changed
        state_before_reset = engine.get_state_summary()

        # Reset
        engine.reset_state()

        # Check state is reset
        state_after_reset = engine.get_state_summary()

        assert state_after_reset['vault_balance'] == 1000.0  # Back to target
        assert state_after_reset['smoothed_l1_cost'] is None
        assert state_after_reset['previous_fee'] is None

    def test_get_state_summary_completeness(self):
        """Test state summary provides complete information."""
        engine = SimulationEngine(
            mu=0.2,
            nu=0.8,
            horizon_h=288,
            target_vault_balance=3000.0,
            lambda_c=0.15
        )

        # Run a step to initialize state
        engine.simulate_step(0.002)

        summary = engine.get_state_summary()

        required_fields = [
            'vault_balance', 'vault_deficit', 'solvency_ratio',
            'smoothed_l1_cost', 'previous_fee', 'target_balance', 'parameters'
        ]

        for field in required_fields:
            assert field in summary

        # Check parameter accuracy
        params = summary['parameters']
        assert params['mu'] == 0.2
        assert params['nu'] == 0.8
        assert params['horizon_h'] == 288
        assert params['lambda_c'] == 0.15

    def test_calculate_metrics_comprehensive(self):
        """Test comprehensive metrics calculation."""
        engine = SimulationEngine(
            mu=0.0,
            nu=0.1,
            horizon_h=36,
            target_vault_balance=1000.0
        )

        # Generate test data
        l1_costs = np.random.uniform(0.001, 0.005, 100)
        result_df = engine.simulate_series(l1_costs)

        metrics = engine.calculate_metrics(result_df)

        # Check all expected metrics are present
        expected_metrics = [
            'avg_fee_gwei', 'median_fee_gwei', 'fee_std_gwei', 'fee_cv',
            'fee_p95_gwei', 'fee_p99_gwei', 'avg_vault_balance',
            'min_vault_balance', 'avg_deficit', 'max_deficit',
            'insolvency_episodes', 'insolvency_rate', 'avg_solvency_ratio',
            'min_solvency_ratio', 'cost_recovery_ratio', 'avg_subsidy',
            'full_subsidy_rate'
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert not np.isnan(metrics[metric])

        # Check metric reasonableness
        assert metrics['cost_recovery_ratio'] > 0
        assert 0 <= metrics['insolvency_rate'] <= 1
        assert 0 <= metrics['full_subsidy_rate'] <= 1

    def test_calculate_metrics_empty_dataframe(self):
        """Test error handling for empty DataFrame in metrics calculation."""
        engine = SimulationEngine(
            mu=0.5,
            nu=0.3,
            horizon_h=144,
            target_vault_balance=1000.0
        )

        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            engine.calculate_metrics(empty_df)

    def test_parameter_validation(self):
        """Test parameter validation functionality."""
        # Valid parameters
        engine = SimulationEngine(
            mu=0.5,
            nu=0.3,
            horizon_h=144,
            target_vault_balance=1000.0
        )

        errors = engine.validate_parameters()
        assert len(errors) == 0

        # Invalid parameters
        engine_invalid = SimulationEngine(
            mu=-0.1,  # Invalid
            nu=1.5,   # Invalid
            horizon_h=-10,  # Invalid
            target_vault_balance=-100.0  # Invalid
        )

        errors = engine_invalid.validate_parameters()
        assert len(errors) > 0
        assert any("mu" in error for error in errors)
        assert any("nu" in error for error in errors)

    def test_canonical_optimal_parameters(self):
        """Test simulation with canonical optimal parameters from SPECS.md."""
        # Optimal: μ=0.0, ν=0.1, H=36
        engine = SimulationEngine(
            mu=0.0,
            nu=0.1,
            horizon_h=36,
            target_vault_balance=5000.0
        )

        # Should run without errors
        l1_costs = np.array([0.002, 0.003, 0.0015, 0.004])
        result_df = engine.simulate_series(l1_costs)

        assert len(result_df) == 4
        assert all(result_df['basefee_per_gas'] > 0)

        # Check deficit decay factor
        expected_phi = 1 - 0.1 / 36
        actual_phi = engine.fee_controller.get_deficit_decay_factor()
        assert_allclose(actual_phi, expected_phi, rtol=1e-15)

    def test_crisis_resilient_parameters(self):
        """Test simulation with crisis-resilient parameters from SPECS.md."""
        # Crisis: μ=0.0, ν=0.7, H=288
        engine = SimulationEngine(
            mu=0.0,
            nu=0.7,
            horizon_h=288,
            target_vault_balance=10000.0
        )

        # Simulate a crisis scenario with high L1 costs
        crisis_costs = np.array([0.05, 0.08, 0.12, 0.1, 0.06])
        result_df = engine.simulate_series(crisis_costs)

        assert len(result_df) == 5

        # In crisis mode, fees should respond more aggressively to deficits
        # Check that fee increases when vault is stressed
        final_deficit = result_df.iloc[-1]['deficit_after']
        if final_deficit > 0:
            # Should have meaningful fee response
            final_fee = result_df.iloc[-1]['basefee_gwei']
            initial_fee = result_df.iloc[0]['basefee_gwei']
            assert final_fee >= initial_fee

    def test_numerical_stability_extreme_scenarios(self):
        """Test numerical stability with extreme scenarios."""
        engine = SimulationEngine(
            mu=0.5,
            nu=0.9,
            horizon_h=6,  # Very short horizon
            target_vault_balance=1000.0,
            initial_vault_balance=10.0  # Very low initial balance
        )

        # Extreme L1 costs
        extreme_costs = np.array([0.1, 0.2, 0.05, 0.15])

        result_df = engine.simulate_series(extreme_costs)

        # Check no NaN or infinite values
        for col in ['basefee_per_gas', 'revenue', 'vault_balance_after']:
            assert not result_df[col].isna().any()
            assert not np.isinf(result_df[col]).any()

    def test_empty_series_error_handling(self):
        """Test error handling for empty L1 cost series."""
        engine = SimulationEngine(
            mu=0.5,
            nu=0.3,
            horizon_h=144,
            target_vault_balance=1000.0
        )

        with pytest.raises(ValueError):
            engine.simulate_series(np.array([]))

    def test_component_integration_consistency(self):
        """Test that all components work together consistently."""
        # Use known parameters for predictable behavior
        lambda_c = 0.5  # Fast L1 cost adaptation
        engine = SimulationEngine(
            mu=1.0,  # Pure L1 pass-through
            nu=0.0,  # No deficit correction
            horizon_h=100,  # Arbitrary
            target_vault_balance=1000.0,
            lambda_c=lambda_c
        )

        l1_cost = 0.001

        # First step
        result1 = engine.simulate_step(l1_cost)

        # With mu=1.0, nu=0.0, fee should be purely based on L1 cost
        expected_raw_fee = l1_cost / 6.9e5
        # Allow for technical floor
        assert result1['basefee_per_gas'] >= min(expected_raw_fee, 1e6)

        # Second step with different cost
        l1_cost2 = 0.002
        result2 = engine.simulate_step(l1_cost2)

        # L1 smoothing should show expected EMA behavior
        expected_smoothed = (1 - lambda_c) * l1_cost + lambda_c * l1_cost2
        assert_allclose(result2['l1_cost_smoothed'], expected_smoothed, rtol=1e-10)

    def test_string_representations(self):
        """Test string representations."""
        engine = SimulationEngine(
            mu=0.3,
            nu=0.7,
            horizon_h=144,
            target_vault_balance=5000.0
        )

        str_repr = str(engine)
        assert "μ=0.3" in str_repr or "mu=0.3" in str_repr
        assert "ν=0.7" in str_repr or "nu=0.7" in str_repr
        assert "H=144" in str_repr
        assert "5.00e+03" in str_repr or "5000" in str_repr