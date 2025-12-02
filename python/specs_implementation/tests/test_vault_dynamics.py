"""
Unit tests for VaultDynamics

Tests the vault dynamics implementation against mathematical specifications
from SPECS.md Section 4.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from ..core.vault_dynamics import VaultDynamics


class TestVaultDynamics:
    """Test suite for VaultDynamics mathematical operations."""

    def test_initialization_valid_params(self):
        """Test vault dynamics initialization with valid parameters."""
        target = 1000.0
        vault = VaultDynamics(target_balance=target)

        assert vault.target_balance == target
        assert vault.vault_balance == target  # Default initial balance
        assert len(vault.balance_history) == 1
        assert len(vault.deficit_history) == 1

        # With custom initial balance
        initial = 500.0
        vault = VaultDynamics(target_balance=target, initial_balance=initial)
        assert vault.vault_balance == initial

    def test_initialization_invalid_params(self):
        """Test vault dynamics initialization with invalid parameters."""
        with pytest.raises(ValueError):
            VaultDynamics(target_balance=0.0)

        with pytest.raises(ValueError):
            VaultDynamics(target_balance=-100.0)

    def test_deficit_calculation_formula(self):
        """Test deficit calculation: D(t) = T - V(t)"""
        target = 1000.0
        initial = 800.0
        vault = VaultDynamics(target_balance=target, initial_balance=initial)

        # Deficit case (below target)
        deficit = vault.calculate_deficit()
        expected_deficit = target - initial
        assert_allclose(deficit, expected_deficit, rtol=1e-15)
        assert deficit > 0  # Positive = deficit

        # Surplus case (above target)
        vault.vault_balance = 1200.0
        surplus = vault.calculate_deficit()
        expected_surplus = target - 1200.0
        assert_allclose(surplus, expected_surplus, rtol=1e-15)
        assert surplus < 0  # Negative = surplus

    def test_subsidy_calculation_formula(self):
        """Test subsidy calculation: S(t) = min(C_L1(t), V(t))"""
        vault = VaultDynamics(target_balance=1000.0, initial_balance=300.0)

        # Case 1: Vault balance >= L1 cost (full subsidy)
        l1_cost = 200.0
        subsidy = vault.calculate_subsidy(l1_cost)
        assert subsidy == l1_cost

        # Case 2: Vault balance < L1 cost (partial subsidy)
        l1_cost = 500.0
        subsidy = vault.calculate_subsidy(l1_cost)
        assert subsidy == vault.vault_balance

        # Case 3: Empty vault
        vault.vault_balance = 0.0
        subsidy = vault.calculate_subsidy(l1_cost)
        assert subsidy == 0.0

    def test_subsidy_calculation_negative_cost(self):
        """Test error handling for negative L1 cost in subsidy calculation."""
        vault = VaultDynamics(target_balance=1000.0)

        with pytest.raises(ValueError):
            vault.calculate_subsidy(-10.0)

    def test_vault_update_formula(self):
        """Test vault update: V(t+1) = V(t) + R(t) - S(t)"""
        initial_balance = 500.0
        vault = VaultDynamics(target_balance=1000.0, initial_balance=initial_balance)

        revenue = 300.0
        l1_cost = 200.0

        # Expected subsidy: min(l1_cost, vault_balance) = min(200, 500) = 200
        expected_subsidy = 200.0

        # Expected new balance: 500 + 300 - 200 = 600
        expected_new_balance = initial_balance + revenue - expected_subsidy

        subsidy, new_balance = vault.update_vault(revenue, l1_cost)

        assert_allclose(subsidy, expected_subsidy, rtol=1e-15)
        assert_allclose(new_balance, expected_new_balance, rtol=1e-15)
        assert_allclose(vault.vault_balance, expected_new_balance, rtol=1e-15)

    def test_vault_update_insolvency_case(self):
        """Test vault update when insufficient funds for full subsidy."""
        initial_balance = 50.0
        vault = VaultDynamics(target_balance=1000.0, initial_balance=initial_balance)

        revenue = 100.0
        l1_cost = 200.0  # More than vault balance

        # Expected subsidy: min(200, 50) = 50 (vault emptied)
        expected_subsidy = 50.0

        # Expected new balance: 50 + 100 - 50 = 100
        expected_new_balance = 100.0

        subsidy, new_balance = vault.update_vault(revenue, l1_cost)

        assert_allclose(subsidy, expected_subsidy, rtol=1e-15)
        assert_allclose(new_balance, expected_new_balance, rtol=1e-15)

    def test_vault_update_negative_inputs(self):
        """Test error handling for negative revenue or L1 cost."""
        vault = VaultDynamics(target_balance=1000.0)

        with pytest.raises(ValueError):
            vault.update_vault(-10.0, 100.0)  # Negative revenue

        with pytest.raises(ValueError):
            vault.update_vault(100.0, -10.0)  # Negative L1 cost

    def test_solvency_checks(self):
        """Test solvency checking functionality."""
        vault = VaultDynamics(target_balance=1000.0, initial_balance=200.0)

        # Above threshold
        assert vault.is_solvent(threshold=100.0)

        # Below threshold
        assert not vault.is_solvent(threshold=300.0)

        # Exactly at threshold
        assert vault.is_solvent(threshold=200.0)

    def test_solvency_ratio(self):
        """Test solvency ratio calculation: V(t)/T"""
        target = 500.0
        balance = 250.0
        vault = VaultDynamics(target_balance=target, initial_balance=balance)

        ratio = vault.get_solvency_ratio()
        expected_ratio = balance / target
        assert_allclose(ratio, expected_ratio, rtol=1e-15)

    def test_history_tracking(self):
        """Test that balance and deficit history is tracked correctly."""
        vault = VaultDynamics(target_balance=1000.0, initial_balance=800.0)

        # Initial state should be recorded
        assert len(vault.balance_history) == 1
        assert len(vault.deficit_history) == 1
        assert vault.balance_history[0] == 800.0
        assert vault.deficit_history[0] == 200.0

        # Update vault
        vault.update_vault(revenue=300.0, l1_cost=100.0)

        # History should be updated
        assert len(vault.balance_history) == 2
        assert len(vault.deficit_history) == 2

        # Check values: 800 + 300 - 100 = 1000
        assert vault.balance_history[1] == 1000.0
        assert vault.deficit_history[1] == 0.0  # At target

    def test_simulate_step_detailed_output(self):
        """Test simulate_step provides detailed step information."""
        vault = VaultDynamics(target_balance=1000.0, initial_balance=600.0)

        revenue = 250.0
        l1_cost = 150.0

        result = vault.simulate_step(revenue, l1_cost)

        # Check all expected fields are present
        required_fields = [
            'revenue', 'l1_cost', 'subsidy',
            'vault_balance_before', 'vault_balance_after',
            'deficit_before', 'deficit_after', 'fully_subsidized'
        ]

        for field in required_fields:
            assert field in result

        # Check values
        assert result['revenue'] == revenue
        assert result['l1_cost'] == l1_cost
        assert result['subsidy'] == l1_cost  # Full subsidy possible
        assert result['vault_balance_before'] == 600.0
        assert result['vault_balance_after'] == 700.0  # 600 + 250 - 150
        assert result['deficit_before'] == 400.0  # 1000 - 600
        assert result['deficit_after'] == 300.0  # 1000 - 700
        assert result['fully_subsidized'] is True

    def test_process_series_mathematical_consistency(self):
        """Test process_series produces mathematically consistent results."""
        vault = VaultDynamics(target_balance=1000.0, initial_balance=500.0)

        revenues = np.array([200.0, 150.0, 300.0])
        l1_costs = np.array([100.0, 200.0, 50.0])

        result = vault.process_series(revenues, l1_costs)

        # Check array lengths
        assert len(result['vault_balances']) == len(revenues) + 1  # Includes initial
        assert len(result['deficits']) == len(revenues) + 1
        assert len(result['subsidies']) == len(revenues)

        # Manually calculate expected results
        expected_balances = [500.0]  # Initial
        expected_deficits = [500.0]  # 1000 - 500

        balance = 500.0
        for i, (revenue, l1_cost) in enumerate(zip(revenues, l1_costs)):
            subsidy = min(l1_cost, balance)
            balance = balance + revenue - subsidy

            expected_balances.append(balance)
            expected_deficits.append(1000.0 - balance)

        assert_allclose(result['vault_balances'], expected_balances, rtol=1e-15)
        assert_allclose(result['deficits'], expected_deficits, rtol=1e-15)

    def test_process_series_mismatched_lengths(self):
        """Test error handling for mismatched input array lengths."""
        vault = VaultDynamics(target_balance=1000.0)

        with pytest.raises(ValueError):
            vault.process_series(
                np.array([100.0, 200.0]),
                np.array([50.0])
            )

    def test_reset_state_functionality(self):
        """Test state reset functionality."""
        vault = VaultDynamics(target_balance=1000.0, initial_balance=600.0)

        # Make some updates
        vault.update_vault(100.0, 50.0)
        vault.update_vault(200.0, 80.0)

        assert len(vault.balance_history) > 1
        assert vault.vault_balance != 600.0

        # Reset to default (target balance)
        vault.reset_state()

        assert vault.vault_balance == 1000.0
        assert len(vault.balance_history) == 1
        assert len(vault.deficit_history) == 1
        assert vault.balance_history[0] == 1000.0

        # Reset to specific balance
        vault.reset_state(initial_balance=300.0)

        assert vault.vault_balance == 300.0
        assert vault.balance_history[0] == 300.0

    def test_deficit_recursion_approximation(self):
        """Test approximate deficit recursion from SPECS.md Section 4.3."""
        # Set up scenario for testing D(t+1) = φ*D(t) + (1-μ)*C_L1(t)
        # This requires coordination with fee controller, but we can test the vault part

        vault = VaultDynamics(target_balance=1000.0, initial_balance=800.0)

        # Initial deficit
        initial_deficit = vault.calculate_deficit()  # 200.0

        # Simulate revenue that would come from formula with μ=0.3, ν=0.2, H=50
        l1_cost = 100.0
        mu = 0.3
        # Revenue would be: μ*L1_cost + ν*deficit/H = 0.3*100 + something
        # For simplicity, let's use exact L1 cost coverage
        revenue = mu * l1_cost  # Partial coverage

        vault.update_vault(revenue, l1_cost)

        # Check that deficit changed as expected
        new_deficit = vault.calculate_deficit()

        # Deficit should increase because revenue < l1_cost
        assert new_deficit > initial_deficit

    def test_extreme_scenarios(self):
        """Test behavior in extreme scenarios."""
        vault = VaultDynamics(target_balance=1000.0, initial_balance=0.0)

        # Empty vault scenario
        assert vault.is_solvent(threshold=0.0)
        assert not vault.is_solvent(threshold=0.1)

        subsidy = vault.calculate_subsidy(500.0)
        assert subsidy == 0.0

        # Very large revenue
        large_revenue = 1e9
        vault.update_vault(large_revenue, 100.0)
        assert vault.vault_balance > 1000.0  # Way above target

        # Check numerical stability
        assert not np.isnan(vault.vault_balance)
        assert not np.isinf(vault.vault_balance)

    def test_numerical_precision(self):
        """Test numerical precision with very small values."""
        vault = VaultDynamics(target_balance=1.0, initial_balance=0.5)

        # Very small revenue and costs
        small_revenue = 1e-10
        small_cost = 5e-11

        vault.update_vault(small_revenue, small_cost)

        # Should handle small values correctly
        assert vault.vault_balance == 0.5 + small_revenue - small_cost
        assert not np.isnan(vault.vault_balance)

    def test_string_representations(self):
        """Test string representations."""
        vault = VaultDynamics(target_balance=1000.0, initial_balance=750.0)

        str_repr = str(vault)
        assert "T=1.00e+03" in str_repr or "T=1000" in str_repr
        assert "V(t)=7.50e+02" in str_repr or "V(t)=750" in str_repr
        assert "D(t)=2.50e+02" in str_repr or "D(t)=250" in str_repr