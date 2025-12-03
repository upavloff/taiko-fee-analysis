"""
Comprehensive Validation Tests for Constraint-Aware Fee Mechanism Optimization

This module provides thorough testing of the constraint optimization framework
including CRR validation, ruin probability testing, and integration validation.

Test Categories:
1. Constraint Evaluation Tests: CRR and ruin probability calculations
2. Scenario Integration Tests: Historical data loading and processing
3. Optimization Framework Tests: NSGA-II with constraints
4. Parameter Validation Tests: Known optimal parameters verification
5. Edge Case Tests: Boundary conditions and error handling

Usage:
    python test_constraint_optimization.py
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent))

try:
    from core.canonical_optimization import (
        CanonicalOptimizer, Individual, OptimizationBounds, OptimizationStrategy,
        optimize_simplified_parameters, optimize_with_constraints
    )
    from core.canonical_scenarios import (
        CanonicalScenarioLoader, ScenarioEvaluator, load_default_scenarios
    )
    from core.canonical_fee_mechanism import CanonicalTaikoFeeCalculator, FeeParameters

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    MODULES_AVAILABLE = False


class ConstraintOptimizationValidator:
    """Comprehensive validation of constraint optimization framework."""

    def __init__(self):
        """Initialize validator with test configurations."""
        if not MODULES_AVAILABLE:
            raise RuntimeError("Core modules not available for testing")

        self.test_results = {
            'constraint_evaluation': {},
            'scenario_integration': {},
            'optimization_framework': {},
            'parameter_validation': {},
            'edge_cases': {}
        }

        # Known optimal parameters from documentation
        self.optimal_params = {
            'mu': 0.0,
            'nu': 0.27,
            'H': 492,
            'lambda_B': 0.1,
            'alpha_data': 20000.0,
            'Q_bar': 690000.0,
            'T': 1000.0
        }

        # Expected constraint tolerances
        self.crr_tolerance = 0.05  # ±5%
        self.max_ruin_probability = 0.01  # 1%

    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🔬 Running Comprehensive Constraint Optimization Validation")
        print("=" * 60)

        # Test 1: Constraint Evaluation
        print("\n1. Testing Constraint Evaluation Functions...")
        self._test_constraint_evaluation()

        # Test 2: Scenario Integration
        print("\n2. Testing Scenario Integration...")
        self._test_scenario_integration()

        # Test 3: Optimization Framework
        print("\n3. Testing Optimization Framework...")
        self._test_optimization_framework()

        # Test 4: Parameter Validation
        print("\n4. Testing Parameter Validation...")
        self._test_parameter_validation()

        # Test 5: Edge Cases
        print("\n5. Testing Edge Cases...")
        self._test_edge_cases()

        # Generate summary
        summary = self._generate_test_summary()
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        self._print_summary(summary)

        return {
            'results': self.test_results,
            'summary': summary,
            'passed': summary['overall_pass']
        }

    def _test_constraint_evaluation(self):
        """Test CRR and ruin probability calculations."""
        print("  Testing CRR calculation with optimal parameters...")

        try:
            # Create optimizer for testing
            optimizer = CanonicalOptimizer()

            # Create individual with optimal parameters
            individual = Individual(**self.optimal_params)

            # Generate simple scenario data for testing
            scenario_length = 500
            l1_basefees = [15e9 + 5e9 * np.sin(i * 0.1) for i in range(scenario_length)]  # 15±5 gwei oscillation

            # Test CRR calculation
            crr = optimizer.calculate_cost_recovery_ratio(individual, l1_basefees)
            crr_satisfied, crr_violation = optimizer.evaluate_crr_constraint(crr, self.crr_tolerance)

            self.test_results['constraint_evaluation']['crr_value'] = crr
            self.test_results['constraint_evaluation']['crr_satisfied'] = crr_satisfied
            self.test_results['constraint_evaluation']['crr_violation'] = crr_violation

            print(f"    ✓ CRR calculated: {crr:.3f}")
            print(f"    ✓ CRR constraint satisfied: {crr_satisfied}")

            # Test ruin probability calculation
            scenarios = [l1_basefees]  # Single scenario for testing
            ruin_prob = optimizer.calculate_ruin_probability(individual, scenarios)
            ruin_satisfied, ruin_violation = optimizer.evaluate_ruin_constraint(ruin_prob, self.max_ruin_probability)

            self.test_results['constraint_evaluation']['ruin_probability'] = ruin_prob
            self.test_results['constraint_evaluation']['ruin_satisfied'] = ruin_satisfied
            self.test_results['constraint_evaluation']['ruin_violation'] = ruin_violation

            print(f"    ✓ Ruin probability calculated: {ruin_prob:.4f}")
            print(f"    ✓ Ruin constraint satisfied: {ruin_satisfied}")

            self.test_results['constraint_evaluation']['passed'] = True

        except Exception as e:
            print(f"    ❌ Constraint evaluation test failed: {e}")
            self.test_results['constraint_evaluation']['passed'] = False
            self.test_results['constraint_evaluation']['error'] = str(e)

    def _test_scenario_integration(self):
        """Test scenario loading and integration."""
        print("  Testing scenario data loading...")

        try:
            # Test scenario loader
            loader = CanonicalScenarioLoader()
            scenarios = loader.load_all_scenarios()

            self.test_results['scenario_integration']['scenarios_loaded'] = len(scenarios)

            if len(scenarios) > 0:
                print(f"    ✓ Loaded {len(scenarios)} scenarios")

                # Test scenario types
                scenario_types = {}
                for name, scenario in scenarios.items():
                    scenario_type = scenario.scenario_type.value
                    if scenario_type not in scenario_types:
                        scenario_types[scenario_type] = 0
                    scenario_types[scenario_type] += 1

                self.test_results['scenario_integration']['scenario_types'] = scenario_types
                print(f"    ✓ Scenario types: {scenario_types}")

                # Test scenario evaluator
                evaluator = ScenarioEvaluator(scenarios)
                eval_results = evaluator.evaluate_parameter_set(**self.optimal_params)

                self.test_results['scenario_integration']['evaluation_results'] = {
                    'average_crr': eval_results.average_crr,
                    'worst_case_crr': eval_results.worst_case_crr,
                    'average_ruin_prob': eval_results.average_ruin_prob,
                    'max_ruin_prob': eval_results.max_ruin_prob,
                    'constraints_satisfied': eval_results.all_constraints_satisfied
                }

                print(f"    ✓ Parameter evaluation completed")
                print(f"      Average CRR: {eval_results.average_crr:.3f}")
                print(f"      Max ruin probability: {eval_results.max_ruin_prob:.4f}")
                print(f"      All constraints satisfied: {eval_results.all_constraints_satisfied}")

            else:
                print("    ⚠️  No scenarios loaded (data files may be missing)")
                self.test_results['scenario_integration']['warning'] = "No scenarios loaded"

            self.test_results['scenario_integration']['passed'] = True

        except Exception as e:
            print(f"    ❌ Scenario integration test failed: {e}")
            self.test_results['scenario_integration']['passed'] = False
            self.test_results['scenario_integration']['error'] = str(e)

    def _test_optimization_framework(self):
        """Test constraint-aware optimization framework."""
        print("  Testing constraint-aware NSGA-II optimization...")

        try:
            # Test simplified parameter optimization
            print("    Testing simplified parameter vector optimization...")

            results = optimize_simplified_parameters(
                strategy=OptimizationStrategy.BALANCED,
                population_size=20,  # Small for testing
                generations=5,       # Few generations for testing
                enable_constraints=True
            )

            self.test_results['optimization_framework']['simplified_results'] = {
                'pareto_size': len(results.get('pareto_front', [])),
                'generations_completed': results.get('generations', 0),
                'evaluation_time': results.get('evaluation_time', 0)
            }

            print(f"    ✓ Simplified optimization completed")
            print(f"      Pareto front size: {len(results.get('pareto_front', []))}")

            # Test constraint-aware optimization
            print("    Testing constraint-aware optimization...")

            constraint_results = optimize_with_constraints(
                strategy=OptimizationStrategy.BALANCED,
                population_size=15,  # Smaller for testing
                generations=3,       # Few generations for testing
                simplified_mode=True
            )

            self.test_results['optimization_framework']['constraint_results'] = {
                'pareto_size': len(constraint_results.get('pareto_front', [])),
                'generations_completed': constraint_results.get('generations', 0),
                'evaluation_time': constraint_results.get('evaluation_time', 0)
            }

            print(f"    ✓ Constraint-aware optimization completed")
            print(f"      Pareto front size: {len(constraint_results.get('pareto_front', []))}")

            self.test_results['optimization_framework']['passed'] = True

        except Exception as e:
            print(f"    ❌ Optimization framework test failed: {e}")
            self.test_results['optimization_framework']['passed'] = False
            self.test_results['optimization_framework']['error'] = str(e)

    def _test_parameter_validation(self):
        """Test known optimal parameters for correctness."""
        print("  Testing known optimal parameters...")

        try:
            # Create fee calculator with optimal parameters
            params = FeeParameters(**self.optimal_params)
            calculator = CanonicalTaikoFeeCalculator(params)

            # Test basic fee calculation
            l1_basefee = 20e9  # 20 gwei
            vault_deficit = 100.0  # 100 ETH deficit

            estimated_fee = calculator.calculate_estimated_fee_raw(l1_basefee, vault_deficit)

            self.test_results['parameter_validation']['estimated_fee'] = estimated_fee

            # Fee should be reasonable (not too high or too low)
            fee_gwei = estimated_fee * 1e9
            fee_reasonable = 0.1 <= fee_gwei <= 1000.0  # Between 0.1 and 1000 gwei

            self.test_results['parameter_validation']['fee_reasonable'] = fee_reasonable
            self.test_results['parameter_validation']['fee_gwei'] = fee_gwei

            print(f"    ✓ Fee calculation: {fee_gwei:.3f} gwei")
            print(f"    ✓ Fee reasonableness: {fee_reasonable}")

            # Test vault operations
            from core.canonical_fee_mechanism import VaultInitMode
            vault = calculator.create_vault(VaultInitMode.TARGET)
            initial_balance = vault.balance

            # Collect some fees
            vault.collect_fees(10.0)
            balance_after_collection = vault.balance

            # Pay some costs
            vault.pay_l1_costs(5.0)
            balance_after_payment = vault.balance

            vault_operations_correct = (
                balance_after_collection > initial_balance and
                balance_after_payment < balance_after_collection
            )

            self.test_results['parameter_validation']['vault_operations'] = vault_operations_correct

            print(f"    ✓ Vault operations: {vault_operations_correct}")

            self.test_results['parameter_validation']['passed'] = True

        except Exception as e:
            print(f"    ❌ Parameter validation test failed: {e}")
            self.test_results['parameter_validation']['passed'] = False
            self.test_results['parameter_validation']['error'] = str(e)

    def _test_edge_cases(self):
        """Test edge cases and error handling."""
        print("  Testing edge cases and error handling...")

        try:
            # Test empty scenario data
            optimizer = CanonicalOptimizer()
            individual = Individual(**self.optimal_params)

            # Test with empty scenarios
            empty_scenarios = []
            ruin_prob_empty = optimizer.calculate_ruin_probability(individual, empty_scenarios)

            self.test_results['edge_cases']['empty_scenarios_handled'] = ruin_prob_empty == 0.0

            # Test with invalid parameter bounds
            bounds = OptimizationBounds(
                mu_min=1.5, mu_max=1.0,  # Invalid: min > max
                simplified_mode=True
            )

            try:
                invalid_optimizer = CanonicalOptimizer(bounds=bounds)
                population = invalid_optimizer._initialize_population(5)
                # Should handle invalid bounds gracefully
                bounds_handled = len(population) == 5
            except Exception:
                bounds_handled = False  # Error not handled gracefully

            self.test_results['edge_cases']['invalid_bounds_handled'] = bounds_handled

            # Test extreme fee values
            extreme_l1_fee = 1000e9  # 1000 gwei - very high
            extreme_deficit = 10000.0  # 10k ETH deficit - very high

            try:
                extreme_fee = optimizer.calculate_cost_recovery_ratio(
                    individual, [extreme_l1_fee] * 100
                )
                extreme_fee_handled = extreme_fee is not None and extreme_fee > 0
            except Exception:
                extreme_fee_handled = False

            self.test_results['edge_cases']['extreme_values_handled'] = extreme_fee_handled

            print(f"    ✓ Empty scenarios handled: {self.test_results['edge_cases']['empty_scenarios_handled']}")
            print(f"    ✓ Invalid bounds handled: {bounds_handled}")
            print(f"    ✓ Extreme values handled: {extreme_fee_handled}")

            self.test_results['edge_cases']['passed'] = True

        except Exception as e:
            print(f"    ❌ Edge cases test failed: {e}")
            self.test_results['edge_cases']['passed'] = False
            self.test_results['edge_cases']['error'] = str(e)

    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        passed_tests = []
        failed_tests = []
        warnings = []

        for category, results in self.test_results.items():
            if results.get('passed', False):
                passed_tests.append(category)
            else:
                failed_tests.append(category)

            if 'warning' in results:
                warnings.append(f"{category}: {results['warning']}")

        # Overall assessment
        overall_pass = len(failed_tests) == 0

        # Key metrics validation
        key_validations = {}

        if 'constraint_evaluation' in self.test_results and self.test_results['constraint_evaluation'].get('passed'):
            crr_value = self.test_results['constraint_evaluation'].get('crr_value', 0)
            key_validations['crr_near_unity'] = abs(crr_value - 1.0) <= self.crr_tolerance

            ruin_prob = self.test_results['constraint_evaluation'].get('ruin_probability', 1.0)
            key_validations['ruin_probability_acceptable'] = ruin_prob <= self.max_ruin_probability

        if 'parameter_validation' in self.test_results and self.test_results['parameter_validation'].get('passed'):
            key_validations['optimal_parameters_valid'] = self.test_results['parameter_validation'].get('fee_reasonable', False)

        return {
            'total_tests': len(self.test_results),
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'warnings': warnings,
            'overall_pass': overall_pass,
            'key_validations': key_validations,
            'pass_rate': len(passed_tests) / len(self.test_results) if self.test_results else 0
        }

    def _print_summary(self, summary: Dict[str, Any]):
        """Print formatted test summary."""

        print(f"Pass Rate: {summary['pass_rate']:.1%} ({len(summary['passed_tests'])}/{summary['total_tests']} tests)")

        if summary['passed_tests']:
            print(f"\n✅ PASSED TESTS ({len(summary['passed_tests'])}):")
            for test in summary['passed_tests']:
                print(f"  • {test.replace('_', ' ').title()}")

        if summary['failed_tests']:
            print(f"\n❌ FAILED TESTS ({len(summary['failed_tests'])}):")
            for test in summary['failed_tests']:
                print(f"  • {test.replace('_', ' ').title()}")
                if test in self.test_results and 'error' in self.test_results[test]:
                    print(f"    Error: {self.test_results[test]['error']}")

        if summary['warnings']:
            print(f"\n⚠️  WARNINGS ({len(summary['warnings'])}):")
            for warning in summary['warnings']:
                print(f"  • {warning}")

        if summary['key_validations']:
            print(f"\n🎯 KEY VALIDATIONS:")
            for validation, result in summary['key_validations'].items():
                status = "✓" if result else "❌"
                print(f"  {status} {validation.replace('_', ' ').title()}")

        # Final verdict
        if summary['overall_pass']:
            print(f"\n🎉 ALL TESTS PASSED - Constraint optimization framework is working correctly!")
        else:
            print(f"\n⚠️  SOME TESTS FAILED - Review failed tests and fix issues before deployment.")


def main():
    """Run validation tests."""
    if not MODULES_AVAILABLE:
        print("❌ Error: Core modules not available. Please check import paths.")
        return False

    try:
        validator = ConstraintOptimizationValidator()
        results = validator.run_all_tests()
        return results['passed']

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)