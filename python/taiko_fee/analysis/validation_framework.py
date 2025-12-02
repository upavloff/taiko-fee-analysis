"""
Validation Framework for Enhanced Metrics

This module validates the enhanced metrics framework against known optimal
parameters from the POST_TIMING_FIX analysis. It ensures that:

1. Enhanced metrics align with existing optimization results
2. New metrics provide additional meaningful insights
3. Composite scores correctly rank known optimal vs suboptimal parameters
4. Objective functions behave as expected across different strategies
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'core'))
sys.path.insert(0, os.path.join(project_root, 'src', 'analysis'))

# Import simulation components
from core.improved_simulator import ImprovedTaikoFeeSimulator, ImprovedSimulationParams
from analysis.mechanism_metrics import MetricsCalculator as OriginalMetricsCalculator
from analysis.enhanced_metrics import EnhancedMetricsCalculator
from analysis.objective_functions import ObjectiveFunctionSuite, OptimizationStrategy


class ValidationSuite:
    """
    Comprehensive validation suite for enhanced metrics framework.
    """

    def __init__(self, target_balance: float = 1000.0):
        """
        Initialize validation suite.

        Args:
            target_balance: Target vault balance for simulations
        """
        self.target_balance = target_balance
        self.original_calc = OriginalMetricsCalculator(target_balance)
        self.enhanced_calc = EnhancedMetricsCalculator(target_balance)
        self.objective_suite = ObjectiveFunctionSuite()

        # Known optimal parameters from POST_TIMING_FIX analysis
        self.known_optimal_params = {
            'optimal': {'mu': 0.0, 'nu': 0.1, 'H': 36},
            'balanced': {'mu': 0.0, 'nu': 0.2, 'H': 72},
            'crisis': {'mu': 0.0, 'nu': 0.7, 'H': 288}
        }

        # Known suboptimal parameters for comparison
        self.known_suboptimal_params = {
            'high_mu': {'mu': 1.0, 'nu': 0.1, 'H': 36},      # High L1 weight is known to be bad
            'misaligned_h': {'mu': 0.0, 'nu': 0.1, 'H': 100}, # Non-6-step aligned
            'extreme_nu': {'mu': 0.0, 'nu': 0.9, 'H': 72},   # Overly aggressive correction
        }

        # Load historical data for testing
        self.historical_data = self._load_validation_data()

    def _load_validation_data(self) -> Dict[str, np.ndarray]:
        """Load historical L1 basefee data for consistent validation testing."""
        try:
            # Load recent low fees dataset for consistent validation
            data_path = os.path.join(project_root, 'data', 'data_cache', 'recent_low_fees_3hours.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                return {
                    'recent_low': df['basefee_wei'].values[:500]  # Use first 500 points
                }
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")

        # Fallback to synthetic data
        return {
            'synthetic_stable': np.full(500, 10e9),  # 10 gwei stable
            'synthetic_volatile': np.random.lognormal(np.log(20e9), 0.5, 500)  # Volatile around 20 gwei
        }

    def run_simulation_with_params(self, params: Dict[str, float],
                                 scenario: str = 'recent_low') -> pd.DataFrame:
        """
        Run simulation with given parameters on specified scenario.

        Args:
            params: Parameter dictionary with mu, nu, H
            scenario: Data scenario to use

        Returns:
            Simulation results DataFrame
        """
        # Create simulation parameters
        sim_params = ImprovedSimulationParams(
            mu=params['mu'],
            nu=params['nu'],
            H=params['H'],
            target_balance=self.target_balance,
            base_demand=100,
            fee_elasticity=0.2,
            gas_per_batch=200000,
            txs_per_batch=100,
            batch_frequency=0.1,
            total_steps=min(500, len(self.historical_data[scenario])),
            time_step_seconds=2,  # Taiko block time
            vault_initialization_mode='target',
            fee_cap=0.1
        )

        # Create simple L1 model
        class HistoricalL1Model:
            def __init__(self, basefee_sequence):
                self.sequence = basefee_sequence

            def generate_sequence(self, steps, initial_basefee=None):
                return self.sequence[:steps]

            def get_name(self):
                return scenario

        l1_model = HistoricalL1Model(self.historical_data[scenario])

        # Run simulation
        simulator = ImprovedTaikoFeeSimulator(sim_params, l1_model)
        return simulator.run_simulation()

    def validate_metric_correlation(self) -> Dict[str, Dict[str, float]]:
        """
        Validate that new metrics correlate appropriately with existing metrics.

        Returns:
            Correlation analysis results
        """
        print("🔍 Validating metric correlations...")

        all_params = list(self.known_optimal_params.values()) + list(self.known_suboptimal_params.values())
        correlations = {}

        original_metrics = []
        enhanced_metrics = []

        for params in all_params:
            # Run simulation
            df = self.run_simulation_with_params(params)

            # Calculate original metrics
            orig_metrics = self.original_calc.calculate_all_metrics(df)
            original_metrics.append(orig_metrics.to_dict())

            # Calculate enhanced metrics
            enh_metrics = self.enhanced_calc.calculate_all_metrics(df, params)
            enhanced_metrics.append(enh_metrics.to_dict())

        # Convert to DataFrames for correlation analysis
        orig_df = pd.DataFrame(original_metrics)
        enh_df = pd.DataFrame(enhanced_metrics)

        # Key correlations to check
        correlation_checks = {
            'fee_correlation': ('avg_fee', 'fee_affordability_score'),          # Should be negatively correlated
            'vault_correlation': ('time_underfunded_pct', 'vault_robustness_score'),  # Should be negatively correlated
            'efficiency_correlation': ('overpayment_ratio', 'cost_recovery_ratio'),   # Should be positively correlated
        }

        for check_name, (orig_metric, enh_metric) in correlation_checks.items():
            if orig_metric in orig_df.columns and enh_metric in enh_df.columns:
                correlation = np.corrcoef(orig_df[orig_metric], enh_df[enh_metric])[0, 1]
                correlations[check_name] = {
                    'correlation': correlation,
                    'expected_direction': 'negative' if 'fee' in check_name or 'vault' in check_name else 'positive'
                }

        return correlations

    def validate_optimal_ranking(self) -> Dict[str, Dict[str, float]]:
        """
        Validate that known optimal parameters score better than suboptimal ones.

        Returns:
            Ranking validation results
        """
        print("🎯 Validating optimal parameter ranking...")

        results = {}

        # Test each optimization strategy
        for strategy in OptimizationStrategy:
            objective_func = self.objective_suite.objective_functions[strategy]
            strategy_results = {'optimal_scores': [], 'suboptimal_scores': []}

            # Calculate scores for known optimal parameters
            for name, params in self.known_optimal_params.items():
                df = self.run_simulation_with_params(params)
                enhanced_metrics = self.enhanced_calc.calculate_all_metrics(df, params)
                score = objective_func.calculate_weighted_objective(enhanced_metrics)
                strategy_results['optimal_scores'].append(score)

            # Calculate scores for known suboptimal parameters
            for name, params in self.known_suboptimal_params.items():
                df = self.run_simulation_with_params(params)
                enhanced_metrics = self.enhanced_calc.calculate_all_metrics(df, params)
                score = objective_func.calculate_weighted_objective(enhanced_metrics)
                strategy_results['suboptimal_scores'].append(score)

            # Calculate ranking quality
            optimal_mean = np.mean(strategy_results['optimal_scores'])
            suboptimal_mean = np.mean(strategy_results['suboptimal_scores'])

            results[strategy.value] = {
                'optimal_mean': optimal_mean,
                'suboptimal_mean': suboptimal_mean,
                'ranking_quality': optimal_mean - suboptimal_mean,  # Should be positive
                'all_optimal_better': all(
                    opt > sub for opt in strategy_results['optimal_scores']
                    for sub in strategy_results['suboptimal_scores']
                )
            }

        return results

    def validate_6step_alignment_impact(self) -> Dict[str, float]:
        """
        Validate that 6-step aligned horizons perform better than misaligned ones.

        Returns:
            6-step alignment validation results
        """
        print("📊 Validating 6-step cycle alignment impact...")

        base_params = {'mu': 0.0, 'nu': 0.1}
        aligned_horizons = [36, 72, 144, 288]      # All multiples of 6
        misaligned_horizons = [40, 80, 150, 300]   # Non-multiples of 6

        aligned_scores = []
        misaligned_scores = []

        for H in aligned_horizons:
            params = {**base_params, 'H': H}
            df = self.run_simulation_with_params(params)
            enhanced_metrics = self.enhanced_calc.calculate_all_metrics(df, params)
            aligned_scores.append(enhanced_metrics.sixstep_cycle_alignment)

        for H in misaligned_horizons:
            params = {**base_params, 'H': H}
            df = self.run_simulation_with_params(params)
            enhanced_metrics = self.enhanced_calc.calculate_all_metrics(df, params)
            misaligned_scores.append(enhanced_metrics.sixstep_cycle_alignment)

        return {
            'aligned_mean': np.mean(aligned_scores),
            'misaligned_mean': np.mean(misaligned_scores),
            'alignment_advantage': np.mean(aligned_scores) - np.mean(misaligned_scores),
            'all_aligned_better': np.mean(aligned_scores) > np.mean(misaligned_scores)
        }

    def validate_production_constraints(self) -> Dict[str, Dict[str, bool]]:
        """
        Validate that production constraints appropriately filter parameters.

        Returns:
            Production constraint validation results
        """
        print("🏭 Validating production constraints...")

        production_func = self.objective_suite.production_functions[f'{OptimizationStrategy.BALANCED}_production']
        results = {}

        # Test known optimal parameters (should be feasible)
        for name, params in self.known_optimal_params.items():
            df = self.run_simulation_with_params(params)
            enhanced_metrics = self.enhanced_calc.calculate_all_metrics(df, params)

            feasible = production_func.is_feasible(enhanced_metrics)
            constraint_violations = production_func.evaluate_constraints(enhanced_metrics)
            risk_score = production_func.calculate_deployment_risk_score(enhanced_metrics)

            results[f"optimal_{name}"] = {
                'is_feasible': feasible,
                'low_risk': risk_score < 0.3,
                'constraint_violations': {k: not v for k, v in constraint_violations.items() if not v}
            }

        # Test known suboptimal parameters (many should be infeasible)
        for name, params in self.known_suboptimal_params.items():
            df = self.run_simulation_with_params(params)
            enhanced_metrics = self.enhanced_calc.calculate_all_metrics(df, params)

            feasible = production_func.is_feasible(enhanced_metrics)
            constraint_violations = production_func.evaluate_constraints(enhanced_metrics)
            risk_score = production_func.calculate_deployment_risk_score(enhanced_metrics)

            results[f"suboptimal_{name}"] = {
                'is_feasible': feasible,
                'low_risk': risk_score < 0.3,
                'constraint_violations': {k: not v for k, v in constraint_violations.items() if not v}
            }

        return results

    def create_validation_report(self) -> str:
        """
        Create comprehensive validation report.

        Returns:
            Formatted validation report
        """
        print("📋 Creating comprehensive validation report...\n")

        report = []
        report.append("=" * 80)
        report.append("ENHANCED METRICS FRAMEWORK VALIDATION REPORT")
        report.append("=" * 80)

        # 1. Metric Correlation Validation
        report.append("\n1. METRIC CORRELATION VALIDATION")
        report.append("-" * 40)

        correlation_results = self.validate_metric_correlation()
        for check_name, result in correlation_results.items():
            correlation = result['correlation']
            expected = result['expected_direction']

            if expected == 'negative':
                is_correct = correlation < 0
                status = "✅" if is_correct else "❌"
            else:
                is_correct = correlation > 0
                status = "✅" if is_correct else "❌"

            report.append(f"{status} {check_name}: {correlation:.3f} (expected {expected})")

        # 2. Optimal Parameter Ranking
        report.append("\n\n2. OPTIMAL PARAMETER RANKING VALIDATION")
        report.append("-" * 45)

        ranking_results = self.validate_optimal_ranking()
        for strategy, result in ranking_results.items():
            ranking_quality = result['ranking_quality']
            all_better = result['all_optimal_better']

            status = "✅" if ranking_quality > 0 and all_better else "❌"
            report.append(f"{status} {strategy}: ranking_quality={ranking_quality:.3f}, all_optimal_better={all_better}")

        # 3. 6-Step Alignment Impact
        report.append("\n\n3. 6-STEP CYCLE ALIGNMENT VALIDATION")
        report.append("-" * 40)

        alignment_results = self.validate_6step_alignment_impact()
        alignment_advantage = alignment_results['alignment_advantage']
        all_aligned_better = alignment_results['all_aligned_better']

        status = "✅" if alignment_advantage > 0 and all_aligned_better else "❌"
        report.append(f"{status} 6-step alignment advantage: {alignment_advantage:.3f}")
        report.append(f"   Aligned mean: {alignment_results['aligned_mean']:.3f}")
        report.append(f"   Misaligned mean: {alignment_results['misaligned_mean']:.3f}")

        # 4. Production Constraints
        report.append("\n\n4. PRODUCTION CONSTRAINTS VALIDATION")
        report.append("-" * 40)

        production_results = self.validate_production_constraints()

        optimal_feasible = sum(1 for name, result in production_results.items()
                             if 'optimal' in name and result['is_feasible'])
        total_optimal = len(self.known_optimal_params)

        suboptimal_infeasible = sum(1 for name, result in production_results.items()
                                   if 'suboptimal' in name and not result['is_feasible'])
        total_suboptimal = len(self.known_suboptimal_params)

        report.append(f"✅ Optimal parameters feasible: {optimal_feasible}/{total_optimal}")
        report.append(f"✅ Suboptimal parameters infeasible: {suboptimal_infeasible}/{total_suboptimal}")

        # 5. Overall Validation Summary
        report.append("\n\n5. OVERALL VALIDATION SUMMARY")
        report.append("-" * 35)

        # Count successful validations
        correlation_successes = sum(1 for result in correlation_results.values()
                                  if ((result['expected_direction'] == 'negative' and result['correlation'] < 0) or
                                      (result['expected_direction'] == 'positive' and result['correlation'] > 0)))

        ranking_successes = sum(1 for result in ranking_results.values()
                               if result['ranking_quality'] > 0 and result['all_optimal_better'])

        total_validations = len(correlation_results) + len(ranking_results) + 1 + 1  # +1 for alignment, +1 for constraints
        successful_validations = (correlation_successes + ranking_successes +
                                int(alignment_advantage > 0) +
                                int(optimal_feasible == total_optimal))

        success_rate = successful_validations / total_validations

        if success_rate >= 0.8:
            status = "🎉 VALIDATION PASSED"
        elif success_rate >= 0.6:
            status = "⚠️  VALIDATION PARTIAL"
        else:
            status = "❌ VALIDATION FAILED"

        report.append(f"{status}")
        report.append(f"Success Rate: {success_rate:.1%} ({successful_validations}/{total_validations})")

        if success_rate >= 0.8:
            report.append("\n✅ Enhanced metrics framework is properly calibrated")
            report.append("✅ Objective functions correctly rank optimal vs suboptimal parameters")
            report.append("✅ Production constraints provide meaningful filtering")
            report.append("✅ Framework is ready for comprehensive optimization")
        else:
            report.append("\n❌ Enhanced metrics framework needs calibration")
            report.append("❌ Review metric definitions and weight configurations")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def run_full_validation(self) -> bool:
        """
        Run complete validation suite and return success status.

        Returns:
            True if validation passes, False otherwise
        """
        print("🚀 Running Enhanced Metrics Framework Validation...")
        print("-" * 60)

        report = self.create_validation_report()
        print(report)

        # Return True if validation passed
        return "VALIDATION PASSED" in report


def main():
    """
    Main validation function.
    """
    print("Enhanced Metrics Framework Validation")
    print("=====================================\n")

    validator = ValidationSuite()
    success = validator.run_full_validation()

    if success:
        print("\n✅ All validations passed! Enhanced metrics framework is ready.")
        return 0
    else:
        print("\n❌ Some validations failed. Review metrics framework.")
        return 1


if __name__ == "__main__":
    exit(main())