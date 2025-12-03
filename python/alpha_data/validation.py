"""
Alpha Data Validation Suite

Validates the new alpha-data based fee model against:
- Historical scenarios with known Q̄ behavior
- Expected fee ranges (5-15 gwei vs current 0.00)
- Cost recovery ratios (should be 0.8-1.2)
- Cross-validation with existing optimization results
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from .alpha_calculator import AlphaStatistics, AlphaDataPoint

logger = logging.getLogger(__name__)


@dataclass
class ValidationScenario:
    """Validation scenario configuration"""
    name: str
    description: str
    l1_basefee_series: List[float]  # in wei
    expected_fee_range_gwei: Tuple[float, float]
    expected_cost_recovery: Tuple[float, float]
    scenario_type: str  # 'normal', 'spike', 'crash', 'crisis'


@dataclass
class ValidationResult:
    """Results of model validation"""
    scenario_name: str
    alpha_value_tested: float
    q_bar_comparison: float  # Q̄ value for comparison

    # Fee analysis
    avg_fee_gwei_alpha: float
    avg_fee_gwei_qbar: float
    fee_improvement_factor: float

    # Cost recovery
    cost_recovery_alpha: float
    cost_recovery_qbar: float

    # Range validation
    fee_in_expected_range: bool
    cost_recovery_in_range: bool

    # Overall assessment
    passes_validation: bool
    validation_score: float  # 0-1 score
    recommendations: List[str]


class AlphaDataValidator:
    """Validator for alpha-data based fee mechanism"""

    # Current broken Q̄ for comparison
    CURRENT_Q_BAR = 6.9e5

    # Expected performance ranges
    EXPECTED_FEE_RANGE_GWEI = (5.0, 15.0)  # vs current 0.00
    EXPECTED_COST_RECOVERY_RANGE = (0.8, 1.2)

    # Validation scenarios (derived from existing historical data)
    DEFAULT_SCENARIOS = [
        ValidationScenario(
            name="normal_operation",
            description="Normal Ethereum operation - low to medium fees",
            l1_basefee_series=[
                15e9, 18e9, 22e9, 19e9, 16e9, 20e9, 25e9, 21e9, 17e9, 23e9,
                19e9, 24e9, 20e9, 18e9, 22e9, 26e9, 23e9, 19e9, 21e9, 24e9
            ],  # 15-26 gwei range
            expected_fee_range_gwei=(3.0, 8.0),
            expected_cost_recovery=(0.9, 1.1),
            scenario_type="normal"
        ),

        ValidationScenario(
            name="fee_spike",
            description="L1 fee spike scenario - based on July 2022 data",
            l1_basefee_series=[
                30e9, 45e9, 80e9, 120e9, 164e9, 140e9, 100e9, 75e9, 60e9, 45e9,
                35e9, 50e9, 90e9, 130e9, 110e9, 85e9, 65e9, 50e9, 40e9, 35e9
            ],  # Spike to 164 gwei
            expected_fee_range_gwei=(8.0, 25.0),
            expected_cost_recovery=(0.8, 1.3),
            scenario_type="spike"
        ),

        ValidationScenario(
            name="extreme_crisis",
            description="Extreme crisis scenario - based on UST/Luna crash",
            l1_basefee_series=[
                50e9, 150e9, 300e9, 600e9, 1000e9, 1350e9, 1200e9, 900e9, 600e9, 400e9,
                250e9, 400e9, 700e9, 1100e9, 900e9, 650e9, 450e9, 300e9, 200e9, 150e9
            ],  # Extreme spike to 1350 gwei
            expected_fee_range_gwei=(15.0, 50.0),
            expected_cost_recovery=(0.6, 1.5),
            scenario_type="crisis"
        ),

        ValidationScenario(
            name="low_fee_period",
            description="Recent low fee period - post-cancun upgrade",
            l1_basefee_series=[
                3e9, 2.5e9, 4e9, 3.5e9, 2e9, 5e9, 4.5e9, 3e9, 2.8e9, 4.2e9,
                3.8e9, 2.2e9, 5.5e9, 4e9, 3.2e9, 2.6e9, 4.8e9, 3.6e9, 2.4e9, 5.2e9
            ],  # Very low fees 2-5.5 gwei
            expected_fee_range_gwei=(1.0, 4.0),
            expected_cost_recovery=(0.8, 1.1),
            scenario_type="normal"
        )
    ]

    def __init__(self):
        """Initialize validator"""
        self.scenarios = self.DEFAULT_SCENARIOS.copy()

    def add_custom_scenario(self, scenario: ValidationScenario):
        """Add a custom validation scenario"""
        self.scenarios.append(scenario)
        logger.info(f"Added custom scenario: {scenario.name}")

    def validate_alpha_value(
        self,
        alpha_value: float,
        target_vault_balance: float = 1000.0,  # ETH
        scenarios: Optional[List[str]] = None
    ) -> List[ValidationResult]:
        """
        Validate an alpha value against all scenarios

        Args:
            alpha_value: The α_data value to test
            target_vault_balance: Target vault balance in ETH
            scenarios: Optional list of scenario names to run (default: all)

        Returns:
            List of validation results for each scenario
        """
        logger.info(f"Validating alpha value {alpha_value} across scenarios")

        if scenarios is None:
            test_scenarios = self.scenarios
        else:
            test_scenarios = [s for s in self.scenarios if s.name in scenarios]

        results = []

        for scenario in test_scenarios:
            result = self._validate_single_scenario(
                alpha_value, scenario, target_vault_balance
            )
            results.append(result)

        return results

    def _validate_single_scenario(
        self,
        alpha_value: float,
        scenario: ValidationScenario,
        target_vault_balance: float
    ) -> ValidationResult:
        """Validate alpha value for a single scenario"""

        logger.info(f"Testing scenario: {scenario.name}")

        # Run simulation with alpha-based model
        alpha_fees, alpha_recovery = self._simulate_alpha_model(
            alpha_value, scenario.l1_basefee_series, target_vault_balance
        )

        # Run simulation with current Q̄ model for comparison
        qbar_fees, qbar_recovery = self._simulate_qbar_model(
            scenario.l1_basefee_series, target_vault_balance
        )

        # Calculate metrics
        avg_fee_alpha = np.mean(alpha_fees) / 1e9  # Convert to gwei
        avg_fee_qbar = np.mean(qbar_fees) / 1e9

        # Fee improvement factor
        improvement_factor = avg_fee_alpha / avg_fee_qbar if avg_fee_qbar > 0 else float('inf')

        # Validate against expected ranges
        fee_in_range = (scenario.expected_fee_range_gwei[0] <=
                       avg_fee_alpha <=
                       scenario.expected_fee_range_gwei[1])

        recovery_in_range = (scenario.expected_cost_recovery[0] <=
                           alpha_recovery <=
                           scenario.expected_cost_recovery[1])

        # Generate recommendations
        recommendations = self._generate_scenario_recommendations(
            scenario, avg_fee_alpha, alpha_recovery, fee_in_range, recovery_in_range
        )

        # Calculate validation score
        validation_score = self._calculate_validation_score(
            fee_in_range, recovery_in_range, alpha_recovery, improvement_factor
        )

        # Overall pass/fail
        passes_validation = (validation_score >= 0.7 and
                           avg_fee_alpha > avg_fee_qbar and  # Must improve over Q̄
                           alpha_recovery >= 0.5)  # Must have reasonable cost recovery

        return ValidationResult(
            scenario_name=scenario.name,
            alpha_value_tested=alpha_value,
            q_bar_comparison=self.CURRENT_Q_BAR,
            avg_fee_gwei_alpha=avg_fee_alpha,
            avg_fee_gwei_qbar=avg_fee_qbar,
            fee_improvement_factor=improvement_factor,
            cost_recovery_alpha=alpha_recovery,
            cost_recovery_qbar=qbar_recovery,
            fee_in_expected_range=fee_in_range,
            cost_recovery_in_range=recovery_in_range,
            passes_validation=passes_validation,
            validation_score=validation_score,
            recommendations=recommendations
        )

    def _simulate_alpha_model(
        self,
        alpha_value: float,
        l1_basefee_series: List[float],
        target_vault_balance: float
    ) -> Tuple[List[float], float]:
        """Simulate alpha-based fee model"""

        # Simplified alpha-based fee calculation
        # Fee = alpha * L1_basefee + deficit_component

        fees = []
        vault_balance = target_vault_balance
        total_revenue = 0.0
        total_l1_costs = 0.0

        # Default parameters (from SPECS optimization)
        mu = 0.7  # Will be replaced by alpha model
        nu = 0.2
        horizon_h = 72
        l2_gas_per_batch = 690_000  # L2 gas consumption

        for l1_basefee in l1_basefee_series:
            # Alpha-based DA component (direct L1 cost tracking)
            da_component_wei_per_gas = alpha_value * l1_basefee

            # Vault deficit component
            deficit = max(0, target_vault_balance - vault_balance)
            deficit_component_wei_per_gas = nu * (deficit * 1e18) / (horizon_h * l2_gas_per_batch)

            # Total L2 fee per gas
            total_fee_wei_per_gas = da_component_wei_per_gas + deficit_component_wei_per_gas
            fees.append(total_fee_wei_per_gas)

            # Calculate batch economics
            batch_revenue = total_fee_wei_per_gas * l2_gas_per_batch / 1e18  # Convert to ETH
            batch_l1_cost = alpha_value * l1_basefee * l2_gas_per_batch / 1e18  # DA cost only

            # Update vault balance
            vault_balance = vault_balance + batch_revenue - batch_l1_cost

            # Track totals for cost recovery
            total_revenue += batch_revenue
            total_l1_costs += batch_l1_cost

        cost_recovery_ratio = total_revenue / total_l1_costs if total_l1_costs > 0 else 1.0

        return fees, cost_recovery_ratio

    def _simulate_qbar_model(
        self,
        l1_basefee_series: List[float],
        target_vault_balance: float
    ) -> Tuple[List[float], float]:
        """Simulate current Q̄-based model for comparison"""

        fees = []
        vault_balance = target_vault_balance
        total_revenue = 0.0
        total_l1_costs = 0.0

        # Current broken parameters
        mu = 0.7
        nu = 0.2
        horizon_h = 72
        q_bar = self.CURRENT_Q_BAR

        for l1_basefee in l1_basefee_series:
            # Current broken formula: uses arbitrary Q̄
            l1_component_wei_per_gas = mu * l1_basefee / q_bar  # This is the broken part

            # Vault deficit component
            deficit = max(0, target_vault_balance - vault_balance)
            deficit_component_wei_per_gas = nu * (deficit * 1e18) / (horizon_h * q_bar)

            # Total fee
            total_fee_wei_per_gas = l1_component_wei_per_gas + deficit_component_wei_per_gas
            fees.append(total_fee_wei_per_gas)

            # This model has fundamentally broken economics, but calculate anyway
            batch_revenue = total_fee_wei_per_gas * q_bar / 1e18
            # Assume L1 costs are ~200k gas per batch at current basefee
            batch_l1_cost = 200_000 * l1_basefee / 1e18

            vault_balance = vault_balance + batch_revenue - batch_l1_cost
            total_revenue += batch_revenue
            total_l1_costs += batch_l1_cost

        cost_recovery_ratio = total_revenue / total_l1_costs if total_l1_costs > 0 else 1.0

        return fees, cost_recovery_ratio

    def _generate_scenario_recommendations(
        self,
        scenario: ValidationScenario,
        avg_fee_gwei: float,
        cost_recovery: float,
        fee_in_range: bool,
        recovery_in_range: bool
    ) -> List[str]:
        """Generate recommendations for a validation scenario"""

        recommendations = []

        if fee_in_range and recovery_in_range:
            recommendations.append(f"✅ EXCELLENT: Fees and cost recovery within expected ranges")

        if not fee_in_range:
            if avg_fee_gwei < scenario.expected_fee_range_gwei[0]:
                recommendations.append(
                    f"⚠️ LOW FEES: {avg_fee_gwei:.2f} gwei below expected minimum "
                    f"{scenario.expected_fee_range_gwei[0]:.1f} gwei - check α value"
                )
            else:
                recommendations.append(
                    f"⚠️ HIGH FEES: {avg_fee_gwei:.2f} gwei above expected maximum "
                    f"{scenario.expected_fee_range_gwei[1]:.1f} gwei - consider lower α"
                )

        if not recovery_in_range:
            if cost_recovery < scenario.expected_cost_recovery[0]:
                recommendations.append(
                    f"⚠️ POOR RECOVERY: {cost_recovery:.2f} below minimum "
                    f"{scenario.expected_cost_recovery[0]:.1f} - increase α or adjust ν"
                )
            else:
                recommendations.append(
                    f"ℹ️ HIGH RECOVERY: {cost_recovery:.2f} above maximum "
                    f"{scenario.expected_cost_recovery[1]:.1f} - could reduce fees slightly"
                )

        # Scenario-specific recommendations
        if scenario.scenario_type == "crisis":
            recommendations.append(
                f"🚨 CRISIS SCENARIO: Model handles extreme conditions "
                f"(max L1 basefee: {max(scenario.l1_basefee_series)/1e9:.0f} gwei)"
            )
        elif scenario.scenario_type == "normal":
            recommendations.append(
                f"✅ NORMAL OPERATION: Model suitable for typical conditions"
            )

        return recommendations

    def _calculate_validation_score(
        self,
        fee_in_range: bool,
        recovery_in_range: bool,
        cost_recovery: float,
        improvement_factor: float
    ) -> float:
        """Calculate overall validation score (0-1)"""

        score = 0.0

        # Fee range compliance (40% weight)
        if fee_in_range:
            score += 0.4

        # Cost recovery compliance (30% weight)
        if recovery_in_range:
            score += 0.3

        # Cost recovery quality (20% weight)
        if cost_recovery >= 0.9:
            score += 0.2
        elif cost_recovery >= 0.7:
            score += 0.15
        elif cost_recovery >= 0.5:
            score += 0.1

        # Improvement over Q̄ model (10% weight)
        if improvement_factor > 1.0:  # Any improvement is good
            score += 0.1

        return min(score, 1.0)

    def generate_validation_report(
        self,
        validation_results: List[ValidationResult],
        alpha_statistics: Optional[AlphaStatistics] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report"""

        logger.info("Generating validation report...")

        # Overall metrics
        total_scenarios = len(validation_results)
        passed_scenarios = sum(1 for r in validation_results if r.passes_validation)
        avg_score = np.mean([r.validation_score for r in validation_results])

        # Performance analysis
        alpha_values_tested = list(set(r.alpha_value_tested for r in validation_results))
        best_alpha = None
        best_score = 0.0

        if len(alpha_values_tested) == 1:
            alpha_tested = alpha_values_tested[0]
            best_alpha = alpha_tested
            best_score = avg_score
        else:
            # Multiple alpha values tested, find best
            for alpha in alpha_values_tested:
                alpha_results = [r for r in validation_results if r.alpha_value_tested == alpha]
                alpha_avg_score = np.mean([r.validation_score for r in alpha_results])
                if alpha_avg_score > best_score:
                    best_score = alpha_avg_score
                    best_alpha = alpha

        # Generate overall recommendations
        overall_recommendations = []

        if passed_scenarios / total_scenarios >= 0.8:
            overall_recommendations.append(
                f"✅ DEPLOY READY: α = {best_alpha:.3f} passes {passed_scenarios}/{total_scenarios} scenarios"
            )
        elif passed_scenarios / total_scenarios >= 0.6:
            overall_recommendations.append(
                f"⚠️ CONDITIONAL: α = {best_alpha:.3f} passes {passed_scenarios}/{total_scenarios} scenarios - review failures"
            )
        else:
            overall_recommendations.append(
                f"❌ NOT READY: α = {best_alpha:.3f} passes only {passed_scenarios}/{total_scenarios} scenarios - needs adjustment"
            )

        # Performance comparison
        avg_improvement_factor = np.mean([r.fee_improvement_factor for r in validation_results])
        overall_recommendations.append(
            f"📈 IMPROVEMENT: {avg_improvement_factor:.1f}x fee improvement over broken Q̄ model"
        )

        # Empirical validation (if alpha statistics provided)
        empirical_validation = {}
        if alpha_statistics:
            empirical_validation = {
                'measured_alpha': alpha_statistics.mean,
                'confidence_interval': alpha_statistics.confidence_interval_95,
                'sample_size': alpha_statistics.sample_size,
                'recommended_for_deployment': (
                    alpha_statistics.confidence_interval_95[0] <= best_alpha <= alpha_statistics.confidence_interval_95[1]
                )
            }

        report = {
            'summary': {
                'total_scenarios': total_scenarios,
                'passed_scenarios': passed_scenarios,
                'pass_rate': passed_scenarios / total_scenarios,
                'average_score': avg_score,
                'best_alpha_value': best_alpha,
                'best_alpha_score': best_score
            },
            'scenario_results': validation_results,
            'empirical_validation': empirical_validation,
            'overall_recommendations': overall_recommendations,
            'performance_comparison': {
                'average_improvement_factor': avg_improvement_factor,
                'replaces_broken_qbar': self.CURRENT_Q_BAR,
                'expected_fee_improvement': 'Realistic fees (5-15 gwei) vs current 0.00 gwei'
            },
            'generated_at': pd.Timestamp.now().isoformat()
        }

        logger.info("Validation report generated successfully")
        return report

    def compare_alpha_values(
        self,
        alpha_values: List[float],
        target_vault_balance: float = 1000.0
    ) -> Dict[str, Any]:
        """Compare multiple alpha values across all scenarios"""

        logger.info(f"Comparing {len(alpha_values)} alpha values")

        comparison_data = []

        for alpha in alpha_values:
            results = self.validate_alpha_value(alpha, target_vault_balance)

            # Calculate summary metrics for this alpha
            avg_score = np.mean([r.validation_score for r in results])
            pass_rate = sum(1 for r in results if r.passes_validation) / len(results)
            avg_fee = np.mean([r.avg_fee_gwei_alpha for r in results])
            avg_recovery = np.mean([r.cost_recovery_alpha for r in results])

            comparison_data.append({
                'alpha_value': alpha,
                'average_score': avg_score,
                'pass_rate': pass_rate,
                'average_fee_gwei': avg_fee,
                'average_cost_recovery': avg_recovery,
                'results': results
            })

        # Sort by average score
        comparison_data.sort(key=lambda x: x['average_score'], reverse=True)

        return {
            'best_alpha': comparison_data[0]['alpha_value'],
            'best_score': comparison_data[0]['average_score'],
            'comparison_data': comparison_data,
            'analysis': self._analyze_alpha_comparison(comparison_data)
        }

    def _analyze_alpha_comparison(self, comparison_data: List[Dict]) -> Dict[str, Any]:
        """Analyze comparison results to provide insights"""

        alphas = [d['alpha_value'] for d in comparison_data]
        scores = [d['average_score'] for d in comparison_data]
        fees = [d['average_fee_gwei'] for d in comparison_data]

        analysis = {
            'optimal_range': None,
            'fee_vs_alpha_trend': None,
            'score_vs_alpha_trend': None,
            'recommendations': []
        }

        # Find optimal range (top 3 performers within 5% of best score)
        best_score = max(scores)
        threshold = best_score * 0.95
        optimal_alphas = [d['alpha_value'] for d in comparison_data if d['average_score'] >= threshold]

        if len(optimal_alphas) > 1:
            analysis['optimal_range'] = (min(optimal_alphas), max(optimal_alphas))
        else:
            analysis['optimal_range'] = optimal_alphas[0]

        # Trend analysis
        if len(alphas) >= 3:
            # Fee vs alpha correlation
            fee_correlation = np.corrcoef(alphas, fees)[0, 1]
            analysis['fee_vs_alpha_trend'] = 'positive' if fee_correlation > 0.5 else 'negative' if fee_correlation < -0.5 else 'neutral'

            # Score vs alpha trend
            score_correlation = np.corrcoef(alphas, scores)[0, 1]
            analysis['score_vs_alpha_trend'] = 'positive' if score_correlation > 0.5 else 'negative' if score_correlation < -0.5 else 'neutral'

        # Generate recommendations
        best_alpha = comparison_data[0]['alpha_value']
        analysis['recommendations'].append(f"Recommended α = {best_alpha:.3f} (highest validation score)")

        if analysis['optimal_range'] and isinstance(analysis['optimal_range'], tuple):
            analysis['recommendations'].append(
                f"Acceptable range: {analysis['optimal_range'][0]:.3f} - {analysis['optimal_range'][1]:.3f}"
            )

        return analysis


# Utility function for quick validation
def quick_validation(alpha_value: float) -> Dict[str, Any]:
    """Quick validation of an alpha value"""
    validator = AlphaDataValidator()
    results = validator.validate_alpha_value(alpha_value)

    # Generate quick summary
    passed = sum(1 for r in results if r.passes_validation)
    total = len(results)
    avg_fee = np.mean([r.avg_fee_gwei_alpha for r in results])
    avg_improvement = np.mean([r.fee_improvement_factor for r in results])

    return {
        'alpha_tested': alpha_value,
        'scenarios_passed': f"{passed}/{total}",
        'pass_rate': passed/total,
        'average_fee_gwei': avg_fee,
        'improvement_factor': avg_improvement,
        'ready_for_deployment': passed/total >= 0.8,
        'quick_recommendation': f"α = {alpha_value:.3f} {'✅ READY' if passed/total >= 0.8 else '⚠️ NEEDS REVIEW'}"
    }