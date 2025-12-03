"""
Canonical Taiko Fee Mechanism Metrics Implementation

This module provides the SINGLE SOURCE OF TRUTH for all performance metrics calculations.
All metric computations across the system must use this module to ensure consistency.

Key Features:
- Comprehensive fee mechanism performance metrics
- User experience metrics (affordability, stability, predictability)
- Protocol safety metrics (insolvency protection, deficit control, resilience)
- Economic efficiency metrics (vault utilization, capital efficiency)
- Statistical validation and outlier handling
- Configurable time windows and thresholds
- Detailed mathematical formulations with references

Metrics Categories:
1. User Experience: Fee affordability, stability, predictability
2. Protocol Safety: Vault resilience, deficit control, stress recovery
3. Economic Efficiency: Capital utilization, correction speed
4. System Performance: Throughput, cost coverage, revenue efficiency

Usage:
    calculator = CanonicalMetricsCalculator()
    metrics = calculator.calculate_comprehensive_metrics(simulation_results)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import scipy.stats as stats


class MetricCategory(Enum):
    """Categories of performance metrics."""
    USER_EXPERIENCE = "user_experience"
    PROTOCOL_SAFETY = "protocol_safety"
    ECONOMIC_EFFICIENCY = "economic_efficiency"
    SYSTEM_PERFORMANCE = "system_performance"


@dataclass
class MetricThresholds:
    """Configurable thresholds for metric evaluation."""

    # Fee stability thresholds
    excellent_cv: float = 0.5           # CV < 0.5 = excellent stability
    good_cv: float = 1.0                # CV < 1.0 = good stability

    # Deficit duration thresholds
    excellent_deficit_pct: float = 5.0  # < 5% time underfunded = excellent
    good_deficit_pct: float = 15.0      # < 15% time underfunded = good

    # L1 tracking error thresholds
    excellent_tracking: float = 0.3     # Normalized error < 0.3 = excellent
    good_tracking: float = 0.6          # Normalized error < 0.6 = good

    # Vault insolvency thresholds
    insolvency_threshold: float = 0.01   # < 1% of target = near insolvency
    critical_deficit: float = 0.5       # 50% of target = critical deficit

    # Fee affordability thresholds (in gwei)
    expensive_fee: float = 10.0         # > 10 gwei = expensive
    reasonable_fee: float = 5.0         # < 5 gwei = reasonable

    # Predictability window sizes (in steps)
    short_term_window: int = 1800       # 1 hour
    long_term_window: int = 10800       # 6 hours


@dataclass
class ComprehensiveMetrics:
    """Complete set of fee mechanism performance metrics."""

    # User Experience Metrics
    average_fee_gwei: float
    fee_stability_cv: float
    fee_affordability_score: float
    fee_predictability_1h: float
    fee_predictability_6h: float
    fee_rate_of_change_p95: float

    # Protocol Safety Metrics
    time_underfunded_pct: float
    max_deficit_ratio: float
    insolvency_protection_score: float
    vault_stress_resilience: float
    deficit_recovery_rate: float
    underfunding_resistance: float

    # Economic Efficiency Metrics
    vault_utilization_score: float
    capital_efficiency: float
    cost_coverage_ratio: float
    revenue_efficiency: float
    deficit_correction_rate: float

    # System Performance Metrics
    l1_tracking_error: float
    correlation_with_l1: float
    transaction_throughput: float
    fee_revenue_total: float
    l1_cost_total: float
    net_revenue: float

    # Statistical Validation
    simulation_length: int
    data_completeness: float
    outlier_percentage: float

    # Overall Scores
    user_experience_score: float
    protocol_safety_score: float
    economic_efficiency_score: float
    overall_performance_score: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for serialization."""
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}


class CanonicalMetricsCalculator:
    """
    SINGLE SOURCE OF TRUTH for all fee mechanism performance metrics.

    This class provides authoritative implementations of all metrics used
    throughout the system for evaluation and optimization.
    """

    def __init__(self, thresholds: Optional[MetricThresholds] = None):
        """Initialize calculator with configurable thresholds."""
        self.thresholds = thresholds or MetricThresholds()

        # Configuration
        self.taiko_block_time = 2.0      # seconds
        self.target_vault_balance = 1000.0  # ETH
        self.min_fee_eth = 1e-8          # Minimum fee floor

    def calculate_comprehensive_metrics(self, simulation_results: Dict[str, List[float]]) -> ComprehensiveMetrics:
        """
        Calculate all performance metrics from simulation results.

        Args:
            simulation_results: Dictionary with simulation time series data

        Returns:
            ComprehensiveMetrics object with all calculated metrics
        """

        # Validate and preprocess data
        validated_results = self._validate_simulation_data(simulation_results)

        # Extract key arrays
        fees = np.array(validated_results['estimatedFee'])
        vault_balances = np.array(validated_results['vaultBalance'])
        vault_deficits = np.array(validated_results.get('vaultDeficit', []))
        l1_basefees = np.array(validated_results['l1Basefee'])
        tx_volumes = np.array(validated_results['transactionVolume'])
        fees_collected = np.array(validated_results.get('feesCollected', []))
        l1_costs_paid = np.array(validated_results.get('l1CostsPaid', []))

        # Calculate deficits if not provided
        if len(vault_deficits) == 0:
            vault_deficits = np.maximum(0, self.target_vault_balance - vault_balances)

        # User Experience Metrics
        user_metrics = self._calculate_user_experience_metrics(fees, l1_basefees)

        # Protocol Safety Metrics
        safety_metrics = self._calculate_protocol_safety_metrics(vault_balances, vault_deficits)

        # Economic Efficiency Metrics
        efficiency_metrics = self._calculate_economic_efficiency_metrics(
            vault_balances, vault_deficits, fees_collected, l1_costs_paid
        )

        # System Performance Metrics
        performance_metrics = self._calculate_system_performance_metrics(
            fees, l1_basefees, tx_volumes, fees_collected, l1_costs_paid
        )

        # Statistical Validation
        validation_metrics = self._calculate_validation_metrics(validated_results)

        # Overall Scores
        overall_scores = self._calculate_overall_scores(user_metrics, safety_metrics, efficiency_metrics)

        # Combine all metrics
        return ComprehensiveMetrics(
            # User Experience
            average_fee_gwei=user_metrics['average_fee_gwei'],
            fee_stability_cv=user_metrics['fee_stability_cv'],
            fee_affordability_score=user_metrics['fee_affordability_score'],
            fee_predictability_1h=user_metrics['fee_predictability_1h'],
            fee_predictability_6h=user_metrics['fee_predictability_6h'],
            fee_rate_of_change_p95=user_metrics['fee_rate_of_change_p95'],

            # Protocol Safety
            time_underfunded_pct=safety_metrics['time_underfunded_pct'],
            max_deficit_ratio=safety_metrics['max_deficit_ratio'],
            insolvency_protection_score=safety_metrics['insolvency_protection_score'],
            vault_stress_resilience=safety_metrics['vault_stress_resilience'],
            deficit_recovery_rate=safety_metrics['deficit_recovery_rate'],
            underfunding_resistance=safety_metrics['underfunding_resistance'],

            # Economic Efficiency
            vault_utilization_score=efficiency_metrics['vault_utilization_score'],
            capital_efficiency=efficiency_metrics['capital_efficiency'],
            cost_coverage_ratio=efficiency_metrics['cost_coverage_ratio'],
            revenue_efficiency=efficiency_metrics['revenue_efficiency'],
            deficit_correction_rate=efficiency_metrics['deficit_correction_rate'],

            # System Performance
            l1_tracking_error=performance_metrics['l1_tracking_error'],
            correlation_with_l1=performance_metrics['correlation_with_l1'],
            transaction_throughput=performance_metrics['transaction_throughput'],
            fee_revenue_total=performance_metrics['fee_revenue_total'],
            l1_cost_total=performance_metrics['l1_cost_total'],
            net_revenue=performance_metrics['net_revenue'],

            # Validation
            simulation_length=validation_metrics['simulation_length'],
            data_completeness=validation_metrics['data_completeness'],
            outlier_percentage=validation_metrics['outlier_percentage'],

            # Overall Scores
            user_experience_score=overall_scores['user_experience_score'],
            protocol_safety_score=overall_scores['protocol_safety_score'],
            economic_efficiency_score=overall_scores['economic_efficiency_score'],
            overall_performance_score=overall_scores['overall_performance_score']
        )

    def _validate_simulation_data(self, results: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Validate and clean simulation data."""
        required_fields = ['estimatedFee', 'vaultBalance', 'l1Basefee', 'transactionVolume']

        # Check required fields
        for field in required_fields:
            if field not in results:
                raise ValueError(f"Missing required field: {field}")
            if not results[field]:
                raise ValueError(f"Empty data for required field: {field}")

        # Check data consistency
        lengths = {field: len(results[field]) for field in required_fields}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Inconsistent data lengths: {lengths}")

        # Clean data (remove NaN, inf, negative values where inappropriate)
        cleaned_results = {}
        for field, data in results.items():
            cleaned_data = []
            for value in data:
                if np.isnan(value) or np.isinf(value):
                    # Replace with interpolated or default values
                    if field in ['estimatedFee', 'vaultBalance']:
                        cleaned_data.append(cleaned_data[-1] if cleaned_data else 0.0)
                    else:
                        cleaned_data.append(0.0)
                elif field in ['estimatedFee', 'transactionVolume'] and value < 0:
                    cleaned_data.append(0.0)
                else:
                    cleaned_data.append(value)
            cleaned_results[field] = cleaned_data

        return cleaned_results

    def _calculate_user_experience_metrics(self, fees: np.ndarray, l1_basefees: np.ndarray) -> Dict[str, float]:
        """Calculate metrics related to user experience."""

        # Average fee in gwei
        average_fee_gwei = np.mean(fees) * 1e9

        # Fee stability (coefficient of variation)
        fee_stability_cv = np.std(fees) / (np.mean(fees) + 1e-12)

        # Fee affordability score (logarithmic penalty for high fees)
        fee_affordability_score = self._calculate_affordability_score(fees)

        # Fee predictability over different time windows
        fee_predictability_1h = self._calculate_predictability(fees, self.thresholds.short_term_window)
        fee_predictability_6h = self._calculate_predictability(fees, self.thresholds.long_term_window)

        # Fee rate of change (95th percentile)
        fee_rate_of_change_p95 = self._calculate_rate_of_change_p95(fees)

        return {
            'average_fee_gwei': average_fee_gwei,
            'fee_stability_cv': fee_stability_cv,
            'fee_affordability_score': fee_affordability_score,
            'fee_predictability_1h': fee_predictability_1h,
            'fee_predictability_6h': fee_predictability_6h,
            'fee_rate_of_change_p95': fee_rate_of_change_p95
        }

    def _calculate_protocol_safety_metrics(self, vault_balances: np.ndarray, vault_deficits: np.ndarray) -> Dict[str, float]:
        """Calculate metrics related to protocol safety and vault management."""

        # Time underfunded percentage
        significant_deficit = vault_deficits > (self.target_vault_balance * 0.01)  # > 1% of target
        time_underfunded_pct = (np.sum(significant_deficit) / len(vault_deficits)) * 100.0

        # Maximum deficit ratio
        max_deficit_ratio = np.max(vault_deficits) / self.target_vault_balance

        # Insolvency protection score
        insolvency_protection_score = self._calculate_insolvency_protection(vault_balances)

        # Vault stress resilience (recovery after stress events)
        vault_stress_resilience = self._calculate_stress_resilience(vault_balances)

        # Deficit recovery rate
        deficit_recovery_rate = self._calculate_deficit_recovery_rate(vault_deficits)

        # Underfunding resistance
        underfunding_resistance = self._calculate_underfunding_resistance(vault_deficits)

        return {
            'time_underfunded_pct': time_underfunded_pct,
            'max_deficit_ratio': max_deficit_ratio,
            'insolvency_protection_score': insolvency_protection_score,
            'vault_stress_resilience': vault_stress_resilience,
            'deficit_recovery_rate': deficit_recovery_rate,
            'underfunding_resistance': underfunding_resistance
        }

    def _calculate_economic_efficiency_metrics(self,
                                             vault_balances: np.ndarray,
                                             vault_deficits: np.ndarray,
                                             fees_collected: np.ndarray,
                                             l1_costs_paid: np.ndarray) -> Dict[str, float]:
        """Calculate metrics related to economic efficiency."""

        # Vault utilization score (deviation from target)
        vault_utilization_score = self._calculate_vault_utilization(vault_balances)

        # Capital efficiency (return on capital)
        capital_efficiency = self._calculate_capital_efficiency(vault_balances, fees_collected, l1_costs_paid)

        # Cost coverage ratio
        cost_coverage_ratio = self._calculate_cost_coverage_ratio(fees_collected, l1_costs_paid)

        # Revenue efficiency
        revenue_efficiency = self._calculate_revenue_efficiency(fees_collected, l1_costs_paid)

        # Deficit correction rate
        deficit_correction_rate = self._calculate_deficit_correction_rate_efficiency(vault_deficits)

        return {
            'vault_utilization_score': vault_utilization_score,
            'capital_efficiency': capital_efficiency,
            'cost_coverage_ratio': cost_coverage_ratio,
            'revenue_efficiency': revenue_efficiency,
            'deficit_correction_rate': deficit_correction_rate
        }

    def _calculate_system_performance_metrics(self,
                                            fees: np.ndarray,
                                            l1_basefees: np.ndarray,
                                            tx_volumes: np.ndarray,
                                            fees_collected: np.ndarray,
                                            l1_costs_paid: np.ndarray) -> Dict[str, float]:
        """Calculate system-level performance metrics."""

        # L1 tracking error
        l1_tracking_error = self._calculate_l1_tracking_error(fees, l1_basefees)

        # Correlation with L1
        correlation_with_l1 = self._calculate_correlation_with_l1(fees, l1_basefees)

        # Transaction throughput
        transaction_throughput = np.mean(tx_volumes)

        # Financial metrics
        fee_revenue_total = np.sum(fees_collected) if len(fees_collected) > 0 else 0.0
        l1_cost_total = np.sum(l1_costs_paid) if len(l1_costs_paid) > 0 else 0.0
        net_revenue = fee_revenue_total - l1_cost_total

        return {
            'l1_tracking_error': l1_tracking_error,
            'correlation_with_l1': correlation_with_l1,
            'transaction_throughput': transaction_throughput,
            'fee_revenue_total': fee_revenue_total,
            'l1_cost_total': l1_cost_total,
            'net_revenue': net_revenue
        }

    def _calculate_validation_metrics(self, results: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate data validation and quality metrics."""

        simulation_length = len(results['estimatedFee'])

        # Data completeness (percentage of non-zero values)
        total_points = sum(len(data) for data in results.values())
        non_zero_points = sum(np.count_nonzero(data) for data in results.values())
        data_completeness = (non_zero_points / total_points) * 100.0 if total_points > 0 else 0.0

        # Outlier percentage (using Z-score > 3)
        fees = np.array(results['estimatedFee'])
        z_scores = np.abs(stats.zscore(fees))
        outlier_percentage = (np.sum(z_scores > 3) / len(fees)) * 100.0

        return {
            'simulation_length': simulation_length,
            'data_completeness': data_completeness,
            'outlier_percentage': outlier_percentage
        }

    def _calculate_overall_scores(self,
                                user_metrics: Dict[str, float],
                                safety_metrics: Dict[str, float],
                                efficiency_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate overall composite scores."""

        # User Experience Score (0-1, higher is better)
        ux_components = [
            self._normalize_affordability_score(user_metrics['fee_affordability_score']),
            self._normalize_stability_score(user_metrics['fee_stability_cv']),
            user_metrics['fee_predictability_1h'],
            user_metrics['fee_predictability_6h']
        ]
        user_experience_score = np.mean(ux_components)

        # Protocol Safety Score (0-1, higher is better)
        safety_components = [
            self._normalize_underfunded_time(safety_metrics['time_underfunded_pct']),
            safety_metrics['insolvency_protection_score'],
            safety_metrics['vault_stress_resilience'],
            safety_metrics['underfunding_resistance']
        ]
        protocol_safety_score = np.mean(safety_components)

        # Economic Efficiency Score (0-1, higher is better)
        efficiency_components = [
            efficiency_metrics['vault_utilization_score'],
            min(1.0, efficiency_metrics['capital_efficiency']),
            min(1.0, efficiency_metrics['cost_coverage_ratio']),
            efficiency_metrics['deficit_correction_rate']
        ]
        economic_efficiency_score = np.mean(efficiency_components)

        # Overall Performance Score (weighted average)
        overall_performance_score = (
            0.4 * user_experience_score +
            0.4 * protocol_safety_score +
            0.2 * economic_efficiency_score
        )

        return {
            'user_experience_score': user_experience_score,
            'protocol_safety_score': protocol_safety_score,
            'economic_efficiency_score': economic_efficiency_score,
            'overall_performance_score': overall_performance_score
        }

    # Helper methods for specific metric calculations

    def _calculate_affordability_score(self, fees: np.ndarray) -> float:
        """Calculate fee affordability score using logarithmic scaling."""
        avg_fee_gwei = np.mean(fees) * 1e9
        if avg_fee_gwei <= 0:
            return 1.0

        # Logarithmic penalty: score decreases as fees increase
        # Score = 1 at 1 gwei, 0.5 at 10 gwei, 0 at 100 gwei
        score = max(0.0, 1.0 - np.log10(max(avg_fee_gwei, 0.1)) / 2.0)
        return min(1.0, score)

    def _calculate_predictability(self, fees: np.ndarray, window_size: int) -> float:
        """Calculate fee predictability over specified window size."""
        if len(fees) < window_size:
            window_size = len(fees)

        if window_size < 2:
            return 1.0

        predictabilities = []
        for i in range(len(fees) - window_size + 1):
            window = fees[i:i + window_size]
            cv = np.std(window) / (np.mean(window) + 1e-12)
            predictability = max(0.0, 1.0 - cv)
            predictabilities.append(predictability)

        return np.mean(predictabilities) if predictabilities else 1.0

    def _calculate_rate_of_change_p95(self, fees: np.ndarray) -> float:
        """Calculate 95th percentile of fee rate of change."""
        if len(fees) < 2:
            return 0.0

        rate_changes = []
        for i in range(1, len(fees)):
            if fees[i-1] > 0:
                change = abs(fees[i] - fees[i-1]) / fees[i-1]
                rate_changes.append(change)

        return np.percentile(rate_changes, 95) if rate_changes else 0.0

    def _calculate_insolvency_protection(self, vault_balances: np.ndarray) -> float:
        """Calculate insolvency protection score."""
        min_balance = np.min(vault_balances)
        critical_threshold = self.target_vault_balance * self.thresholds.insolvency_threshold

        if min_balance >= critical_threshold:
            return 1.0
        elif min_balance >= 0:
            # Partial score based on how close to threshold
            return min_balance / critical_threshold
        else:
            # Negative balance (insolvency)
            return 0.0

    def _calculate_stress_resilience(self, vault_balances: np.ndarray) -> float:
        """Calculate vault stress resilience (recovery speed after stress events)."""
        stress_threshold = self.target_vault_balance * 0.8  # 20% below target
        recovery_threshold = self.target_vault_balance * 0.9  # 10% below target

        recovery_times = []
        in_stress = False
        stress_start = 0

        for i, balance in enumerate(vault_balances):
            if balance < stress_threshold and not in_stress:
                in_stress = True
                stress_start = i
            elif balance >= recovery_threshold and in_stress:
                in_stress = False
                recovery_time = i - stress_start
                recovery_times.append(recovery_time)

        if not recovery_times:
            return 1.0  # No stress events

        # Score based on average recovery time (faster = better)
        avg_recovery = np.mean(recovery_times)
        max_acceptable_recovery = 100  # steps
        score = max(0.0, 1.0 - avg_recovery / max_acceptable_recovery)
        return score

    def _calculate_deficit_recovery_rate(self, vault_deficits: np.ndarray) -> float:
        """Calculate rate of deficit recovery."""
        if np.all(vault_deficits == 0):
            return 1.0

        deficit_reductions = []
        in_deficit = False
        deficit_start = 0
        max_deficit = 0

        for i, deficit in enumerate(vault_deficits):
            if deficit > 0 and not in_deficit:
                in_deficit = True
                deficit_start = i
                max_deficit = deficit
            elif deficit > 0 and in_deficit:
                max_deficit = max(max_deficit, deficit)
            elif deficit == 0 and in_deficit:
                in_deficit = False
                recovery_time = i - deficit_start
                if recovery_time > 0:
                    recovery_rate = max_deficit / recovery_time
                    deficit_reductions.append(recovery_rate)

        if not deficit_reductions:
            return 0.5  # Some deficit but no complete recoveries

        avg_recovery_rate = np.mean(deficit_reductions)
        # Normalize by reasonable recovery rate
        normalized_rate = min(1.0, avg_recovery_rate / 10.0)  # 10 ETH per step = perfect
        return normalized_rate

    def _calculate_underfunding_resistance(self, vault_deficits: np.ndarray) -> float:
        """Calculate resistance to underfunding."""
        max_deficit_ratio = np.max(vault_deficits) / self.target_vault_balance
        resistance = max(0.0, 1.0 - max_deficit_ratio)
        return resistance

    def _calculate_vault_utilization(self, vault_balances: np.ndarray) -> float:
        """Calculate vault utilization efficiency."""
        deviations = np.abs(vault_balances - self.target_vault_balance) / self.target_vault_balance
        avg_deviation = np.mean(deviations)
        utilization = max(0.0, 1.0 - avg_deviation)
        return utilization

    def _calculate_capital_efficiency(self,
                                    vault_balances: np.ndarray,
                                    fees_collected: np.ndarray,
                                    l1_costs_paid: np.ndarray) -> float:
        """Calculate capital efficiency (return on capital)."""
        if len(fees_collected) == 0 or len(l1_costs_paid) == 0:
            return 0.0

        net_revenue = np.sum(fees_collected) - np.sum(l1_costs_paid)
        avg_capital = np.mean(vault_balances)

        if avg_capital <= 0:
            return 0.0

        return max(0.0, net_revenue / avg_capital)

    def _calculate_cost_coverage_ratio(self, fees_collected: np.ndarray, l1_costs_paid: np.ndarray) -> float:
        """Calculate cost coverage ratio."""
        if len(fees_collected) == 0 or len(l1_costs_paid) == 0:
            return 0.0

        total_fees = np.sum(fees_collected)
        total_costs = np.sum(l1_costs_paid)

        if total_costs == 0:
            return float('inf') if total_fees > 0 else 1.0

        return total_fees / total_costs

    def _calculate_revenue_efficiency(self, fees_collected: np.ndarray, l1_costs_paid: np.ndarray) -> float:
        """Calculate revenue efficiency."""
        cost_coverage = self._calculate_cost_coverage_ratio(fees_collected, l1_costs_paid)
        if cost_coverage >= 1.0:
            return min(1.0, (cost_coverage - 1.0) / 0.2 + 0.5)  # Scale 1.0-1.2 coverage to 0.5-1.0 score
        else:
            return cost_coverage * 0.5  # Below 1.0 coverage scales to 0.0-0.5 score

    def _calculate_deficit_correction_rate_efficiency(self, vault_deficits: np.ndarray) -> float:
        """Calculate deficit correction rate for efficiency metrics."""
        return self._calculate_deficit_recovery_rate(vault_deficits)

    def _calculate_l1_tracking_error(self, fees: np.ndarray, l1_basefees: np.ndarray) -> float:
        """Calculate normalized tracking error between fees and L1 costs."""
        # Normalize both series
        fees_normalized = fees / (np.mean(fees) + 1e-12)
        l1_normalized = l1_basefees / (np.mean(l1_basefees) + 1e-12)

        # Calculate normalized standard deviation of differences
        differences = fees_normalized - l1_normalized
        tracking_error = np.std(differences)

        return tracking_error

    def _calculate_correlation_with_l1(self, fees: np.ndarray, l1_basefees: np.ndarray) -> float:
        """Calculate correlation coefficient between fees and L1 basefees."""
        if len(fees) < 2 or len(l1_basefees) < 2:
            return 0.0

        correlation = np.corrcoef(fees, l1_basefees)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    # Normalization helper methods

    def _normalize_affordability_score(self, affordability_score: float) -> float:
        """Normalize affordability score to 0-1 range."""
        return max(0.0, min(1.0, affordability_score))

    def _normalize_stability_score(self, cv: float) -> float:
        """Convert coefficient of variation to stability score (0-1)."""
        if cv <= self.thresholds.excellent_cv:
            return 1.0
        elif cv <= self.thresholds.good_cv:
            return 1.0 - (cv - self.thresholds.excellent_cv) / (self.thresholds.good_cv - self.thresholds.excellent_cv) * 0.5
        else:
            return max(0.0, 0.5 - (cv - self.thresholds.good_cv) / 2.0)

    def _normalize_underfunded_time(self, underfunded_pct: float) -> float:
        """Convert underfunded time percentage to score (0-1)."""
        if underfunded_pct <= self.thresholds.excellent_deficit_pct:
            return 1.0
        elif underfunded_pct <= self.thresholds.good_deficit_pct:
            range_size = self.thresholds.good_deficit_pct - self.thresholds.excellent_deficit_pct
            return 1.0 - (underfunded_pct - self.thresholds.excellent_deficit_pct) / range_size * 0.5
        else:
            return max(0.0, 0.5 - (underfunded_pct - self.thresholds.good_deficit_pct) / 50.0)


# Convenience functions for quick metric calculations

def calculate_basic_metrics(simulation_results: Dict[str, List[float]]) -> Dict[str, float]:
    """Calculate basic performance metrics quickly."""
    calculator = CanonicalMetricsCalculator()
    comprehensive = calculator.calculate_comprehensive_metrics(simulation_results)

    return {
        'average_fee_gwei': comprehensive.average_fee_gwei,
        'fee_stability_cv': comprehensive.fee_stability_cv,
        'time_underfunded_pct': comprehensive.time_underfunded_pct,
        'l1_tracking_error': comprehensive.l1_tracking_error,
        'overall_score': comprehensive.overall_performance_score
    }


def calculate_user_experience_score(simulation_results: Dict[str, List[float]]) -> float:
    """Calculate user experience score only."""
    calculator = CanonicalMetricsCalculator()
    comprehensive = calculator.calculate_comprehensive_metrics(simulation_results)
    return comprehensive.user_experience_score


def calculate_protocol_safety_score(simulation_results: Dict[str, List[float]]) -> float:
    """Calculate protocol safety score only."""
    calculator = CanonicalMetricsCalculator()
    comprehensive = calculator.calculate_comprehensive_metrics(simulation_results)
    return comprehensive.protocol_safety_score


def validate_metric_thresholds(average_fee_gwei: float,
                              fee_cv: float,
                              underfunded_pct: float,
                              tracking_error: float) -> Dict[str, str]:
    """Validate metrics against standard thresholds."""
    thresholds = MetricThresholds()
    results = {}

    # Fee affordability
    if average_fee_gwei <= thresholds.reasonable_fee:
        results['affordability'] = 'excellent'
    elif average_fee_gwei <= thresholds.expensive_fee:
        results['affordability'] = 'good'
    else:
        results['affordability'] = 'poor'

    # Fee stability
    if fee_cv <= thresholds.excellent_cv:
        results['stability'] = 'excellent'
    elif fee_cv <= thresholds.good_cv:
        results['stability'] = 'good'
    else:
        results['stability'] = 'poor'

    # Underfunded time
    if underfunded_pct <= thresholds.excellent_deficit_pct:
        results['safety'] = 'excellent'
    elif underfunded_pct <= thresholds.good_deficit_pct:
        results['safety'] = 'good'
    else:
        results['safety'] = 'poor'

    # L1 tracking
    if tracking_error <= thresholds.excellent_tracking:
        results['tracking'] = 'excellent'
    elif tracking_error <= thresholds.good_tracking:
        results['tracking'] = 'good'
    else:
        results['tracking'] = 'poor'

    return results