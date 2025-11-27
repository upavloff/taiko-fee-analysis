"""
Revised Metrics Framework for Taiko Fee Mechanism Optimization

Based on protocol researcher feedback, this module implements rigorously justified
metrics focused on fundamental protocol objectives, eliminating poorly justified
proxy metrics and L1 correlation bias.

Key Principles:
1. User Experience: Affordability + Stability (not L1 responsiveness)
2. Protocol Safety: Insolvency risk + Deficit-weighted duration
3. Economic Efficiency: Vault utilization without over/under capitalization
4. Remove: L1 correlation metrics, implementation complexity, mechanism overhead

Constraints (not metrics):
- 6-step alignment: H must be multiple of 6 (batch cycle alignment)
- Production feasibility: Basic safety thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class RevisedMechanismMetrics:
    """
    Rigorously justified metrics container.

    Only includes metrics with clear mathematical definitions and
    direct relevance to protocol performance.
    """

    # === PRIMARY: User Experience ===
    fee_affordability: float              # -log(1 + avg_fee_eth × 1000) - exponential penalty
    fee_stability: float                  # 1 - CV over meaningful time windows
    fee_predictability_1h: float         # Predictability over 1-hour windows (30 steps)
    fee_predictability_6h: float         # Predictability over 6-hour windows (180 steps)

    # === SECONDARY: Protocol Safety ===
    insolvency_probability: float         # P(vault_balance < critical_threshold)
    deficit_weighted_duration: float     # ∑(deficit_magnitude × duration)^2 / total_time
    vault_stress_resilience: float       # Worst-case deficit recovery under stress scenarios
    max_continuous_underfunding: float   # Maximum continuous time below target

    # === TERTIARY: Economic Efficiency ===
    vault_utilization_efficiency: float  # How efficiently vault maintains target (minimize excess capital)
    deficit_correction_rate: float       # Speed of deficit correction (deficit_reduction / time)
    capital_efficiency: float            # Minimize idle capital while maintaining safety

    # === COMPOSITE SCORES ===
    user_experience_score: float         # Weighted combination of UX metrics
    protocol_safety_score: float         # Weighted combination of safety metrics
    economic_efficiency_score: float     # Weighted combination of efficiency metrics

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for analysis."""
        return {k: v for k, v in self.__dict__.items()}

    def get_pareto_objectives(self) -> Tuple[float, float, float]:
        """Return three main objectives for Pareto optimization (for minimization)."""
        return (
            -self.user_experience_score,      # Minimize negative UX (maximize UX)
            -self.protocol_safety_score,      # Minimize negative safety (maximize safety)
            -self.economic_efficiency_score   # Minimize negative efficiency (maximize efficiency)
        )


class RevisedMetricsCalculator:
    """
    Revised metrics calculator with rigorous mathematical definitions.

    Focus on fundamental protocol objectives without proxy metrics.
    """

    def __init__(self, target_balance: float, taiko_block_time: float = 2.0):
        """
        Initialize with protocol parameters.

        Args:
            target_balance: Target vault balance (ETH)
            taiko_block_time: Taiko L2 block time (seconds)
        """
        self.target_balance = target_balance
        self.taiko_block_time = taiko_block_time

        # Critical thresholds for safety analysis
        self.critical_threshold = 0.1 * target_balance    # 10% of target = critical insolvency risk
        self.significant_deficit_threshold = 0.05 * target_balance  # 5% deficit is significant

    def calculate_user_experience_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate user experience metrics focused on what users actually care about.

        Users want:
        1. Low fees (affordability)
        2. Predictable fees (stability over relevant time horizons)
        """
        fees = df['estimated_fee'].values

        # Fee Affordability: Exponential penalty for high fees
        # Formula: -log(1 + avg_fee_eth × 1000)
        # Justification: Users are exponentially sensitive to fee increases
        avg_fee = np.mean(fees)
        fee_affordability = -np.log(1 + avg_fee * 1000) if avg_fee > 0 else 1.0

        # Fee Stability: Overall coefficient of variation
        # Formula: 1 - (std_dev / mean)
        # Justification: Lower relative volatility = better UX
        fee_cv = np.std(fees) / np.mean(fees) if np.mean(fees) > 0 else 0
        fee_stability = max(0, 1 - fee_cv)

        # Fee Predictability over User-Relevant Time Windows
        # 1-hour predictability (30 steps @ 2s per step)
        predictability_1h = self._calculate_fee_predictability(fees, window=30)

        # 6-hour predictability (180 steps @ 2s per step)
        predictability_6h = self._calculate_fee_predictability(fees, window=180)

        return {
            'fee_affordability': fee_affordability,
            'fee_stability': fee_stability,
            'fee_predictability_1h': predictability_1h,
            'fee_predictability_6h': predictability_6h
        }

    def _calculate_fee_predictability(self, fees: np.ndarray, window: int) -> float:
        """
        Calculate fee predictability over a specific time window.

        Measures how well fees can be predicted by users planning transactions.
        Uses rolling standard deviation normalized by rolling mean.
        """
        if len(fees) < window:
            return 1.0  # Default to perfect predictability for short sequences

        fee_series = pd.Series(fees)
        rolling_std = fee_series.rolling(window=window).std()
        rolling_mean = fee_series.rolling(window=window).mean()

        # Avoid division by zero
        rolling_cv = rolling_std / rolling_mean.where(rolling_mean > 0, np.inf)

        # Average predictability (1 - average rolling CV)
        avg_rolling_cv = rolling_cv.mean()
        return max(0, 1 - avg_rolling_cv) if np.isfinite(avg_rolling_cv) else 0

    def calculate_protocol_safety_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate protocol safety metrics focused on actual solvency risks.

        Protocol safety concerns:
        1. Insolvency probability (critical vault balance)
        2. Deficit severity and duration (magnitude matters)
        3. Recovery resilience under stress
        """
        vault_balance = df['vault_balance'].values
        deficit = df['vault_deficit'].values

        # Insolvency Probability: P(vault_balance < critical_threshold)
        # Formula: fraction of time below critical threshold
        # Justification: Critical threshold represents actual protocol risk
        insolvency_events = vault_balance < self.critical_threshold
        insolvency_probability = np.mean(insolvency_events)

        # Deficit-Weighted Duration: Account for both magnitude and duration
        # Formula: ∑(deficit_magnitude × duration)^2 / total_time
        # Justification: Large deficits are exponentially more dangerous than small ones
        deficit_weighted_duration = self._calculate_deficit_weighted_duration(deficit)

        # Vault Stress Resilience: Recovery capability under pressure
        # Formula: Average recovery rate during deficit periods
        # Justification: Measures protocol's ability to self-correct
        vault_stress_resilience = self._calculate_vault_stress_resilience(deficit)

        # Maximum Continuous Underfunding: Longest period below target
        # Formula: Maximum consecutive periods with deficit > significant_threshold
        # Justification: Extended underfunding periods increase systemic risk
        max_continuous_underfunding = self._calculate_max_continuous_underfunding(deficit)

        return {
            'insolvency_probability': insolvency_probability,
            'deficit_weighted_duration': deficit_weighted_duration,
            'vault_stress_resilience': vault_stress_resilience,
            'max_continuous_underfunding': max_continuous_underfunding
        }

    def _calculate_deficit_weighted_duration(self, deficit: np.ndarray) -> float:
        """
        Calculate deficit-weighted duration that accounts for magnitude.

        Large deficits are exponentially more dangerous than small ones.
        """
        # Only consider significant deficits
        significant_deficits = np.maximum(0, deficit - self.significant_deficit_threshold)

        # Weight by magnitude squared (exponential penalty for large deficits)
        weighted_deficit_time = np.sum(significant_deficits ** 2)

        # Normalize by total time and target balance squared
        total_time = len(deficit)
        normalization_factor = total_time * (self.target_balance ** 2)

        return weighted_deficit_time / normalization_factor if normalization_factor > 0 else 0

    def _calculate_vault_stress_resilience(self, deficit: np.ndarray) -> float:
        """
        Calculate vault's ability to recover from deficit periods.

        Measures the rate of deficit reduction during stress periods.
        """
        if len(deficit) < 2:
            return 1.0

        # Find periods when vault is under stress (above significant deficit)
        stress_periods = deficit > self.significant_deficit_threshold

        if not np.any(stress_periods):
            return 1.0  # No stress periods = perfect resilience

        # Calculate deficit reduction rates during stress periods
        deficit_changes = np.diff(deficit)
        stress_period_changes = deficit_changes[stress_periods[:-1]]  # Align with diff array

        if len(stress_period_changes) == 0:
            return 1.0

        # Average deficit reduction rate (negative = good, positive = bad)
        avg_deficit_reduction = -np.mean(stress_period_changes)  # Negative because reduction is good

        # Normalize to [0, 1] scale where 1 = excellent resilience
        # Use sigmoid function to map reduction rates to [0, 1]
        resilience = 1 / (1 + np.exp(-avg_deficit_reduction * self.target_balance))

        return resilience

    def _calculate_max_continuous_underfunding(self, deficit: np.ndarray) -> float:
        """
        Calculate maximum continuous period below target balance.

        Extended underfunding increases systemic risk.
        """
        # Find periods above significant deficit threshold
        underfunded_periods = deficit > self.significant_deficit_threshold

        if not np.any(underfunded_periods):
            return 0.0

        # Find continuous runs of underfunding
        continuous_periods = []
        current_period = 0

        for is_underfunded in underfunded_periods:
            if is_underfunded:
                current_period += 1
            else:
                if current_period > 0:
                    continuous_periods.append(current_period)
                current_period = 0

        # Don't forget the last period if simulation ends during underfunding
        if current_period > 0:
            continuous_periods.append(current_period)

        max_period = max(continuous_periods) if continuous_periods else 0

        # Normalize by total simulation length
        return max_period / len(deficit)

    def calculate_economic_efficiency_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate economic efficiency metrics focused on capital utilization.

        Economic efficiency concerns:
        1. Vault utilization (avoid excess idle capital)
        2. Deficit correction speed (minimize time to recovery)
        3. Overall capital efficiency
        """
        vault_balance = df['vault_balance'].values
        deficit = df['vault_deficit'].values

        # Vault Utilization Efficiency: How close vault stays to optimal level
        # Formula: 1 - avg(|vault_balance - target_balance|) / target_balance
        # Justification: Excess capital is opportunity cost, insufficient capital is risk
        avg_deviation = np.mean(np.abs(vault_balance - self.target_balance))
        vault_utilization_efficiency = max(0, 1 - avg_deviation / self.target_balance)

        # Deficit Correction Rate: Speed of deficit resolution
        # Formula: Average rate of deficit reduction during deficit periods
        # Justification: Faster correction = more efficient mechanism
        deficit_correction_rate = self._calculate_deficit_correction_rate(deficit)

        # Capital Efficiency: Minimize wasted capital while maintaining safety
        # Formula: Target utilization adjusted for safety performance
        # Justification: Balance between capital efficiency and protocol safety
        capital_efficiency = self._calculate_capital_efficiency(vault_balance, deficit)

        return {
            'vault_utilization_efficiency': vault_utilization_efficiency,
            'deficit_correction_rate': deficit_correction_rate,
            'capital_efficiency': capital_efficiency
        }

    def _calculate_deficit_correction_rate(self, deficit: np.ndarray) -> float:
        """
        Calculate the rate at which deficits are corrected.

        Faster correction indicates more efficient mechanism.
        """
        if len(deficit) < 2:
            return 1.0

        # Find periods with significant deficit
        deficit_periods = deficit > self.significant_deficit_threshold

        if not np.any(deficit_periods):
            return 1.0  # No deficits to correct = perfect efficiency

        # Calculate deficit changes during deficit periods
        deficit_changes = np.diff(deficit)
        deficit_period_changes = deficit_changes[deficit_periods[:-1]]

        if len(deficit_period_changes) == 0:
            return 1.0

        # Average deficit reduction rate (normalized by target balance)
        avg_reduction_rate = -np.mean(deficit_period_changes) / self.target_balance

        # Convert to efficiency score [0, 1] where higher is better
        efficiency = min(1.0, max(0.0, avg_reduction_rate * 100))  # Scale factor for reasonable range

        return efficiency

    def _calculate_capital_efficiency(self, vault_balance: np.ndarray, deficit: np.ndarray) -> float:
        """
        Calculate overall capital efficiency balancing utilization and safety.
        """
        # Base efficiency from vault utilization
        avg_balance = np.mean(vault_balance)
        base_efficiency = min(1.0, avg_balance / self.target_balance)

        # Penalty for safety issues (time with significant deficit)
        significant_deficit_time = np.mean(deficit > self.significant_deficit_threshold)
        safety_penalty = significant_deficit_time * 0.5  # 50% penalty for safety issues

        # Penalty for excess capitalization
        excess_capital = max(0, avg_balance - self.target_balance * 1.5)  # 150% is considered excess
        excess_penalty = (excess_capital / self.target_balance) * 0.3  # 30% penalty rate

        efficiency = base_efficiency - safety_penalty - excess_penalty
        return max(0.0, min(1.0, efficiency))

    def calculate_composite_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate composite scores for multi-objective optimization.

        Weights based on protocol priorities:
        - User Experience: Focus on affordability and short-term predictability
        - Protocol Safety: Emphasize insolvency risk and deficit management
        - Economic Efficiency: Balance utilization with safety
        """

        # User Experience Score (equal weights for simplicity)
        ux_score = (
            0.4 * metrics['fee_affordability'] +
            0.3 * metrics['fee_stability'] +
            0.2 * metrics['fee_predictability_1h'] +
            0.1 * metrics['fee_predictability_6h']
        )

        # Protocol Safety Score (emphasize critical risks)
        safety_score = (
            0.4 * (1 - metrics['insolvency_probability']) +  # Higher insolvency = lower score
            0.3 * (1 - metrics['deficit_weighted_duration']) +  # Higher deficit duration = lower score
            0.2 * metrics['vault_stress_resilience'] +
            0.1 * (1 - metrics['max_continuous_underfunding'])  # Longer underfunding = lower score
        )

        # Economic Efficiency Score (balanced approach)
        efficiency_score = (
            0.4 * metrics['vault_utilization_efficiency'] +
            0.3 * metrics['deficit_correction_rate'] +
            0.3 * metrics['capital_efficiency']
        )

        return {
            'user_experience_score': ux_score,
            'protocol_safety_score': safety_score,
            'economic_efficiency_score': efficiency_score
        }

    def calculate_all_metrics(self, df: pd.DataFrame,
                            params: Dict[str, float] = None) -> RevisedMechanismMetrics:
        """
        Calculate all revised metrics.

        Args:
            df: Simulation results DataFrame
            params: Parameter dictionary (for constraint checking only)

        Returns:
            RevisedMechanismMetrics object
        """

        # Calculate all metric groups
        ux_metrics = self.calculate_user_experience_metrics(df)
        safety_metrics = self.calculate_protocol_safety_metrics(df)
        efficiency_metrics = self.calculate_economic_efficiency_metrics(df)

        # Combine all metrics
        all_metrics = {**ux_metrics, **safety_metrics, **efficiency_metrics}

        # Calculate composite scores
        composite_scores = self.calculate_composite_scores(all_metrics)
        all_metrics.update(composite_scores)

        return RevisedMechanismMetrics(**all_metrics)


# Constraint checking functions (not metrics)

def check_6step_alignment_constraint(H: int) -> bool:
    """
    Check if H satisfies 6-step batch cycle alignment constraint.

    Args:
        H: Horizon parameter

    Returns:
        True if H is multiple of 6 (aligned with batch cycles)
    """
    return H % 6 == 0


def check_basic_safety_constraints(metrics: RevisedMechanismMetrics) -> Dict[str, bool]:
    """
    Check basic safety constraints for production deployment.

    Args:
        metrics: Calculated metrics

    Returns:
        Dictionary of constraint_name -> is_satisfied
    """
    # Much more relaxed constraints based on feedback
    constraints = {
        'low_insolvency_risk': metrics.insolvency_probability < 0.20,  # Max 20% insolvency risk (relaxed)
        'acceptable_deficit_duration': metrics.deficit_weighted_duration < 0.5,  # More lenient deficit levels
        'min_vault_resilience': metrics.vault_stress_resilience > 0.1,  # Lower minimum recovery capability
        'reasonable_fee_affordability': metrics.fee_affordability > -15,  # More lenient fee levels
        'min_fee_stability': metrics.fee_stability > 0.1  # Lower minimum stability requirement
    }

    return constraints


def is_configuration_feasible(metrics: RevisedMechanismMetrics, H: int) -> bool:
    """
    Check if a configuration meets all constraints.

    Args:
        metrics: Calculated metrics
        H: Horizon parameter for 6-step constraint

    Returns:
        True if configuration is feasible for deployment
    """
    # Check 6-step alignment constraint
    if not check_6step_alignment_constraint(H):
        return False

    # Check safety constraints
    safety_constraints = check_basic_safety_constraints(metrics)
    return all(safety_constraints.values())