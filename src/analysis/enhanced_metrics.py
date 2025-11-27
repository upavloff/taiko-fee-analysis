"""
Enhanced Metrics Framework for Taiko Fee Mechanism Optimization

This module implements a mathematically rigorous metrics framework designed for
multi-objective optimization of the Taiko fee mechanism. Each metric is carefully
defined with clear justification for its role in protocol and user experience optimization.

Key improvements over the original framework:
1. Mathematically justified objective functions
2. Rigorous definitions with clear thresholds
3. Production-ready constraints (governance lag, implementation costs)
4. Adversarial robustness metrics
5. 6-step batch cycle alignment analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class EnhancedMechanismMetrics:
    """
    Comprehensive metrics container with mathematical rigor.

    Each metric includes:
    - Clear mathematical definition
    - Justification for inclusion
    - Normalization for multi-objective optimization
    """

    # === PRIMARY: User Experience Metrics ===
    # These directly impact end users and should be heavily weighted

    fee_affordability_score: float      # log(1 + avg_fee_eth × 1000) - penalizes high fees heavily
    fee_predictability_score: float     # 1 - coefficient_of_variation - rewards stable fees
    fee_responsiveness_score: float     # 1 / (1 + fee_change_lag) - rewards fast L1 tracking
    fee_rate_of_change_p95: float       # 95th percentile of |fee_t+1 - fee_t| / fee_t
    user_cost_burden: float             # total_user_fees / total_protocol_value

    # === SECONDARY: Protocol Stability Metrics ===
    # Critical for protocol security and robustness

    vault_robustness_score: float       # 1 - P(deficit > 0.5×target_balance)
    crisis_resilience_score: float      # 1 - max_deficit_duration / simulation_length
    capital_efficiency_score: float     # avg_vault_utilization (penalize over-capitalization)
    vault_insolvency_risk: float        # P(balance < 0.1×target_balance)
    l1_spike_response_time: float       # Steps to reach 90% of equilibrium after L1 shock

    # === TERTIARY: Economic Efficiency Metrics ===
    # System-level optimization for long-term sustainability

    cost_recovery_ratio: float          # min(1, total_fees_collected / total_l1_costs)
    mechanism_overhead_score: float     # 1 - (computation_cost / total_fees)
    sixstep_cycle_alignment: float      # How well H aligns with 6-step batch cycles
    deficit_correction_efficiency: float # Time to correct large deficits (improved calculation)

    # === QUATERNARY: Production Considerations ===
    # Real-world deployment constraints

    governance_stability_score: float   # Parameter change frequency (implementation cost)
    implementation_complexity: float    # Simplicity score for audit/explanation
    backwards_compatibility_score: float # Smooth transition from current parameters

    # === ADVERSARIAL ROBUSTNESS METRICS ===
    # Resistance to attacks and extreme conditions

    mev_attack_resistance: float        # Resistance to coordinated fee manipulation
    extreme_volatility_survival: float  # Performance under 100x L1 fee spikes
    demand_shock_resilience: float      # Performance under 10x volume changes

    # === COMPOSITE OBJECTIVE SCORES ===
    # Weighted combinations for multi-objective optimization

    user_experience_composite: float    # Primary optimization target
    protocol_stability_composite: float # Secondary optimization target
    economic_efficiency_composite: float # Tertiary optimization target
    production_readiness_composite: float # Deployment feasibility
    overall_optimization_score: float   # Grand unified score

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for DataFrame compatibility."""
        return {k: v for k, v in self.__dict__.items()}

    def get_pareto_objectives(self) -> Tuple[float, float, float]:
        """Return the three main objectives for Pareto optimization."""
        return (
            -self.user_experience_composite,      # Minimize (negative for minimization)
            -self.protocol_stability_composite,   # Minimize (negative for minimization)
            -self.economic_efficiency_composite   # Minimize (negative for minimization)
        )


class EnhancedMetricsCalculator:
    """
    Enhanced metrics calculator with rigorous mathematical definitions.

    Each calculation includes:
    - Detailed docstring with mathematical formula
    - Justification for the metric's importance
    - Edge case handling
    - Normalization for optimization
    """

    def __init__(self, target_balance: float,
                 taiko_block_time: float = 2.0,
                 eth_block_time: float = 12.0):
        """
        Initialize with protocol-specific parameters.

        Args:
            target_balance: Target vault balance (ETH)
            taiko_block_time: Taiko L2 block time (seconds)
            eth_block_time: Ethereum L1 block time (seconds)
        """
        self.target_balance = target_balance
        self.taiko_block_time = taiko_block_time
        self.eth_block_time = eth_block_time
        self.batch_cycle_steps = int(eth_block_time / taiko_block_time)  # 6 steps

        # Thresholds for risk assessment (more aggressive than original)
        self.critical_deficit_threshold = 0.5 * target_balance  # 50% deficit is critical
        self.insolvency_threshold = 0.1 * target_balance        # 10% deficit is insolvency risk
        self.extreme_fee_threshold = 0.01                        # 0.01 ETH fee is extreme

    def calculate_user_experience_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate user experience metrics with mathematical rigor.

        These metrics directly impact user adoption and protocol competitiveness.
        """
        fees = df['estimated_fee'].values

        # Fee Affordability: Heavily penalize high fees using log scale
        # Formula: -log(1 + avg_fee_eth × 1000)
        # Justification: Users are exponentially sensitive to fee increases
        avg_fee = np.mean(fees)
        fee_affordability = -np.log(1 + avg_fee * 1000) if avg_fee > 0 else 1.0

        # Fee Predictability: Reward stable fees
        # Formula: 1 - coefficient_of_variation
        # Justification: Unpredictable fees create terrible UX for dApps/users
        fee_cv = np.std(fees) / np.mean(fees) if np.mean(fees) > 0 else 0
        fee_predictability = max(0, 1 - fee_cv)

        # Fee Responsiveness: How quickly fees respond to L1 changes
        # Formula: 1 / (1 + response_lag_steps)
        # Justification: Slow response creates arbitrage opportunities and poor UX
        response_lag = self._calculate_response_lag(
            fees, df['estimated_l1_cost'].values
        )
        fee_responsiveness = 1 / (1 + response_lag)

        # Fee Rate of Change: 95th percentile of period-over-period changes
        # Formula: percentile_95(|fee_t+1 - fee_t| / fee_t)
        # Justification: Large fee jumps break user mental models and dApp UX
        if len(fees) > 1:
            fee_changes = np.abs(np.diff(fees)) / fees[:-1]
            fee_changes = fee_changes[np.isfinite(fee_changes)]  # Remove inf/nan
            fee_rate_change_p95 = np.percentile(fee_changes, 95) if len(fee_changes) > 0 else 0
        else:
            fee_rate_change_p95 = 0

        # User Cost Burden: What fraction of transaction value goes to fees
        # Formula: total_fees / total_transaction_value
        # Justification: High cost burden makes the protocol uncompetitive
        total_fees = df['fee_collected'].sum()
        total_tx_volume = df['transaction_volume'].sum()
        # Estimate transaction value (assume avg tx value = 100 ETH for scaling)
        estimated_tx_value = total_tx_volume * 0.1  # Conservative estimate
        user_cost_burden = total_fees / estimated_tx_value if estimated_tx_value > 0 else 1.0

        return {
            'fee_affordability_score': fee_affordability,
            'fee_predictability_score': fee_predictability,
            'fee_responsiveness_score': fee_responsiveness,
            'fee_rate_of_change_p95': fee_rate_change_p95,
            'user_cost_burden': user_cost_burden
        }

    def calculate_protocol_stability_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate protocol stability metrics for security and robustness.

        These metrics ensure the protocol can survive extreme market conditions.
        """
        vault_balance = df['vault_balance'].values
        deficit = df['vault_deficit'].values

        # Vault Robustness: Probability of avoiding critical deficits
        # Formula: 1 - P(deficit > 0.5×target_balance)
        # Justification: Large deficits threaten protocol solvency
        critical_deficit_events = np.sum(deficit > self.critical_deficit_threshold)
        vault_robustness = 1 - (critical_deficit_events / len(deficit))

        # Crisis Resilience: How quickly the protocol recovers from crises
        # Formula: 1 - max_continuous_deficit_duration / total_simulation_time
        # Justification: Extended crises can cause bank runs and protocol abandonment
        max_deficit_duration = self._calculate_max_deficit_duration(deficit)
        crisis_resilience = 1 - (max_deficit_duration / len(deficit))

        # Capital Efficiency: Avoid over-capitalization while maintaining safety
        # Formula: avg_vault_utilization (normalized around target)
        # Justification: Excess capital is opportunity cost for the protocol
        avg_utilization = np.mean(vault_balance) / self.target_balance
        capital_efficiency = 1 - abs(avg_utilization - 1) if avg_utilization > 0 else 0

        # Vault Insolvency Risk: Probability of approaching insolvency
        # Formula: P(balance < 0.1×target_balance)
        # Justification: Near-insolvency events damage protocol reputation
        insolvency_events = np.sum(vault_balance < self.insolvency_threshold)
        vault_insolvency_risk = insolvency_events / len(vault_balance)

        # L1 Spike Response Time: How quickly fees adapt to L1 changes
        # Formula: Steps to reach 90% of new equilibrium after shock
        # Justification: Slow response creates MEV opportunities
        l1_spike_response_time = self._calculate_l1_response_time(df)

        return {
            'vault_robustness_score': vault_robustness,
            'crisis_resilience_score': crisis_resilience,
            'capital_efficiency_score': capital_efficiency,
            'vault_insolvency_risk': vault_insolvency_risk,
            'l1_spike_response_time': l1_spike_response_time
        }

    def calculate_economic_efficiency_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate economic efficiency metrics for long-term sustainability.

        These metrics ensure the mechanism is economically sound and efficient.
        """
        total_fees = df['fee_collected'].sum()
        total_l1_costs = df['l1_cost_paid'].sum()

        # Cost Recovery Ratio: How well fees cover L1 costs
        # Formula: min(1, total_fees_collected / total_l1_costs)
        # Justification: Insufficient cost coverage threatens long-term viability
        cost_recovery_ratio = min(1.0, total_fees / total_l1_costs) if total_l1_costs > 0 else 1.0

        # Mechanism Overhead: Computational/gas costs relative to fees
        # Formula: 1 - (estimated_mechanism_cost / total_fees)
        # Justification: High overhead reduces capital efficiency
        # Note: Simplified estimate - in production would use actual gas costs
        estimated_mechanism_cost = len(df) * 0.0001  # Rough estimate per step
        mechanism_overhead = 1 - (estimated_mechanism_cost / total_fees) if total_fees > 0 else 0

        return {
            'cost_recovery_ratio': cost_recovery_ratio,
            'mechanism_overhead_score': max(0, mechanism_overhead)  # Ensure non-negative
        }

    def calculate_batch_cycle_metrics(self, df: pd.DataFrame, H: int) -> Dict[str, float]:
        """
        Calculate metrics specific to the 6-step batch cycle alignment.

        Taiko's 6-step batch cycles create natural resonant frequencies in the mechanism.
        """
        # 6-Step Cycle Alignment: How well H aligns with batch cycles
        # Formula: Alignment score based on H % 6
        # Justification: Misaligned horizons create suboptimal deficit correction
        if H % self.batch_cycle_steps == 0:
            sixstep_alignment = 1.0  # Perfect alignment
        else:
            remainder = H % self.batch_cycle_steps
            # Score decreases as we get further from alignment
            sixstep_alignment = 1 - (min(remainder, self.batch_cycle_steps - remainder) / self.batch_cycle_steps)

        # Enhanced Deficit Correction Efficiency
        # Formula: Weighted by batch cycle alignment
        # Justification: Batch-aligned correction is more efficient
        deficit = df['vault_deficit'].values
        base_correction_efficiency = self._calculate_deficit_correction_efficiency(deficit)
        aligned_correction_efficiency = base_correction_efficiency * (0.5 + 0.5 * sixstep_alignment)

        return {
            'sixstep_cycle_alignment': sixstep_alignment,
            'deficit_correction_efficiency': aligned_correction_efficiency
        }

    def calculate_production_readiness_metrics(self, params: Dict) -> Dict[str, float]:
        """
        Calculate production deployment readiness metrics.

        These metrics assess real-world implementation feasibility.
        """
        mu, nu, H = params.get('mu', 0), params.get('nu', 0), params.get('H', 144)

        # Governance Stability: How often would optimal parameters change?
        # Formula: Based on parameter stability across scenarios
        # Justification: Frequent changes increase governance overhead
        # Note: This would be calculated across multiple scenarios in practice
        governance_stability = 0.8  # Placeholder - would calculate from scenario variance

        # Implementation Complexity: How easy to understand/audit?
        # Formula: Simple parameters get higher scores
        # Justification: Complex parameters increase security risks and governance overhead
        complexity_penalty = 0
        if mu == 0:
            complexity_penalty += 0.2  # μ=0 is conceptually simple
        if nu in [0.1, 0.2, 0.5, 0.7]:  # Round numbers
            complexity_penalty += 0.2
        if H % self.batch_cycle_steps == 0:  # Batch-aligned
            complexity_penalty += 0.3
        if H in [36, 72, 144, 288]:  # Common horizons
            complexity_penalty += 0.3

        implementation_complexity = min(1.0, complexity_penalty)

        # Backwards Compatibility: How smooth is transition from current params?
        # Formula: Distance-based score from current parameters
        # Justification: Radical changes increase migration risks
        # Current params (from POST_TIMING_FIX): μ=0.0, ν=0.1, H=36
        current_params = {'mu': 0.0, 'nu': 0.1, 'H': 36}

        mu_distance = abs(mu - current_params['mu'])
        nu_distance = abs(nu - current_params['nu'])
        h_distance = abs(H - current_params['H']) / current_params['H']  # Relative distance

        total_distance = mu_distance + nu_distance + h_distance
        backwards_compatibility = max(0, 1 - total_distance / 3)  # Normalize

        return {
            'governance_stability_score': governance_stability,
            'implementation_complexity': implementation_complexity,
            'backwards_compatibility_score': backwards_compatibility
        }

    def calculate_adversarial_robustness_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate adversarial robustness metrics for security assessment.

        These metrics test resistance to various attack vectors.
        """
        fees = df['estimated_fee'].values
        vault_balance = df['vault_balance'].values

        # MEV Attack Resistance: How hard is it to manipulate fees for profit?
        # Formula: Variance in fee response to volume changes
        # Justification: Predictable fee responses can be exploited
        fee_variance = np.var(fees)
        # Higher variance = harder to predict = better resistance
        mev_resistance = min(1.0, fee_variance * 1000)  # Scale factor

        # Extreme Volatility Survival: Performance during 100x L1 spikes
        # Formula: Vault survival during simulated extreme events
        # Justification: Real-world events can exceed historical data
        # Note: Would be calculated from synthetic stress tests in practice
        extreme_volatility_survival = 0.7  # Placeholder - needs stress testing

        # Demand Shock Resilience: Performance under 10x volume changes
        # Formula: Fee stability during volume shocks
        # Justification: Viral dApps can create sudden demand spikes
        volume = df['transaction_volume'].values
        if len(volume) > 1:
            volume_changes = np.abs(np.diff(volume)) / volume[:-1]
            volume_changes = volume_changes[np.isfinite(volume_changes)]
            max_volume_change = np.max(volume_changes) if len(volume_changes) > 0 else 0
            demand_shock_resilience = max(0, 1 - max_volume_change / 10)  # Scale by 10x
        else:
            demand_shock_resilience = 1.0

        return {
            'mev_attack_resistance': mev_resistance,
            'extreme_volatility_survival': extreme_volatility_survival,
            'demand_shock_resilience': demand_shock_resilience
        }

    def calculate_composite_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate weighted composite scores for multi-objective optimization.

        Weights are based on protocol priorities and stakeholder impact analysis.
        """
        # User Experience Composite (40% weight in overall score)
        # Heavy focus on affordability and predictability
        ux_composite = (
            0.4 * metrics['fee_affordability_score'] +
            0.3 * metrics['fee_predictability_score'] +
            0.2 * metrics['fee_responsiveness_score'] +
            0.1 * (1 - metrics['fee_rate_of_change_p95'])  # Lower is better
        )

        # Protocol Stability Composite (35% weight in overall score)
        # Focus on robustness and crisis resilience
        stability_composite = (
            0.3 * metrics['vault_robustness_score'] +
            0.3 * metrics['crisis_resilience_score'] +
            0.2 * metrics['capital_efficiency_score'] +
            0.2 * (1 - metrics['vault_insolvency_risk'])  # Lower risk is better
        )

        # Economic Efficiency Composite (15% weight in overall score)
        efficiency_composite = (
            0.4 * metrics['cost_recovery_ratio'] +
            0.3 * metrics['mechanism_overhead_score'] +
            0.3 * metrics['deficit_correction_efficiency']
        )

        # Production Readiness Composite (10% weight in overall score)
        production_composite = (
            0.4 * metrics['implementation_complexity'] +
            0.3 * metrics['backwards_compatibility_score'] +
            0.3 * metrics['governance_stability_score']
        )

        # Overall Optimization Score
        overall_score = (
            0.40 * ux_composite +
            0.35 * stability_composite +
            0.15 * efficiency_composite +
            0.10 * production_composite
        )

        return {
            'user_experience_composite': ux_composite,
            'protocol_stability_composite': stability_composite,
            'economic_efficiency_composite': efficiency_composite,
            'production_readiness_composite': production_composite,
            'overall_optimization_score': overall_score
        }

    def calculate_all_metrics(self, df: pd.DataFrame,
                            params: Dict[str, Union[float, int]] = None) -> EnhancedMechanismMetrics:
        """
        Calculate comprehensive enhanced metrics suite.

        Args:
            df: Simulation results DataFrame
            params: Parameter dictionary with mu, nu, H values

        Returns:
            EnhancedMechanismMetrics object with all calculated metrics
        """
        if params is None:
            params = {'mu': 0.0, 'nu': 0.1, 'H': 36}  # Default values

        # Calculate all metric groups
        ux_metrics = self.calculate_user_experience_metrics(df)
        stability_metrics = self.calculate_protocol_stability_metrics(df)
        efficiency_metrics = self.calculate_economic_efficiency_metrics(df)
        batch_metrics = self.calculate_batch_cycle_metrics(df, params.get('H', 36))
        production_metrics = self.calculate_production_readiness_metrics(params)
        robustness_metrics = self.calculate_adversarial_robustness_metrics(df)

        # Combine all metrics
        all_metrics = {
            **ux_metrics,
            **stability_metrics,
            **efficiency_metrics,
            **batch_metrics,
            **production_metrics,
            **robustness_metrics
        }

        # Calculate composite scores
        composite_scores = self.calculate_composite_scores(all_metrics)
        all_metrics.update(composite_scores)

        return EnhancedMechanismMetrics(**all_metrics)

    # === Helper Methods ===

    def _calculate_response_lag(self, fees: np.ndarray, l1_costs: np.ndarray) -> float:
        """Calculate response lag using cross-correlation analysis."""
        if len(fees) < 10:
            return 0

        # Normalize series for correlation analysis
        fees_norm = (fees - np.mean(fees)) / np.std(fees) if np.std(fees) > 0 else fees
        l1_norm = (l1_costs - np.mean(l1_costs)) / np.std(l1_costs) if np.std(l1_costs) > 0 else l1_costs

        # Cross-correlation with limited lag range
        max_lag = min(20, len(fees) // 4)
        correlations = []

        for lag in range(max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(fees_norm, l1_norm)[0, 1]
            else:
                corr = np.corrcoef(fees_norm[lag:], l1_norm[:-lag])[0, 1]

            correlations.append(abs(corr) if not np.isnan(corr) else 0)

        # Return lag with highest correlation
        return np.argmax(correlations) if correlations else 0

    def _calculate_max_deficit_duration(self, deficit: np.ndarray) -> int:
        """Calculate maximum continuous duration above critical deficit."""
        above_threshold = deficit > self.critical_deficit_threshold

        if not np.any(above_threshold):
            return 0

        # Find continuous runs above threshold
        durations = []
        current_duration = 0

        for is_above in above_threshold:
            if is_above:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return max(durations) if durations else 0

    def _calculate_l1_response_time(self, df: pd.DataFrame) -> float:
        """Calculate time to reach 90% of equilibrium after L1 shock."""
        l1_costs = df['estimated_l1_cost'].values
        fees = df['estimated_fee'].values

        if len(l1_costs) < 20:
            return 10  # Default moderate response time

        # Detect significant L1 cost changes (>50% change)
        l1_changes = np.abs(np.diff(l1_costs)) / l1_costs[:-1]
        shock_indices = np.where(l1_changes > 0.5)[0]

        if len(shock_indices) == 0:
            return 5  # No shocks detected, assume good responsiveness

        response_times = []
        for shock_idx in shock_indices[:3]:  # Analyze first 3 shocks
            if shock_idx + 20 < len(fees):  # Ensure enough data after shock
                pre_shock_fee = fees[shock_idx]
                post_shock_window = fees[shock_idx+1:shock_idx+21]

                # Find when fee reaches 90% of final adjustment
                target_fee = post_shock_window[-1]  # Assume final value is equilibrium
                adjustment_size = abs(target_fee - pre_shock_fee)

                if adjustment_size > 0:
                    threshold = pre_shock_fee + 0.9 * (target_fee - pre_shock_fee)

                    # Find first time fee crosses 90% threshold
                    for i, fee in enumerate(post_shock_window):
                        if abs(fee - threshold) < 0.1 * adjustment_size:
                            response_times.append(i + 1)
                            break
                    else:
                        response_times.append(20)  # Never reached threshold

        return np.mean(response_times) if response_times else 10

    def _calculate_deficit_correction_efficiency(self, deficit: np.ndarray) -> float:
        """Calculate deficit correction efficiency with enhanced methodology."""
        if len(deficit) < 10:
            return 1.0

        # Find periods of significant deficit (>25% of target)
        significant_deficit_threshold = 0.25 * self.target_balance
        significant_deficit_periods = deficit > significant_deficit_threshold

        if not np.any(significant_deficit_periods):
            return 1.0  # No significant deficits to correct

        # Calculate correction rates during deficit periods
        correction_rates = []
        in_deficit = False
        deficit_start = 0
        max_deficit_in_period = 0

        for t, is_deficit in enumerate(significant_deficit_periods):
            if is_deficit and not in_deficit:
                # Start of deficit period
                deficit_start = t
                max_deficit_in_period = deficit[t]
                in_deficit = True
            elif is_deficit and in_deficit:
                # Continue deficit period
                max_deficit_in_period = max(max_deficit_in_period, deficit[t])
            elif not is_deficit and in_deficit:
                # End of deficit period - calculate correction efficiency
                period_length = t - deficit_start
                deficit_reduced = max_deficit_in_period - deficit[t-1] if t > 0 else max_deficit_in_period

                if period_length > 0 and max_deficit_in_period > 0:
                    correction_rate = deficit_reduced / (period_length * max_deficit_in_period)
                    correction_rates.append(correction_rate)

                in_deficit = False

        # Handle case where deficit period extends to end of simulation
        if in_deficit and len(deficit) > deficit_start:
            period_length = len(deficit) - deficit_start
            deficit_reduced = max_deficit_in_period - deficit[-1]
            if period_length > 0 and max_deficit_in_period > 0:
                correction_rate = deficit_reduced / (period_length * max_deficit_in_period)
                correction_rates.append(correction_rate)

        if correction_rates:
            avg_correction_rate = np.mean(correction_rates)
            # Convert to 0-1 score (higher correction rate = better efficiency)
            return min(1.0, avg_correction_rate * 10)  # Scale factor
        else:
            return 0.5  # Some deficits exist but never corrected