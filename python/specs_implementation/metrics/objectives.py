"""
Soft Objectives Implementation

Implements SPECS.md Section 7: Multi-Objective Calculation

Mathematical Formulas:

7.1 UX Objective:
J_UX(θ) = w₁F̄ + w₂CV_F + w₃J₀.₉₅ + w₄CV₁ₕ + w₅CV₆ₕ

7.2 Robustness Objective:
J_Robust(θ) = u₁DWD + u₂L_max

7.3 Capital Efficiency Objective:
J_CapEff(θ) = V̄/Q̄
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .constraints import SimulationResults


class StakeholderProfile(Enum):
    """Different stakeholder optimization profiles"""
    PROTOCOL_LAUNCH = "protocol_launch"     # Safety-first for new protocol
    USER_CENTRIC = "user_centric"           # Low fees, good UX
    OPERATOR_FOCUSED = "operator_focused"   # Capital efficiency focus
    BALANCED = "balanced"                   # Balanced across all objectives
    STRESS_TESTED = "stress_tested"         # Optimized for crisis scenarios


@dataclass
class ObjectiveWeights:
    """Weight parameters for multi-objective calculation - expressing IMPORTANCE, not scaling"""
    # UX objective weights (Section 7.1) - Importance weights for normalized [0,1] metrics
    w1_avg_fee: float = 0.4         # Average fee level importance
    w2_cv_global: float = 0.3       # Global coefficient of variation importance
    w3_jump_p95: float = 0.2        # 95th percentile relative jump importance
    w4_cv_1h: float = 0.05         # Rolling 1h CV importance (reduced correlation)
    w5_cv_6h: float = 0.05         # Rolling 6h CV importance (reduced correlation)

    # Robustness objective weights (Section 7.2) - Importance weights for normalized [0,1] metrics
    u1_dwd: float = 0.7             # Deficit-weighted duration importance
    u2_l_max: float = 0.3           # Max underfunding streak importance


@dataclass
class ObjectiveResults:
    """Results of objective calculation"""
    # Individual UX metrics
    avg_fee_gwei: float
    cv_global: float
    jump_p95: float
    cv_1h: float
    cv_6h: float

    # Individual robustness metrics
    deficit_weighted_duration: float
    max_underfunding_streak: int

    # Individual capital efficiency metric
    capital_efficiency_eth_per_gas: float

    # Aggregated objectives
    ux_objective: float
    robustness_objective: float
    capital_efficiency_objective: float


class ObjectiveCalculator:
    """SPECS.md Section 7: Multi-Objective Calculation"""

    def __init__(self, weights: Optional[ObjectiveWeights] = None, stakeholder_profile: Optional[StakeholderProfile] = None):
        """
        Initialize objective calculator with stakeholder-specific weights

        Args:
            weights: Objective weight parameters (overrides stakeholder_profile if provided)
            stakeholder_profile: Predefined stakeholder optimization profile
        """
        if weights is not None:
            self.weights = weights
        elif stakeholder_profile is not None:
            self.weights = self.get_stakeholder_weights(stakeholder_profile)
        else:
            self.weights = ObjectiveWeights()  # Use normalized defaults

    def calculate_all_objectives(
        self,
        simulation_results: SimulationResults,
        V_min: float = 0.0
    ) -> ObjectiveResults:
        """
        Calculate all objectives for simulation results

        Args:
            simulation_results: Complete simulation trajectory
            V_min: Minimum vault threshold for robustness calculation

        Returns:
            ObjectiveResults with all individual and aggregated metrics
        """
        # Calculate individual UX metrics
        avg_fee_gwei = self._calculate_average_fee_gwei(simulation_results)
        cv_global = self._calculate_global_cv(simulation_results)
        jump_p95 = self._calculate_jump_p95(simulation_results)
        cv_1h = self._calculate_rolling_cv(simulation_results, window_hours=1)
        cv_6h = self._calculate_rolling_cv(simulation_results, window_hours=6)

        # Calculate individual robustness metrics
        dwd = self._calculate_deficit_weighted_duration(simulation_results, V_min)
        l_max = self._calculate_max_underfunding_streak(simulation_results, V_min)

        # Calculate capital efficiency metric
        cap_eff = self._calculate_capital_efficiency(simulation_results)

        # Calculate aggregated objectives
        ux_obj = self.calculate_ux_objective(
            avg_fee_gwei, cv_global, jump_p95, cv_1h, cv_6h
        )
        robust_obj = self.calculate_robustness_objective(
            dwd, l_max, len(simulation_results.vault_balances)
        )
        cap_eff_obj = self.calculate_capital_efficiency_objective(simulation_results)

        return ObjectiveResults(
            avg_fee_gwei=avg_fee_gwei,
            cv_global=cv_global,
            jump_p95=jump_p95,
            cv_1h=cv_1h,
            cv_6h=cv_6h,
            deficit_weighted_duration=dwd,
            max_underfunding_streak=l_max,
            capital_efficiency_eth_per_gas=cap_eff,
            ux_objective=ux_obj,
            robustness_objective=robust_obj,
            capital_efficiency_objective=cap_eff_obj
        )

    def calculate_ux_objective(
        self,
        avg_fee_gwei: float,
        cv_global: float,
        jump_p95: float,
        cv_1h: float,
        cv_6h: float
    ) -> float:
        """
        Calculate UX objective with proper normalization

        Each metric is first normalized to [0,1] where 1.0 = better,
        then importance weights are applied.

        Args:
            avg_fee_gwei: Average fee level in gwei
            cv_global: Global coefficient of variation
            jump_p95: 95th percentile relative jump
            cv_1h: Rolling 1h CV
            cv_6h: Rolling 6h CV

        Returns:
            Weighted UX objective (higher is better for maximization)
        """
        # Normalize all components to [0,1] before weighting
        norm_fee = self._normalize_average_fee(avg_fee_gwei)
        norm_cv_global = self._normalize_cv(cv_global)
        norm_jump = self._normalize_jump_p95(jump_p95)
        norm_cv_1h = self._normalize_cv(cv_1h)
        norm_cv_6h = self._normalize_cv(cv_6h)

        # Apply importance weights to normalized metrics
        ux_objective = (
            self.weights.w1_avg_fee * norm_fee +
            self.weights.w2_cv_global * norm_cv_global +
            self.weights.w3_jump_p95 * norm_jump +
            self.weights.w4_cv_1h * norm_cv_1h +
            self.weights.w5_cv_6h * norm_cv_6h
        )
        return ux_objective

    def calculate_robustness_objective(
        self,
        deficit_weighted_duration: float,
        max_underfunding_streak: int,
        simulation_length: int,
        target_balance: float = 1.0
    ) -> float:
        """
        Calculate robustness objective with proper normalization

        Each metric is first normalized to [0,1] where 1.0 = better,
        then importance weights are applied.

        Args:
            deficit_weighted_duration: Severity of underfunding
            max_underfunding_streak: Max consecutive underfunded batches
            simulation_length: Length of simulation for normalization
            target_balance: Target vault balance for normalization

        Returns:
            Weighted robustness objective (higher is better for maximization)
        """
        # Normalize all components to [0,1] before weighting
        norm_dwd = self._normalize_deficit_weighted_duration(
            deficit_weighted_duration, simulation_length, target_balance
        )
        norm_streak = self._normalize_max_underfunding_streak(
            max_underfunding_streak, simulation_length
        )

        # Apply importance weights to normalized metrics
        robustness_objective = (
            self.weights.u1_dwd * norm_dwd +
            self.weights.u2_l_max * norm_streak
        )
        return robustness_objective

    def calculate_capital_efficiency_objective(
        self,
        simulation_results: SimulationResults
    ) -> float:
        """
        Calculate capital efficiency objective with proper normalization

        Capital efficiency is first calculated as V̄/Q̄, then normalized to [0,1]
        where 1.0 = better (more capital efficient)

        Args:
            simulation_results: Complete simulation trajectory

        Returns:
            Normalized capital efficiency objective (higher is better)
        """
        # Calculate raw capital efficiency
        raw_cap_eff = self._calculate_capital_efficiency(simulation_results)

        # Normalize to [0,1] where lower capital per gas is better
        normalized_cap_eff = self._normalize_capital_efficiency(raw_cap_eff)

        return normalized_cap_eff

    def get_pareto_objectives(
        self,
        simulation_results: SimulationResults,
        V_min: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Calculate three separate normalized objectives for NSGA-II Pareto optimization

        This method returns the three primary objectives as separate values for
        true multi-objective optimization, rather than combining them with weights.
        All objectives are normalized to [0,1] where 1.0 = better performance.

        Args:
            simulation_results: Complete simulation trajectory
            V_min: Minimum vault threshold for robustness calculation

        Returns:
            Tuple of (ux_objective, robustness_objective, capital_efficiency_objective)
            Each objective is normalized to [0,1] for fair Pareto comparison
        """
        # Calculate individual UX metrics
        avg_fee_gwei = self._calculate_average_fee_gwei(simulation_results)
        cv_global = self._calculate_global_cv(simulation_results)
        jump_p95 = self._calculate_jump_p95(simulation_results)
        cv_1h = self._calculate_rolling_cv(simulation_results, window_hours=1)
        cv_6h = self._calculate_rolling_cv(simulation_results, window_hours=6)

        # Calculate individual robustness metrics
        dwd = self._calculate_deficit_weighted_duration(simulation_results, V_min)
        l_max = self._calculate_max_underfunding_streak(simulation_results, V_min)

        # Calculate individual capital efficiency metric
        raw_cap_eff = self._calculate_capital_efficiency(simulation_results)

        # Calculate the three Pareto objectives (all normalized)
        ux_objective = self.calculate_ux_objective(
            avg_fee_gwei, cv_global, jump_p95, cv_1h, cv_6h
        )
        robustness_objective = self.calculate_robustness_objective(
            dwd, l_max, len(simulation_results.vault_balances)
        )
        capital_efficiency_objective = self._normalize_capital_efficiency(raw_cap_eff)

        return (ux_objective, robustness_objective, capital_efficiency_objective)

    def _calculate_average_fee_gwei(
        self,
        simulation_results: SimulationResults
    ) -> float:
        """Calculate average fee level in gwei"""
        fees_wei = simulation_results.fees_per_gas
        avg_fee_wei = np.mean(fees_wei)
        return avg_fee_wei / 1e9  # Convert to gwei

    @staticmethod
    def get_stakeholder_weights(profile: StakeholderProfile) -> ObjectiveWeights:
        """
        Get predefined weight configurations for different stakeholder profiles.

        Args:
            profile: Stakeholder optimization profile

        Returns:
            ObjectiveWeights configured for the specified stakeholder
        """
        if profile == StakeholderProfile.PROTOCOL_LAUNCH:
            # Safety-first for protocol launch phase
            return ObjectiveWeights(
                w1_avg_fee=0.2,         # Low fee importance (safety first)
                w2_cv_global=0.4,       # Stability very important
                w3_jump_p95=0.3,        # NO big jumps during launch
                w4_cv_1h=0.05,         # Reduced correlation noise
                w5_cv_6h=0.05,         # Minimal
                u1_dwd=0.8,            # HIGH safety priority
                u2_l_max=0.2           # Backup safety metric
            )
        elif profile == StakeholderProfile.USER_CENTRIC:
            # Optimize for end user experience
            return ObjectiveWeights(
                w1_avg_fee=0.6,         # STRONG focus on low fees
                w2_cv_global=0.25,      # Predictability matters
                w3_jump_p95=0.1,        # Some fee spikes acceptable for low avg fees
                w4_cv_1h=0.025,        # Reduced correlation
                w5_cv_6h=0.025,        # Minimal
                u1_dwd=0.6,            # Basic safety only
                u2_l_max=0.4           # Moderate safety
            )
        elif profile == StakeholderProfile.OPERATOR_FOCUSED:
            # Capital efficiency and business metrics
            return ObjectiveWeights(
                w1_avg_fee=0.3,         # Moderate fee concern
                w2_cv_global=0.2,       # Some variability OK
                w3_jump_p95=0.4,        # Business needs predictability
                w4_cv_1h=0.05,         # Low priority
                w5_cv_6h=0.05,         # Minimal
                u1_dwd=0.5,            # Reasonable safety
                u2_l_max=0.5           # Business continuity important
            )
        elif profile == StakeholderProfile.STRESS_TESTED:
            # Optimized for crisis scenarios
            return ObjectiveWeights(
                w1_avg_fee=0.1,         # Fees secondary during crisis
                w2_cv_global=0.2,       # Some variability expected
                w3_jump_p95=0.6,        # Large jumps acceptable in crisis
                w4_cv_1h=0.05,         # Minimal
                w5_cv_6h=0.05,         # Minimal
                u1_dwd=0.9,            # MAXIMUM safety priority
                u2_l_max=0.1           # Crisis robustness
            )
        else:  # StakeholderProfile.BALANCED or default
            # Balanced optimization across all objectives
            return ObjectiveWeights(
                w1_avg_fee=0.4,         # Standard fee concern
                w2_cv_global=0.3,       # Baseline stability
                w3_jump_p95=0.2,        # Important jump protection
                w4_cv_1h=0.05,         # Moderate smoothness
                w5_cv_6h=0.05,         # Lower smoothness
                u1_dwd=0.7,            # Balanced safety
                u2_l_max=0.3           # Supporting metric
            )

    def _calculate_global_cv(
        self,
        simulation_results: SimulationResults
    ) -> float:
        """
        Calculate global coefficient of variation

        CV_F = sqrt(Var[F_L2(t)]) / F̄
        """
        fees = simulation_results.fees_per_gas
        if len(fees) == 0:
            return 0.0

        mean_fee = np.mean(fees)
        if mean_fee == 0:
            return 0.0

        std_fee = np.std(fees, ddof=1)
        cv_global = std_fee / mean_fee
        return cv_global

    def _calculate_jump_p95(
        self,
        simulation_results: SimulationResults
    ) -> float:
        """
        Calculate 95th percentile relative jump

        J_0.95 = p95 of |F_L2(t+1) - F_L2(t)| / F_L2(t)
        """
        fees = simulation_results.fees_per_gas
        if len(fees) < 2:
            return 0.0

        # Calculate relative jumps
        fee_diffs = np.abs(np.diff(fees))
        relative_jumps = np.divide(
            fee_diffs,
            fees[:-1],
            out=np.zeros_like(fee_diffs),
            where=fees[:-1] != 0
        )

        if len(relative_jumps) == 0:
            return 0.0

        jump_p95 = np.percentile(relative_jumps, 95)
        return jump_p95

    def _calculate_rolling_cv(
        self,
        simulation_results: SimulationResults,
        window_hours: int
    ) -> float:
        """
        Calculate rolling coefficient of variation

        Args:
            simulation_results: Simulation data
            window_hours: Window size in hours

        Returns:
            Average CV over sliding windows
        """
        fees = simulation_results.fees_per_gas

        # Convert window from hours to batches
        # Assuming 12s per batch (L1 block time)
        batches_per_hour = 3600 / 12  # 300 batches per hour
        window_batches = int(window_hours * batches_per_hour)

        if len(fees) < window_batches:
            # Fall back to global CV if insufficient data
            return self._calculate_global_cv(simulation_results)

        cvs = []
        for i in range(len(fees) - window_batches + 1):
            window_fees = fees[i:i + window_batches]
            window_mean = np.mean(window_fees)

            if window_mean > 0:
                window_std = np.std(window_fees, ddof=1)
                cv = window_std / window_mean
                cvs.append(cv)

        if len(cvs) == 0:
            return 0.0

        return np.mean(cvs)

    def _calculate_deficit_weighted_duration(
        self,
        simulation_results: SimulationResults,
        V_min: float
    ) -> float:
        """
        Calculate deficit-weighted duration

        DWD = E[Σt (V_min - V(t))+]
        """
        vault_balances = simulation_results.vault_balances

        # Calculate positive deficits below V_min
        deficits = np.maximum(0, V_min - vault_balances)

        # Sum total deficit-weighted duration
        dwd = np.sum(deficits)
        return dwd

    def _calculate_max_underfunding_streak(
        self,
        simulation_results: SimulationResults,
        V_min: float
    ) -> int:
        """
        Calculate maximum continuous underfunding streak

        L_max = max length of consecutive batches with V(t) < V_min
        """
        vault_balances = simulation_results.vault_balances

        # Identify underfunded periods
        underfunded = vault_balances < V_min

        if not np.any(underfunded):
            return 0

        # Find consecutive streaks
        streaks = []
        current_streak = 0

        for is_underfunded in underfunded:
            if is_underfunded:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                    current_streak = 0

        # Don't forget the last streak if it extends to the end
        if current_streak > 0:
            streaks.append(current_streak)

        return max(streaks) if streaks else 0

    def _calculate_capital_efficiency(
        self,
        simulation_results: SimulationResults
    ) -> float:
        """
        Calculate capital efficiency

        CapEff = V̄/Q̄
        """
        avg_vault_balance = np.mean(simulation_results.vault_balances)
        Q_bar = simulation_results.Q_bar

        if Q_bar == 0:
            return float('inf')

        capital_efficiency = avg_vault_balance / Q_bar
        return capital_efficiency

    # ========================================
    # METRIC NORMALIZATION FUNCTIONS
    # All metrics normalized to [0,1] range where 1.0 = better
    # ========================================

    def _normalize_average_fee(self, avg_fee_gwei: float) -> float:
        """
        Normalize average fee to [0,1] range using log scale.

        Based on enhanced_metrics.py approach:
        Higher fees are exponentially worse for users.
        Target: 0-50 gwei maps to [1.0, 0.0]
        """
        if avg_fee_gwei <= 0:
            return 1.0
        # Log scale normalization: exp(-fee_gwei/10)
        # Maps: 0 gwei -> 1.0, 10 gwei -> ~0.37, 50+ gwei -> near 0
        normalized = np.exp(-avg_fee_gwei / 10)
        return max(0.0, min(1.0, normalized))

    def _normalize_cv(self, cv: float) -> float:
        """
        Normalize coefficient of variation to [0,1] range.

        Lower CV is better (more predictable fees)
        Formula: max(0, 1 - cv)
        """
        return max(0.0, 1 - cv)

    def _normalize_jump_p95(self, jump_p95: float) -> float:
        """
        Normalize 95th percentile fee jump to [0,1] range.

        Lower jumps are better (more stable UX)
        Target: 0-100% jump maps to [1.0, 0.0]
        """
        if jump_p95 <= 0:
            return 1.0
        # Exponential decay: exp(-jump_p95 * 2)
        # Maps: 0% -> 1.0, 50% -> ~0.37, 100% -> ~0.14
        normalized = np.exp(-jump_p95 * 2)
        return max(0.0, min(1.0, normalized))

    def _normalize_deficit_weighted_duration(
        self,
        dwd: float,
        simulation_length: int,
        target_balance: float = 1.0
    ) -> float:
        """
        Normalize deficit-weighted duration to [0,1] range.

        Lower DWD is better (less time in deficit)
        Normalize by simulation length and target balance.
        """
        if dwd <= 0:
            return 1.0

        # Worst case: entire simulation at 100% deficit
        worst_case_dwd = simulation_length * target_balance

        if worst_case_dwd <= 0:
            return 1.0

        # Linear normalization: 1 - (actual_dwd / worst_case_dwd)
        normalized = max(0.0, 1.0 - (dwd / worst_case_dwd))
        return normalized

    def _normalize_max_underfunding_streak(
        self,
        max_streak: int,
        simulation_length: int
    ) -> float:
        """
        Normalize maximum underfunding streak to [0,1] range.

        Shorter streaks are better (more robust)
        """
        if max_streak <= 0:
            return 1.0

        if simulation_length <= 0:
            return 0.0

        # Linear normalization: 1 - (streak / simulation_length)
        normalized = max(0.0, 1.0 - (max_streak / simulation_length))
        return normalized

    def _normalize_capital_efficiency(
        self,
        capital_efficiency: float,
        target_balance: float = 1.0
    ) -> float:
        """
        Normalize capital efficiency to [0,1] range.

        Lower capital per gas is better (more efficient)
        Target efficiency around 1.0 (target_balance/Q_bar) should score ~0.8
        """
        if capital_efficiency <= 0:
            return 1.0

        # Target capital efficiency (target_balance / Q_bar) for normalization
        target_efficiency = target_balance / 6.9e5  # Q_bar from SPECS

        # Exponential decay: exp(-capital_efficiency / target_efficiency)
        normalized = np.exp(-capital_efficiency / target_efficiency)
        return max(0.0, min(1.0, normalized))