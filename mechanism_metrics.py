"""
Metrics Framework for Taiko Fee Mechanism Analysis

This module provides comprehensive metrics for evaluating the performance
of the fee mechanism across multiple dimensions: vault stability, user experience,
system efficiency, and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings

@dataclass
class MechanismMetrics:
    """Container for all mechanism evaluation metrics."""

    # Vault Stability Metrics
    avg_vault_balance: float
    vault_balance_std: float
    avg_deficit: float
    deficit_std: float
    time_underfunded_pct: float
    time_overfunded_pct: float
    max_deficit: float
    max_surplus: float

    # User Experience Metrics
    avg_fee: float
    fee_std: float
    fee_cv: float  # Coefficient of variation
    fee_95th_percentile: float
    fee_99th_percentile: float
    fee_volatility: float  # Rolling volatility measure

    # System Efficiency Metrics
    l1_tracking_error: float  # How well fees track L1 costs
    response_lag: float       # Time to respond to L1 changes
    overpayment_ratio: float  # Ratio of total fees to total L1 costs
    vault_utilization: float  # Average vault usage relative to target

    # Risk Metrics
    insolvency_probability: float  # P(vault balance < minimum)
    insolvency_duration: float     # Average duration when insolvent
    fee_shock_frequency: float     # Frequency of large fee increases
    var_95_deficit: float          # 95% VaR for vault deficit

    # Mechanism-Specific Metrics
    deficit_correction_efficiency: float  # How quickly deficits are corrected
    l1_sensitivity: float                 # Sensitivity to L1 changes
    demand_elasticity_realized: float     # Realized elasticity of demand

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for easy manipulation."""
        return {
            'avg_vault_balance': self.avg_vault_balance,
            'vault_balance_std': self.vault_balance_std,
            'avg_deficit': self.avg_deficit,
            'deficit_std': self.deficit_std,
            'time_underfunded_pct': self.time_underfunded_pct,
            'time_overfunded_pct': self.time_overfunded_pct,
            'max_deficit': self.max_deficit,
            'max_surplus': self.max_surplus,
            'avg_fee': self.avg_fee,
            'fee_std': self.fee_std,
            'fee_cv': self.fee_cv,
            'fee_95th_percentile': self.fee_95th_percentile,
            'fee_99th_percentile': self.fee_99th_percentile,
            'fee_volatility': self.fee_volatility,
            'l1_tracking_error': self.l1_tracking_error,
            'response_lag': self.response_lag,
            'overpayment_ratio': self.overpayment_ratio,
            'vault_utilization': self.vault_utilization,
            'insolvency_probability': self.insolvency_probability,
            'insolvency_duration': self.insolvency_duration,
            'fee_shock_frequency': self.fee_shock_frequency,
            'var_95_deficit': self.var_95_deficit,
            'deficit_correction_efficiency': self.deficit_correction_efficiency,
            'l1_sensitivity': self.l1_sensitivity,
            'demand_elasticity_realized': self.demand_elasticity_realized
        }


class MetricsCalculator:
    """Calculates comprehensive metrics from simulation results."""

    def __init__(self, target_balance: float, min_balance: float = 0):
        self.target_balance = target_balance
        self.min_balance = min_balance

    def calculate_vault_stability_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate vault stability metrics."""
        vault_balance = df['vault_balance'].values
        deficit = df['vault_deficit'].values

        metrics = {
            'avg_vault_balance': np.mean(vault_balance),
            'vault_balance_std': np.std(vault_balance),
            'avg_deficit': np.mean(deficit),
            'deficit_std': np.std(deficit),
            'time_underfunded_pct': np.mean(deficit > 0) * 100,
            'time_overfunded_pct': np.mean(deficit < 0) * 100,
            'max_deficit': np.max(deficit),
            'max_surplus': -np.min(deficit)  # Max negative deficit
        }

        return metrics

    def calculate_user_experience_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate user experience metrics."""
        fees = df['estimated_fee'].values

        # Remove any extreme outliers for volatility calculation
        fees_clean = fees[fees < np.percentile(fees, 99.5)]

        # Rolling volatility (20-period)
        fee_series = pd.Series(fees)
        rolling_vol = fee_series.rolling(window=20).std()
        avg_volatility = rolling_vol.mean()

        metrics = {
            'avg_fee': np.mean(fees),
            'fee_std': np.std(fees),
            'fee_cv': np.std(fees) / np.mean(fees) if np.mean(fees) > 0 else 0,
            'fee_95th_percentile': np.percentile(fees, 95),
            'fee_99th_percentile': np.percentile(fees, 99),
            'fee_volatility': avg_volatility
        }

        return metrics

    def calculate_system_efficiency_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate system efficiency metrics."""
        fees = df['estimated_fee'].values
        l1_costs = df['estimated_l1_cost'].values
        vault_balance = df['vault_balance'].values
        total_fees_collected = df['fee_collected'].sum()
        total_l1_costs = df['l1_cost_paid'].sum()

        # L1 tracking error (correlation between fees and L1 costs)
        if len(l1_costs) > 1 and np.std(l1_costs) > 0:
            correlation = np.corrcoef(fees, l1_costs)[0, 1]
            tracking_error = 1 - correlation if not np.isnan(correlation) else 1
        else:
            tracking_error = 1

        # Response lag using cross-correlation
        response_lag = self._calculate_response_lag(fees, l1_costs)

        # Overpayment ratio
        overpayment_ratio = total_fees_collected / total_l1_costs if total_l1_costs > 0 else np.inf

        # Vault utilization
        vault_utilization = np.mean(vault_balance) / self.target_balance

        metrics = {
            'l1_tracking_error': tracking_error,
            'response_lag': response_lag,
            'overpayment_ratio': overpayment_ratio,
            'vault_utilization': vault_utilization
        }

        return metrics

    def _calculate_response_lag(self, fees: np.ndarray, l1_costs: np.ndarray) -> float:
        """Calculate response lag using cross-correlation."""
        if len(fees) < 10:
            return 0

        # Normalize series
        fees_norm = (fees - np.mean(fees)) / np.std(fees) if np.std(fees) > 0 else fees
        l1_norm = (l1_costs - np.mean(l1_costs)) / np.std(l1_costs) if np.std(l1_costs) > 0 else l1_costs

        # Cross-correlation
        max_lag = min(50, len(fees) // 4)  # Don't look beyond 25% of series length
        xcorr = np.correlate(fees_norm, l1_norm, mode='full')

        # Find lag with maximum correlation
        lags = np.arange(-max_lag, max_lag + 1)
        center_idx = len(xcorr) // 2
        relevant_xcorr = xcorr[center_idx - max_lag:center_idx + max_lag + 1]

        max_idx = np.argmax(np.abs(relevant_xcorr))
        return abs(lags[max_idx])

    def calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-related metrics."""
        vault_balance = df['vault_balance'].values
        deficit = df['vault_deficit'].values
        fees = df['estimated_fee'].values

        # Insolvency analysis
        insolvent_periods = vault_balance < self.min_balance
        insolvency_probability = np.mean(insolvent_periods) if len(insolvent_periods) > 0 else 0

        # Calculate average duration of insolvency periods
        insolvency_durations = []
        current_duration = 0
        for is_insolvent in insolvent_periods:
            if is_insolvent:
                current_duration += 1
            else:
                if current_duration > 0:
                    insolvency_durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            insolvency_durations.append(current_duration)

        avg_insolvency_duration = np.mean(insolvency_durations) if insolvency_durations else 0

        # Fee shock analysis (periods where fees increase by >100%)
        fee_changes = np.diff(fees) / fees[:-1]
        fee_shocks = fee_changes > 1.0  # 100% increase
        fee_shock_frequency = np.mean(fee_shocks) if len(fee_shocks) > 0 else 0

        # Value at Risk for deficit
        var_95_deficit = np.percentile(deficit, 95)

        metrics = {
            'insolvency_probability': insolvency_probability,
            'insolvency_duration': avg_insolvency_duration,
            'fee_shock_frequency': fee_shock_frequency,
            'var_95_deficit': var_95_deficit
        }

        return metrics

    def calculate_mechanism_specific_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate mechanism-specific metrics."""
        deficit = df['vault_deficit'].values
        fees = df['estimated_fee'].values
        l1_costs = df['estimated_l1_cost'].values
        volume = df['transaction_volume'].values

        # Deficit correction efficiency (how quickly large deficits are corrected)
        deficit_correction_efficiency = self._calculate_deficit_correction_efficiency(deficit)

        # L1 sensitivity (how much fees change per unit change in L1 costs)
        l1_sensitivity = self._calculate_l1_sensitivity(fees, l1_costs)

        # Realized demand elasticity
        elasticity_realized = self._calculate_realized_elasticity(fees, volume)

        metrics = {
            'deficit_correction_efficiency': deficit_correction_efficiency,
            'l1_sensitivity': l1_sensitivity,
            'demand_elasticity_realized': elasticity_realized
        }

        return metrics

    def _calculate_deficit_correction_efficiency(self, deficit: np.ndarray) -> float:
        """Calculate how efficiently large deficits are corrected."""
        if len(deficit) < 10:
            return 0

        # Find periods of large deficit (>50% of target)
        large_deficit_threshold = 0.5 * self.target_balance
        large_deficit_periods = deficit > large_deficit_threshold

        if not np.any(large_deficit_periods):
            return 1.0  # No large deficits to correct

        # Calculate average time to correct large deficits
        correction_times = []
        in_deficit = False
        deficit_start = 0

        for t, is_large_deficit in enumerate(large_deficit_periods):
            if is_large_deficit and not in_deficit:
                deficit_start = t
                in_deficit = True
            elif not is_large_deficit and in_deficit:
                correction_times.append(t - deficit_start)
                in_deficit = False

        if correction_times:
            avg_correction_time = np.mean(correction_times)
            # Efficiency is inverse of correction time (normalized)
            max_reasonable_time = 100  # steps
            efficiency = max(0, 1 - avg_correction_time / max_reasonable_time)
        else:
            efficiency = 0.5  # Some deficits never corrected

        return efficiency

    def _calculate_l1_sensitivity(self, fees: np.ndarray, l1_costs: np.ndarray) -> float:
        """Calculate sensitivity of fees to L1 cost changes."""
        if len(fees) < 2 or len(l1_costs) < 2:
            return 0

        # Use linear regression to estimate sensitivity
        try:
            slope, _, r_value, _, _ = stats.linregress(l1_costs, fees)
            # Weight by R-squared to account for fit quality
            sensitivity = abs(slope) * (r_value ** 2)
        except:
            sensitivity = 0

        return sensitivity

    def _calculate_realized_elasticity(self, fees: np.ndarray, volume: np.ndarray) -> float:
        """Calculate realized demand elasticity."""
        if len(fees) < 2 or len(volume) < 2:
            return 0

        # Remove zero volumes and fees to avoid log(0)
        valid_idx = (fees > 0) & (volume > 0)
        if np.sum(valid_idx) < 2:
            return 0

        fees_valid = fees[valid_idx]
        volume_valid = volume[valid_idx]

        try:
            # Log-linear regression: log(volume) = a + b*log(fee)
            log_fees = np.log(fees_valid)
            log_volume = np.log(volume_valid)
            slope, _, r_value, _, _ = stats.linregress(log_fees, log_volume)
            # Weight by R-squared
            elasticity = abs(slope) * (r_value ** 2)
        except:
            elasticity = 0

        return elasticity

    def calculate_all_metrics(self, df: pd.DataFrame) -> MechanismMetrics:
        """Calculate all metrics and return as MechanismMetrics object."""
        vault_metrics = self.calculate_vault_stability_metrics(df)
        ux_metrics = self.calculate_user_experience_metrics(df)
        efficiency_metrics = self.calculate_system_efficiency_metrics(df)
        risk_metrics = self.calculate_risk_metrics(df)
        mechanism_metrics = self.calculate_mechanism_specific_metrics(df)

        # Combine all metrics
        all_metrics = {**vault_metrics, **ux_metrics, **efficiency_metrics,
                      **risk_metrics, **mechanism_metrics}

        return MechanismMetrics(**all_metrics)


class ParameterSweepAnalyzer:
    """Analyzes results from parameter sweeps."""

    def __init__(self, target_balance: float):
        self.target_balance = target_balance
        self.metrics_calculator = MetricsCalculator(target_balance)

    def run_parameter_sweep(self,
                           simulator_class,
                           l1_model,
                           param_ranges: Dict[str, List[float]],
                           base_params: dict,
                           n_trials: int = 5) -> pd.DataFrame:
        """
        Run parameter sweep and collect metrics.

        Args:
            simulator_class: TaikoFeeSimulator class
            l1_model: L1 dynamics model instance
            param_ranges: Dictionary of parameter names to lists of values
            base_params: Base parameter set
            n_trials: Number of Monte Carlo trials per parameter set

        Returns:
            DataFrame with parameter combinations and metrics
        """
        from itertools import product
        from fee_mechanism_simulator import SimulationParams

        results = []

        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        for param_combo in product(*param_values):
            param_dict = dict(zip(param_names, param_combo))

            # Run multiple trials for this parameter set
            trial_metrics = []

            for trial in range(n_trials):
                # Create parameters
                params_dict = {**base_params, **param_dict}
                params = SimulationParams(**params_dict)

                # Run simulation
                simulator = simulator_class(params, l1_model)
                df = simulator.run_simulation()

                # Calculate metrics
                metrics = self.metrics_calculator.calculate_all_metrics(df)
                trial_metrics.append(metrics.to_dict())

            # Average metrics across trials
            avg_metrics = {}
            for key in trial_metrics[0].keys():
                values = [m[key] for m in trial_metrics]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f"{key}_std"] = np.std(values)

            # Add parameter values
            result = {**param_dict, **avg_metrics}
            results.append(result)

        return pd.DataFrame(results)

    def analyze_mu_zero_viability(self,
                                 simulator_class,
                                 l1_models: List,
                                 base_params: dict) -> Dict[str, pd.DataFrame]:
        """
        Specifically analyze whether μ=0 is viable across different L1 scenarios.
        """
        from fee_mechanism_simulator import SimulationParams

        mu_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        nu_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        results = {}

        for l1_model in l1_models:
            model_results = []

            for mu in mu_values:
                for nu in nu_values:
                    params_dict = {**base_params, 'mu': mu, 'nu': nu}
                    params = SimulationParams(**params_dict)

                    # Run multiple trials
                    trial_metrics = []
                    for trial in range(3):  # Fewer trials for detailed analysis
                        simulator = simulator_class(params, l1_model)
                        df = simulator.run_simulation()
                        metrics = self.metrics_calculator.calculate_all_metrics(df)
                        trial_metrics.append(metrics.to_dict())

                    # Average results
                    avg_metrics = {}
                    for key in trial_metrics[0].keys():
                        values = [m[key] for m in trial_metrics]
                        avg_metrics[key] = np.mean(values)

                    result = {'mu': mu, 'nu': nu, **avg_metrics}
                    model_results.append(result)

            results[l1_model.get_name()] = pd.DataFrame(model_results)

        return results