"""
Alpha Data Calculator

Processes Taiko L1 DA transaction data to calculate α_data statistics:
- α_i = gas_used_da[i] / l2_gas[i] for each batch
- Statistical analysis: mean, median, confidence intervals
- Regime detection: calldata vs blob mode analysis
- Time trend analysis and regime changes

This replaces the crude Q̄ = 690,000 constant with empirical measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
import logging

from .taiko_da_fetcher import DATransaction, AlphaDataPoint

logger = logging.getLogger(__name__)


@dataclass
class AlphaStatistics:
    """Statistical summary of alpha data"""
    mean: float
    median: float
    std: float
    p5: float
    p25: float
    p75: float
    p95: float
    confidence_interval_95: Tuple[float, float]
    sample_size: int
    regime: str  # 'blob', 'calldata', 'mixed'


@dataclass
class RegimeAnalysis:
    """Analysis of different DA modes"""
    blob_stats: AlphaStatistics
    calldata_stats: AlphaStatistics
    regime_transition_blocks: List[int]
    dominant_regime: str
    regime_stability: float


class AlphaCalculator:
    """
    Calculator for α_data analysis from Taiko L1 DA transactions
    """

    # Expected alpha ranges based on theoretical analysis
    EXPECTED_ALPHA_BLOB = (0.15, 0.20)  # EIP-4844 blob mode
    EXPECTED_ALPHA_CALLDATA = (0.22, 0.28)  # Traditional calldata mode

    # L2 gas estimation parameters
    DEFAULT_L2_GAS_PER_BATCH = 690_000  # From current Q̄ calibration
    L2_TXS_PER_BATCH_ESTIMATE = 100  # Estimated transactions per batch
    L2_GAS_PER_TX_ESTIMATE = 21_000  # Estimated gas per L2 transaction

    def __init__(
        self,
        l2_gas_estimation_method: str = 'fixed',  # 'fixed', 'dynamic', 'external'
        confidence_level: float = 0.95
    ):
        """
        Initialize alpha calculator

        Args:
            l2_gas_estimation_method: How to estimate L2 gas usage
                - 'fixed': Use fixed value based on current calibration
                - 'dynamic': Try to estimate from L1 batch patterns
                - 'external': Require external L2 gas data
            confidence_level: Confidence level for statistical intervals
        """
        self.l2_gas_estimation_method = l2_gas_estimation_method
        self.confidence_level = confidence_level

    def calculate_alpha_series(
        self,
        da_transactions: List[DATransaction],
        l2_gas_data: Optional[Dict[int, int]] = None
    ) -> List[AlphaDataPoint]:
        """
        Calculate α_data series from DA transactions

        Args:
            da_transactions: List of DA transactions from TaikoDAFetcher
            l2_gas_data: Optional mapping of batch_id -> L2 gas used

        Returns:
            List of alpha data points with calculated α values
        """
        logger.info(f"Calculating alpha series from {len(da_transactions)} DA transactions")

        alpha_points = []

        for i, tx in enumerate(da_transactions):
            # Estimate L2 gas for this batch
            l2_gas = self._estimate_l2_gas(tx, l2_gas_data)

            if l2_gas > 0 and tx.gas_used > 0:
                # Calculate α_i = L1_DA_gas / L2_gas
                alpha_value = tx.gas_used / l2_gas

                # Calculate confidence based on estimation method
                confidence = self._calculate_confidence(tx, l2_gas, l2_gas_data)

                alpha_point = AlphaDataPoint(
                    batch_id=i,  # Use transaction index as batch ID for now
                    alpha_value=alpha_value,
                    l1_da_gas=tx.gas_used,
                    l2_gas=l2_gas,
                    timestamp=tx.timestamp or 0,
                    is_blob_mode=tx.is_blob_mode,
                    confidence=confidence
                )

                alpha_points.append(alpha_point)

        logger.info(f"Generated {len(alpha_points)} alpha data points")
        return alpha_points

    def _estimate_l2_gas(
        self,
        tx: DATransaction,
        l2_gas_data: Optional[Dict[int, int]]
    ) -> int:
        """Estimate L2 gas consumption for a DA transaction"""

        if self.l2_gas_estimation_method == 'external':
            # Use external L2 gas data if available
            if l2_gas_data and tx.l2_batch_id in l2_gas_data:
                return l2_gas_data[tx.l2_batch_id]
            else:
                logger.warning(f"External L2 gas data not found for batch {tx.l2_batch_id}")
                return 0

        elif self.l2_gas_estimation_method == 'dynamic':
            # Try to estimate from transaction data size and patterns
            if tx.is_blob_mode:
                # Blob mode: estimate based on blob capacity
                # Each blob can hold ~128KB of data, estimate L2 gas from utilization
                estimated_txs = min(tx.data_size / 100, self.L2_TXS_PER_BATCH_ESTIMATE)
                return int(estimated_txs * self.L2_GAS_PER_TX_ESTIMATE)
            else:
                # Calldata mode: estimate from calldata size
                # Rough heuristic: more calldata = more L2 transactions
                estimated_txs = min(tx.data_size / 200, self.L2_TXS_PER_BATCH_ESTIMATE)
                return int(estimated_txs * self.L2_GAS_PER_TX_ESTIMATE)

        else:  # 'fixed'
            # Use fixed estimation based on current Q̄ calibration
            return self.DEFAULT_L2_GAS_PER_BATCH

    def _calculate_confidence(
        self,
        tx: DATransaction,
        l2_gas: int,
        l2_gas_data: Optional[Dict[int, int]]
    ) -> float:
        """Calculate confidence level for alpha measurement"""

        if self.l2_gas_estimation_method == 'external' and l2_gas_data:
            return 0.95  # High confidence with actual L2 data

        elif self.l2_gas_estimation_method == 'dynamic':
            # Confidence based on data availability
            if tx.data_size > 1000 and not tx.is_blob_mode:
                return 0.75  # Medium confidence for calldata with substantial data
            elif tx.is_blob_mode:
                return 0.80  # Good confidence for blob mode
            else:
                return 0.50  # Low confidence for minimal data

        else:  # 'fixed'
            return 0.60  # Moderate confidence with fixed estimation

    def calculate_statistics(self, alpha_points: List[AlphaDataPoint]) -> AlphaStatistics:
        """Calculate comprehensive statistics for alpha data"""

        if not alpha_points:
            raise ValueError("No alpha points provided")

        values = [point.alpha_value for point in alpha_points]
        regime = self._determine_primary_regime(alpha_points)

        # Basic statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)

        # Percentiles
        p5 = np.percentile(values, 5)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        p95 = np.percentile(values, 95)

        # Confidence interval for the mean
        n = len(values)
        se = std_val / np.sqrt(n)
        alpha = 1 - self.confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n-1)
        ci_margin = t_critical * se
        confidence_interval = (mean_val - ci_margin, mean_val + ci_margin)

        return AlphaStatistics(
            mean=mean_val,
            median=median_val,
            std=std_val,
            p5=p5,
            p25=p25,
            p75=p75,
            p95=p95,
            confidence_interval_95=confidence_interval,
            sample_size=n,
            regime=regime
        )

    def _determine_primary_regime(self, alpha_points: List[AlphaDataPoint]) -> str:
        """Determine primary DA regime (blob/calldata/mixed)"""
        blob_count = sum(1 for point in alpha_points if point.is_blob_mode)
        total_count = len(alpha_points)

        if blob_count / total_count > 0.8:
            return 'blob'
        elif blob_count / total_count < 0.2:
            return 'calldata'
        else:
            return 'mixed'

    def analyze_regimes(self, alpha_points: List[AlphaDataPoint]) -> RegimeAnalysis:
        """Analyze different DA regimes separately"""

        # Separate by regime
        blob_points = [p for p in alpha_points if p.is_blob_mode]
        calldata_points = [p for p in alpha_points if not p.is_blob_mode]

        # Calculate statistics for each regime
        blob_stats = None
        if blob_points:
            blob_stats = self.calculate_statistics(blob_points)

        calldata_stats = None
        if calldata_points:
            calldata_stats = self.calculate_statistics(calldata_points)

        # Detect regime transitions
        transitions = self._detect_regime_transitions(alpha_points)

        # Determine dominant regime
        if len(blob_points) > len(calldata_points):
            dominant_regime = 'blob'
        elif len(calldata_points) > len(blob_points):
            dominant_regime = 'calldata'
        else:
            dominant_regime = 'mixed'

        # Calculate regime stability
        regime_stability = self._calculate_regime_stability(alpha_points)

        return RegimeAnalysis(
            blob_stats=blob_stats,
            calldata_stats=calldata_stats,
            regime_transition_blocks=transitions,
            dominant_regime=dominant_regime,
            regime_stability=regime_stability
        )

    def _detect_regime_transitions(self, alpha_points: List[AlphaDataPoint]) -> List[int]:
        """Detect regime transition points"""
        transitions = []

        if len(alpha_points) < 10:
            return transitions

        # Look for consecutive switches in regime
        current_regime = alpha_points[0].is_blob_mode
        consecutive_same = 0
        min_consecutive = 5  # Minimum consecutive points to confirm regime

        for i, point in enumerate(alpha_points[1:], 1):
            if point.is_blob_mode == current_regime:
                consecutive_same += 1
            else:
                if consecutive_same >= min_consecutive:
                    # Confirmed regime transition
                    transitions.append(i)
                    current_regime = point.is_blob_mode
                consecutive_same = 1

        return transitions

    def _calculate_regime_stability(self, alpha_points: List[AlphaDataPoint]) -> float:
        """Calculate stability of regime (0 = highly unstable, 1 = very stable)"""
        if len(alpha_points) < 2:
            return 1.0

        # Count regime switches
        switches = 0
        for i in range(1, len(alpha_points)):
            if alpha_points[i].is_blob_mode != alpha_points[i-1].is_blob_mode:
                switches += 1

        # Normalize by maximum possible switches
        max_switches = len(alpha_points) - 1
        stability = 1.0 - (switches / max_switches) if max_switches > 0 else 1.0

        return stability

    def validate_against_templates(self, stats: AlphaStatistics) -> Dict[str, Any]:
        """Validate calculated alpha against theoretical templates"""
        validation = {
            'regime': stats.regime,
            'measured_mean': stats.mean,
            'expected_range': None,
            'within_expected': False,
            'deviation_percent': None,
            'recommendation': None
        }

        # Determine expected range based on regime
        if stats.regime == 'blob':
            expected_range = self.EXPECTED_ALPHA_BLOB
        elif stats.regime == 'calldata':
            expected_range = self.EXPECTED_ALPHA_CALLDATA
        else:  # mixed
            expected_range = (self.EXPECTED_ALPHA_BLOB[0], self.EXPECTED_ALPHA_CALLDATA[1])

        validation['expected_range'] = expected_range

        # Check if within expected range
        within_range = expected_range[0] <= stats.mean <= expected_range[1]
        validation['within_expected'] = within_range

        # Calculate deviation from expected center
        expected_center = (expected_range[0] + expected_range[1]) / 2
        deviation_percent = abs(stats.mean - expected_center) / expected_center * 100
        validation['deviation_percent'] = deviation_percent

        # Generate recommendation
        if within_range:
            validation['recommendation'] = f"GOOD: Measured α = {stats.mean:.3f} within expected range"
        else:
            if stats.mean < expected_range[0]:
                validation['recommendation'] = f"LOW: Measured α = {stats.mean:.3f} below expected minimum {expected_range[0]:.3f}"
            else:
                validation['recommendation'] = f"HIGH: Measured α = {stats.mean:.3f} above expected maximum {expected_range[1]:.3f}"

        return validation

    def generate_analysis_report(
        self,
        alpha_points: List[AlphaDataPoint],
        save_plots: bool = True,
        output_dir: str = "./analysis_output"
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""

        logger.info("Generating alpha analysis report...")

        # Calculate overall statistics
        overall_stats = self.calculate_statistics(alpha_points)

        # Analyze regimes
        regime_analysis = self.analyze_regimes(alpha_points)

        # Validate against templates
        validation = self.validate_against_templates(overall_stats)

        # Time series analysis
        time_analysis = self._analyze_time_trends(alpha_points)

        # Generate plots if requested
        plot_paths = []
        if save_plots:
            plot_paths = self._generate_analysis_plots(alpha_points, overall_stats, regime_analysis, output_dir)

        # Compile report
        report = {
            'overall_statistics': overall_stats,
            'regime_analysis': regime_analysis,
            'template_validation': validation,
            'time_analysis': time_analysis,
            'data_quality': self._assess_data_quality(alpha_points),
            'recommendations': self._generate_recommendations(overall_stats, regime_analysis, validation),
            'plot_paths': plot_paths,
            'generated_at': datetime.now().isoformat()
        }

        logger.info("Alpha analysis report generated successfully")
        return report

    def _analyze_time_trends(self, alpha_points: List[AlphaDataPoint]) -> Dict[str, Any]:
        """Analyze time trends in alpha data"""
        if len(alpha_points) < 10:
            return {'trend': 'insufficient_data', 'slope': 0, 'r_squared': 0}

        # Extract time series data
        times = [p.timestamp for p in alpha_points if p.timestamp > 0]
        alphas = [p.alpha_value for p in alpha_points if p.timestamp > 0]

        if len(times) < 10:
            return {'trend': 'insufficient_time_data', 'slope': 0, 'r_squared': 0}

        # Linear regression for trend analysis
        times_normalized = [(t - min(times)) / 3600 for t in times]  # Convert to hours
        slope, intercept, r_value, p_value, std_err = stats.linregress(times_normalized, alphas)

        trend_direction = 'stable'
        if abs(slope) > 0.001:  # Significant trend threshold
            trend_direction = 'increasing' if slope > 0 else 'decreasing'

        return {
            'trend': trend_direction,
            'slope': slope,
            'slope_per_hour': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    def _assess_data_quality(self, alpha_points: List[AlphaDataPoint]) -> Dict[str, Any]:
        """Assess the quality of alpha data"""
        if not alpha_points:
            return {'quality': 'no_data', 'score': 0.0}

        # Calculate quality metrics
        total_points = len(alpha_points)
        high_confidence_points = sum(1 for p in alpha_points if p.confidence >= 0.8)
        medium_confidence_points = sum(1 for p in alpha_points if 0.6 <= p.confidence < 0.8)

        high_conf_ratio = high_confidence_points / total_points
        medium_conf_ratio = medium_confidence_points / total_points

        # Calculate overall quality score
        quality_score = high_conf_ratio * 1.0 + medium_conf_ratio * 0.7

        quality_level = 'poor'
        if quality_score >= 0.8:
            quality_level = 'excellent'
        elif quality_score >= 0.6:
            quality_level = 'good'
        elif quality_score >= 0.4:
            quality_level = 'fair'

        return {
            'quality': quality_level,
            'score': quality_score,
            'total_points': total_points,
            'high_confidence_points': high_confidence_points,
            'medium_confidence_points': medium_confidence_points,
            'high_confidence_ratio': high_conf_ratio
        }

    def _generate_recommendations(
        self,
        overall_stats: AlphaStatistics,
        regime_analysis: RegimeAnalysis,
        validation: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Deployment recommendation based on validation
        if validation['within_expected']:
            recommendations.append(
                f"✅ DEPLOY: Use measured α = {overall_stats.mean:.3f} "
                f"(95% CI: {overall_stats.confidence_interval_95[0]:.3f}-{overall_stats.confidence_interval_95[1]:.3f})"
            )
        else:
            recommendations.append(
                f"⚠️ INVESTIGATE: Measured α = {overall_stats.mean:.3f} outside expected range {validation['expected_range']}"
            )

        # Regime-specific recommendations
        if regime_analysis.dominant_regime == 'mixed' and regime_analysis.regime_stability < 0.8:
            recommendations.append(
                "🔄 CONSIDER: Regime-aware α model due to mixed blob/calldata usage and frequent transitions"
            )

        # Statistical recommendations
        if overall_stats.std / overall_stats.mean > 0.3:  # High coefficient of variation
            recommendations.append(
                f"📊 NOTE: High variability (CV = {(overall_stats.std/overall_stats.mean):.2f}), "
                "consider rolling average implementation"
            )

        # Sample size recommendations
        if overall_stats.sample_size < 100:
            recommendations.append(
                f"📈 COLLECT: Sample size ({overall_stats.sample_size}) is small, "
                "gather more data for robust estimation"
            )

        return recommendations

    def _generate_analysis_plots(
        self,
        alpha_points: List[AlphaDataPoint],
        overall_stats: AlphaStatistics,
        regime_analysis: RegimeAnalysis,
        output_dir: str
    ) -> List[str]:
        """Generate analysis plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        plot_paths = []

        # 1. Alpha distribution histogram
        plt.figure(figsize=(10, 6))
        alphas = [p.alpha_value for p in alpha_points]
        plt.hist(alphas, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(overall_stats.mean, color='red', linestyle='--', label=f'Mean: {overall_stats.mean:.3f}')
        plt.axvline(overall_stats.median, color='green', linestyle='--', label=f'Median: {overall_stats.median:.3f}')
        plt.xlabel('Alpha Value (L1 DA gas / L2 gas)')
        plt.ylabel('Frequency')
        plt.title('Alpha Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = f"{output_dir}/alpha_distribution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plot_paths.append(plot_path)
        plt.close()

        # 2. Time series plot
        if len(alpha_points) > 1:
            plt.figure(figsize=(12, 6))
            timestamps = [p.timestamp for p in alpha_points if p.timestamp > 0]
            alphas_ts = [p.alpha_value for p in alpha_points if p.timestamp > 0]

            if len(timestamps) > 0:
                plt.plot(timestamps, alphas_ts, 'o-', alpha=0.7, markersize=3)
                plt.axhline(overall_stats.mean, color='red', linestyle='--', alpha=0.8)
                plt.xlabel('Timestamp')
                plt.ylabel('Alpha Value')
                plt.title('Alpha Time Series')
                plt.grid(True, alpha=0.3)

                plot_path = f"{output_dir}/alpha_time_series.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plot_paths.append(plot_path)
            plt.close()

        # 3. Regime comparison (if both regimes exist)
        if regime_analysis.blob_stats and regime_analysis.calldata_stats:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Blob mode data
            blob_alphas = [p.alpha_value for p in alpha_points if p.is_blob_mode]
            ax1.hist(blob_alphas, bins=20, alpha=0.7, color='blue', label='Blob Mode')
            ax1.axvline(regime_analysis.blob_stats.mean, color='darkblue', linestyle='--')
            ax1.set_xlabel('Alpha Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Blob Mode Alpha Distribution')
            ax1.grid(True, alpha=0.3)

            # Calldata mode data
            calldata_alphas = [p.alpha_value for p in alpha_points if not p.is_blob_mode]
            ax2.hist(calldata_alphas, bins=20, alpha=0.7, color='orange', label='Calldata Mode')
            ax2.axvline(regime_analysis.calldata_stats.mean, color='darkorange', linestyle='--')
            ax2.set_xlabel('Alpha Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Calldata Mode Alpha Distribution')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = f"{output_dir}/regime_comparison.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plot_paths.append(plot_path)
            plt.close()

        return plot_paths

    def export_results(
        self,
        alpha_points: List[AlphaDataPoint],
        report: Dict[str, Any],
        output_dir: str = "./analysis_output"
    ):
        """Export alpha analysis results"""
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        # Export alpha data points as CSV
        alpha_df = pd.DataFrame([
            {
                'batch_id': p.batch_id,
                'alpha_value': p.alpha_value,
                'l1_da_gas': p.l1_da_gas,
                'l2_gas': p.l2_gas,
                'timestamp': p.timestamp,
                'is_blob_mode': p.is_blob_mode,
                'confidence': p.confidence
            } for p in alpha_points
        ])

        alpha_df.to_csv(f"{output_dir}/alpha_data_points.csv", index=False)

        # Export report as JSON (excluding non-serializable objects)
        serializable_report = self._make_serializable(report)
        with open(f"{output_dir}/alpha_analysis_report.json", 'w') as f:
            json.dump(serializable_report, f, indent=2)

        logger.info(f"Results exported to {output_dir}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


# Utility function for quick analysis
def quick_alpha_calculation(
    da_transactions: List[DATransaction],
    l2_gas_data: Optional[Dict[int, int]] = None
) -> Dict[str, Any]:
    """Quick alpha calculation and analysis"""

    calculator = AlphaCalculator()

    # Calculate alpha series
    alpha_points = calculator.calculate_alpha_series(da_transactions, l2_gas_data)

    if not alpha_points:
        return {'error': 'No valid alpha points calculated'}

    # Generate basic statistics
    stats = calculator.calculate_statistics(alpha_points)
    validation = calculator.validate_against_templates(stats)

    return {
        'alpha_mean': stats.mean,
        'alpha_median': stats.median,
        'alpha_std': stats.std,
        'confidence_interval_95': stats.confidence_interval_95,
        'sample_size': stats.sample_size,
        'regime': stats.regime,
        'within_expected_range': validation['within_expected'],
        'expected_range': validation['expected_range'],
        'recommendation': validation['recommendation']
    }