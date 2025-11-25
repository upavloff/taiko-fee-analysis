"""
Comprehensive Analysis Script for Taiko Fee Mechanism

This script runs the full analysis suite to answer the key research questions:
1. Impact of L1 basefee dynamics on the mechanism
2. Role of μ and ν parameters
3. Viability of μ=0 approach
4. Effects of fee caps and other constraints

Usage:
    python run_analysis.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from fee_mechanism_simulator import (
    TaikoFeeSimulator, SimulationParams,
    GeometricBrownianMotion, RegimeSwitchingModel, SpikeEventsModel
)
from advanced_mechanisms import (
    MultiTimescaleSimulator, AdvancedSimulationParams, OptimalControlBenchmark
)
from mechanism_metrics import MetricsCalculator, ParameterSweepAnalyzer

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


def create_l1_scenarios() -> List:
    """Create different L1 dynamics scenarios for testing."""

    scenarios = [
        GeometricBrownianMotion(mu=0.0, sigma=0.3),  # Baseline: no drift, moderate volatility
        GeometricBrownianMotion(mu=0.0, sigma=0.6),  # High volatility
        GeometricBrownianMotion(mu=0.0, sigma=0.1),  # Low volatility
        RegimeSwitchingModel(),                       # Regime switching
        SpikeEventsModel(                            # Spike events on baseline
            GeometricBrownianMotion(mu=0.0, sigma=0.2),
            spike_probability=0.002
        )
    ]

    return scenarios


def run_baseline_comparison():
    """Compare basic mechanism vs advanced variants across L1 scenarios."""

    print("Running baseline mechanism comparison...")

    l1_scenarios = create_l1_scenarios()
    target_balance = 1000.0

    # Base parameters
    base_params = {
        'mu': 0.5,
        'nu': 0.3,
        'H': 144,  # 1 day at 12s blocks
        'target_balance': target_balance,
        'base_demand': 100,
        'fee_elasticity': 0.2,
        'total_steps': 2000,  # Shorter for initial analysis
        'batch_frequency': 0.1
    }

    results = []

    for l1_model in l1_scenarios:
        print(f"  Testing {l1_model.get_name()}...")

        # Test standard mechanism
        params = SimulationParams(**base_params)
        simulator = TaikoFeeSimulator(params, l1_model)
        df_standard = simulator.run_simulation()

        metrics_calc = MetricsCalculator(target_balance)
        metrics_standard = metrics_calc.calculate_all_metrics(df_standard)

        # Test advanced mechanism
        advanced_params = AdvancedSimulationParams(**base_params,
                                                   dynamic_target=True,
                                                   use_predictive_l1=True)
        advanced_simulator = MultiTimescaleSimulator(advanced_params, l1_model)
        df_advanced = advanced_simulator.run_simulation()
        metrics_advanced = metrics_calc.calculate_all_metrics(df_advanced)

        # Store results
        results.append({
            'scenario': l1_model.get_name(),
            'mechanism': 'Standard',
            **metrics_standard.to_dict()
        })

        results.append({
            'scenario': l1_model.get_name(),
            'mechanism': 'Advanced',
            **metrics_advanced.to_dict()
        })

    return pd.DataFrame(results)


def analyze_mu_zero_viability():
    """Comprehensive analysis of μ=0 viability across scenarios."""

    print("Analyzing μ=0 viability...")

    l1_scenarios = create_l1_scenarios()
    target_balance = 1000.0

    # Parameters for μ=0 analysis
    base_params = {
        'target_balance': target_balance,
        'base_demand': 100,
        'fee_elasticity': 0.2,
        'total_steps': 3000,  # Longer simulation for stability analysis
        'batch_frequency': 0.1,
        'H': 144
    }

    analyzer = ParameterSweepAnalyzer(target_balance)

    # Compare μ=0 vs μ>0 across scenarios
    results = analyzer.analyze_mu_zero_viability(
        TaikoFeeSimulator,
        l1_scenarios,
        base_params
    )

    return results


def run_parameter_sweeps():
    """Run comprehensive parameter sweeps."""

    print("Running parameter sweeps...")

    l1_model = GeometricBrownianMotion(mu=0.0, sigma=0.3)  # Use baseline scenario
    target_balance = 1000.0

    base_params = {
        'target_balance': target_balance,
        'base_demand': 100,
        'fee_elasticity': 0.2,
        'total_steps': 2000,
        'batch_frequency': 0.1
    }

    # Define parameter ranges
    param_ranges = {
        'mu': [0.0, 0.25, 0.5, 0.75, 1.0],
        'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
        'H': [72, 144, 288, 576]  # 0.5, 1, 2, 4 days
    }

    analyzer = ParameterSweepAnalyzer(target_balance)
    results = analyzer.run_parameter_sweep(
        TaikoFeeSimulator,
        l1_model,
        param_ranges,
        base_params,
        n_trials=3
    )

    return results


def analyze_fee_caps_impact():
    """Analyze impact of different fee cap levels."""

    print("Analyzing fee cap impact...")

    l1_model = SpikeEventsModel(
        GeometricBrownianMotion(mu=0.0, sigma=0.3),
        spike_probability=0.003  # More frequent spikes for cap testing
    )

    target_balance = 1000.0
    base_params = {
        'mu': 0.5,
        'nu': 0.3,
        'H': 144,
        'target_balance': target_balance,
        'base_demand': 100,
        'fee_elasticity': 0.2,
        'total_steps': 2000,
        'batch_frequency': 0.1
    }

    # Test different cap levels
    cap_multipliers = [None, 2.0, 5.0, 10.0]  # None = no cap
    results = []

    for cap_mult in cap_multipliers:
        print(f"  Testing cap multiplier: {cap_mult}")

        if cap_mult is None:
            fee_cap = None
            cap_label = "No Cap"
        else:
            base_fee = 0.05  # Estimate of typical fee
            fee_cap = base_fee * cap_mult
            cap_label = f"{cap_mult}x Cap"

        params = SimulationParams(**base_params, fee_cap=fee_cap)

        # Run multiple trials
        trial_results = []
        for trial in range(5):
            simulator = TaikoFeeSimulator(params, l1_model)
            df = simulator.run_simulation()

            metrics_calc = MetricsCalculator(target_balance)
            metrics = metrics_calc.calculate_all_metrics(df)
            trial_results.append(metrics.to_dict())

        # Average across trials
        avg_metrics = {}
        for key in trial_results[0].keys():
            values = [m[key] for m in trial_results]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)

        avg_metrics['cap_label'] = cap_label
        avg_metrics['cap_multiplier'] = cap_mult
        results.append(avg_metrics)

    return pd.DataFrame(results)


def test_extreme_scenarios():
    """Test mechanism under extreme market conditions."""

    print("Testing extreme scenarios...")

    target_balance = 1000.0
    base_params = {
        'mu': 0.5,
        'nu': 0.3,
        'H': 144,
        'target_balance': target_balance,
        'base_demand': 100,
        'fee_elasticity': 0.2,
        'total_steps': 1500,  # Shorter for extreme scenarios
        'batch_frequency': 0.1
    }

    # Extreme scenarios
    extreme_scenarios = [
        ('Very High Volatility', GeometricBrownianMotion(mu=0.0, sigma=1.0)),
        ('Persistent High Fees', GeometricBrownianMotion(mu=0.001, sigma=0.2)),  # Positive drift
        ('Frequent Spikes', SpikeEventsModel(
            GeometricBrownianMotion(mu=0.0, sigma=0.2),
            spike_probability=0.01,
            spike_magnitude_range=(5.0, 20.0)
        )),
        ('High Elasticity', GeometricBrownianMotion(mu=0.0, sigma=0.3))  # Will test with high fee elasticity
    ]

    results = []

    for scenario_name, l1_model in extreme_scenarios:
        print(f"  Testing {scenario_name}...")

        # Adjust parameters for high elasticity scenario
        if scenario_name == 'High Elasticity':
            test_params = {**base_params, 'fee_elasticity': 1.0}  # Very elastic demand
        else:
            test_params = base_params

        params = SimulationParams(**test_params)
        simulator = TaikoFeeSimulator(params, l1_model)
        df = simulator.run_simulation()

        metrics_calc = MetricsCalculator(target_balance)
        metrics = metrics_calc.calculate_all_metrics(df)

        result = {'scenario': scenario_name, **metrics.to_dict()}
        results.append(result)

    return pd.DataFrame(results)


def generate_optimal_control_comparison():
    """Compare mechanism performance vs theoretical optimum."""

    print("Generating optimal control comparison...")

    l1_model = GeometricBrownianMotion(mu=0.0, sigma=0.3)
    target_balance = 1000.0

    base_params = {
        'mu': 0.5,
        'nu': 0.3,
        'H': 144,
        'target_balance': target_balance,
        'base_demand': 100,
        'fee_elasticity': 0.2,
        'total_steps': 1000,  # Shorter for optimal control
        'batch_frequency': 0.1
    }

    # Generate L1 sequence
    l1_sequence = l1_model.generate_sequence(base_params['total_steps'])

    # Run mechanism simulation
    params = SimulationParams(**base_params)
    simulator = TaikoFeeSimulator(params, l1_model)
    df = simulator.run_simulation()

    # Calculate optimal benchmark
    benchmark = OptimalControlBenchmark(params)
    volume_sequence = df['transaction_volume'].values

    optimal_metrics = benchmark.calculate_optimal_metrics(l1_sequence, volume_sequence)

    # Mechanism metrics
    metrics_calc = MetricsCalculator(target_balance)
    mechanism_metrics = metrics_calc.calculate_all_metrics(df)

    comparison = {
        'mechanism_avg_fee': mechanism_metrics.avg_fee,
        'mechanism_fee_std': mechanism_metrics.fee_std,
        'mechanism_fee_cv': mechanism_metrics.fee_cv,
        'optimal_avg_fee': optimal_metrics['optimal_avg_fee'],
        'optimal_fee_std': optimal_metrics['optimal_fee_std'],
        'optimal_fee_cv': optimal_metrics['optimal_fee_cv'],
        'efficiency_ratio': optimal_metrics['optimal_fee_cv'] / mechanism_metrics.fee_cv if mechanism_metrics.fee_cv > 0 else 1
    }

    return comparison


def main():
    """Run complete analysis suite."""

    print("=== Taiko Fee Mechanism Analysis ===\n")

    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)

    results = {}

    # 1. Baseline comparison
    try:
        baseline_results = run_baseline_comparison()
        results['baseline_comparison'] = baseline_results
        baseline_results.to_csv('results/baseline_comparison.csv', index=False)
        print("✓ Baseline comparison completed\n")
    except Exception as e:
        print(f"✗ Baseline comparison failed: {e}\n")

    # 2. μ=0 viability analysis
    try:
        mu_zero_results = analyze_mu_zero_viability()
        results['mu_zero_analysis'] = mu_zero_results

        # Save results for each scenario
        for scenario, df in mu_zero_results.items():
            safe_name = scenario.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            df.to_csv(f'results/mu_zero_{safe_name}.csv', index=False)

        print("✓ μ=0 viability analysis completed\n")
    except Exception as e:
        print(f"✗ μ=0 viability analysis failed: {e}\n")

    # 3. Parameter sweeps
    try:
        param_sweep_results = run_parameter_sweeps()
        results['parameter_sweeps'] = param_sweep_results
        param_sweep_results.to_csv('results/parameter_sweeps.csv', index=False)
        print("✓ Parameter sweeps completed\n")
    except Exception as e:
        print(f"✗ Parameter sweeps failed: {e}\n")

    # 4. Fee caps analysis
    try:
        fee_caps_results = analyze_fee_caps_impact()
        results['fee_caps'] = fee_caps_results
        fee_caps_results.to_csv('results/fee_caps_analysis.csv', index=False)
        print("✓ Fee caps analysis completed\n")
    except Exception as e:
        print(f"✗ Fee caps analysis failed: {e}\n")

    # 5. Extreme scenarios
    try:
        extreme_results = test_extreme_scenarios()
        results['extreme_scenarios'] = extreme_results
        extreme_results.to_csv('results/extreme_scenarios.csv', index=False)
        print("✓ Extreme scenarios testing completed\n")
    except Exception as e:
        print(f"✗ Extreme scenarios testing failed: {e}\n")

    # 6. Optimal control comparison
    try:
        optimal_comparison = generate_optimal_control_comparison()
        results['optimal_comparison'] = optimal_comparison

        # Save as single-row DataFrame
        pd.DataFrame([optimal_comparison]).to_csv('results/optimal_comparison.csv', index=False)
        print("✓ Optimal control comparison completed\n")
    except Exception as e:
        print(f"✗ Optimal control comparison failed: {e}\n")

    print("=== Analysis Summary ===")
    print(f"Results saved to 'results/' directory")
    print(f"Completed analyses: {len(results)}")

    return results


if __name__ == "__main__":
    results = main()