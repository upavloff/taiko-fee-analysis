"""
Main Execution Script for Taiko Fee Mechanism Analysis

Run this script to execute the complete analysis suite and generate
visualizations for the Taiko fee mechanism research.

Usage:
    python main.py [--quick] [--visualize-only]

Options:
    --quick: Run a faster analysis with reduced parameters
    --visualize-only: Only generate visualizations from existing results
"""

import sys
import argparse
import time
from datetime import datetime

def run_full_analysis(quick_mode=False):
    """Run the complete analysis suite."""

    print(f"Starting Taiko Fee Mechanism Analysis at {datetime.now()}")
    print("=" * 60)

    start_time = time.time()

    try:
        # Import and run analysis
        print("Importing analysis modules...")
        from run_analysis import main as run_analysis_main

        if quick_mode:
            print("Running in QUICK MODE (reduced parameters)...")
            # Modify global parameters for quicker analysis
            import fee_mechanism_simulator
            # Reduce simulation steps for quick mode
            original_steps = 2000
            fee_mechanism_simulator.SimulationParams.__dataclass_fields__['total_steps'].default = 500
            print(f"Reduced simulation steps from {original_steps} to 500")

        print("\nRunning comprehensive analysis...")
        results = run_analysis_main()

        analysis_time = time.time() - start_time
        print(f"\nAnalysis completed in {analysis_time:.1f} seconds")

        return True

    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_visualizations():
    """Generate all visualizations."""

    print("\nGenerating visualizations...")
    viz_start_time = time.time()

    try:
        from create_visualizations import main as create_viz_main
        create_viz_main()

        viz_time = time.time() - viz_start_time
        print(f"Visualizations completed in {viz_time:.1f} seconds")

        return True

    except Exception as e:
        print(f"Visualization generation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_example_notebook():
    """Create an example Jupyter notebook for interactive analysis."""

    notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Taiko Fee Mechanism Analysis\\n",
    "\\n",
    "This notebook provides an interactive interface for exploring the Taiko fee mechanism analysis.\\n",
    "\\n",
    "## Key Research Questions:\\n",
    "1. Can we set μ=0 and rely only on deficit correction?\\n",
    "2. How do different L1 dynamics affect the mechanism?\\n",
    "3. What are optimal parameter ranges for (μ, ν, H)?\\n",
    "4. How do fee caps affect performance?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "# Import our analysis modules\\n",
    "from fee_mechanism_simulator import *\\n",
    "from mechanism_metrics import *\\n",
    "from advanced_mechanisms import *\\n",
    "\\n",
    "# Set up plotting\\n",
    "plt.style.use('default')\\n",
    "sns.set_style(\\"whitegrid\\")\\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Quick Start: Single Simulation Example"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create a simple example simulation\\n",
    "\\n",
    "# Parameters\\n",
    "params = SimulationParams(\\n",
    "    mu=0.5,  # Try changing this to 0.0 to test μ=0\\n",
    "    nu=0.3,\\n",
    "    H=144,\\n",
    "    target_balance=1000,\\n",
    "    total_steps=1000\\n",
    ")\\n",
    "\\n",
    "# L1 dynamics model\\n",
    "l1_model = GeometricBrownianMotion(mu=0.0, sigma=0.3)\\n",
    "\\n",
    "# Run simulation\\n",
    "simulator = TaikoFeeSimulator(params, l1_model)\\n",
    "df = simulator.run_simulation()\\n",
    "\\n",
    "print(f\\"Simulation completed with {len(df)} time steps\\")\\n",
    "print(f\\"Average estimated fee: {df['estimated_fee'].mean():.4f} ETH\\")\\n",
    "print(f\\"Final vault balance: {df['vault_balance'].iloc[-1]:.2f} ETH\\")\\n",
    "print(f\\"Target balance: {params.target_balance} ETH\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualize Simulation Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create time series plots\\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\\n",
    "\\n",
    "# Convert time steps to hours\\n",
    "time_hours = df['time_step'] * 12 / 3600\\n",
    "\\n",
    "# L1 basefee\\n",
    "axes[0,0].plot(time_hours, df['l1_basefee'] / 1e9)\\n",
    "axes[0,0].set_title('L1 Basefee')\\n",
    "axes[0,0].set_ylabel('Basefee (gwei)')\\n",
    "axes[0,0].grid(True, alpha=0.3)\\n",
    "\\n",
    "# Estimated fees\\n",
    "axes[0,1].plot(time_hours, df['estimated_fee'], color='orange')\\n",
    "axes[0,1].set_title('Estimated Fee')\\n",
    "axes[0,1].set_ylabel('Fee (ETH)')\\n",
    "axes[0,1].grid(True, alpha=0.3)\\n",
    "\\n",
    "# Vault balance\\n",
    "axes[1,0].plot(time_hours, df['vault_balance'], color='blue')\\n",
    "axes[1,0].axhline(y=params.target_balance, color='blue', linestyle='--', alpha=0.7)\\n",
    "axes[1,0].set_title('Vault Balance')\\n",
    "axes[1,0].set_ylabel('Balance (ETH)')\\n",
    "axes[1,0].set_xlabel('Time (hours)')\\n",
    "axes[1,0].grid(True, alpha=0.3)\\n",
    "\\n",
    "# Vault deficit\\n",
    "axes[1,1].plot(time_hours, df['vault_deficit'], color='red')\\n",
    "axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)\\n",
    "axes[1,1].set_title('Vault Deficit')\\n",
    "axes[1,1].set_ylabel('Deficit (ETH)')\\n",
    "axes[1,1].set_xlabel('Time (hours)')\\n",
    "axes[1,1].grid(True, alpha=0.3)\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Calculate Performance Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Calculate comprehensive metrics\\n",
    "metrics_calc = MetricsCalculator(target_balance=params.target_balance)\\n",
    "metrics = metrics_calc.calculate_all_metrics(df)\\n",
    "\\n",
    "# Display key metrics\\n",
    "print(\\"Performance Metrics:\\")\\n",
    "print(\\"-\\" * 30)\\n",
    "print(f\\"Average Fee: {metrics.avg_fee:.4f} ETH\\")\\n",
    "print(f\\"Fee Std Dev: {metrics.fee_std:.4f} ETH\\")\\n",
    "print(f\\"Fee CV: {metrics.fee_cv:.3f}\\")\\n",
    "print(f\\"95th Percentile Fee: {metrics.fee_95th_percentile:.4f} ETH\\")\\n",
    "print(f\\"Time Underfunded: {metrics.time_underfunded_pct:.1f}%\\")\\n",
    "print(f\\"Time Overfunded: {metrics.time_overfunded_pct:.1f}%\\")\\n",
    "print(f\\"Max Deficit: {metrics.max_deficit:.2f} ETH\\")\\n",
    "print(f\\"Insolvency Probability: {metrics.insolvency_probability:.3f}\\")\\n",
    "print(f\\"L1 Tracking Error: {metrics.l1_tracking_error:.3f}\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Compare μ=0 vs μ>0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Quick comparison of μ=0 vs μ=0.5\\n",
    "\\n",
    "mu_values = [0.0, 0.5]\\n",
    "comparison_results = []\\n",
    "\\n",
    "for mu in mu_values:\\n",
    "    params_test = SimulationParams(\\n",
    "        mu=mu, nu=0.3, H=144, target_balance=1000, total_steps=500\\n",
    "    )\\n",
    "    \\n",
    "    simulator_test = TaikoFeeSimulator(params_test, l1_model)\\n",
    "    df_test = simulator_test.run_simulation()\\n",
    "    \\n",
    "    metrics_test = metrics_calc.calculate_all_metrics(df_test)\\n",
    "    \\n",
    "    comparison_results.append({\\n",
    "        'mu': mu,\\n",
    "        'avg_fee': metrics_test.avg_fee,\\n",
    "        'fee_cv': metrics_test.fee_cv,\\n",
    "        'time_underfunded_pct': metrics_test.time_underfunded_pct,\\n",
    "        'max_deficit': metrics_test.max_deficit\\n",
    "    })\\n",
    "\\n",
    "comparison_df = pd.DataFrame(comparison_results)\\n",
    "print(\\"μ=0 vs μ>0 Comparison:\\")\\n",
    "print(comparison_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Test Different L1 Scenarios"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Test mechanism under different L1 dynamics\\n",
    "\\n",
    "l1_scenarios = [\\n",
    "    (\\"Low Volatility\\", GeometricBrownianMotion(mu=0.0, sigma=0.1)),\\n",
    "    (\\"Medium Volatility\\", GeometricBrownianMotion(mu=0.0, sigma=0.3)),\\n",
    "    (\\"High Volatility\\", GeometricBrownianMotion(mu=0.0, sigma=0.6)),\\n",
    "    (\\"Regime Switching\\", RegimeSwitchingModel()),\\n",
    "    (\\"Spike Events\\", SpikeEventsModel(\\n",
    "        GeometricBrownianMotion(mu=0.0, sigma=0.2), spike_probability=0.005\\n",
    "    ))\\n",
    "]\\n",
    "\\n",
    "scenario_results = []\\n",
    "\\n",
    "base_params = SimulationParams(mu=0.5, nu=0.3, H=144, target_balance=1000, total_steps=500)\\n",
    "\\n",
    "for scenario_name, l1_model_test in l1_scenarios:\\n",
    "    simulator_scenario = TaikoFeeSimulator(base_params, l1_model_test)\\n",
    "    df_scenario = simulator_scenario.run_simulation()\\n",
    "    metrics_scenario = metrics_calc.calculate_all_metrics(df_scenario)\\n",
    "    \\n",
    "    scenario_results.append({\\n",
    "        'scenario': scenario_name,\\n",
    "        'avg_fee': metrics_scenario.avg_fee,\\n",
    "        'fee_cv': metrics_scenario.fee_cv,\\n",
    "        'time_underfunded_pct': metrics_scenario.time_underfunded_pct\\n",
    "    })\\n",
    "\\n",
    "scenario_df = pd.DataFrame(scenario_results)\\n",
    "print(\\"L1 Scenario Comparison:\\")\\n",
    "print(scenario_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Interactive Parameter Explorer\\n",
    "\\n",
    "Use the widgets below to explore how different parameters affect the mechanism:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Install ipywidgets if not available: pip install ipywidgets\\n",
    "try:\\n",
    "    from ipywidgets import interact, FloatSlider, IntSlider\\n",
    "    \\n",
    "    def explore_parameters(mu=0.5, nu=0.3, H=144, elasticity=0.2):\\n",
    "        params_explore = SimulationParams(\\n",
    "            mu=mu, nu=nu, H=int(H), fee_elasticity=elasticity,\\n",
    "            target_balance=1000, total_steps=300  # Shorter for interactivity\\n",
    "        )\\n",
    "        \\n",
    "        l1_explore = GeometricBrownianMotion(mu=0.0, sigma=0.3)\\n",
    "        simulator_explore = TaikoFeeSimulator(params_explore, l1_explore)\\n",
    "        df_explore = simulator_explore.run_simulation()\\n",
    "        \\n",
    "        metrics_explore = metrics_calc.calculate_all_metrics(df_explore)\\n",
    "        \\n",
    "        print(f\\"μ={mu:.2f}, ν={nu:.2f}, H={int(H)}, elasticity={elasticity:.2f}\\")\\n",
    "        print(f\\"Average Fee: {metrics_explore.avg_fee:.4f} ETH\\")\\n",
    "        print(f\\"Fee CV: {metrics_explore.fee_cv:.3f}\\")\\n",
    "        print(f\\"Time Underfunded: {metrics_explore.time_underfunded_pct:.1f}%\\")\\n",
    "        \\n",
    "        # Quick plot\\n",
    "        plt.figure(figsize=(12, 4))\\n",
    "        time_hours = df_explore['time_step'] * 12 / 3600\\n",
    "        \\n",
    "        plt.subplot(1, 2, 1)\\n",
    "        plt.plot(time_hours, df_explore['estimated_fee'])\\n",
    "        plt.title('Estimated Fee')\\n",
    "        plt.ylabel('Fee (ETH)')\\n",
    "        plt.xlabel('Time (hours)')\\n",
    "        plt.grid(True, alpha=0.3)\\n",
    "        \\n",
    "        plt.subplot(1, 2, 2)\\n",
    "        plt.plot(time_hours, df_explore['vault_deficit'])\\n",
    "        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)\\n",
    "        plt.title('Vault Deficit')\\n",
    "        plt.ylabel('Deficit (ETH)')\\n",
    "        plt.xlabel('Time (hours)')\\n",
    "        plt.grid(True, alpha=0.3)\\n",
    "        \\n",
    "        plt.tight_layout()\\n",
    "        plt.show()\\n",
    "    \\n",
    "    # Create interactive widgets\\n",
    "    interact(explore_parameters,\\n",
    "             mu=FloatSlider(value=0.5, min=0.0, max=1.0, step=0.1, description='μ'),\\n",
    "             nu=FloatSlider(value=0.3, min=0.1, max=0.9, step=0.1, description='ν'),\\n",
    "             H=IntSlider(value=144, min=24, max=576, step=24, description='H'),\\n",
    "             elasticity=FloatSlider(value=0.2, min=0.0, max=1.0, step=0.1, description='Elasticity'))\\n",
    "             \\n",
    "except ImportError:\\n",
    "    print(\\"Install ipywidgets for interactive exploration: pip install ipywidgets\\")\\n",
    "    print(\\"Then restart the kernel and run this cell again.\\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''

    with open('taiko_fee_analysis.ipynb', 'w') as f:
        f.write(notebook_content)

    print("Created example Jupyter notebook: taiko_fee_analysis.ipynb")


def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(description='Taiko Fee Mechanism Analysis')
    parser.add_argument('--quick', action='store_true',
                       help='Run faster analysis with reduced parameters')
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only generate visualizations from existing results')
    parser.add_argument('--create-notebook', action='store_true',
                       help='Create example Jupyter notebook')

    args = parser.parse_args()

    if args.create_notebook:
        create_example_notebook()
        return

    total_start_time = time.time()

    # Run analysis unless visualize-only
    if not args.visualize_only:
        success = run_full_analysis(quick_mode=args.quick)
        if not success:
            print("Analysis failed. Exiting.")
            return

    # Generate visualizations
    viz_success = generate_visualizations()

    total_time = time.time() - total_start_time

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total execution time: {total_time:.1f} seconds")

    if not args.visualize_only:
        print("✓ Comprehensive parameter analysis completed")
    print("✓ Visualizations generated")

    print("\nResults saved in 'results/' directory:")
    print("- CSV files with detailed metrics")
    print("- PNG files with publication-quality charts")
    print("- Summary dashboard with key insights")

    print("\nNext steps:")
    print("1. Review results/summary_dashboard.png for key findings")
    print("2. Check results/mu_zero_viability_analysis.png for μ=0 analysis")
    print("3. Examine individual CSV files for detailed metrics")
    print("4. Run 'python main.py --create-notebook' for interactive exploration")


if __name__ == "__main__":
    main()