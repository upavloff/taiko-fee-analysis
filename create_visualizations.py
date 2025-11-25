"""
Visualization Script for Taiko Fee Mechanism Analysis

Creates publication-quality charts and visualizations from the analysis results
to answer the key research questions about the fee mechanism.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")

# Custom color palette
COLORS = {
    'mu_zero': '#e74c3c',      # Red for μ=0
    'mu_positive': '#3498db',   # Blue for μ>0
    'baseline': '#2ecc71',      # Green for baseline
    'advanced': '#9b59b6',      # Purple for advanced
    'optimal': '#f39c12'        # Orange for optimal
}


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def create_mu_zero_analysis_plots(mu_zero_results: Dict[str, pd.DataFrame]) -> None:
    """Create visualizations for μ=0 viability analysis."""

    print("Creating μ=0 analysis plots...")

    # Create subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('μ=0 Viability Analysis Across L1 Scenarios', fontsize=16, y=0.98)

    scenarios = list(mu_zero_results.keys())
    metrics_to_plot = [
        ('avg_fee', 'Average Fee', 'Fee'),
        ('fee_cv', 'Fee Variability (CV)', 'Coefficient of Variation'),
        ('time_underfunded_pct', 'Time Underfunded (%)', 'Percentage'),
        ('max_deficit', 'Maximum Deficit', 'Deficit'),
        ('insolvency_probability', 'Insolvency Probability', 'Probability'),
        ('l1_tracking_error', 'L1 Tracking Error', 'Error')
    ]

    for idx, (metric, title, ylabel) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]

        # Collect data across scenarios
        scenario_data = []
        for scenario in scenarios:
            df = mu_zero_results[scenario]
            mu_zero_data = df[df['mu'] == 0.0][metric].values
            mu_positive_data = df[df['mu'] > 0][metric].values

            scenario_data.append({
                'scenario': scenario.replace('(μ=0.00, σ=0.30)', '').replace('GBM', '').strip(),
                'mu_zero': np.mean(mu_zero_data) if len(mu_zero_data) > 0 else np.nan,
                'mu_positive': np.mean(mu_positive_data) if len(mu_positive_data) > 0 else np.nan
            })

        plot_df = pd.DataFrame(scenario_data)

        # Bar plot comparing μ=0 vs μ>0
        x = np.arange(len(scenarios))
        width = 0.35

        ax.bar(x - width/2, plot_df['mu_zero'], width, label='μ=0',
               color=COLORS['mu_zero'], alpha=0.8)
        ax.bar(x + width/2, plot_df['mu_positive'], width, label='μ>0',
               color=COLORS['mu_positive'], alpha=0.8)

        ax.set_xlabel('L1 Scenario')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([s.split('(')[0] for s in scenarios], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/mu_zero_viability_analysis.png')
    plt.close()

    # Create heatmap for μ vs ν parameter space
    print("Creating μ vs ν parameter heatmap...")

    # Use first scenario for detailed parameter analysis
    main_scenario = scenarios[0]
    df = mu_zero_results[main_scenario]

    # Create pivot table for heatmap
    pivot_data = df.pivot(index='nu', columns='mu', values='fee_cv')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=pivot_data.median().median(), ax=ax)
    ax.set_title(f'Fee Variability (CV) Heatmap: μ vs ν\nScenario: {main_scenario}')
    ax.set_xlabel('μ (L1 Cost Weight)')
    ax.set_ylabel('ν (Deficit Correction Weight)')

    plt.tight_layout()
    plt.savefig('results/mu_nu_heatmap.png')
    plt.close()


def create_parameter_sweep_visualization(param_results: pd.DataFrame) -> None:
    """Create visualizations for parameter sweep results."""

    print("Creating parameter sweep visualizations...")

    # Key metrics to visualize
    key_metrics = ['avg_fee', 'fee_cv', 'time_underfunded_pct', 'max_deficit']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Parameter Sweep Analysis', fontsize=16)

    for idx, metric in enumerate(key_metrics):
        ax = axes[idx // 2, idx % 2]

        # Create scatter plot with μ on x-axis, colored by ν
        scatter = ax.scatter(param_results['mu'], param_results[metric],
                           c=param_results['nu'], s=60, alpha=0.7,
                           cmap='viridis')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('ν (Deficit Weight)')

        # Highlight μ=0 points
        mu_zero_data = param_results[param_results['mu'] == 0.0]
        ax.scatter(mu_zero_data['mu'], mu_zero_data[metric],
                  s=100, facecolors='none', edgecolors='red', linewidth=2,
                  label='μ=0')

        ax.set_xlabel('μ (L1 Cost Weight)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Parameters')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('results/parameter_sweep_analysis.png')
    plt.close()

    # Create box plots for μ=0 vs others
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('μ=0 vs μ>0 Performance Comparison', fontsize=16)

    param_results['mu_category'] = param_results['mu'].apply(
        lambda x: 'μ=0' if x == 0.0 else 'μ>0'
    )

    for idx, metric in enumerate(key_metrics):
        ax = axes[idx // 2, idx % 2]

        sns.boxplot(data=param_results, x='mu_category', y=metric, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/mu_zero_comparison_boxplots.png')
    plt.close()


def create_time_series_plots():
    """Create example time series plots showing mechanism dynamics."""

    print("Creating time series examples...")

    from fee_mechanism_simulator import (
        TaikoFeeSimulator, SimulationParams,
        GeometricBrownianMotion, SpikeEventsModel
    )

    # Set up scenarios for time series plots
    scenarios = [
        ('Baseline L1 Dynamics', GeometricBrownianMotion(mu=0.0, sigma=0.3)),
        ('L1 with Spike Events', SpikeEventsModel(
            GeometricBrownianMotion(mu=0.0, sigma=0.2),
            spike_probability=0.005,
            spike_magnitude_range=(3.0, 8.0)
        ))
    ]

    for scenario_name, l1_model in scenarios:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Mechanism Dynamics: {scenario_name}', fontsize=16)

        # Compare μ=0 vs μ=0.5
        mu_values = [0.0, 0.5]
        colors = [COLORS['mu_zero'], COLORS['mu_positive']]

        for col, (mu, color) in enumerate(zip(mu_values, colors)):
            params = SimulationParams(
                mu=mu, nu=0.3, H=144, target_balance=1000,
                total_steps=1000, fee_elasticity=0.2
            )

            simulator = TaikoFeeSimulator(params, l1_model)
            df = simulator.run_simulation()

            # Convert time steps to hours
            time_hours = df['time_step'] * 12 / 3600

            # Plot L1 basefee
            axes[0, col].plot(time_hours, df['l1_basefee'] / 1e9,
                             color='gray', alpha=0.7, label='L1 Basefee')
            axes[0, col].set_ylabel('L1 Basefee (gwei)')
            axes[0, col].set_title(f'μ={mu}')
            axes[0, col].grid(True, alpha=0.3)

            # Plot estimated fees
            axes[1, col].plot(time_hours, df['estimated_fee'],
                             color=color, linewidth=2, label='Estimated Fee')
            axes[1, col].set_ylabel('Estimated Fee (ETH)')
            axes[1, col].grid(True, alpha=0.3)

            # Plot vault balance and deficit
            ax_balance = axes[2, col]
            ax_deficit = ax_balance.twinx()

            ax_balance.plot(time_hours, df['vault_balance'],
                           color='blue', linewidth=2, label='Vault Balance')
            ax_balance.axhline(y=1000, color='blue', linestyle='--', alpha=0.7, label='Target')

            ax_deficit.plot(time_hours, df['vault_deficit'],
                           color='red', linewidth=1, alpha=0.8, label='Deficit')
            ax_deficit.axhline(y=0, color='red', linestyle='-', alpha=0.5)

            ax_balance.set_ylabel('Vault Balance (ETH)', color='blue')
            ax_deficit.set_ylabel('Vault Deficit (ETH)', color='red')
            ax_balance.set_xlabel('Time (hours)')

            ax_balance.grid(True, alpha=0.3)
            ax_balance.legend(loc='upper left')
            ax_deficit.legend(loc='upper right')

        plt.tight_layout()
        safe_name = scenario_name.replace(' ', '_').lower()
        plt.savefig(f'results/time_series_{safe_name}.png')
        plt.close()


def create_fee_cap_analysis_plots(fee_cap_results: pd.DataFrame) -> None:
    """Create visualizations for fee cap analysis."""

    print("Creating fee cap analysis plots...")

    metrics = ['avg_fee', 'fee_95th_percentile', 'time_underfunded_pct', 'max_deficit']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Impact of Fee Caps on Mechanism Performance', fontsize=16)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Bar plot
        bars = ax.bar(fee_cap_results['cap_label'], fee_cap_results[metric],
                     color=sns.color_palette("viridis", len(fee_cap_results)),
                     alpha=0.8)

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Fee Cap Level')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('results/fee_cap_analysis.png')
    plt.close()


def create_extreme_scenarios_plots(extreme_results: pd.DataFrame) -> None:
    """Create visualizations for extreme scenario analysis."""

    print("Creating extreme scenarios plots...")

    # Radar chart for extreme scenarios
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    metrics = ['avg_fee', 'fee_cv', 'time_underfunded_pct', 'max_deficit',
              'insolvency_probability']

    # Normalize metrics to 0-1 scale for radar chart
    norm_data = extreme_results[metrics].copy()
    for col in metrics:
        norm_data[col] = (norm_data[col] - norm_data[col].min()) / (norm_data[col].max() - norm_data[col].min())

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle

    colors = sns.color_palette("husl", len(extreme_results))

    for idx, scenario in enumerate(extreme_results['scenario']):
        values = norm_data.iloc[idx].values
        values = np.concatenate((values, [values[0]]))  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=scenario, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title('Extreme Scenarios Performance Radar Chart', size=16, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('results/extreme_scenarios_radar.png')
    plt.close()


def create_summary_dashboard(all_results: Dict) -> None:
    """Create a summary dashboard with key insights."""

    print("Creating summary dashboard...")

    fig = plt.figure(figsize=(20, 16))

    # Create grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Taiko Fee Mechanism Analysis - Summary Dashboard', fontsize=20, y=0.98)

    # 1. μ=0 viability summary (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    if 'mu_zero_analysis' in all_results:
        # Aggregate μ=0 performance across scenarios
        scenario_performance = []
        for scenario, df in all_results['mu_zero_analysis'].items():
            mu_zero_avg = df[df['mu'] == 0.0]['fee_cv'].mean()
            mu_pos_avg = df[df['mu'] > 0]['fee_cv'].mean()
            ratio = mu_zero_avg / mu_pos_avg if mu_pos_avg > 0 else 1
            scenario_performance.append(ratio)

        scenarios = list(all_results['mu_zero_analysis'].keys())
        bars = ax1.bar(range(len(scenarios)), scenario_performance,
                      color=COLORS['mu_zero'], alpha=0.8)
        ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
        ax1.set_ylabel('μ=0 Fee CV / μ>0 Fee CV')
        ax1.set_title('μ=0 Viability: Fee Variability Ratio')
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels([s.split('(')[0][:10] for s in scenarios], rotation=45)

        # Add text annotations
        for bar, ratio in zip(bars, scenario_performance):
            color = 'green' if ratio < 1.2 else 'red'
            ax1.text(bar.get_x() + bar.get_width()/2., ratio + 0.05,
                    f'{ratio:.2f}', ha='center', va='bottom', color=color, weight='bold')

    # 2. Parameter sensitivity (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    if 'parameter_sweeps' in all_results:
        param_df = all_results['parameter_sweeps']
        # Show how metrics change with μ
        for metric, color in zip(['fee_cv', 'time_underfunded_pct'], ['blue', 'red']):
            ax2_twin = ax2.twinx() if metric == 'time_underfunded_pct' else ax2

            mu_grouped = param_df.groupby('mu')[metric].mean()
            ax2_twin.plot(mu_grouped.index, mu_grouped.values,
                         'o-', color=color, linewidth=2, markersize=6, label=metric)
            ax2_twin.set_ylabel(metric.replace('_', ' ').title(), color=color)

        ax2.set_xlabel('μ (L1 Cost Weight)')
        ax2.set_title('Parameter Sensitivity')
        ax2.grid(True, alpha=0.3)

    # 3. Fee cap effectiveness (middle left)
    ax3 = fig.add_subplot(gs[1, :2])
    if 'fee_caps' in all_results:
        cap_df = all_results['fee_caps']
        cap_labels = cap_df['cap_label']
        effectiveness = cap_df['avg_fee'] / cap_df['time_underfunded_pct'].replace(0, 0.1)  # Fee efficiency

        bars = ax3.bar(cap_labels, effectiveness, color=sns.color_palette("viridis", len(cap_df)))
        ax3.set_ylabel('Fee Efficiency (Fee/Underfunding)')
        ax3.set_title('Fee Cap Effectiveness')
        ax3.tick_params(axis='x', rotation=45)

    # 4. Extreme scenario robustness (middle right)
    ax4 = fig.add_subplot(gs[1, 2:])
    if 'extreme_scenarios' in all_results:
        extreme_df = all_results['extreme_scenarios']
        scenarios = extreme_df['scenario']
        robustness_score = (1 / extreme_df['fee_cv']) * (1 - extreme_df['insolvency_probability'])

        bars = ax4.bar(scenarios, robustness_score, color=COLORS['baseline'], alpha=0.8)
        ax4.set_ylabel('Robustness Score')
        ax4.set_title('Extreme Scenario Robustness')
        ax4.tick_params(axis='x', rotation=45)

    # 5. Key recommendations (bottom)
    ax5 = fig.add_subplot(gs[2:, :])
    ax5.axis('off')

    # Generate recommendations based on results
    recommendations = generate_recommendations(all_results)

    ax5.text(0.05, 0.95, 'Key Recommendations:', fontsize=16, weight='bold',
            transform=ax5.transAxes, va='top')

    for i, rec in enumerate(recommendations):
        ax5.text(0.05, 0.85 - i*0.12, f"{i+1}. {rec}", fontsize=12,
                transform=ax5.transAxes, va='top', wrap=True)

    plt.tight_layout()
    plt.savefig('results/summary_dashboard.png')
    plt.close()


def generate_recommendations(all_results: Dict) -> List[str]:
    """Generate key recommendations based on analysis results."""

    recommendations = []

    # Analyze μ=0 viability
    if 'mu_zero_analysis' in all_results:
        mu_zero_viable = True
        for scenario, df in all_results['mu_zero_analysis'].items():
            mu_zero_performance = df[df['mu'] == 0.0]
            if not mu_zero_performance.empty:
                if mu_zero_performance['fee_cv'].mean() > 0.5:  # High variability threshold
                    mu_zero_viable = False
                    break

        if mu_zero_viable:
            recommendations.append(
                "μ=0 is viable: Pure deficit-based control provides adequate stability "
                "across most L1 scenarios while simplifying the mechanism"
            )
        else:
            recommendations.append(
                "μ=0 shows instability: Include L1 cost tracking (μ>0) for better "
                "responsiveness to rapid L1 fee changes"
            )

    # Parameter recommendations
    if 'parameter_sweeps' in all_results:
        param_df = all_results['parameter_sweeps']
        best_params = param_df.loc[param_df['fee_cv'].idxmin()]
        recommendations.append(
            f"Optimal parameters: μ={best_params['mu']:.2f}, ν={best_params['nu']:.2f}, "
            f"H={int(best_params['H'])} steps for best fee stability"
        )

    # Fee cap recommendations
    if 'fee_caps' in all_results:
        cap_df = all_results['fee_caps']
        best_cap = cap_df.loc[cap_df['time_underfunded_pct'].idxmin()]
        recommendations.append(
            f"Fee cap recommendation: {best_cap['cap_label']} provides good balance "
            "between user protection and vault solvency"
        )

    # System robustness
    if 'extreme_scenarios' in all_results:
        recommendations.append(
            "Consider dynamic target adjustment and predictive L1 cost estimation "
            "for improved performance under extreme market conditions"
        )

    # General mechanism design
    recommendations.append(
        "Implement gradual fee adjustments and circuit breakers to prevent "
        "user experience degradation during market volatility"
    )

    return recommendations[:5]  # Limit to top 5 recommendations


def main():
    """Create all visualizations from analysis results."""

    setup_plot_style()

    # Create output directory
    os.makedirs('results', exist_ok=True)

    print("Loading analysis results...")

    # Load results
    all_results = {}

    try:
        # Load parameter sweeps
        if os.path.exists('results/parameter_sweeps.csv'):
            all_results['parameter_sweeps'] = pd.read_csv('results/parameter_sweeps.csv')

        # Load μ=0 analysis results
        mu_zero_results = {}
        for file in os.listdir('results'):
            if file.startswith('mu_zero_') and file.endswith('.csv'):
                scenario_name = file.replace('mu_zero_', '').replace('.csv', '').replace('_', ' ')
                mu_zero_results[scenario_name] = pd.read_csv(f'results/{file}')

        if mu_zero_results:
            all_results['mu_zero_analysis'] = mu_zero_results

        # Load other results
        for result_file, key in [
            ('fee_caps_analysis.csv', 'fee_caps'),
            ('extreme_scenarios.csv', 'extreme_scenarios'),
            ('optimal_comparison.csv', 'optimal_comparison')
        ]:
            if os.path.exists(f'results/{result_file}'):
                all_results[key] = pd.read_csv(f'results/{result_file}')

        print(f"Loaded {len(all_results)} result sets")

        # Create visualizations
        if 'mu_zero_analysis' in all_results:
            create_mu_zero_analysis_plots(all_results['mu_zero_analysis'])

        if 'parameter_sweeps' in all_results:
            create_parameter_sweep_visualization(all_results['parameter_sweeps'])

        create_time_series_plots()

        if 'fee_caps' in all_results:
            create_fee_cap_analysis_plots(all_results['fee_caps'])

        if 'extreme_scenarios' in all_results:
            create_extreme_scenarios_plots(all_results['extreme_scenarios'])

        # Create summary dashboard
        create_summary_dashboard(all_results)

        print("✓ All visualizations created successfully!")
        print("Check the 'results/' directory for generated plots.")

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()