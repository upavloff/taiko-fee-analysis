"""
Improved Fee Mechanism Simulator with Proper Vault Initialization

This module fixes the fee visualization issues and provides better
analysis of fee mechanism behavior with realistic starting conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from dataclasses import dataclass, replace
import seaborn as sns

# Import from same directory (will work when sys.path includes this dir)
from fee_mechanism_simulator import TaikoFeeSimulator, SimulationParams, FeeVault


@dataclass
class ImprovedSimulationParams(SimulationParams):
    """Enhanced simulation parameters with vault initialization options."""

    # Vault initialization
    initial_vault_balance: Optional[float] = None  # If None, uses target_balance
    vault_initialization_mode: str = "target"      # "target", "deficit", "surplus", "custom"
    initial_deficit_ratio: float = 0.0             # For deficit mode: deficit as % of target

    # Enhanced L1 cost estimation
    l1_cost_estimation_window: int = 20             # EWMA window size
    l1_cost_outlier_threshold: float = 3.0         # Sigma threshold for outlier rejection

    # Fee bounds
    min_estimated_fee: float = 1e-8                # Minimum fee (prevents zero/negative)
    max_fee_change_per_step: float = 0.5           # Max 50% change per step

    def __post_init__(self):
        """Set initial vault balance based on mode."""
        if self.initial_vault_balance is None:
            if self.vault_initialization_mode == "target":
                self.initial_vault_balance = self.target_balance
            elif self.vault_initialization_mode == "deficit":
                deficit = self.target_balance * self.initial_deficit_ratio
                self.initial_vault_balance = self.target_balance - deficit
            elif self.vault_initialization_mode == "surplus":
                surplus = self.target_balance * self.initial_deficit_ratio
                self.initial_vault_balance = self.target_balance + surplus
            else:  # custom - use provided value
                self.initial_vault_balance = self.target_balance


class ImprovedFeeVault(FeeVault):
    """Enhanced vault with better initialization and tracking."""

    def __init__(self, initial_balance: float, target_balance: float):
        super().__init__(initial_balance, target_balance)

        # Track initialization for analysis
        self.initial_balance = initial_balance
        self.initial_deficit = target_balance - initial_balance

        # Enhanced history tracking
        self.history['deficit'] = [self.deficit]
        self.history['deficit_ratio'] = [self.deficit / target_balance]

    def collect_fees(self, amount: float):
        """Enhanced fee collection with history tracking."""
        super().collect_fees(amount)
        self.history['deficit'].append(self.deficit)
        self.history['deficit_ratio'].append(self.deficit / self.target)

    def pay_l1_costs(self, amount: float):
        """Enhanced cost payment with history tracking."""
        super().pay_l1_costs(amount)
        self.history['deficit'].append(self.deficit)
        self.history['deficit_ratio'].append(self.deficit / self.target)


class ImprovedTaikoFeeSimulator(TaikoFeeSimulator):
    """Enhanced simulator with better initialization and fee smoothing."""

    def __init__(self, params: ImprovedSimulationParams, l1_model):
        # Store improved params first
        self.improved_params = params

        # Convert to base params for parent class
        base_params = SimulationParams(**{k: v for k, v in params.__dict__.items()
                                        if k in SimulationParams.__dataclass_fields__})

        # Initialize parent without calling reset_state (we'll do it ourselves)
        self.params = base_params
        self.l1_model = l1_model
        self.vault = FeeVault(base_params.target_balance, base_params.target_balance)

        # Now call our custom reset
        self.reset_state()

    def estimate_l1_cost_per_tx(self, l1_basefee: float) -> float:
        """Enhanced L1 cost estimation with outlier rejection."""

        # Calculate raw cost
        gas_cost_per_tx = self.params.gas_per_batch / self.params.txs_per_batch
        raw_cost = l1_basefee * gas_cost_per_tx / 1e18

        # Add to history
        self.l1_cost_history.append(raw_cost)

        # Keep only recent history
        window = self.improved_params.l1_cost_estimation_window
        if len(self.l1_cost_history) > window:
            self.l1_cost_history = self.l1_cost_history[-window:]

        # Outlier rejection if we have enough history
        if len(self.l1_cost_history) >= 5:
            costs_array = np.array(self.l1_cost_history)
            mean_cost = np.mean(costs_array)
            std_cost = np.std(costs_array)
            threshold = self.improved_params.l1_cost_outlier_threshold

            # Reject if too far from mean
            if abs(raw_cost - mean_cost) > threshold * std_cost:
                # Use previous EWMA value instead of outlier
                raw_cost = self.l1_cost_ewma

        # Update EWMA
        self.l1_cost_ewma = (1 - self.ewma_alpha) * self.l1_cost_ewma + self.ewma_alpha * raw_cost
        return self.l1_cost_ewma

    def calculate_estimated_fee(self, l1_cost_estimate: float) -> float:
        """Enhanced fee calculation with smoothing and bounds."""

        # Standard fee calculation
        l1_component = self.params.mu * l1_cost_estimate
        deficit = self.vault.deficit
        deficit_component = self.params.nu * deficit / self.params.H

        estimated_fee = l1_component + deficit_component

        # Apply fee caps
        if self.params.fee_cap is not None:
            estimated_fee = min(estimated_fee, self.params.fee_cap)

        # Apply minimum fee
        estimated_fee = max(estimated_fee, self.improved_params.min_estimated_fee)

        # Smooth large fee changes
        if self.previous_estimated_fee is not None:
            max_change = self.improved_params.max_fee_change_per_step
            fee_change_ratio = abs(estimated_fee - self.previous_estimated_fee) / self.previous_estimated_fee

            if fee_change_ratio > max_change:
                # Limit the change
                direction = 1 if estimated_fee > self.previous_estimated_fee else -1
                estimated_fee = self.previous_estimated_fee * (1 + direction * max_change)

        self.previous_estimated_fee = estimated_fee
        return estimated_fee

    def reset_state(self):
        """Enhanced state reset."""
        super().reset_state()

        # Reset improved components
        self.vault = ImprovedFeeVault(
            self.improved_params.initial_vault_balance,
            self.improved_params.target_balance
        )
        self.l1_cost_history = []
        self.previous_estimated_fee = None


# ImprovedAnalyzer class temporarily removed to fix import issues
# TODO: Move to separate analysis module to avoid circular dependencies

# class ImprovedAnalyzer:
#     """Enhanced analyzer for better fee mechanism analysis."""
#
#     def __init__(self):
#         from ..data.rpc_data_fetcher import ImprovedRealDataIntegrator
#         from ..analysis.mechanism_metrics import MetricsCalculator
#         self.data_integrator = ImprovedRealDataIntegrator()
#
#     def analyze_vault_initialization_impact(self, l1_basefees: np.ndarray) -> Dict:
#        """Analyze how different vault initializations affect performance."""
#
#        print("Analyzing vault initialization impact...")
#
#        initialization_modes = [
#            ("Fully Funded (Target)", "target", 0.0),
#            ("20% Underfunded", "deficit", 0.2),
#            ("50% Underfunded", "deficit", 0.5),
#            ("20% Overfunded", "surplus", 0.2),
#        ]
#
#        results = {}
#
#        for mode_name, mode_type, ratio in initialization_modes:
#            print(f"  Testing {mode_name}...")
#
#            params = ImprovedSimulationParams(
#                mu=0.5, nu=0.3, H=144,
#                target_balance=1000,
#                vault_initialization_mode=mode_type,
#                initial_deficit_ratio=ratio,
#                total_steps=len(l1_basefees)
#            )
#
#            # Create simple L1 model
#            class SimpleL1Model:
#                def __init__(self, sequence):
#                    self.sequence = sequence
#
#                def generate_sequence(self, steps, initial_basefee=None):
#                    return self.sequence[:steps] if steps <= len(self.sequence) else \
#                           np.tile(self.sequence, (steps // len(self.sequence)) + 1)[:steps]
#
#                def get_name(self):
#                    return "Test Data"
#
#            l1_model = SimpleL1Model(l1_basefees)
#            simulator = ImprovedTaikoFeeSimulator(params, l1_model)
#            df = simulator.run_simulation()
#
#            # Calculate metrics
#            metrics_calc = MetricsCalculator(1000)
#            metrics = metrics_calc.calculate_all_metrics(df)
#
#            results[mode_name] = {
#                'dataframe': df,
#                'metrics': metrics,
#                'initial_balance': params.initial_vault_balance,
#                'initial_deficit': params.target_balance - params.initial_vault_balance
#            }
#
#        return results
#
#    def create_improved_visualization(self, results: Dict, title: str = "Vault Initialization Impact"):
#        """Create improved visualization focusing on realistic fee ranges."""
#
#        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#        fig.suptitle(title, fontsize=16)
#
#        # Define colors for each initialization mode
#        colors = ['blue', 'orange', 'red', 'green']
#
#        # Plot 1: Estimated Fees (with better y-axis scaling)
#        ax = axes[0, 0]
#        fee_data_for_scaling = []
#
#        for (mode_name, data), color in zip(results.items(), colors):
#            df = data['dataframe']
#            time_hours = df['time_step'] * 12 / 3600
#
#            ax.plot(time_hours, df['estimated_fee'], label=mode_name,
#                   alpha=0.8, linewidth=1.5, color=color)
#
#            # Collect fees for scaling (exclude extreme outliers)
#            fees_clean = df['estimated_fee'][df['estimated_fee'] < df['estimated_fee'].quantile(0.95)]
#            fee_data_for_scaling.extend(fees_clean)
#
#        ax.set_title('Estimated Fees Over Time')
#        ax.set_xlabel('Time (hours)')
#        ax.set_ylabel('Estimated Fee (ETH)')
#        ax.legend()
#        ax.grid(True, alpha=0.3)
#
#        # Set y-axis to show meaningful range (excluding extreme outliers)
#        if fee_data_for_scaling:
#            y_max = np.percentile(fee_data_for_scaling, 99)  # 99th percentile
#            y_min = np.percentile(fee_data_for_scaling, 1)   # 1st percentile
#            ax.set_ylim(max(0, y_min * 0.9), y_max * 1.1)
#
#        # Plot 2: Vault Balance
#        ax = axes[0, 1]
#        for (mode_name, data), color in zip(results.items(), colors):
#            df = data['dataframe']
#            time_hours = df['time_step'] * 12 / 3600
#            ax.plot(time_hours, df['vault_balance'], label=mode_name,
#                   alpha=0.8, linewidth=1.5, color=color)
#
#        ax.axhline(y=1000, color='black', linestyle='--', alpha=0.5, label='Target')
#        ax.set_title('Vault Balance Evolution')
#        ax.set_xlabel('Time (hours)')
#        ax.set_ylabel('Vault Balance (ETH)')
#        ax.legend()
#        ax.grid(True, alpha=0.3)
#
#        # Plot 3: Vault Deficit
#        ax = axes[0, 2]
#        for (mode_name, data), color in zip(results.items(), colors):
#            df = data['dataframe']
#            time_hours = df['time_step'] * 12 / 3600
#            ax.plot(time_hours, df['vault_deficit'], label=mode_name,
#                   alpha=0.8, linewidth=1.5, color=color)
#
#        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
#        ax.set_title('Vault Deficit Over Time')
#        ax.set_xlabel('Time (hours)')
#        ax.set_ylabel('Deficit (ETH)')
#        ax.legend()
#        ax.grid(True, alpha=0.3)
#
#        # Plot 4: Fee Distribution (Log scale for better visibility)
#        ax = axes[1, 0]
#        for (mode_name, data), color in zip(results.items(), colors):
#            df = data['dataframe']
#            # Remove extreme outliers for visualization
#            fees_clean = df['estimated_fee'][df['estimated_fee'] < df['estimated_fee'].quantile(0.99)]
#            ax.hist(fees_clean, bins=30, alpha=0.5, label=mode_name,
#                   color=color, density=True)
#
#        ax.set_title('Fee Distribution (99th percentile)')
#        ax.set_xlabel('Estimated Fee (ETH)')
#        ax.set_ylabel('Density')
#        ax.set_yscale('log')
#        ax.legend()
#        ax.grid(True, alpha=0.3)
#
#        # Plot 5: Performance Metrics Table
#        ax = axes[1, 1]
#        ax.axis('off')
#
#        # Prepare metrics table
#        metrics_data = []
#        for mode_name, data in results.items():
#            metrics = data['metrics']
#            metrics_data.append([
#                mode_name[:15],  # Truncate long names
#                f"{metrics.avg_fee:.2e}",
#                f"{metrics.fee_cv:.3f}",
#                f"{metrics.time_underfunded_pct:.1f}%",
#                f"{metrics.max_deficit:.0f}"
#            ])
#
#        table = ax.table(cellText=metrics_data,
#                        colLabels=['Init Mode', 'Avg Fee', 'Fee CV', 'Underfunded%', 'Max Deficit'],
#                        cellLoc='center', loc='center')
#        table.auto_set_font_size(False)
#        table.set_fontsize(9)
#        table.scale(1.2, 2.0)
#        ax.set_title('Performance Metrics Comparison')
#
#        # Plot 6: L1 Basefee Reference
#        ax = axes[1, 2]
#        # Use L1 data from first result
#        first_df = list(results.values())[0]['dataframe']
#        time_hours = first_df['time_step'] * 12 / 3600
#        ax.plot(time_hours, first_df['l1_basefee'] / 1e9, color='gray', alpha=0.7)
#        ax.set_title('L1 Basefee Reference')
#        ax.set_xlabel('Time (hours)')
#        ax.set_ylabel('L1 Basefee (gwei)')
#        ax.grid(True, alpha=0.3)
#
#        plt.tight_layout()
#
#        # Save plot
#        filename = f"results/improved_vault_initialization_analysis.png"
#        plt.savefig(filename, dpi=300, bbox_inches='tight')
#        print(f"Improved visualization saved to {filename}")
#
#        return fig
#
#
#def test_with_real_data():
#    """Test improved simulator with real RPC data."""
#
#    print("Testing improved simulator with real data...")
#
#    try:
#        # Get recent real data (last 3 days)
#        from datetime import datetime, timedelta
#
#        end_date = datetime.now()
#        start_date = end_date - timedelta(days=3)
#
#        integrator = ImprovedRealDataIntegrator()
#        df = integrator.get_real_basefee_data(
#            start_date.strftime('%Y-%m-%d'),
#            end_date.strftime('%Y-%m-%d'),
#            provider='ethereum_public'
#        )
#
#        print(f"Fetched {len(df)} real basefee records")
#        print(f"Basefee range: {df['basefee_gwei'].min():.1f} - {df['basefee_gwei'].max():.1f} gwei")
#
#        # Analyze with different vault initializations
#        analyzer = ImprovedAnalyzer()
#        results = analyzer.analyze_vault_initialization_impact(df['basefee_wei'].values[:300])
#
#        # Create visualization
#        analyzer.create_improved_visualization(results, "Real Data: Vault Initialization Impact")
#
#        print("\n=== ANALYSIS SUMMARY ===")
#        for mode_name, data in results.items():
#            metrics = data['metrics']
#            print(f"\n{mode_name}:")
#            print(f"  Initial Balance: {data['initial_balance']:.0f} ETH")
#            print(f"  Average Fee: {metrics.avg_fee:.2e} ETH")
#            print(f"  Fee Variability (CV): {metrics.fee_cv:.3f}")
#            print(f"  Time Underfunded: {metrics.time_underfunded_pct:.1f}%")
#            print(f"  Max Deficit: {metrics.max_deficit:.0f} ETH")
#
#    except Exception as e:
#        print(f"Real data test failed: {e}")
#        print("Falling back to synthetic data test...")
#
#        # Generate synthetic volatile data
#        np.random.seed(42)
#        synthetic_basefees = []
#
#        base_fee = 30e9  # 30 gwei
#        for i in range(500):
#            # Add random walk with occasional spikes
#            change = np.random.normal(0, 0.05)  # 5% volatility
#            if np.random.random() < 0.02:  # 2% chance of spike
#                change += np.random.uniform(1, 3)  # 100-300% spike
#
#            base_fee *= (1 + change)
#            base_fee = max(base_fee, 1e9)  # Minimum 1 gwei
#            synthetic_basefees.append(base_fee)
#
#        print(f"Using {len(synthetic_basefees)} synthetic basefee points")
#
#        # Run analysis
#        analyzer = ImprovedAnalyzer()
#        results = analyzer.analyze_vault_initialization_impact(np.array(synthetic_basefees))
#
#        # Create visualization
#        analyzer.create_improved_visualization(results, "Synthetic Data: Vault Initialization Impact")
#
#
#def main():
#    """Run improved simulator tests."""
#    test_with_real_data()
#
#
#if __name__ == "__main__":
#    main()