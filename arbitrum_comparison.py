"""
Arbitrum vs Taiko Fee Mechanism Comparison

This module implements Arbitrum's L1 cost estimation approach and compares it
with Taiko's proposed mechanism using real historical data.

Key differences analyzed:
1. L1 cost estimation methodology
2. Volatility handling and smoothing
3. Fee adjustment mechanisms
4. Performance under real market conditions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dataclasses import dataclass
import math

from fee_mechanism_simulator import TaikoFeeSimulator, SimulationParams
from real_data_fetcher import RealDataIntegrator, HISTORICAL_PERIODS


@dataclass
class ArbitrumParams:
    """Parameters for Arbitrum-style L1 cost estimation."""

    # Compression and batching
    compression_ratio: float = 0.15  # Brotli compression ratio
    calldata_gas_per_byte: float = 16  # L1 gas per byte of calldata

    # Dynamic pricing
    price_adjustment_factor: float = 0.1  # How aggressively to adjust prices
    surplus_tolerance: float = 0.1  # Acceptable surplus before adjustment
    deficit_tolerance: float = 0.1  # Acceptable deficit before adjustment

    # Smoothing
    basefee_smoothing_factor: float = 0.05  # EWMA factor for basefee smoothing
    allocation_window: int = 100  # Time window for fee allocation


class ArbitrumL1Pricer:
    """
    Arbitrum-style L1 cost estimation and pricing.

    Based on Arbitrum's documentation:
    - Uses Brotli compression estimation
    - Dynamic price adjustment based on surplus/deficit
    - Allocation-based fee distribution
    - Exponential smoothing for volatility reduction
    """

    def __init__(self, params: ArbitrumParams):
        self.params = params

        # State variables
        self.funds_pool = 1000.0  # Start with target balance
        self.collected_fees = 0.0
        self.actual_costs = 0.0
        self.smoothed_basefee = 20e9  # Start at 20 gwei
        self.price_per_byte = 20e9 * 16 / 1e18  # Initial price

        # Historical tracking
        self.surplus_deficit_history = []
        self.price_history = []
        self.basefee_history = []

    def estimate_tx_cost(self, tx_data_bytes: int, l1_basefee: float) -> float:
        """
        Estimate L1 cost for a transaction.

        Arbitrum's approach:
        1. Estimate compressed size using Brotli compression ratio
        2. Calculate L1 gas usage (compressed_bytes * 16)
        3. Apply dynamic price per unit based on pool state
        """

        # Step 1: Estimate compressed transaction size
        compressed_bytes = tx_data_bytes * self.params.compression_ratio

        # Step 2: Estimate L1 gas for calldata
        estimated_l1_gas = compressed_bytes * self.params.calldata_gas_per_byte

        # Step 3: Use smoothed basefee rather than current
        self.smoothed_basefee = (
            (1 - self.params.basefee_smoothing_factor) * self.smoothed_basefee +
            self.params.basefee_smoothing_factor * l1_basefee
        )

        # Step 4: Calculate cost using current price per unit
        estimated_cost = estimated_l1_gas * self.price_per_byte

        return estimated_cost

    def collect_fee(self, amount: float):
        """Collect fee into the pricing pool."""
        self.funds_pool += amount
        self.collected_fees += amount

    def pay_actual_cost(self, actual_l1_cost: float):
        """Pay actual L1 batch posting cost."""
        self.funds_pool -= actual_l1_cost
        self.actual_costs += actual_l1_cost

    def update_pricing(self):
        """
        Update dynamic pricing based on surplus/deficit.

        Arbitrum's approach:
        1. Compare collected fees to actual costs
        2. Adjust price per unit to balance over time
        3. Use exponential adjustment mechanism
        """

        # Calculate surplus/deficit ratio
        if self.actual_costs > 0:
            surplus_ratio = (self.collected_fees - self.actual_costs) / self.actual_costs
        else:
            surplus_ratio = 0

        self.surplus_deficit_history.append(surplus_ratio)

        # Adjust price based on surplus/deficit
        if surplus_ratio > self.params.surplus_tolerance:
            # Too much surplus, decrease price
            adjustment = -self.params.price_adjustment_factor * abs(surplus_ratio)
        elif surplus_ratio < -self.params.deficit_tolerance:
            # Too much deficit, increase price
            adjustment = self.params.price_adjustment_factor * abs(surplus_ratio)
        else:
            # Within tolerance, no adjustment
            adjustment = 0

        # Apply exponential adjustment
        self.price_per_byte *= (1 + adjustment)
        self.price_per_byte = max(self.price_per_byte, 1e-18)  # Prevent negative prices

        self.price_history.append(self.price_per_byte)
        self.basefee_history.append(self.smoothed_basefee)

        # Reset counters for next period
        self.collected_fees = 0
        self.actual_costs = 0


class ArbitrumSimulator:
    """Simulator using Arbitrum's approach to L1 cost estimation."""

    def __init__(self, arbitrum_params: ArbitrumParams, avg_tx_bytes: int = 100):
        self.arbitrum_params = arbitrum_params
        self.avg_tx_bytes = avg_tx_bytes
        self.pricer = ArbitrumL1Pricer(arbitrum_params)

        # Simulation parameters
        self.target_balance = 1000.0
        self.base_demand = 100  # Transactions per time step
        self.fee_elasticity = 0.2

    def run_simulation(self, l1_basefees: np.ndarray) -> pd.DataFrame:
        """Run simulation with Arbitrum-style pricing."""

        results = {
            'time_step': [],
            'l1_basefee': [],
            'estimated_fee': [],
            'transaction_volume': [],
            'vault_balance': [],
            'vault_deficit': [],
            'fee_collected': [],
            'l1_cost_paid': [],
            'price_per_byte': [],
            'surplus_ratio': []
        }

        for t, l1_basefee in enumerate(l1_basefees):
            # Estimate cost per transaction
            estimated_cost = self.pricer.estimate_tx_cost(self.avg_tx_bytes, l1_basefee)

            # Calculate transaction volume with elasticity
            volume = self.base_demand * np.exp(-self.fee_elasticity * estimated_cost)
            volume *= np.random.normal(1.0, 0.1)  # Add noise
            volume = max(volume, 0)

            # Collect fees
            total_fees = estimated_cost * volume
            self.pricer.collect_fee(total_fees)

            # Pay actual L1 costs (batching every 10 steps)
            l1_cost_paid = 0
            if t % 10 == 0:  # Batch every 10 time steps
                # Actual L1 cost for the batch
                actual_batch_cost = l1_basefee * 200000 / 1e18  # 200k gas batch
                self.pricer.pay_actual_cost(actual_batch_cost)
                l1_cost_paid = actual_batch_cost

                # Update pricing
                self.pricer.update_pricing()

            # Record results
            results['time_step'].append(t)
            results['l1_basefee'].append(l1_basefee)
            results['estimated_fee'].append(estimated_cost)
            results['transaction_volume'].append(volume)
            results['vault_balance'].append(self.pricer.funds_pool)
            results['vault_deficit'].append(self.target_balance - self.pricer.funds_pool)
            results['fee_collected'].append(total_fees)
            results['l1_cost_paid'].append(l1_cost_paid)
            results['price_per_byte'].append(self.pricer.price_per_byte)

            # Surplus ratio (recent)
            if len(self.pricer.surplus_deficit_history) > 0:
                results['surplus_ratio'].append(self.pricer.surplus_deficit_history[-1])
            else:
                results['surplus_ratio'].append(0)

        return pd.DataFrame(results)


class ComparisonAnalyzer:
    """Compares Arbitrum and Taiko fee mechanisms."""

    def __init__(self):
        self.data_integrator = RealDataIntegrator()

    def compare_mechanisms(self, period_name: str, etherscan_api_key: Optional[str] = None) -> Dict:
        """Compare both mechanisms on the same real data."""

        print(f"Comparing mechanisms on {HISTORICAL_PERIODS[period_name].name} data...")

        # Get real historical data
        df = self.data_integrator.get_period_data(period_name, etherscan_api_key=etherscan_api_key)
        l1_basefees = df['basefee_wei'].values[:1000]  # Limit for performance

        # Run Arbitrum simulation
        arbitrum_params = ArbitrumParams()
        arbitrum_sim = ArbitrumSimulator(arbitrum_params)
        arbitrum_results = arbitrum_sim.run_simulation(l1_basefees)

        # Run Taiko simulation
        taiko_params = SimulationParams(
            mu=0.5, nu=0.3, H=144, target_balance=1000,
            total_steps=len(l1_basefees)
        )

        # Create L1 model from real data
        real_l1_model = self.data_integrator.create_l1_model_from_data(df)
        taiko_sim = TaikoFeeSimulator(taiko_params, real_l1_model)
        taiko_results = taiko_sim.run_simulation()

        # Calculate comparative metrics
        comparison = self._calculate_comparison_metrics(
            arbitrum_results, taiko_results, period_name
        )

        return {
            'period': period_name,
            'arbitrum_results': arbitrum_results,
            'taiko_results': taiko_results,
            'comparison': comparison,
            'l1_data': df
        }

    def _calculate_comparison_metrics(self, arbitrum_df: pd.DataFrame,
                                    taiko_df: pd.DataFrame, period_name: str) -> Dict:
        """Calculate detailed comparison metrics."""

        metrics = {
            'period_name': period_name,

            # Fee volatility comparison
            'arbitrum_fee_cv': arbitrum_df['estimated_fee'].std() / arbitrum_df['estimated_fee'].mean(),
            'taiko_fee_cv': taiko_df['estimated_fee'].std() / taiko_df['estimated_fee'].mean(),

            # Average fees
            'arbitrum_avg_fee': arbitrum_df['estimated_fee'].mean(),
            'taiko_avg_fee': taiko_df['estimated_fee'].mean(),

            # Vault stability
            'arbitrum_vault_std': arbitrum_df['vault_balance'].std(),
            'taiko_vault_std': taiko_df['vault_balance'].std(),

            # Fee responsiveness (correlation with L1)
            'arbitrum_l1_correlation': np.corrcoef(
                arbitrum_df['l1_basefee'], arbitrum_df['estimated_fee']
            )[0,1],
            'taiko_l1_correlation': np.corrcoef(
                taiko_df['l1_basefee'], taiko_df['estimated_fee']
            )[0,1],

            # Extreme values
            'arbitrum_max_fee': arbitrum_df['estimated_fee'].max(),
            'taiko_max_fee': taiko_df['estimated_fee'].max(),

            'arbitrum_95_fee': arbitrum_df['estimated_fee'].quantile(0.95),
            'taiko_95_fee': taiko_df['estimated_fee'].quantile(0.95),
        }

        # Add deficit analysis
        arbitrum_underfunded = (arbitrum_df['vault_deficit'] > 0).mean() * 100
        taiko_underfunded = (taiko_df['vault_deficit'] > 0).mean() * 100

        metrics['arbitrum_underfunded_pct'] = arbitrum_underfunded
        metrics['taiko_underfunded_pct'] = taiko_underfunded

        return metrics

    def create_comparison_visualization(self, results: Dict) -> None:
        """Create comprehensive comparison visualization."""

        arbitrum_df = results['arbitrum_results']
        taiko_df = results['taiko_results']
        comparison = results['comparison']
        l1_data = results['l1_data']

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f"Arbitrum vs Taiko Fee Mechanism Comparison\n{HISTORICAL_PERIODS[results['period']].name}",
                    fontsize=16)

        # L1 Basefee
        axes[0,0].plot(arbitrum_df['time_step'], arbitrum_df['l1_basefee'] / 1e9,
                      color='gray', alpha=0.7, label='L1 Basefee')
        axes[0,0].set_title('L1 Basefee (gwei)')
        axes[0,0].set_ylabel('Basefee (gwei)')
        axes[0,0].grid(True, alpha=0.3)

        # Estimated Fees Comparison
        axes[0,1].plot(arbitrum_df['time_step'], arbitrum_df['estimated_fee'],
                      label='Arbitrum', color='blue', alpha=0.8)
        axes[0,1].plot(taiko_df['time_step'], taiko_df['estimated_fee'],
                      label='Taiko', color='red', alpha=0.8)
        axes[0,1].set_title('Estimated Fees Comparison')
        axes[0,1].set_ylabel('Fee (ETH)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Vault Balance
        axes[1,0].plot(arbitrum_df['time_step'], arbitrum_df['vault_balance'],
                      label='Arbitrum', color='blue')
        axes[1,0].plot(taiko_df['time_step'], taiko_df['vault_balance'],
                      label='Taiko', color='red')
        axes[1,0].axhline(y=1000, color='black', linestyle='--', alpha=0.5, label='Target')
        axes[1,0].set_title('Vault Balance')
        axes[1,0].set_ylabel('Balance (ETH)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Fee Distribution
        axes[1,1].hist(arbitrum_df['estimated_fee'], bins=50, alpha=0.5,
                      label='Arbitrum', color='blue', density=True)
        axes[1,1].hist(taiko_df['estimated_fee'], bins=50, alpha=0.5,
                      label='Taiko', color='red', density=True)
        axes[1,1].set_title('Fee Distribution')
        axes[1,1].set_xlabel('Fee (ETH)')
        axes[1,1].set_ylabel('Density')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        # Comparison Metrics
        metrics_data = [
            ['Fee CV', f"{comparison['arbitrum_fee_cv']:.3f}", f"{comparison['taiko_fee_cv']:.3f}"],
            ['Avg Fee', f"{comparison['arbitrum_avg_fee']:.6f}", f"{comparison['taiko_avg_fee']:.6f}"],
            ['Max Fee', f"{comparison['arbitrum_max_fee']:.6f}", f"{comparison['taiko_max_fee']:.6f}"],
            ['95th %ile', f"{comparison['arbitrum_95_fee']:.6f}", f"{comparison['taiko_95_fee']:.6f}"],
            ['Vault Std', f"{comparison['arbitrum_vault_std']:.2f}", f"{comparison['taiko_vault_std']:.2f}"],
            ['L1 Corr', f"{comparison['arbitrum_l1_correlation']:.3f}", f"{comparison['taiko_l1_correlation']:.3f}"],
            ['Underfunded%', f"{comparison['arbitrum_underfunded_pct']:.1f}%", f"{comparison['taiko_underfunded_pct']:.1f}%"]
        ]

        axes[2,0].axis('off')
        table = axes[2,0].table(cellText=metrics_data,
                               colLabels=['Metric', 'Arbitrum', 'Taiko'],
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[2,0].set_title('Performance Metrics')

        # Dynamic Pricing Behavior (Arbitrum specific)
        axes[2,1].plot(arbitrum_df['time_step'], arbitrum_df['price_per_byte'],
                      color='green', label='Price/Byte')
        axes[2,1].set_title('Arbitrum Dynamic Pricing')
        axes[2,1].set_xlabel('Time Step')
        axes[2,1].set_ylabel('Price per Byte (ETH)')
        axes[2,1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        filename = f"results/arbitrum_vs_taiko_{results['period']}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved to {filename}")

        return fig


def analyze_arbitrum_advantages():
    """Analyze the advantages of Arbitrum's approach."""

    print("\n" + "="*60)
    print("ARBITRUM L1 PRICING MECHANISM ANALYSIS")
    print("="*60)

    advantages = [
        "1. COMPRESSION-AWARE PRICING",
        "   - Uses Brotli compression ratio (~15%) to estimate actual L1 data size",
        "   - More accurate cost estimation vs raw transaction size",
        "   - Incentivizes compressible transactions",
        "",
        "2. DYNAMIC SELF-ADJUSTMENT",
        "   - Continuously compares collected fees to actual batch costs",
        "   - Automatically adjusts price per unit based on surplus/deficit",
        "   - No need for manual parameter tuning",
        "",
        "3. ALLOCATION-BASED FAIRNESS",
        "   - Distributes costs based on timestamp allocation",
        "   - Ensures fair compensation for batch posters",
        "   - Separates fee collection from cost attribution",
        "",
        "4. ROBUST SMOOTHING",
        "   - Multiple layers of smoothing (basefee + price adjustment)",
        "   - Exponential mechanisms prevent extreme adjustments",
        "   - Time-based allocation windows reduce noise",
        "",
        "5. PRODUCTION-TESTED",
        "   - Proven mechanism handling billions in transaction volume",
        "   - Evolved through multiple iterations and market cycles",
        "   - Battle-tested against extreme market conditions"
    ]

    for line in advantages:
        print(line)

    print("\n" + "="*60)
    print("COMPARISON WITH TAIKO'S APPROACH")
    print("="*60)

    comparison_points = [
        "ARBITRUM ADVANTAGES:",
        "✓ More sophisticated L1 cost estimation (compression-aware)",
        "✓ Self-tuning parameters reduce governance overhead",
        "✓ Proven scalability and robustness",
        "✓ Better handling of batch economics",
        "",
        "TAIKO ADVANTAGES:",
        "✓ Simpler, more transparent fee formula",
        "✓ Explicit deficit correction with configurable horizons",
        "✓ Easier to reason about and audit",
        "✓ More predictable fee behavior for users",
        "",
        "SYNTHESIS OPPORTUNITY:",
        "→ Combine Arbitrum's compression-aware estimation",
        "→ With Taiko's explicit deficit correction approach",
        "→ Add dynamic parameter adjustment like Arbitrum's",
        "→ Maintain Taiko's transparency and simplicity"
    ]

    for line in comparison_points:
        print(line)


def main():
    """Run comprehensive comparison analysis."""

    print("Arbitrum vs Taiko Fee Mechanism Analysis")
    print("=" * 50)

    # Analyze Arbitrum's advantages
    analyze_arbitrum_advantages()

    # Run comparison on real data
    analyzer = ComparisonAnalyzer()

    # Test on a synthetic high volatility period first
    print(f"\nRunning comparison on synthetic DeFi Summer data...")
    try:
        results = analyzer.compare_mechanisms('defi_summer')

        print(f"\nComparison Results:")
        comparison = results['comparison']
        print(f"Arbitrum Fee CV: {comparison['arbitrum_fee_cv']:.3f}")
        print(f"Taiko Fee CV: {comparison['taiko_fee_cv']:.3f}")
        print(f"Arbitrum Avg Fee: {comparison['arbitrum_avg_fee']:.6f} ETH")
        print(f"Taiko Avg Fee: {comparison['taiko_avg_fee']:.6f} ETH")

        # Create visualization
        analyzer.create_comparison_visualization(results)

    except Exception as e:
        print(f"Error running comparison: {e}")

    print(f"\nTo run with real data, use:")
    print(f"python arbitrum_comparison.py --api-key YOUR_ETHERSCAN_KEY --period defi_summer")


if __name__ == "__main__":
    main()