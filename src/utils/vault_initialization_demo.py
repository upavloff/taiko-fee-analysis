"""
Quick demonstration of vault initialization impact on fee visualization

Shows how proper vault initialization eliminates the extreme initial fees
that dominate the visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fee_mechanism_simulator import TaikoFeeSimulator, SimulationParams, GeometricBrownianMotion


def compare_vault_initializations():
    """Compare different vault initialization strategies."""

    print("Demonstrating vault initialization impact on fees...")

    # Generate volatile L1 basefee data
    np.random.seed(42)
    l1_model = GeometricBrownianMotion(mu=0.0, sigma=0.5)  # High volatility

    # Test scenarios
    scenarios = [
        ("Empty Vault (Original)", 0, 1000),      # Start empty
        ("Target Balance", 1000, 1000),           # Start at target
        ("50% Underfunded", 500, 1000),           # Start 50% underfunded
        ("20% Overfunded", 1200, 1000),           # Start 20% overfunded
    ]

    results = {}

    for scenario_name, initial_balance, target_balance in scenarios:
        print(f"  Running {scenario_name}...")

        # Create custom parameters
        params = SimulationParams(
            mu=0.5, nu=0.3, H=144,
            target_balance=target_balance,
            total_steps=300  # Shorter for quick demo
        )

        simulator = TaikoFeeSimulator(params, l1_model)

        # Manually set initial vault balance
        simulator.vault.balance = initial_balance

        # Run simulation
        df = simulator.run_simulation()
        results[scenario_name] = df

    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Impact of Vault Initialization on Fee Behavior', fontsize=16)

    colors = ['red', 'blue', 'orange', 'green']

    # Plot 1: Estimated Fees (Full Range)
    ax = axes[0, 0]
    for (scenario, df), color in zip(results.items(), colors):
        time_hours = df['time_step'] * 12 / 3600
        ax.plot(time_hours, df['estimated_fee'], label=scenario, color=color, alpha=0.8)

    ax.set_title('Estimated Fees - Full Range')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Estimated Fee (ETH)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Estimated Fees (Zoomed - excluding first 10 steps)
    ax = axes[0, 1]
    for (scenario, df), color in zip(results.items(), colors):
        df_zoom = df[10:]  # Skip first 10 steps
        time_hours = df_zoom['time_step'] * 12 / 3600
        ax.plot(time_hours, df_zoom['estimated_fee'], label=scenario, color=color, alpha=0.8)

    ax.set_title('Estimated Fees - Zoomed (after step 10)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Estimated Fee (ETH)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Vault Balance Evolution
    ax = axes[1, 0]
    for (scenario, df), color in zip(results.items(), colors):
        time_hours = df['time_step'] * 12 / 3600
        ax.plot(time_hours, df['vault_balance'], label=scenario, color=color, alpha=0.8)

    ax.axhline(y=1000, color='black', linestyle='--', alpha=0.5, label='Target')
    ax.set_title('Vault Balance Evolution')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Vault Balance (ETH)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Fee Statistics Comparison
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate stats
    stats_data = []
    for scenario, df in results.items():
        df_clean = df[10:]  # Exclude first 10 steps
        stats_data.append([
            scenario[:15],
            f"{df['estimated_fee'].iloc[0]:.2e}",  # Initial fee
            f"{df_clean['estimated_fee'].mean():.2e}",  # Average fee (after initial)
            f"{df_clean['estimated_fee'].std():.2e}",   # Fee volatility
            f"{(df['vault_deficit'] > 0).mean()*100:.0f}%"  # % time underfunded
        ])

    table = ax.table(cellText=stats_data,
                    colLabels=['Scenario', 'Initial Fee', 'Avg Fee', 'Fee Std', '% Underfunded'],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title('Fee Statistics Comparison')

    plt.tight_layout()

    # Save the plot
    filename = "results/vault_initialization_demo.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Demo visualization saved to {filename}")

    # Print summary
    print(f"\n=== KEY FINDINGS ===")
    for scenario, df in results.items():
        df_clean = df[10:]  # Exclude initialization period
        print(f"\n{scenario}:")
        print(f"  Initial Fee: {df['estimated_fee'].iloc[0]:.2e} ETH")
        print(f"  Avg Fee (after init): {df_clean['estimated_fee'].mean():.2e} ETH")
        print(f"  Max Fee: {df['estimated_fee'].max():.2e} ETH")
        print(f"  Initial Vault Balance: {df['vault_balance'].iloc[0]:.0f} ETH")
        print(f"  Time Underfunded: {(df['vault_deficit'] > 0).mean()*100:.0f}%")

    return results


def demonstrate_l1_cost_calculation():
    """Demonstrate how L1 basefee gets converted to C_L1."""

    print(f"\n=== L1 BASEFEE TO C_L1 CONVERSION DEMO ===")

    # Example basefees (in wei)
    example_basefees = [
        (10e9, "10 gwei - Low"),
        (30e9, "30 gwei - Medium"),
        (100e9, "100 gwei - High"),
        (300e9, "300 gwei - Crisis")
    ]

    # Default parameters
    gas_per_batch = 200000
    txs_per_batch = 100

    print(f"Parameters: {gas_per_batch:,} gas per batch, {txs_per_batch} txs per batch")
    print(f"Gas per transaction: {gas_per_batch/txs_per_batch:,.0f}")
    print()

    print("L1 Basefee → C_L1 Conversion:")
    print("Basefee (gwei) | Cost per TX (ETH) | Daily Cost (100 tx/step)")
    print("-" * 60)

    for basefee_wei, description in example_basefees:
        # Step 1: Calculate gas cost per transaction
        gas_cost_per_tx = gas_per_batch / txs_per_batch

        # Step 2: Calculate cost per transaction in ETH
        cost_per_tx_eth = basefee_wei * gas_cost_per_tx / 1e18

        # Step 3: Daily cost estimate (assuming 100 tx per 12-second step)
        steps_per_day = 24 * 60 * 60 / 12  # 7200 steps
        daily_cost = cost_per_tx_eth * 100 * steps_per_day

        basefee_gwei = basefee_wei / 1e9
        print(f"{basefee_gwei:>8.0f} gwei   | {cost_per_tx_eth:.2e} ETH     | {daily_cost:.2f} ETH")


if __name__ == "__main__":
    # Run the demonstrations
    compare_vault_initializations()
    demonstrate_l1_cost_calculation()