#!/usr/bin/env python3
"""Diagnose L1 cost calculation and fee mechanism issues"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the specs_implementation to path
sys.path.append(str(Path(__file__).parent / "python" / "specs_implementation"))

from core.simulation_engine import SimulationEngine
from core.fee_controller import FeeController

def diagnose_l1_costs():
    """Analyze L1 cost calculation and fee mechanism behavior"""

    print("🔍 DIAGNOSING L1 COST CALCULATION & FEE MECHANISM")
    print("=" * 65)

    # Create realistic L1 basefee scenarios
    scenarios = {
        "Normal Period": np.full(50, 20e9),      # 20 gwei constant
        "Moderate Spike": np.full(50, 50e9),     # 50 gwei constant
        "Crisis Period": np.full(50, 150e9),     # 150 gwei constant
        "Volatile Period": np.concatenate([
            np.full(20, 20e9),
            np.full(10, 100e9),
            np.full(20, 30e9)
        ])
    }

    for scenario_name, basefees_wei in scenarios.items():
        print(f"\n📊 SCENARIO: {scenario_name}")
        print("-" * 50)

        # Current L1 cost calculation (from optimizer)
        txs_per_batch = 100
        gas_per_tx = max(200_000 / txs_per_batch, 200)  # = 2000
        l1_costs_eth = (basefees_wei * gas_per_tx) / 1e18

        print(f"L1 Basefees: {basefees_wei[0]/1e9:.1f} gwei")
        print(f"Gas per TX: {gas_per_tx:,.0f}")
        print(f"L1 Cost per batch: {l1_costs_eth[0]:.6f} ETH")

        # Test different μ values to see fee response
        mu_tests = [0.0, 0.2, 0.5, 0.8, 1.0]

        print("\nFee Response Test (μ varies, ν=0.2, H=144):")
        print("μ     | Avg Fee (gwei) | L1 Component | Deficit Component")
        print("------|----------------|---------------|------------------")

        for mu in mu_tests:
            try:
                # Create simulation engine
                engine = SimulationEngine(
                    mu=mu,
                    nu=0.2,
                    horizon_h=144,
                    lambda_c=0.1,
                    target_vault_balance=1.0,
                    q_bar=6.9e5,
                    f_min=0,
                    f_max=float('inf')
                )

                # Run simulation
                sim_results = engine.simulate_series(l1_costs_eth)
                fees_gwei = sim_results['basefee_per_gas'] / 1e9
                avg_fee = np.mean(fees_gwei)

                # Try to calculate components manually
                l1_component = mu * (l1_costs_eth[0] * 1e18) / 6.9e5  # Convert back to wei per gas
                l1_comp_gwei = l1_component / 1e9

                print(f"{mu:5.1f} | {avg_fee:14.6f} | {l1_comp_gwei:13.6f} | {'TBD':>16}")

            except Exception as e:
                print(f"{mu:5.1f} | ERROR: {str(e)[:30]}")

        # Calculate revenue vs L1 costs
        q_bar = 6.9e5
        revenue_per_batch_eth = (20e9 * q_bar) / 1e18  # Assume 20 gwei fee
        l1_cost_per_batch_eth = l1_costs_eth[0]

        print(f"\n💰 Economics Analysis:")
        print(f"   Revenue/batch (20 gwei): {revenue_per_batch_eth:.6f} ETH")
        print(f"   L1 cost/batch:           {l1_cost_per_batch_eth:.6f} ETH")
        print(f"   L1 cost ratio:           {l1_cost_per_batch_eth/revenue_per_batch_eth:.2%}")

        if l1_cost_per_batch_eth > revenue_per_batch_eth:
            print("   ❌ L1 costs > revenue → μ=0 optimal (ignore L1)")
        else:
            print("   ✅ Revenue > L1 costs → μ>0 might be viable")

def test_fee_controller_directly():
    """Test the fee controller in isolation"""

    print(f"\n\n🎛️  DIRECT FEE CONTROLLER TEST")
    print("=" * 65)

    # Create fee controller
    controller = FeeController(
        mu=0.5,
        nu=0.2,
        horizon_h=144,
        q_bar=6.9e5,
        f_min=0,
        f_max=float('inf')
    )

    # Test scenarios
    l1_costs = [0.0001, 0.001, 0.01]  # ETH
    deficits = [0, 0.5, 2.0]  # ETH

    print("L1 Cost | Deficit | Raw Fee (gwei) | Components")
    print("--------|---------|----------------|---------------------------")

    for l1_cost in l1_costs:
        for deficit in deficits:
            # Convert to expected units
            smoothed_l1_cost = l1_cost  # ETH
            deficit_per_gas = deficit / 6.9e5  # ETH per gas

            # Calculate fee
            raw_fee_wei = controller.calculate_fee(
                smoothed_l1_cost_eth=smoothed_l1_cost,
                deficit_eth=deficit,
                current_basefee=1e9  # 1 gwei baseline
            )

            raw_fee_gwei = raw_fee_wei / 1e9

            # Calculate components
            l1_term = controller.mu * (smoothed_l1_cost * 1e18) / controller.q_bar
            deficit_term = controller.nu * (deficit * 1e18) / (controller.horizon_h * controller.q_bar)

            print(f"{l1_cost:7.4f} | {deficit:7.1f} | {raw_fee_gwei:14.6f} | L1:{l1_term/1e9:.3f} Def:{deficit_term/1e9:.3f}")

if __name__ == "__main__":
    diagnose_l1_costs()
    test_fee_controller_directly()