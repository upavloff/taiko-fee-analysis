#!/usr/bin/env python3
"""Quick test of the fixed simulation with realistic inputs"""

import sys
import numpy as np
from pathlib import Path

# Add the specs_implementation to path
sys.path.append(str(Path(__file__).parent / "python" / "specs_implementation"))

from core.simulation_engine import SimulationEngine
from core.units import gwei_to_wei, l1_basefee_to_batch_cost

def test_fixed_simulation():
    """Test simulation with the unit fix"""

    print("🧪 Testing Fixed Simulation Engine")
    print("=" * 45)

    # Create realistic scenario: 50 gwei L1 for 10 batches
    l1_basefees_gwei = [50.0] * 10
    l1_basefees_wei = [gwei_to_wei(gwei) for gwei in l1_basefees_gwei]
    l1_costs_wei = [l1_basefee_to_batch_cost(wei, gas_per_tx=2000).value for wei in l1_basefees_wei]

    print(f"L1 basefees: {l1_basefees_gwei[0]} gwei")
    print(f"L1 cost per batch: {l1_costs_wei[0]:,} wei = {l1_costs_wei[0]/1e18:.6f} ETH")
    print()

    # Test different μ values
    mu_values = [0.0, 0.3, 0.6, 1.0]

    print("μ     | Fee (gwei) | Fee (wei)     | L1 Component")
    print("------|------------|---------------|-------------")

    for mu in mu_values:
        try:
            engine = SimulationEngine(
                mu=mu, nu=0.1, horizon_h=144, lambda_c=0.1,
                target_vault_balance=1.0, q_bar=690_000,
                f_min=0, f_max=float('inf')
            )

            # Run one step
            result = engine.simulate_step(l1_costs_wei[0])
            fee_wei = result['basefee_per_gas']
            fee_gwei = fee_wei / 1e9

            # Calculate expected L1 component
            expected_l1_comp = (mu * l1_costs_wei[0] / 690_000) / 1e9

            print(f"{mu:5.1f} | {fee_gwei:10.3f} | {fee_wei:13,.0f} | {expected_l1_comp:11.3f}")

        except Exception as e:
            print(f"{mu:5.1f} | ERROR: {str(e)[:50]}")

    print("\n🔍 Analysis:")
    print("- μ=0.0: Should be ~0 (no L1 tracking)")
    print("- μ=0.6: Should be ~43 gwei (60% of L1 cost)")
    print("- μ=1.0: Should be ~72 gwei (100% of L1 cost)")

if __name__ == "__main__":
    test_fixed_simulation()