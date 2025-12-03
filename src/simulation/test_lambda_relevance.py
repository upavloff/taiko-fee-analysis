#!/usr/bin/env python3
"""Test to see if lambda_c matters when mu > 0"""

import sys
import numpy as np
from pathlib import Path

# Add the specs_implementation to path
sys.path.append(str(Path(__file__).parent / "python" / "specs_implementation"))

from specs_nsga_ii_optimizer import SpecsIndividual
from core.simulation_engine import SimulationEngine

def test_lambda_impact():
    """Test how different lambda_c values affect results when mu > 0"""

    # Create synthetic L1 data with volatility
    np.random.seed(42)
    basefees_wei = np.concatenate([
        np.random.normal(20e9, 5e9, 50),   # Normal period: ~20 gwei ± 5
        np.random.normal(100e9, 20e9, 30)  # Crisis period: ~100 gwei ± 20
    ])

    # Convert to L1 costs
    txs_per_batch = 100
    gas_per_tx = max(200_000 / txs_per_batch, 200)
    l1_costs = (basefees_wei * gas_per_tx) / 1e18

    print("🔍 Testing Lambda_C Impact with Forced L1 Usage (μ=0.5)")
    print("=" * 65)
    print("Testing how L1 smoothing affects fees when L1 costs are used")
    print(f"L1 cost range: {min(l1_costs):.6f} - {max(l1_costs):.6f} ETH")
    print()

    # Test different lambda_c values with fixed mu=0.5 (force L1 usage)
    lambda_tests = [0.01, 0.1, 0.3, 0.5, 0.8, 1.0]

    results = []
    for lambda_c in lambda_tests:
        # Create individual with fixed mu=0.5, varying lambda_c
        individual = SpecsIndividual(mu=0.5, nu=0.2, H=144, lambda_c=lambda_c)

        try:
            # Create simulation engine
            engine = SimulationEngine(
                mu=individual.mu,
                nu=individual.nu,
                horizon_h=individual.H,
                lambda_c=individual.lambda_c,
                target_vault_balance=1.0,
                q_bar=6.9e5
            )

            # Run simulation
            sim_results = engine.simulate_series(l1_costs)

            # Calculate basic stats
            fees_gwei = sim_results['basefee_per_gas'] / 1e9
            avg_fee = np.mean(fees_gwei)
            fee_cv = np.std(fees_gwei) / (np.mean(fees_gwei) + 1e-10)

            results.append({
                'lambda_c': lambda_c,
                'avg_fee': avg_fee,
                'fee_cv': fee_cv,
                'min_fee': np.min(fees_gwei),
                'max_fee': np.max(fees_gwei)
            })

        except Exception as e:
            results.append({
                'lambda_c': lambda_c,
                'avg_fee': float('nan'),
                'fee_cv': float('nan'),
                'error': str(e)[:50]
            })

    # Display results
    print("λ_C    | Avg Fee | Fee CV | Min Fee | Max Fee | Notes")
    print("-------|---------|--------|---------|---------|--------")

    for r in results:
        if 'error' in r:
            print(f"{r['lambda_c']:6.2f} | ERROR: {r['error']}")
        else:
            print(f"{r['lambda_c']:6.2f} | {r['avg_fee']:7.4f} | {r['fee_cv']:6.4f} | "
                  f"{r['min_fee']:7.4f} | {r['max_fee']:7.4f} |")

    # Analysis
    valid_results = [r for r in results if 'error' not in r]
    if len(valid_results) > 1:
        fees = [r['avg_fee'] for r in valid_results]
        cvs = [r['fee_cv'] for r in valid_results]
        lambdas = [r['lambda_c'] for r in valid_results]

        print(f"\n📊 Analysis:")
        print(f"   Fee range: {min(fees):.4f} - {max(fees):.4f} gwei")
        print(f"   CV range:  {min(cvs):.4f} - {max(cvs):.4f}")
        print(f"   Lambda_C matters: {max(fees) - min(fees) > 0.001}")

        # Find best lambda_c (lowest CV for stability)
        best_idx = cvs.index(min(cvs))
        print(f"   Best λ_C for stability: {lambdas[best_idx]:.2f} (CV={cvs[best_idx]:.4f})")

if __name__ == "__main__":
    test_lambda_impact()