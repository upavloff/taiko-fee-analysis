#!/usr/bin/env python3
"""Quick test to see optimal lambda_c values"""

import sys
import numpy as np
from pathlib import Path

# Add the specs_implementation to path
sys.path.append(str(Path(__file__).parent / "python" / "specs_implementation"))

from specs_nsga_ii_optimizer import SpecsNSGAII, SpecsIndividual, StakeholderProfile

def quick_lambda_test():
    """Run 1 generation to see lambda_c values in optimal solutions"""

    # Create synthetic data
    np.random.seed(42)
    synthetic_data = {"synthetic": np.random.exponential(20e9, 100)}  # ~20 gwei basefees

    optimizer = SpecsNSGAII(
        population_size=20,
        max_generations=1,  # Just 1 generation
        crossover_rate=0.9,
        mutation_rate=0.1
    )

    print("🎯 Quick Lambda_C Test (1 generation, 20 individuals)")
    print("=" * 60)

    # Test one stakeholder
    results = optimizer.optimize(synthetic_data)

    print(f"\n📊 Top 10 solutions with lambda_c values:")
    print("Rank | μ      | ν      | H   | λ_C    | UX     | Robust | CapEff")
    print("-" * 70)

    for i, sol in enumerate(results[:10]):
        print(f"{i+1:4d} | {sol.mu:6.3f} | {sol.nu:6.3f} | {sol.H:3d} | {sol.lambda_c:6.3f} | "
              f"{sol.ux_objective:6.3f} | {sol.robustness_objective:6.3f} | {sol.capital_efficiency_objective:6.3f}")

    # Analyze lambda_c distribution
    lambda_values = [sol.lambda_c for sol in results]
    print(f"\n🔍 Lambda_C Analysis:")
    print(f"   Min:    {min(lambda_values):.3f}")
    print(f"   Max:    {max(lambda_values):.3f}")
    print(f"   Mean:   {np.mean(lambda_values):.3f}")
    print(f"   Median: {np.median(lambda_values):.3f}")
    print(f"   Std:    {np.std(lambda_values):.3f}")

if __name__ == "__main__":
    quick_lambda_test()