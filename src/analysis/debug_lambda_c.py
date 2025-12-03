#!/usr/bin/env python3
"""Quick test to see lambda_c parameter values being generated"""

import sys
import numpy as np
from pathlib import Path

# Add the specs_implementation to path
sys.path.append(str(Path(__file__).parent / "python" / "specs_implementation"))

# Import from our optimizer
from specs_nsga_ii_optimizer import SpecsNSGAII, SpecsIndividual

def test_lambda_c_generation():
    """Test what lambda_c values are being generated"""
    optimizer = SpecsNSGAII()

    print("🔍 Testing lambda_c parameter generation:")
    print(f"   Range: {optimizer.lambda_c_bounds}")
    print("\n📊 Sample of 20 random individuals:")
    print("Individual | μ      | ν      | H   | λ_C")
    print("-" * 45)

    for i in range(20):
        individual = optimizer.create_random_individual()
        print(f"{i+1:10d} | {individual.mu:6.3f} | {individual.nu:6.3f} | {individual.H:3d} | {individual.lambda_c:6.3f}")

    print(f"\n🎯 Lambda_C bounds: {optimizer.lambda_c_bounds}")
    print("   Range: [0.01, 1.0] - from very slow (0.01) to instantaneous (1.0) L1 adaptation")

if __name__ == "__main__":
    test_lambda_c_generation()