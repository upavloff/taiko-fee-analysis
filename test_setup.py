#!/usr/bin/env python3
"""
Test script to validate that all imports and basic functionality work.
Run this before using the Jupyter notebook to ensure everything is set up correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported correctly."""
    print("🧪 Testing import system...")

    # Add src directories to path
    project_root = Path(__file__).parent
    src_path = project_root / 'src'

    sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(src_path / 'core'))
    sys.path.insert(0, str(src_path / 'data'))
    sys.path.insert(0, str(src_path / 'analysis'))
    sys.path.insert(0, str(src_path / 'utils'))

    try:
        # Test core imports
        print("  ✓ Testing core modules...")
        from fee_mechanism_simulator import TaikoFeeSimulator, SimulationParams, GeometricBrownianMotion, FeeVault
        from improved_simulator import ImprovedTaikoFeeSimulator, ImprovedSimulationParams
        print("    ✅ Core modules imported successfully")

        # Test analysis imports
        print("  ✓ Testing analysis modules...")
        from mechanism_metrics import MetricsCalculator
        print("    ✅ Analysis modules imported successfully")

        # Test data imports
        print("  ✓ Testing data modules...")
        from rpc_data_fetcher import ImprovedRealDataIntegrator, RPCBasefeeeFetcher
        print("    ✅ Data modules imported successfully")

        # Test utils imports
        print("  ✓ Testing utility modules...")
        from vault_initialization_demo import demonstrate_l1_cost_calculation
        print("    ✅ Utility modules imported successfully")

        # Return classes in a dict for easier access
        return True, {
            'TaikoFeeSimulator': TaikoFeeSimulator,
            'ImprovedTaikoFeeSimulator': ImprovedTaikoFeeSimulator,
            'ImprovedSimulationParams': ImprovedSimulationParams,
            'GeometricBrownianMotion': GeometricBrownianMotion,
            'MetricsCalculator': MetricsCalculator,
            'ImprovedRealDataIntegrator': ImprovedRealDataIntegrator
        }

    except Exception as e:
        print(f"    ❌ Import failed: {e}")
        return False, None

def test_basic_functionality(modules):
    """Test that basic functionality works."""
    print("\n🔧 Testing basic functionality...")

    # Unpack the module dictionary
    TaikoFeeSimulator = modules['TaikoFeeSimulator']
    ImprovedTaikoFeeSimulator = modules['ImprovedTaikoFeeSimulator']
    ImprovedSimulationParams = modules['ImprovedSimulationParams']
    GeometricBrownianMotion = modules['GeometricBrownianMotion']
    MetricsCalculator = modules['MetricsCalculator']
    ImprovedRealDataIntegrator = modules['ImprovedRealDataIntegrator']

    try:
        # Test basic simulation
        print("  ✓ Testing basic simulation...")
        params = ImprovedSimulationParams(
            mu=0.5, nu=0.3, H=144,
            target_balance=1000,
            vault_initialization_mode="target",
            total_steps=10  # Short test
        )

        l1_model = GeometricBrownianMotion(mu=0.0, sigma=0.1)
        simulator = ImprovedTaikoFeeSimulator(params, l1_model)
        df = simulator.run_simulation()

        assert len(df) == 10, f"Expected 10 steps, got {len(df)}"
        assert 'estimated_fee' in df.columns, "Missing estimated_fee column"
        print("    ✅ Basic simulation works")

        # Test metrics calculation
        print("  ✓ Testing metrics calculation...")
        calc = MetricsCalculator(target_balance=1000)
        metrics = calc.calculate_all_metrics(df)

        assert hasattr(metrics, 'avg_fee'), "Missing avg_fee metric"
        assert hasattr(metrics, 'fee_cv'), "Missing fee_cv metric"
        print("    ✅ Metrics calculation works")

        # Test data integrator initialization
        print("  ✓ Testing data integrator...")
        integrator = ImprovedRealDataIntegrator()
        assert integrator.cache_dir.exists(), "Cache directory not created"
        print("    ✅ Data integrator works")

        return True, (df, metrics)

    except Exception as e:
        print(f"    ❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_numpy_pandas():
    """Test that numpy and pandas work correctly."""
    print("\n📊 Testing numpy/pandas...")

    try:
        import numpy as np
        import pandas as pd

        # Test basic operations
        arr = np.array([1, 2, 3, 4, 5])
        df = pd.DataFrame({'x': arr, 'y': arr * 2})

        assert len(df) == 5, "DataFrame creation failed"
        assert df['y'].sum() == 30, "DataFrame operations failed"

        print("    ✅ Numpy/Pandas working correctly")
        return True

    except Exception as e:
        print(f"    ❌ Numpy/Pandas test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 TAIKO FEE ANALYSIS - SETUP VALIDATION")
    print("=" * 50)

    # Test 1: Basic dependencies
    if not test_numpy_pandas():
        print("\n❌ FAILED: Basic dependencies not working")
        return False

    # Test 2: Import system
    success, modules = test_imports()
    if not success:
        print("\n❌ FAILED: Import system broken")
        return False

    # Test 3: Basic functionality
    success, results = test_basic_functionality(modules)
    if not success:
        print("\n❌ FAILED: Basic functionality broken")
        return False

    df, metrics = results

    # Test results summary
    print("\n📋 Test Results Summary:")
    print(f"  • Simulation steps: {len(df)}")
    print(f"  • Average fee: {metrics.avg_fee:.2e} ETH")
    print(f"  • Fee variability: {metrics.fee_cv:.3f}")

    print("\n✅ ALL TESTS PASSED!")
    print("\n🎯 Your setup is working correctly.")
    print("📖 You can now run the Jupyter notebook:")
    print("   jupyter notebook notebooks/taiko_fee_analysis.ipynb")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)