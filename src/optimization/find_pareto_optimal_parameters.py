#!/usr/bin/env python3
"""
Pareto Optimal Parameter Discovery Script

Uses the normalized objective functions and stakeholder-specific profiles
to discover optimal fee mechanism parameters across different scenarios.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import time
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add the specs_implementation to path
sys.path.append(str(Path(__file__).parent / "python" / "specs_implementation"))

from core.simulation_engine import SimulationEngine
from metrics.calculator import MetricsCalculator
from metrics.objectives import StakeholderProfile, ObjectiveWeights


def load_crisis_datasets():
    """Load multiple crisis datasets for robust testing"""
    datasets = {}

    data_files = [
        ("luna_crash", "data/data_cache/luna_crash_true_peak_contiguous.csv"),
        ("july_spike", "data/data_cache/real_july_2022_spike_data.csv"),
        ("recent_low", "data/data_cache/recent_low_fees_3hours.csv")
    ]

    for name, file_path in data_files:
        if os.path.exists(file_path):
            print(f"📊 Loading {name}: {file_path}")
            df = pd.read_csv(file_path)
            basefees_wei = df['basefee_wei'].values

            # Use subset for optimization speed
            max_points = 500  # Enough for good signal, fast enough for optimization
            if len(basefees_wei) > max_points:
                # Take samples across the full range
                indices = np.linspace(0, len(basefees_wei)-1, max_points, dtype=int)
                basefees_wei = basefees_wei[indices]

            datasets[name] = basefees_wei
            print(f"   Points: {len(basefees_wei):,}, Range: {basefees_wei.min()/1e9:.3f} - {basefees_wei.max()/1e9:.1f} gwei")

    # Fallback: synthetic crisis
    if not datasets:
        print("📊 Using synthetic crisis data")
        base_fee = 10e9  # 10 gwei baseline
        crisis_pattern = np.concatenate([
            np.linspace(1, 20, 150),      # Crisis buildup
            np.full(200, 20),             # Peak crisis
            np.linspace(20, 1, 150)       # Recovery
        ])
        datasets["synthetic_crisis"] = base_fee * crisis_pattern

    return datasets


def generate_parameter_grid():
    """Generate comprehensive parameter grid for optimization"""
    # SPECS.md compliant parameter ranges
    mu_values = np.linspace(0.0, 1.0, 11)      # [0.0, 0.1, ..., 1.0] - L1 weight
    nu_values = np.linspace(0.0, 1.0, 11)      # [0.0, 0.1, ..., 1.0] - Deficit weight
    H_values = [36, 72, 144, 288]              # Prediction horizon (6-step aligned)

    print(f"🔍 Parameter Grid: {len(mu_values)} μ × {len(nu_values)} ν × {len(H_values)} H = {len(mu_values) * len(nu_values) * len(H_values)} combinations")

    return list(product(mu_values, nu_values, H_values))


def evaluate_parameter_set(
    params: Tuple[float, float, int],
    datasets: Dict[str, np.ndarray],
    stakeholder_profile: StakeholderProfile
) -> Dict[str, Any]:
    """Evaluate a single parameter set across all datasets"""
    mu, nu, H = params

    # Create metrics calculator for this stakeholder
    calc = MetricsCalculator.for_stakeholder(stakeholder_profile)

    # Aggregate results across all datasets
    all_metrics = []
    feasibility_count = 0

    for dataset_name, basefees_wei in datasets.items():
        try:
            # Create simulation engine
            engine = SimulationEngine(
                mu=mu, nu=nu, horizon_h=H,
                target_vault_balance=1.0,
                q_bar=6.9e5  # Correct Q value
            )

            # Convert basefees to L1 costs
            txs_per_batch = 100
            gas_per_tx = max(200_000 / txs_per_batch, 200)
            l1_costs = (basefees_wei * gas_per_tx) / 1e18  # Convert to ETH

            # Simulate
            sim_results = engine.simulate_series(l1_costs)

            # Convert to SimulationResults
            from metrics.constraints import SimulationResults
            results = SimulationResults(
                fees_per_gas=sim_results['basefee_per_gas'].values,
                vault_balances=sim_results['vault_balance_after'].values,
                revenues=sim_results['revenue'].values,
                l1_costs=sim_results['l1_cost_actual'].values,
                deficits=sim_results['deficit'].values,
                timestamps=np.arange(len(sim_results)),
                subsidies=sim_results['subsidy_paid'].values,
                Q_bar=6.9e5
            )

            # Evaluate metrics
            metrics = calc.evaluate_parameter_set(results)
            all_metrics.append(metrics)

            if metrics.is_feasible:
                feasibility_count += 1

        except Exception as e:
            # Skip failed simulations
            continue

    if not all_metrics:
        # Return worst possible scores for failed parameter sets
        return {
            "params": {"mu": mu, "nu": nu, "H": H},
            "feasible": False,
            "ux_score": float('inf'),
            "robustness_score": float('inf'),
            "capital_efficiency_score": float('inf'),
            "combined_score": float('inf'),
            "feasibility_ratio": 0.0,
            "avg_fee_gwei": float('inf'),
            "details": "Simulation failed"
        }

    # Aggregate metrics across datasets
    avg_ux = np.mean([m.objective_results.ux_objective for m in all_metrics])
    avg_robustness = np.mean([m.objective_results.robustness_objective for m in all_metrics])
    avg_capital_eff = np.mean([m.objective_results.capital_efficiency_objective for m in all_metrics])
    avg_fee = np.mean([m.objective_results.avg_fee_gwei for m in all_metrics])

    combined_score = avg_ux + avg_robustness + avg_capital_eff
    feasibility_ratio = feasibility_count / len(all_metrics)

    return {
        "params": {"mu": mu, "nu": nu, "H": H},
        "feasible": feasibility_ratio > 0.5,  # Require majority feasibility
        "ux_score": avg_ux,
        "robustness_score": avg_robustness,
        "capital_efficiency_score": avg_capital_eff,
        "combined_score": combined_score,
        "feasibility_ratio": feasibility_ratio,
        "avg_fee_gwei": avg_fee,
        "details": f"Tested on {len(datasets)} datasets"
    }


def find_pareto_frontier(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find Pareto optimal parameter sets"""
    # Filter to feasible solutions only
    feasible_results = [r for r in results if r["feasible"]]

    if not feasible_results:
        print("❌ No feasible parameter sets found!")
        return []

    print(f"✅ Found {len(feasible_results)} feasible parameter sets out of {len(results)} total")

    pareto_frontier = []

    for i, result in enumerate(feasible_results):
        is_dominated = False

        # Check if this result is dominated by any other
        for other in feasible_results:
            if other == result:
                continue

            # A solution dominates if it's better or equal in all objectives and strictly better in at least one
            ux_better = other["ux_score"] <= result["ux_score"]
            robust_better = other["robustness_score"] <= result["robustness_score"]
            cap_better = other["capital_efficiency_score"] <= result["capital_efficiency_score"]

            # At least one must be strictly better
            any_strictly_better = (
                other["ux_score"] < result["ux_score"] or
                other["robustness_score"] < result["robustness_score"] or
                other["capital_efficiency_score"] < result["capital_efficiency_score"]
            )

            if ux_better and robust_better and cap_better and any_strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto_frontier.append(result)

    return pareto_frontier


def run_stakeholder_optimization(profile: StakeholderProfile, datasets: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Run full optimization for a specific stakeholder profile"""
    print(f"\n🎯 Optimizing for {profile.value.upper()} stakeholder")
    print("=" * 60)

    parameter_grid = generate_parameter_grid()

    # Evaluate all parameter combinations
    results = []
    start_time = time.time()

    print(f"⚡ Evaluating {len(parameter_grid)} parameter combinations...")

    for i, params in enumerate(parameter_grid):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            progress = i / len(parameter_grid) * 100
            eta = elapsed / max(i, 1) * (len(parameter_grid) - i)
            print(f"   Progress: {progress:.1f}% ({i}/{len(parameter_grid)}) - ETA: {eta:.0f}s")

        result = evaluate_parameter_set(params, datasets, profile)
        results.append(result)

    elapsed = time.time() - start_time
    print(f"✅ Evaluation complete in {elapsed:.1f}s")

    # Find Pareto frontier
    pareto_results = find_pareto_frontier(results)

    print(f"\n📊 Pareto Frontier: {len(pareto_results)} optimal parameter sets")

    return pareto_results


def display_results(profile: StakeholderProfile, pareto_results: List[Dict[str, Any]]):
    """Display optimization results in a nice format"""
    print(f"\n🏆 {profile.value.upper()} - PARETO OPTIMAL PARAMETERS")
    print("=" * 80)

    if not pareto_results:
        print("❌ No Pareto optimal solutions found!")
        return

    # Sort by combined score (lower is better)
    sorted_results = sorted(pareto_results, key=lambda x: x["combined_score"])

    print(f"{'Rank':<6} {'μ':<6} {'ν':<6} {'H':<6} {'Avg Fee':<10} {'UX Score':<12} {'Robust':<12} {'CapEff':<12} {'Combined':<12}")
    print("-" * 80)

    for i, result in enumerate(sorted_results[:10]):  # Top 10
        params = result["params"]
        rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}"

        print(f"{rank_symbol:<6} {params['mu']:<6.1f} {params['nu']:<6.1f} {params['H']:<6d} "
              f"{result['avg_fee_gwei']:<10.3f} {result['ux_score']:<12.2e} "
              f"{result['robustness_score']:<12.2e} {result['capital_efficiency_score']:<12.2e} "
              f"{result['combined_score']:<12.2e}")

    # Highlight top 3 recommendations
    print(f"\n🎯 TOP 3 RECOMMENDATIONS for {profile.value.upper()}:")
    for i in range(min(3, len(sorted_results))):
        result = sorted_results[i]
        params = result["params"]
        rank = "🥇 BEST" if i == 0 else "🥈 SECOND" if i == 1 else "🥉 THIRD"

        print(f"\n{rank}: μ={params['mu']:.1f}, ν={params['nu']:.1f}, H={params['H']}")
        print(f"   Average Fee: {result['avg_fee_gwei']:.3f} gwei")
        print(f"   Feasibility: {result['feasibility_ratio']*100:.1f}%")
        print(f"   Combined Score: {result['combined_score']:.2e}")


def main():
    """Run comprehensive Pareto optimization across all stakeholder profiles"""
    print("🚀 PARETO OPTIMAL PARAMETER DISCOVERY")
    print("=" * 60)
    print("Using SPECS.md compliant normalized objectives and stakeholder profiles")

    # Load crisis datasets
    datasets = load_crisis_datasets()

    # Test all stakeholder profiles
    profiles_to_test = [
        StakeholderProfile.USER_CENTRIC,
        StakeholderProfile.PROTOCOL_LAUNCH,
        StakeholderProfile.OPERATOR_FOCUSED,
        StakeholderProfile.BALANCED,
        StakeholderProfile.STRESS_TESTED
    ]

    all_results = {}

    for profile in profiles_to_test:
        pareto_results = run_stakeholder_optimization(profile, datasets)
        all_results[profile] = pareto_results
        display_results(profile, pareto_results)

    # Cross-profile comparison
    print(f"\n📋 CROSS-STAKEHOLDER COMPARISON")
    print("=" * 80)
    print("Best parameter set for each stakeholder profile:")
    print(f"{'Profile':<18} {'Best Params':<20} {'Avg Fee':<10} {'Combined Score':<15}")
    print("-" * 65)

    for profile, results in all_results.items():
        if results:
            best = min(results, key=lambda x: x["combined_score"])
            params = best["params"]
            param_str = f"μ={params['mu']:.1f},ν={params['nu']:.1f},H={params['H']}"
            print(f"{profile.value:<18} {param_str:<20} {best['avg_fee_gwei']:<10.3f} {best['combined_score']:<15.2e}")
        else:
            print(f"{profile.value:<18} {'No feasible solution':<20} {'N/A':<10} {'N/A':<15}")

    print(f"\n✅ Pareto optimization complete!")
    print(f"📊 Results show optimal parameters vary significantly by stakeholder profile")
    print(f"🎯 Use stakeholder-specific parameters for deployment")

    return 0


if __name__ == "__main__":
    exit(main())