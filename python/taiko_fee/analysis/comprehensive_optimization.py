"""
Comprehensive Multi-Objective Optimization for Taiko Fee Mechanism

This script runs the complete optimization process using NSGA-II with the enhanced
metrics framework, generating true Pareto frontiers across multiple scenarios.

Key Features:
1. Multi-scenario optimization across historical datasets
2. 6-step batch cycle aligned parameter space exploration
3. Production-ready constraint integration
4. Comprehensive result analysis and visualization
5. Parameter robustness validation

Usage:
    python comprehensive_optimization.py --scenario recent_low --generations 50
    python comprehensive_optimization.py --scenario all --population 200
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import json
import time
from datetime import datetime

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'core'))
sys.path.insert(0, os.path.join(project_root, 'src', 'analysis'))

from analysis.nsga_ii_optimizer import NSGAII, ParameterSpace, EvolutionOperators, Individual
from analysis.enhanced_metrics import EnhancedMetricsCalculator
from analysis.objective_functions import ObjectiveFunctionSuite, OptimizationStrategy


class ComprehensiveOptimizer:
    """
    Complete optimization suite for Taiko fee mechanism parameters.
    """

    def __init__(self, output_dir: str = "results/comprehensive_optimization"):
        """
        Initialize comprehensive optimizer.

        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = output_dir
        self.project_root = project_root
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Historical datasets for optimization
        self.scenarios = self._load_historical_scenarios()

        # Base simulation parameters
        self.base_sim_params = {
            'target_balance': 1000.0,
            'base_demand': 100,
            'fee_elasticity': 0.2,
            'gas_per_batch': 200000,
            'txs_per_batch': 100,
            'batch_frequency': 0.1,
            'time_step_seconds': 2,
            'vault_initialization_mode': 'target',
            'fee_cap': 0.1,
            'total_steps': 500
        }

        # Enhanced parameter space with 6-step alignment
        self.parameter_space = ParameterSpace(
            mu_bounds=(0.0, 1.0),
            nu_bounds=(0.02, 1.0),
            H_bounds=(6, 576),
            H_alignment_bias=0.8  # Strong bias toward 6-step alignment
        )

    def _load_historical_scenarios(self) -> Dict[str, np.ndarray]:
        """
        Load historical L1 basefee scenarios for optimization.

        Returns:
            Dictionary of scenario_name -> basefee_sequence
        """
        scenarios = {}
        data_cache_dir = os.path.join(self.project_root, 'data', 'data_cache')

        # Define available scenarios
        scenario_files = {
            'recent_low': 'recent_low_fees_3hours.csv',
            'july_spike': 'real_july_2022_spike_data.csv',
            'luna_crash': 'luna_crash_true_peak_contiguous.csv',
            'pepe_crisis': 'may_2023_pepe_crisis_data.csv'
        }

        for scenario_name, filename in scenario_files.items():
            filepath = os.path.join(data_cache_dir, filename)

            try:
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    # Use first 500 points for consistent comparison
                    scenarios[scenario_name] = df['basefee_wei'].values[:500]
                    print(f"✓ Loaded {scenario_name}: {len(scenarios[scenario_name])} data points")
                else:
                    print(f"⚠ Scenario file not found: {filepath}")

            except Exception as e:
                print(f"✗ Error loading {scenario_name}: {e}")

        # Add synthetic scenarios if no historical data
        if not scenarios:
            print("⚠ No historical data found, using synthetic scenarios")
            scenarios = {
                'synthetic_stable': np.full(500, 10e9),  # 10 gwei stable
                'synthetic_volatile': np.random.lognormal(np.log(20e9), 0.5, 500)
            }

        return scenarios

    def run_single_scenario_optimization(self,
                                       scenario_name: str,
                                       population_size: int = 100,
                                       max_generations: int = 50,
                                       n_workers: int = 4,
                                       production_constraints: bool = True) -> Tuple[List[Individual], Dict]:
        """
        Run NSGA-II optimization on a single scenario.

        Args:
            scenario_name: Name of scenario to optimize
            population_size: Population size for NSGA-II
            max_generations: Number of generations
            n_workers: Number of parallel workers
            production_constraints: Whether to apply production constraints

        Returns:
            Tuple of (pareto_front, optimization_stats)
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        print(f"\n🎯 Optimizing for scenario: {scenario_name}")
        print(f"   Scenario characteristics:")

        scenario_data = self.scenarios[scenario_name]
        basefee_gwei = scenario_data / 1e9  # Convert to gwei for display

        print(f"     Length: {len(scenario_data)} steps")
        print(f"     Range: {basefee_gwei.min():.1f} - {basefee_gwei.max():.1f} gwei")
        print(f"     Mean: {basefee_gwei.mean():.1f} gwei")
        print(f"     Volatility (CV): {basefee_gwei.std()/basefee_gwei.mean():.3f}")

        # Setup NSGA-II optimizer
        evolution_ops = EvolutionOperators(
            mutation_rate=0.1,
            crossover_rate=0.9,
            eta_m=20.0,    # Moderate mutation
            eta_c=20.0     # Moderate crossover
        )

        optimizer = NSGAII(
            population_size=population_size,
            max_generations=max_generations,
            parameter_space=self.parameter_space,
            evolution_ops=evolution_ops,
            n_workers=n_workers
        )

        # Run optimization
        start_time = time.time()
        population, history = optimizer.optimize(
            scenario_data=scenario_data,
            simulation_params=self.base_sim_params,
            production_constraints=production_constraints,
            verbose=True
        )

        optimization_time = time.time() - start_time

        # Extract Pareto front
        pareto_front = optimizer.get_pareto_front(feasible_only=production_constraints)

        # Compile statistics
        stats = {
            'scenario_name': scenario_name,
            'optimization_time': optimization_time,
            'total_evaluations': population_size * (max_generations + 1),
            'pareto_front_size': len(pareto_front),
            'feasible_solutions': sum(1 for ind in population if ind.is_feasible),
            'total_solutions': len(population),
            'history': history
        }

        print(f"\n✅ Optimization completed for {scenario_name}")
        print(f"   Time: {optimization_time:.1f} seconds")
        print(f"   Pareto front: {len(pareto_front)} solutions")
        print(f"   Feasible rate: {stats['feasible_solutions']}/{stats['total_solutions']} ({100*stats['feasible_solutions']/stats['total_solutions']:.1f}%)")

        return pareto_front, stats

    def analyze_pareto_solutions(self, pareto_front: List[Individual],
                               scenario_name: str) -> Dict:
        """
        Analyze Pareto front solutions and extract insights.

        Args:
            pareto_front: List of Pareto optimal individuals
            scenario_name: Name of the scenario

        Returns:
            Analysis results dictionary
        """
        if not pareto_front:
            return {'error': 'No Pareto solutions found'}

        print(f"\n📊 Analyzing Pareto front for {scenario_name}...")

        analysis = {
            'scenario_name': scenario_name,
            'num_solutions': len(pareto_front),
            'parameter_ranges': {},
            'objective_ranges': {},
            'insights': {},
            'recommended_solutions': {}
        }

        # Extract parameter ranges
        mus = [ind.mu for ind in pareto_front]
        nus = [ind.nu for ind in pareto_front]
        Hs = [ind.H for ind in pareto_front]

        analysis['parameter_ranges'] = {
            'mu': {'min': min(mus), 'max': max(mus), 'mean': np.mean(mus)},
            'nu': {'min': min(nus), 'max': max(nus), 'mean': np.mean(nus)},
            'H': {'min': min(Hs), 'max': max(Hs), 'mean': np.mean(Hs)}
        }

        # Extract objective ranges
        ux_objs = [-ind.ux_objective for ind in pareto_front]  # Convert back to positive
        stability_objs = [-ind.stability_objective for ind in pareto_front]
        efficiency_objs = [-ind.efficiency_objective for ind in pareto_front]

        analysis['objective_ranges'] = {
            'user_experience': {'min': min(ux_objs), 'max': max(ux_objs), 'mean': np.mean(ux_objs)},
            'protocol_stability': {'min': min(stability_objs), 'max': max(stability_objs), 'mean': np.mean(stability_objs)},
            'economic_efficiency': {'min': min(efficiency_objs), 'max': max(efficiency_objs), 'mean': np.mean(efficiency_objs)}
        }

        # 6-step alignment analysis
        aligned_solutions = [ind for ind in pareto_front if ind.H % 6 == 0]
        alignment_rate = len(aligned_solutions) / len(pareto_front)

        analysis['insights']['6_step_alignment_rate'] = alignment_rate
        analysis['insights']['6_step_aligned_count'] = len(aligned_solutions)

        # μ=0 analysis (known optimal pattern)
        zero_mu_solutions = [ind for ind in pareto_front if abs(ind.mu) < 0.01]
        zero_mu_rate = len(zero_mu_solutions) / len(pareto_front)

        analysis['insights']['zero_mu_rate'] = zero_mu_rate
        analysis['insights']['zero_mu_count'] = len(zero_mu_solutions)

        # Find recommended solutions for different strategies
        strategies = {
            'user_focused': max(pareto_front, key=lambda x: -x.ux_objective),
            'stability_focused': max(pareto_front, key=lambda x: -x.stability_objective),
            'efficiency_focused': max(pareto_front, key=lambda x: -x.efficiency_objective),
            'balanced': min(pareto_front, key=lambda x: abs(x.ux_objective + x.stability_objective + x.efficiency_objective) / 3)
        }

        for strategy_name, solution in strategies.items():
            analysis['recommended_solutions'][strategy_name] = {
                'mu': solution.mu,
                'nu': solution.nu,
                'H': solution.H,
                'ux_score': -solution.ux_objective,
                'stability_score': -solution.stability_objective,
                'efficiency_score': -solution.efficiency_objective
            }

        # Print key insights
        print(f"   Parameter ranges:")
        print(f"     μ: {analysis['parameter_ranges']['mu']['min']:.3f} - {analysis['parameter_ranges']['mu']['max']:.3f}")
        print(f"     ν: {analysis['parameter_ranges']['nu']['min']:.3f} - {analysis['parameter_ranges']['nu']['max']:.3f}")
        print(f"     H: {int(analysis['parameter_ranges']['H']['min'])} - {int(analysis['parameter_ranges']['H']['max'])}")

        print(f"   Key insights:")
        print(f"     6-step aligned: {alignment_rate:.1%} ({len(aligned_solutions)}/{len(pareto_front)})")
        print(f"     μ=0 solutions: {zero_mu_rate:.1%} ({len(zero_mu_solutions)}/{len(pareto_front)})")

        return analysis

    def run_multi_scenario_optimization(self,
                                      scenarios: Optional[List[str]] = None,
                                      population_size: int = 100,
                                      max_generations: int = 50,
                                      n_workers: int = 4) -> Dict[str, Dict]:
        """
        Run optimization across multiple scenarios.

        Args:
            scenarios: List of scenario names (None = all scenarios)
            population_size: Population size for NSGA-II
            max_generations: Number of generations
            n_workers: Number of parallel workers

        Returns:
            Dictionary of scenario_name -> results
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())

        print(f"\n🚀 Running multi-scenario optimization")
        print(f"   Scenarios: {scenarios}")
        print(f"   Population: {population_size} | Generations: {max_generations}")

        results = {}
        total_start_time = time.time()

        for scenario_name in scenarios:
            try:
                # Run single scenario optimization
                pareto_front, stats = self.run_single_scenario_optimization(
                    scenario_name=scenario_name,
                    population_size=population_size,
                    max_generations=max_generations,
                    n_workers=n_workers,
                    production_constraints=True
                )

                # Analyze results
                analysis = self.analyze_pareto_solutions(pareto_front, scenario_name)

                # Store results
                results[scenario_name] = {
                    'pareto_front': pareto_front,
                    'optimization_stats': stats,
                    'analysis': analysis
                }

                # Save intermediate results
                self.save_scenario_results(scenario_name, results[scenario_name])

            except Exception as e:
                print(f"❌ Error optimizing scenario {scenario_name}: {e}")
                results[scenario_name] = {'error': str(e)}

        total_time = time.time() - total_start_time

        print(f"\n🎉 Multi-scenario optimization completed!")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Successful scenarios: {len([r for r in results.values() if 'error' not in r])}")

        # Generate comprehensive analysis
        self.generate_comprehensive_report(results)

        return results

    def save_scenario_results(self, scenario_name: str, results: Dict) -> None:
        """Save results for a single scenario."""
        scenario_dir = os.path.join(self.output_dir, f"{scenario_name}_{self.timestamp}")
        os.makedirs(scenario_dir, exist_ok=True)

        # Save Pareto front
        if 'pareto_front' in results:
            pareto_df = pd.DataFrame([ind.to_dict() for ind in results['pareto_front']])
            pareto_df.to_csv(os.path.join(scenario_dir, 'pareto_front.csv'), index=False)

        # Save optimization stats
        if 'optimization_stats' in results:
            with open(os.path.join(scenario_dir, 'optimization_stats.json'), 'w') as f:
                # Convert numpy types for JSON serialization
                stats_serializable = self._make_json_serializable(results['optimization_stats'])
                json.dump(stats_serializable, f, indent=2)

        # Save analysis
        if 'analysis' in results:
            with open(os.path.join(scenario_dir, 'analysis.json'), 'w') as f:
                analysis_serializable = self._make_json_serializable(results['analysis'])
                json.dump(analysis_serializable, f, indent=2)

        print(f"   💾 Results saved to {scenario_dir}")

    def _make_json_serializable(self, obj) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def generate_comprehensive_report(self, results: Dict[str, Dict]) -> None:
        """Generate comprehensive optimization report."""
        print(f"\n📋 Generating comprehensive report...")

        report_path = os.path.join(self.output_dir, f"comprehensive_report_{self.timestamp}.md")

        with open(report_path, 'w') as f:
            f.write("# Comprehensive Taiko Fee Mechanism Optimization Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")

            # Count successful optimizations
            successful_scenarios = [name for name, result in results.items() if 'error' not in result]
            total_pareto_solutions = sum(len(results[name]['pareto_front'])
                                       for name in successful_scenarios)

            f.write(f"- **Scenarios analyzed**: {len(successful_scenarios)}/{len(results)}\n")
            f.write(f"- **Total Pareto solutions found**: {total_pareto_solutions}\n")
            f.write(f"- **Optimization method**: NSGA-II Multi-Objective Evolutionary Algorithm\n")
            f.write(f"- **Parameter space**: 6-step aligned, continuous exploration\n\n")

            # Cross-scenario insights
            f.write("## Cross-Scenario Insights\n\n")

            if successful_scenarios:
                # Aggregate parameter preferences
                all_pareto_solutions = []
                for scenario_name in successful_scenarios:
                    all_pareto_solutions.extend(results[scenario_name]['pareto_front'])

                if all_pareto_solutions:
                    # μ analysis
                    zero_mu_count = sum(1 for ind in all_pareto_solutions if abs(ind.mu) < 0.01)
                    zero_mu_rate = zero_mu_count / len(all_pareto_solutions)

                    # 6-step alignment
                    aligned_count = sum(1 for ind in all_pareto_solutions if ind.H % 6 == 0)
                    aligned_rate = aligned_count / len(all_pareto_solutions)

                    # ν ranges
                    nu_values = [ind.nu for ind in all_pareto_solutions]
                    H_values = [ind.H for ind in all_pareto_solutions]

                    f.write(f"### Parameter Preferences Across All Scenarios\n\n")
                    f.write(f"- **μ=0 preference**: {zero_mu_rate:.1%} of Pareto solutions use μ≈0\n")
                    f.write(f"- **6-step alignment**: {aligned_rate:.1%} of Pareto solutions use H divisible by 6\n")
                    f.write(f"- **ν range**: {min(nu_values):.3f} - {max(nu_values):.3f} (mean: {np.mean(nu_values):.3f})\n")
                    f.write(f"- **H range**: {min(H_values)} - {max(H_values)} steps (mean: {np.mean(H_values):.0f})\n\n")

            # Scenario-specific results
            f.write("## Scenario-Specific Results\n\n")

            for scenario_name in successful_scenarios:
                result = results[scenario_name]
                analysis = result['analysis']

                f.write(f"### {scenario_name.title().replace('_', ' ')}\n\n")

                pareto_count = analysis['num_solutions']
                f.write(f"- **Pareto solutions found**: {pareto_count}\n")

                # Parameter insights
                param_ranges = analysis['parameter_ranges']
                f.write(f"- **Parameter ranges**:\n")
                f.write(f"  - μ: {param_ranges['mu']['min']:.3f} - {param_ranges['mu']['max']:.3f}\n")
                f.write(f"  - ν: {param_ranges['nu']['min']:.3f} - {param_ranges['nu']['max']:.3f}\n")
                f.write(f"  - H: {int(param_ranges['H']['min'])} - {int(param_ranges['H']['max'])}\n")

                # Recommended solutions
                f.write(f"- **Recommended parameters**:\n")
                for strategy, solution in analysis['recommended_solutions'].items():
                    f.write(f"  - {strategy.replace('_', ' ').title()}: μ={solution['mu']:.3f}, ν={solution['nu']:.3f}, H={solution['H']}\n")

                f.write("\n")

            f.write("## Methodology\n\n")
            f.write("- **Algorithm**: NSGA-II (Non-dominated Sorting Genetic Algorithm II)\n")
            f.write("- **Objectives**: User Experience, Protocol Stability, Economic Efficiency\n")
            f.write("- **Constraints**: Production-ready feasibility constraints\n")
            f.write("- **Parameter Space**: Continuous with 6-step batch cycle alignment bias\n")
            f.write("- **Evaluation**: Enhanced metrics framework with 25+ quantitative measures\n\n")

            f.write("## Files Generated\n\n")
            for scenario_name in successful_scenarios:
                f.write(f"- `{scenario_name}_{self.timestamp}/`: Complete results for {scenario_name} scenario\n")

        print(f"   📄 Comprehensive report saved to {report_path}")


def main():
    """Main optimization runner."""
    parser = argparse.ArgumentParser(description="Run comprehensive Taiko fee mechanism optimization")

    parser.add_argument('--scenario', type=str, default='all',
                      help='Scenario to optimize: recent_low, july_spike, luna_crash, pepe_crisis, or all')
    parser.add_argument('--population', type=int, default=100,
                      help='Population size for NSGA-II')
    parser.add_argument('--generations', type=int, default=50,
                      help='Number of generations')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of parallel workers')
    parser.add_argument('--output', type=str, default='analysis/results/comprehensive_optimization',
                      help='Output directory')

    args = parser.parse_args()

    print("🎯 Comprehensive Taiko Fee Mechanism Optimization")
    print("=" * 60)

    # Initialize optimizer
    optimizer = ComprehensiveOptimizer(output_dir=args.output)

    if args.scenario == 'all':
        # Run multi-scenario optimization
        results = optimizer.run_multi_scenario_optimization(
            scenarios=None,  # All scenarios
            population_size=args.population,
            max_generations=args.generations,
            n_workers=args.workers
        )
    else:
        # Run single scenario
        if args.scenario not in optimizer.scenarios:
            print(f"❌ Unknown scenario: {args.scenario}")
            print(f"Available scenarios: {list(optimizer.scenarios.keys())}")
            return 1

        pareto_front, stats = optimizer.run_single_scenario_optimization(
            scenario_name=args.scenario,
            population_size=args.population,
            max_generations=args.generations,
            n_workers=args.workers
        )

        analysis = optimizer.analyze_pareto_solutions(pareto_front, args.scenario)

        results = {args.scenario: {
            'pareto_front': pareto_front,
            'optimization_stats': stats,
            'analysis': analysis
        }}

        optimizer.save_scenario_results(args.scenario, results[args.scenario])

    print("\n🎉 Optimization completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())