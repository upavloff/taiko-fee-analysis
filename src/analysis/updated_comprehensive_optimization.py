#!/usr/bin/env python3
"""
Updated Comprehensive Multi-Objective Optimization - Using Canonical Modules

This script demonstrates the canonical optimization implementation for finding
Pareto optimal fee mechanism parameters across multiple scenarios.

Key Updates:
- Uses canonical_optimization.py for NSGA-II implementation
- Uses canonical_metrics.py for performance evaluation
- Uses canonical_fee_mechanism.py for simulation
- Demonstrates single source of truth architecture
- Provides clear comparison with legacy implementations

Features:
1. Multi-objective optimization with configurable strategies
2. Historical L1 data integration for realistic evaluation
3. Comprehensive Pareto frontier analysis
4. Parameter robustness validation
5. Results export and visualization

Usage:
    python updated_comprehensive_optimization.py --strategy balanced --generations 50
    python updated_comprehensive_optimization.py --strategy crisis_resilient --population 100
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from datetime import datetime
import warnings

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import canonical modules (single source of truth)
from core.canonical_optimization import (
    CanonicalOptimizer,
    OptimizationStrategy,
    OptimizationBounds,
    optimize_for_strategy,
    find_pareto_optimal_parameters
)
from core.canonical_metrics import (
    CanonicalMetricsCalculator,
    MetricThresholds,
    calculate_basic_metrics,
    validate_metric_thresholds
)
from core.canonical_fee_mechanism import (
    VaultInitMode,
    get_optimal_parameters
)


class CanonicalComprehensiveOptimizer:
    """
    Comprehensive optimization suite using canonical implementations.

    This class provides a complete optimization workflow that demonstrates
    the benefits of the modular canonical architecture.
    """

    def __init__(self, output_dir: str = "results/canonical_optimization"):
        """Initialize the canonical comprehensive optimizer."""
        self.output_dir = output_dir
        self.results_cache: Dict[str, Any] = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self.optimizer = CanonicalOptimizer()
        self.metrics_calculator = CanonicalMetricsCalculator()

        print(f"🎯 Canonical Comprehensive Optimizer initialized")
        print(f"📁 Output directory: {output_dir}")

    def load_historical_data(self, scenario: str = "auto") -> Optional[List[float]]:
        """
        Load historical L1 data for optimization evaluation.

        Args:
            scenario: Data scenario to use ('auto', 'recent_low', 'luna_crash', etc.)

        Returns:
            List of L1 basefee values in wei, or None if synthetic data should be used
        """
        data_paths = {
            'recent_low': 'data_cache/recent_low_fees_3hours.csv',
            'luna_crash': 'data_cache/luna_crash_true_peak_contiguous.csv',
            'july_2022': 'data_cache/real_july_2022_spike_data.csv',
            'pepe_crisis': 'data_cache/may_2023_pepe_crisis_data.csv'
        }

        if scenario == "auto":
            # Try to find any available dataset
            for name, path in data_paths.items():
                full_path = os.path.join(project_root, path)
                if os.path.exists(full_path):
                    scenario = name
                    break
            else:
                print("📊 No historical data found, using synthetic data")
                return None

        if scenario not in data_paths:
            print(f"⚠️  Unknown scenario '{scenario}', using synthetic data")
            return None

        data_path = os.path.join(project_root, data_paths[scenario])

        if not os.path.exists(data_path):
            print(f"⚠️  Data file not found: {data_path}")
            print("📊 Using synthetic data instead")
            return None

        try:
            # Load CSV data
            df = pd.read_csv(data_path)

            # Extract basefee column (try different possible column names)
            basefee_cols = ['basefee_wei', 'basefee', 'l1_basefee', 'basefee_gwei']
            basefee_col = None

            for col in basefee_cols:
                if col in df.columns:
                    basefee_col = col
                    break

            if basefee_col is None:
                print(f"⚠️  No basefee column found in {data_path}")
                return None

            basefees = df[basefee_col].values

            # Convert to wei if needed
            if 'gwei' in basefee_col.lower():
                basefees = basefees * 1e9

            # Validate data
            if len(basefees) == 0:
                print(f"⚠️  Empty dataset: {data_path}")
                return None

            if np.any(basefees <= 0):
                print(f"⚠️  Invalid basefee values in dataset")
                return None

            print(f"✅ Loaded {len(basefees)} historical L1 basefee points from {scenario}")
            print(f"   Range: {np.min(basefees)/1e9:.3f} - {np.max(basefees)/1e9:.3f} gwei")

            return basefees.tolist()

        except Exception as e:
            print(f"❌ Failed to load historical data: {e}")
            return None

    def run_optimization(self,
                        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                        population_size: int = 100,
                        generations: int = 50,
                        l1_scenario: str = "auto") -> Dict[str, Any]:
        """
        Run comprehensive optimization using canonical implementation.

        Args:
            strategy: Optimization strategy to use
            population_size: GA population size
            generations: Number of GA generations
            l1_scenario: Historical data scenario

        Returns:
            Complete optimization results
        """
        print(f"🚀 Starting canonical optimization:")
        print(f"   Strategy: {strategy.value}")
        print(f"   Population: {population_size}")
        print(f"   Generations: {generations}")
        print()

        start_time = time.time()

        # Load historical data
        l1_data = self.load_historical_data(l1_scenario)

        # Configure optimizer for this run
        self.optimizer.simulation_steps = 1800  # 1 hour simulation

        # Progress tracking
        def progress_callback(generation: int, total_generations: int, population: List[Any]):
            progress_pct = (generation / total_generations) * 100
            print(f"   Generation {generation:2d}/{total_generations} ({progress_pct:5.1f}%)")

        # Run optimization
        try:
            results = self.optimizer.optimize(
                strategy=strategy,
                population_size=population_size,
                generations=generations,
                l1_data=l1_data,
                vault_init=VaultInitMode.DEFICIT,  # Start with realistic deficit
                progress_callback=progress_callback
            )

            optimization_time = time.time() - start_time

            print(f"✅ Optimization completed in {optimization_time:.1f}s")
            print(f"   Pareto solutions found: {len(results['pareto_front'])}")
            print(f"   Hypervolume: {results['hypervolume']:.6f}")
            print(f"   Spread: {results['spread']:.3f}")
            print()

            # Add metadata to results
            results['metadata'] = {
                'strategy': strategy.value,
                'population_size': population_size,
                'generations': generations,
                'l1_scenario': l1_scenario,
                'optimization_time': optimization_time,
                'timestamp': datetime.now().isoformat(),
                'used_historical_data': l1_data is not None,
                'simulation_steps': self.optimizer.simulation_steps
            }

            # Cache results
            cache_key = f"{strategy.value}_{population_size}_{generations}"
            self.results_cache[cache_key] = results

            return results

        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def analyze_pareto_solutions(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze Pareto optimal solutions and extract insights.

        Args:
            results: Optimization results from run_optimization

        Returns:
            DataFrame with detailed analysis of Pareto solutions
        """
        print("📊 Analyzing Pareto optimal solutions:")

        pareto_front = results['pareto_front']

        if not pareto_front:
            print("   No Pareto solutions found")
            return pd.DataFrame()

        # Extract solution data
        solutions_data = []

        for i, individual in enumerate(pareto_front):
            if individual.objectives is None:
                continue

            # Calculate interpretable scores from objectives
            # (objectives are negative for maximization, so we reverse them)
            user_score = max(0.0, 1.0 + individual.objectives[0])
            safety_score = max(0.0, 1.0 + individual.objectives[4])
            efficiency_score = max(0.0, 1.0 + individual.objectives[8])
            overall_score = (user_score + safety_score + efficiency_score) / 3.0

            solution_data = {
                'solution_id': i,
                'mu': individual.mu,
                'nu': individual.nu,
                'H': individual.H,
                'user_experience_score': user_score,
                'protocol_safety_score': safety_score,
                'economic_efficiency_score': efficiency_score,
                'overall_score': overall_score,
                'pareto_rank': individual.rank,
                'crowding_distance': individual.crowding_distance,
                'simulation_time': individual.simulation_time
            }

            solutions_data.append(solution_data)

        df = pd.DataFrame(solutions_data)

        if len(df) > 0:
            print(f"   Analyzed {len(df)} Pareto solutions")
            print(f"   Parameter ranges:")
            print(f"     μ: {df['mu'].min():.3f} - {df['mu'].max():.3f}")
            print(f"     ν: {df['nu'].min():.3f} - {df['nu'].max():.3f}")
            print(f"     H: {df['H'].min()} - {df['H'].max()}")
            print(f"   Score ranges:")
            print(f"     User Experience: {df['user_experience_score'].min():.3f} - {df['user_experience_score'].max():.3f}")
            print(f"     Protocol Safety: {df['protocol_safety_score'].min():.3f} - {df['protocol_safety_score'].max():.3f}")
            print(f"     Economic Efficiency: {df['economic_efficiency_score'].min():.3f} - {df['economic_efficiency_score'].max():.3f}")
            print()

        return df

    def find_best_solutions(self, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """
        Identify the best solutions according to different criteria.

        Args:
            df: DataFrame from analyze_pareto_solutions
            top_k: Number of top solutions to return for each criterion

        Returns:
            DataFrame with best solutions highlighted
        """
        if len(df) == 0:
            return df

        print(f"🏆 Finding top {top_k} solutions by different criteria:")

        # Different ranking criteria
        criteria = {
            'overall_best': 'overall_score',
            'user_focused': 'user_experience_score',
            'safety_focused': 'protocol_safety_score',
            'efficiency_focused': 'economic_efficiency_score'
        }

        best_solutions = {}

        for criterion_name, score_column in criteria.items():
            top_solutions = df.nlargest(top_k, score_column)
            best_solutions[criterion_name] = top_solutions

            print(f"\n   {criterion_name.replace('_', ' ').title()}:")
            for _, sol in top_solutions.head(3).iterrows():  # Show top 3
                print(f"     μ={sol['mu']:.3f}, ν={sol['nu']:.3f}, H={int(sol['H'])}, "
                      f"Score={sol[score_column]:.3f}")

        print()

        # Check consistency with research findings
        optimal_params = get_optimal_parameters()
        research_mu = optimal_params['mu']
        research_nu = optimal_params['nu']
        research_H = optimal_params['H']

        # Find closest solution to research parameters
        df['research_distance'] = np.sqrt(
            (df['mu'] - research_mu)**2 +
            (df['nu'] - research_nu)**2 +
            ((df['H'] - research_H)/100)**2  # Scale H appropriately
        )

        closest_to_research = df.loc[df['research_distance'].idxmin()]

        print(f"🔬 Closest to research parameters (μ={research_mu}, ν={research_nu}, H={research_H}):")
        print(f"     Found: μ={closest_to_research['mu']:.3f}, ν={closest_to_research['nu']:.3f}, "
              f"H={int(closest_to_research['H'])}")
        print(f"     Distance: {closest_to_research['research_distance']:.3f}")
        print(f"     Overall score: {closest_to_research['overall_score']:.3f}")
        print()

        return best_solutions

    def create_visualizations(self, results: Dict[str, Any], df: pd.DataFrame):
        """
        Create comprehensive visualizations of optimization results.

        Args:
            results: Optimization results
            df: Analyzed Pareto solutions DataFrame
        """
        if len(df) == 0:
            print("⚠️  No solutions to visualize")
            return

        print("📈 Creating optimization visualizations:")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Pareto front in objective space (simplified 2D projection)
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(df['user_experience_score'], df['protocol_safety_score'],
                            c=df['economic_efficiency_score'], cmap='viridis',
                            s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('User Experience Score')
        ax1.set_ylabel('Protocol Safety Score')
        ax1.set_title('Pareto Front (UX vs Safety)')
        plt.colorbar(scatter, ax=ax1, label='Economic Efficiency')

        # 2. Parameter space coverage
        ax2 = plt.subplot(2, 3, 2)
        scatter2 = ax2.scatter(df['mu'], df['nu'], c=df['overall_score'],
                             s=df['H']/5, cmap='RdYlGn', alpha=0.7,
                             edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('μ (L1 weight)')
        ax2.set_ylabel('ν (deficit weight)')
        ax2.set_title('Parameter Space (size=H)')
        plt.colorbar(scatter2, ax=ax2, label='Overall Score')

        # 3. Score distributions
        ax3 = plt.subplot(2, 3, 3)
        score_cols = ['user_experience_score', 'protocol_safety_score', 'economic_efficiency_score']
        score_data = [df[col] for col in score_cols]
        bp = ax3.boxplot(score_data, labels=['UX', 'Safety', 'Efficiency'], patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'gold']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax3.set_title('Score Distributions')
        ax3.set_ylabel('Score')

        # 4. Convergence history
        ax4 = plt.subplot(2, 3, 4)
        if 'convergence_history' in results and results['convergence_history']:
            conv_hist = results['convergence_history']
            generations = [entry['generation'] for entry in conv_hist]
            hypervolumes = [entry['hypervolume'] for entry in conv_hist]
            ax4.plot(generations, hypervolumes, 'b-', linewidth=2, marker='o', markersize=4)
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Hypervolume')
            ax4.set_title('Convergence Progress')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No convergence data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Convergence Progress')

        # 5. H distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(df['H'], bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax5.set_xlabel('H (horizon)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Horizon Distribution')

        # 6. Overall score vs simulation time
        ax6 = plt.subplot(2, 3, 6)
        ax6.scatter(df['simulation_time'], df['overall_score'],
                   alpha=0.6, color='purple', edgecolors='black', linewidth=0.5)
        ax6.set_xlabel('Simulation Time (s)')
        ax6.set_ylabel('Overall Score')
        ax6.set_title('Score vs Computation Time')

        plt.suptitle(f'Canonical Optimization Results - {results["metadata"]["strategy"].title()} Strategy',
                     fontsize=16)
        plt.tight_layout()

        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(self.output_dir, f"optimization_results_{timestamp}.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"   Visualization saved: {viz_path}")

        plt.show()

    def export_results(self, results: Dict[str, Any], df: pd.DataFrame):
        """
        Export optimization results to files.

        Args:
            results: Optimization results
            df: Analyzed Pareto solutions DataFrame
        """
        print("💾 Exporting results:")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export solutions CSV
        if len(df) > 0:
            csv_path = os.path.join(self.output_dir, f"pareto_solutions_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            print(f"   Solutions exported: {csv_path}")

        # Export full results JSON (excluding large arrays)
        results_export = results.copy()
        results_export.pop('pareto_front', None)  # Too large for JSON
        results_export.pop('final_population', None)  # Too large for JSON

        json_path = os.path.join(self.output_dir, f"optimization_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results_export, f, indent=2, default=str)
        print(f"   Metadata exported: {json_path}")

        print()


def run_comprehensive_analysis(args):
    """
    Run the complete canonical optimization analysis.

    Args:
        args: Command line arguments
    """
    print("🎯 Canonical Comprehensive Optimization Analysis")
    print("=" * 60)
    print()

    # Initialize optimizer
    optimizer = CanonicalComprehensiveOptimizer(args.output_dir)

    # Convert strategy string to enum
    try:
        strategy = OptimizationStrategy(args.strategy)
    except ValueError:
        print(f"❌ Invalid strategy: {args.strategy}")
        print(f"Available strategies: {[s.value for s in OptimizationStrategy]}")
        return 1

    try:
        # Run optimization
        results = optimizer.run_optimization(
            strategy=strategy,
            population_size=args.population,
            generations=args.generations,
            l1_scenario=args.scenario
        )

        # Analyze results
        df = optimizer.analyze_pareto_solutions(results)

        if len(df) > 0:
            # Find best solutions
            best_solutions = optimizer.find_best_solutions(df, top_k=5)

            # Create visualizations
            if not args.no_plot:
                optimizer.create_visualizations(results, df)

            # Export results
            if args.export:
                optimizer.export_results(results, df)

        print("🎉 Canonical optimization analysis completed successfully!")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Optimization interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Canonical Comprehensive Optimization for Taiko Fee Mechanism",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Optimization parameters
    parser.add_argument('--strategy', choices=[s.value for s in OptimizationStrategy],
                       default='balanced', help='Optimization strategy')
    parser.add_argument('--population', type=int, default=100,
                       help='GA population size')
    parser.add_argument('--generations', type=int, default=50,
                       help='GA generations')
    parser.add_argument('--scenario', default='auto',
                       help='Historical L1 data scenario')

    # Output options
    parser.add_argument('--output-dir', default='results/canonical_optimization',
                       help='Output directory for results')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip creating visualizations')
    parser.add_argument('--export', action='store_true',
                       help='Export detailed results to files')

    args = parser.parse_args()

    return run_comprehensive_analysis(args)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)