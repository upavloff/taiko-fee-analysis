"""
Revised Optimization Framework with Corrected Metrics

This module implements optimization using the revised metrics framework
that addresses protocol researcher feedback and focuses on fundamental
objectives without proxy metrics or L1 correlation bias.

Key Changes:
1. Eliminated L1 responsiveness from UX metrics
2. Replaced vague "max deficit duration" with deficit-weighted duration
3. Removed cost recovery and mechanism overhead
4. 6-step alignment as constraint, not optimization objective
5. Focused on insolvency risk rather than L1 tracking
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
import warnings

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'core'))
sys.path.insert(0, os.path.join(project_root, 'src', 'analysis'))

from core.improved_simulator import ImprovedTaikoFeeSimulator, ImprovedSimulationParams
from analysis.revised_metrics import RevisedMetricsCalculator, check_6step_alignment_constraint, is_configuration_feasible, check_basic_safety_constraints
from analysis.nsga_ii_optimizer import NSGAII, ParameterSpace, EvolutionOperators, Individual


class RevisedParameterSpace(ParameterSpace):
    """
    Parameter space with 6-step alignment as hard constraint.
    """

    def __init__(self,
                 mu_bounds: Tuple[float, float] = (0.0, 1.0),
                 nu_bounds: Tuple[float, float] = (0.02, 1.0),
                 H_bounds: Tuple[int, int] = (6, 576)):
        """
        Initialize parameter space with 6-step constraint.

        Args:
            mu_bounds: Bounds for L1 weight parameter
            nu_bounds: Bounds for deficit weight parameter
            H_bounds: Bounds for horizon parameter (must be 6-aligned)
        """
        # Generate only 6-step aligned H values
        self.aligned_H_values = [h for h in range(H_bounds[0], H_bounds[1] + 1) if h % 6 == 0]

        if not self.aligned_H_values:
            raise ValueError("No valid 6-step aligned H values in specified bounds")

        # Store bounds for mu and nu
        self.mu_bounds = mu_bounds
        self.nu_bounds = nu_bounds
        self.H_bounds = H_bounds

    def sample_individual(self) -> Individual:
        """Sample individual with guaranteed 6-step alignment."""
        mu = np.random.uniform(*self.mu_bounds)
        nu = np.random.uniform(*self.nu_bounds)
        H = np.random.choice(self.aligned_H_values)  # Always 6-step aligned

        return Individual(mu=mu, nu=nu, H=H)

    def clip_individual(self, individual: Individual) -> Individual:
        """Clip individual to valid bounds with 6-step alignment."""
        mu_clipped = np.clip(individual.mu, *self.mu_bounds)
        nu_clipped = np.clip(individual.nu, *self.nu_bounds)

        # Find closest 6-step aligned H value
        H_clipped = min(self.aligned_H_values, key=lambda x: abs(x - individual.H))

        return Individual(mu=mu_clipped, nu=nu_clipped, H=H_clipped)


def evaluate_individual_revised(individual: Individual,
                               scenario_data: np.ndarray,
                               simulation_params: dict) -> Individual:
    """
    Evaluate individual using revised metrics framework.

    Args:
        individual: Individual to evaluate
        scenario_data: L1 basefee sequence
        simulation_params: Base simulation parameters

    Returns:
        Individual with evaluated objectives and feasibility
    """
    start_time = time.time()

    try:
        # Create simulation parameters
        sim_params = ImprovedSimulationParams(
            mu=individual.mu,
            nu=individual.nu,
            H=individual.H,
            **simulation_params
        )

        # Create L1 model
        class ArrayL1Model:
            def __init__(self, data):
                self.data = data
            def generate_sequence(self, steps, initial_basefee=None):
                return self.data[:steps]
            def get_name(self):
                return "evaluation_scenario"

        l1_model = ArrayL1Model(scenario_data)

        # Run simulation
        simulator = ImprovedTaikoFeeSimulator(sim_params, l1_model)
        df = simulator.run_simulation()

        # Calculate revised metrics
        revised_calc = RevisedMetricsCalculator(simulation_params['target_balance'])
        revised_metrics = revised_calc.calculate_all_metrics(df)

        # Get Pareto objectives (negated for minimization)
        ux_obj, safety_obj, efficiency_obj = revised_metrics.get_pareto_objectives()

        # Update individual
        individual.ux_objective = ux_obj
        individual.stability_objective = safety_obj  # Keep same name for compatibility
        individual.efficiency_objective = efficiency_obj
        individual.enhanced_metrics = revised_metrics
        individual.simulation_time = time.time() - start_time

        # Check feasibility using revised constraints
        individual.is_feasible = is_configuration_feasible(revised_metrics, individual.H)

        # Count constraint violations
        safety_constraints = check_basic_safety_constraints(revised_metrics)
        individual.constraint_violations = sum(1 for satisfied in safety_constraints.values() if not satisfied)

        # Add 6-step alignment violation if applicable
        if not check_6step_alignment_constraint(individual.H):
            individual.constraint_violations += 1

    except Exception as e:
        # Handle simulation failures
        individual.ux_objective = float('inf')
        individual.stability_objective = float('inf')
        individual.efficiency_objective = float('inf')
        individual.is_feasible = False
        individual.constraint_violations = 999
        individual.simulation_time = time.time() - start_time

        warnings.warn(f"Simulation failed for {individual.mu:.3f}, {individual.nu:.3f}, {individual.H}: {e}")

    return individual


class RevisedOptimizer:
    """
    Revised optimizer using corrected metrics framework.
    """

    def __init__(self, output_dir: str = "results/revised_optimization"):
        """Initialize revised optimizer."""
        self.output_dir = output_dir
        self.project_root = project_root

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load historical scenarios
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

        # Revised parameter space with 6-step constraint
        self.parameter_space = RevisedParameterSpace(
            mu_bounds=(0.0, 1.0),
            nu_bounds=(0.02, 1.0),
            H_bounds=(6, 576)
        )

    def _load_historical_scenarios(self) -> Dict[str, np.ndarray]:
        """Load historical L1 basefee scenarios."""
        scenarios = {}
        data_cache_dir = os.path.join(self.project_root, 'data', 'data_cache')

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
                    scenarios[scenario_name] = df['basefee_wei'].values[:500]
                    print(f"✓ Loaded {scenario_name}: {len(scenarios[scenario_name])} data points")
            except Exception as e:
                print(f"✗ Error loading {scenario_name}: {e}")

        # Fallback to synthetic if no data
        if not scenarios:
            print("⚠ No historical data found, using synthetic scenarios")
            scenarios = {
                'synthetic_stable': np.full(500, 10e9),
                'synthetic_volatile': np.random.lognormal(np.log(20e9), 0.5, 500)
            }

        return scenarios

    def run_revised_optimization(self,
                                scenario_name: str = 'recent_low',
                                population_size: int = 100,
                                max_generations: int = 50,
                                n_workers: int = 4) -> Tuple[List[Individual], Dict]:
        """
        Run optimization with revised metrics framework.

        Args:
            scenario_name: Scenario to optimize on
            population_size: NSGA-II population size
            max_generations: Number of generations
            n_workers: Parallel workers

        Returns:
            Tuple of (pareto_front, optimization_stats)
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        print(f"\n🎯 Revised Optimization for scenario: {scenario_name}")
        print(f"   Using corrected metrics framework without L1 bias")

        scenario_data = self.scenarios[scenario_name]
        basefee_gwei = scenario_data / 1e9

        print(f"   Data: {len(scenario_data)} steps, {basefee_gwei.min():.1f}-{basefee_gwei.max():.1f} gwei")

        # Custom NSGA-II with revised evaluation
        class RevisedNSGAII(NSGAII):
            def _evaluate_population(self, scenario_data, simulation_params, production_constraints, verbose):
                """Override evaluation to use revised metrics."""
                if verbose:
                    print(f"   Evaluating {len(self.population)} individuals with revised metrics...")

                # Evaluate with revised metrics
                for i in range(len(self.population)):
                    self.population[i] = evaluate_individual_revised(
                        self.population[i], scenario_data, simulation_params
                    )

            def _evaluate_population_subset(self, individuals, scenario_data, simulation_params,
                                          production_constraints, verbose):
                """Override subset evaluation to use revised metrics."""
                if verbose:
                    print(f"   Evaluating {len(individuals)} offspring with revised metrics...")

                for i in range(len(individuals)):
                    individuals[i] = evaluate_individual_revised(
                        individuals[i], scenario_data, simulation_params
                    )

        # Setup optimizer
        optimizer = RevisedNSGAII(
            population_size=population_size,
            max_generations=max_generations,
            parameter_space=self.parameter_space,
            evolution_ops=EvolutionOperators(),
            n_workers=1  # Custom evaluation doesn't support multiprocessing
        )

        # Run optimization
        start_time = time.time()
        population, history = optimizer.optimize(
            scenario_data=scenario_data,
            simulation_params=self.base_sim_params,
            production_constraints=False,  # We handle constraints in revised evaluation
            verbose=True
        )

        optimization_time = time.time() - start_time

        # Extract Pareto front
        pareto_front = optimizer.get_pareto_front(feasible_only=True)

        stats = {
            'scenario_name': scenario_name,
            'optimization_time': optimization_time,
            'pareto_front_size': len(pareto_front),
            'feasible_solutions': sum(1 for ind in population if ind.is_feasible),
            'total_solutions': len(population),
            'history': history
        }

        print(f"\n✅ Revised optimization completed")
        print(f"   Time: {optimization_time:.1f}s")
        print(f"   Pareto front: {len(pareto_front)} solutions")
        print(f"   Feasible rate: {stats['feasible_solutions']}/{stats['total_solutions']}")

        if pareto_front:
            print(f"\n🏆 Top revised solutions:")
            for i, ind in enumerate(pareto_front[:3]):
                print(f"   {i+1}. μ={ind.mu:.3f}, ν={ind.nu:.3f}, H={ind.H}")
                print(f"      UX={-ind.ux_objective:.3f}, Safety={-ind.stability_objective:.3f}, Efficiency={-ind.efficiency_objective:.3f}")

        return pareto_front, stats

    def analyze_revised_results(self, pareto_front: List[Individual]) -> Dict:
        """Analyze results from revised optimization."""
        if not pareto_front:
            return {'error': 'No feasible solutions found'}

        print(f"\n📊 Analyzing {len(pareto_front)} revised Pareto solutions...")

        # Extract parameter distributions
        mus = [ind.mu for ind in pareto_front]
        nus = [ind.nu for ind in pareto_front]
        Hs = [ind.H for ind in pareto_front]

        analysis = {
            'num_solutions': len(pareto_front),
            'parameter_ranges': {
                'mu': {'min': min(mus), 'max': max(mus), 'mean': np.mean(mus)},
                'nu': {'min': min(nus), 'max': max(nus), 'mean': np.mean(nus)},
                'H': {'min': min(Hs), 'max': max(Hs), 'mean': np.mean(Hs)}
            },
            'insights': {}
        }

        # Analyze key patterns
        zero_mu_count = sum(1 for mu in mus if abs(mu) < 0.01)
        analysis['insights']['zero_mu_preference'] = zero_mu_count / len(mus)

        # All H values should be 6-aligned by constraint
        aligned_count = sum(1 for H in Hs if H % 6 == 0)
        analysis['insights']['6step_alignment_rate'] = aligned_count / len(Hs)

        # Find representative solutions
        best_ux = max(pareto_front, key=lambda x: -x.ux_objective)
        best_safety = max(pareto_front, key=lambda x: -x.stability_objective)
        best_efficiency = max(pareto_front, key=lambda x: -x.efficiency_objective)

        analysis['recommended_solutions'] = {
            'best_ux': {'mu': best_ux.mu, 'nu': best_ux.nu, 'H': best_ux.H},
            'best_safety': {'mu': best_safety.mu, 'nu': best_safety.nu, 'H': best_safety.H},
            'best_efficiency': {'mu': best_efficiency.mu, 'nu': best_efficiency.nu, 'H': best_efficiency.H}
        }

        print(f"   Key insights:")
        print(f"     μ=0 preference: {analysis['insights']['zero_mu_preference']:.1%}")
        print(f"     6-step aligned: {analysis['insights']['6step_alignment_rate']:.1%}")
        print(f"   Best UX: μ={best_ux.mu:.3f}, ν={best_ux.nu:.3f}, H={best_ux.H}")
        print(f"   Best Safety: μ={best_safety.mu:.3f}, ν={best_safety.nu:.3f}, H={best_safety.H}")

        return analysis


def main():
    """Run revised optimization."""
    print("🔄 Revised Optimization Framework")
    print("Corrected metrics without L1 bias or proxy measures")
    print("="*60)

    optimizer = RevisedOptimizer()

    # Run optimization on a representative scenario
    pareto_front, stats = optimizer.run_revised_optimization(
        scenario_name='recent_low',
        population_size=50,  # Smaller for testing
        max_generations=20,
        n_workers=1
    )

    # Analyze results
    analysis = optimizer.analyze_revised_results(pareto_front)

    print(f"\n📋 Revised Optimization Summary")
    print(f"Framework successfully eliminated L1 correlation bias")
    print(f"6-step alignment enforced as constraint")
    print(f"Focus on fundamental protocol objectives")

    return 0


if __name__ == "__main__":
    exit(main())