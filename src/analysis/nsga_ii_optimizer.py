"""
NSGA-II Multi-Objective Optimization for Taiko Fee Mechanism

This module implements the NSGA-II (Non-dominated Sorting Genetic Algorithm II)
for continuous multi-objective optimization of Taiko fee mechanism parameters.

Key Features:
1. True Pareto frontier generation (not just constraint satisfaction)
2. Continuous parameter space exploration
3. 6-step batch cycle alignment bias
4. Elitist selection with crowding distance
5. Production constraint integration
6. Parallel simulation evaluation

References:
- Deb, K., et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II"
- IEEE Transactions on Evolutionary Computation, 2002
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import time
import os
import sys

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'core'))
sys.path.insert(0, os.path.join(project_root, 'src', 'analysis'))

from core.improved_simulator import ImprovedTaikoFeeSimulator, ImprovedSimulationParams
from analysis.enhanced_metrics import EnhancedMetricsCalculator
from analysis.objective_functions import ObjectiveFunctionSuite, OptimizationStrategy, ProductionObjectiveFunction


@dataclass
class Individual:
    """
    Individual solution in the NSGA-II population.

    Represents a specific parameter configuration and its evaluation results.
    """

    # Parameters
    mu: float          # L1 weight [0.0, 1.0]
    nu: float          # Deficit weight [0.02, 1.0]
    H: int             # Horizon steps (biased toward 6-multiples)

    # Objectives (for minimization - negative of actual values)
    ux_objective: Optional[float] = None           # User experience (minimize)
    stability_objective: Optional[float] = None   # Protocol stability (minimize)
    efficiency_objective: Optional[float] = None  # Economic efficiency (minimize)

    # NSGA-II metadata
    domination_count: int = 0                     # Number of individuals dominating this one
    dominated_solutions: List[int] = None         # Indices of solutions this one dominates
    rank: Optional[int] = None                    # Pareto rank (0 = non-dominated front)
    crowding_distance: float = 0.0               # Crowding distance for diversity

    # Constraint handling
    constraint_violations: int = 0                # Number of constraint violations
    is_feasible: bool = True                     # Whether solution satisfies all constraints

    # Additional metadata
    enhanced_metrics: Optional[Any] = None        # Full enhanced metrics object
    simulation_time: Optional[float] = None       # Time to evaluate this individual

    def __post_init__(self):
        """Initialize dominated solutions list if not provided."""
        if self.dominated_solutions is None:
            self.dominated_solutions = []

    def dominates(self, other: 'Individual') -> bool:
        """
        Check if this individual dominates another.

        Individual A dominates B if:
        1. A is no worse than B in all objectives
        2. A is strictly better than B in at least one objective
        3. Both individuals are feasible, or A is feasible and B is not
        """
        # Constraint-based domination
        if self.is_feasible and not other.is_feasible:
            return True
        if not self.is_feasible and other.is_feasible:
            return False
        if not self.is_feasible and not other.is_feasible:
            # Both infeasible - compare constraint violations
            return self.constraint_violations < other.constraint_violations

        # Both feasible - compare objectives
        if any(obj is None for obj in [self.ux_objective, self.stability_objective, self.efficiency_objective]):
            return False
        if any(obj is None for obj in [other.ux_objective, other.stability_objective, other.efficiency_objective]):
            return True

        # Check domination conditions
        at_least_one_better = False

        objectives_self = [self.ux_objective, self.stability_objective, self.efficiency_objective]
        objectives_other = [other.ux_objective, other.stability_objective, other.efficiency_objective]

        for obj_self, obj_other in zip(objectives_self, objectives_other):
            if obj_self > obj_other:  # Worse in this objective
                return False
            if obj_self < obj_other:  # Better in this objective
                at_least_one_better = True

        return at_least_one_better

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'mu': self.mu,
            'nu': self.nu,
            'H': self.H,
            'ux_objective': self.ux_objective,
            'stability_objective': self.stability_objective,
            'efficiency_objective': self.efficiency_objective,
            'rank': self.rank,
            'crowding_distance': self.crowding_distance,
            'is_feasible': self.is_feasible,
            'constraint_violations': self.constraint_violations,
            'simulation_time': self.simulation_time
        }


class ParameterSpace:
    """
    Defines the parameter space for optimization with intelligent constraints.
    """

    def __init__(self,
                 mu_bounds: Tuple[float, float] = (0.0, 1.0),
                 nu_bounds: Tuple[float, float] = (0.02, 1.0),
                 H_bounds: Tuple[int, int] = (6, 576),
                 H_alignment_bias: float = 0.8):
        """
        Initialize parameter space.

        Args:
            mu_bounds: Bounds for L1 weight parameter
            nu_bounds: Bounds for deficit weight parameter
            H_bounds: Bounds for horizon parameter
            H_alignment_bias: Probability bias toward 6-step aligned values
        """
        self.mu_bounds = mu_bounds
        self.nu_bounds = nu_bounds
        self.H_bounds = H_bounds
        self.H_alignment_bias = H_alignment_bias

        # Pre-compute preferred H values (multiples of 6)
        self.aligned_H_values = [h for h in range(H_bounds[0], H_bounds[1] + 1) if h % 6 == 0]

    def sample_individual(self) -> Individual:
        """
        Sample a random individual from the parameter space with intelligent biases.
        """
        # Sample mu uniformly
        mu = np.random.uniform(*self.mu_bounds)

        # Sample nu uniformly
        nu = np.random.uniform(*self.nu_bounds)

        # Sample H with bias toward 6-step alignment
        if np.random.random() < self.H_alignment_bias and self.aligned_H_values:
            # Choose from aligned values
            H = np.random.choice(self.aligned_H_values)
        else:
            # Choose uniformly from range
            H = np.random.randint(*self.H_bounds)

        return Individual(mu=mu, nu=nu, H=H)

    def clip_individual(self, individual: Individual) -> Individual:
        """
        Clip individual parameters to valid bounds.
        """
        mu_clipped = np.clip(individual.mu, *self.mu_bounds)
        nu_clipped = np.clip(individual.nu, *self.nu_bounds)
        H_clipped = int(np.clip(individual.H, *self.H_bounds))

        return Individual(mu=mu_clipped, nu=nu_clipped, H=H_clipped)


class EvolutionOperators:
    """
    Genetic operators for NSGA-II evolution.
    """

    def __init__(self,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9,
                 eta_m: float = 20.0,    # Mutation distribution index
                 eta_c: float = 20.0):   # Crossover distribution index
        """
        Initialize evolution operators.

        Args:
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            eta_m: Mutation distribution index (higher = more localized)
            eta_c: Crossover distribution index (higher = more localized)
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.eta_m = eta_m
        self.eta_c = eta_c

    def simulated_binary_crossover(self, parent1: Individual, parent2: Individual,
                                 parameter_space: ParameterSpace) -> Tuple[Individual, Individual]:
        """
        Simulated Binary Crossover (SBX) for continuous parameters.
        """
        if np.random.random() > self.crossover_rate:
            return parent1, parent2

        # Crossover for continuous parameters (mu, nu)
        beta = self._get_beta_value(self.eta_c)

        # mu crossover
        mu1 = 0.5 * ((1 + beta) * parent1.mu + (1 - beta) * parent2.mu)
        mu2 = 0.5 * ((1 - beta) * parent1.mu + (1 + beta) * parent2.mu)

        # nu crossover
        nu1 = 0.5 * ((1 + beta) * parent1.nu + (1 - beta) * parent2.nu)
        nu2 = 0.5 * ((1 - beta) * parent1.nu + (1 + beta) * parent2.nu)

        # H crossover (discrete)
        if np.random.random() < 0.5:
            H1, H2 = parent1.H, parent2.H
        else:
            H1, H2 = parent2.H, parent1.H

        child1 = Individual(mu=mu1, nu=nu1, H=H1)
        child2 = Individual(mu=mu2, nu=nu2, H=H2)

        # Clip to bounds
        child1 = parameter_space.clip_individual(child1)
        child2 = parameter_space.clip_individual(child2)

        return child1, child2

    def polynomial_mutation(self, individual: Individual,
                          parameter_space: ParameterSpace) -> Individual:
        """
        Polynomial mutation for continuous parameters.
        """
        mu = individual.mu
        nu = individual.nu
        H = individual.H

        # Mutate mu
        if np.random.random() < self.mutation_rate:
            delta = self._get_mutation_delta(mu, parameter_space.mu_bounds, self.eta_m)
            mu = mu + delta

        # Mutate nu
        if np.random.random() < self.mutation_rate:
            delta = self._get_mutation_delta(nu, parameter_space.nu_bounds, self.eta_m)
            nu = nu + delta

        # Mutate H (discrete mutation)
        if np.random.random() < self.mutation_rate:
            # Small perturbation with bias toward 6-aligned values
            if np.random.random() < 0.3:  # 30% chance of large jump to aligned value
                if parameter_space.aligned_H_values:
                    H = np.random.choice(parameter_space.aligned_H_values)
            else:  # 70% chance of small perturbation
                perturbation = np.random.choice([-18, -12, -6, 6, 12, 18])  # 6-aligned perturbations
                H = H + perturbation

        mutated = Individual(mu=mu, nu=nu, H=H)
        return parameter_space.clip_individual(mutated)

    def _get_beta_value(self, eta: float) -> float:
        """Generate beta value for SBX crossover."""
        u = np.random.random()

        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (eta + 1))
        else:
            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))

        return beta

    def _get_mutation_delta(self, value: float, bounds: Tuple[float, float], eta: float) -> float:
        """Generate mutation delta for polynomial mutation."""
        delta_max = bounds[1] - bounds[0]

        # Normalized value
        y = (value - bounds[0]) / delta_max

        u = np.random.random()

        if u < 0.5:
            delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1
        else:
            delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))

        return delta_q * delta_max


def evaluate_individual_wrapper(args) -> Individual:
    """
    Wrapper function for parallel evaluation of individuals.

    This function is defined at module level to support multiprocessing.
    """
    individual, scenario_data, simulation_params, production_constraints = args
    return evaluate_individual(individual, scenario_data, simulation_params, production_constraints)


def evaluate_individual(individual: Individual,
                       scenario_data: np.ndarray,
                       simulation_params: dict,
                       production_constraints: bool = True) -> Individual:
    """
    Evaluate an individual by running simulation and calculating objectives.

    Args:
        individual: Individual to evaluate
        scenario_data: L1 basefee sequence
        simulation_params: Base simulation parameters
        production_constraints: Whether to apply production constraints

    Returns:
        Individual with evaluated objectives and metadata
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

        # Calculate enhanced metrics
        enhanced_calc = EnhancedMetricsCalculator(simulation_params['target_balance'])
        enhanced_metrics = enhanced_calc.calculate_all_metrics(
            df, {'mu': individual.mu, 'nu': individual.nu, 'H': individual.H}
        )

        # Get Pareto objectives (negated for minimization)
        ux_obj, stability_obj, efficiency_obj = enhanced_metrics.get_pareto_objectives()

        # Update individual
        individual.ux_objective = ux_obj
        individual.stability_objective = stability_obj
        individual.efficiency_objective = efficiency_obj
        individual.enhanced_metrics = enhanced_metrics
        individual.simulation_time = time.time() - start_time

        # Apply constraints if requested
        if production_constraints:
            from analysis.objective_functions import OptimizationStrategy, ObjectiveFunctionSuite
            suite = ObjectiveFunctionSuite()
            production_func = suite.production_functions[f'{OptimizationStrategy.BALANCED}_production']

            individual.is_feasible = production_func.is_feasible(enhanced_metrics)
            constraint_violations = production_func.evaluate_constraints(enhanced_metrics)
            individual.constraint_violations = sum(1 for satisfied in constraint_violations.values() if not satisfied)

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


class NSGAII:
    """
    NSGA-II Multi-Objective Evolutionary Algorithm implementation.
    """

    def __init__(self,
                 population_size: int = 100,
                 max_generations: int = 50,
                 parameter_space: Optional[ParameterSpace] = None,
                 evolution_ops: Optional[EvolutionOperators] = None,
                 n_workers: int = 4):
        """
        Initialize NSGA-II optimizer.

        Args:
            population_size: Size of population (must be even)
            max_generations: Maximum number of generations
            parameter_space: Parameter space configuration
            evolution_ops: Evolution operators configuration
            n_workers: Number of parallel workers for evaluation
        """
        self.population_size = population_size if population_size % 2 == 0 else population_size + 1
        self.max_generations = max_generations
        self.parameter_space = parameter_space or ParameterSpace()
        self.evolution_ops = evolution_ops or EvolutionOperators()
        self.n_workers = min(n_workers, os.cpu_count() or 1)

        # Evolution tracking
        self.generation = 0
        self.population: List[Individual] = []
        self.history: List[Dict] = []

    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Fast non-dominated sorting algorithm.

        Returns:
            List of fronts, where each front is a list of individuals
        """
        # Reset domination data
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []

        # Calculate domination relationships
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i != j:
                    if p.dominates(q):
                        p.dominated_solutions.append(j)
                    elif q.dominates(p):
                        p.domination_count += 1

        # Find first front (non-dominated solutions)
        fronts = [[]]
        for individual in population:
            if individual.domination_count == 0:
                individual.rank = 0
                fronts[0].append(individual)

        # Find subsequent fronts
        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []

            for individual in fronts[front_index]:
                for j in individual.dominated_solutions:
                    dominated_individual = population[j]
                    dominated_individual.domination_count -= 1

                    if dominated_individual.domination_count == 0:
                        dominated_individual.rank = front_index + 1
                        next_front.append(dominated_individual)

            fronts.append(next_front)
            front_index += 1

        # Remove empty last front
        if not fronts[-1]:
            fronts.pop()

        return fronts

    def calculate_crowding_distance(self, front: List[Individual]) -> None:
        """
        Calculate crowding distance for individuals in a front.
        """
        if len(front) == 0:
            return

        # Initialize distances
        for individual in front:
            individual.crowding_distance = 0

        # Number of objectives
        n_objectives = 3

        for obj_index in range(n_objectives):
            # Sort front by objective
            if obj_index == 0:  # UX objective
                front.sort(key=lambda x: x.ux_objective if x.ux_objective is not None else float('inf'))
            elif obj_index == 1:  # Stability objective
                front.sort(key=lambda x: x.stability_objective if x.stability_objective is not None else float('inf'))
            else:  # Efficiency objective
                front.sort(key=lambda x: x.efficiency_objective if x.efficiency_objective is not None else float('inf'))

            # Set boundary points to infinity
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calculate distances for intermediate points
            if len(front) > 2:
                # Get objective range
                if obj_index == 0:
                    obj_range = front[-1].ux_objective - front[0].ux_objective
                elif obj_index == 1:
                    obj_range = front[-1].stability_objective - front[0].stability_objective
                else:
                    obj_range = front[-1].efficiency_objective - front[0].efficiency_objective

                if obj_range == 0:
                    continue

                for i in range(1, len(front) - 1):
                    if obj_index == 0:
                        distance = (front[i+1].ux_objective - front[i-1].ux_objective) / obj_range
                    elif obj_index == 1:
                        distance = (front[i+1].stability_objective - front[i-1].stability_objective) / obj_range
                    else:
                        distance = (front[i+1].efficiency_objective - front[i-1].efficiency_objective) / obj_range

                    front[i].crowding_distance += distance

    def environmental_selection(self, population: List[Individual]) -> List[Individual]:
        """
        Environmental selection using non-dominated sorting and crowding distance.
        """
        fronts = self.fast_non_dominated_sort(population)

        selected = []
        for front in fronts:
            if len(selected) + len(front) <= self.population_size:
                # Include entire front
                self.calculate_crowding_distance(front)
                selected.extend(front)
            else:
                # Include part of front based on crowding distance
                remaining_spots = self.population_size - len(selected)
                self.calculate_crowding_distance(front)

                # Sort by crowding distance (descending)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                selected.extend(front[:remaining_spots])
                break

        return selected

    def optimize(self,
                scenario_data: np.ndarray,
                simulation_params: dict,
                production_constraints: bool = True,
                verbose: bool = True) -> Tuple[List[Individual], List[Dict]]:
        """
        Run NSGA-II optimization.

        Args:
            scenario_data: L1 basefee data for evaluation
            simulation_params: Base simulation parameters
            production_constraints: Whether to apply production constraints
            verbose: Whether to print progress

        Returns:
            Tuple of (final_population, optimization_history)
        """
        if verbose:
            print(f"🚀 Starting NSGA-II Optimization")
            print(f"   Population size: {self.population_size}")
            print(f"   Max generations: {self.max_generations}")
            print(f"   Parallel workers: {self.n_workers}")
            print(f"   Scenario length: {len(scenario_data)} steps")

        # Initialize population
        if verbose:
            print("\n🧬 Initializing population...")

        self.population = [self.parameter_space.sample_individual() for _ in range(self.population_size)]

        # Evaluate initial population
        self._evaluate_population(scenario_data, simulation_params, production_constraints, verbose)

        # Evolution loop
        for generation in range(self.max_generations):
            self.generation = generation

            if verbose:
                print(f"\n🧬 Generation {generation + 1}/{self.max_generations}")

            # Create offspring population
            offspring = self._create_offspring()

            # Evaluate offspring
            self._evaluate_population_subset(offspring, scenario_data, simulation_params,
                                           production_constraints, verbose)

            # Environmental selection
            combined_population = self.population + offspring
            self.population = self.environmental_selection(combined_population)

            # Record generation statistics
            self._record_generation_stats(verbose)

        if verbose:
            print("\n✅ Optimization completed!")
            self._print_final_summary()

        return self.population, self.history

    def _evaluate_population(self, scenario_data: np.ndarray, simulation_params: dict,
                           production_constraints: bool, verbose: bool) -> None:
        """Evaluate entire population in parallel."""
        if verbose:
            print(f"   Evaluating {len(self.population)} individuals...")

        evaluation_args = [
            (ind, scenario_data, simulation_params, production_constraints)
            for ind in self.population
        ]

        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                self.population = list(executor.map(evaluate_individual_wrapper, evaluation_args))
        else:
            self.population = [evaluate_individual_wrapper(args) for args in evaluation_args]

    def _evaluate_population_subset(self, individuals: List[Individual],
                                  scenario_data: np.ndarray, simulation_params: dict,
                                  production_constraints: bool, verbose: bool) -> None:
        """Evaluate subset of population in parallel."""
        if verbose:
            print(f"   Evaluating {len(individuals)} offspring...")

        evaluation_args = [
            (ind, scenario_data, simulation_params, production_constraints)
            for ind in individuals
        ]

        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                evaluated = list(executor.map(evaluate_individual_wrapper, evaluation_args))
        else:
            evaluated = [evaluate_individual_wrapper(args) for args in evaluation_args]

        # Update individuals in-place
        for i, evaluated_ind in enumerate(evaluated):
            individuals[i] = evaluated_ind

    def _create_offspring(self) -> List[Individual]:
        """Create offspring population through selection and variation."""
        offspring = []

        # Tournament selection and reproduction
        for _ in range(self.population_size // 2):
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            child1, child2 = self.evolution_ops.simulated_binary_crossover(
                parent1, parent2, self.parameter_space
            )

            # Mutation
            child1 = self.evolution_ops.polynomial_mutation(child1, self.parameter_space)
            child2 = self.evolution_ops.polynomial_mutation(child2, self.parameter_space)

            offspring.extend([child1, child2])

        return offspring

    def _tournament_selection(self, tournament_size: int = 2) -> Individual:
        """Binary tournament selection based on NSGA-II criteria."""
        candidates = np.random.choice(self.population, size=tournament_size, replace=False)

        # Compare based on rank first, then crowding distance
        best = candidates[0]
        for candidate in candidates[1:]:
            if self._is_better(candidate, best):
                best = candidate

        return best

    def _is_better(self, ind1: Individual, ind2: Individual) -> bool:
        """Compare two individuals using NSGA-II criteria."""
        # First compare feasibility
        if ind1.is_feasible and not ind2.is_feasible:
            return True
        if not ind1.is_feasible and ind2.is_feasible:
            return False

        # If both infeasible, prefer fewer violations
        if not ind1.is_feasible and not ind2.is_feasible:
            return ind1.constraint_violations < ind2.constraint_violations

        # Both feasible - compare rank and crowding distance
        if ind1.rank != ind2.rank:
            return ind1.rank < ind2.rank  # Lower rank is better

        return ind1.crowding_distance > ind2.crowding_distance  # Higher crowding distance is better

    def _record_generation_stats(self, verbose: bool) -> None:
        """Record statistics for current generation."""
        # Calculate fronts for stats
        fronts = self.fast_non_dominated_sort(self.population)

        feasible_count = sum(1 for ind in self.population if ind.is_feasible)
        avg_simulation_time = np.mean([ind.simulation_time for ind in self.population if ind.simulation_time])

        stats = {
            'generation': self.generation,
            'population_size': len(self.population),
            'feasible_count': feasible_count,
            'front_0_size': len(fronts[0]) if fronts else 0,
            'total_fronts': len(fronts),
            'avg_simulation_time': avg_simulation_time
        }

        # Add objective statistics for feasible individuals
        feasible_inds = [ind for ind in self.population if ind.is_feasible]
        if feasible_inds:
            stats.update({
                'avg_ux_objective': np.mean([ind.ux_objective for ind in feasible_inds]),
                'avg_stability_objective': np.mean([ind.stability_objective for ind in feasible_inds]),
                'avg_efficiency_objective': np.mean([ind.efficiency_objective for ind in feasible_inds]),
            })

        self.history.append(stats)

        if verbose:
            print(f"   Feasible: {feasible_count}/{len(self.population)} | "
                  f"Pareto front: {stats['front_0_size']} | "
                  f"Fronts: {stats['total_fronts']} | "
                  f"Avg eval time: {avg_simulation_time:.2f}s")

    def _print_final_summary(self) -> None:
        """Print final optimization summary."""
        fronts = self.fast_non_dominated_sort(self.population)
        pareto_front = fronts[0] if fronts else []

        feasible_pareto = [ind for ind in pareto_front if ind.is_feasible]

        print(f"\n📊 Final Results:")
        print(f"   Total individuals: {len(self.population)}")
        print(f"   Feasible individuals: {sum(1 for ind in self.population if ind.is_feasible)}")
        print(f"   Pareto front size: {len(pareto_front)}")
        print(f"   Feasible Pareto solutions: {len(feasible_pareto)}")

        if feasible_pareto:
            print(f"\n🎯 Best Feasible Pareto Solutions:")
            for i, ind in enumerate(feasible_pareto[:5]):  # Show top 5
                print(f"   {i+1}. μ={ind.mu:.3f}, ν={ind.nu:.3f}, H={ind.H} | "
                      f"UX={-ind.ux_objective:.3f}, Stab={-ind.stability_objective:.3f}, Eff={-ind.efficiency_objective:.3f}")

    def get_pareto_front(self, feasible_only: bool = True) -> List[Individual]:
        """
        Get the Pareto front from current population.

        Args:
            feasible_only: Whether to return only feasible solutions

        Returns:
            List of individuals on Pareto front
        """
        fronts = self.fast_non_dominated_sort(self.population)
        pareto_front = fronts[0] if fronts else []

        if feasible_only:
            return [ind for ind in pareto_front if ind.is_feasible]

        return pareto_front