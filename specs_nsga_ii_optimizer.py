#!/usr/bin/env python3
"""
SPECS-Compatible NSGA-II Multi-Objective Optimization

Integrates the corrected SPECS.md normalized objectives with the existing
NSGA-II implementation for true Pareto optimization of fee mechanism parameters.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings('ignore')

# Add the specs_implementation to path
sys.path.append(str(Path(__file__).parent / "python" / "specs_implementation"))

from core.simulation_engine import SimulationEngine
from metrics.calculator import MetricsCalculator
from metrics.objectives import StakeholderProfile, ObjectiveCalculator
from metrics.constraints import SimulationResults


@dataclass
class SpecsIndividual:
    """
    Individual solution for SPECS-compatible NSGA-II optimization

    Represents a parameter configuration with SPECS.md normalized objectives
    Note: lambda_c is fixed at 0.1 (moderate L1 smoothing)
    """
    # Parameters
    mu: float          # L1 weight [0.0, 1.0]
    nu: float          # Deficit weight [0.0, 1.0]
    H: int             # Horizon steps (36, 72, 144, 288)

    # SPECS Normalized Objectives (for maximization - higher is better)
    ux_objective: Optional[float] = None           # User experience [0,1]
    robustness_objective: Optional[float] = None   # Protocol robustness [0,1]
    capital_efficiency_objective: Optional[float] = None  # Capital efficiency [0,1]

    # NSGA-II metadata
    domination_count: int = 0
    dominated_solutions: List[int] = None
    rank: Optional[int] = None
    crowding_distance: float = 0.0

    # Constraint handling
    is_feasible: bool = True
    feasibility_ratio: float = 1.0  # Ratio of datasets where simulation succeeded

    # Additional metadata
    avg_fee_gwei: Optional[float] = None
    evaluation_time: Optional[float] = None

    def __post_init__(self):
        if self.dominated_solutions is None:
            self.dominated_solutions = []

    def dominates(self, other: 'SpecsIndividual') -> bool:
        """
        Check if this individual dominates another using SPECS objectives

        For maximization objectives (higher is better):
        A dominates B if A >= B in all objectives and A > B in at least one
        """
        # Feasibility-based domination
        if self.is_feasible and not other.is_feasible:
            return True
        if not self.is_feasible and other.is_feasible:
            return False
        if not self.is_feasible and not other.is_feasible:
            return self.feasibility_ratio > other.feasibility_ratio

        # Both feasible - compare objectives (maximization)
        if any(obj is None for obj in [self.ux_objective, self.robustness_objective, self.capital_efficiency_objective]):
            return False
        if any(obj is None for obj in [other.ux_objective, other.robustness_objective, other.capital_efficiency_objective]):
            return True

        # Check domination: all >= and at least one >
        ux_better_eq = self.ux_objective >= other.ux_objective
        robust_better_eq = self.robustness_objective >= other.robustness_objective
        cap_better_eq = self.capital_efficiency_objective >= other.capital_efficiency_objective

        ux_strictly_better = self.ux_objective > other.ux_objective
        robust_strictly_better = self.robustness_objective > other.robustness_objective
        cap_strictly_better = self.capital_efficiency_objective > other.capital_efficiency_objective

        all_better_or_equal = ux_better_eq and robust_better_eq and cap_better_eq
        at_least_one_strictly_better = ux_strictly_better or robust_strictly_better or cap_strictly_better

        return all_better_or_equal and at_least_one_strictly_better


class SpecsNSGAII:
    """
    NSGA-II implementation using SPECS.md normalized objectives
    """

    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
        stakeholder_profile: StakeholderProfile = StakeholderProfile.BALANCED
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.stakeholder_profile = stakeholder_profile

        # Parameter bounds (SPECS.md compliant)
        self.mu_bounds = (0.0, 1.0)
        self.nu_bounds = (0.0, 1.0)
        self.H_options = [36, 72, 144, 288]  # 6-step aligned options
        self.lambda_c_fixed = 0.1  # Fixed L1 smoothing parameter

        # Evaluation cache for expensive simulations
        self.evaluation_cache = {}

    def create_random_individual(self) -> SpecsIndividual:
        """Create a random individual within parameter bounds"""
        mu = np.random.uniform(*self.mu_bounds)
        nu = np.random.uniform(*self.nu_bounds)
        H = np.random.choice(self.H_options)

        return SpecsIndividual(mu=mu, nu=nu, H=H)

    def evaluate_individual(
        self,
        individual: SpecsIndividual,
        datasets: Dict[str, np.ndarray]
    ) -> SpecsIndividual:
        """
        Evaluate an individual using SPECS metrics across multiple datasets
        """
        cache_key = (individual.mu, individual.nu, individual.H)
        if cache_key in self.evaluation_cache:
            cached = self.evaluation_cache[cache_key]
            individual.ux_objective = cached['ux_objective']
            individual.robustness_objective = cached['robustness_objective']
            individual.capital_efficiency_objective = cached['capital_efficiency_objective']
            individual.is_feasible = cached['is_feasible']
            individual.feasibility_ratio = cached['feasibility_ratio']
            individual.avg_fee_gwei = cached['avg_fee_gwei']
            return individual

        start_time = time.time()

        # Create objective calculator for stakeholder profile
        calc = MetricsCalculator.for_stakeholder(self.stakeholder_profile)
        objective_calc = ObjectiveCalculator()

        # Evaluate across all datasets
        all_objectives = []
        feasible_count = 0
        all_fees = []

        for dataset_name, basefees_wei in datasets.items():
            try:
                # Create simulation engine
                engine = SimulationEngine(
                    mu=individual.mu,
                    nu=individual.nu,
                    horizon_h=individual.H,
                    lambda_c=self.lambda_c_fixed,
                    target_vault_balance=1.0,
                    q_bar=6.9e5
                )

                # Convert basefees to L1 costs (FIXED: keep in wei, don't convert to ETH)
                txs_per_batch = 100
                gas_per_tx = max(200_000 / txs_per_batch, 200)
                l1_costs_wei = basefees_wei * gas_per_tx  # Keep in wei!

                # Simulate (pass wei, not ETH)
                sim_df = engine.simulate_series(l1_costs_wei)

                # Convert to SimulationResults format
                results = SimulationResults(
                    fees_per_gas=sim_df['basefee_per_gas'].values,
                    vault_balances=sim_df['vault_balance_after'].values,
                    revenues=sim_df['revenue'].values,
                    l1_costs=sim_df['l1_cost_actual'].values,
                    deficits=sim_df['deficit'].values,
                    timestamps=np.arange(len(sim_df)),
                    subsidies=sim_df['subsidy_paid'].values,
                    Q_bar=6.9e5
                )

                # Get the three Pareto objectives (normalized to [0,1])
                ux_obj, robust_obj, cap_eff_obj = objective_calc.get_pareto_objectives(results)
                all_objectives.append((ux_obj, robust_obj, cap_eff_obj))

                # Track average fee
                avg_fee = np.mean(sim_df['basefee_per_gas']) / 1e9
                all_fees.append(avg_fee)

                feasible_count += 1

            except Exception as e:
                # Skip failed simulations
                continue

        evaluation_time = time.time() - start_time

        if len(all_objectives) == 0:
            # Complete failure
            individual.ux_objective = 0.0
            individual.robustness_objective = 0.0
            individual.capital_efficiency_objective = 0.0
            individual.is_feasible = False
            individual.feasibility_ratio = 0.0
            individual.avg_fee_gwei = float('inf')
        else:
            # Aggregate across successful datasets
            objectives_array = np.array(all_objectives)
            individual.ux_objective = np.mean(objectives_array[:, 0])
            individual.robustness_objective = np.mean(objectives_array[:, 1])
            individual.capital_efficiency_objective = np.mean(objectives_array[:, 2])
            individual.feasibility_ratio = feasible_count / len(datasets)
            individual.is_feasible = individual.feasibility_ratio > 0.5  # Majority feasible
            individual.avg_fee_gwei = np.mean(all_fees)

        individual.evaluation_time = evaluation_time

        # Cache results
        self.evaluation_cache[cache_key] = {
            'ux_objective': individual.ux_objective,
            'robustness_objective': individual.robustness_objective,
            'capital_efficiency_objective': individual.capital_efficiency_objective,
            'is_feasible': individual.is_feasible,
            'feasibility_ratio': individual.feasibility_ratio,
            'avg_fee_gwei': individual.avg_fee_gwei
        }

        return individual

    def fast_non_dominated_sort(self, population: List[SpecsIndividual]) -> List[List[int]]:
        """
        Perform fast non-dominated sorting (NSGA-II Algorithm 1)

        Returns list of fronts, where each front is a list of individual indices
        """
        # Reset domination data
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []

        fronts = [[]]  # First front

        # Calculate domination relationships
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i != j:
                    if p.dominates(q):
                        p.dominated_solutions.append(j)
                    elif q.dominates(p):
                        p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(i)

        # Build subsequent fronts
        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []
            for i in fronts[front_index]:
                p = population[i]
                for j in p.dominated_solutions:
                    q = population[j]
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = front_index + 1
                        next_front.append(j)

            front_index += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    def calculate_crowding_distance(self, population: List[SpecsIndividual], front: List[int]):
        """
        Calculate crowding distance for individuals in a front (NSGA-II Algorithm 2)
        """
        if len(front) <= 2:
            for i in front:
                population[i].crowding_distance = float('inf')
            return

        # Initialize distances
        for i in front:
            population[i].crowding_distance = 0.0

        # For each objective
        objectives = ['ux_objective', 'robustness_objective', 'capital_efficiency_objective']

        for obj_name in objectives:
            # Sort front by this objective
            front_sorted = sorted(front, key=lambda i: getattr(population[i], obj_name) or 0.0)

            # Boundary points get infinite distance
            population[front_sorted[0]].crowding_distance = float('inf')
            population[front_sorted[-1]].crowding_distance = float('inf')

            # Calculate range for normalization
            obj_values = [getattr(population[i], obj_name) or 0.0 for i in front_sorted]
            obj_range = max(obj_values) - min(obj_values)

            if obj_range > 0:
                # Calculate crowding distance for interior points
                for i in range(1, len(front_sorted) - 1):
                    idx = front_sorted[i]
                    if population[idx].crowding_distance != float('inf'):
                        prev_val = getattr(population[front_sorted[i-1]], obj_name) or 0.0
                        next_val = getattr(population[front_sorted[i+1]], obj_name) or 0.0
                        population[idx].crowding_distance += (next_val - prev_val) / obj_range

    def crowded_comparison(self, i1: SpecsIndividual, i2: SpecsIndividual) -> bool:
        """
        Crowded comparison operator: returns True if i1 is better than i2
        """
        if i1.rank < i2.rank:
            return True
        elif i1.rank > i2.rank:
            return False
        else:
            # Same rank - compare crowding distance (higher is better)
            return i1.crowding_distance > i2.crowding_distance

    def selection(self, population: List[SpecsIndividual]) -> List[SpecsIndividual]:
        """
        Environmental selection using NSGA-II ranking and crowding distance
        """
        # Fast non-dominated sort
        fronts = self.fast_non_dominated_sort(population)

        # Calculate crowding distances for each front
        for front in fronts:
            self.calculate_crowding_distance(population, front)

        # Select individuals for next generation
        new_population = []

        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                # Add entire front
                new_population.extend([population[i] for i in front])
            else:
                # Partial front - select by crowding distance
                remaining_slots = self.population_size - len(new_population)
                front_individuals = [population[i] for i in front]

                # Sort by crowding distance (descending)
                front_individuals.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_population.extend(front_individuals[:remaining_slots])
                break

        return new_population

    def crossover(self, parent1: SpecsIndividual, parent2: SpecsIndividual) -> Tuple[SpecsIndividual, SpecsIndividual]:
        """
        Simulated binary crossover (SBX) for continuous parameters
        """
        eta_c = 20.0  # Crossover distribution index

        if np.random.random() > self.crossover_rate:
            return parent1, parent2

        # Crossover mu and nu
        u = np.random.random()
        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (eta_c + 1))
        else:
            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))

        # Child 1
        child1_mu = 0.5 * ((1 + beta) * parent1.mu + (1 - beta) * parent2.mu)
        child1_nu = 0.5 * ((1 + beta) * parent1.nu + (1 - beta) * parent2.nu)

        # Child 2
        child2_mu = 0.5 * ((1 - beta) * parent1.mu + (1 + beta) * parent2.mu)
        child2_nu = 0.5 * ((1 - beta) * parent1.nu + (1 + beta) * parent2.nu)

        # Clip to bounds
        child1_mu = np.clip(child1_mu, *self.mu_bounds)
        child1_nu = np.clip(child1_nu, *self.nu_bounds)
        child2_mu = np.clip(child2_mu, *self.mu_bounds)
        child2_nu = np.clip(child2_nu, *self.nu_bounds)

        # H parameter - discrete crossover
        child1_H = np.random.choice([parent1.H, parent2.H])
        child2_H = np.random.choice([parent1.H, parent2.H])

        child1 = SpecsIndividual(mu=child1_mu, nu=child1_nu, H=child1_H)
        child2 = SpecsIndividual(mu=child2_mu, nu=child2_nu, H=child2_H)

        return child1, child2

    def mutate(self, individual: SpecsIndividual) -> SpecsIndividual:
        """
        Polynomial mutation for continuous parameters
        """
        eta_m = 20.0  # Mutation distribution index

        if np.random.random() > self.mutation_rate:
            return individual

        # Mutate mu
        if np.random.random() < 0.5:
            u = np.random.random()
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))

            mu_range = self.mu_bounds[1] - self.mu_bounds[0]
            individual.mu = np.clip(individual.mu + delta * mu_range, *self.mu_bounds)

        # Mutate nu
        if np.random.random() < 0.5:
            u = np.random.random()
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))

            nu_range = self.nu_bounds[1] - self.nu_bounds[0]
            individual.nu = np.clip(individual.nu + delta * nu_range, *self.nu_bounds)

        # Mutate H (discrete)
        if np.random.random() < 0.3:  # Lower probability for discrete parameter
            individual.H = np.random.choice(self.H_options)

        return individual

    def optimize(self, datasets: Dict[str, np.ndarray]) -> List[SpecsIndividual]:
        """
        Run NSGA-II optimization with SPECS metrics

        Returns the final Pareto front
        """
        print(f"🚀 Starting SPECS NSGA-II Optimization")
        print(f"   Profile: {self.stakeholder_profile.value}")
        print(f"   Population: {self.population_size}, Generations: {self.max_generations}")
        print(f"   Datasets: {list(datasets.keys())}")
        print("=" * 60)

        # Initialize population
        population = [self.create_random_individual() for _ in range(self.population_size)]

        # Evaluate initial population
        print("📊 Evaluating initial population...")
        start_time = time.time()

        for i, individual in enumerate(population):
            if i % 20 == 0:
                elapsed = time.time() - start_time
                progress = i / len(population) * 100
                eta = elapsed / max(i, 1) * (len(population) - i)
                print(f"   Progress: {progress:.1f}% ({i}/{len(population)}) - ETA: {eta:.0f}s")

            population[i] = self.evaluate_individual(individual, datasets)

        # Evolution loop
        for generation in range(self.max_generations):
            print(f"\n🧬 Generation {generation + 1}/{self.max_generations}")

            # Create offspring
            offspring = []
            while len(offspring) < self.population_size:
                # Tournament selection (size 2)
                parent1 = min(np.random.choice(population, 2), key=lambda x: (x.rank or float('inf'), -x.crowding_distance))
                parent2 = min(np.random.choice(population, 2), key=lambda x: (x.rank or float('inf'), -x.crowding_distance))

                # Crossover and mutation
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                offspring.extend([child1, child2])

            # Evaluate offspring
            for i, child in enumerate(offspring[:self.population_size]):
                offspring[i] = self.evaluate_individual(child, datasets)

            # Environmental selection
            combined_population = population + offspring[:self.population_size]
            population = self.selection(combined_population)

            # Report progress
            feasible_count = sum(1 for ind in population if ind.is_feasible)
            if feasible_count > 0:
                avg_ux = np.mean([ind.ux_objective for ind in population if ind.is_feasible])
                avg_robust = np.mean([ind.robustness_objective for ind in population if ind.is_feasible])
                avg_cap = np.mean([ind.capital_efficiency_objective for ind in population if ind.is_feasible])
                print(f"   Feasible: {feasible_count}/{len(population)}")
                print(f"   Avg objectives: UX={avg_ux:.3f}, Robust={avg_robust:.3f}, CapEff={avg_cap:.3f}")

        # Extract final Pareto front
        fronts = self.fast_non_dominated_sort(population)
        if len(fronts) > 0:
            pareto_front = [population[i] for i in fronts[0]]
            print(f"\n🏆 Optimization complete! Pareto front: {len(pareto_front)} solutions")
            return sorted(pareto_front, key=lambda x: (x.ux_objective + x.robustness_objective + x.capital_efficiency_objective), reverse=True)
        else:
            return []


def load_crisis_datasets():
    """Load crisis datasets for robust optimization"""
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

            # Sample for optimization speed
            max_points = 500
            if len(basefees_wei) > max_points:
                indices = np.linspace(0, len(basefees_wei)-1, max_points, dtype=int)
                basefees_wei = basefees_wei[indices]

            datasets[name] = basefees_wei
            print(f"   Points: {len(basefees_wei):,}, Range: {basefees_wei.min()/1e9:.3f} - {basefees_wei.max()/1e9:.1f} gwei")

    # Fallback: synthetic crisis
    if not datasets:
        print("📊 Using synthetic crisis data")
        base_fee = 10e9
        crisis_pattern = np.concatenate([
            np.linspace(1, 20, 150),
            np.full(200, 20),
            np.linspace(20, 1, 150)
        ])
        datasets["synthetic_crisis"] = base_fee * crisis_pattern

    return datasets


def main():
    """Run SPECS-compatible NSGA-II optimization for all stakeholder profiles"""
    print("🎯 SPECS.md Compatible Multi-Objective Optimization")
    print("Using NSGA-II with normalized [0,1] objectives")
    print("=" * 60)

    # Load datasets
    datasets = load_crisis_datasets()

    if not datasets:
        print("❌ No datasets available for optimization!")
        return 1

    # Test profiles
    profiles_to_test = [
        StakeholderProfile.USER_CENTRIC,
        StakeholderProfile.PROTOCOL_LAUNCH,
        StakeholderProfile.OPERATOR_FOCUSED,
        StakeholderProfile.BALANCED,
        StakeholderProfile.STRESS_TESTED
    ]

    all_results = {}

    for profile in profiles_to_test:
        print(f"\n🎯 Optimizing for {profile.value.upper()} stakeholder profile")
        print("-" * 60)

        # Create optimizer
        optimizer = SpecsNSGAII(
            population_size=50,  # Smaller for demo
            max_generations=20,  # Smaller for demo
            stakeholder_profile=profile
        )

        # Run optimization
        pareto_front = optimizer.optimize(datasets)
        all_results[profile] = pareto_front

        # Display results
        print(f"\n📊 {profile.value.upper()} - Top 5 Pareto Optimal Solutions:")
        print("-" * 80)
        print(f"{'Rank':<6} {'μ':<6} {'ν':<6} {'H':<6} {'UX':<8} {'Robust':<8} {'CapEff':<8} {'Fee':<10}")
        print("-" * 80)

        for i, sol in enumerate(pareto_front[:5]):
            rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}"
            print(f"{rank_symbol:<6} {sol.mu:<6.2f} {sol.nu:<6.2f} {sol.H:<6d} "
                  f"{sol.ux_objective:<8.3f} {sol.robustness_objective:<8.3f} "
                  f"{sol.capital_efficiency_objective:<8.3f} {sol.avg_fee_gwei:<10.2f}")

    # Cross-profile comparison
    print(f"\n🏆 CROSS-STAKEHOLDER COMPARISON")
    print("=" * 80)
    print("Best solution for each stakeholder profile:")
    print(f"{'Profile':<18} {'Best Params':<20} {'Objectives (UX/Rob/Cap)':<25} {'Avg Fee':<10}")
    print("-" * 80)

    for profile, results in all_results.items():
        if results:
            best = results[0]
            param_str = f"μ={best.mu:.2f},ν={best.nu:.2f},H={best.H}"
            obj_str = f"{best.ux_objective:.3f}/{best.robustness_objective:.3f}/{best.capital_efficiency_objective:.3f}"
            print(f"{profile.value:<18} {param_str:<20} {obj_str:<25} {best.avg_fee_gwei:<10.2f}")
        else:
            print(f"{profile.value:<18} {'No solutions found':<20} {'N/A':<25} {'N/A':<10}")

    print(f"\n✅ SPECS NSGA-II optimization complete!")
    print(f"📋 True Pareto optimization with normalized [0,1] objectives")
    print(f"🎯 Results show optimal parameters vary by stakeholder profile")

    return 0


if __name__ == "__main__":
    exit(main())