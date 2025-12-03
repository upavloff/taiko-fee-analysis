"""
Canonical Taiko Fee Mechanism Optimization Implementation

This module provides the SINGLE SOURCE OF TRUTH for all optimization algorithms.
All optimization-related components must use this module to ensure consistency.

Key Features:
- NSGA-II multi-objective optimization (authoritative implementation)
- Configurable objective functions for different stakeholder priorities
- Pareto frontier generation and analysis
- Parameter bounds and constraint validation
- Parallel evaluation for performance
- Comprehensive convergence metrics

Optimization Strategies:
- USER_CENTRIC: Maximize user experience (low fees, stability)
- PROTOCOL_CENTRIC: Maximize protocol safety (vault resilience)
- BALANCED: Balance all objectives equally
- CRISIS_RESILIENT: Optimize for extreme market conditions

Usage:
    optimizer = CanonicalOptimizer()
    results = optimizer.optimize(
        strategy=OptimizationStrategy.BALANCED,
        population_size=100,
        generations=50
    )
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
import warnings
import time

from .canonical_fee_mechanism import (
    CanonicalTaikoFeeCalculator,
    FeeParameters,
    VaultState,
    VaultInitMode
)


class OptimizationStrategy(Enum):
    """Predefined optimization strategies for different deployment scenarios."""
    USER_CENTRIC = "user_centric"           # Maximize user experience
    PROTOCOL_CENTRIC = "protocol_centric"   # Maximize protocol robustness
    BALANCED = "balanced"                   # Balance all concerns
    CRISIS_RESILIENT = "crisis_resilient"   # Prepare for extreme volatility
    CAPITAL_EFFICIENT = "capital_efficient" # Minimize capital requirements


@dataclass
class OptimizationBounds:
    """Parameter bounds for optimization search space."""

    mu_min: float = 0.0
    mu_max: float = 1.0
    nu_min: float = 0.0
    nu_max: float = 1.0
    H_min: int = 24          # ~1 minute (24 * 2s steps)
    H_max: int = 1800        # ~1 hour (1800 * 2s steps)

    # Constraint: H must be multiple of batch_interval for resonance
    batch_interval: int = 6  # Enforce 6-step alignment


@dataclass
class Individual:
    """Individual solution in the optimization population."""

    # Parameters (search variables)
    mu: float
    nu: float
    H: int

    # Objectives (to be minimized - negative of actual objectives for maximization)
    objectives: Optional[List[float]] = None

    # Optimization metadata
    rank: int = -1                    # Pareto dominance rank
    crowding_distance: float = 0.0    # Diversity measure
    constraint_violation: float = 0.0 # Constraint penalty

    # Evaluation metadata
    simulation_time: float = 0.0
    evaluation_id: Optional[str] = None


@dataclass
class ObjectiveWeights:
    """Weights for different objective categories."""

    # User Experience objectives
    fee_affordability: float = 1.0      # Prefer lower average fees
    fee_stability: float = 1.0          # Prefer stable fees (low CV)
    fee_predictability_1h: float = 1.0  # Short-term predictability
    fee_predictability_6h: float = 0.5  # Long-term predictability

    # Protocol Safety objectives
    insolvency_protection: float = 1.0   # Prevent vault depletion
    deficit_duration: float = 1.0       # Minimize underfunded time
    vault_stress: float = 1.0           # Stress test resilience
    underfunding_resistance: float = 0.5 # General underfunding avoidance

    # Economic Efficiency objectives
    vault_utilization: float = 1.0      # Efficient capital usage
    deficit_correction: float = 1.0     # Fast deficit recovery
    capital_efficiency: float = 1.0     # Performance per capital unit

    @classmethod
    def for_strategy(cls, strategy: OptimizationStrategy) -> 'ObjectiveWeights':
        """Create objective weights for specific optimization strategy."""

        if strategy == OptimizationStrategy.USER_CENTRIC:
            return cls(
                # Emphasize user experience
                fee_affordability=2.0,
                fee_stability=2.0,
                fee_predictability_1h=1.5,
                fee_predictability_6h=1.0,
                # Minimum safety requirements
                insolvency_protection=1.0,
                deficit_duration=0.5,
                vault_stress=0.5,
                # Efficiency secondary
                vault_utilization=0.5,
                deficit_correction=0.5,
                capital_efficiency=0.5
            )

        elif strategy == OptimizationStrategy.PROTOCOL_CENTRIC:
            return cls(
                # Basic user experience
                fee_affordability=0.5,
                fee_stability=1.0,
                fee_predictability_1h=0.5,
                fee_predictability_6h=0.5,
                # Maximize protocol safety
                insolvency_protection=2.0,
                deficit_duration=2.0,
                vault_stress=2.0,
                underfunding_resistance=1.5,
                # High efficiency for safety
                vault_utilization=1.5,
                deficit_correction=2.0,
                capital_efficiency=1.0
            )

        elif strategy == OptimizationStrategy.CRISIS_RESILIENT:
            return cls(
                # Accept higher fees for safety
                fee_affordability=0.3,
                fee_stability=1.5,
                fee_predictability_1h=1.0,
                fee_predictability_6h=0.5,
                # Maximum safety focus
                insolvency_protection=3.0,
                deficit_duration=3.0,
                vault_stress=3.0,
                underfunding_resistance=2.0,
                # Rapid recovery essential
                vault_utilization=1.0,
                deficit_correction=3.0,
                capital_efficiency=1.5
            )

        elif strategy == OptimizationStrategy.CAPITAL_EFFICIENT:
            return cls(
                # Moderate user experience
                fee_affordability=1.0,
                fee_stability=1.0,
                fee_predictability_1h=1.0,
                fee_predictability_6h=0.5,
                # Basic safety
                insolvency_protection=1.0,
                deficit_duration=1.0,
                vault_stress=1.0,
                underfunding_resistance=0.5,
                # Maximum efficiency focus
                vault_utilization=2.0,
                deficit_correction=1.5,
                capital_efficiency=3.0
            )

        else:  # BALANCED
            return cls()  # All weights = 1.0


class CanonicalOptimizer:
    """
    SINGLE SOURCE OF TRUTH for Taiko fee mechanism optimization.

    Implements NSGA-II multi-objective optimization with configurable
    objectives and constraints for different deployment scenarios.
    """

    def __init__(self, bounds: Optional[OptimizationBounds] = None):
        """Initialize optimizer with parameter bounds."""
        self.bounds = bounds or OptimizationBounds()

        # Simulation configuration
        self.simulation_steps = 1800  # 1 hour at 2s per step
        self.use_parallel_evaluation = True
        self.max_workers = 4

        # Algorithm configuration
        self.crossover_rate = 0.9
        self.mutation_rate = 0.1
        self.tournament_size = 2

        # Convergence tracking
        self.convergence_history: List[Dict] = []

    def optimize(self,
                strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                population_size: int = 100,
                generations: int = 50,
                l1_data: Optional[List[float]] = None,
                vault_init: VaultInitMode = VaultInitMode.TARGET,
                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run NSGA-II optimization for specified strategy.

        Args:
            strategy: Optimization strategy defining objective priorities
            population_size: Number of individuals per generation
            generations: Number of evolutionary generations
            l1_data: Historical L1 basefee data (wei) for simulation
            vault_init: Vault initialization mode
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary containing optimization results and analysis
        """

        start_time = time.time()

        # Initialize population
        population = self._initialize_population(population_size)

        # Get objective weights for strategy
        weights = ObjectiveWeights.for_strategy(strategy)

        # Evaluate initial population
        population = self._evaluate_population(population, weights, l1_data, vault_init)

        # Evolution loop
        for gen in range(generations):
            # Progress callback
            if progress_callback:
                progress_callback(gen, generations, population)

            # Create offspring through selection, crossover, mutation
            offspring = self._create_offspring(population, population_size)

            # Evaluate offspring
            offspring = self._evaluate_population(offspring, weights, l1_data, vault_init)

            # Combine parent and offspring populations
            combined = population + offspring

            # Environmental selection (NSGA-II)
            population = self._environmental_selection(combined, population_size)

            # Track convergence metrics
            self._update_convergence_history(gen, population)

        # Extract final results
        pareto_front = [ind for ind in population if ind.rank == 0]

        return {
            'pareto_front': pareto_front,
            'final_population': population,
            'strategy': strategy,
            'convergence_history': self.convergence_history,
            'optimization_time': time.time() - start_time,
            'hypervolume': self._calculate_hypervolume(pareto_front),
            'spread': self._calculate_spread(pareto_front),
            'n_pareto_solutions': len(pareto_front)
        }

    def _initialize_population(self, size: int) -> List[Individual]:
        """Initialize random population within bounds."""
        population = []

        for _ in range(size):
            # Random parameters within bounds
            mu = np.random.uniform(self.bounds.mu_min, self.bounds.mu_max)
            nu = np.random.uniform(self.bounds.nu_min, self.bounds.nu_max)

            # H with batch interval alignment
            h_range = self.bounds.H_max - self.bounds.H_min
            h_steps = h_range // self.bounds.batch_interval
            h_multiplier = np.random.randint(0, h_steps + 1)
            H = self.bounds.H_min + h_multiplier * self.bounds.batch_interval

            individual = Individual(mu=mu, nu=nu, H=H)
            population.append(individual)

        return population

    def _evaluate_population(self,
                           population: List[Individual],
                           weights: ObjectiveWeights,
                           l1_data: Optional[List[float]],
                           vault_init: VaultInitMode) -> List[Individual]:
        """Evaluate population objectives using simulation."""

        if self.use_parallel_evaluation:
            return self._evaluate_parallel(population, weights, l1_data, vault_init)
        else:
            return self._evaluate_sequential(population, weights, l1_data, vault_init)

    def _evaluate_individual(self,
                           individual: Individual,
                           weights: ObjectiveWeights,
                           l1_data: Optional[List[float]],
                           vault_init: VaultInitMode) -> Individual:
        """Evaluate single individual through simulation."""

        start_time = time.time()

        try:
            # Create fee calculator with individual's parameters
            params = FeeParameters(
                mu=individual.mu,
                nu=individual.nu,
                H=individual.H
            )
            calculator = CanonicalTaikoFeeCalculator(params)

            # Create vault
            vault = calculator.create_vault(vault_init, deficit_ratio=0.2)

            # Run simulation
            simulation_results = self._run_simulation(calculator, vault, l1_data)

            # Calculate objectives
            objectives = self._calculate_objectives(simulation_results, weights)
            individual.objectives = objectives

            # Calculate constraint violations
            individual.constraint_violation = self._calculate_constraint_violation(individual)

        except Exception as e:
            # Handle evaluation failures gracefully
            warnings.warn(f"Individual evaluation failed: {e}")
            individual.objectives = [float('inf')] * 11  # Worst possible values
            individual.constraint_violation = float('inf')

        individual.simulation_time = time.time() - start_time
        return individual

    def _evaluate_sequential(self, population, weights, l1_data, vault_init):
        """Sequential evaluation for debugging or small populations."""
        return [self._evaluate_individual(ind, weights, l1_data, vault_init)
                for ind in population]

    def _evaluate_parallel(self, population, weights, l1_data, vault_init):
        """Parallel evaluation for performance."""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._evaluate_individual, ind, weights, l1_data, vault_init)
                for ind in population
            ]
            return [future.result() for future in futures]

    def _run_simulation(self,
                       calculator: CanonicalTaikoFeeCalculator,
                       vault: VaultState,
                       l1_data: Optional[List[float]]) -> Dict[str, List[float]]:
        """Run fee mechanism simulation for evaluation."""

        # Use provided L1 data or generate synthetic data
        if l1_data is None:
            l1_data = self._generate_synthetic_l1_data()

        # Ensure we have enough data
        if len(l1_data) < self.simulation_steps:
            # Extend by repeating the pattern
            repeats = (self.simulation_steps // len(l1_data)) + 1
            l1_data = (l1_data * repeats)[:self.simulation_steps]

        # Simulation results storage
        results = {
            'timeStep': [],
            'l1Basefee': [],
            'estimatedFee': [],
            'transactionVolume': [],
            'vaultBalance': [],
            'vaultDeficit': [],
            'feesCollected': [],
            'l1CostsPaid': []
        }

        # Run simulation
        for step in range(self.simulation_steps):
            l1_basefee_wei = l1_data[step]

            # Calculate L1 cost and estimated fee
            l1_cost = calculator.calculate_l1_cost_per_tx(l1_basefee_wei)
            estimated_fee = calculator.calculate_estimated_fee(l1_cost, vault.deficit)

            # Calculate transaction volume
            tx_volume = calculator.calculate_transaction_volume(estimated_fee)

            # Collect fees (every step - 2s intervals)
            fees_collected = estimated_fee * tx_volume
            vault.collect_fees(fees_collected)

            # Pay L1 costs (every batch_interval steps - 12s intervals)
            l1_costs_paid = 0.0
            if step % calculator.params.batch_interval_steps == 0:
                l1_costs_paid = calculator.calculate_l1_batch_cost(l1_basefee_wei)
                vault.pay_l1_costs(l1_costs_paid)

            # Record results
            results['timeStep'].append(step)
            results['l1Basefee'].append(l1_basefee_wei / 1e9)  # Convert to gwei
            results['estimatedFee'].append(estimated_fee)
            results['transactionVolume'].append(tx_volume)
            results['vaultBalance'].append(vault.balance)
            results['vaultDeficit'].append(vault.deficit)
            results['feesCollected'].append(fees_collected)
            results['l1CostsPaid'].append(l1_costs_paid)

        return results

    def _generate_synthetic_l1_data(self) -> List[float]:
        """Generate synthetic L1 basefee data for evaluation."""
        # Simple GBM model for L1 basefee simulation
        np.random.seed(42)  # Deterministic for fair comparison

        initial_basefee = 15e9  # 15 gwei in wei
        volatility = 0.3
        drift = 0.0
        dt = 1.0 / (24 * 3600 / 2)  # 2-second steps

        basefees = [initial_basefee]
        for _ in range(self.simulation_steps - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            dS = drift * basefees[-1] * dt + volatility * basefees[-1] * dW
            new_basefee = max(basefees[-1] + dS, 1e6)  # Floor at 0.001 gwei
            basefees.append(new_basefee)

        return basefees

    def _calculate_objectives(self,
                            simulation_results: Dict[str, List[float]],
                            weights: ObjectiveWeights) -> List[float]:
        """Calculate weighted objective values from simulation results."""

        fees = np.array(simulation_results['estimatedFee'])
        vault_balances = np.array(simulation_results['vaultBalance'])
        vault_deficits = np.array(simulation_results['vaultDeficit'])

        # User Experience Objectives (higher is better, so negate for minimization)
        fee_affordability = -self._calculate_fee_affordability(fees) * weights.fee_affordability
        fee_stability = -self._calculate_fee_stability(fees) * weights.fee_stability
        fee_pred_1h = -self._calculate_fee_predictability(fees, 1800) * weights.fee_predictability_1h  # 1h
        fee_pred_6h = -self._calculate_fee_predictability(fees, 10800) * weights.fee_predictability_6h  # 6h

        # Protocol Safety Objectives (higher is better, so negate for minimization)
        insolvency_protection = -self._calculate_insolvency_protection(vault_balances) * weights.insolvency_protection
        deficit_duration = -self._calculate_deficit_duration(vault_deficits) * weights.deficit_duration
        vault_stress = -self._calculate_vault_stress_resilience(vault_balances) * weights.vault_stress
        underfunding_resistance = -self._calculate_underfunding_resistance(vault_deficits) * weights.underfunding_resistance

        # Economic Efficiency Objectives (higher is better, so negate for minimization)
        vault_utilization = -self._calculate_vault_utilization(vault_balances) * weights.vault_utilization
        deficit_correction = -self._calculate_deficit_correction_rate(vault_deficits) * weights.deficit_correction
        capital_efficiency = -self._calculate_capital_efficiency(simulation_results) * weights.capital_efficiency

        return [
            fee_affordability, fee_stability, fee_pred_1h, fee_pred_6h,
            insolvency_protection, deficit_duration, vault_stress, underfunding_resistance,
            vault_utilization, deficit_correction, capital_efficiency
        ]

    def _calculate_fee_affordability(self, fees: np.ndarray) -> float:
        """Calculate fee affordability score (higher is better)."""
        avg_fee_gwei = np.mean(fees) * 1e9  # Convert to gwei
        # Logarithmic penalty for high fees
        affordability = max(0.0, 1.0 - np.log10(max(avg_fee_gwei, 0.1)))
        return affordability

    def _calculate_fee_stability(self, fees: np.ndarray) -> float:
        """Calculate fee stability score (higher is better)."""
        if len(fees) < 2:
            return 1.0
        cv = np.std(fees) / (np.mean(fees) + 1e-8)
        stability = max(0.0, 1.0 - cv)
        return stability

    def _calculate_fee_predictability(self, fees: np.ndarray, window_size: int) -> float:
        """Calculate fee predictability over specified window."""
        if len(fees) < window_size:
            return self._calculate_fee_stability(fees)

        predictabilities = []
        for i in range(len(fees) - window_size + 1):
            window = fees[i:i + window_size]
            cv = np.std(window) / (np.mean(window) + 1e-8)
            predictability = max(0.0, 1.0 - cv)
            predictabilities.append(predictability)

        return np.mean(predictabilities)

    def _calculate_insolvency_protection(self, vault_balances: np.ndarray) -> float:
        """Calculate insolvency protection score."""
        min_balance = np.min(vault_balances)
        if min_balance >= 0:
            return 1.0
        else:
            # Penalize based on depth of insolvency
            return max(0.0, 1.0 + min_balance / 1000.0)  # Assuming 1000 ETH target

    def _calculate_deficit_duration(self, vault_deficits: np.ndarray) -> float:
        """Calculate deficit duration control score."""
        deficit_steps = np.sum(vault_deficits > 0.01)  # > 1% of target
        deficit_fraction = deficit_steps / len(vault_deficits)
        return max(0.0, 1.0 - deficit_fraction)

    def _calculate_vault_stress_resilience(self, vault_balances: np.ndarray) -> float:
        """Calculate vault stress resilience score."""
        # Measure recovery speed after stress events
        balance_changes = np.diff(vault_balances)
        recovery_periods = []

        in_stress = False
        stress_start = 0

        for i, balance in enumerate(vault_balances):
            if balance < 800 and not in_stress:  # 20% below target
                in_stress = True
                stress_start = i
            elif balance >= 900 and in_stress:  # 10% below target
                in_stress = False
                recovery_periods.append(i - stress_start)

        if not recovery_periods:
            return 1.0

        avg_recovery_time = np.mean(recovery_periods)
        resilience = max(0.0, 1.0 - avg_recovery_time / 100.0)  # Normalize by 100 steps
        return resilience

    def _calculate_underfunding_resistance(self, vault_deficits: np.ndarray) -> float:
        """Calculate underfunding resistance score."""
        max_deficit_ratio = np.max(vault_deficits) / 1000.0  # Normalize by target balance
        resistance = max(0.0, 1.0 - max_deficit_ratio)
        return resistance

    def _calculate_vault_utilization(self, vault_balances: np.ndarray) -> float:
        """Calculate vault utilization efficiency."""
        target_balance = 1000.0
        deviations = np.abs(vault_balances - target_balance) / target_balance
        avg_deviation = np.mean(deviations)
        utilization = max(0.0, 1.0 - avg_deviation)
        return utilization

    def _calculate_deficit_correction_rate(self, vault_deficits: np.ndarray) -> float:
        """Calculate deficit correction speed."""
        if np.sum(vault_deficits > 0) == 0:
            return 1.0

        # Find deficit correction episodes
        deficit_reductions = []
        in_deficit = False
        deficit_start = 0
        max_deficit = 0

        for i, deficit in enumerate(vault_deficits):
            if deficit > 0 and not in_deficit:
                in_deficit = True
                deficit_start = i
                max_deficit = deficit
            elif deficit > 0 and in_deficit:
                max_deficit = max(max_deficit, deficit)
            elif deficit == 0 and in_deficit:
                in_deficit = False
                correction_time = i - deficit_start
                if correction_time > 0:
                    correction_rate = max_deficit / correction_time
                    deficit_reductions.append(correction_rate)

        if not deficit_reductions:
            return 0.0

        avg_correction_rate = np.mean(deficit_reductions)
        # Normalize by reasonable correction rate (e.g., 1 ETH per step)
        return min(1.0, avg_correction_rate)

    def _calculate_capital_efficiency(self, simulation_results: Dict[str, List[float]]) -> float:
        """Calculate capital efficiency score."""
        fees_collected = np.sum(simulation_results['feesCollected'])
        l1_costs_paid = np.sum(simulation_results['l1CostsPaid'])
        vault_balances = np.array(simulation_results['vaultBalance'])

        # Revenue efficiency
        net_revenue = fees_collected - l1_costs_paid
        avg_capital = np.mean(vault_balances)

        if avg_capital == 0:
            return 0.0

        capital_efficiency = net_revenue / avg_capital
        return max(0.0, min(1.0, capital_efficiency))

    def _calculate_constraint_violation(self, individual: Individual) -> float:
        """Calculate constraint violation penalty."""
        violation = 0.0

        # Parameter bounds (should be enforced during generation)
        if individual.mu < self.bounds.mu_min or individual.mu > self.bounds.mu_max:
            violation += 1.0
        if individual.nu < self.bounds.nu_min or individual.nu > self.bounds.nu_max:
            violation += 1.0
        if individual.H < self.bounds.H_min or individual.H > self.bounds.H_max:
            violation += 1.0

        # Batch interval alignment
        if individual.H % self.bounds.batch_interval != 0:
            violation += 1.0

        return violation

    def _create_offspring(self, population: List[Individual], size: int) -> List[Individual]:
        """Create offspring through selection, crossover, and mutation."""
        offspring = []

        for _ in range(size):
            # Tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # Crossover
            if np.random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = Individual(mu=parent1.mu, nu=parent1.nu, H=parent1.H)

            # Mutation
            child = self._mutate(child)

            offspring.append(child)

        return offspring

    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection."""
        tournament = np.random.choice(population, self.tournament_size, replace=False)

        # Select best individual from tournament (lowest rank, highest crowding distance)
        best = tournament[0]
        for individual in tournament[1:]:
            if (individual.rank < best.rank or
                (individual.rank == best.rank and individual.crowding_distance > best.crowding_distance)):
                best = individual

        return best

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Simulated binary crossover for real parameters."""
        eta_c = 20.0  # Crossover distribution index

        # Crossover for mu
        if np.random.random() < 0.5:
            beta = self._calculate_crossover_beta(eta_c)
            mu = 0.5 * ((1 + beta) * parent1.mu + (1 - beta) * parent2.mu)
        else:
            mu = parent1.mu

        # Crossover for nu
        if np.random.random() < 0.5:
            beta = self._calculate_crossover_beta(eta_c)
            nu = 0.5 * ((1 + beta) * parent1.nu + (1 - beta) * parent2.nu)
        else:
            nu = parent1.nu

        # Discrete crossover for H
        H = parent1.H if np.random.random() < 0.5 else parent2.H

        # Ensure bounds
        mu = np.clip(mu, self.bounds.mu_min, self.bounds.mu_max)
        nu = np.clip(nu, self.bounds.nu_min, self.bounds.nu_max)
        H = np.clip(H, self.bounds.H_min, self.bounds.H_max)

        # Ensure H alignment
        H = self._align_to_batch_interval(H)

        return Individual(mu=mu, nu=nu, H=H)

    def _calculate_crossover_beta(self, eta_c: float) -> float:
        """Calculate beta parameter for simulated binary crossover."""
        u = np.random.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
        return beta

    def _mutate(self, individual: Individual) -> Individual:
        """Polynomial mutation for real parameters."""
        eta_m = 50.0  # Mutation distribution index

        # Mutate mu
        if np.random.random() < self.mutation_rate:
            mu = self._polynomial_mutation(individual.mu, self.bounds.mu_min, self.bounds.mu_max, eta_m)
        else:
            mu = individual.mu

        # Mutate nu
        if np.random.random() < self.mutation_rate:
            nu = self._polynomial_mutation(individual.nu, self.bounds.nu_min, self.bounds.nu_max, eta_m)
        else:
            nu = individual.nu

        # Mutate H (discrete)
        if np.random.random() < self.mutation_rate:
            h_steps = (self.bounds.H_max - self.bounds.H_min) // self.bounds.batch_interval
            step_offset = np.random.randint(-2, 3)  # ±2 steps
            new_step = (individual.H - self.bounds.H_min) // self.bounds.batch_interval + step_offset
            new_step = np.clip(new_step, 0, h_steps)
            H = self.bounds.H_min + new_step * self.bounds.batch_interval
        else:
            H = individual.H

        return Individual(mu=mu, nu=nu, H=H)

    def _polynomial_mutation(self, value: float, lower: float, upper: float, eta_m: float) -> float:
        """Polynomial mutation for bounded real parameters."""
        u = np.random.random()

        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))

        mutated = value + delta * (upper - lower)
        return np.clip(mutated, lower, upper)

    def _align_to_batch_interval(self, H: int) -> int:
        """Align H to batch interval for resonance."""
        remainder = H % self.bounds.batch_interval
        if remainder == 0:
            return H
        else:
            # Round to nearest aligned value
            if remainder <= self.bounds.batch_interval // 2:
                return H - remainder
            else:
                return H + (self.bounds.batch_interval - remainder)

    def _environmental_selection(self, population: List[Individual], size: int) -> List[Individual]:
        """NSGA-II environmental selection."""
        # Fast non-dominated sorting
        fronts = self._fast_non_dominated_sort(population)

        # Assign ranks
        for rank, front in enumerate(fronts):
            for individual in front:
                individual.rank = rank

        # Select individuals for next generation
        selected = []
        front_index = 0

        while len(selected) + len(fronts[front_index]) <= size:
            # Add entire front
            for individual in fronts[front_index]:
                selected.append(individual)
            front_index += 1

        # Add remaining individuals from next front using crowding distance
        if len(selected) < size:
            remaining = size - len(selected)
            last_front = fronts[front_index]

            # Calculate crowding distances for last front
            self._calculate_crowding_distance(last_front)

            # Sort by crowding distance (descending)
            last_front.sort(key=lambda x: x.crowding_distance, reverse=True)

            # Add individuals with highest crowding distance
            for i in range(remaining):
                selected.append(last_front[i])

        return selected

    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Fast non-dominated sorting algorithm."""
        fronts = [[]]

        # Initialize domination structures
        for individual in population:
            individual.domination_count = 0
            individual.dominated_individuals = []

        # Compare all pairs
        for i, p in enumerate(population):
            for q in population[i+1:]:
                if self._dominates(p, q):
                    p.dominated_individuals.append(q)
                    q.domination_count += 1
                elif self._dominates(q, p):
                    q.dominated_individuals.append(p)
                    p.domination_count += 1

            # Add to first front if non-dominated
            if p.domination_count == 0:
                fronts[0].append(p)

        # Create subsequent fronts
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []

            for individual in fronts[current_front]:
                for dominated in individual.dominated_individuals:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        next_front.append(dominated)

            if next_front:
                fronts.append(next_front)
            current_front += 1

        # Remove empty fronts
        return [front for front in fronts if front]

    def _dominates(self, individual1: Individual, individual2: Individual) -> bool:
        """Check if individual1 dominates individual2."""
        if individual1.objectives is None or individual2.objectives is None:
            return False

        # Handle constraint violations
        if individual1.constraint_violation < individual2.constraint_violation:
            return True
        elif individual1.constraint_violation > individual2.constraint_violation:
            return False

        # Check objective domination (all objectives are for minimization)
        better_in_any = False
        for obj1, obj2 in zip(individual1.objectives, individual2.objectives):
            if obj1 > obj2:  # Worse in this objective
                return False
            elif obj1 < obj2:  # Better in this objective
                better_in_any = True

        return better_in_any

    def _calculate_crowding_distance(self, front: List[Individual]) -> None:
        """Calculate crowding distance for individuals in a front."""
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return

        # Initialize distances
        for individual in front:
            individual.crowding_distance = 0.0

        n_objectives = len(front[0].objectives)

        # Calculate distance for each objective
        for obj_index in range(n_objectives):
            # Sort front by this objective
            front.sort(key=lambda x: x.objectives[obj_index])

            # Boundary individuals get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calculate objective range
            obj_min = front[0].objectives[obj_index]
            obj_max = front[-1].objectives[obj_index]
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # Calculate distances for intermediate individuals
            for i in range(1, len(front) - 1):
                if front[i].crowding_distance != float('inf'):
                    distance = (front[i + 1].objectives[obj_index] -
                              front[i - 1].objectives[obj_index]) / obj_range
                    front[i].crowding_distance += distance

    def _update_convergence_history(self, generation: int, population: List[Individual]) -> None:
        """Track convergence metrics."""
        pareto_front = [ind for ind in population if ind.rank == 0]

        metrics = {
            'generation': generation,
            'pareto_size': len(pareto_front),
            'hypervolume': self._calculate_hypervolume(pareto_front),
            'spread': self._calculate_spread(pareto_front),
            'avg_objectives': self._calculate_average_objectives(pareto_front)
        }

        self.convergence_history.append(metrics)

    def _calculate_hypervolume(self, front: List[Individual]) -> float:
        """Calculate hypervolume indicator (simplified 2D version)."""
        if not front or not front[0].objectives:
            return 0.0

        # Extract first two objectives for 2D hypervolume
        points = [(ind.objectives[0], ind.objectives[1]) for ind in front]

        # Sort points
        points.sort()

        # Calculate hypervolume with reference point (0, 0)
        hypervolume = 0.0
        prev_x = 0.0

        for x, y in points:
            hypervolume += (x - prev_x) * abs(y)
            prev_x = x

        return hypervolume

    def _calculate_spread(self, front: List[Individual]) -> float:
        """Calculate spread indicator."""
        if len(front) <= 1:
            return 0.0

        # Calculate distances between consecutive solutions
        distances = []
        objectives_matrix = np.array([ind.objectives for ind in front])

        for i in range(len(front) - 1):
            dist = np.linalg.norm(objectives_matrix[i] - objectives_matrix[i + 1])
            distances.append(dist)

        if not distances:
            return 0.0

        mean_distance = np.mean(distances)
        variance = np.var(distances)

        # Spread is coefficient of variation of distances
        return np.sqrt(variance) / (mean_distance + 1e-8)

    def _calculate_average_objectives(self, front: List[Individual]) -> List[float]:
        """Calculate average objective values for front."""
        if not front:
            return []

        objectives_matrix = np.array([ind.objectives for ind in front])
        return np.mean(objectives_matrix, axis=0).tolist()


# Convenience functions for common optimization scenarios
def optimize_for_strategy(strategy: OptimizationStrategy,
                         population_size: int = 100,
                         generations: int = 50) -> Dict[str, Any]:
    """Quick optimization for specific strategy."""
    optimizer = CanonicalOptimizer()
    return optimizer.optimize(
        strategy=strategy,
        population_size=population_size,
        generations=generations
    )


def find_pareto_optimal_parameters(l1_data: Optional[List[float]] = None) -> List[Individual]:
    """Find Pareto optimal parameters using balanced strategy."""
    results = optimize_for_strategy(OptimizationStrategy.BALANCED)
    return results['pareto_front']


def validate_parameter_set(mu: float, nu: float, H: int,
                          l1_data: Optional[List[float]] = None) -> Dict[str, float]:
    """Validate specific parameter set and return objective scores."""
    optimizer = CanonicalOptimizer()
    individual = Individual(mu=mu, nu=nu, H=H)
    weights = ObjectiveWeights.for_strategy(OptimizationStrategy.BALANCED)

    evaluated = optimizer._evaluate_individual(individual, weights, l1_data, VaultInitMode.TARGET)
    return {
        'objectives': evaluated.objectives,
        'constraint_violation': evaluated.constraint_violation,
        'simulation_time': evaluated.simulation_time
    }