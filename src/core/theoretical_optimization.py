"""
Theoretical Framework Optimization Implementation

This module implements the EXACT specification from SUMMARY.md for parameter optimization.
It provides a clean separation between the theoretical framework and the existing
canonical implementation, allowing for direct comparison and validation.

Objectives from SUMMARY.md Section 2.3:
- J_UX(θ): User Experience objectives
- J_safe(θ): Safety robustness objectives
- J_eff(θ): Capital efficiency objectives

Hard constraints from Section 2.1-2.2:
- Cost Recovery Ratio: 1-ε ≤ CRR(θ) ≤ 1+ε
- Ruin Probability: ρ_ruin(θ) ≤ ε_ruin
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import time

from .canonical_fee_mechanism import (
    CanonicalTaikoFeeCalculator,
    FeeParameters,
    VaultState,
    VaultInitMode
)


class StakeholderProfile(Enum):
    """Stakeholder profiles with different objective priorities."""
    END_USER = "end_user"              # Minimize fees, maximize predictability
    PROTOCOL_DAO = "protocol_dao"      # Balance all objectives equally
    VAULT_OPERATOR = "vault_operator"  # Minimize capital requirements
    SEQUENCER = "sequencer"            # Maximize revenue stability
    CRISIS_MANAGER = "crisis_manager"  # Maximize robustness


@dataclass
class TheoreticalObjectiveWeights:
    """
    Objective weights following SUMMARY.md specification.

    UX Objectives (Section 2.3.1):
    - a1: Fee stability weight (CV penalty)
    - a2: Fee jumpiness weight (relative change penalty)
    - a3: High fee penalty weight (p95 cap enforcement)

    Safety Objectives (Section 2.3.2):
    - b1: Deficit-weighted duration penalty
    - b2: Maximum deficit depth penalty
    - b3: Recovery time penalty

    Efficiency Objectives (Section 2.3.3):
    - c1: Target size penalty (capital cost)
    - c2: Vault deviation penalty (utilization)
    - c3: Capital efficiency penalty (ROI)
    """
    # UX weights
    a1_stability: float = 1.0
    a2_jumpiness: float = 1.0
    a3_high_fees: float = 1.0

    # Safety weights
    b1_deficit_duration: float = 1.0
    b2_max_deficit: float = 1.0
    b3_recovery_time: float = 1.0

    # Efficiency weights
    c1_target_size: float = 1.0
    c2_vault_deviation: float = 1.0
    c3_capital_efficiency: float = 1.0

    # UX cap for high fee penalty (gwei)
    F_ux_cap: float = 100.0

    @classmethod
    def for_profile(cls, profile: StakeholderProfile) -> 'TheoreticalObjectiveWeights':
        """Create objective weights for specific stakeholder profile."""

        if profile == StakeholderProfile.END_USER:
            # Users prioritize low, stable, predictable fees
            return cls(
                a1_stability=3.0,    # High stability weight
                a2_jumpiness=2.0,    # Moderate jumpiness penalty
                a3_high_fees=5.0,    # Strong high fee penalty
                b1_deficit_duration=0.5,  # Lower safety concern
                b2_max_deficit=0.5,
                b3_recovery_time=0.3,
                c1_target_size=0.1,  # Don't care about capital costs
                c2_vault_deviation=0.1,
                c3_capital_efficiency=0.1,
                F_ux_cap=50.0       # Low fee tolerance
            )

        elif profile == StakeholderProfile.VAULT_OPERATOR:
            # Vault operators want minimal capital with good returns
            return cls(
                a1_stability=0.5,    # Some stability needed
                a2_jumpiness=0.3,
                a3_high_fees=0.3,    # Higher fees = more revenue
                b1_deficit_duration=1.0,  # Moderate safety
                b2_max_deficit=1.5,
                b3_recovery_time=1.0,
                c1_target_size=3.0,  # Minimize capital requirements
                c2_vault_deviation=2.0,  # Efficient capital usage
                c3_capital_efficiency=3.0,  # Maximize returns
                F_ux_cap=200.0      # Higher fee tolerance
            )

        elif profile == StakeholderProfile.SEQUENCER:
            # Sequencers want stable revenue streams
            return cls(
                a1_stability=2.0,    # Stable fees = stable revenue
                a2_jumpiness=1.0,
                a3_high_fees=0.5,    # Higher fees = more tips
                b1_deficit_duration=1.5,  # Need protocol stability
                b2_max_deficit=1.5,
                b3_recovery_time=1.0,
                c1_target_size=1.0,
                c2_vault_deviation=1.5,  # Efficient operations
                c3_capital_efficiency=2.0,  # Revenue focused
                F_ux_cap=150.0
            )

        elif profile == StakeholderProfile.CRISIS_MANAGER:
            # Crisis managers prioritize robustness above all
            return cls(
                a1_stability=1.0,
                a2_jumpiness=0.5,    # Accept volatility for safety
                a3_high_fees=0.2,    # Accept high fees in crisis
                b1_deficit_duration=3.0,  # Maximum safety focus
                b2_max_deficit=3.0,
                b3_recovery_time=3.0,
                c1_target_size=0.5,  # Accept higher capital costs
                c2_vault_deviation=0.5,
                c3_capital_efficiency=0.5,
                F_ux_cap=500.0      # High crisis fee tolerance
            )

        else:  # PROTOCOL_DAO (balanced)
            return cls()  # All weights = 1.0


@dataclass
class TheoreticalMetrics:
    """Metrics calculated according to SUMMARY.md formulas."""

    # UX metrics (Section 2.3.1)
    median_fee: float = 0.0              # m_F(θ)
    p95_fee: float = 0.0                 # F_95(θ)
    fee_variance: float = 0.0            # Var_F(θ)
    fee_cv: float = 0.0                  # CV_F(θ)
    p95_relative_jump: float = 0.0       # J_ΔF(θ)

    # Safety metrics (Section 2.3.2)
    deficit_weighted_duration: float = 0.0   # DD(θ) = Σ(T-V(t))_+
    max_deficit_depth: float = 0.0           # D_max(θ) = max(T-V(t))_+
    recovery_time_shock: float = 0.0         # Recovery time after synthetic shock

    # Efficiency metrics (Section 2.3.3)
    target_size: float = 0.0                 # T (capital locked)
    avg_vault_deviation: float = 0.0         # E[|V(t)-T|]
    capital_per_throughput: float = 0.0      # T/E[Q(t)]

    # Hard constraints
    cost_recovery_ratio: float = 1.0         # CRR(θ) = R(θ)/C_L1
    ruin_probability: float = 0.0            # ρ_ruin(θ)


class TheoreticalOptimizer:
    """
    Parameter optimizer implementing SUMMARY.md theoretical framework.

    Solves the optimization problem from Section 2.4:

    min_θ w_UX * J_UX(θ) + w_safe * J_safe(θ) + w_eff * J_eff(θ)
    s.t.  1-ε_CRR ≤ CRR(θ) ≤ 1+ε_CRR
          ρ_ruin(θ) ≤ ε_ruin
    """

    def __init__(self,
                 epsilon_crr: float = 0.05,
                 epsilon_ruin: float = 0.01):
        """
        Initialize optimizer with constraint tolerances.

        Args:
            epsilon_crr: CRR constraint tolerance (±5%)
            epsilon_ruin: Maximum ruin probability (1%)
        """
        self.epsilon_crr = epsilon_crr
        self.epsilon_ruin = epsilon_ruin

        # Parameter bounds
        self.bounds = {
            'mu': (0.0, 1.0),
            'nu': (0.0, 1.0),
            'H': (6, 1440),  # 6 steps (12s) to 1440 steps (48 min)
            'lambda_B': (0.1, 0.9),
            'T': (500.0, 2000.0)  # Target balance bounds (ETH)
        }

        # Simulation parameters
        self.simulation_steps = 2160  # 1.2 hours at 2s per step

    def optimize_for_profile(self,
                           profile: StakeholderProfile,
                           population_size: int = 50,
                           generations: int = 30,
                           scenario_data: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Run optimization for specific stakeholder profile.

        Args:
            profile: Stakeholder profile defining objective weights
            population_size: GA population size
            generations: Number of GA generations
            scenario_data: Historical L1 data scenarios for constraint evaluation

        Returns:
            Optimization results with Pareto front and analysis
        """
        start_time = time.time()

        # Get objective weights for profile
        weights = TheoreticalObjectiveWeights.for_profile(profile)

        # Load scenario data if not provided
        if scenario_data is None:
            scenario_data = self._load_historical_scenarios()

        # Initialize population
        population = self._initialize_population(population_size)

        # Evolution loop with constraint handling
        best_solutions = []
        convergence_history = []

        for gen in range(generations):
            # Evaluate population
            evaluated_population = []
            for individual in population:
                metrics = self._evaluate_individual(individual, scenario_data)
                objectives = self._calculate_objectives(metrics, weights)
                constraint_violation = self._calculate_constraint_violation(metrics)

                evaluated_population.append({
                    'parameters': individual,
                    'metrics': metrics,
                    'objectives': objectives,
                    'constraint_violation': constraint_violation,
                    'feasible': constraint_violation == 0
                })

            # Selection and reproduction
            population = self._genetic_operations(evaluated_population, population_size)

            # Track best feasible solution
            feasible_solutions = [ind for ind in evaluated_population if ind['feasible']]
            if feasible_solutions:
                best = min(feasible_solutions, key=lambda x: sum(x['objectives']))
                best_solutions.append(best)

            # Convergence tracking
            convergence_history.append({
                'generation': gen,
                'best_objective': sum(best['objectives']) if feasible_solutions else float('inf'),
                'feasible_count': len(feasible_solutions),
                'avg_constraint_violation': np.mean([ind['constraint_violation']
                                                   for ind in evaluated_population])
            })

        return {
            'profile': profile,
            'best_solutions': best_solutions,
            'final_population': evaluated_population,
            'convergence_history': convergence_history,
            'optimization_time': time.time() - start_time,
            'objective_weights': weights
        }

    def _load_historical_scenarios(self) -> List[List[float]]:
        """Load historical L1 basefee scenarios for constraint evaluation."""
        try:
            from .canonical_scenarios import get_scenario_list_for_optimization
            scenarios = get_scenario_list_for_optimization()
            print(f"Loaded {len(scenarios)} historical scenarios for constraint evaluation")
            return scenarios
        except ImportError:
            warnings.warn("canonical_scenarios not available - using synthetic data")
            return self._generate_synthetic_scenarios(5)

    def _generate_synthetic_scenarios(self, n_scenarios: int) -> List[List[float]]:
        """Generate synthetic L1 basefee scenarios."""
        scenarios = []
        np.random.seed(42)  # Reproducible

        for _ in range(n_scenarios):
            # GBM with different volatility regimes
            volatility = np.random.uniform(0.2, 0.8)
            initial_fee = np.random.uniform(10e9, 50e9)  # 10-50 gwei

            scenario = [initial_fee]
            for step in range(self.simulation_steps):
                dW = np.random.normal(0, np.sqrt(1/1800))  # 2s time step
                dS = volatility * scenario[-1] * dW
                new_fee = max(scenario[-1] + dS, 1e6)  # 0.001 gwei floor
                scenario.append(new_fee)

            scenarios.append(scenario[1:])  # Remove initial value

        return scenarios

    def _initialize_population(self, size: int) -> List[Dict[str, float]]:
        """Initialize random parameter population."""
        population = []

        for _ in range(size):
            individual = {
                'mu': np.random.uniform(*self.bounds['mu']),
                'nu': np.random.uniform(*self.bounds['nu']),
                'H': int(np.random.uniform(*self.bounds['H'])),
                'lambda_B': np.random.uniform(*self.bounds['lambda_B']),
                'T': np.random.uniform(*self.bounds['T'])
            }

            # Ensure H is multiple of 6 for batch alignment
            individual['H'] = (individual['H'] // 6) * 6
            if individual['H'] < 6:
                individual['H'] = 6

            population.append(individual)

        return population

    def _evaluate_individual(self,
                           individual: Dict[str, float],
                           scenario_data: List[List[float]]) -> TheoreticalMetrics:
        """Evaluate individual using theoretical framework metrics."""

        # Create fee calculator
        params = FeeParameters(
            mu=individual['mu'],
            nu=individual['nu'],
            H=int(individual['H']),
            lambda_B=individual['lambda_B'],
            T=individual['T']
        )
        calculator = CanonicalTaikoFeeCalculator(params)

        # Run primary simulation
        primary_results = self._run_simulation(calculator, scenario_data[0] if scenario_data else None)

        # Calculate UX metrics
        fees = np.array(primary_results['estimatedFee'])
        median_fee = np.median(fees)
        p95_fee = np.percentile(fees, 95)
        fee_variance = np.var(fees)
        fee_cv = np.std(fees) / (np.mean(fees) + 1e-8)

        # Calculate relative jumps
        relative_jumps = np.abs(np.diff(fees)) / (fees[:-1] + 1e-8)
        p95_relative_jump = np.percentile(relative_jumps, 95)

        # Calculate safety metrics
        vault_balances = np.array(primary_results['vaultBalance'])
        vault_deficits = np.maximum(0, individual['T'] - vault_balances)

        deficit_weighted_duration = np.sum(vault_deficits)
        max_deficit_depth = np.max(vault_deficits)
        recovery_time_shock = self._calculate_recovery_time(vault_balances, individual['T'])

        # Calculate efficiency metrics
        avg_vault_deviation = np.mean(np.abs(vault_balances - individual['T']))
        tx_volumes = np.array(primary_results['transactionVolume'])
        avg_throughput = np.mean(tx_volumes) if len(tx_volumes) > 0 else 1.0
        capital_per_throughput = individual['T'] / avg_throughput

        # Calculate hard constraints
        cost_recovery_ratio = self._calculate_crr(calculator, scenario_data[0] if scenario_data else None)
        ruin_probability = self._calculate_ruin_probability(calculator, scenario_data)

        return TheoreticalMetrics(
            median_fee=median_fee,
            p95_fee=p95_fee,
            fee_variance=fee_variance,
            fee_cv=fee_cv,
            p95_relative_jump=p95_relative_jump,
            deficit_weighted_duration=deficit_weighted_duration,
            max_deficit_depth=max_deficit_depth,
            recovery_time_shock=recovery_time_shock,
            target_size=individual['T'],
            avg_vault_deviation=avg_vault_deviation,
            capital_per_throughput=capital_per_throughput,
            cost_recovery_ratio=cost_recovery_ratio,
            ruin_probability=ruin_probability
        )

    def _run_simulation(self,
                       calculator: CanonicalTaikoFeeCalculator,
                       l1_data: Optional[List[float]]) -> Dict[str, List[float]]:
        """Run fee mechanism simulation."""

        # Use provided data or generate synthetic
        if l1_data is None or len(l1_data) < self.simulation_steps:
            l1_data = self._generate_synthetic_l1_data()

        # Create vault in target state
        vault = calculator.create_vault(VaultInitMode.TARGET)

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

        for step in range(min(self.simulation_steps, len(l1_data))):
            l1_basefee_wei = l1_data[step]

            # Calculate fee and volume
            l1_cost = calculator.calculate_l1_cost_per_tx(l1_basefee_wei)
            estimated_fee = calculator.calculate_estimated_fee(l1_cost, vault.deficit)
            tx_volume = calculator.calculate_transaction_volume(estimated_fee)

            # Vault operations
            fees_collected = estimated_fee * tx_volume
            vault.collect_fees(fees_collected)

            l1_costs_paid = 0.0
            if step % calculator.params.batch_interval_steps == 0:
                l1_costs_paid = calculator.calculate_l1_batch_cost(l1_basefee_wei)
                vault.pay_l1_costs(l1_costs_paid)

            # Record results
            results['timeStep'].append(step)
            results['l1Basefee'].append(l1_basefee_wei / 1e9)
            results['estimatedFee'].append(estimated_fee)
            results['transactionVolume'].append(tx_volume)
            results['vaultBalance'].append(vault.balance)
            results['vaultDeficit'].append(vault.deficit)
            results['feesCollected'].append(fees_collected)
            results['l1CostsPaid'].append(l1_costs_paid)

        return results

    def _generate_synthetic_l1_data(self) -> List[float]:
        """Generate synthetic L1 basefee data."""
        np.random.seed(42)

        basefees = []
        current_fee = 20e9  # 20 gwei

        for _ in range(self.simulation_steps):
            # Add volatility with mean reversion
            drift = -0.1 * (current_fee - 15e9) / 15e9  # Mean revert to 15 gwei
            volatility = 0.3
            dW = np.random.normal(0, np.sqrt(1/1800))  # 2s timestep

            dS = drift * current_fee * (1/1800) + volatility * current_fee * dW
            current_fee = max(current_fee + dS, 1e6)  # Floor at 0.001 gwei
            basefees.append(current_fee)

        return basefees

    def _calculate_recovery_time(self, vault_balances: np.ndarray, target: float) -> float:
        """Calculate recovery time after stress events."""
        recovery_times = []

        # Find stress periods (below 80% of target)
        stress_threshold = 0.8 * target
        recovery_threshold = 0.95 * target

        in_stress = False
        stress_start = 0

        for i, balance in enumerate(vault_balances):
            if balance < stress_threshold and not in_stress:
                in_stress = True
                stress_start = i
            elif balance >= recovery_threshold and in_stress:
                in_stress = False
                recovery_time = i - stress_start
                recovery_times.append(recovery_time)

        return np.mean(recovery_times) if recovery_times else 0.0

    def _calculate_crr(self,
                      calculator: CanonicalTaikoFeeCalculator,
                      scenario_data: Optional[List[float]]) -> float:
        """Calculate Cost Recovery Ratio for constraint evaluation."""
        if scenario_data is None:
            return 1.0

        vault = calculator.create_vault(VaultInitMode.TARGET)
        total_revenue = 0.0
        total_costs = 0.0

        # Limit scenario for computational efficiency
        scenario_length = min(len(scenario_data), 1080)  # 36 minutes

        for step in range(scenario_length):
            l1_basefee_wei = scenario_data[step]

            # Calculate revenue
            l1_cost = calculator.calculate_l1_cost_per_tx(l1_basefee_wei)
            estimated_fee = calculator.calculate_estimated_fee(l1_cost, vault.deficit)
            tx_volume = calculator.calculate_transaction_volume(estimated_fee)

            revenue = estimated_fee * tx_volume
            total_revenue += revenue
            vault.collect_fees(revenue)

            # Calculate costs (every batch interval)
            if step % calculator.params.batch_interval_steps == 0:
                l1_cost_batch = calculator.calculate_l1_batch_cost(l1_basefee_wei)
                total_costs += l1_cost_batch
                vault.pay_l1_costs(l1_cost_batch)

        return total_revenue / total_costs if total_costs > 0 else 1.0

    def _calculate_ruin_probability(self,
                                   calculator: CanonicalTaikoFeeCalculator,
                                   scenario_data: List[List[float]]) -> float:
        """Calculate ruin probability across scenarios."""
        if not scenario_data:
            return 0.0

        ruin_count = 0
        critical_threshold = 0.1 * calculator.params.T  # 10% of target

        for scenario in scenario_data[:5]:  # Limit to 5 scenarios for speed
            vault = calculator.create_vault(VaultInitMode.TARGET)

            # Run scenario and check for ruin
            for step in range(min(len(scenario), 540)):  # 18 minutes
                l1_basefee_wei = scenario[step]

                l1_cost = calculator.calculate_l1_cost_per_tx(l1_basefee_wei)
                estimated_fee = calculator.calculate_estimated_fee(l1_cost, vault.deficit)
                tx_volume = calculator.calculate_transaction_volume(estimated_fee)

                revenue = estimated_fee * tx_volume
                vault.collect_fees(revenue)

                if step % calculator.params.batch_interval_steps == 0:
                    l1_cost_batch = calculator.calculate_l1_batch_cost(l1_basefee_wei)
                    vault.pay_l1_costs(l1_cost_batch)

                # Check for ruin
                if vault.balance < critical_threshold:
                    ruin_count += 1
                    break

        return ruin_count / min(len(scenario_data), 5)

    def _calculate_objectives(self,
                            metrics: TheoreticalMetrics,
                            weights: TheoreticalObjectiveWeights) -> List[float]:
        """
        Calculate objective functions according to SUMMARY.md Section 2.3.

        J_UX(θ) = a1*CV_F + a2*J_ΔF + a3*max(0, F95-F_cap)
        J_safe(θ) = b1*DD + b2*D_max + b3*RecoveryTime
        J_eff(θ) = c1*T + c2*|V-T| + c3*CapEff
        """

        # UX objective (Section 2.3.1)
        high_fee_penalty = max(0, metrics.p95_fee * 1e9 - weights.F_ux_cap)  # Convert to gwei
        j_ux = (weights.a1_stability * metrics.fee_cv +
                weights.a2_jumpiness * metrics.p95_relative_jump +
                weights.a3_high_fees * high_fee_penalty / 100.0)  # Normalize penalty

        # Safety objective (Section 2.3.2)
        j_safe = (weights.b1_deficit_duration * metrics.deficit_weighted_duration / 1000.0 +  # Normalize by target
                  weights.b2_max_deficit * metrics.max_deficit_depth / 1000.0 +
                  weights.b3_recovery_time * metrics.recovery_time_shock / 100.0)  # Normalize by steps

        # Efficiency objective (Section 2.3.3)
        j_eff = (weights.c1_target_size * metrics.target_size / 1000.0 +  # Normalize by 1000 ETH
                 weights.c2_vault_deviation * metrics.avg_vault_deviation / 1000.0 +
                 weights.c3_capital_efficiency * metrics.capital_per_throughput / 1e6)  # Normalize by 1M

        return [j_ux, j_safe, j_eff]

    def _calculate_constraint_violation(self, metrics: TheoreticalMetrics) -> float:
        """Calculate constraint violation penalty."""
        violation = 0.0

        # CRR constraint: 1-ε ≤ CRR ≤ 1+ε
        crr_min = 1.0 - self.epsilon_crr
        crr_max = 1.0 + self.epsilon_crr

        if metrics.cost_recovery_ratio < crr_min:
            violation += (crr_min - metrics.cost_recovery_ratio) * 10.0
        elif metrics.cost_recovery_ratio > crr_max:
            violation += (metrics.cost_recovery_ratio - crr_max) * 10.0

        # Ruin probability constraint
        if metrics.ruin_probability > self.epsilon_ruin:
            violation += (metrics.ruin_probability - self.epsilon_ruin) * 50.0

        return violation

    def _genetic_operations(self,
                           population: List[Dict],
                           target_size: int) -> List[Dict[str, float]]:
        """Simple genetic algorithm operations."""
        # Select best individuals (constraint handling)
        feasible = [ind for ind in population if ind['feasible']]
        infeasible = [ind for ind in population if not ind['feasible']]

        # Sort feasible by objective value, infeasible by constraint violation
        if feasible:
            feasible.sort(key=lambda x: sum(x['objectives']))
        if infeasible:
            infeasible.sort(key=lambda x: x['constraint_violation'])

        # Select survivors
        survivors = feasible[:target_size//2] + infeasible[:target_size//4]

        # Generate offspring
        offspring = []
        while len(offspring) + len(survivors) < target_size:
            if len(survivors) >= 2:
                parent1 = np.random.choice(survivors)
                parent2 = np.random.choice(survivors)
                child = self._crossover(parent1['parameters'], parent2['parameters'])
                child = self._mutate(child)
                offspring.append(child)
            else:
                # Random individual if not enough survivors
                offspring.append(self._initialize_population(1)[0])

        return [ind['parameters'] for ind in survivors] + offspring

    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Simulated binary crossover."""
        child = {}
        eta_c = 20.0

        for param in ['mu', 'nu', 'lambda_B', 'T']:
            if np.random.random() < 0.5:
                u = np.random.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))

                child[param] = 0.5 * ((1 + beta) * parent1[param] + (1 - beta) * parent2[param])
                child[param] = np.clip(child[param], *self.bounds[param])
            else:
                child[param] = parent1[param]

        # Discrete crossover for H
        child['H'] = parent1['H'] if np.random.random() < 0.5 else parent2['H']
        child['H'] = int(np.clip(child['H'], *self.bounds['H']))
        child['H'] = (child['H'] // 6) * 6  # Ensure multiple of 6
        if child['H'] < 6:
            child['H'] = 6

        return child

    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Polynomial mutation."""
        mutated = individual.copy()
        eta_m = 50.0
        mutation_rate = 0.1

        for param in ['mu', 'nu', 'lambda_B', 'T']:
            if np.random.random() < mutation_rate:
                u = np.random.random()
                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))

                range_val = self.bounds[param][1] - self.bounds[param][0]
                mutated[param] += delta * range_val
                mutated[param] = np.clip(mutated[param], *self.bounds[param])

        # Discrete mutation for H
        if np.random.random() < mutation_rate:
            h_values = list(range(6, 1440, 6))
            mutated['H'] = np.random.choice(h_values)

        return mutated


def run_stakeholder_analysis(population_size: int = 50,
                           generations: int = 30) -> Dict[str, Any]:
    """
    Run parameter optimization for all stakeholder profiles.

    This function implements a comprehensive stakeholder analysis following
    the theoretical framework from SUMMARY.md.

    Returns:
        Dictionary containing results for each stakeholder profile
    """
    print("🎯 Running Theoretical Framework Parameter Optimization")
    print("=" * 60)

    optimizer = TheoreticalOptimizer()
    results = {}

    for profile in StakeholderProfile:
        print(f"\n📊 Optimizing for {profile.value.replace('_', ' ').title()} Profile...")

        start_time = time.time()
        profile_results = optimizer.optimize_for_profile(
            profile=profile,
            population_size=population_size,
            generations=generations
        )

        runtime = time.time() - start_time
        best_solutions = profile_results['best_solutions']

        print(f"   ⏱️  Runtime: {runtime:.1f}s")
        print(f"   ✅ Feasible solutions found: {len(best_solutions)}")

        if best_solutions:
            best = best_solutions[-1]  # Final best solution
            params = best['parameters']
            print(f"   🎯 Best parameters: μ={params['mu']:.3f}, ν={params['nu']:.3f}, H={params['H']}")
            print(f"   📈 Objectives: {[f'{obj:.3f}' for obj in best['objectives']]}")

        results[profile] = profile_results

    return results


# Convenience function for quick profile comparison
def compare_stakeholder_optima() -> Dict[str, Dict[str, float]]:
    """Quick comparison of optimal parameters across stakeholder profiles."""

    results = run_stakeholder_analysis(population_size=30, generations=20)
    comparison = {}

    for profile, profile_results in results.items():
        if profile_results['best_solutions']:
            best = profile_results['best_solutions'][-1]
            params = best['parameters']
            metrics = best['metrics']

            comparison[profile.value] = {
                'mu': params['mu'],
                'nu': params['nu'],
                'H': params['H'],
                'lambda_B': params['lambda_B'],
                'T': params['T'],
                'median_fee_gwei': metrics.median_fee * 1e9,
                'fee_cv': metrics.fee_cv,
                'crr': metrics.cost_recovery_ratio,
                'ruin_prob': metrics.ruin_probability,
                'objective_sum': sum(best['objectives'])
            }

    return comparison