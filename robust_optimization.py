"""
Robust Parameter Optimization with Realistic Constraints

This script fixes the issues identified in the diagnostic and provides
a working implementation of the theoretical framework from SUMMARY.md.
"""

import sys
sys.path.append('src')

import numpy as np
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from src.core.canonical_fee_mechanism import CanonicalTaikoFeeCalculator, FeeParameters, VaultInitMode
from src.core.theoretical_optimization import StakeholderProfile, TheoreticalObjectiveWeights


@dataclass
class ParameterSet:
    """Parameter set for evaluation."""
    mu: float
    nu: float
    H: int
    lambda_B: float = 0.365
    T: float = 1000.0

    def to_dict(self):
        return {
            'mu': self.mu,
            'nu': self.nu,
            'H': self.H,
            'lambda_B': self.lambda_B,
            'T': self.T
        }


class RobustOptimizer:
    """
    Robust optimizer that works with realistic data and constraints.
    """

    def __init__(self):
        # Realistic parameter bounds based on research
        self.bounds = {
            'mu': (0.0, 0.6),
            'nu': (0.0, 1.0),
            'H': (60, 600),  # 2-20 minutes at 2s steps
            'lambda_B': (0.3, 0.4),
            'T': (500, 2000)
        }

        # Relaxed but meaningful constraints
        self.constraints = {
            'min_crr': 0.80,  # 80% cost recovery minimum
            'max_crr': 1.30,  # 130% cost recovery maximum
            'max_ruin_prob': 0.15,  # 15% ruin probability max
            'max_median_fee_gwei': 200.0,  # 200 gwei max median fee
        }

    def evaluate_parameter_set(self, params: ParameterSet) -> Dict[str, Any]:
        """Evaluate a single parameter set."""

        try:
            # Create fee calculator with realistic parameters
            fee_params = FeeParameters(
                mu=params.mu,
                nu=params.nu,
                H=params.H,
                lambda_B=params.lambda_B,
                T=params.T,
                alpha_data=20000,  # Fixed at realistic value
                Q_bar=690000,      # Fixed at realistic value
                F_min=1e-9,        # Very low minimum (0.001 gwei)
                F_max=500e-9       # High but realistic maximum (500 gwei)
            )
            calculator = CanonicalTaikoFeeCalculator(fee_params)

            # Run realistic simulation
            metrics = self._run_evaluation_simulation(calculator)

            # Calculate theoretical framework objectives
            objectives = self._calculate_theoretical_objectives(metrics, params)

            # Check constraints
            constraint_violation = self._check_constraints(metrics)

            return {
                'params': params,
                'metrics': metrics,
                'objectives': objectives,
                'constraint_violation': constraint_violation,
                'feasible': constraint_violation == 0
            }

        except Exception as e:
            return {
                'params': params,
                'metrics': None,
                'objectives': [float('inf')] * 3,
                'constraint_violation': float('inf'),
                'feasible': False,
                'error': str(e)
            }

    def _run_evaluation_simulation(self, calculator: CanonicalTaikoFeeCalculator) -> Dict[str, Any]:
        """Run realistic fee mechanism simulation."""

        # Create vault starting at target
        vault = calculator.create_vault(VaultInitMode.TARGET)

        # Generate realistic L1 data with volatility
        l1_data = self._generate_realistic_l1_data()

        # Simulation results
        results = {
            'fees': [],
            'vault_balances': [],
            'vault_deficits': [],
            'tx_volumes': [],
            'l2_revenues': [],
            'l1_costs': []
        }

        total_l2_revenue = 0.0
        total_l1_cost = 0.0

        # Run simulation for 30 minutes (900 steps at 2s each)
        for step in range(900):
            l1_basefee_wei = l1_data[step]

            # Calculate fee using canonical mechanism
            l1_cost_per_tx = calculator.calculate_l1_cost_per_tx(l1_basefee_wei)
            estimated_fee = calculator.calculate_estimated_fee(l1_cost_per_tx, vault.deficit)

            # Limit fee to reasonable range
            estimated_fee = max(1e-12, min(estimated_fee, 500e-9))  # 0.001 - 500 gwei

            # Calculate transaction volume (with realistic demand curve)
            fee_gwei = estimated_fee * 1e9
            if fee_gwei < 1.0:
                tx_volume = 25000  # High volume for low fees
            elif fee_gwei < 10.0:
                tx_volume = 20000 - (fee_gwei * 1000)  # Linear decrease
            elif fee_gwei < 50.0:
                tx_volume = max(5000, 15000 - (fee_gwei * 200))
            else:
                tx_volume = max(1000, 10000 - (fee_gwei * 100))

            # Vault operations
            l2_revenue = estimated_fee * tx_volume * calculator.params.gas_per_tx
            vault.collect_fees(l2_revenue)
            total_l2_revenue += l2_revenue

            # L1 costs (every 6 steps = 12s batch interval)
            l1_cost_step = 0.0
            if step % 6 == 0:
                l1_cost_step = calculator.calculate_l1_batch_cost(l1_basefee_wei)
                vault.pay_l1_costs(l1_cost_step)
                total_l1_cost += l1_cost_step

            # Record results
            results['fees'].append(estimated_fee)
            results['vault_balances'].append(vault.balance)
            results['vault_deficits'].append(vault.deficit)
            results['tx_volumes'].append(tx_volume)
            results['l2_revenues'].append(l2_revenue)
            results['l1_costs'].append(l1_cost_step)

        # Calculate summary metrics
        fees = np.array(results['fees'])
        balances = np.array(results['vault_balances'])
        deficits = np.array(results['vault_deficits'])

        return {
            'median_fee_gwei': np.median(fees) * 1e9,
            'p95_fee_gwei': np.percentile(fees, 95) * 1e9,
            'fee_cv': np.std(fees) / (np.mean(fees) + 1e-12),
            'relative_jumps': np.abs(np.diff(fees)) / (fees[:-1] + 1e-12),
            'vault_balances': balances,
            'vault_deficits': deficits,
            'cost_recovery_ratio': total_l2_revenue / max(total_l1_cost, 1e-6),
            'min_balance': np.min(balances),
            'max_deficit': np.max(deficits),
            'total_l2_revenue': total_l2_revenue,
            'total_l1_cost': total_l1_cost
        }

    def _generate_realistic_l1_data(self) -> List[float]:
        """Generate realistic L1 basefee data with proper volatility."""
        np.random.seed(42)

        # Start at reasonable L1 basefee (15 gwei)
        basefees = []
        current_fee = 15e9

        for step in range(900):
            # Realistic volatility with occasional spikes
            if np.random.random() < 0.02:  # 2% chance of spike
                spike_factor = np.random.uniform(2.0, 5.0)
                current_fee *= spike_factor
            else:
                # Mean-reverting random walk
                drift = -0.1 * (current_fee - 15e9) / 15e9  # Revert to 15 gwei
                volatility = 0.2
                dt = 1.0 / 1800  # 2-second steps
                dW = np.random.normal(0, np.sqrt(dt))

                dS = drift * current_fee * dt + volatility * current_fee * dW
                current_fee = max(current_fee + dS, 1e9)  # Floor at 1 gwei

            # Decay spikes gradually
            if current_fee > 30e9:
                current_fee *= 0.95

            basefees.append(current_fee)

        return basefees

    def _calculate_theoretical_objectives(self, metrics: Dict[str, Any], params: ParameterSet) -> List[float]:
        """Calculate SUMMARY.md theoretical framework objectives."""

        # UX Objective (Section 2.3.1)
        # J_UX = a1*CV_F + a2*J_ΔF + a3*max(0, F95-F_cap)
        fee_cv = metrics['fee_cv']
        p95_relative_jump = np.percentile(metrics['relative_jumps'], 95)
        high_fee_penalty = max(0, metrics['p95_fee_gwei'] - 50.0) / 100.0  # Penalty for >50 gwei

        j_ux = fee_cv + p95_relative_jump + high_fee_penalty

        # Safety Objective (Section 2.3.2)
        # J_safe = b1*DD + b2*D_max + b3*RecoveryTime
        deficit_weighted_duration = np.sum(metrics['vault_deficits'])
        max_deficit = metrics['max_deficit']
        recovery_time = self._calculate_recovery_time(metrics['vault_balances'], params.T)

        j_safe = (deficit_weighted_duration / 1000.0 +  # Normalize
                  max_deficit / 1000.0 +
                  recovery_time / 100.0)

        # Efficiency Objective (Section 2.3.3)
        # J_eff = c1*T + c2*|V-T| + c3*CapEff
        target_penalty = params.T / 1000.0  # Capital cost
        avg_deviation = np.mean(np.abs(metrics['vault_balances'] - params.T)) / 1000.0
        capital_efficiency = params.T / max(metrics['total_l2_revenue'], 1e-6)

        j_eff = target_penalty + avg_deviation + capital_efficiency

        return [j_ux, j_safe, j_eff]

    def _calculate_recovery_time(self, balances: np.ndarray, target: float) -> float:
        """Calculate average recovery time from stress events."""
        stress_threshold = 0.8 * target
        recovery_threshold = 0.95 * target

        recovery_times = []
        in_stress = False
        stress_start = 0

        for i, balance in enumerate(balances):
            if balance < stress_threshold and not in_stress:
                in_stress = True
                stress_start = i
            elif balance >= recovery_threshold and in_stress:
                in_stress = False
                recovery_times.append(i - stress_start)

        return np.mean(recovery_times) if recovery_times else 0.0

    def _check_constraints(self, metrics: Dict[str, Any]) -> float:
        """Check hard constraints and return violation penalty."""
        violation = 0.0

        # CRR constraint
        crr = metrics['cost_recovery_ratio']
        if crr < self.constraints['min_crr']:
            violation += (self.constraints['min_crr'] - crr) * 10.0
        elif crr > self.constraints['max_crr']:
            violation += (crr - self.constraints['max_crr']) * 10.0

        # Ruin probability (simplified: check if balance goes below 10% of target)
        min_balance_ratio = metrics['min_balance'] / 1000.0  # Assuming T=1000 baseline
        if min_balance_ratio < 0.1:
            violation += (0.1 - min_balance_ratio) * 20.0

        # Median fee constraint
        if metrics['median_fee_gwei'] > self.constraints['max_median_fee_gwei']:
            violation += (metrics['median_fee_gwei'] - self.constraints['max_median_fee_gwei']) / 10.0

        return violation

    def optimize_for_stakeholder_profiles(self) -> Dict[str, Any]:
        """Run optimization for all stakeholder profiles using grid search."""

        print("🎯 ROBUST STAKEHOLDER PROFILE OPTIMIZATION")
        print("=" * 60)
        print("Using theoretical framework from SUMMARY.md with realistic constraints")
        print()

        # Define stakeholder profiles with weights
        profiles = {
            'end_user': TheoreticalObjectiveWeights(
                a1_stability=3.0, a2_jumpiness=2.0, a3_high_fees=5.0,
                b1_deficit_duration=0.5, b2_max_deficit=0.5, b3_recovery_time=0.3,
                c1_target_size=0.1, c2_vault_deviation=0.1, c3_capital_efficiency=0.1,
                F_ux_cap=30.0
            ),
            'protocol_dao': TheoreticalObjectiveWeights(),  # Balanced
            'vault_operator': TheoreticalObjectiveWeights(
                a1_stability=0.5, a2_jumpiness=0.3, a3_high_fees=0.3,
                b1_deficit_duration=1.0, b2_max_deficit=1.5, b3_recovery_time=1.0,
                c1_target_size=3.0, c2_vault_deviation=2.0, c3_capital_efficiency=3.0,
                F_ux_cap=150.0
            ),
            'sequencer': TheoreticalObjectiveWeights(
                a1_stability=2.0, a2_jumpiness=1.0, a3_high_fees=0.5,
                b1_deficit_duration=1.5, b2_max_deficit=1.5, b3_recovery_time=1.0,
                c1_target_size=1.0, c2_vault_deviation=1.5, c3_capital_efficiency=2.0,
                F_ux_cap=100.0
            ),
            'crisis_manager': TheoreticalObjectiveWeights(
                a1_stability=1.0, a2_jumpiness=0.5, a3_high_fees=0.2,
                b1_deficit_duration=3.0, b2_max_deficit=3.0, b3_recovery_time=3.0,
                c1_target_size=0.5, c2_vault_deviation=0.5, c3_capital_efficiency=0.5,
                F_ux_cap=300.0
            )
        }

        results = {}

        for profile_name, weights in profiles.items():
            print(f"📊 Optimizing for {profile_name.replace('_', ' ').title()}...")

            start_time = time.time()
            best_result = self._grid_search_optimization(weights)
            runtime = time.time() - start_time

            print(f"   ⏱️  Runtime: {runtime:.1f}s")

            if best_result['feasible']:
                params = best_result['params']
                metrics = best_result['metrics']
                print(f"   ✅ Feasible solution found!")
                print(f"   🎯 Parameters: μ={params.mu:.3f}, ν={params.nu:.3f}, H={params.H}")
                print(f"   📈 Fee: {metrics['median_fee_gwei']:.2f} gwei (CV: {metrics['fee_cv']:.3f})")
                print(f"   💰 CRR: {metrics['cost_recovery_ratio']:.3f}")
                print(f"   🏦 Min Balance: {metrics['min_balance']:.0f} ETH")
                print(f"   🎪 Objectives: {[f'{obj:.3f}' for obj in best_result['objectives']]}")
            else:
                print(f"   ❌ No feasible solution found (best violation: {best_result['constraint_violation']:.3f})")

            results[profile_name] = best_result
            print()

        return results

    def _grid_search_optimization(self, weights: TheoreticalObjectiveWeights) -> Dict[str, Any]:
        """Perform grid search optimization for given weights."""

        # Coarse grid for efficiency
        mu_values = np.linspace(0.0, 0.5, 6)
        nu_values = np.linspace(0.0, 1.0, 11)
        H_values = [60, 120, 240, 360, 480, 600]

        best_result = None
        best_objective = float('inf')

        for mu in mu_values:
            for nu in nu_values:
                for H in H_values:
                    params = ParameterSet(mu=mu, nu=nu, H=H)
                    result = self.evaluate_parameter_set(params)

                    if result.get('error'):
                        continue

                    # Calculate weighted objective
                    objectives = result['objectives']
                    weighted_objective = (weights.a1_stability * objectives[0] +
                                        weights.b1_deficit_duration * objectives[1] +
                                        weights.c1_target_size * objectives[2])

                    # Prefer feasible solutions, then best objective
                    if result['feasible']:
                        if best_result is None or not best_result['feasible'] or weighted_objective < best_objective:
                            best_result = result
                            best_objective = weighted_objective
                    elif best_result is None or (not best_result['feasible'] and
                                               result['constraint_violation'] < best_result['constraint_violation']):
                        best_result = result

        return best_result or {'feasible': False, 'constraint_violation': float('inf')}


if __name__ == "__main__":
    optimizer = RobustOptimizer()
    results = optimizer.optimize_for_stakeholder_profiles()

    print("🎯 OPTIMIZATION SUMMARY")
    print("=" * 40)

    for profile_name, result in results.items():
        if result['feasible']:
            params = result['params']
            print(f"{profile_name.replace('_', ' ').title()}: "
                  f"μ={params.mu:.3f}, ν={params.nu:.3f}, H={params.H} "
                  f"(Fee: {result['metrics']['median_fee_gwei']:.1f} gwei)")
        else:
            print(f"{profile_name.replace('_', ' ').title()}: No feasible solution")