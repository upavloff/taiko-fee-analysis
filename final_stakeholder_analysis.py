"""
Final Stakeholder Analysis Using SUMMARY.md Theoretical Framework

This implementation provides a working stakeholder analysis by:
1. Using the existing canonical fee mechanism correctly
2. Implementing SUMMARY.md theoretical objectives
3. Evaluating known good parameter sets for different stakeholder profiles
4. Providing meaningful comparisons and recommendations

As a senior protocol researcher, this delivers actionable insights.
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from dataclasses import dataclass

from src.core.canonical_fee_mechanism import CanonicalTaikoFeeCalculator, FeeParameters, VaultInitMode


@dataclass
class StakeholderObjectives:
    """Stakeholder-specific objective weights following SUMMARY.md Section 2.3"""

    # UX weights (Section 2.3.1)
    ux_stability_weight: float      # a1 - fee stability (CV penalty)
    ux_jumpiness_weight: float      # a2 - fee jumpiness penalty
    ux_high_fee_weight: float       # a3 - high fee penalty

    # Safety weights (Section 2.3.2)
    safety_duration_weight: float   # b1 - deficit duration penalty
    safety_depth_weight: float      # b2 - max deficit penalty
    safety_recovery_weight: float   # b3 - recovery time penalty

    # Efficiency weights (Section 2.3.3)
    eff_capital_weight: float       # c1 - target size penalty
    eff_deviation_weight: float     # c2 - vault deviation penalty
    eff_throughput_weight: float    # c3 - capital efficiency penalty

    # UX parameters
    fee_tolerance_gwei: float       # F_UX-cap from SUMMARY.md


class TheoreticalFrameworkAnalyzer:
    """
    Implements the theoretical framework from SUMMARY.md Section 2 for
    stakeholder-specific parameter analysis.
    """

    def __init__(self):
        # Define stakeholder profiles based on real-world priorities
        self.stakeholder_profiles = {
            'end_user': StakeholderObjectives(
                ux_stability_weight=3.0,    # Users care most about predictability
                ux_jumpiness_weight=2.0,
                ux_high_fee_weight=5.0,     # Strong aversion to high fees
                safety_duration_weight=0.5, # Less concern about protocol details
                safety_depth_weight=0.5,
                safety_recovery_weight=0.3,
                eff_capital_weight=0.1,     # Don't care about capital efficiency
                eff_deviation_weight=0.1,
                eff_throughput_weight=0.1,
                fee_tolerance_gwei=20.0     # Low tolerance for high fees
            ),

            'protocol_dao': StakeholderObjectives(
                ux_stability_weight=1.0,    # Balanced approach
                ux_jumpiness_weight=1.0,
                ux_high_fee_weight=1.0,
                safety_duration_weight=1.0,
                safety_depth_weight=1.0,
                safety_recovery_weight=1.0,
                eff_capital_weight=1.0,
                eff_deviation_weight=1.0,
                eff_throughput_weight=1.0,
                fee_tolerance_gwei=100.0    # Moderate tolerance
            ),

            'vault_operator': StakeholderObjectives(
                ux_stability_weight=0.5,    # Revenue stability more important than UX
                ux_jumpiness_weight=0.3,
                ux_high_fee_weight=0.3,     # Higher fees = more revenue
                safety_duration_weight=1.0,
                safety_depth_weight=1.5,    # Care about vault safety
                safety_recovery_weight=1.0,
                eff_capital_weight=3.0,     # Minimize capital requirements
                eff_deviation_weight=2.0,
                eff_throughput_weight=3.0,  # Maximize capital efficiency
                fee_tolerance_gwei=200.0    # Accept higher fees for efficiency
            ),

            'sequencer': StakeholderObjectives(
                ux_stability_weight=2.0,    # Want predictable revenue stream
                ux_jumpiness_weight=1.0,
                ux_high_fee_weight=0.5,     # Higher fees = more priority fees
                safety_duration_weight=1.5, # Need protocol stability
                safety_depth_weight=1.5,
                safety_recovery_weight=1.0,
                eff_capital_weight=1.0,
                eff_deviation_weight=1.5,
                eff_throughput_weight=2.0,  # Revenue efficiency important
                fee_tolerance_gwei=150.0
            ),

            'crisis_manager': StakeholderObjectives(
                ux_stability_weight=1.0,
                ux_jumpiness_weight=0.5,    # Accept volatility for robustness
                ux_high_fee_weight=0.2,     # Accept high fees in crisis
                safety_duration_weight=3.0, # Maximum safety focus
                safety_depth_weight=3.0,
                safety_recovery_weight=3.0,
                eff_capital_weight=0.5,     # Accept capital costs for safety
                eff_deviation_weight=0.5,
                eff_throughput_weight=0.5,
                fee_tolerance_gwei=500.0    # Very high tolerance in crisis
            )
        }

        # Research-validated parameter sets from existing analysis
        self.parameter_candidates = [
            {'name': 'optimal_balanced', 'mu': 0.0, 'nu': 0.27, 'H': 492},
            {'name': 'conservative', 'mu': 0.0, 'nu': 0.48, 'H': 492},
            {'name': 'crisis_resilient', 'mu': 0.0, 'nu': 0.88, 'H': 120},
            {'name': 'user_friendly', 'mu': 0.1, 'nu': 0.1, 'H': 720},
            {'name': 'rapid_response', 'mu': 0.2, 'nu': 0.6, 'H': 180},
            {'name': 'capital_efficient', 'mu': 0.0, 'nu': 0.35, 'H': 300},
            {'name': 'stability_focused', 'mu': 0.05, 'nu': 0.25, 'H': 600},
            {'name': 'da_responsive', 'mu': 0.4, 'nu': 0.4, 'H': 240},
        ]

    def analyze_parameter_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a parameter set using canonical fee mechanism and
        calculate theoretical framework metrics.
        """

        try:
            # Create fee calculator
            fee_params = FeeParameters(
                mu=params['mu'],
                nu=params['nu'],
                H=params['H'],
                lambda_B=0.365,
                T=1000.0,
                alpha_data=20000,
                Q_bar=690000
            )
            calculator = CanonicalTaikoFeeCalculator(fee_params)

            # Run simulation
            metrics = self._run_canonical_simulation(calculator)

            # Calculate theoretical objectives
            objectives = self._calculate_theoretical_metrics(metrics, fee_params)

            return {
                'params': params,
                'simulation_metrics': metrics,
                'theoretical_objectives': objectives,
                'success': True
            }

        except Exception as e:
            return {
                'params': params,
                'error': str(e),
                'success': False
            }

    def _run_canonical_simulation(self, calculator: CanonicalTaikoFeeCalculator) -> Dict[str, Any]:
        """Run canonical fee mechanism simulation with realistic data."""

        # Create vault starting at target
        vault = calculator.create_vault(VaultInitMode.TARGET, deficit_ratio=0.1)

        # Generate realistic L1 basefee scenario
        l1_data = self._generate_l1_scenario()

        # Simulation results
        results = {
            'fees': [],
            'vault_balances': [],
            'vault_deficits': [],
            'l2_revenues': [],
            'l1_costs': []
        }

        total_l2_revenue = 0.0
        total_l1_cost = 0.0

        # Run for 1 hour (1800 steps at 2s each)
        for step in range(1800):
            l1_basefee_wei = l1_data[step % len(l1_data)]

            # Calculate estimated fee
            l1_cost_per_tx = calculator.calculate_l1_cost_per_tx(l1_basefee_wei)
            estimated_fee = calculator.calculate_estimated_fee(l1_cost_per_tx, vault.deficit)

            # Realistic transaction volume based on fee
            tx_volume = self._calculate_demand(estimated_fee)

            # Revenue collection
            l2_revenue = estimated_fee * tx_volume * calculator.params.gas_per_tx
            vault.collect_fees(l2_revenue)
            total_l2_revenue += l2_revenue

            # L1 cost payment (every 6 steps)
            l1_cost = 0.0
            if step % 6 == 0:
                l1_cost = calculator.calculate_l1_batch_cost(l1_basefee_wei)
                vault.pay_l1_costs(l1_cost)
                total_l1_cost += l1_cost

            # Record results
            results['fees'].append(estimated_fee)
            results['vault_balances'].append(vault.balance)
            results['vault_deficits'].append(vault.deficit)
            results['l2_revenues'].append(l2_revenue)
            results['l1_costs'].append(l1_cost)

        # Calculate summary statistics
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
            'avg_balance_deviation': np.mean(np.abs(balances - 1000.0)),
            'recovery_episodes': self._count_recovery_episodes(balances)
        }

    def _generate_l1_scenario(self) -> List[float]:
        """Generate realistic L1 basefee scenario with volatility."""
        np.random.seed(42)  # Reproducible

        # Base scenario: trending from 15 to 25 gwei with volatility
        scenario = []
        base_fee = 15e9

        for step in range(1800):
            # Trend component
            trend = (step / 1800) * 10e9  # Increase by 10 gwei over period

            # Volatility component
            volatility = 0.3 * base_fee * np.random.normal(0, 0.1)

            # Occasional spikes
            if np.random.random() < 0.01:  # 1% chance
                spike = base_fee * np.random.uniform(2, 4)
                volatility += spike

            current_fee = max(base_fee + trend + volatility, 1e9)  # Floor at 1 gwei
            scenario.append(current_fee)

        return scenario

    def _calculate_demand(self, fee_per_gas: float) -> int:
        """Calculate transaction demand based on fee level."""
        fee_gwei = fee_per_gas * 1e9

        if fee_gwei < 1.0:
            return 30000
        elif fee_gwei < 5.0:
            return max(20000, 30000 - int(fee_gwei * 2000))
        elif fee_gwei < 20.0:
            return max(10000, 20000 - int(fee_gwei * 500))
        elif fee_gwei < 50.0:
            return max(5000, 15000 - int(fee_gwei * 200))
        else:
            return max(2000, 10000 - int(fee_gwei * 100))

    def _count_recovery_episodes(self, balances: np.ndarray) -> int:
        """Count how many times vault recovers from stress."""
        stress_threshold = 800.0  # 80% of 1000 ETH target
        recovery_threshold = 950.0  # 95% of target

        episodes = 0
        in_stress = False

        for balance in balances:
            if balance < stress_threshold and not in_stress:
                in_stress = True
            elif balance >= recovery_threshold and in_stress:
                in_stress = False
                episodes += 1

        return episodes

    def _calculate_theoretical_metrics(self, metrics: Dict[str, Any], params: FeeParameters) -> Dict[str, float]:
        """Calculate theoretical framework metrics from SUMMARY.md Section 2.3"""

        # UX metrics (Section 2.3.1)
        median_fee = metrics['median_fee_gwei']
        p95_fee = metrics['p95_fee_gwei']
        fee_cv = metrics['fee_cv']
        p95_jump = np.percentile(metrics['relative_jumps'], 95) if len(metrics['relative_jumps']) > 0 else 0

        # Safety metrics (Section 2.3.2)
        deficit_weighted_duration = np.sum(metrics['vault_deficits'])
        max_deficit_depth = metrics['max_deficit']
        recovery_time = 1.0 / max(metrics['recovery_episodes'], 1)  # Inverse of recovery frequency

        # Efficiency metrics (Section 2.3.3)
        target_size = params.T
        avg_vault_deviation = metrics['avg_balance_deviation']
        capital_per_throughput = target_size / max(metrics['cost_recovery_ratio'], 0.1)

        return {
            # UX objectives
            'median_fee_gwei': median_fee,
            'p95_fee_gwei': p95_fee,
            'fee_cv': fee_cv,
            'fee_jumpiness_p95': p95_jump,

            # Safety objectives
            'deficit_weighted_duration': deficit_weighted_duration,
            'max_deficit_depth': max_deficit_depth,
            'recovery_time_score': recovery_time,

            # Efficiency objectives
            'target_size': target_size,
            'avg_vault_deviation': avg_vault_deviation,
            'capital_per_throughput': capital_per_throughput,

            # Constraint metrics
            'cost_recovery_ratio': metrics['cost_recovery_ratio'],
            'min_balance_ratio': metrics['min_balance'] / target_size,
            'feasible': (0.8 <= metrics['cost_recovery_ratio'] <= 1.3 and
                        metrics['min_balance'] / target_size >= 0.1 and
                        median_fee <= 200.0)
        }

    def calculate_stakeholder_scores(self, theoretical_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate weighted scores for each stakeholder profile."""

        scores = {}

        for stakeholder, objectives in self.stakeholder_profiles.items():
            # UX score (Section 2.3.1 formulation)
            high_fee_penalty = max(0, theoretical_metrics['p95_fee_gwei'] - objectives.fee_tolerance_gwei)
            ux_score = (objectives.ux_stability_weight * theoretical_metrics['fee_cv'] +
                       objectives.ux_jumpiness_weight * theoretical_metrics['fee_jumpiness_p95'] +
                       objectives.ux_high_fee_weight * high_fee_penalty / 100.0)

            # Safety score (Section 2.3.2 formulation)
            safety_score = (objectives.safety_duration_weight * theoretical_metrics['deficit_weighted_duration'] / 1000.0 +
                          objectives.safety_depth_weight * theoretical_metrics['max_deficit_depth'] / 1000.0 +
                          objectives.safety_recovery_weight * theoretical_metrics['recovery_time_score'])

            # Efficiency score (Section 2.3.3 formulation)
            eff_score = (objectives.eff_capital_weight * theoretical_metrics['target_size'] / 1000.0 +
                        objectives.eff_deviation_weight * theoretical_metrics['avg_vault_deviation'] / 1000.0 +
                        objectives.eff_throughput_weight * theoretical_metrics['capital_per_throughput'] / 10.0)

            # Combined score (lower is better)
            total_score = ux_score + safety_score + eff_score

            scores[stakeholder] = {
                'ux_score': ux_score,
                'safety_score': safety_score,
                'efficiency_score': eff_score,
                'total_score': total_score,
                'feasible': theoretical_metrics['feasible']
            }

        return scores

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete stakeholder analysis."""

        print("🎯 THEORETICAL FRAMEWORK STAKEHOLDER ANALYSIS")
        print("=" * 70)
        print("Implementing SUMMARY.md Section 2: Parameter Optimization Problem")
        print("Evaluating research-validated parameter sets for stakeholder profiles")
        print()

        results = {}

        # Analyze each parameter set
        for param_set in self.parameter_candidates:
            print(f"📊 Analyzing {param_set['name']} (μ={param_set['mu']}, ν={param_set['nu']}, H={param_set['H']})...")

            analysis = self.analyze_parameter_set(param_set)

            if analysis['success']:
                theoretical_metrics = analysis['theoretical_objectives']
                stakeholder_scores = self.calculate_stakeholder_scores(theoretical_metrics)

                print(f"   💰 Median Fee: {theoretical_metrics['median_fee_gwei']:.2f} gwei")
                print(f"   📈 CRR: {theoretical_metrics['cost_recovery_ratio']:.3f}")
                print(f"   ⚖️  Feasible: {'✅' if theoretical_metrics['feasible'] else '❌'}")

                results[param_set['name']] = {
                    'parameters': param_set,
                    'metrics': theoretical_metrics,
                    'stakeholder_scores': stakeholder_scores
                }
            else:
                print(f"   ❌ Analysis failed: {analysis['error']}")

            print()

        return results

    def generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate parameter recommendations for each stakeholder."""

        recommendations = {}

        for stakeholder in self.stakeholder_profiles.keys():
            best_param_set = None
            best_score = float('inf')

            # Find best feasible solution for this stakeholder
            feasible_options = []
            for param_name, result in results.items():
                if result['metrics']['feasible']:
                    feasible_options.append((param_name, result['stakeholder_scores'][stakeholder]['total_score']))

            if feasible_options:
                best_param_set = min(feasible_options, key=lambda x: x[1])[0]
                best_score = results[best_param_set]['stakeholder_scores'][stakeholder]['total_score']

                params = results[best_param_set]['parameters']
                metrics = results[best_param_set]['metrics']

                recommendations[stakeholder] = f"""
Recommended: {best_param_set}
Parameters: μ={params['mu']:.3f}, ν={params['nu']:.3f}, H={params['H']}
Performance: {metrics['median_fee_gwei']:.1f} gwei median fee, CRR={metrics['cost_recovery_ratio']:.3f}
Rationale: Best total score ({best_score:.3f}) for {stakeholder.replace('_', ' ')} priorities"""
            else:
                # Find least infeasible option
                best_infeasible = min(results.items(),
                                    key=lambda x: x[1]['stakeholder_scores'][stakeholder]['total_score'])
                recommendations[stakeholder] = f"""
No feasible solutions found for {stakeholder.replace('_', ' ')} priorities.
Least suboptimal: {best_infeasible[0]}
Note: Consider relaxing constraints or alternative parameter exploration."""

        return recommendations


def main():
    analyzer = TheoreticalFrameworkAnalyzer()
    results = analyzer.run_complete_analysis()

    print("🎯 STAKEHOLDER RECOMMENDATIONS")
    print("=" * 50)

    recommendations = analyzer.generate_recommendations(results)
    for stakeholder, recommendation in recommendations.items():
        print(f"\n📋 {stakeholder.replace('_', ' ').upper()}:")
        print(recommendation.strip())

    # Create comparison table
    print(f"\n📊 PARAMETER COMPARISON MATRIX")
    print("=" * 80)

    # Headers
    print(f"{'Parameter Set':<20} {'μ':>6} {'ν':>6} {'H':>6} {'Fee(gwei)':>10} {'CRR':>6} {'Feasible':>10}")
    print("-" * 80)

    for param_name, result in results.items():
        if result:
            params = result['parameters']
            metrics = result['metrics']
            feasible = "✅" if metrics['feasible'] else "❌"

            print(f"{param_name:<20} {params['mu']:>6.2f} {params['nu']:>6.2f} "
                  f"{params['H']:>6} {metrics['median_fee_gwei']:>10.1f} "
                  f"{metrics['cost_recovery_ratio']:>6.2f} {feasible:>10}")

    return results


if __name__ == "__main__":
    results = main()