"""
Constraint-Aware Fee Mechanism Optimization Demo

This demo showcases the formal optimization framework for fee mechanism parameters
with hard constraints (CRR, ruin probability) and soft objectives (UX, safety,
capital efficiency).

Features:
- Interactive parameter optimization with constraint visualization
- Pareto frontier analysis with constraint boundaries
- Historical scenario evaluation and stress testing
- Comparison of simplified θ=(μ,ν,H,λ_B) vs full parameter vectors
- Real-time constraint satisfaction monitoring

Usage:
    python demo_constraint_optimization.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from typing import Dict, List, Any, Optional
import warnings

# Add core modules to path
sys.path.append(str(Path(__file__).parent))

try:
    from core.canonical_optimization import (
        CanonicalOptimizer, Individual, OptimizationBounds, OptimizationStrategy,
        optimize_simplified_parameters, optimize_with_constraints
    )
    from core.canonical_scenarios import (
        CanonicalScenarioLoader, ScenarioEvaluator, load_default_scenarios
    )

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    MODULES_AVAILABLE = False

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")


class ConstraintOptimizationDemo:
    """Interactive demonstration of constraint-aware optimization."""

    def __init__(self):
        """Initialize demo with configuration."""
        if not MODULES_AVAILABLE:
            raise RuntimeError("Core modules not available for demo")

        # Demo configuration
        self.population_size = 50
        self.generations = 20
        self.scenarios = None

        # Results storage
        self.optimization_results = {}

        # Known optimal parameters for reference
        self.reference_params = {
            'mu': 0.0, 'nu': 0.27, 'H': 492, 'lambda_B': 0.1,
            'alpha_data': 20000.0, 'Q_bar': 690000.0, 'T': 1000.0
        }

    def run_full_demo(self):
        """Run complete constraint optimization demonstration."""
        print("🚀 CONSTRAINT-AWARE FEE MECHANISM OPTIMIZATION DEMO")
        print("=" * 70)

        # Load scenarios
        self._load_scenarios()

        # Demo 1: Simplified Parameter Optimization
        print("\n📊 DEMO 1: Simplified Parameter Vector Optimization")
        print("-" * 50)
        self._demo_simplified_optimization()

        # Demo 2: Constraint Evaluation
        print("\n🔒 DEMO 2: Constraint Evaluation and Validation")
        print("-" * 50)
        self._demo_constraint_evaluation()

        # Demo 3: Pareto Frontier Analysis
        print("\n📈 DEMO 3: Pareto Frontier with Constraint Boundaries")
        print("-" * 50)
        self._demo_pareto_analysis()

        # Demo 4: Scenario-Based Stress Testing
        print("\n🧪 DEMO 4: Scenario-Based Stress Testing")
        print("-" * 50)
        self._demo_stress_testing()

        # Demo 5: Parameter Space Exploration
        print("\n🗺️  DEMO 5: Parameter Space Constraint Visualization")
        print("-" * 50)
        self._demo_parameter_space()

        print("\n" + "=" * 70)
        print("✅ DEMO COMPLETED - All constraint optimization features demonstrated!")
        print("=" * 70)

    def _load_scenarios(self):
        """Load scenarios for constraint evaluation."""
        print("Loading evaluation scenarios...")

        try:
            loader = CanonicalScenarioLoader()
            scenario_data = loader.load_all_scenarios()

            if len(scenario_data) > 0:
                self.scenarios = list(scenario_data.values())
                scenario_types = {}
                total_steps = 0

                for scenario in self.scenarios:
                    scenario_type = scenario.scenario_type.value
                    if scenario_type not in scenario_types:
                        scenario_types[scenario_type] = 0
                    scenario_types[scenario_type] += 1
                    total_steps += scenario.duration_steps

                print(f"✓ Loaded {len(self.scenarios)} scenarios:")
                for scenario_type, count in scenario_types.items():
                    print(f"  • {scenario_type}: {count} scenarios")
                print(f"✓ Total simulation steps across all scenarios: {total_steps:,}")

            else:
                print("⚠️  No scenarios loaded - using synthetic data for demo")
                self._create_synthetic_scenarios()

        except Exception as e:
            print(f"⚠️  Error loading scenarios: {e}")
            print("Creating synthetic scenarios for demo...")
            self._create_synthetic_scenarios()

    def _create_synthetic_scenarios(self):
        """Create synthetic scenarios for demo when real data unavailable."""
        from core.canonical_scenarios import ScenarioData, ScenarioType

        # Create simple synthetic scenarios
        scenarios = []

        # Stable scenario
        stable_data = [15e9] * 1000  # 15 gwei constant
        scenarios.append(ScenarioData(
            name="synthetic_stable",
            scenario_type=ScenarioType.STRESS,
            l1_basefee_wei=stable_data,
            description="Stable 15 gwei baseline"
        ))

        # Volatile scenario
        volatile_data = [15e9 + 10e9 * np.sin(i * 0.02) for i in range(1000)]  # 15±10 gwei sine wave
        scenarios.append(ScenarioData(
            name="synthetic_volatile",
            scenario_type=ScenarioType.STRESS,
            l1_basefee_wei=volatile_data,
            description="Volatile 15±10 gwei oscillation"
        ))

        # Spike scenario
        spike_data = [15e9] * 200 + [100e9] * 100 + [15e9] * 200  # 100 gwei spike
        scenarios.append(ScenarioData(
            name="synthetic_spike",
            scenario_type=ScenarioType.STRESS,
            l1_basefee_wei=spike_data,
            description="100 gwei fee spike"
        ))

        self.scenarios = scenarios
        print(f"✓ Created {len(scenarios)} synthetic scenarios for demo")

    def _demo_simplified_optimization(self):
        """Demonstrate simplified parameter vector optimization."""
        print("Running simplified θ=(μ,ν,H,λ_B) optimization...")

        start_time = time.time()

        # Run optimization with constraints
        results = optimize_simplified_parameters(
            strategy=OptimizationStrategy.BALANCED,
            population_size=self.population_size,
            generations=self.generations,
            enable_constraints=True
        )

        optimization_time = time.time() - start_time
        self.optimization_results['simplified'] = results

        # Display results
        pareto_front = results.get('pareto_front', [])
        print(f"✓ Optimization completed in {optimization_time:.2f}s")
        print(f"✓ Pareto frontier size: {len(pareto_front)}")

        if pareto_front:
            # Analyze constraint satisfaction
            feasible_solutions = [ind for ind in pareto_front if ind.constraint_violation == 0.0]
            constraint_satisfaction_rate = len(feasible_solutions) / len(pareto_front)

            print(f"✓ Constraint satisfaction rate: {constraint_satisfaction_rate:.2%}")

            # Show best solution
            if feasible_solutions:
                best_solution = min(feasible_solutions, key=lambda x: sum(x.objectives) if x.objectives else float('inf'))
                print(f"✓ Best feasible solution found:")
                print(f"  μ={best_solution.mu:.3f}, ν={best_solution.nu:.3f}, H={best_solution.H}, λ_B={best_solution.lambda_B:.3f}")

                if hasattr(best_solution, 'crr_violation') and hasattr(best_solution, 'ruin_probability'):
                    print(f"  CRR violation: {best_solution.crr_violation:.4f}")
                    print(f"  Ruin probability: {best_solution.ruin_probability:.4f}")
            else:
                print("⚠️  No fully feasible solutions found - constraints may be too strict")

    def _demo_constraint_evaluation(self):
        """Demonstrate constraint evaluation with reference parameters."""
        print("Evaluating constraints for reference optimal parameters...")

        if not self.scenarios:
            print("⚠️  No scenarios available for constraint evaluation")
            return

        # Create evaluator
        scenario_dict = {f"scenario_{i}": scenario for i, scenario in enumerate(self.scenarios)}
        evaluator = ScenarioEvaluator(scenario_dict)

        # Evaluate reference parameters
        eval_results = evaluator.evaluate_parameter_set(**self.reference_params)

        print(f"✓ Reference parameter evaluation:")
        print(f"  μ={self.reference_params['mu']}, ν={self.reference_params['nu']}, H={self.reference_params['H']}")
        print(f"  Average CRR: {eval_results.average_crr:.3f}")
        print(f"  Worst-case CRR: {eval_results.worst_case_crr:.3f}")
        print(f"  Average ruin probability: {eval_results.average_ruin_prob:.4f}")
        print(f"  Max ruin probability: {eval_results.max_ruin_prob:.4f}")
        print(f"  CRR constraint satisfied: {eval_results.crr_constraint_satisfied}")
        print(f"  Ruin constraint satisfied: {eval_results.ruin_constraint_satisfied}")
        print(f"  All constraints satisfied: {eval_results.all_constraints_satisfied}")

    def _demo_pareto_analysis(self):
        """Demonstrate Pareto frontier analysis with constraint visualization."""
        print("Generating Pareto frontier visualization...")

        if 'simplified' not in self.optimization_results:
            print("⚠️  No optimization results available for analysis")
            return

        try:
            # Create visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Constraint-Aware Optimization Results', fontsize=16, fontweight='bold')

            pareto_front = self.optimization_results['simplified'].get('pareto_front', [])

            if not pareto_front:
                print("⚠️  No Pareto front data available")
                return

            # Extract data
            feasible = [ind for ind in pareto_front if ind.constraint_violation == 0.0]
            infeasible = [ind for ind in pareto_front if ind.constraint_violation > 0.0]

            # Plot 1: Parameter space (μ vs ν)
            if feasible:
                ax1.scatter([ind.mu for ind in feasible], [ind.nu for ind in feasible],
                          c='green', label='Feasible', alpha=0.7, s=60)
            if infeasible:
                ax1.scatter([ind.mu for ind in infeasible], [ind.nu for ind in infeasible],
                          c='red', label='Infeasible', alpha=0.5, s=40)

            # Add reference point
            ax1.scatter([self.reference_params['mu']], [self.reference_params['nu']],
                       c='gold', marker='★', s=200, label='Reference Optimal', edgecolor='black')

            ax1.set_xlabel('μ (L1 weight)')
            ax1.set_ylabel('ν (deficit weight)')
            ax1.set_title('Parameter Space: μ vs ν')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Constraint violations
            if pareto_front:
                violations = [ind.constraint_violation for ind in pareto_front]
                crr_violations = [getattr(ind, 'crr_violation', 0) for ind in pareto_front]
                ruin_probs = [getattr(ind, 'ruin_probability', 0) for ind in pareto_front]

                ax2.hist(violations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.axvline(x=0, color='red', linestyle='--', label='Feasible threshold')
                ax2.set_xlabel('Total Constraint Violation')
                ax2.set_ylabel('Number of Solutions')
                ax2.set_title('Distribution of Constraint Violations')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Plot 3: CRR vs Ruin Probability
                colors = ['green' if v == 0 else 'red' for v in violations]
                scatter = ax3.scatter(crr_violations, ruin_probs, c=colors, alpha=0.7)
                ax3.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='CRR tolerance (±5%)')
                ax3.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='Ruin probability limit (1%)')
                ax3.set_xlabel('CRR Constraint Violation')
                ax3.set_ylabel('Ruin Probability')
                ax3.set_title('Hard Constraints: CRR vs Ruin Probability')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                # Plot 4: H parameter distribution
                H_values = [ind.H for ind in pareto_front]
                colors_H = ['green' if violations[i] == 0 else 'red' for i in range(len(H_values))]
                ax4.scatter(H_values, [i for i in range(len(H_values))], c=colors_H, alpha=0.7)
                ax4.axvline(x=self.reference_params['H'], color='gold', linestyle='--', linewidth=2, label='Reference H=492')
                ax4.set_xlabel('H (prediction horizon)')
                ax4.set_ylabel('Solution Index')
                ax4.set_title('Prediction Horizon Distribution')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            print(f"✓ Pareto frontier visualization completed")
            print(f"  Total solutions: {len(pareto_front)}")
            print(f"  Feasible solutions: {len(feasible)}")
            print(f"  Infeasible solutions: {len(infeasible)}")

        except Exception as e:
            print(f"⚠️  Visualization error: {e}")

    def _demo_stress_testing(self):
        """Demonstrate scenario-based stress testing."""
        print("Running stress testing across scenarios...")

        if not self.scenarios:
            print("⚠️  No scenarios available for stress testing")
            return

        # Test multiple parameter configurations
        test_configs = [
            {'name': 'Reference', **self.reference_params},
            {'name': 'Conservative', 'mu': 0.0, 'nu': 0.48, 'H': 492, 'lambda_B': 0.1, 'alpha_data': 20000.0, 'Q_bar': 690000.0, 'T': 1500.0},
            {'name': 'Aggressive', 'mu': 0.0, 'nu': 0.88, 'H': 120, 'lambda_B': 0.3, 'alpha_data': 20000.0, 'Q_bar': 690000.0, 'T': 800.0}
        ]

        try:
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Stress Testing Results Across Scenarios', fontsize=14, fontweight='bold')

            scenario_dict = {f"scenario_{i}": scenario for i, scenario in enumerate(self.scenarios)}
            evaluator = ScenarioEvaluator(scenario_dict)

            config_names = []
            crr_values = []
            ruin_probabilities = []
            constraint_satisfaction = []

            for config in test_configs:
                name = config.pop('name')
                config_names.append(name)

                print(f"  Testing {name} configuration...")

                eval_results = evaluator.evaluate_parameter_set(**config)

                crr_values.append(eval_results.average_crr)
                ruin_probabilities.append(eval_results.max_ruin_prob)
                constraint_satisfaction.append(1 if eval_results.all_constraints_satisfied else 0)

                print(f"    CRR: {eval_results.average_crr:.3f}, Ruin prob: {eval_results.max_ruin_prob:.4f}, Satisfied: {eval_results.all_constraints_satisfied}")

            # Plot CRR comparison
            bars1 = ax1.bar(config_names, crr_values, alpha=0.7, color=['gold', 'green', 'orange'])
            ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Target CRR = 1.0')
            ax1.axhline(y=0.95, color='red', linestyle=':', alpha=0.5, label='CRR tolerance (±5%)')
            ax1.axhline(y=1.05, color='red', linestyle=':', alpha=0.5)
            ax1.set_ylabel('Average Cost Recovery Ratio')
            ax1.set_title('CRR Performance Across Configurations')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Add value labels on bars
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{crr_values[i]:.3f}',
                        ha='center', va='bottom', fontweight='bold')

            # Plot ruin probability comparison
            colors = ['green' if ruin_probabilities[i] <= 0.01 else 'red' for i in range(len(ruin_probabilities))]
            bars2 = ax2.bar(config_names, ruin_probabilities, alpha=0.7, color=colors)
            ax2.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Ruin probability limit (1%)')
            ax2.set_ylabel('Maximum Ruin Probability')
            ax2.set_title('Ruin Probability Across Configurations')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Add value labels on bars
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0005, f'{ruin_probabilities[i]:.4f}',
                        ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.show()

            print(f"✓ Stress testing completed")
            print(f"  Configurations satisfying constraints: {sum(constraint_satisfaction)}/{len(test_configs)}")

        except Exception as e:
            print(f"⚠️  Stress testing error: {e}")

    def _demo_parameter_space(self):
        """Demonstrate parameter space exploration with constraint overlays."""
        print("Exploring parameter space with constraint visualization...")

        try:
            # Create parameter grid for μ and ν
            mu_range = np.linspace(0, 0.5, 20)
            nu_range = np.linspace(0.1, 0.9, 20)

            mu_grid, nu_grid = np.meshgrid(mu_range, nu_range)

            # Initialize constraint satisfaction grid
            constraint_grid = np.zeros_like(mu_grid)
            crr_grid = np.ones_like(mu_grid)

            if not self.scenarios:
                print("⚠️  No scenarios available for parameter space analysis")
                return

            # Create evaluator
            scenario_dict = {f"scenario_{i}": scenario for i, scenario in enumerate(self.scenarios[:3])}  # Use first 3 scenarios for speed
            evaluator = ScenarioEvaluator(scenario_dict)

            print("  Evaluating parameter grid (this may take a while)...")

            # Evaluate each point in the grid
            total_points = mu_grid.size
            evaluated_points = 0

            for i in range(mu_grid.shape[0]):
                for j in range(mu_grid.shape[1]):
                    mu_val = mu_grid[i, j]
                    nu_val = nu_grid[i, j]

                    # Use reference values for other parameters
                    test_params = {
                        'mu': mu_val, 'nu': nu_val, 'H': self.reference_params['H'],
                        'lambda_B': self.reference_params['lambda_B'], 'alpha_data': self.reference_params['alpha_data'],
                        'Q_bar': self.reference_params['Q_bar'], 'T': self.reference_params['T']
                    }

                    try:
                        eval_results = evaluator.evaluate_parameter_set(**test_params)
                        constraint_grid[i, j] = 1 if eval_results.all_constraints_satisfied else 0
                        crr_grid[i, j] = eval_results.average_crr

                        evaluated_points += 1
                        if evaluated_points % 50 == 0:
                            progress = evaluated_points / total_points * 100
                            print(f"    Progress: {progress:.1f}% ({evaluated_points}/{total_points})")

                    except Exception:
                        # Handle evaluation failures
                        constraint_grid[i, j] = 0
                        crr_grid[i, j] = 0.5

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Parameter Space Analysis: μ vs ν', fontsize=14, fontweight='bold')

            # Plot 1: Constraint satisfaction
            im1 = ax1.contourf(mu_grid, nu_grid, constraint_grid, levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.7)
            ax1.contour(mu_grid, nu_grid, constraint_grid, levels=[0.5], colors='black', linewidths=2)
            ax1.scatter([self.reference_params['mu']], [self.reference_params['nu']],
                       c='gold', marker='★', s=200, label='Reference Optimal', edgecolor='black')
            ax1.set_xlabel('μ (L1 weight)')
            ax1.set_ylabel('ν (deficit weight)')
            ax1.set_title('Constraint Satisfaction Regions')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Constraints Satisfied')
            cbar1.set_ticks([0, 1])
            cbar1.set_ticklabels(['No', 'Yes'])

            # Plot 2: CRR heatmap
            im2 = ax2.contourf(mu_grid, nu_grid, crr_grid, levels=20, cmap='RdYlGn')
            ax2.contour(mu_grid, nu_grid, crr_grid, levels=[0.95, 1.05], colors='red', linewidths=2, linestyles=['--', '--'])
            ax2.scatter([self.reference_params['mu']], [self.reference_params['nu']],
                       c='gold', marker='★', s=200, label='Reference Optimal', edgecolor='black')
            ax2.set_xlabel('μ (L1 weight)')
            ax2.set_ylabel('ν (deficit weight)')
            ax2.set_title('Cost Recovery Ratio (CRR)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Average CRR')

            plt.tight_layout()
            plt.show()

            # Calculate statistics
            feasible_points = np.sum(constraint_grid)
            total_points = constraint_grid.size
            feasible_percentage = (feasible_points / total_points) * 100

            print(f"✓ Parameter space analysis completed")
            print(f"  Total points evaluated: {total_points}")
            print(f"  Feasible points: {feasible_points} ({feasible_percentage:.1f}%)")
            print(f"  Infeasible points: {total_points - feasible_points} ({100 - feasible_percentage:.1f}%)")

        except Exception as e:
            print(f"⚠️  Parameter space analysis error: {e}")


def main():
    """Run constraint optimization demonstration."""
    if not MODULES_AVAILABLE:
        print("❌ Error: Core modules not available. Please check import paths.")
        return

    try:
        demo = ConstraintOptimizationDemo()
        demo.run_full_demo()

    except KeyboardInterrupt:
        print("\n⏸️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()