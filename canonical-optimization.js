/**
 * Canonical Taiko Fee Mechanism Optimization (JavaScript)
 *
 * This is the SINGLE SOURCE OF TRUTH for optimization logic in JavaScript.
 * It mirrors the Python canonical implementation to ensure consistency.
 *
 * Key Features:
 * - NSGA-II multi-objective optimization
 * - Parameter validation and constraint checking
 * - Optimization strategies (balanced, crisis-resilient, etc.)
 * - Performance evaluation and Pareto ranking
 */

import { CanonicalTaikoFeeCalculator, FeeParameters, VaultInitMode, validateFeeParameters } from './canonical-fee-mechanism.js';
import { CanonicalMetricsCalculator, calculateBasicMetrics } from './canonical-metrics.js';

// Optimization strategies enum
export const OptimizationStrategy = {
    BALANCED: "balanced",
    USER_FOCUSED: "user_focused",
    PROTOCOL_SAFETY: "protocol_safety",
    CRISIS_RESILIENT: "crisis_resilient",
    ECONOMIC_EFFICIENCY: "economic_efficiency"
};

// Individual solution class for NSGA-II
export class Individual {
    constructor(mu, nu, H) {
        this.mu = mu;
        this.nu = nu;
        this.H = H;
        this.objectives = [0, 0, 0]; // [user_experience, protocol_safety, economic_efficiency]
        this.constraintViolation = 0;
        this.rank = 0;
        this.crowdingDistance = 0;
        this.dominatedCount = 0;
        this.dominatedSolutions = [];
    }

    /**
     * Check if this individual dominates another
     * @param {Individual} other - Other individual to compare
     * @returns {boolean} True if this dominates other
     */
    dominates(other) {
        let atLeastOneBetter = false;

        for (let i = 0; i < this.objectives.length; i++) {
            if (this.objectives[i] < other.objectives[i]) {
                return false; // This is worse in at least one objective
            }
            if (this.objectives[i] > other.objectives[i]) {
                atLeastOneBetter = true;
            }
        }

        return atLeastOneBetter;
    }

    /**
     * Calculate distance between two individuals in parameter space
     * @param {Individual} other - Other individual
     * @returns {number} Euclidean distance
     */
    distanceTo(other) {
        const dMu = this.mu - other.mu;
        const dNu = this.nu - other.nu;
        const dH = (this.H - other.H) / 1000; // Normalize H scale
        return Math.sqrt(dMu * dMu + dNu * dNu + dH * dH);
    }
}

// Main canonical optimizer class
export class CanonicalOptimizer {
    constructor(strategy = OptimizationStrategy.BALANCED) {
        this.strategy = strategy;
        this.populationSize = 100;
        this.generations = 50;
        this.crossoverProbability = 0.9;
        this.mutationProbability = 0.1;

        // Parameter bounds
        this.bounds = {
            mu: [0.0, 1.0],
            nu: [0.0, 1.0],
            H: [6, 1440] // 6 steps (1 min) to 1440 steps (8 hours)
        };

        this.validHValues = this._generateValidHValues();
    }

    /**
     * Generate valid H values (multiples of 6)
     * @returns {number[]} Array of valid H values
     */
    _generateValidHValues() {
        const values = [];
        for (let h = 6; h <= 1440; h += 6) {
            values.push(h);
        }
        return values;
    }

    /**
     * Create random individual with valid parameters
     * @returns {Individual} Random individual
     */
    _createRandomIndividual() {
        const mu = Math.random() * (this.bounds.mu[1] - this.bounds.mu[0]) + this.bounds.mu[0];
        const nu = Math.random() * (this.bounds.nu[1] - this.bounds.nu[0]) + this.bounds.nu[0];
        const H = this.validHValues[Math.floor(Math.random() * this.validHValues.length)];

        return new Individual(mu, nu, H);
    }

    /**
     * Evaluate individual's performance using simulation
     * @param {Individual} individual - Individual to evaluate
     * @returns {Object} Evaluation results
     */
    async evaluateIndividual(individual) {
        try {
            // Validate parameters
            if (!validateFeeParameters(individual.mu, individual.nu, individual.H)) {
                individual.constraintViolation = 1;
                individual.objectives = [0, 0, 0];
                return { valid: false, reason: "Invalid parameters" };
            }

            // Create calculator with individual's parameters
            const params = new FeeParameters({
                mu: individual.mu,
                nu: individual.nu,
                H: individual.H
            });
            const calculator = new CanonicalTaikoFeeCalculator(params);

            // Run quick simulation
            const simulationResults = await this._runQuickSimulation(calculator);

            // Calculate comprehensive metrics
            const metricsCalculator = new CanonicalMetricsCalculator();
            const metrics = metricsCalculator.calculateComprehensiveMetrics(simulationResults);

            // Set objectives (higher is better for NSGA-II)
            individual.objectives = [
                metrics.userExperienceScore,
                metrics.protocolSafetyScore,
                metrics.economicEfficiencyScore
            ];
            individual.constraintViolation = 0;

            return {
                valid: true,
                metrics: metrics,
                simulationTime: simulationResults.simulationTime || 0
            };

        } catch (error) {
            individual.constraintViolation = 1;
            individual.objectives = [0, 0, 0];
            return { valid: false, reason: error.message };
        }
    }

    /**
     * Run quick simulation for parameter evaluation
     * @param {CanonicalTaikoFeeCalculator} calculator - Fee calculator
     * @returns {Object} Simulation results
     */
    async _runQuickSimulation(calculator) {
        const startTime = Date.now();

        // Create vault with deficit
        const vault = calculator.createVault(VaultInitMode.DEFICIT, { deficit_ratio: 0.1 });

        // Generate synthetic L1 data (short simulation for speed)
        const steps = 120; // 4 minutes
        const baseFee = 20e9; // 20 gwei

        const results = {
            timeStep: [],
            l1Basefee: [],
            estimatedFee: [],
            transactionVolume: [],
            vaultBalance: [],
            vaultDeficit: [],
            feesCollected: [],
            l1CostsPaid: []
        };

        for (let step = 0; step < steps; step++) {
            // Simulate variable L1 basefee
            const l1BasefeeWei = baseFee * (1 + 0.1 * Math.sin(step / 10));

            // Calculate fee and volume
            const l1Cost = calculator.calculateL1CostPerTx(l1BasefeeWei);
            const estimatedFee = calculator.calculateEstimatedFee(l1Cost, vault.deficit);
            const txVolume = calculator.calculateTransactionVolume(estimatedFee);

            // Vault operations
            const feesCollected = estimatedFee * txVolume;
            vault.collectFees(feesCollected);

            let l1CostsPaid = 0;
            if (step % calculator.params.batch_interval_steps === 0) {
                l1CostsPaid = calculator.calculateL1BatchCost(l1BasefeeWei);
                vault.payL1Costs(l1CostsPaid);
            }

            // Record results
            results.timeStep.push(step);
            results.l1Basefee.push(l1BasefeeWei / 1e9); // Convert to gwei
            results.estimatedFee.push(estimatedFee);
            results.transactionVolume.push(txVolume);
            results.vaultBalance.push(vault.balance);
            results.vaultDeficit.push(vault.deficit);
            results.feesCollected.push(feesCollected);
            results.l1CostsPaid.push(l1CostsPaid);
        }

        results.simulationTime = (Date.now() - startTime) / 1000;
        return results;
    }

    /**
     * Perform NSGA-II selection
     * @param {Individual[]} population - Current population
     * @returns {Individual[]} Selected population
     */
    _nsgaIISelection(population) {
        // Fast non-dominated sorting
        const fronts = this._fastNonDominatedSort(population);

        // Calculate crowding distance for each front
        fronts.forEach(front => this._calculateCrowdingDistance(front));

        // Select individuals for next generation
        const selected = [];
        let frontIndex = 0;

        while (selected.length < this.populationSize && frontIndex < fronts.length) {
            const front = fronts[frontIndex];

            if (selected.length + front.length <= this.populationSize) {
                // Include entire front
                selected.push(...front);
            } else {
                // Sort front by crowding distance and select best
                front.sort((a, b) => b.crowdingDistance - a.crowdingDistance);
                const remaining = this.populationSize - selected.length;
                selected.push(...front.slice(0, remaining));
            }

            frontIndex++;
        }

        return selected;
    }

    /**
     * Fast non-dominated sorting algorithm
     * @param {Individual[]} population - Population to sort
     * @returns {Individual[][]} Array of fronts
     */
    _fastNonDominatedSort(population) {
        const fronts = [[]];

        // Initialize domination properties
        population.forEach(individual => {
            individual.dominatedSolutions = [];
            individual.dominatedCount = 0;
        });

        // Find domination relationships
        for (let i = 0; i < population.length; i++) {
            for (let j = 0; j < population.length; j++) {
                if (i !== j) {
                    if (population[i].dominates(population[j])) {
                        population[i].dominatedSolutions.push(population[j]);
                    } else if (population[j].dominates(population[i])) {
                        population[i].dominatedCount++;
                    }
                }
            }

            if (population[i].dominatedCount === 0) {
                population[i].rank = 0;
                fronts[0].push(population[i]);
            }
        }

        // Build subsequent fronts
        let frontIndex = 0;
        while (fronts[frontIndex].length > 0) {
            const nextFront = [];

            fronts[frontIndex].forEach(individual => {
                individual.dominatedSolutions.forEach(dominated => {
                    dominated.dominatedCount--;
                    if (dominated.dominatedCount === 0) {
                        dominated.rank = frontIndex + 1;
                        nextFront.push(dominated);
                    }
                });
            });

            if (nextFront.length > 0) {
                fronts.push(nextFront);
            }
            frontIndex++;
        }

        return fronts.filter(front => front.length > 0);
    }

    /**
     * Calculate crowding distance for individuals in a front
     * @param {Individual[]} front - Front to calculate crowding distance for
     */
    _calculateCrowdingDistance(front) {
        if (front.length === 0) return;

        // Initialize crowding distance
        front.forEach(individual => individual.crowdingDistance = 0);

        const numObjectives = front[0].objectives.length;

        for (let objIndex = 0; objIndex < numObjectives; objIndex++) {
            // Sort by objective value
            front.sort((a, b) => a.objectives[objIndex] - b.objectives[objIndex]);

            // Set boundary points to infinite distance
            front[0].crowdingDistance = Infinity;
            front[front.length - 1].crowdingDistance = Infinity;

            // Calculate objective range
            const objRange = front[front.length - 1].objectives[objIndex] - front[0].objectives[objIndex];

            if (objRange > 0) {
                for (let i = 1; i < front.length - 1; i++) {
                    const distance = (front[i + 1].objectives[objIndex] - front[i - 1].objectives[objIndex]) / objRange;
                    front[i].crowdingDistance += distance;
                }
            }
        }
    }
}

/**
 * Validate a specific parameter set
 * @param {number} mu - L1 weight parameter
 * @param {number} nu - Deficit weight parameter
 * @param {number} H - Prediction horizon
 * @returns {Object} Validation results
 */
export async function validateParameterSet(mu, nu, H) {
    const individual = new Individual(mu, nu, H);
    const optimizer = new CanonicalOptimizer();

    const startTime = Date.now();
    const result = await optimizer.evaluateIndividual(individual);
    const simulationTime = (Date.now() - startTime) / 1000;

    return {
        valid: result.valid,
        constraint_violation: individual.constraintViolation,
        objectives: individual.objectives,
        simulation_time: simulationTime,
        metrics: result.metrics || null
    };
}

/**
 * Get recommended parameter sets for different strategies
 * @param {string} strategy - Optimization strategy
 * @returns {Object} Parameter recommendations
 */
export function getStrategyParameters(strategy = OptimizationStrategy.BALANCED) {
    const strategies = {
        [OptimizationStrategy.BALANCED]: {
            mu: 0.0, nu: 0.27, H: 492,
            description: "Balanced approach optimizing all objectives equally"
        },
        [OptimizationStrategy.USER_FOCUSED]: {
            mu: 0.0, nu: 0.15, H: 288,
            description: "Prioritizes user experience with lower, more stable fees"
        },
        [OptimizationStrategy.PROTOCOL_SAFETY]: {
            mu: 0.0, nu: 0.48, H: 492,
            description: "Conservative approach prioritizing protocol safety"
        },
        [OptimizationStrategy.CRISIS_RESILIENT]: {
            mu: 0.0, nu: 0.88, H: 120,
            description: "Aggressive deficit recovery for crisis situations"
        },
        [OptimizationStrategy.ECONOMIC_EFFICIENCY]: {
            mu: 0.0, nu: 0.35, H: 360,
            description: "Balanced cost recovery with economic efficiency focus"
        }
    };

    return strategies[strategy] || strategies[OptimizationStrategy.BALANCED];
}

// Export classes and functions
export default {
    OptimizationStrategy,
    Individual,
    CanonicalOptimizer,
    validateParameterSet,
    getStrategyParameters
};