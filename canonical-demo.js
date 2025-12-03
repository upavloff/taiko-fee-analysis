/**
 * Canonical Modules Demo
 *
 * Demonstrates how to use the canonical JavaScript modules
 * in place of local fee simulation implementations.
 */

import {
    createDefaultCalculator,
    createBalancedCalculator,
    createCrisisCalculator,
    VaultInitMode,
    getOptimalParameters
} from './canonical-fee-mechanism.js';

import {
    CanonicalMetricsCalculator,
    calculateBasicMetrics,
    validateMetricThresholds
} from './canonical-metrics.js';

import {
    getStrategyParameters,
    OptimizationStrategy
} from './canonical-optimization.js';

// Example: Replace local fee calculation with canonical implementation
export class CanonicalTaikoSimulator {
    constructor(strategy = 'optimal') {
        // Use canonical calculator instead of local implementation
        switch (strategy) {
            case 'optimal':
                this.calculator = createDefaultCalculator();
                break;
            case 'balanced':
                this.calculator = createBalancedCalculator();
                break;
            case 'crisis':
                this.calculator = createCrisisCalculator();
                break;
            default:
                this.calculator = createDefaultCalculator();
        }

        // Initialize vault and metrics calculator
        this.vault = this.calculator.createVault(VaultInitMode.TARGET);
        this.metricsCalculator = new CanonicalMetricsCalculator();

        // Simulation state
        this.results = {
            timeStep: [],
            l1Basefee: [],
            estimatedFee: [],
            transactionVolume: [],
            vaultBalance: [],
            vaultDeficit: [],
            feesCollected: [],
            l1CostsPaid: []
        };
    }

    /**
     * Run a single simulation step using canonical implementations
     * @param {number} l1BasefeeWei - L1 basefee in wei
     * @param {number} step - Current step number
     */
    step(l1BasefeeWei, step) {
        // Use canonical fee calculation
        const l1Cost = this.calculator.calculateL1CostPerTx(l1BasefeeWei);
        const estimatedFee = this.calculator.calculateEstimatedFee(l1Cost, this.vault.deficit);
        const txVolume = this.calculator.calculateTransactionVolume(estimatedFee);

        // Vault operations using canonical vault
        const feesCollected = estimatedFee * txVolume;
        this.vault.collectFees(feesCollected);

        let l1CostsPaid = 0;
        if (step % this.calculator.params.batch_interval_steps === 0) {
            l1CostsPaid = this.calculator.calculateL1BatchCost(l1BasefeeWei);
            this.vault.payL1Costs(l1CostsPaid);
        }

        // Record results
        this.results.timeStep.push(step);
        this.results.l1Basefee.push(l1BasefeeWei / 1e9); // Convert to gwei
        this.results.estimatedFee.push(estimatedFee);
        this.results.transactionVolume.push(txVolume);
        this.results.vaultBalance.push(this.vault.balance);
        this.results.vaultDeficit.push(this.vault.deficit);
        this.results.feesCollected.push(feesCollected);
        this.results.l1CostsPaid.push(l1CostsPaid);
    }

    /**
     * Get performance metrics using canonical metrics calculation
     * @returns {Object} Performance metrics
     */
    getMetrics() {
        // Use canonical metrics calculation
        const basicMetrics = calculateBasicMetrics(this.results);
        const comprehensiveMetrics = this.metricsCalculator.calculateComprehensiveMetrics(this.results);
        const thresholdGrades = validateMetricThresholds(
            basicMetrics.average_fee_gwei,
            basicMetrics.fee_stability_cv,
            basicMetrics.time_underfunded_pct,
            basicMetrics.l1_tracking_error
        );

        return {
            basic: basicMetrics,
            comprehensive: comprehensiveMetrics,
            grades: thresholdGrades
        };
    }

    /**
     * Get current parameters from canonical source
     * @returns {Object} Current parameters
     */
    getParameters() {
        return {
            mu: this.calculator.params.mu,
            nu: this.calculator.params.nu,
            H: this.calculator.params.H,
            target_balance: this.calculator.params.target_balance
        };
    }
}

// Example usage functions
export function demonstrateCanonicalUsage() {
    console.log('🚀 Canonical Modules Demo');
    console.log('='.repeat(40));

    // 1. Get optimal parameters from canonical source
    const optimalParams = getOptimalParameters();
    console.log('📊 Optimal Parameters:', optimalParams);

    // 2. Create simulator with canonical modules
    const simulator = new CanonicalTaikoSimulator('optimal');
    console.log('✅ Simulator created with canonical modules');

    // 3. Run short simulation
    const baseFee = 20e9; // 20 gwei
    for (let step = 0; step < 10; step++) {
        const l1Fee = baseFee * (1 + 0.1 * Math.sin(step / 3));
        simulator.step(l1Fee, step);
    }

    // 4. Get metrics using canonical calculations
    const metrics = simulator.getMetrics();
    console.log('📈 Performance Metrics:');
    console.log(`   Average Fee: ${metrics.basic.average_fee_gwei.toFixed(2)} gwei`);
    console.log(`   Fee Stability: ${metrics.basic.fee_stability_cv.toFixed(3)}`);
    console.log(`   Overall Score: ${metrics.basic.overall_score.toFixed(3)}`);

    // 5. Show strategy parameters
    console.log('\n🎯 Available Strategies:');
    Object.values(OptimizationStrategy).forEach(strategy => {
        const params = getStrategyParameters(strategy);
        console.log(`   ${strategy}: μ=${params.mu}, ν=${params.nu}, H=${params.H}`);
    });

    return {
        simulator,
        metrics,
        optimalParams
    };
}

// Migration guide for existing code
export const MIGRATION_GUIDE = {
    'Old Pattern': 'Replace with Canonical Module',
    'Examples': {
        'Local fee calculation': 'import { createDefaultCalculator } from "./canonical-fee-mechanism.js"',
        'Local metrics calculation': 'import { calculateBasicMetrics } from "./canonical-metrics.js"',
        'Parameter validation': 'import { validateFeeParameters } from "./canonical-fee-mechanism.js"',
        'Optimization parameters': 'import { getStrategyParameters } from "./canonical-optimization.js"'
    },
    'Benefits': [
        'Single source of truth ensures consistency',
        'Automatic updates when canonical modules change',
        'Reduced code duplication and maintenance',
        'Guaranteed compatibility with Python research'
    ]
};

// Export for easy testing
export default {
    CanonicalTaikoSimulator,
    demonstrateCanonicalUsage,
    MIGRATION_GUIDE
};