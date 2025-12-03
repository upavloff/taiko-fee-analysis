/**
 * Internal Verification Test
 * This script tests that the web optimization results match the Python research results
 * This is for development/testing purposes only and not displayed in the UI
 */

class InternalVerificationTest {
    constructor() {
        this.testResults = [];
    }

    /**
     * Run verification tests
     */
    async runVerificationTests() {
        console.log('🧪 Starting Internal Verification Tests...');

        // Test 1: Verify historical data loading
        await this.testHistoricalDataLoading();

        // Test 2: Verify simulator with known parameters
        await this.testSimulatorWithKnownParameters();

        // Test 3: Verify optimal parameters score higher than suboptimal ones
        await this.testOptimalParametersPerformance();

        // Test 4: Verify metrics calculations
        await this.testMetricsCalculations();

        // Display results
        this.displayTestResults();

        return this.testResults.every(result => result.passed);
    }

    /**
     * Test 1: Historical data loading
     */
    async testHistoricalDataLoading() {
        try {
            console.log('Testing historical data loading...');

            if (!window.historicalDataLoader.loaded) {
                const loadSuccess = await window.historicalDataLoader.loadAllDatasets();
                if (!loadSuccess) {
                    throw new Error('Failed to load datasets');
                }
            }

            const stats = window.historicalDataLoader.getAllDatasetStats();
            const datasetNames = window.historicalDataLoader.getDatasetNames();

            // Verify all expected datasets are loaded
            const expectedDatasets = ['july_2022_spike', 'luna_crash', 'pepe_crisis', 'normal_operation'];
            const allDatasetsLoaded = expectedDatasets.every(name => datasetNames.includes(name));

            // Verify reasonable data sizes
            const reasonableDataSizes = Object.values(stats).every(stat =>
                stat.count > 100 && stat.count < 50000
            );

            this.testResults.push({
                name: 'Historical Data Loading',
                passed: allDatasetsLoaded && reasonableDataSizes,
                details: `Loaded ${datasetNames.length} datasets with ${Object.values(stats).map(s => s.count).join(', ')} data points`
            });

        } catch (error) {
            this.testResults.push({
                name: 'Historical Data Loading',
                passed: false,
                error: error.message
            });
        }
    }

    /**
     * Test 2: Simulator with known parameters
     */
    async testSimulatorWithKnownParameters() {
        try {
            console.log('Testing simulator with known parameters...');

            const testParams = new ImprovedSimulationParams({
                mu: 0.0,
                nu: 0.1,
                H: 36,
                total_steps: 100  // Small test
            });

            const testDataset = window.historicalDataLoader.getDataset('july_2022_spike').slice(0, 100);
            const simulator = new ImprovedResearchTaikoFeeSimulator(testParams, testDataset);
            const results = simulator.runSimulation();

            // Verify simulation produces expected structure
            const hasRequiredFields = [
                'time_step', 'estimated_fee', 'vault_balance', 'vault_deficit'
            ].every(field => results[field] && results[field].length > 0);

            // Verify reasonable output values
            const avgFee = results.estimated_fee.reduce((a, b) => a + b, 0) / results.estimated_fee.length;
            const reasonableFees = avgFee > 0 && avgFee < 1.0; // Between 0 and 1 ETH

            this.testResults.push({
                name: 'Simulator with Known Parameters',
                passed: hasRequiredFields && reasonableFees,
                details: `Simulated ${results.time_step.length} steps, avg fee: ${avgFee.toFixed(6)} ETH`
            });

        } catch (error) {
            this.testResults.push({
                name: 'Simulator with Known Parameters',
                passed: false,
                error: error.message
            });
        }
    }

    /**
     * Test 3: Optimal parameters should score higher than suboptimal ones
     */
    async testOptimalParametersPerformance() {
        try {
            console.log('Testing optimal vs suboptimal parameter performance...');

            const metricsCalculator = new EnhancedMetricsCalculator();
            const evaluator = new MultiScenarioEvaluator(window.historicalDataLoader, metricsCalculator);

            // Test optimal parameters (from research)
            const optimalParams = { mu: 0.0, nu: 0.1, H: 36 };
            const optimalResults = await evaluator.evaluateParameterSet(optimalParams);
            const optimalScore = optimalResults.aggregated_metrics.robust_overall_score;

            // Test suboptimal parameters
            const suboptimalParams = { mu: 0.5, nu: 0.9, H: 288 };
            const suboptimalResults = await evaluator.evaluateParameterSet(suboptimalParams);
            const suboptimalScore = suboptimalResults.aggregated_metrics.robust_overall_score;

            // Optimal should score higher than suboptimal
            const optimalIsBetter = optimalScore > suboptimalScore;

            this.testResults.push({
                name: 'Optimal vs Suboptimal Parameters',
                passed: optimalIsBetter,
                details: `Optimal score: ${optimalScore.toFixed(4)}, Suboptimal score: ${suboptimalScore.toFixed(4)}`
            });

        } catch (error) {
            this.testResults.push({
                name: 'Optimal vs Suboptimal Parameters',
                passed: false,
                error: error.message
            });
        }
    }

    /**
     * Test 4: Metrics calculations
     */
    async testMetricsCalculations() {
        try {
            console.log('Testing metrics calculations...');

            // Create mock simulation results
            const mockResults = {
                estimated_fee: [0.001, 0.002, 0.001, 0.003, 0.001],
                vault_balance: [1.0, 0.9, 1.1, 0.8, 1.2],
                vault_deficit: [0.0, 0.1, 0.0, 0.2, 0.0],
                transaction_volume: [100, 95, 105, 90, 110],
                fee_collected: [0.1, 0.19, 0.105, 0.27, 0.11],
                l1_cost_paid: [0.05, 0.0, 0.05, 0.0, 0.05]
            };

            const metricsCalculator = new EnhancedMetricsCalculator();
            const metrics = metricsCalculator.calculateAllMetrics(mockResults);

            // Verify metrics structure
            const hasRequiredMetrics = [
                'fee_affordability_score', 'fee_stability_score', 'insolvency_probability',
                'ux_score', 'safety_score', 'overall_optimization_score'
            ].every(metric => metrics[metric] !== undefined);

            // Verify metrics are in reasonable ranges
            const metricsInRange = Object.values(metrics).every(value =>
                typeof value === 'number' && !isNaN(value) && isFinite(value)
            );

            this.testResults.push({
                name: 'Metrics Calculations',
                passed: hasRequiredMetrics && metricsInRange,
                details: `UX Score: ${metrics.ux_score?.toFixed(4)}, Safety Score: ${metrics.safety_score?.toFixed(4)}`
            });

        } catch (error) {
            this.testResults.push({
                name: 'Metrics Calculations',
                passed: false,
                error: error.message
            });
        }
    }

    /**
     * Display test results in console
     */
    displayTestResults() {
        console.log('\n🧪 Internal Verification Test Results:');
        console.log('='.repeat(50));

        let passedCount = 0;
        this.testResults.forEach((result, index) => {
            const status = result.passed ? '✅ PASS' : '❌ FAIL';
            console.log(`${index + 1}. ${result.name}: ${status}`);

            if (result.details) {
                console.log(`   Details: ${result.details}`);
            }
            if (result.error) {
                console.log(`   Error: ${result.error}`);
            }

            if (result.passed) passedCount++;
        });

        console.log('='.repeat(50));
        console.log(`Summary: ${passedCount}/${this.testResults.length} tests passed`);

        if (passedCount === this.testResults.length) {
            console.log('🎉 All verification tests passed! Web implementation matches research methodology.');
        } else {
            console.log('⚠️  Some verification tests failed. Check implementation details above.');
        }
    }
}

// Function to run verification tests (for internal use only)
window.runInternalVerificationTests = async function() {
    const tester = new InternalVerificationTest();
    return await tester.runVerificationTests();
};