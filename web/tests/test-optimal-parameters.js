/**
 * Quick test script to verify optimal parameters in the web interface
 * Run this in the browser console after the page loads
 */

async function testOptimalParametersInWebInterface() {
    console.log('🧪 Testing optimal parameters in web interface...');

    try {
        // Wait for all components to be available
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Load historical data if not already loaded
        if (!window.historicalDataLoader.loaded) {
            console.log('Loading historical data...');
            await window.historicalDataLoader.loadAllDatasets();
        }

        // Test the optimal parameters from research
        const optimalParams = {
            mu: 0.0,
            nu: 0.1,
            H: 36
        };

        // Create NSGA-II evaluator
        const nsgaii = new window.NSGAII({
            populationSize: 10,  // Small test
            maxGenerations: 1,
            crossoverRate: 0.9,
            mutationRate: 0.1,
            weights: {
                w1_fee_affordability: 0.4,
                w2_fee_stability: 0.3,
                w3_fee_predictability_1h: 0.15,
                w4_fee_predictability_6h: 0.15,
                w5_insolvency_protection: 0.4,
                w6_deficit_duration: 0.3,
                w7_vault_stress: 0.15,
                w8_continuous_underfunding: 0.15,
                w9_vault_utilization: 0.6,
                w10_deficit_correction: 0.25,
                w11_capital_efficiency: 0.15
            }
        });

        // Create individual with optimal parameters
        const individual = {
            mu: optimalParams.mu,
            nu: optimalParams.nu,
            H: optimalParams.H,
            objectives: []
        };

        // Test evaluation
        console.log('Evaluating optimal parameters...');
        const simulationResults = await nsgaii.runSimulation(individual);
        const scores = nsgaii.calculateCompositeScores(simulationResults);

        console.log('✅ Optimal Parameters Test Results:');
        console.log('Parameters:', optimalParams);
        console.log('UX Score:', scores.uxScore?.toFixed(4));
        console.log('Safety Score:', scores.safetyScore?.toFixed(4));
        console.log('Efficiency Score:', scores.efficiencyScore?.toFixed(4));
        console.log('Overall Score:', scores.overallScore?.toFixed(4));
        console.log('Using Real Simulation:', !simulationResults.is_fallback);

        // Test a suboptimal set for comparison
        const suboptimalIndividual = {
            mu: 0.5,
            nu: 0.9,
            H: 288,
            objectives: []
        };

        console.log('Evaluating suboptimal parameters...');
        const suboptimalResults = await nsgaii.runSimulation(suboptimalIndividual);
        const suboptimalScores = nsgaii.calculateCompositeScores(suboptimalResults);

        console.log('📊 Suboptimal Parameters Test Results:');
        console.log('Parameters:', { mu: 0.5, nu: 0.9, H: 288 });
        console.log('UX Score:', suboptimalScores.uxScore?.toFixed(4));
        console.log('Safety Score:', suboptimalScores.safetyScore?.toFixed(4));
        console.log('Overall Score:', suboptimalScores.overallScore?.toFixed(4));

        // Comparison
        const optimalIsBetter = scores.overallScore > suboptimalScores.overallScore;
        console.log(`\n🎯 Comparison Result: Optimal parameters ${optimalIsBetter ? 'ARE' : 'ARE NOT'} better`);
        console.log(`Score difference: ${(scores.overallScore - suboptimalScores.overallScore).toFixed(4)}`);

        return {
            optimal: scores,
            suboptimal: suboptimalScores,
            optimalIsBetter: optimalIsBetter
        };

    } catch (error) {
        console.error('❌ Test failed:', error);
        return { error: error.message };
    }
}

// Auto-export for console use
window.testOptimalParametersInWebInterface = testOptimalParametersInWebInterface;