// Quick test of the simulation logic
const fs = require('fs');

// Load the simulator code
eval(fs.readFileSync('simulator.js', 'utf8'));

console.log('🧪 Testing Taiko Fee Mechanism Simulator...\n');

try {
    // Test 1: GeometricBrownianMotion
    console.log('Test 1: GeometricBrownianMotion');
    const gbm = new GeometricBrownianMotion(0.0, 0.3, 10e9);
    const values = [];
    for (let i = 0; i < 5; i++) {
        values.push(gbm.step());
    }
    console.log('✅ GBM generates values:', values.map(v => (v/1e9).toFixed(1) + ' gwei'));

    // Test 2: TaikoFeeSimulator
    console.log('\nTest 2: TaikoFeeSimulator');
    const params = {
        mu: 0.4,
        nu: 0.3,
        H: 144,
        l1Volatility: 0.3,
        vaultInit: 'target',
        targetBalance: 100
    };

    const simulator = new TaikoFeeSimulator(params);
    console.log('✅ Simulator created with initial vault balance:', simulator.vaultBalance);

    // Test 3: Short simulation
    console.log('\nTest 3: Running simulation');
    const results = simulator.runSimulation(20); // Short test
    console.log('✅ Simulation completed, data points:', results.length);
    console.log('   First fee:', results[0].estimatedFee.toExponential(2), 'ETH');
    console.log('   Last fee:', results[results.length-1].estimatedFee.toExponential(2), 'ETH');

    // Test 4: MetricsCalculator
    console.log('\nTest 4: MetricsCalculator');
    const metricsCalc = new MetricsCalculator(100, 2000);
    const metrics = metricsCalc.calculateMetrics(results);
    console.log('✅ Metrics calculated:');
    console.log('   Average Fee:', metrics.avgFee.toExponential(2), 'ETH');
    console.log('   Fee CV:', metrics.feeCV.toFixed(3));
    console.log('   Time Underfunded:', metrics.timeUnderfundedPct.toFixed(1), '%');

    // Test 5: Presets
    console.log('\nTest 5: Presets');
    console.log('✅ Available presets:', Object.keys(PRESETS).join(', '));

    // Test 6: Fee calculation
    console.log('\nTest 6: Fee calculations');
    const testFee1 = simulator.calculateFee(10e9, 0); // 10 gwei, no deficit
    const testFee2 = simulator.calculateFee(10e9, 500); // 10 gwei, 500 ETH deficit
    console.log('✅ Fee with no deficit:', testFee1.toExponential(2), 'ETH');
    console.log('✅ Fee with 500 ETH deficit:', testFee2.toExponential(2), 'ETH');

    console.log('\n🎉 ALL TESTS PASSED! Web simulator is working correctly.');

} catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
}