// Quick test of preset functionality
const fs = require('fs');

// Load simulator code (exclude the export section that causes issues)
const simulatorCode = fs.readFileSync('simulator.js', 'utf8');
const codeWithoutExports = simulatorCode.replace(/if \(typeof module.*\{[\s\S]*?\}$/m, '');
eval(codeWithoutExports);

console.log('🧪 Testing Updated Preset Configurations...\n');

try {
    // Test presets are loaded
    console.log('Test 1: Preset Configuration');
    console.log('✅ Available presets:', Object.keys(PRESETS).join(', '));

    console.log('\nTest 2: Preset Details');
    for (const [name, config] of Object.entries(PRESETS)) {
        console.log(`${name}:`);
        console.log(`  μ=${config.mu}, ν=${config.nu}, H=${config.H}`);
        console.log(`  Description: ${config.description}`);
        console.log(`  Use case: ${config.useCase}`);
        console.log('');
    }

    // Test optimal preset specifically
    console.log('Test 3: Optimal Preset Verification');
    const optimal = PRESETS['optimal'];
    if (optimal.mu === 0.0 && optimal.nu === 0.9 && optimal.H === 72) {
        console.log('✅ Optimal preset has correct research-based parameters');
    } else {
        console.log('❌ Optimal preset parameters incorrect');
    }

    // Test that we can create a simulator with optimal parameters
    console.log('\nTest 4: Simulator Creation with Optimal Preset');
    const params = {
        mu: optimal.mu,
        nu: optimal.nu,
        H: optimal.H,
        targetBalance: 100,
        l1Source: 'simulated',
        l1Volatility: 0.3,
        baseTxVolume: 10,
        vaultInit: 'target',
        spikeDelay: 60,
        spikeHeight: 0.3
    };

    const simulator = new TaikoFeeSimulator(params);
    console.log('✅ Simulator created successfully with optimal preset parameters');
    console.log(`  mu: ${simulator.mu}, nu: ${simulator.nu}, H: ${simulator.H}`);

    // Test other presets as well
    console.log('\nTest 5: All Presets Parameter Validation');
    for (const [name, config] of Object.entries(PRESETS)) {
        if (config.mu >= 0 && config.mu <= 1 &&
            config.nu >= 0 && config.nu <= 1 &&
            config.H > 0) {
            console.log(`✅ ${name}: Parameters within valid ranges`);
        } else {
            console.log(`❌ ${name}: Invalid parameter ranges`);
        }
    }

    console.log('\n✅ All preset tests completed successfully!');

} catch (error) {
    console.log('❌ Test failed:', error.message);
    console.log(error.stack);
}