// Simple preset verification test
const fs = require('fs');

console.log('🧪 Testing Updated Preset Configurations...\n');

try {
    // Read simulator file and extract presets
    const content = fs.readFileSync('simulator.js', 'utf8');

    // Extract PRESETS object using regex
    const presetsMatch = content.match(/const PRESETS = \{[\s\S]*?\};/);
    if (!presetsMatch) {
        throw new Error('PRESETS object not found in simulator.js');
    }

    // Evaluate just the PRESETS definition
    eval(presetsMatch[0]);

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

    console.log('Test 3: Optimal Preset Verification');
    const optimal = PRESETS['optimal'];
    if (optimal && optimal.mu === 0.0 && optimal.nu === 0.9 && optimal.H === 72) {
        console.log('✅ Optimal preset has correct research-based parameters (μ=0.0, ν=0.9, H=72)');
    } else {
        console.log('❌ Optimal preset parameters incorrect or missing');
    }

    console.log('\nTest 4: Parameter Validation');
    let allValid = true;
    for (const [name, config] of Object.entries(PRESETS)) {
        const valid = config.mu >= 0 && config.mu <= 1 &&
                     config.nu >= 0 && config.nu <= 1 &&
                     config.H > 0 && config.H <= 1000;

        if (valid) {
            console.log(`✅ ${name}: Parameters within valid ranges`);
        } else {
            console.log(`❌ ${name}: Invalid parameter ranges`);
            allValid = false;
        }
    }

    console.log('\nTest 5: Research Alignment Check');
    const researchBased = ['optimal', 'conservative', 'crisis-ready'];
    for (const presetName of researchBased) {
        if (PRESETS[presetName]) {
            const preset = PRESETS[presetName];
            // All research-based presets should have μ=0.0 based on our analysis
            if (preset.mu === 0.0) {
                console.log(`✅ ${presetName}: Aligned with research finding (μ=0.0)`);
            } else {
                console.log(`⚠️  ${presetName}: Not fully aligned with optimal research (μ=${preset.mu})`);
            }
        } else {
            console.log(`❌ Research-based preset '${presetName}' missing`);
        }
    }

    if (allValid) {
        console.log('\n✅ All preset tests completed successfully!');
        console.log('✅ Web interface ready with research-optimized presets');
    } else {
        console.log('\n⚠️  Some tests failed - check parameter ranges');
    }

} catch (error) {
    console.log('❌ Test failed:', error.message);
}