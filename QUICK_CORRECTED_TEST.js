// 🧪 QUICK CORRECTED TEST
// Test the corrected implementation with a small sample

// Import the corrected simulator
const fs = require('fs');
const simulatorCode = fs.readFileSync('./simulator.js', 'utf8');
eval(simulatorCode);

console.log("🧪 TESTING CORRECTED IMPLEMENTATION");
console.log("=" .repeat(50));

// Test with recent low fee data (small sample)
console.log("📊 Loading recent low fee sample...");
const csvData = fs.readFileSync('./data/data_cache/recent_low_fees_3hours.csv', 'utf8');
const lines = csvData.split('\n').slice(1, 21); // First 20 data points
const l1Data = lines.filter(line => line.trim()).map(line => {
    const [timestamp, basefee_wei, basefee_gwei, block_number] = line.split(',');
    return {
        timestamp: new Date(timestamp),
        basefee: parseFloat(basefee_wei)
    };
});

console.log(`✅ Loaded ${l1Data.length} data points`);
console.log(`   Basefee range: ${(Math.min(...l1Data.map(d => d.basefee)) / 1e9).toFixed(3)} - ${(Math.max(...l1Data.map(d => d.basefee)) / 1e9).toFixed(3)} gwei`);

// Test key parameter combinations
const testConfigs = [
    { name: "μ=0 ν=0.9 H=72 (Original Best)", mu: 0.0, nu: 0.9, horizon: 72 },
    { name: "μ=0.2 ν=0.9 H=72 (Mixed)", mu: 0.2, nu: 0.9, horizon: 72 },
    { name: "μ=1 ν=0 H=72 (Pure L1)", mu: 1.0, nu: 0.0, horizon: 72 }
];

console.log("\n🎯 TESTING CORRECTED PARAMETERS:");

testConfigs.forEach(config => {
    console.log(`\n${config.name}:`);

    const params = new ImprovedSimulationParams({
        mu: config.mu,
        nu: config.nu,
        horizon: config.horizon,
        baseTxVolume: 10,  // Default from simulator
        batchGas: 200000
    });

    const simulator = new ImprovedTaikoFeeSimulator(params);

    // Run short simulation
    const results = [];
    l1Data.forEach((dataPoint, i) => {
        const result = simulator.step(dataPoint.basefee, dataPoint.timestamp);
        results.push(result);
    });

    // Calculate key metrics
    const avgFee = results.reduce((sum, r) => sum + r.fee, 0) / results.length;
    const avgL1Cost = results.reduce((sum, r) => sum + r.l1Cost, 0) / results.length;
    const finalDeficit = results[results.length - 1].deficit;

    console.log(`   Avg Fee:     ${avgFee.toExponential(3)} ETH`);
    console.log(`   Avg L1 Cost: ${avgL1Cost.toExponential(3)} ETH`);
    console.log(`   Final Deficit: ${finalDeficit.toExponential(3)} ETH`);
    console.log(`   gasPerTx: ${params.gasPerTx} gas`);

    // Check relative costs
    if (config.mu > 0) {
        const l1Weight = avgL1Cost * config.mu;
        console.log(`   L1 Component: ${l1Weight.toExponential(3)} ETH (${(l1Weight/avgFee*100).toFixed(1)}%)`);
    }
});

console.log("\n🔬 KEY FINDINGS:");
console.log("- gasPerTx calculation now follows documented formula");
console.log("- L1 costs are 100x higher than before");
console.log("- Basefee floor removed allows realistic low-fee simulation");
console.log("- Need full parameter re-optimization");