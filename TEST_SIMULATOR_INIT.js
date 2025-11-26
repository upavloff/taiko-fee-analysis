// 🧪 TEST SIMULATOR INITIALIZATION
// Check if the simulator initializes properly after parameter name changes

console.log("🧪 TESTING SIMULATOR INITIALIZATION");
console.log("=" .repeat(50));

// Load the simulator code
try {
    const fs = require('fs');
    const simulatorCode = fs.readFileSync('./simulator.js', 'utf8');

    // Check if the code contains any obvious syntax errors
    console.log("✅ Simulator file loaded successfully");

    // Try to evaluate the classes
    eval(simulatorCode);

    console.log("✅ Simulator code evaluated without syntax errors");

    // Test creating a simulator instance
    const testParams = {
        mu: 0.0,
        nu: 0.0,
        H: 72,
        txsPerBatch: 100,  // Using new parameter name
        minFee: 1e-8,
        targetBalance: 1.0,
        l1Volatility: 0.3,
        feeElasticity: 0.5
    };

    console.log("🎯 Testing simulator initialization...");
    const simulator = new TaikoFeeSimulator(testParams);

    console.log("✅ Simulator initialized successfully");
    console.log(`   txsPerBatch: ${simulator.txsPerBatch}`);
    console.log(`   gasPerTx: ${simulator.gasPerTx}`);
    console.log(`   mu: ${simulator.mu}`);
    console.log(`   nu: ${simulator.nu}`);

    // Test a simple step
    const testBasefee = 0.075e9; // 0.075 gwei in wei
    const testTimestamp = new Date();

    console.log("🎯 Testing simulation step...");
    const result = simulator.step(testBasefee, testTimestamp);

    console.log("✅ Simulation step completed");
    console.log(`   Fee: ${(result.fee * 1e9).toFixed(6)} gwei`);
    console.log(`   L1 Cost: ${(result.l1Cost * 1e9).toFixed(6)} gwei`);
    console.log(`   Deficit: ${(result.deficit * 1e9).toFixed(6)} gwei`);

    console.log("\n🏆 DIAGNOSIS:");
    console.log("Simulator appears to be working correctly.");
    console.log("If graphs aren't showing, issue is likely in:");
    console.log("1. Web interface parameter passing");
    console.log("2. Chart rendering code");
    console.log("3. Data processing pipeline");

} catch (error) {
    console.log("❌ ERROR:", error.message);
    console.log("   This explains why graphs aren't showing values!");
}