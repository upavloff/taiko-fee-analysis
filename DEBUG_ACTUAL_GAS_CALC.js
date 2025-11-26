// 🔍 DEBUG ACTUAL GAS CALCULATION
// Test what gasPerTx is actually calculated in the current implementation

console.log("🔍 DEBUGGING ACTUAL GAS CALCULATION");
console.log("=" .repeat(50));

// Test the current formula with different baseTxVolume values
function testGasCalculation(batchGas, baseTxVolume) {
    const gasPerTx = Math.max(batchGas / Math.max(baseTxVolume, 1), 2000);
    return gasPerTx;
}

const batchGas = 200000;
const testValues = [1, 10, 100, 1000];

console.log("Current JavaScript Formula: max(batchGas / max(baseTxVolume, 1), 2000)");
console.log(`batchGas = ${batchGas}`);
console.log();

testValues.forEach(baseTxVolume => {
    const result = testGasCalculation(batchGas, baseTxVolume);
    console.log(`baseTxVolume = ${baseTxVolume}: gasPerTx = ${result} gas`);
});

console.log();
console.log("🤔 ANALYSIS:");
console.log("For gasPerTx to be ~200 gas:");
console.log("200 = max(200000 / baseTxVolume, 2000)");
console.log("This is impossible because max(anything, 2000) ≥ 2000");
console.log("So gasPerTx should be at least 2000 gas");

console.log();
console.log("🕵️ POSSIBLE ISSUES:");
console.log("1. Different formula being used somewhere else");
console.log("2. Parameters being overridden after calculation");
console.log("3. Display/conversion error in charts");
console.log("4. Mixed up with old backup code");

// Check what would give ~200 gas
console.log();
console.log("🧮 REVERSE CALCULATION:");
console.log("For gasPerTx = 200 to occur:");
console.log("- batchGas / baseTxVolume = 200");
console.log("- 200000 / baseTxVolume = 200");
console.log("- baseTxVolume = 200000 / 200 = 1000");
console.log("- BUT max(200, 2000) = 2000, not 200");
console.log("- So this scenario is mathematically impossible with current formula");