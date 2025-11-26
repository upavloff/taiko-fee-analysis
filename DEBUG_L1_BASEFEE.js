// DEBUG: Check what L1 basefee values the simulator is actually generating
console.log('🔍 DEBUGGING L1 BASEFEE VALUES FROM SIMULATOR');

class EIP1559BaseFeeSimulator {
    constructor(mu, sigma, initialValue = 10e9, seed = 42) {
        this.mu = mu; this.sigma = sigma; this.currentBaseFee = initialValue;
        this.seed = seed; this.rng = this.seedableRandom(seed);
        console.log(`Initial basefee: ${initialValue} wei = ${initialValue/1e9} gwei`);
    }

    step() {
        // Simplified - just return the current value for debugging
        return this.currentBaseFee;
    }

    seedableRandom(seed) {
        let state = seed;
        return () => {
            state = (state * 1664525 + 1013904223) % Math.pow(2, 32);
            return state / Math.pow(2, 32);
        };
    }
}

// Test the simulator
const l1Model = new EIP1559BaseFeeSimulator(0.0, 0.3, 10e9, 42);

console.log('\n📊 GENERATED L1 BASEFEE VALUES:');
for (let i = 0; i < 5; i++) {
    const basefeeWei = l1Model.step();
    const basefeeGwei = basefeeWei / 1e9;
    console.log(`Step ${i}: ${basefeeWei} wei = ${basefeeGwei} gwei`);
}

console.log('\n🎯 COMPARISON:');
console.log('- Real recent data: ~0.075 gwei');
console.log('- Simulator initial: 10 gwei (133x higher!)');
console.log('\n💡 ROOT CAUSE: Simulator using 10 gwei instead of realistic 0.075 gwei values!');

// Test with realistic initial value
console.log('\n🔧 TESTING WITH REALISTIC INITIAL VALUE:');
const realisticL1Model = new EIP1559BaseFeeSimulator(0.0, 0.3, 75000000, 42); // 0.075 gwei
for (let i = 0; i < 3; i++) {
    const basefeeWei = realisticL1Model.step();
    const basefeeGwei = basefeeWei / 1e9;
    console.log(`Realistic Step ${i}: ${basefeeWei} wei = ${basefeeGwei} gwei`);

    // Calculate expected L1 cost
    const gasPerTx = 200;
    const l1Cost = (basefeeWei * gasPerTx) / 1e18 * 1e9; // gwei
    console.log(`Expected L1 cost: ${l1Cost} gwei`);
}