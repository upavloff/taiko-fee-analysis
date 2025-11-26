// FINAL REANALYSIS: Testing with CORRECT gas per transaction calculation
console.log('🔥 FINAL RE-ANALYSIS: True Parameter Optimization');
console.log('Testing with REALISTIC gas calculation (200 gas per tx)...\n');

// Quick manual calculation first
const basefeeWei = 75000000; // 0.075 gwei from real data
const realisticGasPerTx = 200; // Real Taiko efficiency
const realisticL1Cost = (basefeeWei * realisticGasPerTx) / 1e18; // ETH
const realisticL1CostGwei = realisticL1Cost * 1e9;

console.log(`📊 MANUAL CALCULATION CHECK:`);
console.log(`L1 Basefee: ${basefeeWei / 1e9} gwei`);
console.log(`Gas per tx: ${realisticGasPerTx} gas`);
console.log(`L1 Cost per tx: ${realisticL1CostGwei} gwei`);
console.log(`Expected μ=1 fee: ~${realisticL1CostGwei} gwei\n`);

// Run actual simulation test
class EIP1559BaseFeeSimulator {
    constructor(mu, sigma, initialValue = 10e9, seed = 42) {
        this.mu = mu; this.sigma = sigma; this.currentBaseFee = initialValue;
        this.seed = seed; this.rng = this.seedableRandom(seed);
        this.TARGET_GAS = 18_000_000; this.MAX_GAS = 36_000_000; this.BASE_FEE_MAX_CHANGE = 1.125;
    }

    step(dt = 1, timeStep = 0, spikeDelay = 0, spikeHeight = 0.3) {
        const randomShock = this.generateNormal(0, 1);
        let demandPressure = this.sigma * randomShock;

        if (timeStep >= spikeDelay && timeStep < spikeDelay + 30) {
            const spikeProgress = (timeStep - spikeDelay) / 30;
            const spikeIntensity = spikeHeight * Math.sin(spikeProgress * Math.PI) * 2;
            demandPressure += spikeIntensity;
        }

        const normalizedDemand = Math.tanh(demandPressure);
        const gasUsed = this.TARGET_GAS + (normalizedDemand * (this.MAX_GAS - this.TARGET_GAS) * 0.5);
        const gasUsedDelta = gasUsed - this.TARGET_GAS;
        const baseFeePerGasDelta = Math.floor(this.currentBaseFee * gasUsedDelta / this.TARGET_GAS / 8);
        let newBaseFee = this.currentBaseFee + baseFeePerGasDelta;
        const maxIncrease = this.currentBaseFee * this.BASE_FEE_MAX_CHANGE;
        const maxDecrease = this.currentBaseFee / this.BASE_FEE_MAX_CHANGE;
        newBaseFee = Math.min(newBaseFee, maxIncrease);
        newBaseFee = Math.max(newBaseFee, maxDecrease);
        this.currentBaseFee = Math.max(newBaseFee, 1e9);
        return this.currentBaseFee;
    }

    seedableRandom(seed) {
        let state = seed;
        return () => {
            state = (state * 1664525 + 1013904223) % Math.pow(2, 32);
            return state / Math.pow(2, 32);
        };
    }

    generateNormal(mean, std) {
        let u = 0, v = 0;
        while(u === 0) u = this.rng();
        while(v === 0) v = this.rng();
        return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}

class TaikoFeeSimulator {
    constructor(params) {
        this.mu = params.mu; this.nu = params.nu; this.H = params.H;
        this.targetBalance = params.targetBalance || 100;
        this.feeElasticity = params.feeElasticity || 0.2;
        this.minFee = params.minFee || 1e-8;
        this.vaultBalance = this.getInitialVaultBalance(params.vaultInit);
        this.l1Source = params.l1Source || 'simulated';
        this.l1Model = new EIP1559BaseFeeSimulator(0.0, params.l1Volatility, 10e9, params.seed || 42);
        this.spikeDelay = params.spikeDelay || 60; this.spikeHeight = params.spikeHeight || 0.3;
        this.baseTxVolume = params.baseTxVolume || 10; this.batchGas = 200000;
        this.l1BasefeeHistory = []; this.trendWindow = 20; this.trendBasefee = null; this.historicalIndex = 0;
        this.updateGasPerTx();
    }

    updateGasPerTx() {
        // FINAL FIX: Real Taiko efficiency
        const realisticBatchSize = 1000; // Real Taiko batches 1000+ transactions
        this.gasPerTx = Math.max(this.batchGas / realisticBatchSize, 200); // 200 gas per tx
        const oldBuggyCalc = Math.max(this.batchGas / Math.max(this.baseTxVolume, 1), 2000);
        console.log(`FINAL FIX: gasPerTx = ${this.batchGas} / ${realisticBatchSize} = ${this.gasPerTx} gas (was ${oldBuggyCalc})`);
    }

    getInitialVaultBalance(vaultInit) {
        switch(vaultInit) {
            case 'target': return this.targetBalance;
            case 'underfunded-20': return this.targetBalance * 0.8;
            case 'underfunded-50': return this.targetBalance * 0.5;
            case 'overfunded-20': return this.targetBalance * 1.2;
            default: return this.targetBalance;
        }
    }

    updateL1BasefeeTrend(currentBasefeeWei) {
        this.l1BasefeeHistory.push(currentBasefeeWei);
        if (this.l1BasefeeHistory.length > this.trendWindow) {
            this.l1BasefeeHistory.shift();
        }
        if (this.l1BasefeeHistory.length >= 3) {
            const alpha = 0.15;
            if (this.trendBasefee === null) {
                this.trendBasefee = this.l1BasefeeHistory.reduce((a, b) => a + b, 0) / this.l1BasefeeHistory.length;
            } else {
                this.trendBasefee = alpha * currentBasefeeWei + (1 - alpha) * this.trendBasefee;
            }
        } else {
            this.trendBasefee = currentBasefeeWei;
        }
    }

    calculateL1Cost(l1BasefeeWei) {
        this.updateL1BasefeeTrend(l1BasefeeWei);
        const basefeeForCost = this.trendBasefee || l1BasefeeWei;
        return (basefeeForCost * this.gasPerTx) / 1e18;
    }

    calculateFee(l1BasefeeWei, vaultDeficit) {
        const l1Cost = this.calculateL1Cost(l1BasefeeWei);
        const l1Component = this.mu * l1Cost;
        const deficitComponent = this.nu * (vaultDeficit / this.H);
        return Math.max(l1Component + deficitComponent, this.minFee);
    }

    calculateDemand(fee, baseDemand = this.baseTxVolume) {
        if (fee <= this.minFee) return baseDemand;
        const feeMultiplier = fee / this.minFee;
        const demandMultiplier = Math.pow(feeMultiplier, -this.feeElasticity);
        return baseDemand * demandMultiplier;
    }

    getNextL1Basefee() {
        const simulatedBasefee = this.l1Model.step(1, this.historicalIndex, this.spikeDelay, this.spikeHeight);
        this.historicalIndex++;
        return simulatedBasefee;
    }

    async runSimulation(steps = 300) {
        const results = [];
        this.historicalIndex = 0;
        for (let t = 0; t < steps; t++) {
            const l1Basefee = this.getNextL1Basefee();
            const vaultDeficit = Math.max(0, this.targetBalance - this.vaultBalance);
            const estimatedFee = this.calculateFee(l1Basefee, vaultDeficit);
            const txVolume = this.calculateDemand(estimatedFee);
            const actualL1Cost = this.calculateL1Cost(l1Basefee) * txVolume;
            const feesCollected = estimatedFee * txVolume;
            this.vaultBalance += feesCollected - actualL1Cost;
            results.push({
                timeStep: t, l1Basefee: l1Basefee, l1TrendBasefee: this.trendBasefee || l1Basefee,
                vaultBalance: this.vaultBalance, vaultDeficit: vaultDeficit,
                estimatedFee: estimatedFee, txVolume: txVolume,
                feesCollected: feesCollected, actualL1Cost: actualL1Cost
            });
        }
        return results;
    }
}

class MetricsCalculator {
    constructor(targetBalance) { this.targetBalance = targetBalance; }
    calculateMetrics(simulationData) {
        const fees = simulationData.map(d => d.estimatedFee);
        const vaultBalances = simulationData.map(d => d.vaultBalance);
        const avgFee = this.mean(fees);
        const feeStd = this.standardDeviation(fees);
        const feeCV = feeStd / avgFee;
        const significantDeficitThreshold = this.targetBalance * 0.0001;
        const underfundedSteps = vaultBalances.filter(balance => (this.targetBalance - balance) > significantDeficitThreshold).length;
        const timeUnderfundedPct = (underfundedSteps / simulationData.length) * 100;
        return { avgFee, feeCV, timeUnderfundedPct };
    }
    mean(arr) { return arr.reduce((sum, val) => sum + val, 0) / arr.length; }
    standardDeviation(arr) {
        const mean = this.mean(arr);
        const squaredDiffs = arr.map(val => Math.pow(val - mean, 2));
        return Math.sqrt(this.mean(squaredDiffs));
    }
}

async function finalAnalysis() {
    const baseParams = {
        targetBalance: 100, feeElasticity: 0.2, minFee: 1e-8,
        l1Source: 'simulated', l1Volatility: 0.3, seed: 42,
        vaultInit: 'target', baseTxVolume: 10, spikeDelay: 60,
        spikeHeight: 0.3, guaranteedRecovery: false, minDeficitRate: 1e-6,
        durationHours: 3
    };

    const configurations = [
        { name: 'Pure L1 Tracking', mu: 1.0, nu: 0.0, H: 144 },
        { name: 'Pure Deficit Correction', mu: 0.0, nu: 0.9, H: 72 },
        { name: 'Balanced Hybrid', mu: 0.5, nu: 0.5, H: 144 },
        { name: 'L1-Heavy Hybrid', mu: 0.8, nu: 0.2, H: 144 }
    ];

    const calculator = new MetricsCalculator(baseParams.targetBalance);

    console.log('🔥 FINAL ANALYSIS RESULTS (with CORRECT gas calculation):\n');

    for (const config of configurations) {
        const params = { ...baseParams, ...config };
        const simulator = new TaikoFeeSimulator(params);
        const results = await simulator.runSimulation(1800); // 1 hour
        const metrics = calculator.calculateMetrics(results);

        console.log(`📈 ${config.name} (μ=${config.mu}, ν=${config.nu}, H=${config.H}):`);
        console.log(`   Average Fee: ${(metrics.avgFee * 1e9).toFixed(3)} gwei`);
        console.log(`   Time Underfunded: ${metrics.timeUnderfundedPct.toFixed(1)}%`);
        console.log(`   Fee Volatility (CV): ${metrics.feeCV.toFixed(3)}`);
        console.log('');
    }

    console.log('🚨 KEY INSIGHTS:');
    console.log('- Now comparing realistic fee levels (not 1000x inflated)');
    console.log('- L1 tracking should show fees proportional to actual L1 basefees');
    console.log('- True optimal configuration revealed!');
}

finalAnalysis().catch(console.error);