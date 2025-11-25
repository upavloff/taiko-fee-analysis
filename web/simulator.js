// Taiko Fee Mechanism Simulator - JavaScript Implementation
// Ported from Python analysis for web interface

class GeometricBrownianMotion {
    constructor(mu, sigma, initialValue = 10e9, seed = 42) {
        this.mu = mu;           // Drift (trend)
        this.sigma = sigma;     // Volatility
        this.currentValue = initialValue;  // Current basefee in wei
        this.seed = seed;       // Random seed for reproducible results
        this.rng = this.seedableRandom(seed);
    }

    step(dt = 1) {
        // Generate next basefee value using GBM formula
        const randomShock = this.generateNormal(0, 1);
        const drift = (this.mu - 0.5 * this.sigma * this.sigma) * dt;
        const diffusion = this.sigma * Math.sqrt(dt) * randomShock;

        this.currentValue *= Math.exp(drift + diffusion);

        // Ensure basefee doesn't go below 1 gwei
        this.currentValue = Math.max(this.currentValue, 1e9);

        return this.currentValue;
    }

    seedableRandom(seed) {
        // Simple seedable random number generator (LCG)
        let state = seed;
        return () => {
            state = (state * 1664525 + 1013904223) % Math.pow(2, 32);
            return state / Math.pow(2, 32);
        };
    }

    generateNormal(mean, std) {
        // Box-Muller transform for normal distribution using seeded random
        let u = 0, v = 0;
        while(u === 0) u = this.rng(); // Converting [0,1) to (0,1)
        while(v === 0) v = this.rng();
        return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}

class TaikoFeeSimulator {
    constructor(params) {
        this.mu = params.mu;                    // L1 cost weight
        this.nu = params.nu;                    // Deficit weight
        this.H = params.H;                      // Time horizon
        this.targetBalance = params.targetBalance || 1000;
        this.feeElasticity = params.feeElasticity || 0.2;
        this.minFee = params.minFee || 1e-8;

        // Vault initialization
        this.vaultBalance = this.getInitialVaultBalance(params.vaultInit);

        // L1 model parameters
        this.l1Model = new GeometricBrownianMotion(0.0, params.l1Volatility, 10e9, params.seed || 42);

        // Transaction parameters
        this.gasPerTx = 2000;          // Gas per transaction
        this.baseTxVolume = 100;       // Base transaction volume per step
    }

    getInitialVaultBalance(vaultInit) {
        switch(vaultInit) {
            case 'target':
                return this.targetBalance;
            case 'underfunded-20':
                return this.targetBalance * 0.8;
            case 'underfunded-50':
                return this.targetBalance * 0.5;
            case 'overfunded-20':
                return this.targetBalance * 1.2;
            default:
                return this.targetBalance;
        }
    }

    calculateL1Cost(l1BasefeeWei) {
        // Calculate L1 cost per transaction in ETH
        return (l1BasefeeWei * this.gasPerTx) / 1e18;
    }

    calculateFee(l1BasefeeWei, vaultDeficit) {
        const l1Cost = this.calculateL1Cost(l1BasefeeWei);
        const l1Component = this.mu * l1Cost;
        const deficitComponent = this.nu * (vaultDeficit / this.H);

        return Math.max(l1Component + deficitComponent, this.minFee);
    }

    calculateDemand(fee, baseDemand = this.baseTxVolume) {
        // Simple demand model with price elasticity
        if (fee <= this.minFee) return baseDemand;

        const feeMultiplier = fee / this.minFee;
        const demandMultiplier = Math.pow(feeMultiplier, -this.feeElasticity);

        return baseDemand * demandMultiplier;
    }

    runSimulation(steps = 300) {
        const results = [];

        for (let t = 0; t < steps; t++) {
            // Get current L1 basefee
            const l1Basefee = this.l1Model.step();

            // Calculate vault deficit
            const vaultDeficit = Math.max(0, this.targetBalance - this.vaultBalance);

            // Calculate current fee
            const estimatedFee = this.calculateFee(l1Basefee, vaultDeficit);

            // Calculate transaction demand based on fee
            const txVolume = this.calculateDemand(estimatedFee);

            // Calculate actual L1 costs
            const actualL1Cost = this.calculateL1Cost(l1Basefee) * txVolume;

            // Calculate fees collected
            const feesCollected = estimatedFee * txVolume;

            // Update vault balance
            this.vaultBalance += feesCollected - actualL1Cost;

            // Store results
            results.push({
                timeStep: t,
                l1Basefee: l1Basefee,
                vaultBalance: this.vaultBalance,
                vaultDeficit: vaultDeficit,
                estimatedFee: estimatedFee,
                txVolume: txVolume,
                feesCollected: feesCollected,
                actualL1Cost: actualL1Cost
            });
        }

        return results;
    }
}

class MetricsCalculator {
    constructor(targetBalance) {
        this.targetBalance = targetBalance;
    }

    calculateMetrics(simulationData) {
        const fees = simulationData.map(d => d.estimatedFee);
        const vaultBalances = simulationData.map(d => d.vaultBalance);
        const l1Basefees = simulationData.map(d => d.l1Basefee);

        // Average fee
        const avgFee = this.mean(fees);

        // Fee coefficient of variation
        const feeStd = this.standardDeviation(fees);
        const feeCV = feeStd / avgFee;

        // Time underfunded percentage
        const underfundedSteps = vaultBalances.filter(balance => balance < this.targetBalance).length;
        const timeUnderfundedPct = (underfundedSteps / simulationData.length) * 100;

        // L1 tracking error (simplified)
        const l1Costs = l1Basefees.map(basefee => (basefee * 2000) / 1e18);
        const normalizedFees = fees.map(fee => fee / this.mean(fees));
        const normalizedL1Costs = l1Costs.map(cost => cost / this.mean(l1Costs));
        const trackingError = this.standardDeviation(normalizedFees.map((fee, i) => fee - normalizedL1Costs[i]));

        // 95th percentile fee
        const sortedFees = [...fees].sort((a, b) => a - b);
        const fee95thPercentile = sortedFees[Math.floor(sortedFees.length * 0.95)];

        // Vault balance statistics
        const vaultBalanceStd = this.standardDeviation(vaultBalances);
        const maxDeficit = Math.max(...simulationData.map(d => d.vaultDeficit));

        return {
            avgFee,
            feeCV,
            timeUnderfundedPct,
            l1TrackingError: trackingError,
            fee95thPercentile,
            vaultBalanceStd,
            maxDeficit
        };
    }

    mean(arr) {
        return arr.reduce((sum, val) => sum + val, 0) / arr.length;
    }

    standardDeviation(arr) {
        const mean = this.mean(arr);
        const squaredDiffs = arr.map(val => Math.pow(val - mean, 2));
        return Math.sqrt(this.mean(squaredDiffs));
    }
}

// Preset configurations
const PRESETS = {
    'fee-stability': {
        mu: 0.0,
        nu: 0.4,
        H: 144,
        description: 'Minimizes fee volatility for best user experience',
        useCase: 'DeFi protocols requiring predictable transaction costs'
    },
    'vault-management': {
        mu: 0.6,
        nu: 0.5,
        H: 96,
        description: 'Optimizes vault funding and reduces underfunding periods',
        useCase: 'L2s concerned about treasury management'
    },
    'l1-tracking': {
        mu: 0.8,
        nu: 0.3,
        H: 72,
        description: 'Closely follows L1 costs for accurate fee pricing',
        useCase: 'Applications that need fees reflecting true L1 costs'
    },
    'balanced': {
        mu: 0.4,
        nu: 0.3,
        H: 144,
        description: 'Good compromise across all metrics',
        useCase: 'General purpose L2 with diverse application mix'
    },
    'conservative': {
        mu: 0.2,
        nu: 0.2,
        H: 288,
        description: 'Ultra-safe parameters for risk-averse deployments',
        useCase: 'Initial launch or high-stakes financial applications'
    },
    'aggressive': {
        mu: 0.7,
        nu: 0.6,
        H: 48,
        description: 'Fast-adapting parameters for dynamic environments',
        useCase: 'High-frequency trading, MEV-sensitive applications'
    }
};

// Export for use in main application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TaikoFeeSimulator,
        MetricsCalculator,
        GeometricBrownianMotion,
        PRESETS
    };
}