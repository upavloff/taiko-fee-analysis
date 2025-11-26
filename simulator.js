// Taiko Fee Mechanism Simulator - JavaScript Implementation
// Ported from Python analysis for web interface

class EIP1559BaseFeeSimulator {
    constructor(mu, sigma, initialValue = 10e9, seed = 42) {
        this.mu = mu;           // Drift (trend)
        this.sigma = sigma;     // Volatility
        this.currentBaseFee = initialValue;  // Current basefee in wei
        this.seed = seed;       // Random seed for reproducible results
        this.rng = this.seedableRandom(seed);

        // EIP-1559 constants
        this.TARGET_GAS = 15_000_000;  // 15M gas target per block
        this.MAX_GAS = 30_000_000;     // 30M gas max per block
        this.BASE_FEE_MAX_CHANGE = 1.125;  // Max 12.5% change per block
    }

    step(dt = 1, timeStep = 0, spikeDelay = 0, spikeHeight = 0.3) {
        // Generate demand pressure based on volatility and spikes
        const randomShock = this.generateNormal(0, 1);
        let demandPressure = this.sigma * randomShock;

        // Add controlled spike after delay
        if (timeStep >= spikeDelay && timeStep < spikeDelay + 30) {
            // Create realistic spike wave that follows EIP-1559 limits
            const spikeProgress = (timeStep - spikeDelay) / 30;
            const spikeIntensity = spikeHeight * Math.sin(spikeProgress * Math.PI) * 2;
            demandPressure += spikeIntensity;
        }

        // Convert demand pressure to block gas usage
        // demandPressure > 0 = high demand, < 0 = low demand
        const normalizedDemand = Math.tanh(demandPressure); // Bound between -1 and 1
        const gasUsed = this.TARGET_GAS + (normalizedDemand * (this.MAX_GAS - this.TARGET_GAS) * 0.5);

        // EIP-1559 basefee adjustment formula
        const gasUsedDelta = gasUsed - this.TARGET_GAS;
        const baseFeePerGasDelta = Math.floor(this.currentBaseFee * gasUsedDelta / this.TARGET_GAS / 8);

        // Apply the change with EIP-1559 limits
        let newBaseFee = this.currentBaseFee + baseFeePerGasDelta;

        // Enforce max change rate (12.5% per block)
        const maxIncrease = this.currentBaseFee * this.BASE_FEE_MAX_CHANGE;
        const maxDecrease = this.currentBaseFee / this.BASE_FEE_MAX_CHANGE;

        newBaseFee = Math.min(newBaseFee, maxIncrease);
        newBaseFee = Math.max(newBaseFee, maxDecrease);

        // Ensure basefee doesn't go below 1 gwei
        this.currentBaseFee = Math.max(newBaseFee, 1e9);

        return this.currentBaseFee;
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

// Historical data cache
const HISTORICAL_DATA = {
    recent: null,
    may2022: null,
    june2022: null
};

// Data file mapping
const DATA_FILES = {
    recent: 'data_cache/recent_low_fees_3hours.csv',
    may2022: 'data_cache/may_crash_basefee_data.csv',
    june2022: 'data_cache/high_volatility_basefee_data.csv'
    // Note: All periods are post-EIP-1559 with valid basefee data (EIP-1559 activated Aug 5, 2021)
};

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

        // L1 data source configuration
        this.l1Source = params.l1Source || 'simulated';
        this.historicalPeriod = params.historicalPeriod || 'may2022';
        this.historicalData = null;
        this.historicalIndex = 0;


        // L1 model parameters (for simulated mode)
        this.l1Model = new EIP1559BaseFeeSimulator(0.0, params.l1Volatility, 10e9, params.seed || 42);
        this.spikeDelay = params.spikeDelay || 60;  // Delay spike by 60 steps (10 minutes)
        this.spikeHeight = params.spikeHeight || 0.3;  // Spike intensity

        // Transaction parameters
        this.baseTxVolume = params.baseTxVolume || 10;  // Expected transaction volume per step
        this.batchGas = 200000;        // Gas cost for L1 batch submission

        // Calculate gas per tx based on expected volume (economies of scale)
        this.updateGasPerTx();
    }

    updateGasPerTx() {
        // Economies of scale: more transactions = lower gas cost per tx
        // Floor at 2000 gas to account for minimum batch overhead
        this.gasPerTx = Math.max(this.batchGas / Math.max(this.baseTxVolume, 1), 2000);
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

    async loadHistoricalData() {
        if (this.l1Source !== 'historical') {
            return;
        }

        const period = this.historicalPeriod;

        // Check if already cached
        if (HISTORICAL_DATA[period]) {
            this.historicalData = HISTORICAL_DATA[period];
            this.historicalIndex = 0;
            return;
        }

        try {
            const response = await fetch(DATA_FILES[period]);
            if (!response.ok) {
                throw new Error(`Failed to load ${period} data: ${response.statusText}`);
            }

            const csvText = await response.text();
            const lines = csvText.trim().split('\n');
            const headers = lines[0].split(',');

            const data = lines.slice(1).map(line => {
                const values = line.split(',');
                return {
                    timestamp: values[0],
                    basefee_wei: parseFloat(values[1]),
                    basefee_gwei: parseFloat(values[2]),
                    block_number: parseInt(values[3])
                };
            });

            HISTORICAL_DATA[period] = data;
            this.historicalData = data;
            this.historicalIndex = 0;

        } catch (error) {
            console.error('Failed to load historical data:', error);
            // Fallback to simulated data
            this.l1Source = 'simulated';
        }
    }

    getNextL1Basefee() {
        if (this.l1Source === 'historical' && this.historicalData) {
            // Use modulo to cycle through historical data
            const dataPoint = this.historicalData[this.historicalIndex % this.historicalData.length];
            const basefeeGwei = dataPoint.basefee_wei / 1e9;
            this.historicalIndex++;
            return dataPoint.basefee_wei;
        } else {
            const simulatedBasefee = this.l1Model.step(1, this.historicalIndex, this.spikeDelay, this.spikeHeight);
            const simulatedGwei = simulatedBasefee / 1e9;
            this.historicalIndex++;
            return simulatedBasefee;
        }
    }

    async runSimulation(steps = 300) {
        // Load historical data if needed
        await this.loadHistoricalData();

        // Reset historical index to start from beginning of selected period
        this.historicalIndex = 0;

        // For historical data, use the full length of available data instead of fixed steps
        const actualSteps = (this.l1Source === 'historical' && this.historicalData)
            ? this.historicalData.length
            : steps;

        const results = [];

        for (let t = 0; t < actualSteps; t++) {
            // Get current L1 basefee
            const l1Basefee = this.getNextL1Basefee();

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