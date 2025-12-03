// Taiko Fee Mechanism Simulator - JavaScript Implementation
// Ported from Python analysis for web interface

class EIP1559BaseFeeSimulator {
    constructor(mu, sigma, initialValue = 10e9, seed = 42) {
        this.mu = mu;           // Drift (trend)
        this.sigma = sigma;     // Volatility
        this.currentBaseFee = initialValue;  // Current basefee in wei
        this.seed = seed;       // Random seed for reproducible results
        this.rng = this.seedableRandom(seed);

        // EIP-1559 constants (updated 2024 values)
        this.TARGET_GAS = 18_000_000;  // 18M gas target per block (current)
        this.MAX_GAS = 36_000_000;     // 36M gas max per block (current)
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

        // Allow natural basefee dynamics for realistic sub-gwei periods
        // Removed artificial 1 gwei floor to match documented low-fee datasets (0.055-0.092 gwei)
        this.currentBaseFee = Math.max(newBaseFee, 1e6); // 0.001 gwei minimum (technical floor)

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
    may2023: null,
    recent: null,
    may2022: null,
    june2022: null
};

// Data file mapping
const DATA_FILES = {
    may2023: 'data/may_2023_pepe_crisis_data.csv',
    recent: 'data/recent_low_fees_3hours.csv',
    may2022: 'data/luna_crash_true_peak_contiguous.csv',
    june2022: 'data/real_july_2022_spike_data.csv'
    // Note: All periods are post-EIP-1559 with valid basefee data (EIP-1559 activated Aug 5, 2021)
    // may2023 contains PEPE memecoin crisis data showing extreme volatility (60-184 gwei)
    // june2022 contains real July 1, 2022 data showing actual market volatility (7-88 gwei)
};

class TaikoFeeSimulator {
    constructor(params) {

        this.mu = params.mu;                    // L1 cost weight
        this.nu = params.nu;                    // Deficit weight
        this.H = params.H;                      // Time horizon
        this.targetBalance = params.targetBalance || 100;
        this.feeElasticity = params.feeElasticity || 0.2;
        this.minFee = params.minFee || 1e-8;

        // Alpha-data parameters
        this.alpha_data = params.alpha_data || 0.22;  // L1 DA gas per L2 gas ratio (empirically measured)

        // Initialize Alpha-data components
        this.specsEngine = new AlphaSimulationEngine({
            alpha_data: this.alpha_data,
            nu: this.nu,
            H: this.H,
            targetBalance: this.targetBalance,
            vaultInit: params.vaultInit,
            l1Source: params.l1Source || 'simulated',
            historicalPeriod: params.historicalPeriod || 'may2022',
            l1Volatility: params.l1Volatility || 0.1,
            seed: params.seed || 42,
            txsPerBatch: params.txsPerBatch || 100
        });

        // Legacy parameters for backward compatibility
        this.guaranteedRecovery = params.guaranteedRecovery || false;
        this.minDeficitRate = params.minDeficitRate || 1e-6;

        // Debug constructor parameters
        console.log(`TaikoFeeSimulator initialized with Alpha-data engine: alpha_data=${this.alpha_data}, nu=${this.nu}, H=${this.H}`);

        // Vault initialization (delegated to SPECS engine)
        this.vaultBalance = this.specsEngine.vault_balance;

        // L1 data source configuration
        this.l1Source = params.l1Source || 'simulated';
        this.historicalPeriod = params.historicalPeriod || 'may2022';
        this.historicalData = null;
        this.historicalIndex = 0;


        // L1 model parameters (for simulated mode) - Use realistic 2025 basefee levels
        const realisticBasefee = 0.075e9; // 0.075 gwei in wei (realistic 2025 levels)
        this.l1Model = new EIP1559BaseFeeSimulator(0.0, params.l1Volatility, realisticBasefee, params.seed || 42);
        console.log(`L1 Simulator initialized with realistic basefee: ${realisticBasefee/1e9} gwei (was 10 gwei)`);
        this.spikeDelay = params.spikeDelaySteps || params.spikeDelay || 60;  // Use calculated steps or fallback
        this.spikeHeight = params.spikeHeight || 0.3;  // Spike intensity

        // Transaction parameters (aligned with Python implementation)
        this.txsPerBatch = params.txsPerBatch || 100;  // Transactions per L1 batch (matches Python default)
        this.batchGas = 200000;        // Gas cost for L1 batch submission

        // L1 basefee trend tracking for cost estimation
        this.l1BasefeeHistory = [];
        this.trendWindow = 20;  // 20-step window for trend calculation
        this.trendBasefee = null;  // Trend-based basefee estimate

        // Calculate gas per tx based on expected volume (economies of scale)
        this.updateGasPerTx();
    }

    updateGasPerTx() {
        // CORRECTED: max(200,000 / Expected Tx Volume, 200) - fixed from bug analysis
        // This implements economies of scale with a 200 gas minimum for overhead
        const baseGasPerTx = this.batchGas / this.txsPerBatch;
        this.gasPerTx = Math.max(baseGasPerTx, 200);

        console.log(`gasPerTx = max(${this.batchGas} / ${this.txsPerBatch}, 200) = max(${baseGasPerTx}, 200) = ${this.gasPerTx} gas`);
        console.log(`L1 cost per tx = basefee * ${this.gasPerTx} / 1e18`);
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

    updateL1BasefeeTrend(currentBasefeeWei) {
        // Add current basefee to history
        this.l1BasefeeHistory.push(currentBasefeeWei);

        // Maintain window size
        if (this.l1BasefeeHistory.length > this.trendWindow) {
            this.l1BasefeeHistory.shift();
        }

        // Calculate trend-based estimate (exponentially weighted moving average)
        if (this.l1BasefeeHistory.length >= 3) {
            // Use EWMA with alpha = 0.15 for smoothing
            const alpha = 0.15;
            if (this.trendBasefee === null) {
                // Initialize with simple average of first few points
                this.trendBasefee = this.l1BasefeeHistory.reduce((a, b) => a + b, 0) / this.l1BasefeeHistory.length;
            } else {
                this.trendBasefee = alpha * currentBasefeeWei + (1 - alpha) * this.trendBasefee;
            }
        } else {
            // Not enough history, use spot price
            this.trendBasefee = currentBasefeeWei;
        }
    }

    calculateL1Cost(l1BasefeeWei) {
        // Update trend tracking
        this.updateL1BasefeeTrend(l1BasefeeWei);

        // Use trend basefee for cost estimation instead of spot price
        const basefeeForCost = this.trendBasefee || l1BasefeeWei;

        // Calculate L1 cost per L2 transaction based on amortized batch costs
        return (basefeeForCost * this.gasPerTx) / 1e18;
    }

    calculateFee(l1BasefeeWei, vaultDeficit) {
        const l1Cost = this.calculateL1Cost(l1BasefeeWei);
        const l1Component = this.mu * l1Cost;

        let deficitComponent = this.nu * (vaultDeficit / this.H);

        // Apply guaranteed recovery logic if enabled
        if (this.guaranteedRecovery && vaultDeficit > 0) {
            // Ensure minimum deficit correction rate to prevent asymptotic stalling
            const standardCorrection = this.nu * (vaultDeficit / this.H);
            const minimumCorrection = this.minDeficitRate;
            const wasStandardUsed = standardCorrection >= minimumCorrection;
            deficitComponent = Math.max(standardCorrection, minimumCorrection);

            // Debug logging (log 2% of the time to monitor effectiveness)
            if (Math.random() < 0.02) {
                console.log(`Guaranteed Recovery: deficit=${vaultDeficit.toFixed(6)}, standard=${standardCorrection.toExponential(3)}, minimum=${minimumCorrection.toExponential(3)}, used=${deficitComponent.toExponential(3)}, source=${wasStandardUsed ? 'standard' : 'minimum'}`);
            }
        }

        return Math.max(l1Component + deficitComponent, this.minFee);
    }

    calculateDemand(fee, baseDemand = this.txsPerBatch) {
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
            console.log(`Loading historical data for period: ${period}, file: ${DATA_FILES[period]}`);
            const response = await fetch(DATA_FILES[period]);
            if (!response.ok) {
                throw new Error(`Failed to load ${period} data: ${response.statusText}`);
            }

            const csvText = await response.text();
            const lines = csvText.trim().split('\n');
            const headers = lines[0].split(',').map(h => h.trim());

            // Support any column order (e.g., Luna crash dataset uses block_number first)
            const colIndex = {
                timestamp: headers.indexOf('timestamp'),
                basefee_wei: headers.indexOf('basefee_wei'),
                basefee_gwei: headers.indexOf('basefee_gwei'),
                block_number: headers.indexOf('block_number')
            };

            const data = lines.slice(1).map(line => {
                const values = line.split(',').map(v => v.trim());

                const getVal = (key, fallbackIdx) => {
                    const idx = colIndex[key];
                    return idx >= 0 ? values[idx] : values[fallbackIdx];
                };

                const timestamp = getVal('timestamp', 0);
                const basefeeWei = parseFloat(getVal('basefee_wei', 1));
                const basefeeGwei = parseFloat(getVal('basefee_gwei', 2));
                const rawBlock = getVal('block_number', 3);
                const blockNumber = (typeof rawBlock === 'string' && rawBlock.startsWith('0x'))
                    ? parseInt(rawBlock, 16)
                    : parseInt(rawBlock, 10);

                return {
                    timestamp,
                    basefee_wei: basefeeWei,
                    basefee_gwei: basefeeGwei,
                    block_number: blockNumber
                };
            });

            HISTORICAL_DATA[period] = data;
            this.historicalData = data;
            this.historicalIndex = 0;

            console.log(`Successfully loaded ${data.length} data points for ${period}. Basefee range: ${data[0]?.basefee_gwei?.toFixed(3)} - ${data[data.length-1]?.basefee_gwei?.toFixed(3)} gwei`);

        } catch (error) {
            console.error('Failed to load historical data:', error);
            // Fallback to simulated data
            this.l1Source = 'simulated';
        }
    }

    getNextL1Basefee() {
        if (this.l1Source === 'historical' && this.historicalData && this.historicalData.length > 0) {
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
        // Delegate to SPECS engine for SPECS.md compliant simulation
        const specsResults = await this.specsEngine.runSimulation(steps);

        // Convert SPECS results to legacy format for UI compatibility
        const results = [];

        for (let t = 0; t < specsResults.timestamp.length; t++) {
            results.push({
                timeStep: specsResults.timestamp[t],
                l1ElapsedSeconds: t * 12,  // Approximate L1 timing
                l2ElapsedSeconds: t * 2,   // Taiko L2 timing
                timestampLabel: null,
                l1Basefee: specsResults.l1_basefee[t],
                l1TrendBasefee: specsResults.smoothed_l1_cost ? specsResults.smoothed_l1_cost[t] * 1e18 / this.specsEngine.batch_gas_cost * this.specsEngine.txs_per_batch : specsResults.l1_basefee[t],
                vaultBalance: specsResults.vault_balance[t],
                vaultDeficit: specsResults.vault_deficit[t],
                estimatedFee: specsResults.estimated_fee[t],
                txVolume: this.specsEngine.txs_per_batch,  // Fixed for now
                feesCollected: specsResults.fee_collected[t],
                actualL1Cost: specsResults.l1_cost_paid[t],
                isL1BatchStep: (t % 6 === 0)  // Every 6th step
            });
        }

        // Update internal state to match SPECS engine
        this.vaultBalance = this.specsEngine.vault_balance;

        return results;
    }
}

class MetricsCalculator {
    constructor(targetBalance, gasPerTx) {
        this.targetBalance = targetBalance;
        this.gasPerTx = gasPerTx;
    }

    calculateMetrics(simulationData) {
        // Convert per-transaction fees to per-gas fees for display
        const fees = simulationData.map(d => (d.estimatedFee * 1e9) / this.gasPerTx); // Convert to per-gas gwei
        const vaultBalances = simulationData.map(d => d.vaultBalance);
        const l1Basefees = simulationData.map(d => d.l1Basefee);

        // Average fee (now in per-gas gwei)
        const avgFee = this.mean(fees);

        // Fee coefficient of variation
        const feeStd = this.standardDeviation(fees);
        const feeCV = feeStd / avgFee;

        // Time underfunded percentage (with 0.01% threshold for meaningful underfunding)
        // Only deficits > 0.01% of target balance are considered meaningfully underfunded
        const significantDeficitThreshold = this.targetBalance * 0.0001; // 0.01% of target
        const underfundedSteps = vaultBalances.filter(balance =>
            (this.targetBalance - balance) > significantDeficitThreshold
        ).length;
        const timeUnderfundedPct = (underfundedSteps / simulationData.length) * 100;

        // L1 tracking error (simplified)
        const l1Costs = l1Basefees.map(basefee => (basefee * this.gasPerTx) / 1e18);
        const normalizedFees = fees.map(fee => fee / this.mean(fees));
        const normalizedL1Costs = l1Costs.map(cost => cost / this.mean(l1Costs));
        const trackingError = this.standardDeviation(normalizedFees.map((fee, i) => fee - normalizedL1Costs[i]));

        // 95th percentile fee (now in per-gas gwei)
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

// Revised optimization framework presets for optimal fee mechanism performance
const PRESETS = {
    'deployment-ready': {
        mu: 0.7,
        alpha_data: 0.18,
        nu: 0.2,
        H: 72,
        description: '🚀 DEPLOYMENT READY: Production-calibrated parameters',
        objective: 'Validated for immediate production deployment with realistic fee outputs',
        constraints: 'Calibrated μ=0.7, α_data=0.18 produces optimal L1 cost tracking',
        tradeoffs: 'Balanced approach: healthy cost recovery with stable vault economics',
        riskProfile: 'Production validated - proper balance of L1 weight and DA efficiency',
        useCase: 'PRODUCTION: μ=0.7, α=0.18, ν=0.2, H=72. Balanced L1 tracking with alpha efficiency.'
    },
    'conservative': {
        mu: 0.5,
        alpha_data: 0.16,
        nu: 0.2,
        H: 72,
        description: '🛡️ CONSERVATIVE: Lower fees with robust operation',
        objective: 'Prioritizes user affordability while maintaining healthy economics',
        constraints: 'Lower μ (0.5) and α_data (0.16) for reduced fee sensitivity',
        tradeoffs: 'Blob-mode biased pricing with reduced L1 pass-through',
        riskProfile: 'Conservative approach - lower L1 tracking, maximum affordability',
        useCase: 'AFFORDABLE: μ=0.5, α=0.16, ν=0.2, H=72. Conservative configuration.'
    },
    'responsive': {
        mu: 1.0,
        alpha_data: 0.20,
        nu: 0.3,
        H: 48,
        description: '⚡ RESPONSIVE: High market responsiveness',
        objective: 'Maximum responsiveness to L1 market conditions and vault state',
        constraints: 'Full L1 tracking (μ=1.0) with higher α_data (0.20) and aggressive deficit correction',
        tradeoffs: 'Full L1 cost pass-through with faster response to market changes',
        riskProfile: 'Market-responsive - maximum L1 sensitivity with quick adaptation',
        useCase: 'MARKET-REACTIVE: μ=1.0, α=0.20, ν=0.3, H=48. Full L1 tracking configuration.'
    }
};

// Export for use in main application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TaikoFeeSimulator,
        MetricsCalculator,
        EIP1559BaseFeeSimulator,
        PRESETS
    };
}

// Also export to global window object for browser usage
if (typeof window !== 'undefined') {
    window.TaikoFeeSimulator = TaikoFeeSimulator;
    window.MetricsCalculator = MetricsCalculator;
    window.EIP1559BaseFeeSimulator = EIP1559BaseFeeSimulator;
    window.PRESETS = PRESETS;
}
