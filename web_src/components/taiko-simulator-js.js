/**
 * JavaScript Implementation of the Taiko Fee Mechanism Simulator
 * Based on the Python implementation in src/core/improved_simulator.py
 */

/**
 * Fee Vault implementation
 */
class FeeVault {
    constructor(initialBalance, target) {
        this.balance = initialBalance;
        this.target = target;
    }

    get deficit() {
        return Math.max(0, this.target - this.balance);
    }

    get surplus() {
        return Math.max(0, this.balance - this.target);
    }

    collectFees(amount) {
        this.balance += amount;
    }

    payL1Costs(amount) {
        this.balance -= amount;
    }
}

/**
 * Improved Fee Vault with enhanced functionality
 */
class ImprovedFeeVault extends FeeVault {
    constructor(initialBalance, target) {
        super(initialBalance, target);
        this.history = {
            balance: [initialBalance],
            deficit: [this.deficit],
            surplus: [this.surplus]
        };
    }

    collectFees(amount) {
        super.collectFees(amount);
        this.recordState();
    }

    payL1Costs(amount) {
        super.payL1Costs(amount);
        this.recordState();
    }

    recordState() {
        this.history.balance.push(this.balance);
        this.history.deficit.push(this.deficit);
        this.history.surplus.push(this.surplus);
    }
}

/**
 * Simulation Parameters
 */
class SimulationParams {
    constructor(options = {}) {
        // Fee mechanism parameters
        this.mu = options.mu || 0.0;           // L1 weight
        this.nu = options.nu || 0.1;           // Deficit weight
        this.H = options.H || 36;              // Prediction horizon (steps)

        // System parameters
        this.txs_per_batch = options.txs_per_batch || 100;
        // Implement documented formula: max(200,000/Expected Tx Volume, 2,000)
        const expectedTxVolume = this.txs_per_batch || 100;
        this.gas_per_tx = Math.max(200000 / expectedTxVolume, 2000);
        this.gas_per_batch = this.gas_per_tx * this.txs_per_batch;
        this.target_balance = options.target_balance || 1.0;  // ETH
        this.base_l1_cost = options.base_l1_cost || 0.001;    // ETH per tx

        // Transaction demand parameters
        this.demand_elasticity = options.demand_elasticity || 0.5;
        this.base_tx_demand = options.base_tx_demand || 100;  // txs per step
        this.max_tx_demand = options.max_tx_demand || 1000;   // txs per step

        // Timing parameters (critical for realistic vault economics)
        this.time_step_seconds = options.time_step_seconds || 2;  // 2s per Taiko L2 block
        this.batch_interval_steps = options.batch_interval_steps || 6;  // L1 batch every 12s
    }
}

/**
 * Improved Simulation Parameters
 */
class ImprovedSimulationParams extends SimulationParams {
    constructor(options = {}) {
        super(options);
        this.initial_vault_balance = options.initial_vault_balance || this.target_balance;
        this.total_steps = options.total_steps || 1800;  // Default 1 hour at 2s per step
    }
}

/**
 * Research Taiko Fee Simulator (for multi-scenario evaluation)
 */
class ResearchTaikoFeeSimulator {
    constructor(params, historicalData) {
        this.params = params;
        this.historicalData = historicalData;  // Array of {basefee_gwei, timestamp} objects
        this.vault = new FeeVault(params.target_balance, params.target_balance);

        this.resetState();
    }

    resetState() {
        this.vault = new FeeVault(this.params.target_balance, this.params.target_balance);
        this.timeStep = 0;

        // History tracking
        this.history = {
            time_step: [],
            l1_basefee: [],
            estimated_l1_cost: [],
            estimated_fee: [],
            transaction_volume: [],
            vault_balance: [],
            vault_deficit: [],
            fee_collected: [],
            l1_cost_paid: [],
            batch_occurred: [],
            deficit_ratio: []
        };

        // L1 cost estimation (EWMA)
        this.l1CostEwma = this.params.base_l1_cost;
        this.ewmaAlpha = 0.1;
    }

    /**
     * Estimate L1 cost per transaction based on current basefee
     */
    estimateL1CostPerTx(l1BasefeeGwei) {
        const l1BasefeeWei = l1BasefeeGwei * 1e9;
        const costPerTx = l1BasefeeWei * this.params.gas_per_tx / 1e18;  // Convert to ETH

        // Update EWMA estimate
        this.l1CostEwma = (1 - this.ewmaAlpha) * this.l1CostEwma + this.ewmaAlpha * costPerTx;
        return this.l1CostEwma;
    }

    /**
     * Calculate estimated fee using the Taiko mechanism
     */
    calculateEstimatedFee(l1CostEstimate) {
        // Direct L1 cost component
        const l1Component = this.params.mu * l1CostEstimate;

        // Deficit correction component with H-step prediction
        const deficit = this.vault.deficit;
        const target = this.params.target_balance;

        // Deficit component following documented formula: ν×D/H
        let deficitComponent = 0;
        if (deficit > 0) {
            deficitComponent = this.params.nu * deficit / this.params.H;
        }

        return l1Component + deficitComponent;
    }

    /**
     * Calculate transaction volume based on fee with elasticity
     */
    calculateTransactionVolume(estimatedFee) {
        // Simple demand curve with elasticity
        const baseFee = 0.001;  // Base reference fee (ETH)
        const feeRatio = estimatedFee / baseFee;
        const demandMultiplier = Math.pow(feeRatio, -this.params.demand_elasticity);

        const volume = this.params.base_tx_demand * demandMultiplier;
        return Math.min(volume, this.params.max_tx_demand);
    }

    /**
     * Calculate L1 batch cost when batch is submitted
     */
    calculateL1BatchCost(l1BasefeeGwei) {
        const l1BasefeeWei = l1BasefeeGwei * 1e9;
        return l1BasefeeWei * this.params.gas_per_batch / 1e18;  // Convert to ETH
    }

    /**
     * Execute one simulation time step
     */
    step(l1BasefeeGwei) {
        // Estimate L1 cost per transaction
        const l1CostEstimate = this.estimateL1CostPerTx(l1BasefeeGwei);

        // Calculate estimated fee
        const estimatedFee = this.calculateEstimatedFee(l1CostEstimate);

        // Calculate transaction volume (with fee elasticity)
        const txVolume = this.calculateTransactionVolume(estimatedFee);

        // ALWAYS collect fees (every 2s Taiko L2 block)
        const totalFees = estimatedFee * txVolume;
        this.vault.collectFees(totalFees);

        // ONLY pay L1 costs when batch is submitted (every 12s = every 6 Taiko steps)
        const batchOccurred = (this.timeStep % this.params.batch_interval_steps === 0);
        let l1CostPaid = 0;
        if (batchOccurred) {
            l1CostPaid = this.calculateL1BatchCost(l1BasefeeGwei);
            this.vault.payL1Costs(l1CostPaid);
        }

        // Record history
        this.history.time_step.push(this.timeStep);
        this.history.l1_basefee.push(l1BasefeeGwei);
        this.history.estimated_l1_cost.push(l1CostEstimate);
        this.history.estimated_fee.push(estimatedFee);
        this.history.transaction_volume.push(txVolume);
        this.history.vault_balance.push(this.vault.balance);
        this.history.vault_deficit.push(this.vault.deficit);
        this.history.fee_collected.push(totalFees);
        this.history.l1_cost_paid.push(l1CostPaid);
        this.history.batch_occurred.push(batchOccurred);
        this.history.deficit_ratio.push(this.vault.deficit / this.params.target_balance);

        this.timeStep++;
    }

    /**
     * Run complete simulation on historical data
     */
    runSimulation(maxSteps = null) {
        this.resetState();

        const dataLength = this.historicalData.length;
        const totalSteps = maxSteps || Math.min(dataLength, this.params.total_steps || dataLength);

        for (let i = 0; i < totalSteps && i < dataLength; i++) {
            const dataPoint = this.historicalData[i];
            this.step(dataPoint.basefee_gwei);
        }

        return this.history;
    }

    /**
     * Get simulation results summary
     */
    getResultsSummary() {
        if (this.history.time_step.length === 0) {
            throw new Error('No simulation data available. Run simulation first.');
        }

        const fees = this.history.estimated_fee;
        const deficits = this.history.vault_deficit;
        const balances = this.history.vault_balance;

        return {
            steps: this.history.time_step.length,
            avg_fee_eth: fees.reduce((a, b) => a + b, 0) / fees.length,
            max_fee_eth: Math.max(...fees),
            min_fee_eth: Math.min(...fees),
            avg_deficit: deficits.reduce((a, b) => a + b, 0) / deficits.length,
            max_deficit: Math.max(...deficits),
            final_balance: balances[balances.length - 1],
            deficit_duration: deficits.filter(d => d > 0).length,
            insolvency_events: balances.filter(b => b < 0).length
        };
    }
}

/**
 * Improved Research Taiko Fee Simulator with enhanced features
 */
class ImprovedResearchTaikoFeeSimulator extends ResearchTaikoFeeSimulator {
    constructor(params, historicalData) {
        super(params, historicalData);
        this.improvedParams = params;

        // Enhanced state tracking
        this.l1CostHistory = [];
        this.l1BasefeeHistory = [];
        this.trendBasefee = null;
        this.previousEstimatedFee = null;

        this.resetState();
    }

    resetState() {
        super.resetState();

        // Use improved vault
        this.vault = new ImprovedFeeVault(
            this.improvedParams.initial_vault_balance,
            this.improvedParams.target_balance
        );

        this.l1CostHistory = [];
        this.l1BasefeeHistory = [];
        this.trendBasefee = null;
        this.previousEstimatedFee = null;
    }

    /**
     * Enhanced L1 cost estimation with trend analysis
     */
    estimateL1CostPerTx(l1BasefeeGwei) {
        // Track L1 basefee history for trend calculation
        this.l1BasefeeHistory.push(l1BasefeeGwei);

        // Keep only recent history for trend calculation
        const trendWindow = 20;
        if (this.l1BasefeeHistory.length > trendWindow) {
            this.l1BasefeeHistory = this.l1BasefeeHistory.slice(-trendWindow);
        }

        // Calculate trend basefee (EWMA)
        if (this.l1BasefeeHistory.length >= 3) {
            if (this.trendBasefee === null) {
                // Initialize with mean of first few points
                this.trendBasefee = this.l1BasefeeHistory.reduce((a, b) => a + b, 0) / this.l1BasefeeHistory.length;
            } else {
                // Update EWMA trend with alpha = 0.15
                const alphaTrend = 0.15;
                this.trendBasefee = alphaTrend * l1BasefeeGwei + (1 - alphaTrend) * this.trendBasefee;
            }
        } else {
            this.trendBasefee = l1BasefeeGwei;
        }

        // Use trend basefee for cost calculation instead of spot price
        const rawCost = this.trendBasefee * 1e9 * this.params.gas_per_tx / 1e18;

        // Add to cost history for outlier rejection
        this.l1CostHistory.push(rawCost);
        if (this.l1CostHistory.length > 50) {
            this.l1CostHistory = this.l1CostHistory.slice(-50);
        }

        // Simple outlier rejection (remove top 10% outliers)
        const sortedCosts = [...this.l1CostHistory].sort((a, b) => a - b);
        const percentile90 = sortedCosts[Math.floor(sortedCosts.length * 0.9)];
        const filteredCost = Math.min(rawCost, percentile90);

        // Update EWMA estimate with filtered cost
        this.l1CostEwma = (1 - this.ewmaAlpha) * this.l1CostEwma + this.ewmaAlpha * filteredCost;
        return this.l1CostEwma;
    }
}

// Export for global use
window.ResearchTaikoFeeSimulator = ResearchTaikoFeeSimulator;
window.ImprovedResearchTaikoFeeSimulator = ImprovedResearchTaikoFeeSimulator;
window.SimulationParams = SimulationParams;
window.ImprovedSimulationParams = ImprovedSimulationParams;