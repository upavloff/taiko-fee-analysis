/**
 * SPECS.md Compliant Simulator
 *
 * Implements the exact mathematical formulations from SPECS.md:
 * - Section 3: Fee Rule (Per-Gas Basefee)
 * - Section 4: Vault Dynamics
 * - Maintains compatibility with existing web interface
 */

/**
 * SPECS.md Section 3: Fee Controller
 */
class SpecsFeeController {
    constructor(mu, nu, H, Q_bar) {
        this.mu = mu;           // L1 weight parameter [0,1]
        this.nu = nu;           // Deficit weight parameter [0,1]
        this.H = H;             // Amortization horizon (batches)
        this.Q_bar = Q_bar;     // Average gas per batch (calibrated at 2.0e4)

        // Optional bounds and rate limiting
        this.F_min = 1e-9;      // Minimum fee (wei per gas)
        this.F_max = 1000e9;    // Maximum fee (wei per gas)
        this.kappa_up = 1.0;    // Max relative up move per batch (disabled)
        this.kappa_down = 1.0;  // Max relative down move per batch (disabled)
    }

    /**
     * Calculate raw per-gas basefee
     * SPECS.md 3.1: f^raw(t) = μ * Ĉ_L1(t)/Q̄ + ν * D(t)/(H*Q̄)
     */
    calculateRawFee(C_hat_L1, D_t) {
        const l1_component = this.mu * (C_hat_L1 / this.Q_bar);
        const deficit_component = this.nu * (D_t / (this.H * this.Q_bar));

        return l1_component + deficit_component;
    }

    /**
     * Apply static min/max bounds
     * SPECS.md 3.2: f^clip(t) = min(F_max, max(F_min, f^raw(t)))
     */
    applyBounds(f_raw) {
        return Math.min(this.F_max, Math.max(this.F_min, f_raw));
    }

    /**
     * Apply rate limiting (optional)
     * SPECS.md 3.3: F_L2(t-1)(1-κ_↓) ≤ F_L2(t) ≤ F_L2(t-1)(1+κ_↑)
     */
    applyRateLimits(f_clip, F_L2_prev) {
        if (F_L2_prev <= 0) return f_clip;

        const min_fee = F_L2_prev * (1 - this.kappa_down);
        const max_fee = F_L2_prev * (1 + this.kappa_up);

        return Math.min(max_fee, Math.max(min_fee, f_clip));
    }

    /**
     * Complete fee calculation pipeline
     */
    calculateFee(C_hat_L1, D_t, F_L2_prev = null) {
        const f_raw = this.calculateRawFee(C_hat_L1, D_t);
        const f_clip = this.applyBounds(f_raw);

        if (F_L2_prev !== null) {
            return this.applyRateLimits(f_clip, F_L2_prev);
        }

        return f_clip;
    }
}

/**
 * SPECS.md Section 4: Vault Dynamics
 */
class SpecsVaultDynamics {
    constructor(target_balance, Q_bar) {
        this.T = target_balance;    // Target vault level (ETH)
        this.Q_bar = Q_bar;         // Average gas per batch
    }

    /**
     * Subsidy rule
     * SPECS.md 4.1: S(t) = min(C_L1(t), V(t))
     */
    calculateSubsidy(C_L1_t, V_t) {
        return Math.min(C_L1_t, V_t);
    }

    /**
     * Revenue calculation
     * R(t) = F_L2(t) * Q̄
     */
    calculateRevenue(F_L2_t) {
        return F_L2_t * this.Q_bar;
    }

    /**
     * Vault balance update
     * SPECS.md 4.2: V(t+1) = V(t) + R(t) - S(t)
     */
    updateVaultBalance(V_t, R_t, S_t) {
        return V_t + R_t - S_t;
    }

    /**
     * Deficit calculation
     * D(t) = T - V(t)
     */
    calculateDeficit(V_t) {
        return this.T - V_t;
    }
}

/**
 * L1 Cost Smoother using EMA
 */
class SpecsL1CostSmoother {
    constructor(alpha = 0.15) {
        this.alpha = alpha;         // EMA smoothing parameter
        this.C_hat_L1 = null;      // Current smoothed estimate
    }

    /**
     * Update smoothed L1 cost estimate
     */
    update(C_L1_actual) {
        if (this.C_hat_L1 === null) {
            this.C_hat_L1 = C_L1_actual;
        } else {
            this.C_hat_L1 = this.alpha * C_L1_actual + (1 - this.alpha) * this.C_hat_L1;
        }
        return this.C_hat_L1;
    }

    /**
     * Get current smoothed estimate
     */
    getSmoothedCost() {
        return this.C_hat_L1 || 0;
    }
}

/**
 * SPECS.md Compliant Simulation Engine
 *
 * Integrates fee controller and vault dynamics for complete simulation
 */
class SpecsSimulationEngine {
    constructor(params) {
        // Core parameters - using SPECS optimal from PARETO analysis
        this.mu = params.mu || 0.7;  // L1 weight (balanced L1 tracking)
        this.nu = params.nu || 0.2;  // Deficit weight (moderate correction)
        this.H = params.H || 72;     // Horizon (balanced responsiveness)
        this.Q_bar = params.Q_bar || 2.0e4;  // From SPECS.md calibration (corrected)
        this.target_balance = params.targetBalance || 1000.0;  // ETH

        // Initialize components
        this.fee_controller = new SpecsFeeController(this.mu, this.nu, this.H, this.Q_bar);
        this.vault_dynamics = new SpecsVaultDynamics(this.target_balance, this.Q_bar);
        this.l1_smoother = new SpecsL1CostSmoother();

        // State tracking
        this.vault_balance = params.vaultInit === 'target' ? this.target_balance :
                           params.vaultInit === 'underfunded-20' ? this.target_balance * 0.8 :
                           params.vaultInit === 'underfunded-50' ? this.target_balance * 0.5 :
                           params.vaultInit === 'overfunded-20' ? this.target_balance * 1.2 :
                           this.target_balance;

        this.previous_fee = 1e9;  // 1 gwei initial fee
        this.time_step = 0;

        // Historical data integration
        this.l1Source = params.l1Source || 'simulated';
        this.historicalPeriod = params.historicalPeriod || 'may2022';
        this.historicalData = null;
        this.historicalIndex = 0;

        // L1 simulation for non-historical mode
        if (this.l1Source === 'simulated') {
            this.l1Model = new EIP1559BaseFeeSimulator(0.0, params.l1Volatility || 0.1, 0.075e9, params.seed || 42);
        }

        // Gas cost calculation (matches Python implementation)
        this.batch_gas_cost = 200000;  // Gas for L1 batch submission
        this.txs_per_batch = params.txsPerBatch || 100;
    }

    /**
     * Calculate L1 cost per transaction
     */
    calculateL1Cost(l1_basefee_wei) {
        // Update smoothed cost estimate
        const gas_per_tx = Math.max(this.batch_gas_cost / this.txs_per_batch, 200);
        const l1_cost_per_tx = (l1_basefee_wei * gas_per_tx) / 1e18;  // Convert to ETH

        return this.l1_smoother.update(l1_cost_per_tx);
    }

    /**
     * Get L1 basefee for current step
     */
    getL1Basefee() {
        if (this.l1Source === 'historical' && this.historicalData && this.historicalData.length > 0) {
            const dataPoint = this.historicalData[this.historicalIndex % this.historicalData.length];
            this.historicalIndex++;
            return dataPoint.basefee_wei;
        } else {
            // Use simulated L1 data
            return this.l1Model.step(1, this.time_step, 0, 0);
        }
    }

    /**
     * Execute single simulation step
     *
     * Returns step results compatible with existing interface
     */
    simulateStep() {
        // Get L1 basefee for this step
        const l1_basefee_wei = this.getL1Basefee();

        // Calculate L1 cost and update smoother
        const C_L1_actual = this.calculateL1Cost(l1_basefee_wei);
        const C_hat_L1 = this.l1_smoother.getSmoothedCost();

        // Calculate current deficit
        const D_t = this.vault_dynamics.calculateDeficit(this.vault_balance);

        // Calculate L2 fee using SPECS.md formula
        const F_L2_t = this.fee_controller.calculateFee(C_hat_L1, D_t, this.previous_fee);

        // Calculate revenue and subsidy
        const R_t = this.vault_dynamics.calculateRevenue(F_L2_t);
        const S_t = this.vault_dynamics.calculateSubsidy(C_L1_actual, this.vault_balance);

        // Update vault balance
        const V_t_plus_1 = this.vault_dynamics.updateVaultBalance(this.vault_balance, R_t, S_t);

        // Prepare step results (compatible with existing interface)
        const step_result = {
            timeStep: this.time_step,
            l1Basefee: l1_basefee_wei / 1e9,  // Convert to gwei for display
            l1Cost: C_L1_actual,
            smoothedL1Cost: C_hat_L1,
            estimatedFee: F_L2_t / 1e9,  // Convert to gwei for display
            feeWeiPerGas: F_L2_t,  // Keep wei for calculations
            revenue: R_t,
            subsidy: S_t,
            vaultBalance: V_t_plus_1,
            vaultDeficit: this.vault_dynamics.calculateDeficit(V_t_plus_1),
            targetBalance: this.target_balance
        };

        // Update state for next step
        this.vault_balance = V_t_plus_1;
        this.previous_fee = F_L2_t;
        this.time_step++;

        return step_result;
    }

    /**
     * Run complete simulation
     */
    async runSimulation(steps = 300) {
        // Load historical data if needed
        await this.loadHistoricalData();

        const results = [];

        for (let i = 0; i < steps; i++) {
            const step_result = this.simulateStep();
            results.push(step_result);
        }

        return this.formatSimulationResults(results);
    }

    /**
     * Load historical data (compatible with existing system)
     */
    async loadHistoricalData() {
        if (this.l1Source !== 'historical') {
            return;
        }

        const period = this.historicalPeriod;

        // Check if data is already cached globally
        if (window.HISTORICAL_DATA && window.HISTORICAL_DATA[period]) {
            this.historicalData = window.HISTORICAL_DATA[period];
            this.historicalIndex = 0;
            return;
        }

        // Map period names to file paths
        const DATA_FILES = {
            may2023: 'data_cache/may_2023_pepe_crisis_data.csv',
            recent: 'data_cache/recent_low_fees_3hours.csv',
            may2022: 'data_cache/luna_crash_true_peak_contiguous.csv',
            june2022: 'data_cache/real_july_2022_spike_data.csv'
        };

        try {
            const response = await fetch(DATA_FILES[period]);
            const csvText = await response.text();
            this.historicalData = this.parseCSV(csvText);

            // Cache globally
            if (!window.HISTORICAL_DATA) window.HISTORICAL_DATA = {};
            window.HISTORICAL_DATA[period] = this.historicalData;

            this.historicalIndex = 0;
        } catch (error) {
            console.warn(`Failed to load historical data for ${period}:`, error);
            // Fall back to simulated data
            this.l1Source = 'simulated';
        }
    }

    /**
     * Parse CSV data (compatible with existing format)
     */
    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        const data = [];

        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            if (values.length >= 3) {
                data.push({
                    timestamp: values[0],
                    basefee_wei: parseFloat(values[1]),
                    basefee_gwei: parseFloat(values[2])
                });
            }
        }

        return data;
    }

    /**
     * Format simulation results for compatibility with existing interface
     */
    formatSimulationResults(results) {
        return {
            timestamp: results.map(r => r.timeStep),
            l1_basefee: results.map(r => r.l1Basefee),
            estimated_fee: results.map(r => r.estimatedFee),
            vault_balance: results.map(r => r.vaultBalance),
            vault_deficit: results.map(r => r.vaultDeficit),
            fee_collected: results.map(r => r.revenue),
            l1_cost_paid: results.map(r => r.subsidy),

            // Additional SPECS-specific data
            smoothed_l1_cost: results.map(r => r.smoothedL1Cost),
            raw_l1_cost: results.map(r => r.l1Cost)
        };
    }

    /**
     * Calculate metrics compatible with existing system
     */
    calculateMetrics(simulationData) {
        if (!simulationData || simulationData.length === 0) {
            return this.getEmptyMetrics();
        }

        const fees = simulationData.map(d => d.estimatedFee);
        const vaultBalances = simulationData.map(d => d.vaultBalance);

        // Basic metrics
        const avgFee = this.mean(fees);
        const feeStd = this.standardDeviation(fees);
        const feeCV = feeStd / avgFee;

        // Vault metrics
        const avgVault = this.mean(vaultBalances);
        const underfundedSteps = vaultBalances.filter(balance =>
            (this.target_balance - balance) > this.target_balance * 0.0001
        ).length;
        const timeUnderfundedPct = (underfundedSteps / simulationData.length) * 100;

        return {
            avgFee,
            feeStd,
            feeCV,
            avgVault,
            timeUnderfundedPct,
            maxDeficit: Math.max(...simulationData.map(d => Math.max(0, this.target_balance - d.vaultBalance))),

            // SPECS-specific metrics
            costRecoveryRatio: this.calculateCostRecoveryRatio(simulationData),
            fee99thPercentile: this.percentile(fees, 99)
        };
    }

    /**
     * Calculate cost recovery ratio (SPECS.md constraint)
     */
    calculateCostRecoveryRatio(simulationData) {
        const totalRevenue = simulationData.reduce((sum, d) => sum + d.revenue, 0);
        const totalL1Cost = simulationData.reduce((sum, d) => sum + d.l1Cost, 0);

        return totalL1Cost > 0 ? totalRevenue / totalL1Cost : 1.0;
    }

    // Utility methods
    mean(arr) {
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    standardDeviation(arr) {
        const mean = this.mean(arr);
        const variance = arr.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / arr.length;
        return Math.sqrt(variance);
    }

    percentile(arr, p) {
        const sorted = [...arr].sort((a, b) => a - b);
        const index = Math.ceil((p / 100) * sorted.length) - 1;
        return sorted[Math.max(0, index)];
    }

    getEmptyMetrics() {
        return {
            avgFee: 0, feeStd: 0, feeCV: 0, avgVault: 0,
            timeUnderfundedPct: 0, maxDeficit: 0,
            costRecoveryRatio: 1, fee99thPercentile: 0
        };
    }
}

/**
 * Alpha-Data Based Fee Controller
 *
 * JavaScript implementation of the alpha-data model that replaces
 * the crude Q̄ constant with empirically-measured α_data.
 */
class AlphaFeeController {
    constructor(alpha_data, nu, H, l2_gas_per_batch = 6.9e5, proof_gas_per_batch = 180000) {
        this.alpha_data = alpha_data;           // L1 DA gas per L2 gas (empirically measured)
        this.nu = nu;                          // Deficit weight parameter [0,1]
        this.H = H;                            // Amortization horizon (batches)
        this.l2_gas_per_batch = l2_gas_per_batch;  // L2 gas consumption per batch
        this.proof_gas_per_batch = proof_gas_per_batch;  // L1 proof gas per batch

        // Optional bounds and rate limiting
        this.F_min = 0;                        // Minimum fee (wei per gas)
        this.F_max = 1000e9;                   // Maximum fee (wei per gas)
        this.kappa_up = 1.0;                   // Max relative up move per batch (disabled)
        this.kappa_down = 1.0;                 // Max relative down move per batch (disabled)

        // State tracking
        this.previous_fee = null;
    }

    /**
     * Calculate raw per-gas basefee using alpha-data model
     * New Formula: f^raw(t) = α_data * L1_basefee + ν * D(t)/(H * L2_gas) + proof_component
     */
    calculateRawFee(l1_basefee_wei_per_gas, deficit_wei) {
        // DA Component: Direct L1 cost tracking
        const da_component = this.alpha_data * l1_basefee_wei_per_gas;

        // Deficit Component: Amortize deficit over horizon
        const deficit_component = this.nu * deficit_wei / (this.H * this.l2_gas_per_batch);

        // Proof Component: Amortize proof costs over L2 gas in batch
        const proof_component = this.proof_gas_per_batch * l1_basefee_wei_per_gas / this.l2_gas_per_batch;

        return da_component + deficit_component + proof_component;
    }

    /**
     * Apply static min/max bounds
     */
    applyBounds(f_raw) {
        return Math.min(this.F_max, Math.max(this.F_min, f_raw));
    }

    /**
     * Apply rate limiting
     */
    applyRateLimits(f_clip, F_L2_prev) {
        if (F_L2_prev <= 0) return f_clip;

        const min_fee = F_L2_prev * (1 - this.kappa_down);
        const max_fee = F_L2_prev * (1 + this.kappa_up);

        return Math.min(max_fee, Math.max(min_fee, f_clip));
    }

    /**
     * Complete fee calculation pipeline
     */
    calculateFee(l1_basefee_wei_per_gas, deficit_wei, F_L2_prev = null) {
        const f_raw = this.calculateRawFee(l1_basefee_wei_per_gas, deficit_wei);
        const f_clip = this.applyBounds(f_raw);

        let final_fee = f_clip;
        if (F_L2_prev !== null) {
            final_fee = this.applyRateLimits(f_clip, F_L2_prev);
        }

        this.previous_fee = final_fee;
        return final_fee;
    }

    /**
     * Calculate expected L1 cost for a batch
     */
    calculateExpectedL1Cost(l1_basefee_wei_per_gas) {
        // DA cost
        const da_cost = this.alpha_data * l1_basefee_wei_per_gas * this.l2_gas_per_batch;

        // Proof cost
        const proof_cost = this.proof_gas_per_batch * l1_basefee_wei_per_gas;

        return da_cost + proof_cost;
    }

    /**
     * Calculate batch revenue
     */
    calculateRevenue(basefee_per_gas) {
        return basefee_per_gas * this.l2_gas_per_batch;
    }

    /**
     * Analyze cost recovery
     */
    analyzeCostRecovery(l1_basefee_wei_per_gas, deficit_wei = 0) {
        const l2_fee = this.calculateFee(l1_basefee_wei_per_gas, deficit_wei);
        const l1_cost = this.calculateExpectedL1Cost(l1_basefee_wei_per_gas);
        const revenue = this.calculateRevenue(l2_fee);

        const cost_recovery = l1_cost > 0 ? revenue / l1_cost : Infinity;

        return {
            l1_basefee_gwei: l1_basefee_wei_per_gas / 1e9,
            l2_fee_gwei: l2_fee / 1e9,
            l1_cost_wei: l1_cost,
            l2_revenue_wei: revenue,
            cost_recovery_ratio: cost_recovery,
            net_result_wei: revenue - l1_cost,
            alpha_data_used: this.alpha_data
        };
    }

    /**
     * Reset state
     */
    resetState() {
        this.previous_fee = null;
    }
}

/**
 * Alpha-Data Based Simulation Engine
 *
 * Replaces Q̄-based fee mechanism with empirically-measured α_data model.
 */
class AlphaSimulationEngine {
    constructor(params) {
        // Core alpha-data parameters
        this.alpha_data = params.alpha_data || 0.22;  // Default based on expected range
        this.nu = params.nu || 0.2;                   // Deficit weight
        this.H = params.H || 72;                      // Horizon
        this.target_balance = params.targetBalance || 1000.0;  // ETH
        this.l2_gas_per_batch = params.l2_gas_per_batch || 6.9e5;
        this.proof_gas_per_batch = params.proof_gas_per_batch || 180000;

        // Initialize alpha-based components
        this.fee_controller = new AlphaFeeController(
            this.alpha_data, this.nu, this.H,
            this.l2_gas_per_batch, this.proof_gas_per_batch
        );

        this.vault_dynamics = new SpecsVaultDynamics(this.target_balance, this.l2_gas_per_batch);

        // State tracking
        this.vault_balance = params.vaultInit === 'target' ? this.target_balance :
                           params.vaultInit === 'underfunded-20' ? this.target_balance * 0.8 :
                           params.vaultInit === 'underfunded-50' ? this.target_balance * 0.5 :
                           params.vaultInit === 'overfunded-20' ? this.target_balance * 1.2 :
                           this.target_balance;

        this.time_step = 0;

        // Historical data integration (same as SPECS)
        this.l1Source = params.l1Source || 'simulated';
        this.historicalPeriod = params.historicalPeriod || 'may2022';
        this.historicalData = null;
        this.historicalIndex = 0;

        // L1 simulation for non-historical mode
        if (this.l1Source === 'simulated') {
            this.l1Model = new EIP1559BaseFeeSimulator(0.0, params.l1Volatility || 0.1, 0.075e9, params.seed || 42);
        }
    }

    /**
     * Get L1 basefee for current step
     * Same as SPECS implementation but directly returns basefee
     */
    getL1Basefee() {
        if (this.l1Source === 'historical' && this.historicalData && this.historicalData.length > 0) {
            const dataPoint = this.historicalData[this.historicalIndex % this.historicalData.length];
            this.historicalIndex++;
            return dataPoint.basefee_wei;
        } else {
            // Use simulated L1 data
            return this.l1Model.step(1, this.time_step, 0, 0);
        }
    }

    /**
     * Execute single simulation step with alpha-data model
     */
    simulateStep() {
        // Get L1 basefee for this step
        const l1_basefee_wei = this.getL1Basefee();

        // Calculate current deficit
        const D_t = this.vault_dynamics.calculateDeficit(this.vault_balance);
        const D_t_wei = D_t * 1e18;  // Convert to wei

        // Calculate L2 fee using alpha-data model (direct L1 basefee input)
        const F_L2_t = this.fee_controller.calculateFee(l1_basefee_wei, D_t_wei, this.previous_fee);

        // Calculate actual L1 costs using alpha model
        const actual_l1_cost_wei = this.fee_controller.calculateExpectedL1Cost(l1_basefee_wei);
        const actual_l1_cost_eth = actual_l1_cost_wei / 1e18;

        // Calculate revenue
        const R_t_wei = this.fee_controller.calculateRevenue(F_L2_t);
        const R_t_eth = R_t_wei / 1e18;

        // Calculate subsidy
        const S_t = this.vault_dynamics.calculateSubsidy(actual_l1_cost_eth, this.vault_balance);

        // Update vault balance
        const V_t_plus_1 = this.vault_dynamics.updateVaultBalance(this.vault_balance, R_t_eth, S_t);

        // Cost recovery analysis
        const cost_recovery = actual_l1_cost_eth > 0 ? R_t_eth / actual_l1_cost_eth : Infinity;

        // Prepare step results (compatible with existing interface)
        const step_result = {
            timeStep: this.time_step,
            l1Basefee: l1_basefee_wei / 1e9,  // Convert to gwei for display
            l1Cost: actual_l1_cost_eth,
            estimatedFee: F_L2_t / 1e9,  // Convert to gwei for display
            feeWeiPerGas: F_L2_t,  // Keep wei for calculations
            revenue: R_t_eth,
            subsidy: S_t,
            vaultBalance: V_t_plus_1,
            vaultDeficit: this.vault_dynamics.calculateDeficit(V_t_plus_1),
            targetBalance: this.target_balance,

            // Alpha-specific data
            alpha_data_used: this.alpha_data,
            cost_recovery_ratio: cost_recovery,
            l1_basefee_input: l1_basefee_wei,
            actual_l1_cost_wei: actual_l1_cost_wei,
            net_result_eth: R_t_eth - actual_l1_cost_eth
        };

        // Update state for next step
        this.vault_balance = V_t_plus_1;
        this.previous_fee = F_L2_t;
        this.time_step++;

        return step_result;
    }

    /**
     * Run complete simulation with alpha-data model
     */
    async runSimulation(steps = 300) {
        // Load historical data if needed
        await this.loadHistoricalData();

        const results = [];

        for (let i = 0; i < steps; i++) {
            const step_result = this.simulateStep();
            results.push(step_result);
        }

        return this.formatSimulationResults(results);
    }

    /**
     * Load historical data (same as SPECS implementation)
     */
    async loadHistoricalData() {
        if (this.l1Source !== 'historical') {
            return;
        }

        const period = this.historicalPeriod;

        // Check if data is already cached globally
        if (window.HISTORICAL_DATA && window.HISTORICAL_DATA[period]) {
            this.historicalData = window.HISTORICAL_DATA[period];
            this.historicalIndex = 0;
            return;
        }

        // Map period names to file paths
        const DATA_FILES = {
            may2023: 'data_cache/may_2023_pepe_crisis_data.csv',
            recent: 'data_cache/recent_low_fees_3hours.csv',
            may2022: 'data_cache/luna_crash_true_peak_contiguous.csv',
            june2022: 'data_cache/real_july_2022_spike_data.csv'
        };

        try {
            const response = await fetch(DATA_FILES[period]);
            const csvText = await response.text();
            this.historicalData = this.parseCSV(csvText);

            // Cache globally
            if (!window.HISTORICAL_DATA) window.HISTORICAL_DATA = {};
            window.HISTORICAL_DATA[period] = this.historicalData;

            this.historicalIndex = 0;
        } catch (error) {
            console.warn(`Failed to load historical data for ${period}:`, error);
            // Fall back to simulated data
            this.l1Source = 'simulated';
        }
    }

    /**
     * Parse CSV data (same as SPECS implementation)
     */
    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        const data = [];

        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            if (values.length >= 3) {
                data.push({
                    timestamp: values[0],
                    basefee_wei: parseFloat(values[1]),
                    basefee_gwei: parseFloat(values[2])
                });
            }
        }

        return data;
    }

    /**
     * Format simulation results for compatibility with existing interface
     */
    formatSimulationResults(results) {
        return {
            timestamp: results.map(r => r.timeStep),
            l1_basefee: results.map(r => r.l1Basefee),
            estimated_fee: results.map(r => r.estimatedFee),
            vault_balance: results.map(r => r.vaultBalance),
            vault_deficit: results.map(r => r.vaultDeficit),
            fee_collected: results.map(r => r.revenue),
            l1_cost_paid: results.map(r => r.subsidy),

            // Alpha-specific data
            cost_recovery_ratios: results.map(r => r.cost_recovery_ratio),
            alpha_data_used: results.map(r => r.alpha_data_used),
            net_results: results.map(r => r.net_result_eth)
        };
    }

    /**
     * Calculate metrics with alpha-specific analysis
     */
    calculateMetrics(simulationData) {
        if (!simulationData || simulationData.length === 0) {
            return this.getEmptyMetrics();
        }

        const fees = simulationData.map(d => d.estimatedFee);
        const vaultBalances = simulationData.map(d => d.vaultBalance);
        const costRecoveries = simulationData.map(d => d.cost_recovery_ratio || 1.0);

        // Basic metrics
        const avgFee = this.mean(fees);
        const feeStd = this.standardDeviation(fees);
        const feeCV = feeStd / avgFee;

        // Vault metrics
        const avgVault = this.mean(vaultBalances);
        const underfundedSteps = vaultBalances.filter(balance =>
            (this.target_balance - balance) > this.target_balance * 0.0001
        ).length;
        const timeUnderfundedPct = (underfundedSteps / simulationData.length) * 100;

        // Alpha-specific metrics
        const avgCostRecovery = this.mean(costRecoveries.filter(cr => cr !== Infinity && !isNaN(cr)));
        const realisticFeesAchieved = avgFee >= 5.0;  // Target: 5-15 gwei
        const feeWithinTargetRange = avgFee >= 5.0 && avgFee <= 15.0;
        const costRecoveryHealthy = avgCostRecovery >= 0.8 && avgCostRecovery <= 1.2;

        return {
            avgFee,
            feeStd,
            feeCV,
            avgVault,
            timeUnderfundedPct,
            maxDeficit: Math.max(...simulationData.map(d => Math.max(0, this.target_balance - d.vaultBalance))),

            // Alpha-specific metrics
            avgCostRecovery,
            realisticFeesAchieved,
            feeWithinTargetRange,
            costRecoveryHealthy,
            fee99thPercentile: this.percentile(fees, 99),
            alpha_data_value: this.alpha_data,

            // Model comparison
            alpha_advantage: avgFee > 1.0 ? "Realistic fees achieved" : "Fees still too low"
        };
    }

    /**
     * Compare alpha model with Q̄ model
     */
    compareWithQBarModel(specsEngine, steps = 100) {
        // Run alpha simulation
        const alphaResults = [];
        this.resetState();

        for (let i = 0; i < steps; i++) {
            alphaResults.push(this.simulateStep());
        }

        // Run SPECS Q̄ simulation
        specsEngine.time_step = 0;
        specsEngine.vault_balance = specsEngine.target_balance;
        specsEngine.previous_fee = 1e9;

        const specsResults = [];
        for (let i = 0; i < steps; i++) {
            specsResults.push(specsEngine.simulateStep());
        }

        // Calculate comparison metrics
        const alphaMetrics = this.calculateMetrics(alphaResults);
        const specsMetrics = specsEngine.calculateMetrics(specsResults);

        return {
            alpha_model: {
                avg_fee_gwei: alphaMetrics.avgFee,
                avg_cost_recovery: alphaMetrics.avgCostRecovery,
                realistic_fees: alphaMetrics.realisticFeesAchieved,
                alpha_data: this.alpha_data
            },
            qbar_model: {
                avg_fee_gwei: specsMetrics.avgFee,
                avg_cost_recovery: specsMetrics.costRecoveryRatio || 0,
                realistic_fees: specsMetrics.avgFee >= 5.0,
                q_bar: specsEngine.Q_bar
            },
            improvements: {
                fee_improvement_factor: alphaMetrics.avgFee / specsMetrics.avgFee,
                cost_recovery_improvement: alphaMetrics.avgCostRecovery - (specsMetrics.costRecoveryRatio || 0),
                alpha_advantage: "Direct L1 tracking vs broken Q̄ constant"
            }
        };
    }

    /**
     * Reset simulation state
     */
    resetState() {
        this.fee_controller.resetState();
        this.time_step = 0;
        this.vault_balance = this.target_balance;
        this.previous_fee = null;
        this.historicalIndex = 0;
    }

    // Utility methods (same as SPECS implementation)
    mean(arr) {
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    standardDeviation(arr) {
        const mean = this.mean(arr);
        const variance = arr.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / arr.length;
        return Math.sqrt(variance);
    }

    percentile(arr, p) {
        const sorted = [...arr].sort((a, b) => a - b);
        const index = Math.ceil((p / 100) * sorted.length) - 1;
        return sorted[Math.max(0, index)];
    }

    getEmptyMetrics() {
        return {
            avgFee: 0, feeStd: 0, feeCV: 0, avgVault: 0,
            timeUnderfundedPct: 0, maxDeficit: 0,
            avgCostRecovery: 1, fee99thPercentile: 0,
            realisticFeesAchieved: false, feeWithinTargetRange: false,
            costRecoveryHealthy: false, alpha_data_value: this.alpha_data
        };
    }
}

// Export for use in web interface
if (typeof window !== 'undefined') {
    window.SpecsSimulationEngine = SpecsSimulationEngine;
    window.SpecsFeeController = SpecsFeeController;
    window.SpecsVaultDynamics = SpecsVaultDynamics;
    window.SpecsL1CostSmoother = SpecsL1CostSmoother;

    // Export alpha-data classes
    window.AlphaFeeController = AlphaFeeController;
    window.AlphaSimulationEngine = AlphaSimulationEngine;
}