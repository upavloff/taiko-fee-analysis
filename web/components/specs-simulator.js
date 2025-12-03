/**
 * SPECS.md Compliant Simulator
 *
 * Implements the exact mathematical formulations from SPECS.md:
 * - Section 3: Fee Rule (Per-Gas Basefee)
 * - Section 4: Vault Dynamics
 * - Maintains compatibility with existing web interface
 */


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
 * L1 Basefee Smoother using EMA
 *
 * Implements B̂_L1(t) = (1 - λ_B) × B̂_L1(t-1) + λ_B × B_L1(t)
 *
 * Purpose: Avoid overreacting to single-block noise on L1
 * Units: ETH per L1 gas (same as input)
 */
class L1BasefeeSmoother {
    constructor(lambda_B = 0.15) {
        this.lambda_B = lambda_B;        // EMA smoothing parameter ∈ (0, 1]
        this.b_l1_smooth = null;         // B̂_L1(t): smoothed L1 basefee in ETH per L1 gas
    }

    /**
     * Update smoothed L1 basefee estimate
     *
     * @param {number} b_l1_raw - B_L1(t): Raw L1 basefee in ETH per L1 gas (wei per gas / 1e18)
     * @returns {number} B̂_L1(t): Smoothed L1 basefee in ETH per L1 gas
     */
    update(b_l1_raw) {
        if (this.b_l1_smooth === null) {
            // Initialize with first observation
            this.b_l1_smooth = b_l1_raw;
        } else {
            // EMA update: B̂_L1(t) = (1 - λ_B) × B̂_L1(t-1) + λ_B × B_L1(t)
            this.b_l1_smooth = (1 - this.lambda_B) * this.b_l1_smooth + this.lambda_B * b_l1_raw;
        }
        return this.b_l1_smooth;
    }

    /**
     * Get current smoothed L1 basefee
     * @returns {number} B̂_L1(t) in ETH per L1 gas
     */
    getSmoothedBasefee() {
        return this.b_l1_smooth || 0;
    }

    /**
     * Reset smoother state
     */
    reset() {
        this.b_l1_smooth = null;
    }
}

/**
 * Research Specification Simulation Engine
 *
 * Integrates research fee controller with proper L1 basefee smoothing
 */
class ResearchSimulationEngine {
    constructor(params) {
        // Core parameters - research specification
        this.mu = params.mu || 0.7;              // L1 pass-through weight [0,1]
        this.alpha_data = params.alpha_data || 0.18;  // DA efficiency [ETH per L1 gas]
        this.nu = params.nu || 0.2;              // Deficit weight [0,1]
        this.H = params.H || 72;                 // Amortization horizon [batches]
        this.Q_bar = params.Q_bar || 20000;      // Average L2 gas per batch
        this.target_balance = params.targetBalance || 1000.0;  // ETH

        // Initialize components
        this.fee_controller = new ResearchFeeController({
            mu: this.mu,
            alpha_data: this.alpha_data,
            nu: this.nu,
            H: this.H,
            Q_bar: this.Q_bar
        });
        this.vault_dynamics = new SpecsVaultDynamics(this.target_balance, this.Q_bar);
        this.l1_smoother = new L1BasefeeSmoother(0.15);

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
     * Update L1 basefee smoother
     *
     * @param {number} l1_basefee_wei - Raw L1 basefee in wei per gas
     * @returns {number} Smoothed L1 basefee in ETH per gas
     */
    updateL1BasefeeSmoother(l1_basefee_wei) {
        // Convert to ETH per gas and update smoother
        const b_l1_raw = l1_basefee_wei / 1e18;  // Convert wei per gas → ETH per gas
        return this.l1_smoother.update(b_l1_raw);
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
     * Execute single simulation step using research specification
     *
     * Returns step results compatible with existing interface
     */
    simulateStep() {
        // Get L1 basefee for this step
        const l1_basefee_wei = this.getL1Basefee();

        // Update L1 basefee smoother and get smoothed value
        const b_l1_smooth = this.updateL1BasefeeSmoother(l1_basefee_wei);

        // Calculate current vault deficit
        const D_t = this.vault_dynamics.calculateDeficit(this.vault_balance);

        // Calculate L2 fee using research specification formula
        // Convert ETH values to wei for the fee controller interface
        const l1_basefee_wei = b_l1_smooth * 1e18;  // Convert ETH per gas → wei per gas
        const deficit_wei = D_t * 1e18;             // Convert ETH → wei
        const previous_fee_wei = this.previous_fee ? this.previous_fee * 1e18 : null;

        const F_L2_wei = this.fee_controller.calculateFee(l1_basefee_wei, deficit_wei, previous_fee_wei);
        const F_L2_t = F_L2_wei / 1e18;  // Convert back to ETH per gas for internal calculations

        // Calculate L1 cost for subsidy calculation
        const gas_per_tx = Math.max(this.batch_gas_cost / this.txs_per_batch, 200);
        const C_L1_actual = (l1_basefee_wei * gas_per_tx) / 1e18;  // ETH per transaction

        // Calculate revenue and subsidy
        const R_t = this.vault_dynamics.calculateRevenue(F_L2_t);
        const S_t = this.vault_dynamics.calculateSubsidy(C_L1_actual, this.vault_balance);

        // Update vault balance
        const V_t_plus_1 = this.vault_dynamics.updateVaultBalance(this.vault_balance, R_t, S_t);

        // Prepare step results (compatible with existing interface)
        const step_result = {
            timeStep: this.time_step,
            l1Basefee: l1_basefee_wei / 1e9,  // Convert to gwei for display
            l1BasefeeSmooth: b_l1_smooth * 1e9,  // Convert to gwei for display
            l1Cost: C_L1_actual,
            estimatedFee: F_L2_wei / 1e9,  // Convert to gwei for display
            feeWeiPerGas: F_L2_wei,  // Keep wei per gas for compatibility
            revenue: R_t,
            subsidy: S_t,
            vaultBalance: V_t_plus_1,
            vaultDeficit: this.vault_dynamics.calculateDeficit(V_t_plus_1),
            targetBalance: this.target_balance,
            alpha_data_used: this.alpha_data,  // For compatibility with alpha integration tests
            mu_used: this.mu,                   // L1 pass-through weight
            nu_used: this.nu                    // Deficit weight
        };

        // Update state for next step
        this.vault_balance = V_t_plus_1;
        this.previous_fee = F_L2_t;  // Keep in ETH per gas for internal state
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

        return results;  // Return array directly for compatibility
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
            may2023: 'data/may_2023_pepe_crisis_data.csv',
            recent: 'data/recent_low_fees_3hours.csv',
            may2022: 'data/luna_crash_true_peak_contiguous.csv',
            june2022: 'data/real_july_2022_spike_data.csv'
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
 * Research-Compliant L2 Basefee Controller
 *
 * Implements the exact specification:
 * f_raw(t) = μ × α_data × B̂_L1(t) + ν × D(t) / (H × Q̄) + proof_component
 *
 * All parameters and units are explicitly documented per the research specification.
 */
class ResearchFeeController {
    constructor(params) {
        // === Core Parameters (All Dimensionally Checked) ===

        // μ ∈ [0,1]: L1 pass-through weight (dimensionless)
        // μ → 0: L1 shocks absorbed by vault
        // μ → 1: L1 shocks passed through to users
        this.mu = params.mu;

        // α_data: L1 DA gas per L2 gas (L1 gas / L2 gas)
        // Empirically estimated as: E[L1_DA_gas_per_batch / total_L2_gas_per_batch]
        // Typical range: 0.18 (blob) to 0.26 (calldata)
        this.alpha_data = params.alpha_data;

        // ν ∈ [0,1]: deficit correction weight (dimensionless)
        // Controls how aggressively we repair vault deficits
        this.nu = params.nu;

        // H > 0: amortization horizon (number of batches)
        // Interpretation: "Over ~H batches, correct fraction ν of current deficit"
        this.H = params.H;

        // Q̄: average L2 gas per batch (L2 gas)
        // Acts as conversion factor between "per-batch ETH" and "per-gas ETH"
        this.Q_bar = params.Q_bar || 690000;

        // === Optional Components ===

        // Proof gas cost per batch on L1 (L1 gas)
        this.proof_gas_per_batch = params.proof_gas_per_batch || 180000;

        // λ_B ∈ (0,1]: EMA smoothing parameter for L1 basefee
        this.lambda_B = params.lambda_B || 0.15;

        // === Bounds and Rate Limiting ===
        this.F_min = params.F_min || 0;           // Min fee (ETH per L2 gas)
        this.F_max = params.F_max || 1000e-9;     // Max fee (ETH per L2 gas)
        this.kappa_up = params.kappa_up || 1.0;   // Max relative up move per batch
        this.kappa_down = params.kappa_down || 1.0; // Max relative down move per batch

        // === State Tracking ===
        this.previous_fee = null;                  // For rate limiting

        console.log(`ResearchFeeController initialized:`);
        console.log(`  μ=${this.mu} (L1 pass-through), α_data=${this.alpha_data} (DA efficiency)`);
        console.log(`  ν=${this.nu} (deficit weight), H=${this.H} batches, Q̄=${this.Q_bar} gas`);
        console.log(`  λ_B=${this.lambda_B} (L1 smoothing), proof_gas=${this.proof_gas_per_batch}`);
    }

    /**
     * Calculate raw L2 basefee per L2 gas (Research Specification)
     *
     * f_raw(t) = μ × α_data × B̂_L1(t) + ν × D(t)/(H × Q̄) + proof_component
     *
     * @param {number} b_l1_smooth - B̂_L1(t): Smoothed L1 basefee in ETH per L1 gas
     * @param {number} deficit_eth - D(t): Vault deficit in ETH (T - V(t))
     * @returns {number} f_raw(t): Raw L2 basefee in ETH per L2 gas
     */
    calculateRawFee(b_l1_smooth, deficit_eth) {
        // === DIMENSIONAL ANALYSIS ===
        // b_l1_smooth: ETH / L1_gas
        // deficit_eth: ETH
        // All components must result in: ETH / L2_gas

        // === TERM 1: L1 DA Pass-through ===
        // μ × α_data × B̂_L1(t)
        // Units: [dimensionless] × [L1_gas/L2_gas] × [ETH/L1_gas] = ETH/L2_gas ✓
        //
        // Economic meaning: Instantaneous pass-through of fraction μ of current L1 DA cost per L2 gas
        // - μ → 0: L1 shocks absorbed by vault
        // - μ → 1: L1 shocks passed through to users
        const l1_da_component = this.mu * this.alpha_data * b_l1_smooth;

        // === TERM 2: Vault Deficit Amortization ===
        // ν × D(t) / (H × Q̄)
        // Units: [dimensionless] × [ETH] / ([batches] × [L2_gas/batch]) = ETH/L2_gas ✓
        //
        // Economic meaning: Spread deficit correction over H batches
        // - D(t) in ETH: current deficit (positive) or surplus (negative)
        // - D(t)/H: ETH per batch to correct deficit over H batches
        // - D(t)/(H×Q̄): ETH per L2 gas for deficit correction
        // - ν: fraction of deficit correction applied per batch
        //
        // Expected deficit correction per batch: ν × D(t) / H
        // Over H batches: cumulative correction ≈ ν × D(t)
        const deficit_component = this.nu * deficit_eth / (this.H * this.Q_bar);

        // === TERM 3: L1 Proof Cost Amortization ===
        // proof_gas × B̂_L1(t) / Q̄
        // Units: [L1_gas] × [ETH/L1_gas] / [L2_gas] = ETH/L2_gas ✓
        //
        // Economic meaning: Amortize L1 proof submission costs across L2 gas in batch
        const proof_component = (this.proof_gas_per_batch * b_l1_smooth) / this.Q_bar;

        // === RESULT ===
        const f_raw = l1_da_component + deficit_component + proof_component;

        // Debug logging for dimensional verification
        if (Math.random() < 0.01) { // Log occasionally
            console.log(`Fee calculation (ETH/L2_gas):`);
            console.log(`  L1 DA: ${l1_da_component.toExponential(3)} (μ=${this.mu}, α=${this.alpha_data}, B̂_L1=${b_l1_smooth.toExponential(3)})`);
            console.log(`  Deficit: ${deficit_component.toExponential(3)} (ν=${this.nu}, D=${deficit_eth.toFixed(1)}, H=${this.H})`);
            console.log(`  Proof: ${proof_component.toExponential(3)} (proof_gas=${this.proof_gas_per_batch})`);
            console.log(`  Total: ${f_raw.toExponential(3)} ETH/L2_gas`);
        }

        return f_raw;
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
        // Convert inputs to ETH units for internal calculations
        const b_l1_smooth_eth = l1_basefee_wei_per_gas / 1e18;  // wei per gas → ETH per gas
        const deficit_eth = deficit_wei / 1e18;                 // wei → ETH
        const F_L2_prev_eth = F_L2_prev ? F_L2_prev / 1e18 : null; // wei per gas → ETH per gas

        const f_raw = this.calculateRawFee(b_l1_smooth_eth, deficit_eth);
        const f_clip = this.applyBounds(f_raw);

        let final_fee_eth = f_clip;
        if (F_L2_prev_eth !== null) {
            final_fee_eth = this.applyRateLimits(f_clip, F_L2_prev_eth);
        }

        this.previous_fee = final_fee_eth;
        return final_fee_eth * 1e18;  // Convert back to wei per gas for output
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
        // Core parameters including both μ and α_data
        this.mu = params.mu || 0.7;                   // L1 weight (restored)
        this.alpha_data = params.alpha_data || 0.18;  // Default based on production calibration
        this.nu = params.nu || 0.2;                   // Deficit weight
        this.H = params.H || 72;                      // Horizon
        this.target_balance = params.targetBalance || 1000.0;  // ETH
        this.l2_gas_per_batch = params.l2_gas_per_batch || 6.9e5;
        this.proof_gas_per_batch = params.proof_gas_per_batch || 180000;

        // Initialize alpha-based components with both μ and α_data
        this.fee_controller = new AlphaFeeController(
            this.mu, this.alpha_data, this.nu, this.H,
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

        // Calculate gas per tx for metrics compatibility
        this.batch_gas_cost = 200000;  // Gas for L1 batch submission
        this.txs_per_batch = params.txsPerBatch || 100;
        this.gasPerTx = Math.max(this.batch_gas_cost / this.txs_per_batch, 200);

        console.log(`AlphaSimulationEngine initialized: μ=${this.mu}, α_data=${this.alpha_data}, ν=${this.nu}, H=${this.H}, txs_per_batch=${this.txs_per_batch}, gasPerTx=${this.gasPerTx}`);
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

            // Format each step for UI compatibility (array of objects format)
            const uiCompatibleStep = {
                timeStep: step_result.timeStep,
                l1ElapsedSeconds: step_result.timeStep * 12,  // Approximate L1 timing
                l2ElapsedSeconds: step_result.timeStep * 2,   // Taiko L2 timing
                timestampLabel: null,
                l1Basefee: step_result.l1Basefee,
                l1TrendBasefee: step_result.l1Basefee, // Alpha uses direct basefee, no smoothing
                vaultBalance: step_result.vaultBalance,
                vaultDeficit: step_result.vaultDeficit,
                estimatedFee: step_result.estimatedFee,
                txVolume: this.txs_per_batch || 100, // Transactions per batch
                feesCollected: step_result.revenue,
                actualL1Cost: step_result.l1Cost,
                isL1BatchStep: (step_result.timeStep % 6 === 0),  // Every 6th step

                // Alpha-specific data
                alpha_data_used: step_result.alpha_data_used,
                cost_recovery_ratio: step_result.cost_recovery_ratio,
                net_result_eth: step_result.net_result_eth
            };

            results.push(uiCompatibleStep);
        }

        // Return array format expected by UI (not object format)
        return results;
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
            may2023: 'data/may_2023_pepe_crisis_data.csv',
            recent: 'data/recent_low_fees_3hours.csv',
            may2022: 'data/luna_crash_true_peak_contiguous.csv',
            june2022: 'data/real_july_2022_spike_data.csv'
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
    window.L1BasefeeSmoother = L1BasefeeSmoother;

    // Export research specification classes
    window.ResearchFeeController = ResearchFeeController;
    window.ResearchSimulationEngine = ResearchSimulationEngine;

    // Export with alpha-data naming for compatibility
    window.AlphaSimulationEngine = ResearchSimulationEngine;
}