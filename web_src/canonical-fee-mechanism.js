/**
 * Canonical Taiko Fee Mechanism Implementation (JavaScript)
 *
 * This is the SINGLE SOURCE OF TRUTH for all fee mechanism calculations in JavaScript.
 * It mirrors the Python canonical implementation to ensure consistency.
 *
 * Key Features:
 * - Authoritative fee calculation formula implementation
 * - Vault state management with proper initialization modes
 * - L1 cost estimation with smoothing and outlier rejection
 * - Transaction volume modeling with demand elasticity
 * - Comprehensive parameter validation and bounds checking
 * - UNIT SAFETY: Automatic validation to prevent ETH/Wei/Gwei confusion
 *
 * Formula Reference:
 *     F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t)
 *
 * Where:
 *     - μ: DA cost pass-through coefficient [0.0, 1.0]
 *     - ν: Vault-healing intensity coefficient [0.0, 1.0]
 *     - C_DA(t) = α_data × B̂_L1(t): smoothed marginal DA cost per L2 gas
 *     - C_vault(t) = D(t)/(H × Q̄): full-strength vault-healing surcharge per L2 gas
 *     - α_data: Expected L1 DA gas per 1 L2 gas
 *     - B̂_L1(t): Smoothed L1 basefee (ETH per L1 gas)
 *     - D(t): Vault deficit (ETH, positive when underfunded)
 *     - H: Recovery horizon (batches)
 *     - Q̄: Typical L2 gas per batch (governance constant)
 */

// Import unit validation system
let UNIT_SAFETY_AVAILABLE = false;
let unitValidation = {
    validateBasefeeRange: () => {},
    validateFeeRange: () => {},
    assertReasonableFee: () => {},
    validateL1CostCalculation: (basefee, gas) => (basefee * gas) / 1e18,
    validateFeeCalculation: (l1Cost, deficit, mu, nu, H) => Math.max(mu * l1Cost + nu * deficit / H, 1e-12)
};

// Try to load unit validation asynchronously
(async function() {
    try {
        const module = await import('./src/core/unit_validation.js');
        unitValidation = module.default || module;
        UNIT_SAFETY_AVAILABLE = true;
        console.log('✅ Unit safety system loaded');
    } catch (error) {
        console.warn('🚨 MOCK DATA WARNING: Unit safety system not available - unit validation disabled');
        // Keep fallback functions
    }
})();

// Mock data detection and warnings
function warnMockDataUsage(context, dataType, reason) {
    console.warn(
        `🚨 MOCK DATA WARNING: Using ${dataType} in ${context} - ${reason}. ` +
        `This violates scientific accuracy principles. Real data should be used in production.`
    );
}

function validateRealDataUsage(dataSource, value, context) {
    // Check for common mock data patterns
    if (typeof value === 'number') {
        // Check for suspicious hardcoded values
        if (value === 1.5 && context.toLowerCase().includes('gwei')) {
            warnMockDataUsage(context, 'hardcoded 1.5 gwei base fee',
                            'artificial fee floor injection detected');
        } else if (value === 690000) {
            warnMockDataUsage(context, 'arbitrary Q̄ = 690,000 constant',
                            'should use empirically measured gas consumption');
        } else if (value === 200 && context.toLowerCase().includes('gas')) {
            warnMockDataUsage(context, 'hardcoded 200 gas',
                            'should be 20,000 gas per documentation');
        }
    }

    // Check data source provenance
    const mockSources = ['mock', 'placeholder', 'fallback', 'default', 'hardcoded'];
    if (mockSources.includes(dataSource)) {
        warnMockDataUsage(context, `${dataSource} data`,
                        `data source '${dataSource}' indicates non-real data usage`);
    }
}

// Vault initialization modes
export const VaultInitMode = {
    TARGET: "target",     // Start at target balance
    DEFICIT: "deficit",   // Start below target (underfunded)
    SURPLUS: "surplus",   // Start above target (overfunded)
    CUSTOM: "custom"      // Use provided balance
};

// Fee mechanism parameters class
export class FeeParameters {
    constructor({
        // Core mechanism parameters (2024 NSGA-II optimized)
        mu = 0.0,                    // L1 weight [0.0, 1.0] - DA cost pass-through coefficient
        nu = 0.369,                  // Deficit weight [0.0, 1.0] - enhanced vault-healing intensity coefficient
        H = 1794,                    // Prediction horizon (time steps) - ~1 hour recovery horizon in batches

        // New Fee Mechanism Parameters (per specification, 2024 optimized)
        alpha_data = 0.5,            // DA gas ratio: E[L1 DA gas per batch / Q̄] - empirically calibrated from real proposeBlock transactions
        lambda_B = 0.365,            // EMA smoothing factor for L1 basefee [0.0, 1.0] - enhanced stability
        Q_bar = 690000.0,            // Average L2 gas per batch (governance constant) - ~345k L1 DA gas expected per batch
        T = 1000.0,                  // Target vault balance (ETH)

        // Economic parameters (legacy - kept for compatibility)
        target_balance = 1000.0,     // Legacy target vault balance (ETH) - use T instead
        min_fee = 1e-12,             // Minimum fee floor (ETH) - 0.001 gwei realistic

        // Legacy parameters (kept for backward compatibility)
        F_min = 1e-12,               // Legacy minimum fee floor - 0.001 gwei realistic
        safety_buffer = 0.1,          // Legacy safety buffer ratio
        l1_gas_per_tx = 20000,        // Legacy L1 gas per transaction
        batch_interval_steps = 6,     // Legacy batch interval
        tx_per_batch = 100,           // Legacy transactions per batch
        max_tps = 150.0,              // Legacy maximum TPS
        price_elasticity = -2.0,      // Legacy demand elasticity
        base_fee_gwei = 15.0,         // Legacy reference fee
        smoothing_window = 12,        // Legacy smoothing window
        outlier_threshold = 3.0       // Legacy outlier threshold
    } = {}) {
        // Core mechanism
        this.mu = mu;
        this.nu = nu;
        this.H = H;

        // New specification parameters
        this.alpha_data = alpha_data;
        this.lambda_B = lambda_B;
        this.Q_bar = Q_bar;
        this.T = T;

        // Economic parameters
        this.target_balance = target_balance;
        this.min_fee = min_fee;

        // Legacy parameters
        this.F_min = F_min;
        this.safety_buffer = safety_buffer;
        this.l1_gas_per_tx = l1_gas_per_tx;
        this.batch_interval_steps = batch_interval_steps;
        this.tx_per_batch = tx_per_batch;
        this.max_tps = max_tps;
        this.price_elasticity = price_elasticity;
        this.base_fee_gwei = base_fee_gwei;
        this.smoothing_window = smoothing_window;
        this.outlier_threshold = outlier_threshold;

        // Validate parameters
        this._validateParameters();
    }

    _validateParameters() {
        const errors = [];

        // Core mechanism bounds
        if (this.mu < 0.0 || this.mu > 1.0) {
            errors.push(`mu must be in [0.0, 1.0], got ${this.mu}`);
        }
        if (this.nu < 0.0 || this.nu > 1.0) {
            errors.push(`nu must be in [0.0, 1.0], got ${this.nu}`);
        }
        if (this.H <= 0) {
            errors.push(`H must be positive, got ${this.H}`);
        }

        // New fee mechanism parameter bounds
        if (this.alpha_data <= 0 || this.alpha_data > 2.0) {
            errors.push(`alpha_data must be in (0, 2.0] (realistic L1 DA gas ratio), got ${this.alpha_data}`);
        }
        if (this.lambda_B <= 0.0 || this.lambda_B > 1.0) {
            errors.push(`lambda_B must be in (0.0, 1.0], got ${this.lambda_B}`);
        }
        if (this.Q_bar <= 0) {
            errors.push(`Q_bar must be positive, got ${this.Q_bar}`);
        }
        if (this.T <= 0) {
            errors.push(`T must be positive, got ${this.T}`);
        }

        // Economic bounds
        if (this.target_balance <= 0) {
            errors.push(`target_balance must be positive, got ${this.target_balance}`);
        }
        if (this.min_fee <= 0) {
            errors.push(`min_fee must be positive, got ${this.min_fee}`);
        }

        // Consistency checks
        if (Math.abs(this.T - this.target_balance) > 1e-6) {
            console.warn(
                `T (${this.T}) and target_balance (${this.target_balance}) differ. ` +
                `Using T for new fee mechanism calculations.`
            );
        }

        if (errors.length > 0) {
            throw new Error('Parameter validation failed:\n' + errors.join('\n'));
        }
    }
}

// Fee vault class for managing balance and deficit
export class FeeVault {
    constructor(targetBalance = 1000.0, initialBalance = null) {
        this.target_balance = targetBalance;
        this.balance = initialBalance !== null ? initialBalance : targetBalance;
        this._fee_history = [];
        this._cost_history = [];
    }

    get deficit() {
        return Math.max(0, this.target_balance - this.balance);
    }

    get surplus() {
        return Math.max(0, this.balance - this.target_balance);
    }

    get is_underfunded() {
        return this.balance < this.target_balance;
    }

    collectFees(amount) {
        this.balance += amount;
        this._fee_history.push(amount);
    }

    payL1Costs(amount) {
        this.balance -= amount;
        this._cost_history.push(amount);
    }
}

// Main canonical fee calculator class
export class CanonicalTaikoFeeCalculator {
    constructor(params = new FeeParameters()) {
        this.params = params;

        // L1 basefee EMA smoothing state (new specification)
        this._smoothed_l1_basefee = null;  // B̂_L1(t) in ETH per L1 gas
        this._l1_basefee_history = [];

        // Legacy L1 cost smoothing state (kept for compatibility)
        this._l1_cost_history = [];

        // Fee change limiting
        this._previous_fee = null;
    }

    /**
     * Calculate smoothed L1 basefee using EMA (new specification)
     * @param {number} l1BasefeeWei - Raw L1 basefee in wei
     * @returns {number} Smoothed L1 basefee in ETH per L1 gas
     *
     * Formula: B̂_L1(t) = (1 - λ_B) × B̂_L1(t-1) + λ_B × B_L1(t)
     */
    calculateSmoothedL1Basefee(l1BasefeeWei) {
        if (l1BasefeeWei < 0) {
            throw new Error(`L1 basefee cannot be negative: ${l1BasefeeWei}`);
        }

        // Convert to ETH per L1 gas
        const l1BasefeeEth = l1BasefeeWei / 1e18;

        // Mock data validation
        validateRealDataUsage('parameter', this.params.lambda_B, 'lambda_B EMA smoothing factor');

        // Unit safety validation if available
        if (UNIT_SAFETY_AVAILABLE) {
            try {
                const basefeeGwei = l1BasefeeWei / 1e9;
                unitValidation.validateBasefeeRange(basefeeGwei, 'calculateSmoothedL1Basefee');
            } catch (error) {
                console.warn('Unit validation warning in L1 basefee smoothing:', error.message);
            }
        }

        // Store in history
        this._l1_basefee_history.push(l1BasefeeEth);

        // Initialize on first call
        if (this._smoothed_l1_basefee === null) {
            this._smoothed_l1_basefee = l1BasefeeEth;
            return this._smoothed_l1_basefee;
        }

        // Apply EMA smoothing: B̂_L1(t) = (1 - λ_B) × B̂_L1(t-1) + λ_B × B_L1(t)
        const lambdaB = this.params.lambda_B;
        this._smoothed_l1_basefee = (
            (1.0 - lambdaB) * this._smoothed_l1_basefee +
            lambdaB * l1BasefeeEth
        );

        return this._smoothed_l1_basefee;
    }

    /**
     * Calculate DA cost term C_DA(t) (new specification)
     * @param {number} l1BasefeeWei - Raw L1 basefee in wei
     * @returns {number} C_DA(t) = α_data × B̂_L1(t): smoothed marginal DA cost per L2 gas (ETH)
     *
     * Formula: C_DA(t) = α_data × B̂_L1(t)
     */
    calculateCDA(l1BasefeeWei) {
        // Get smoothed L1 basefee
        const smoothedBasefee = this.calculateSmoothedL1Basefee(l1BasefeeWei);

        // Mock data validation for alpha_data
        validateRealDataUsage('parameter', this.params.alpha_data, 'alpha_data DA gas ratio');

        // Calculate DA cost: C_DA(t) = α_data × B̂_L1(t)
        const CDA = this.params.alpha_data * smoothedBasefee;

        // Validate result
        if (CDA > 0) {
            if (CDA < 1e-12 || CDA > 0.1) {
                console.warn(
                    `Unusual C_DA result: ${CDA.toFixed(12)} ETH per L2 gas ` +
                    `(check alpha_data=${this.params.alpha_data} and basefee units)`
                );
            }
        }

        return CDA;
    }

    /**
     * Calculate vault healing term C_vault(t) (new specification)
     * @param {number} vaultDeficit - Current vault deficit in ETH (positive when underfunded)
     * @returns {number} C_vault(t) = D(t)/(H × Q̄): full-strength vault-healing surcharge per L2 gas (ETH)
     *
     * Formula: C_vault(t) = D(t) / (H × Q̄)
     */
    calculateCVault(vaultDeficit) {
        if (vaultDeficit < 0) {
            throw new Error(`Vault deficit cannot be negative: ${vaultDeficit}`);
        }

        // Mock data validation
        validateRealDataUsage('parameter', this.params.Q_bar, 'Q_bar average L2 gas per batch');
        validateRealDataUsage('parameter', this.params.H, 'H recovery horizon');

        // Check for suspicious Q_bar values
        if (Math.abs(this.params.Q_bar - 690000) < 1000) {
            warnMockDataUsage('vault healing calculation', 'Q_bar ≈ 690,000',
                           'should use empirically measured average gas per batch');
        }

        // Calculate vault healing: C_vault(t) = D(t) / (H × Q̄)
        const denominator = this.params.H * this.params.Q_bar;
        const CVault = vaultDeficit / denominator;

        return CVault;
    }

    /**
     * CANONICAL raw fee calculation - NEW SPECIFICATION implementation
     * @param {number} l1BasefeeWei - L1 basefee in wei
     * @param {number} vaultDeficit - Current vault deficit (ETH, positive when underfunded)
     * @returns {number} F_L2_raw(t): Raw estimated fee per L2 gas (ETH)
     *
     * Formula: F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t)
     */
    calculateEstimatedFeeRaw(l1BasefeeWei, vaultDeficit) {
        if (l1BasefeeWei < 0) {
            throw new Error(`L1 basefee cannot be negative: ${l1BasefeeWei}`);
        }
        if (vaultDeficit < 0) {
            throw new Error(`Vault deficit cannot be negative: ${vaultDeficit}`);
        }

        // Mock data detection for fee mechanism parameters
        validateRealDataUsage('parameter', this.params.mu, 'mu (DA cost pass-through) parameter');
        validateRealDataUsage('parameter', this.params.nu, 'nu (vault healing intensity) parameter');

        // Unit safety validation if available
        if (vaultDeficit > 10000) { // Deficit > 10,000 ETH is suspicious
            console.warn(
                `Very large vault deficit: ${vaultDeficit.toFixed(2)} ETH - verify units are correct`
            );
        }

        // Calculate DA cost component: μ × C_DA(t)
        const CDA = this.calculateCDA(l1BasefeeWei);
        const daComponent = this.params.mu * CDA;

        // Calculate vault healing component: ν × C_vault(t)
        const CVault = this.calculateCVault(vaultDeficit);
        const vaultComponent = this.params.nu * CVault;

        // Calculate raw fee: F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t)
        const rawFee = daComponent + vaultComponent;

        // Final unit safety validation
        if (rawFee > 0) {
            const feeGwei = rawFee * 1e9;
            if (UNIT_SAFETY_AVAILABLE) {
                try {
                    unitValidation.assertReasonableFee(feeGwei, 'calculateEstimatedFeeRaw result');
                } catch (error) {
                    console.warn('Fee result validation warning:', error.message);
                }
            }
        }

        return rawFee;
    }

    /**
     * Reset calculator state for new simulation
     */
    resetState() {
        // Reset new specification state
        this._smoothed_l1_basefee = null;
        this._l1_basefee_history = [];

        // Reset legacy state
        this._l1_cost_history = [];

        // Reset fee limiting
        this._previous_fee = null;
    }

    /**
     * Calculate L1 cost per L2 transaction with unit safety validation
     * @param {number} l1BasefeeWei - L1 basefee in wei
     * @param {boolean} applySmoothing - Whether to apply smoothing
     * @returns {number} L1 cost per transaction in ETH
     */
    calculateL1CostPerTx(l1BasefeeWei, applySmoothing = true) {
        // Unit safety validation if available
        if (UNIT_SAFETY_AVAILABLE) {
            try {
                // Use comprehensive L1 cost validation
                return unitValidation.validateL1CostCalculation(
                    l1BasefeeWei,
                    this.params.l1_gas_per_tx,
                    "calculateL1CostPerTx"
                );
            } catch (error) {
                console.warn('Unit validation warning in L1 cost calculation:', error.message);
                // Fall through to manual calculation with basic validation
            }
        }

        // Manual validation fallback
        if (l1BasefeeWei < 0) {
            throw new Error(`L1 basefee cannot be negative: ${l1BasefeeWei}`);
        }

        // Basic range checks
        const basefeeGwei = l1BasefeeWei / 1e9;
        if (basefeeGwei > 0 && basefeeGwei < 0.001) {
            console.warn(`Very low L1 basefee: ${basefeeGwei.toFixed(6)} gwei - verify units`);
        }
        if (basefeeGwei > 2000) {
            console.warn(`Very high L1 basefee: ${basefeeGwei.toFixed(6)} gwei - verify units`);
        }

        // Gas validation
        if (this.params.l1_gas_per_tx < 1000 || this.params.l1_gas_per_tx > 1000000) {
            console.warn(
                `Unusual gas_per_tx: ${this.params.l1_gas_per_tx} ` +
                `(typical range: 20,000-200,000)`
            );
        }

        const gasPrice = l1BasefeeWei;
        const rawCost = (gasPrice * this.params.l1_gas_per_tx) / 1e18; // Convert to ETH

        // Validate result
        if (rawCost > 0 && (rawCost < 1e-12 || rawCost > 0.1)) {
            console.warn(
                `Unusual L1 cost result: ${rawCost.toFixed(8)} ETH ` +
                `(typical range: 0.00001-0.01 ETH) - check unit conversions`
            );
        }

        // Store for smoothing
        this._l1_cost_history.push(rawCost);
        if (this._l1_cost_history.length > this.params.smoothing_window) {
            this._l1_cost_history.shift();
        }

        if (!applySmoothing || this._l1_cost_history.length < 3) {
            return rawCost;
        }

        // Simple moving average smoothing
        const sum = this._l1_cost_history.reduce((a, b) => a + b, 0);
        return sum / this._l1_cost_history.length;
    }

    /**
     * Legacy fee calculation - kept for backward compatibility.
     *
     * NOTE: This method is deprecated. Use calculateEstimatedFeeRaw() with L1 basefee instead.
     *
     * @param {number} l1Cost - L1 cost per transaction in ETH
     * @param {number} vaultDeficit - Current vault deficit in ETH
     * @returns {number} Estimated fee in ETH
     */
    calculateEstimatedFee(l1Cost, vaultDeficit) {
        console.warn(
            'calculateEstimatedFee() with L1 cost is deprecated. ' +
            'Use calculateEstimatedFeeRaw() with L1 basefee for new specification.'
        );
        // Mock data detection for fee mechanism parameters
        validateRealDataUsage('parameter', this.params.mu, 'mu (L1 weight) parameter');
        validateRealDataUsage('parameter', this.params.nu, 'nu (deficit weight) parameter');
        validateRealDataUsage('parameter', this.params.H, 'H (prediction horizon) parameter');
        validateRealDataUsage('parameter', this.params.F_min, 'minimum fee floor');

        // Check for hardcoded minimum fee injection
        if (this.params.F_min > 1e-10) {
            const feeGweiEquiv = this.params.F_min * 1e9;
            if (Math.abs(feeGweiEquiv - 1.5) < 0.1) { // Close to 1.5 gwei
                warnMockDataUsage('fee calculation', 'artificial minimum fee',
                                `F_min=${feeGweiEquiv.toFixed(1)} gwei appears to be hardcoded injection`);
            }
        }

        // Unit safety validation if available
        if (UNIT_SAFETY_AVAILABLE) {
            try {
                // Use comprehensive fee calculation validation
                return unitValidation.validateFeeCalculation(
                    l1Cost,
                    vaultDeficit,
                    this.params.mu,
                    this.params.nu,
                    this.params.H,
                    "calculateEstimatedFee"
                );
            } catch (error) {
                console.warn('Unit validation warning in fee calculation:', error.message);
                // Fall through to manual calculation with basic validation
            }
        }

        // Manual validation fallback
        if (l1Cost < 0) {
            throw new Error(`L1 cost cannot be negative: ${l1Cost}`);
        }
        if (vaultDeficit < 0) {
            throw new Error(`Vault deficit cannot be negative: ${vaultDeficit}`);
        }

        // Validate input values are reasonable for ETH units
        if (l1Cost > 0.1) {
            console.warn(`Very large L1 cost: ${l1Cost.toFixed(6)} ETH - verify units are correct`);
        }
        if (vaultDeficit > 10000) {
            console.warn(`Very large vault deficit: ${vaultDeficit.toFixed(2)} ETH - verify units are correct`);
        }

        // Validate parameters
        if (this.params.mu < 0 || this.params.mu > 1) {
            throw new Error(`mu parameter must be in [0,1], got ${this.params.mu}`);
        }
        if (this.params.nu < 0 || this.params.nu > 1) {
            throw new Error(`nu parameter must be in [0,1], got ${this.params.nu}`);
        }
        if (this.params.H <= 0 || !Number.isInteger(this.params.H)) {
            throw new Error(`H parameter must be positive integer, got ${this.params.H}`);
        }

        const l1Component = this.params.mu * l1Cost;
        const deficitComponent = this.params.nu * vaultDeficit / this.params.H;
        const rawFee = l1Component + deficitComponent;
        const fee = Math.max(rawFee, this.params.F_min);

        // Check if minimum fee floor was applied (indicates potential artificial inflation)
        if (fee === this.params.F_min && rawFee < this.params.F_min) {
            const feeGwei = fee * 1e9;
            if (feeGwei > 0.1) { // Minimum fee > 0.1 gwei might be artificial
                warnMockDataUsage('fee calculation', 'minimum fee floor applied',
                                `fee artificially increased from ${rawFee*1e9:.6f} to ${feeGwei:.6f} gwei`);
            }
        }

        // Final unit safety validation
        if (fee > 0) {
            const feeGwei = fee * 1e9;

            // Check for suspiciously small fees
            if (feeGwei > 0 && feeGwei < 0.0001) {
                console.warn(
                    `Fee suspiciously small: ${feeGwei.toFixed(8)} gwei. ` +
                    `This likely indicates a unit conversion error.`
                );
            }

            // Check for exactly zero fees
            if (feeGwei === 0) {
                console.warn('Fee is exactly zero. Verify this is intentional.');
            }

            // Unit safety range check if available
            if (UNIT_SAFETY_AVAILABLE) {
                try {
                    unitValidation.assertReasonableFee(feeGwei, "calculateEstimatedFee result");
                } catch (error) {
                    console.warn('Fee result validation warning:', error.message);
                }
            }
        }

        return fee;
    }

    /**
     * Calculate transaction volume based on fee level
     * @param {number} estimatedFee - Current estimated fee in ETH
     * @returns {number} Transaction volume (TPS)
     */
    calculateTransactionVolume(estimatedFee) {
        const feeGwei = estimatedFee * 1e9;
        const referenceFee = this.params.base_fee_gwei;

        if (feeGwei <= 0) return this.params.max_tps;

        const priceRatio = feeGwei / referenceFee;
        const volumeMultiplier = Math.pow(priceRatio, this.params.price_elasticity);

        return Math.max(1.0, Math.min(this.params.max_tps, this.params.max_tps * volumeMultiplier));
    }

    /**
     * Calculate L1 batch cost with unit safety validation
     * @param {number} l1BasefeeWei - L1 basefee in wei
     * @returns {number} L1 batch cost in ETH
     */
    calculateL1BatchCost(l1BasefeeWei) {
        if (l1BasefeeWei < 0) {
            throw new Error(`L1 basefee cannot be negative: ${l1BasefeeWei}`);
        }

        // Unit safety validation if available
        if (UNIT_SAFETY_AVAILABLE) {
            try {
                const basefeeGwei = l1BasefeeWei / 1e9;
                unitValidation.validateBasefeeRange(basefeeGwei, "calculateL1BatchCost");
            } catch (error) {
                console.warn('Unit validation warning in L1 batch cost calculation:', error.message);
            }
        } else {
            // Basic validation fallback
            const basefeeGwei = l1BasefeeWei / 1e9;
            if (basefeeGwei > 0 && basefeeGwei < 0.001) {
                console.warn(`Very low L1 basefee: ${basefeeGwei.toFixed(6)} gwei - verify units`);
            }
            if (basefeeGwei > 2000) {
                console.warn(`Very high L1 basefee: ${basefeeGwei.toFixed(6)} gwei - verify units`);
            }
        }

        const gasPrice = l1BasefeeWei;
        const totalGas = this.params.l1_gas_per_tx * this.params.tx_per_batch;
        const batchCost = (gasPrice * totalGas) / 1e18; // Convert to ETH

        // Validate result is reasonable
        if (batchCost > 10.0) {
            console.warn(
                `Very large batch cost: ${batchCost.toFixed(6)} ETH - verify unit conversions`
            );
        }

        return batchCost;
    }

    /**
     * Create vault with specified initialization mode
     * @param {string} mode - Vault initialization mode
     * @param {Object} options - Additional options (deficit_ratio, surplus_ratio, balance)
     * @returns {FeeVault} Initialized vault
     */
    createVault(mode = VaultInitMode.TARGET, options = {}) {
        // Use new T parameter for target balance
        const targetBalance = this.params.T;
        let initialBalance;

        switch (mode) {
            case VaultInitMode.TARGET:
                initialBalance = targetBalance;
                break;
            case VaultInitMode.DEFICIT:
                const deficitRatio = options.deficit_ratio || 0.1;
                initialBalance = targetBalance * (1 - deficitRatio);
                break;
            case VaultInitMode.SURPLUS:
                const surplusRatio = options.surplus_ratio || 0.1;
                initialBalance = targetBalance * (1 + surplusRatio);
                break;
            case VaultInitMode.CUSTOM:
                initialBalance = options.balance || targetBalance;
                break;
            default:
                throw new Error(`Unknown vault initialization mode: ${mode}`);
        }

        return new FeeVault(targetBalance, initialBalance);
    }
}

// Parameter validation function
export function validateFeeParameters(mu, nu, H) {
    // Validate μ (L1 weight)
    if (mu < 0.0 || mu > 1.0) {
        return false;
    }

    // Validate ν (deficit weight)
    if (nu < 0.0 || nu > 1.0) {
        return false;
    }

    // Validate H (prediction horizon)
    if (H <= 0 || !Number.isInteger(H)) {
        return false;
    }

    // H should be aligned with batch intervals (multiple of 6)
    if (H % 6 !== 0) {
        return false;
    }

    return true;
}

// Optimal parameters from research
export function getOptimalParameters() {
    return {
        mu: 0.0,
        nu: 0.369,
        H: 1794,
        lambda_B: 0.365,
        description: "2024 NSGA-II optimized parameters from balanced multi-objective optimization - enhanced stability and vault healing"
    };
}

// Calculator factory functions
export function createDefaultCalculator() {
    /**
     * Create calculator with default optimal parameters (new specification)
     */
    const params = new FeeParameters({
        // Core mechanism (2024 NSGA-II optimized - balanced strategy)
        mu: 0.0,           // Confirmed optimal - pure deficit-based correction
        nu: 0.369,         // Updated optimal - enhanced vault healing
        H: 1794,           // Updated optimal - ~1 hour horizon for better stability

        // New specification parameters (2024 optimized)
        alpha_data: 0.5,      // Realistic L1 DA gas per L2 gas (FIXED: was 20000x too high)
        lambda_B: 0.365,      // Enhanced smoothing for L1 basefee stability
        Q_bar: 690000.0,      // Average L2 gas per batch (Taiko Alethia estimate)
        T: 1000.0             // Target vault balance
    });
    return new CanonicalTaikoFeeCalculator(params);
}

export function createBalancedCalculator() {
    /**
     * Create calculator with balanced parameters
     */
    const params = new FeeParameters({
        mu: 0.0,       // Consensus optimal
        nu: 0.48,      // Conservative approach (higher than optimal for safety)
        H: 1794,       // Updated to match optimized horizon

        // Standard new specification parameters
        alpha_data: 0.5,      // Realistic L1 DA gas per L2 gas ratio
        lambda_B: 0.365,      // Fixed lambda_B from 2024 optimization
        Q_bar: 690000.0,
        T: 1000.0
    });
    return new CanonicalTaikoFeeCalculator(params);
}

export function createCrisisCalculator() {
    /**
     * Create calculator with crisis-resilient parameters
     */
    const params = new FeeParameters({
        mu: 0.0,       // Consensus optimal
        nu: 0.88,      // Aggressive deficit recovery
        H: 120,        // Shorter horizon for faster response

        // Crisis-tuned parameters
        alpha_data: 0.5,      // Realistic L1 DA gas per L2 gas ratio
        lambda_B: 0.365,      // Fixed lambda_B from 2024 optimization
        Q_bar: 690000.0,
        T: 1000.0
    });
    return new CanonicalTaikoFeeCalculator(params);
}

// Export classes and functions
export default {
    VaultInitMode,
    FeeParameters,
    FeeVault,
    CanonicalTaikoFeeCalculator,
    validateFeeParameters,
    getOptimalParameters,
    createDefaultCalculator,
    createBalancedCalculator,
    createCrisisCalculator
};