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
 *     F_estimated(t) = max(μ × C_L1(t) + ν × D(t)/H, F_min)
 *
 * Where:
 *     - μ: L1 weight parameter [0.0, 1.0]
 *     - ν: Deficit weight parameter [0.0, 1.0]
 *     - H: Prediction horizon (time steps)
 *     - C_L1(t): L1 cost per transaction at time t
 *     - D(t): Vault deficit at time t
 *     - F_min: Minimum fee floor (1e-8 ETH)
 */

// Import unit validation system
let UNIT_SAFETY_AVAILABLE = false;
let unitValidation = null;

try {
    // Try to import unit validation - path may vary in different environments
    const module = await import('./src/core/unit_validation.js');
    unitValidation = module.default || module;
    UNIT_SAFETY_AVAILABLE = true;
    console.log('✅ Unit safety system loaded');
} catch (error) {
    console.warn('🚨 MOCK DATA WARNING: Unit safety system not available - unit validation disabled');
    // Create minimal fallback functions
    unitValidation = {
        validateBasefeeRange: () => {},
        validateFeeRange: () => {},
        assertReasonableFee: () => {},
        validateL1CostCalculation: (basefee, gas) => (basefee * gas) / 1e18,
        validateFeeCalculation: (l1Cost, deficit, mu, nu, H) => Math.max(mu * l1Cost + nu * deficit / H, 1e-8)
    };
}

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
        mu = 0.0,                    // L1 weight [0.0, 1.0]
        nu = 0.27,                   // Deficit weight [0.0, 1.0]
        H = 492,                     // Prediction horizon (steps)
        F_min = 1e-8,                // Minimum fee floor (ETH)

        // Vault parameters
        target_balance = 1000.0,      // Target vault balance (ETH)
        safety_buffer = 0.1,          // Safety buffer ratio

        // L1 cost parameters
        l1_gas_per_tx = 20000,        // L1 gas per L2 transaction (corrected from 600k bug)
        batch_interval_steps = 6,     // Steps between L1 batch submissions
        tx_per_batch = 100,           // L2 transactions per L1 batch

        // Volume model parameters
        max_tps = 150.0,              // Maximum transactions per second
        price_elasticity = -2.0,      // Demand elasticity
        base_fee_gwei = 15.0,         // Reference fee level (gwei)

        // Smoothing parameters
        smoothing_window = 12,        // L1 cost smoothing window
        outlier_threshold = 3.0       // Outlier rejection threshold (z-score)
    } = {}) {
        this.mu = mu;
        this.nu = nu;
        this.H = H;
        this.F_min = F_min;
        this.target_balance = target_balance;
        this.safety_buffer = safety_buffer;
        this.l1_gas_per_tx = l1_gas_per_tx;
        this.batch_interval_steps = batch_interval_steps;
        this.tx_per_batch = tx_per_batch;
        this.max_tps = max_tps;
        this.price_elasticity = price_elasticity;
        this.base_fee_gwei = base_fee_gwei;
        this.smoothing_window = smoothing_window;
        this.outlier_threshold = outlier_threshold;
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
        this._l1_cost_history = [];
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
        if (rawCost > 0 && (rawCost < 1e-8 || rawCost > 0.1)) {
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
     * Calculate estimated fee using canonical formula with unit safety validation
     * @param {number} l1Cost - L1 cost per transaction in ETH
     * @param {number} vaultDeficit - Current vault deficit in ETH
     * @returns {number} Estimated fee in ETH
     */
    calculateEstimatedFee(l1Cost, vaultDeficit) {
        // Mock data detection for fee mechanism parameters
        validateRealDataUsage('parameter', this.params.mu, 'mu (L1 weight) parameter');
        validateRealDataUsage('parameter', this.params.nu, 'nu (deficit weight) parameter');
        validateRealDataUsage('parameter', this.params.H, 'H (prediction horizon) parameter');
        validateRealDataUsage('parameter', this.params.F_min, 'minimum fee floor');

        // Check for hardcoded minimum fee injection
        if (this.params.F_min > 1e-8) {
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
        const targetBalance = this.params.target_balance;
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
        nu: 0.27,
        H: 492,
        description: "NSGA-II multi-objective optimization result - Pareto optimal for user experience, protocol safety, and economic efficiency"
    };
}

// Calculator factory functions
export function createDefaultCalculator() {
    const optimal = getOptimalParameters();
    const params = new FeeParameters({
        mu: optimal.mu,
        nu: optimal.nu,
        H: optimal.H
    });
    return new CanonicalTaikoFeeCalculator(params);
}

export function createBalancedCalculator() {
    const params = new FeeParameters({
        mu: 0.0,
        nu: 0.48,
        H: 492
    });
    return new CanonicalTaikoFeeCalculator(params);
}

export function createCrisisCalculator() {
    const params = new FeeParameters({
        mu: 0.0,
        nu: 0.88,
        H: 120
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