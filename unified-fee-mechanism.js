/**
 * Unified Taiko Fee Mechanism Implementation (JavaScript)
 *
 * This is the JAVASCRIPT MIRROR of src/core/unified_fee_mechanism.py.
 * It implements the identical L2 Sustainability Basefee specification.
 *
 * ALL FORMULAS AND LOGIC MUST MATCH THE PYTHON IMPLEMENTATION EXACTLY.
 */

/**
 * Parameter calibration status levels.
 */
const ParameterCalibrationStatus = {
    CALIBRATED: 'calibrated',      // Based on real Taiko data
    THEORETICAL: 'theoretical',    // Based on theoretical estimates
    UNCALIBRATED: 'uncalibrated'   // No real data, using placeholder
};

/**
 * Unified parameter set for fee mechanism configuration.
 */
class FeeParameters {
    constructor({
        // Core mechanism parameters
        mu = 0.0,                      // DA cost pass-through coefficient [0.0, 1.0]
        nu = 0.5,                      // Vault healing intensity coefficient [0.0, 1.0]
        H = 144,                       // Recovery horizon (batches under typical load)
        lambda_B = 0.3,                // EMA smoothing factor for L1 basefee [0.0, 1.0]

        // System constants (REQUIRE REAL DATA CALIBRATION)
        alpha_data = 0.022,             // Expected L1 DA gas per 1 L2 gas [THEORETICAL ESTIMATE]
        Q_bar = 150000.0,             // Typical L2 gas per batch [CONSERVATIVE ESTIMATE]
        T = 1000.0,                    // Target vault balance (ETH)

        // UX wrapper parameters
        F_min = 1e-12,                 // Minimum sustainability basefee (ETH per L2 gas)
        F_max = 1e-6,                  // Maximum sustainability basefee (ETH per L2 gas)
        kappa_up = 0.1,                // Max relative fee increase per batch [0.0, 1.0]
        kappa_down = 0.1,              // Max relative fee decrease per batch [0.0, 1.0]

        // Parameter calibration status tracking
        alpha_data_status = ParameterCalibrationStatus.THEORETICAL,
        Q_bar_status = ParameterCalibrationStatus.THEORETICAL,
        optimization_status = ParameterCalibrationStatus.UNCALIBRATED
    } = {}) {
        this.mu = mu;
        this.nu = nu;
        this.H = H;
        this.lambda_B = lambda_B;
        this.alpha_data = alpha_data;
        this.Q_bar = Q_bar;
        this.T = T;
        this.F_min = F_min;
        this.F_max = F_max;
        this.kappa_up = kappa_up;
        this.kappa_down = kappa_down;
        this.alpha_data_status = alpha_data_status;
        this.Q_bar_status = Q_bar_status;
        this.optimization_status = optimization_status;

        this._validateRanges();
        this._emitCalibrationWarnings();
    }

    _validateRanges() {
        if (this.mu < 0.0 || this.mu > 1.0) {
            throw new Error(`mu must be in [0.0, 1.0], got ${this.mu}`);
        }
        if (this.nu < 0.0 || this.nu > 1.0) {
            throw new Error(`nu must be in [0.0, 1.0], got ${this.nu}`);
        }
        if (this.lambda_B <= 0.0 || this.lambda_B > 1.0) {
            throw new Error(`lambda_B must be in (0.0, 1.0], got ${this.lambda_B}`);
        }
        if (this.H <= 0) {
            throw new Error(`H must be positive, got ${this.H}`);
        }
        if (this.alpha_data <= 0) {
            throw new Error(`alpha_data must be positive, got ${this.alpha_data}`);
        }
        if (this.Q_bar <= 0) {
            throw new Error(`Q_bar must be positive, got ${this.Q_bar}`);
        }
        if (this.kappa_up < 0.0 || this.kappa_up > 1.0) {
            throw new Error(`kappa_up must be in [0.0, 1.0], got ${this.kappa_up}`);
        }
        if (this.kappa_down < 0.0 || this.kappa_down > 1.0) {
            throw new Error(`kappa_down must be in [0.0, 1.0], got ${this.kappa_down}`);
        }
    }

    _emitCalibrationWarnings() {
        if (this.alpha_data_status !== ParameterCalibrationStatus.CALIBRATED) {
            console.warn(
                `🚨 PARAMETER WARNING: α_data = ${this.alpha_data} is ${this.alpha_data_status.toUpperCase()}. ` +
                `This parameter requires calibration from real Taiko proposeBlock transaction data. ` +
                `Current estimate may be off by orders of magnitude.`
            );
        }

        if (this.Q_bar_status !== ParameterCalibrationStatus.CALIBRATED) {
            console.warn(
                `🚨 PARAMETER WARNING: Q̄ = ${this.Q_bar} is ${this.Q_bar_status.toUpperCase()}. ` +
                `This parameter requires measurement from real Taiko L2 batch sizes.`
            );
        }

        if (this.optimization_status !== ParameterCalibrationStatus.CALIBRATED) {
            console.warn(
                `🚨 OPTIMIZATION WARNING: Parameters (μ=${this.mu}, ν=${this.nu}, H=${this.H}) are ${this.optimization_status.toUpperCase()}. ` +
                `Optimal values require re-optimization with calibrated α_data and Q̄.`
            );
        }
    }
}

/**
 * Vault state tracking.
 */
class VaultState {
    constructor(balance = 1000.0, target = 1000.0) {
        this.balance = balance;
        this.target = target;
    }

    get deficit() {
        return Math.max(0.0, this.target - this.balance);
    }

    get surplus() {
        return Math.max(0.0, this.balance - this.target);
    }
}

/**
 * Unified Fee Calculator implementing the authoritative L2 Sustainability Basefee specification.
 *
 * This implements: F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t)
 * With UX wrapper providing clipping and rate limiting.
 */
class UnifiedFeeCalculator {
    constructor(params = null) {
        this.params = params || new FeeParameters();
        this._smoothed_l1_basefee = null;
        this._last_final_fee = null;

        this._printInitializationStatus();
    }

    _printInitializationStatus() {
        console.log("🏗️ UNIFIED FEE CALCULATOR INITIALIZED");
        console.log(`   Formula: F_L2_raw(t) = μ×C_DA(t) + ν×C_vault(t)`);
        console.log(`   Parameters: μ=${this.params.mu}, ν=${this.params.nu}, H=${this.params.H}`);
        console.log(`   Constants: α_data=${this.params.alpha_data} (${this.params.alpha_data_status})`);
        console.log(`             Q̄=${this.params.Q_bar.toLocaleString()} (${this.params.Q_bar_status})`);

        if (this.params.alpha_data_status !== ParameterCalibrationStatus.CALIBRATED ||
            this.params.Q_bar_status !== ParameterCalibrationStatus.CALIBRATED) {
            console.log(`   ⚠️  UNCALIBRATED parameters detected - see warnings above`);
        }
        console.log();
    }

    updateSmoothedL1Basefee(l1BasefeeWei) {
        const l1BasefeeEthPerGas = l1BasefeeWei / 1e18;

        if (this._smoothed_l1_basefee === null) {
            // Initialize with first observation
            this._smoothed_l1_basefee = l1BasefeeEthPerGas;
        } else {
            // EMA update: B̂_L1(t) = (1 - λ_B) × B̂_L1(t-1) + λ_B × B_L1(t)
            this._smoothed_l1_basefee = (
                (1 - this.params.lambda_B) * this._smoothed_l1_basefee +
                this.params.lambda_B * l1BasefeeEthPerGas
            );
        }

        return this._smoothed_l1_basefee;
    }

    calculateCDA(l1BasefeeWei) {
        const smoothedBasefee = this.updateSmoothedL1Basefee(l1BasefeeWei);
        return this.params.alpha_data * smoothedBasefee;
    }

    calculateCVault(vaultDeficit) {
        return vaultDeficit / (this.params.H * this.params.Q_bar);
    }

    calculateRawFee(l1BasefeeWei, vaultDeficit) {
        const C_DA = this.calculateCDA(l1BasefeeWei);
        const C_vault = this.calculateCVault(vaultDeficit);

        return this.params.mu * C_DA + this.params.nu * C_vault;
    }

    applyClipping(rawFee) {
        return Math.min(Math.max(rawFee, this.params.F_min), this.params.F_max);
    }

    applyRateLimiting(clippedFee) {
        if (this._last_final_fee === null) {
            // Initialize with first fee
            this._last_final_fee = clippedFee;
            return clippedFee;
        }

        // Calculate rate-limited bounds
        const maxIncrease = this._last_final_fee * (1 + this.params.kappa_up);
        const maxDecrease = this._last_final_fee * (1 - this.params.kappa_down);

        // Apply rate limiting
        const finalFee = Math.min(maxIncrease, Math.max(maxDecrease, clippedFee));
        this._last_final_fee = finalFee;

        return finalFee;
    }

    calculateFinalFee(l1BasefeeWei, vaultDeficit) {
        // Step 1: Calculate raw fee
        const rawFee = this.calculateRawFee(l1BasefeeWei, vaultDeficit);

        // Step 2: Apply clipping
        const clippedFee = this.applyClipping(rawFee);

        // Step 3: Apply rate limiting
        const finalFee = this.applyRateLimiting(clippedFee);

        return {
            raw_fee_eth_per_gas: rawFee,
            clipped_fee_eth_per_gas: clippedFee,
            final_fee_eth_per_gas: finalFee,
            final_fee_gwei_per_gas: finalFee * 1e9,
            C_DA: this.calculateCDA(l1BasefeeWei),
            C_vault: this.calculateCVault(vaultDeficit),
            smoothed_l1_basefee: this._smoothed_l1_basefee
        };
    }
}

/**
 * Create calculator with conservative parameter estimates.
 */
function createConservativeCalculator() {
    const params = new FeeParameters({
        // Conservative mechanism parameters
        mu: 0.0,                      // Pure deficit correction (no L1 pass-through)
        nu: 0.3,                      // Moderate vault healing
        H: 144,                       // ~4.8 minute horizon
        lambda_B: 0.2,                // Conservative smoothing

        // Conservative estimates (marked as theoretical)
        alpha_data: 0.22,             // Midpoint of 0.15-0.28 theoretical range
        Q_bar: 200_000.0,             // Conservative batch size estimate
        T: 1000.0,                    // 1000 ETH target

        // Status tracking
        alpha_data_status: ParameterCalibrationStatus.THEORETICAL,
        Q_bar_status: ParameterCalibrationStatus.THEORETICAL,
        optimization_status: ParameterCalibrationStatus.UNCALIBRATED
    });

    return new UnifiedFeeCalculator(params);
}

/**
 * Create calculator with experimental parameters for testing.
 */
function createExperimentalCalculator() {
    const params = new FeeParameters({
        // Experimental parameters
        mu: 0.1,                      // Small L1 component
        nu: 0.7,                      // Aggressive vault healing
        H: 72,                        // ~2.4 minute horizon
        lambda_B: 0.5,                // Responsive smoothing

        // Same estimates (still theoretical)
        alpha_data: 0.22,
        Q_bar: 200_000.0,
        T: 1000.0,

        // Status tracking
        alpha_data_status: ParameterCalibrationStatus.THEORETICAL,
        Q_bar_status: ParameterCalibrationStatus.THEORETICAL,
        optimization_status: ParameterCalibrationStatus.UNCALIBRATED
    });

    return new UnifiedFeeCalculator(params);
}

// Export for both ES6 modules and CommonJS
if (typeof module !== 'undefined' && module.exports) {
    // CommonJS
    module.exports = {
        ParameterCalibrationStatus,
        FeeParameters,
        VaultState,
        UnifiedFeeCalculator,
        createConservativeCalculator,
        createExperimentalCalculator
    };
}

// ES6 modules (for web)
if (typeof window !== 'undefined') {
    window.UnifiedFeeModule = {
        ParameterCalibrationStatus,
        FeeParameters,
        VaultState,
        UnifiedFeeCalculator,
        createConservativeCalculator,
        createExperimentalCalculator
    };
}