/**
 * JavaScript Unit Validation System
 *
 * This module provides unit safety for JavaScript fee mechanism calculations
 * to prevent critical unit conversion bugs (ETH/Wei/Gwei confusion).
 *
 * Key Features:
 * - Runtime validation for unit conversions
 * - Range checking for reasonable fee/basefee values
 * - Clear error messages for debugging unit issues
 * - Consistent validation with Python unit system
 *
 * Critical Safety Goals:
 * 1. Prevent ETH/Wei confusion (10^18 factor errors)
 * 2. Catch unreasonable fee values immediately
 * 3. Ensure consistent units across Python/JavaScript
 * 4. Provide clear error messages for debugging
 *
 * Example Usage:
 *     import { validateFeeRange, validateBasefeeRange, assertReasonableFee } from './unit_validation.js';
 *
 *     const feeGwei = calculateFee(l1Cost, deficit);
 *     validateFeeRange(feeGwei, "final fee calculation");
 *
 *     const l1BasefeeGwei = getL1Basefee();
 *     validateBasefeeRange(l1BasefeeGwei, "L1 data fetch");
 */

// Unit conversion constants
export const WEI_PER_GWEI = 1e9;
export const WEI_PER_ETH = 1e18;
export const GWEI_PER_ETH = 1e9;

// Reasonable bounds for validation (in gwei)
export const MIN_REASONABLE_FEE_GWEI = 0.001;      // 0.001 gwei minimum
export const MAX_REASONABLE_FEE_GWEI = 10000.0;    // 10,000 gwei maximum
export const MIN_REASONABLE_BASEFEE_GWEI = 0.001;  // Historical minimum ~0.055 gwei
export const MAX_REASONABLE_BASEFEE_GWEI = 2000.0; // Historical maximum ~1,352 gwei

// Custom error classes
export class UnitValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = 'UnitValidationError';
    }
}

export class UnitOverflowError extends Error {
    constructor(message) {
        super(message);
        this.name = 'UnitOverflowError';
    }
}

/**
 * Validate numeric value is safe for unit operations
 * @param {number} value - Value to validate
 * @param {string} context - Context for error messages
 * @throws {UnitValidationError} If value is invalid
 */
export function validateNumeric(value, context = "value") {
    if (typeof value !== 'number') {
        throw new UnitValidationError(`${context} must be numeric, got ${typeof value}`);
    }

    if (!Number.isFinite(value)) {
        throw new UnitValidationError(`${context} must be finite, got ${value}`);
    }

    if (value < 0) {
        throw new UnitValidationError(`${context} cannot be negative: ${value}`);
    }
}

/**
 * Safely convert gwei to wei with validation
 * @param {number} gweiValue - Value in gwei
 * @param {string} context - Context for error messages
 * @returns {number} Value in wei
 * @throws {UnitValidationError|UnitOverflowError} If conversion fails
 */
export function gweiToWei(gweiValue, context = "gwei to wei conversion") {
    validateNumeric(gweiValue, `gwei value in ${context}`);

    const weiValue = gweiValue * WEI_PER_GWEI;

    // Check for overflow (JavaScript safe integer limit)
    if (weiValue > Number.MAX_SAFE_INTEGER) {
        throw new UnitOverflowError(
            `Gwei to wei conversion would overflow in ${context}: ${gweiValue} gwei`
        );
    }

    return Math.round(weiValue);
}

/**
 * Safely convert wei to gwei with validation
 * @param {number} weiValue - Value in wei
 * @param {string} context - Context for error messages
 * @returns {number} Value in gwei
 * @throws {UnitValidationError} If conversion fails
 */
export function weiToGwei(weiValue, context = "wei to gwei conversion") {
    validateNumeric(weiValue, `wei value in ${context}`);

    return weiValue / WEI_PER_GWEI;
}

/**
 * Safely convert ETH to wei with validation
 * @param {number} ethValue - Value in ETH
 * @param {string} context - Context for error messages
 * @returns {number} Value in wei
 * @throws {UnitValidationError|UnitOverflowError} If conversion fails
 */
export function ethToWei(ethValue, context = "ETH to wei conversion") {
    validateNumeric(ethValue, `ETH value in ${context}`);

    const weiValue = ethValue * WEI_PER_ETH;

    // Check for overflow
    if (weiValue > Number.MAX_SAFE_INTEGER) {
        throw new UnitOverflowError(
            `ETH to wei conversion would overflow in ${context}: ${ethValue} ETH`
        );
    }

    return Math.round(weiValue);
}

/**
 * Safely convert wei to ETH with validation
 * @param {number} weiValue - Value in wei
 * @param {string} context - Context for error messages
 * @returns {number} Value in ETH
 * @throws {UnitValidationError} If conversion fails
 */
export function weiToEth(weiValue, context = "wei to ETH conversion") {
    validateNumeric(weiValue, `wei value in ${context}`);

    return weiValue / WEI_PER_ETH;
}

/**
 * Validate that fee is in reasonable range
 * @param {number} feeGwei - Fee value in gwei
 * @param {string} context - Context for error messages
 * @throws {UnitValidationError} If fee is outside reasonable range
 */
export function validateFeeRange(feeGwei, context = "fee calculation") {
    validateNumeric(feeGwei, `fee in ${context}`);

    if (feeGwei < MIN_REASONABLE_FEE_GWEI) {
        throw new UnitValidationError(
            `Fee too low in ${context}: ${feeGwei.toFixed(6)} gwei ` +
            `(minimum reasonable: ${MIN_REASONABLE_FEE_GWEI} gwei). ` +
            `This might indicate a unit conversion error.`
        );
    }

    if (feeGwei > MAX_REASONABLE_FEE_GWEI) {
        throw new UnitValidationError(
            `Fee too high in ${context}: ${feeGwei.toFixed(6)} gwei ` +
            `(maximum reasonable: ${MAX_REASONABLE_FEE_GWEI} gwei). ` +
            `This might indicate a unit conversion error.`
        );
    }
}

/**
 * Validate that L1 basefee is in reasonable range
 * @param {number} basefeeGwei - Basefee value in gwei
 * @param {string} context - Context for warning messages
 */
export function validateBasefeeRange(basefeeGwei, context = "basefee") {
    validateNumeric(basefeeGwei, `basefee in ${context}`);

    if (basefeeGwei < MIN_REASONABLE_BASEFEE_GWEI) {
        console.warn(
            `L1 basefee very low in ${context}: ${basefeeGwei.toFixed(6)} gwei ` +
            `(historical minimum ~0.055 gwei)`
        );
    }

    if (basefeeGwei > MAX_REASONABLE_BASEFEE_GWEI) {
        console.warn(
            `L1 basefee very high in ${context}: ${basefeeGwei.toFixed(6)} gwei ` +
            `(historical maximum ~1,352 gwei)`
        );
    }
}

/**
 * Assert that a value is in expected range for its alleged unit
 * @param {number} value - Value to check
 * @param {Array<number>} expectedRange - [min, max] expected range
 * @param {string} expectedUnit - Expected unit name
 * @param {string} context - Context for error messages
 * @throws {UnitValidationError} If value is outside expected range
 */
export function assertNoUnitMismatch(value, expectedRange, expectedUnit, context) {
    validateNumeric(value, `value in ${context}`);

    const [minVal, maxVal] = expectedRange;

    if (value < minVal || value > maxVal) {
        throw new UnitValidationError(
            `Value ${value} outside expected range for ${expectedUnit} ` +
            `in ${context}: expected [${minVal}, ${maxVal}]. ` +
            `This might indicate a unit mismatch (e.g., passing ETH when wei expected).`
        );
    }
}

/**
 * Assert that a fee value is reasonable (not suspiciously near zero)
 * @param {number} feeGwei - Fee value in gwei
 * @param {string} context - Context for error messages
 * @throws {UnitValidationError} If fee is suspiciously small
 */
export function assertReasonableFee(feeGwei, context = "fee result") {
    validateNumeric(feeGwei, `fee in ${context}`);

    // Check for suspiciously small fees (likely unit errors)
    if (feeGwei > 0 && feeGwei < 0.0001) {  // Between 0 and 0.0001 gwei is suspicious
        throw new UnitValidationError(
            `Fee suspiciously small in ${context}: ${feeGwei.toFixed(8)} gwei. ` +
            `This likely indicates a unit conversion error (ETH passed as wei?).`
        );
    }

    // Check for exactly zero fees (might be intentional or error)
    if (feeGwei === 0) {
        console.warn(`Fee is exactly zero in ${context}. Verify this is intentional.`);
    }
}

/**
 * Diagnose potential unit mismatches and suggest corrections
 * @param {number} value - Value to diagnose
 * @param {string} allegedUnit - What unit the value is claimed to be
 * @returns {string} Diagnostic message
 */
export function diagnoseUnitMismatch(value, allegedUnit) {
    let msg = `Diagnosing value ${value} alleged to be ${allegedUnit}:\n`;

    if (allegedUnit.toLowerCase() === "gwei") {
        msg += `  - As gwei: ${value.toFixed(6)} (reasonable: ${MIN_REASONABLE_FEE_GWEI}-${MAX_REASONABLE_FEE_GWEI})\n`;
        msg += `  - If actually wei: ${(value/1e9).toFixed(6)} gwei\n`;
        msg += `  - If actually ETH: ${(value*1e9).toFixed(6)} gwei\n`;

        if (value < 0.001) {
            msg += "  ⚠️  LIKELY ISSUE: Too small for gwei, might be ETH\n";
        } else if (value > 10000) {
            msg += "  ⚠️  LIKELY ISSUE: Too large for gwei, might be wei\n";
        }

    } else if (allegedUnit.toLowerCase() === "wei") {
        msg += `  - As wei: ${value.toLocaleString()}\n`;
        msg += `  - If actually gwei: ${(value*1e9).toLocaleString()} wei\n`;
        msg += `  - If actually ETH: ${(value*1e18).toLocaleString()} wei\n`;

        if (value < 1e15) {  // Less than 0.001 ETH in wei
            msg += "  ⚠️  POSSIBLE ISSUE: Very small for wei, might be ETH or gwei\n";
        }

    } else if (allegedUnit.toLowerCase() === "eth") {
        msg += `  - As ETH: ${value.toFixed(6)}\n`;
        msg += `  - If actually wei: ${(value/1e18).toFixed(6)} ETH\n`;
        msg += `  - If actually gwei: ${(value/1e9).toFixed(6)} ETH\n`;

        if (value > 1.0) {
            msg += "  ⚠️  POSSIBLE ISSUE: Very large for ETH (fee context)\n";
        } else if (value < 1e-10) {
            msg += "  ⚠️  LIKELY ISSUE: Too small for ETH, might be wei\n";
        }
    }

    return msg;
}

/**
 * Validate L1 cost calculation with comprehensive checks
 * @param {number} l1BasefeeWei - L1 basefee in wei
 * @param {number} gasPerTx - Gas per transaction
 * @param {string} context - Context for error messages
 * @returns {number} L1 cost in ETH
 * @throws {UnitValidationError} If inputs are invalid
 */
export function validateL1CostCalculation(l1BasefeeWei, gasPerTx, context = "L1 cost calculation") {
    validateNumeric(l1BasefeeWei, `L1 basefee (wei) in ${context}`);
    validateNumeric(gasPerTx, `gas per tx in ${context}`);

    // Validate basefee is reasonable as wei
    const basefeeGwei = weiToGwei(l1BasefeeWei, `${context} basefee conversion`);
    validateBasefeeRange(basefeeGwei, context);

    // Validate gas amount is reasonable
    if (gasPerTx < 1000 || gasPerTx > 1000000) {
        console.warn(
            `Unusual gas per tx in ${context}: ${gasPerTx} ` +
            `(typical range: 20,000-200,000)`
        );
    }

    // Calculate L1 cost
    const l1CostWei = l1BasefeeWei * gasPerTx;
    const l1CostEth = weiToEth(l1CostWei, `${context} cost conversion`);

    // Validate result is reasonable
    if (l1CostEth < 1e-8 || l1CostEth > 0.1) {
        console.warn(
            `Unusual L1 cost in ${context}: ${l1CostEth.toFixed(8)} ETH ` +
            `(typical range: 0.00001-0.01 ETH)`
        );
    }

    return l1CostEth;
}

/**
 * Validate fee calculation with comprehensive checks
 * @param {number} l1CostEth - L1 cost component in ETH
 * @param {number} deficitEth - Vault deficit in ETH
 * @param {number} mu - L1 weight parameter [0,1]
 * @param {number} nu - Deficit weight parameter [0,1]
 * @param {number} H - Prediction horizon (steps)
 * @param {string} context - Context for error messages
 * @returns {number} Estimated fee in ETH
 * @throws {UnitValidationError} If inputs are invalid
 */
export function validateFeeCalculation(l1CostEth, deficitEth, mu, nu, H, context = "fee calculation") {
    // Validate all inputs
    validateNumeric(l1CostEth, `L1 cost (ETH) in ${context}`);
    validateNumeric(deficitEth, `deficit (ETH) in ${context}`);
    validateNumeric(mu, `mu parameter in ${context}`);
    validateNumeric(nu, `nu parameter in ${context}`);
    validateNumeric(H, `H parameter in ${context}`);

    // Validate parameter ranges
    if (mu < 0 || mu > 1) {
        throw new UnitValidationError(`mu parameter must be in [0,1] in ${context}, got ${mu}`);
    }
    if (nu < 0 || nu > 1) {
        throw new UnitValidationError(`nu parameter must be in [0,1] in ${context}, got ${nu}`);
    }
    if (H <= 0 || !Number.isInteger(H)) {
        throw new UnitValidationError(`H parameter must be positive integer in ${context}, got ${H}`);
    }

    // Calculate fee components
    const l1Component = mu * l1CostEth;
    const deficitComponent = nu * deficitEth / H;
    const totalFee = l1Component + deficitComponent;

    // Apply minimum fee floor
    const minFee = 1e-8; // ETH
    const finalFee = Math.max(totalFee, minFee);

    // Validate result
    const finalFeeGwei = finalFee * GWEI_PER_ETH;
    validateFeeRange(finalFeeGwei, context);

    return finalFee;
}

/**
 * Create unit safety status report
 * @returns {string} Status report
 */
export function createUnitSafetyReport() {
    let report = "🛡️  JavaScript Unit Safety System Status\n";
    report += "=" + "=".repeat(40) + "\n\n";

    report += "✅ Runtime unit validation functions\n";
    report += "✅ Safe conversion helpers with overflow protection\n";
    report += "✅ Reasonable range checking for fees and basefees\n";
    report += "✅ Clear error messages for debugging\n";
    report += "✅ Unit mismatch detection and diagnosis\n";
    report += "✅ Comprehensive fee calculation validation\n";
    report += "✅ L1 cost calculation validation\n";
    report += "✅ Custom error types for unit issues\n\n";

    report += "Configured ranges:\n";
    report += `  - Fee range: ${MIN_REASONABLE_FEE_GWEI}-${MAX_REASONABLE_FEE_GWEI} gwei\n`;
    report += `  - Basefee range: ${MIN_REASONABLE_BASEFEE_GWEI}-${MAX_REASONABLE_BASEFEE_GWEI} gwei\n`;

    return report;
}

/**
 * Quick test for unit conversion correctness
 * @returns {boolean} True if all tests pass
 */
export function runQuickUnitTests() {
    try {
        // Test basic conversions
        const testGwei = 50.0;
        const testWei = gweiToWei(testGwei);
        const backToGwei = weiToGwei(testWei);

        if (Math.abs(backToGwei - testGwei) > 1e-6) {
            console.error("Gwei <-> Wei conversion failed");
            return false;
        }

        // Test fee range validation
        validateFeeRange(1.0, "test");  // Should pass

        try {
            validateFeeRange(0.0001, "test");  // Should fail
            console.error("Fee range validation failed to catch too-small fee");
            return false;
        } catch (e) {
            if (!(e instanceof UnitValidationError)) {
                console.error("Unexpected error type in fee validation");
                return false;
            }
        }

        // Test L1 cost calculation
        const l1Cost = validateL1CostCalculation(50e9, 20000, "test");  // 50 gwei, 20k gas
        if (l1Cost < 1e-6 || l1Cost > 1e-3) {
            console.error("L1 cost calculation produced unreasonable result:", l1Cost);
            return false;
        }

        console.log("✅ JavaScript unit validation tests passed");
        return true;

    } catch (error) {
        console.error("Unit validation test failed:", error.message);
        return false;
    }
}

// Export all validation functions
export default {
    // Constants
    WEI_PER_GWEI,
    WEI_PER_ETH,
    GWEI_PER_ETH,
    MIN_REASONABLE_FEE_GWEI,
    MAX_REASONABLE_FEE_GWEI,
    MIN_REASONABLE_BASEFEE_GWEI,
    MAX_REASONABLE_BASEFEE_GWEI,

    // Error classes
    UnitValidationError,
    UnitOverflowError,

    // Core validation
    validateNumeric,

    // Conversion functions
    gweiToWei,
    weiToGwei,
    ethToWei,
    weiToEth,

    // Range validation
    validateFeeRange,
    validateBasefeeRange,
    assertNoUnitMismatch,
    assertReasonableFee,

    // Comprehensive validation
    validateL1CostCalculation,
    validateFeeCalculation,

    // Debug helpers
    diagnoseUnitMismatch,
    createUnitSafetyReport,
    runQuickUnitTests
};