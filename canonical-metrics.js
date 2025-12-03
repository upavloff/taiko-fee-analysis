/**
 * Canonical Taiko Fee Mechanism Metrics (JavaScript)
 *
 * This is the SINGLE SOURCE OF TRUTH for all performance metrics calculations in JavaScript.
 * It mirrors the Python canonical implementation to ensure consistency.
 *
 * Key Features:
 * - Comprehensive performance scoring system
 * - User experience, protocol safety, and economic efficiency metrics
 * - Threshold-based grading system
 * - Statistical analysis of simulation results
 */

// Comprehensive metrics result class
export class ComprehensiveMetrics {
    constructor({
        userExperienceScore = 0.0,
        protocolSafetyScore = 0.0,
        economicEfficiencyScore = 0.0,
        overallPerformanceScore = 0.0,
        detailedMetrics = {}
    } = {}) {
        this.userExperienceScore = userExperienceScore;
        this.protocolSafetyScore = protocolSafetyScore;
        this.economicEfficiencyScore = economicEfficiencyScore;
        this.overallPerformanceScore = overallPerformanceScore;
        this.detailedMetrics = detailedMetrics;
    }
}

// Canonical metrics calculator class
export class CanonicalMetricsCalculator {
    constructor() {
        this.weights = {
            userExperience: 0.4,
            protocolSafety: 0.35,
            economicEfficiency: 0.25
        };
    }

    /**
     * Calculate comprehensive metrics from simulation results
     * @param {Object} simulationResults - Results from fee mechanism simulation
     * @returns {ComprehensiveMetrics} Comprehensive performance metrics
     */
    calculateComprehensiveMetrics(simulationResults) {
        // Calculate basic metrics first
        const basicMetrics = calculateBasicMetrics(simulationResults);

        // Calculate component scores
        const userExperience = this._calculateUserExperienceScore(basicMetrics, simulationResults);
        const protocolSafety = this._calculateProtocolSafetyScore(basicMetrics, simulationResults);
        const economicEfficiency = this._calculateEconomicEfficiencyScore(basicMetrics, simulationResults);

        // Calculate overall score
        const overallScore = (
            userExperience * this.weights.userExperience +
            protocolSafety * this.weights.protocolSafety +
            economicEfficiency * this.weights.economicEfficiency
        );

        return new ComprehensiveMetrics({
            userExperienceScore: userExperience,
            protocolSafetyScore: protocolSafety,
            economicEfficiencyScore: economicEfficiency,
            overallPerformanceScore: overallScore,
            detailedMetrics: basicMetrics
        });
    }

    _calculateUserExperienceScore(basicMetrics, simulationResults) {
        // Fee predictability (inverse of coefficient of variation)
        const feeStability = Math.max(0, 1 - basicMetrics.fee_stability_cv / 0.5);

        // Fee reasonableness (penalty for very high or very low fees)
        const avgFeeGwei = basicMetrics.average_fee_gwei;
        let feeReasonableness;
        if (avgFeeGwei >= 5 && avgFeeGwei <= 50) {
            feeReasonableness = 1.0;
        } else if (avgFeeGwei < 5) {
            feeReasonableness = avgFeeGwei / 5;
        } else {
            feeReasonableness = Math.max(0, 1 - (avgFeeGwei - 50) / 100);
        }

        return (feeStability * 0.6 + feeReasonableness * 0.4);
    }

    _calculateProtocolSafetyScore(basicMetrics, simulationResults) {
        // Vault health (minimize underfunded time)
        const vaultHealth = Math.max(0, 1 - basicMetrics.time_underfunded_pct / 100);

        // L1 cost coverage (how well fees track L1 costs)
        const trackingScore = Math.max(0, 1 - basicMetrics.l1_tracking_error / 2.0);

        // Revenue stability
        const feesCollected = simulationResults.feesCollected || [];
        const feeVariability = feesCollected.length > 1 ? this._calculateCV(feesCollected) : 0;
        const revenueStability = Math.max(0, 1 - feeVariability / 0.5);

        return (vaultHealth * 0.4 + trackingScore * 0.4 + revenueStability * 0.2);
    }

    _calculateEconomicEfficiencyScore(basicMetrics, simulationResults) {
        // Cost recovery efficiency
        const totalFees = this._sumArray(simulationResults.feesCollected || []);
        const totalCosts = this._sumArray(simulationResults.l1CostsPaid || []);
        const costCoverage = totalCosts > 0 ? Math.min(1.5, totalFees / totalCosts) / 1.5 : 0;

        // Transaction volume efficiency (higher is better, up to a point)
        const avgVolume = this._meanArray(simulationResults.transactionVolume || []);
        const volumeEfficiency = Math.min(1.0, avgVolume / 100); // Normalize to 100 TPS

        // Fee optimization (penalty for excessive fees)
        const feeOptimization = Math.max(0, 1 - Math.max(0, basicMetrics.average_fee_gwei - 30) / 50);

        return (costCoverage * 0.5 + volumeEfficiency * 0.3 + feeOptimization * 0.2);
    }

    _calculateCV(values) {
        if (values.length === 0) return 0;
        const mean = this._meanArray(values);
        if (mean === 0) return 0;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);
        return stdDev / mean;
    }

    _meanArray(arr) {
        if (arr.length === 0) return 0;
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    _sumArray(arr) {
        return arr.reduce((a, b) => a + b, 0);
    }
}

/**
 * Calculate basic metrics from simulation results
 * @param {Object} simulationResults - Simulation data
 * @returns {Object} Basic metrics object
 */
export function calculateBasicMetrics(simulationResults) {
    const fees = simulationResults.estimatedFee || [];
    const l1Basefees = simulationResults.l1Basefee || [];
    const vaultBalances = simulationResults.vaultBalance || [];
    const vaultDeficits = simulationResults.vaultDeficit || [];

    // Convert fees to gwei for analysis
    const feesGwei = fees.map(f => f * 1e9);

    // Average fee
    const averageFeeGwei = feesGwei.length > 0 ?
        feesGwei.reduce((a, b) => a + b, 0) / feesGwei.length : 0;

    // Fee stability (coefficient of variation)
    let feeStabilityCV = 0;
    if (feesGwei.length > 1) {
        const feeStdDev = Math.sqrt(
            feesGwei.reduce((sum, fee) => sum + Math.pow(fee - averageFeeGwei, 2), 0) / feesGwei.length
        );
        feeStabilityCV = averageFeeGwei > 0 ? feeStdDev / averageFeeGwei : 0;
    }

    // Time underfunded percentage
    const underfundedSteps = vaultDeficits.filter(d => d > 0).length;
    const timeUnderfundedPct = vaultDeficits.length > 0 ?
        (underfundedSteps / vaultDeficits.length) * 100 : 0;

    // L1 tracking error
    let l1TrackingError = 0;
    if (feesGwei.length > 0 && l1Basefees.length > 0) {
        const minLength = Math.min(feesGwei.length, l1Basefees.length);
        let sumSquaredError = 0;
        let sumL1Squared = 0;

        for (let i = 0; i < minLength; i++) {
            const error = feesGwei[i] - l1Basefees[i];
            sumSquaredError += error * error;
            sumL1Squared += l1Basefees[i] * l1Basefees[i];
        }

        l1TrackingError = sumL1Squared > 0 ?
            Math.sqrt(sumSquaredError / minLength) / Math.sqrt(sumL1Squared / minLength) : 0;
    }

    // Overall score (simple weighted combination)
    const normalizedFee = Math.max(0, 1 - Math.abs(averageFeeGwei - 20) / 50); // Target ~20 gwei
    const normalizedStability = Math.max(0, 1 - feeStabilityCV / 0.5);
    const normalizedUnderfunded = Math.max(0, 1 - timeUnderfundedPct / 50);
    const normalizedTracking = Math.max(0, 1 - l1TrackingError / 2);

    const overallScore = (
        normalizedFee * 0.25 +
        normalizedStability * 0.25 +
        normalizedUnderfunded * 0.25 +
        normalizedTracking * 0.25
    );

    return {
        average_fee_gwei: averageFeeGwei,
        fee_stability_cv: feeStabilityCV,
        time_underfunded_pct: timeUnderfundedPct,
        l1_tracking_error: l1TrackingError,
        overall_score: overallScore
    };
}

/**
 * Validate metrics against performance thresholds
 * @param {number} averageFeeGwei - Average fee in gwei
 * @param {number} feeCV - Fee coefficient of variation
 * @param {number} underfundedPct - Percentage of time underfunded
 * @param {number} trackingError - L1 tracking error
 * @returns {Object} Performance grades
 */
export function validateMetricThresholds(averageFeeGwei, feeCV, underfundedPct, trackingError) {
    const grades = {};

    // Fee level grading
    if (averageFeeGwei >= 5 && averageFeeGwei <= 30) {
        grades.average_fee = "excellent";
    } else if (averageFeeGwei >= 2 && averageFeeGwei <= 50) {
        grades.average_fee = "good";
    } else {
        grades.average_fee = "poor";
    }

    // Fee stability grading
    if (feeCV <= 0.2) {
        grades.fee_stability = "excellent";
    } else if (feeCV <= 0.5) {
        grades.fee_stability = "good";
    } else {
        grades.fee_stability = "poor";
    }

    // Underfunded time grading
    if (underfundedPct <= 10) {
        grades.vault_health = "excellent";
    } else if (underfundedPct <= 25) {
        grades.vault_health = "good";
    } else {
        grades.vault_health = "poor";
    }

    // L1 tracking grading
    if (trackingError <= 0.3) {
        grades.l1_tracking = "excellent";
    } else if (trackingError <= 0.8) {
        grades.l1_tracking = "good";
    } else {
        grades.l1_tracking = "poor";
    }

    return grades;
}

// Export classes and functions
export default {
    ComprehensiveMetrics,
    CanonicalMetricsCalculator,
    calculateBasicMetrics,
    validateMetricThresholds
};