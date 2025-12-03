/**
 * Stakeholder-Specific Optimization Profiles (JavaScript)
 *
 * This module implements the stakeholder profiles from SUMMARY.md Section 2 for
 * cross-platform optimization. It mirrors the Python stakeholder_profiles.py
 * to ensure identical results between Python and JavaScript implementations.
 */

// Stakeholder types enum
export const StakeholderType = {
    END_USER: "end_user",
    PROTOCOL_DAO: "protocol_dao",
    VAULT_OPERATOR: "vault_operator",
    SEQUENCER: "sequencer",
    CRISIS_MANAGER: "crisis_manager"
};

/**
 * Objective weights implementing SUMMARY.md Section 2.3 theoretical framework.
 *
 * UX Objective (Section 2.3.1):
 *     J_UX(θ) = a1 × CV_F(θ) + a2 × J_ΔF(θ) + a3 × max(0, F95(θ) - F_UX_cap)
 *
 * Safety Objective (Section 2.3.2):
 *     J_safe(θ) = b1 × DD(θ) + b2 × D_max(θ) + b3 × RecoveryTime(θ)
 *
 * Efficiency Objective (Section 2.3.3):
 *     J_eff(θ) = c1 × T + c2 × E[|V(t)-T|] + c3 × CapEff(θ)
 */
export class ObjectiveWeights {
    constructor({
        // UX weights (Section 2.3.1) - higher values = more importance
        a1_fee_stability,        // CV_F(θ) - Coefficient of variation of fees
        a2_fee_jumpiness,        // J_ΔF(θ) - 95th percentile of relative fee jumps
        a3_high_fee_penalty,     // max(0, F95(θ) - F_UX_cap) - High fee penalty

        // Safety weights (Section 2.3.2) - higher values = more importance
        b1_deficit_duration,     // DD(θ) - Deficit-weighted duration Σ(T-V(t))₊
        b2_max_deficit_depth,    // D_max(θ) - Maximum deficit depth
        b3_recovery_time,        // RecoveryTime(θ) - Speed of recovery after shock

        // Efficiency weights (Section 2.3.3) - higher values = more importance
        c1_capital_cost,         // T - Target vault size (capital cost)
        c2_vault_deviation,      // E[|V(t)-T|] - Average vault deviation
        c3_capital_efficiency,   // CapEff(θ) - Capital per throughput ratio

        // Constraint parameters
        fee_tolerance_gwei       // F_UX_cap - Maximum acceptable fee for this stakeholder
    }) {
        this.a1_fee_stability = a1_fee_stability;
        this.a2_fee_jumpiness = a2_fee_jumpiness;
        this.a3_high_fee_penalty = a3_high_fee_penalty;

        this.b1_deficit_duration = b1_deficit_duration;
        this.b2_max_deficit_depth = b2_max_deficit_depth;
        this.b3_recovery_time = b3_recovery_time;

        this.c1_capital_cost = c1_capital_cost;
        this.c2_vault_deviation = c2_vault_deviation;
        this.c3_capital_efficiency = c3_capital_efficiency;

        this.fee_tolerance_gwei = fee_tolerance_gwei;
    }

    /**
     * Get weights normalized within each category to sum to 1.0.
     * @returns {Object} Normalized weights and category totals
     */
    getNormalizedWeights() {
        // UX weights
        const uxSum = this.a1_fee_stability + this.a2_fee_jumpiness + this.a3_high_fee_penalty;
        const uxNorm = Math.max(uxSum, 1e-6);  // Prevent division by zero

        // Safety weights
        const safetySum = this.b1_deficit_duration + this.b2_max_deficit_depth + this.b3_recovery_time;
        const safetyNorm = Math.max(safetySum, 1e-6);

        // Efficiency weights
        const effSum = this.c1_capital_cost + this.c2_vault_deviation + this.c3_capital_efficiency;
        const effNorm = Math.max(effSum, 1e-6);

        return {
            // UX weights (normalized)
            a1_fee_stability: this.a1_fee_stability / uxNorm,
            a2_fee_jumpiness: this.a2_fee_jumpiness / uxNorm,
            a3_high_fee_penalty: this.a3_high_fee_penalty / uxNorm,

            // Safety weights (normalized)
            b1_deficit_duration: this.b1_deficit_duration / safetyNorm,
            b2_max_deficit_depth: this.b2_max_deficit_depth / safetyNorm,
            b3_recovery_time: this.b3_recovery_time / safetyNorm,

            // Efficiency weights (normalized)
            c1_capital_cost: this.c1_capital_cost / effNorm,
            c2_vault_deviation: this.c2_vault_deviation / effNorm,
            c3_capital_efficiency: this.c3_capital_efficiency / effNorm,

            // Raw values for reference
            ux_total_weight: uxSum,
            safety_total_weight: safetySum,
            efficiency_total_weight: effSum
        };
    }
}

/**
 * Complete stakeholder profile with weights and constraints.
 */
export class StakeholderProfile {
    constructor({
        name,
        stakeholder_type,
        description,
        objectives,
        crr_tolerance = 0.05,           // ε_CRR - Cost recovery ratio tolerance (±5%)
        max_ruin_probability = 0.01,    // ε_ruin - Maximum acceptable ruin probability (1%)
        risk_tolerance = 1.0,           // Risk tolerance multiplier
        time_horizon_preference = "medium"  // short/medium/long planning horizon
    }) {
        this.name = name;
        this.stakeholder_type = stakeholder_type;
        this.description = description;
        this.objectives = objectives;
        this.crr_tolerance = crr_tolerance;
        this.max_ruin_probability = max_ruin_probability;
        this.risk_tolerance = risk_tolerance;
        this.time_horizon_preference = time_horizon_preference;
    }
}

// Stakeholder profile definitions (mirrors Python implementation)
export const STAKEHOLDER_PROFILES = {

    [StakeholderType.END_USER]: new StakeholderProfile({
        name: "End User",
        stakeholder_type: StakeholderType.END_USER,
        description: "Users prioritize low, stable, predictable transaction fees above all else",
        objectives: new ObjectiveWeights({
            // UX: Maximum focus - users care most about fees
            a1_fee_stability: 3.0,      // Strong preference for stable fees
            a2_fee_jumpiness: 2.0,      // Hate sudden fee spikes
            a3_high_fee_penalty: 5.0,   // Very strong aversion to high fees

            // Safety: Minimal concern - users don't care about protocol internals
            b1_deficit_duration: 0.5,   // Limited protocol concern
            b2_max_deficit_depth: 0.5,  // Limited protocol concern
            b3_recovery_time: 0.3,      // Limited protocol concern

            // Efficiency: No concern - users don't care about capital efficiency
            c1_capital_cost: 0.1,       // Don't care about protocol capital costs
            c2_vault_deviation: 0.1,    // Don't care about vault management
            c3_capital_efficiency: 0.1, // Don't care about capital efficiency

            fee_tolerance_gwei: 20.0    // Low tolerance for high fees
        }),
        crr_tolerance: 0.1,             // More lenient on cost recovery
        max_ruin_probability: 0.05,     // More tolerant of protocol risk
        risk_tolerance: 0.5,            // Low risk tolerance
        time_horizon_preference: "short"
    }),

    [StakeholderType.PROTOCOL_DAO]: new StakeholderProfile({
        name: "Protocol DAO",
        stakeholder_type: StakeholderType.PROTOCOL_DAO,
        description: "Governance body balancing all stakeholder interests equally",
        objectives: new ObjectiveWeights({
            // UX: Balanced consideration for user experience
            a1_fee_stability: 1.0,      // Moderate stability preference
            a2_fee_jumpiness: 1.0,      // Moderate jumpiness concern
            a3_high_fee_penalty: 1.0,   // Moderate high fee concern

            // Safety: Balanced protocol safety focus
            b1_deficit_duration: 1.0,   // Balanced deficit concern
            b2_max_deficit_depth: 1.0,  // Balanced depth concern
            b3_recovery_time: 1.0,      // Balanced recovery concern

            // Efficiency: Balanced capital efficiency focus
            c1_capital_cost: 1.0,       // Moderate capital cost concern
            c2_vault_deviation: 1.0,    // Moderate vault management concern
            c3_capital_efficiency: 1.0, // Moderate efficiency concern

            fee_tolerance_gwei: 100.0   // Moderate fee tolerance
        }),
        crr_tolerance: 0.05,            // Standard cost recovery requirement
        max_ruin_probability: 0.01,     // Standard safety requirement
        risk_tolerance: 1.0,            // Balanced risk tolerance
        time_horizon_preference: "medium"
    }),

    [StakeholderType.VAULT_OPERATOR]: new StakeholderProfile({
        name: "Vault Operator",
        stakeholder_type: StakeholderType.VAULT_OPERATOR,
        description: "Vault operators prioritize capital efficiency and returns over user experience",
        objectives: new ObjectiveWeights({
            // UX: Lower priority - revenue stability more important than UX
            a1_fee_stability: 0.5,      // Some stability preference for revenue predictability
            a2_fee_jumpiness: 0.3,      // Less concern about fee volatility
            a3_high_fee_penalty: 0.3,   // Higher fees mean more revenue

            // Safety: High priority - vault safety directly affects operators
            b1_deficit_duration: 1.0,   // Care about vault performance
            b2_max_deficit_depth: 1.5,  // Strong concern about vault safety
            b3_recovery_time: 1.0,      // Care about recovery speed

            // Efficiency: Maximum priority - want maximum return on capital
            c1_capital_cost: 3.0,       // Minimize capital requirements
            c2_vault_deviation: 2.0,    // Efficient vault utilization
            c3_capital_efficiency: 3.0, // Maximize capital efficiency

            fee_tolerance_gwei: 200.0   // High tolerance - higher fees = more revenue
        }),
        crr_tolerance: 0.03,            // Stricter cost recovery requirement
        max_ruin_probability: 0.005,    // Lower tolerance for vault risk
        risk_tolerance: 1.5,            // Higher risk tolerance for returns
        time_horizon_preference: "long"
    }),

    [StakeholderType.SEQUENCER]: new StakeholderProfile({
        name: "Sequencer",
        stakeholder_type: StakeholderType.SEQUENCER,
        description: "Sequencers need predictable base fees for priority fee optimization",
        objectives: new ObjectiveWeights({
            // UX: High stability focus - need predictable revenue streams
            a1_fee_stability: 2.0,      // Want predictable base fee for planning
            a2_fee_jumpiness: 1.0,      // Moderate concern about sudden changes
            a3_high_fee_penalty: 0.5,   // Higher fees enable higher priority fees

            // Safety: Moderate focus - need protocol stability for business
            b1_deficit_duration: 1.5,   // Care about protocol health
            b2_max_deficit_depth: 1.5,  // Care about protocol stability
            b3_recovery_time: 1.0,      // Care about system resilience

            // Efficiency: High focus - revenue efficiency important
            c1_capital_cost: 1.0,       // Moderate capital concern
            c2_vault_deviation: 1.5,    // Want efficient vault operation
            c3_capital_efficiency: 2.0, // Want revenue efficiency

            fee_tolerance_gwei: 150.0   // Moderate-high tolerance
        }),
        crr_tolerance: 0.05,            // Standard cost recovery requirement
        max_ruin_probability: 0.01,     // Standard safety requirement
        risk_tolerance: 1.2,            // Moderate-high risk tolerance
        time_horizon_preference: "medium"
    }),

    [StakeholderType.CRISIS_MANAGER]: new StakeholderProfile({
        name: "Crisis Manager",
        stakeholder_type: StakeholderType.CRISIS_MANAGER,
        description: "Crisis response prioritizes maximum protocol robustness over user experience",
        objectives: new ObjectiveWeights({
            // UX: Minimal priority - protocol survival trumps user experience in crisis
            a1_fee_stability: 1.0,      // Some stability preference
            a2_fee_jumpiness: 0.5,      // Accept volatility for robustness
            a3_high_fee_penalty: 0.2,   // Accept very high fees in crisis

            // Safety: Maximum priority - protocol survival is paramount
            b1_deficit_duration: 3.0,   // Maximum deficit control
            b2_max_deficit_depth: 3.0,  // Maximum safety margins
            b3_recovery_time: 3.0,      // Maximum recovery speed

            // Efficiency: Lower priority - accept capital costs for safety
            c1_capital_cost: 0.5,       // Accept higher capital for safety
            c2_vault_deviation: 0.5,    // Accept vault inefficiency for safety
            c3_capital_efficiency: 0.5, // Accept capital inefficiency

            fee_tolerance_gwei: 500.0   // Very high tolerance in crisis
        }),
        crr_tolerance: 0.02,            // Very strict cost recovery in crisis
        max_ruin_probability: 0.001,    // Very low tolerance for failure
        risk_tolerance: 2.0,            // High risk tolerance for extreme measures
        time_horizon_preference: "short"
    })
};

/**
 * Get stakeholder profile by type.
 * @param {StakeholderType} stakeholder_type - The stakeholder type
 * @returns {StakeholderProfile} The stakeholder profile
 */
export function getStakeholderProfile(stakeholder_type) {
    if (!(stakeholder_type in STAKEHOLDER_PROFILES)) {
        throw new Error(`Unknown stakeholder type: ${stakeholder_type}`);
    }
    return STAKEHOLDER_PROFILES[stakeholder_type];
}

/**
 * List all available stakeholder types.
 * @returns {Array<StakeholderType>} Array of stakeholder types
 */
export function listStakeholderTypes() {
    return Object.values(StakeholderType);
}

/**
 * Get summary of all stakeholder priorities.
 * @returns {Object} Map of stakeholder names to descriptions
 */
export function getStakeholderSummary() {
    const summary = {};
    for (const profile of Object.values(STAKEHOLDER_PROFILES)) {
        summary[profile.name] = profile.description;
    }
    return summary;
}

/**
 * Validate that stakeholder weights are reasonable.
 * @param {StakeholderProfile} profile - Profile to validate
 * @returns {Array<string>} Array of validation issues (empty if valid)
 */
export function validateStakeholderWeights(profile) {
    const issues = [];
    const weights = profile.objectives;

    // Check for negative weights
    const weightFields = [
        'a1_fee_stability', 'a2_fee_jumpiness', 'a3_high_fee_penalty',
        'b1_deficit_duration', 'b2_max_deficit_depth', 'b3_recovery_time',
        'c1_capital_cost', 'c2_vault_deviation', 'c3_capital_efficiency'
    ];

    for (const field of weightFields) {
        if (weights[field] < 0) {
            issues.push(`Negative weight detected: ${field} = ${weights[field]}`);
        }
    }

    // Check for all-zero categories
    const uxSum = weights.a1_fee_stability + weights.a2_fee_jumpiness + weights.a3_high_fee_penalty;
    const safetySum = weights.b1_deficit_duration + weights.b2_max_deficit_depth + weights.b3_recovery_time;
    const effSum = weights.c1_capital_cost + weights.c2_vault_deviation + weights.c3_capital_efficiency;

    if (uxSum <= 0) {
        issues.push("UX weights sum to zero - stakeholder needs some UX consideration");
    }
    if (safetySum <= 0) {
        issues.push("Safety weights sum to zero - stakeholder needs some safety consideration");
    }
    if (effSum <= 0) {
        issues.push("Efficiency weights sum to zero - stakeholder needs some efficiency consideration");
    }

    // Check constraint reasonableness
    if (weights.fee_tolerance_gwei <= 0) {
        issues.push("Fee tolerance must be positive");
    }
    if (profile.crr_tolerance <= 0 || profile.crr_tolerance >= 0.5) {
        issues.push("CRR tolerance should be between 0 and 50%");
    }
    if (profile.max_ruin_probability <= 0 || profile.max_ruin_probability >= 0.1) {
        issues.push("Max ruin probability should be between 0 and 10%");
    }

    return issues;
}

/**
 * Demonstrate how stakeholder profiles differ in practice.
 */
export function demonstrateStakeholderDifferences() {
    console.log("=== STAKEHOLDER PROFILE COMPARISON ===");

    for (const [stakeholder_type, profile] of Object.entries(STAKEHOLDER_PROFILES)) {
        console.log(`\n${profile.name}:`);
        console.log(`  Description: ${profile.description}`);

        const weights = profile.objectives.getNormalizedWeights();
        console.log(`  UX Focus: ${weights.ux_total_weight.toFixed(1)} (stability=${weights.a1_fee_stability.toFixed(2)}, jumps=${weights.a2_fee_jumpiness.toFixed(2)}, high_fees=${weights.a3_high_fee_penalty.toFixed(2)})`);
        console.log(`  Safety Focus: ${weights.safety_total_weight.toFixed(1)} (duration=${weights.b1_deficit_duration.toFixed(2)}, depth=${weights.b2_max_deficit_depth.toFixed(2)}, recovery=${weights.b3_recovery_time.toFixed(2)})`);
        console.log(`  Efficiency Focus: ${weights.efficiency_total_weight.toFixed(1)} (capital=${weights.c1_capital_cost.toFixed(2)}, deviation=${weights.c2_vault_deviation.toFixed(2)}, efficiency=${weights.c3_capital_efficiency.toFixed(2)})`);
        console.log(`  Fee Tolerance: ${profile.objectives.fee_tolerance_gwei.toFixed(0)} gwei`);
        console.log(`  Risk Profile: ${profile.risk_tolerance.toFixed(1)}x, CRR±${(profile.crr_tolerance * 100).toFixed(1)}%, Ruin<${(profile.max_ruin_probability * 100).toFixed(1)}%`);
    }
}

// Export for testing
export { STAKEHOLDER_PROFILES };