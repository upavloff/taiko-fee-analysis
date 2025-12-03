"""
Stakeholder-Specific Optimization Profiles Implementation

This module implements the stakeholder profiles from SUMMARY.md Section 2 for
production-ready fee mechanism optimization. Each profile defines different
weights for the three objective categories: UX, Safety, and Efficiency.

Based on real stakeholder priorities from protocol governance discussions.
"""

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


class StakeholderType(Enum):
    """Stakeholder categories for fee mechanism optimization."""
    END_USER = "end_user"
    PROTOCOL_DAO = "protocol_dao"
    VAULT_OPERATOR = "vault_operator"
    SEQUENCER = "sequencer"
    CRISIS_MANAGER = "crisis_manager"


@dataclass
class ObjectiveWeights:
    """
    Objective weights implementing SUMMARY.md Section 2.3 theoretical framework.

    UX Objective (Section 2.3.1):
        J_UX(θ) = a1 × CV_F(θ) + a2 × J_ΔF(θ) + a3 × max(0, F95(θ) - F_UX_cap)

    Safety Objective (Section 2.3.2):
        J_safe(θ) = b1 × DD(θ) + b2 × D_max(θ) + b3 × RecoveryTime(θ)

    Efficiency Objective (Section 2.3.3):
        J_eff(θ) = c1 × T + c2 × E[|V(t)-T|] + c3 × CapEff(θ)
    """

    # UX weights (Section 2.3.1) - higher values = more importance
    a1_fee_stability: float        # CV_F(θ) - Coefficient of variation of fees
    a2_fee_jumpiness: float        # J_ΔF(θ) - 95th percentile of relative fee jumps
    a3_high_fee_penalty: float     # max(0, F95(θ) - F_UX_cap) - High fee penalty

    # Safety weights (Section 2.3.2) - higher values = more importance
    b1_deficit_duration: float     # DD(θ) - Deficit-weighted duration Σ(T-V(t))₊
    b2_max_deficit_depth: float    # D_max(θ) - Maximum deficit depth
    b3_recovery_time: float        # RecoveryTime(θ) - Speed of recovery after shock

    # Efficiency weights (Section 2.3.3) - higher values = more importance
    c1_capital_cost: float         # T - Target vault size (capital cost)
    c2_vault_deviation: float     # E[|V(t)-T|] - Average vault deviation
    c3_capital_efficiency: float  # CapEff(θ) - Capital per throughput ratio

    # Constraint parameters
    fee_tolerance_gwei: float      # F_UX_cap - Maximum acceptable fee for this stakeholder

    def get_normalized_weights(self) -> Dict[str, float]:
        """Get weights normalized within each category to sum to 1.0."""

        # UX weights
        ux_sum = self.a1_fee_stability + self.a2_fee_jumpiness + self.a3_high_fee_penalty
        ux_norm = max(ux_sum, 1e-6)  # Prevent division by zero

        # Safety weights
        safety_sum = self.b1_deficit_duration + self.b2_max_deficit_depth + self.b3_recovery_time
        safety_norm = max(safety_sum, 1e-6)

        # Efficiency weights
        eff_sum = self.c1_capital_cost + self.c2_vault_deviation + self.c3_capital_efficiency
        eff_norm = max(eff_sum, 1e-6)

        return {
            # UX weights (normalized)
            'a1_fee_stability': self.a1_fee_stability / ux_norm,
            'a2_fee_jumpiness': self.a2_fee_jumpiness / ux_norm,
            'a3_high_fee_penalty': self.a3_high_fee_penalty / ux_norm,

            # Safety weights (normalized)
            'b1_deficit_duration': self.b1_deficit_duration / safety_norm,
            'b2_max_deficit_depth': self.b2_max_deficit_depth / safety_norm,
            'b3_recovery_time': self.b3_recovery_time / safety_norm,

            # Efficiency weights (normalized)
            'c1_capital_cost': self.c1_capital_cost / eff_norm,
            'c2_vault_deviation': self.c2_vault_deviation / eff_norm,
            'c3_capital_efficiency': self.c3_capital_efficiency / eff_norm,

            # Raw values for reference
            'ux_total_weight': ux_sum,
            'safety_total_weight': safety_sum,
            'efficiency_total_weight': eff_sum
        }


@dataclass
class StakeholderProfile:
    """Complete stakeholder profile with weights and constraints."""

    name: str
    stakeholder_type: StakeholderType
    description: str
    objectives: ObjectiveWeights

    # Hard constraint parameters (Section 2.1-2.2)
    crr_tolerance: float = 0.05      # ε_CRR - Cost recovery ratio tolerance (±5%)
    max_ruin_probability: float = 0.01  # ε_ruin - Maximum acceptable ruin probability (1%)

    # Additional stakeholder-specific parameters
    risk_tolerance: float = 1.0      # Risk tolerance multiplier
    time_horizon_preference: str = "medium"  # short/medium/long planning horizon


# Stakeholder profile definitions based on real protocol governance priorities
STAKEHOLDER_PROFILES: Dict[StakeholderType, StakeholderProfile] = {

    StakeholderType.END_USER: StakeholderProfile(
        name="End User",
        stakeholder_type=StakeholderType.END_USER,
        description="Users prioritize low, stable, predictable transaction fees above all else",
        objectives=ObjectiveWeights(
            # UX: Maximum focus - users care most about fees
            a1_fee_stability=3.0,      # Strong preference for stable fees
            a2_fee_jumpiness=2.0,      # Hate sudden fee spikes
            a3_high_fee_penalty=5.0,   # Very strong aversion to high fees

            # Safety: Minimal concern - users don't care about protocol internals
            b1_deficit_duration=0.5,   # Limited protocol concern
            b2_max_deficit_depth=0.5,  # Limited protocol concern
            b3_recovery_time=0.3,      # Limited protocol concern

            # Efficiency: No concern - users don't care about capital efficiency
            c1_capital_cost=0.1,       # Don't care about protocol capital costs
            c2_vault_deviation=0.1,    # Don't care about vault management
            c3_capital_efficiency=0.1, # Don't care about capital efficiency

            fee_tolerance_gwei=20.0    # Low tolerance for high fees
        ),
        crr_tolerance=0.1,             # More lenient on cost recovery
        max_ruin_probability=0.05,     # More tolerant of protocol risk
        risk_tolerance=0.5,            # Low risk tolerance
        time_horizon_preference="short"
    ),

    StakeholderType.PROTOCOL_DAO: StakeholderProfile(
        name="Protocol DAO",
        stakeholder_type=StakeholderType.PROTOCOL_DAO,
        description="Governance body balancing all stakeholder interests equally",
        objectives=ObjectiveWeights(
            # UX: Balanced consideration for user experience
            a1_fee_stability=1.0,      # Moderate stability preference
            a2_fee_jumpiness=1.0,      # Moderate jumpiness concern
            a3_high_fee_penalty=1.0,   # Moderate high fee concern

            # Safety: Balanced protocol safety focus
            b1_deficit_duration=1.0,   # Balanced deficit concern
            b2_max_deficit_depth=1.0,  # Balanced depth concern
            b3_recovery_time=1.0,      # Balanced recovery concern

            # Efficiency: Balanced capital efficiency focus
            c1_capital_cost=1.0,       # Moderate capital cost concern
            c2_vault_deviation=1.0,    # Moderate vault management concern
            c3_capital_efficiency=1.0, # Moderate efficiency concern

            fee_tolerance_gwei=100.0   # Moderate fee tolerance
        ),
        crr_tolerance=0.05,            # Standard cost recovery requirement
        max_ruin_probability=0.01,     # Standard safety requirement
        risk_tolerance=1.0,            # Balanced risk tolerance
        time_horizon_preference="medium"
    ),

    StakeholderType.VAULT_OPERATOR: StakeholderProfile(
        name="Vault Operator",
        stakeholder_type=StakeholderType.VAULT_OPERATOR,
        description="Vault operators prioritize capital efficiency and returns over user experience",
        objectives=ObjectiveWeights(
            # UX: Lower priority - revenue stability more important than UX
            a1_fee_stability=0.5,      # Some stability preference for revenue predictability
            a2_fee_jumpiness=0.3,      # Less concern about fee volatility
            a3_high_fee_penalty=0.3,   # Higher fees mean more revenue

            # Safety: High priority - vault safety directly affects operators
            b1_deficit_duration=1.0,   # Care about vault performance
            b2_max_deficit_depth=1.5,  # Strong concern about vault safety
            b3_recovery_time=1.0,      # Care about recovery speed

            # Efficiency: Maximum priority - want maximum return on capital
            c1_capital_cost=3.0,       # Minimize capital requirements
            c2_vault_deviation=2.0,    # Efficient vault utilization
            c3_capital_efficiency=3.0, # Maximize capital efficiency

            fee_tolerance_gwei=200.0   # High tolerance - higher fees = more revenue
        ),
        crr_tolerance=0.03,            # Stricter cost recovery requirement
        max_ruin_probability=0.005,    # Lower tolerance for vault risk
        risk_tolerance=1.5,            # Higher risk tolerance for returns
        time_horizon_preference="long"
    ),

    StakeholderType.SEQUENCER: StakeholderProfile(
        name="Sequencer",
        stakeholder_type=StakeholderType.SEQUENCER,
        description="Sequencers need predictable base fees for priority fee optimization",
        objectives=ObjectiveWeights(
            # UX: High stability focus - need predictable revenue streams
            a1_fee_stability=2.0,      # Want predictable base fee for planning
            a2_fee_jumpiness=1.0,      # Moderate concern about sudden changes
            a3_high_fee_penalty=0.5,   # Higher fees enable higher priority fees

            # Safety: Moderate focus - need protocol stability for business
            b1_deficit_duration=1.5,   # Care about protocol health
            b2_max_deficit_depth=1.5,  # Care about protocol stability
            b3_recovery_time=1.0,      # Care about system resilience

            # Efficiency: High focus - revenue efficiency important
            c1_capital_cost=1.0,       # Moderate capital concern
            c2_vault_deviation=1.5,    # Want efficient vault operation
            c3_capital_efficiency=2.0, # Want revenue efficiency

            fee_tolerance_gwei=150.0   # Moderate-high tolerance
        ),
        crr_tolerance=0.05,            # Standard cost recovery requirement
        max_ruin_probability=0.01,     # Standard safety requirement
        risk_tolerance=1.2,            # Moderate-high risk tolerance
        time_horizon_preference="medium"
    ),

    StakeholderType.CRISIS_MANAGER: StakeholderProfile(
        name="Crisis Manager",
        stakeholder_type=StakeholderType.CRISIS_MANAGER,
        description="Crisis response prioritizes maximum protocol robustness over user experience",
        objectives=ObjectiveWeights(
            # UX: Minimal priority - protocol survival trumps user experience in crisis
            a1_fee_stability=1.0,      # Some stability preference
            a2_fee_jumpiness=0.5,      # Accept volatility for robustness
            a3_high_fee_penalty=0.2,   # Accept very high fees in crisis

            # Safety: Maximum priority - protocol survival is paramount
            b1_deficit_duration=3.0,   # Maximum deficit control
            b2_max_deficit_depth=3.0,  # Maximum safety margins
            b3_recovery_time=3.0,      # Maximum recovery speed

            # Efficiency: Lower priority - accept capital costs for safety
            c1_capital_cost=0.5,       # Accept higher capital for safety
            c2_vault_deviation=0.5,    # Accept vault inefficiency for safety
            c3_capital_efficiency=0.5, # Accept capital inefficiency

            fee_tolerance_gwei=500.0   # Very high tolerance in crisis
        ),
        crr_tolerance=0.02,            # Very strict cost recovery in crisis
        max_ruin_probability=0.001,    # Very low tolerance for failure
        risk_tolerance=2.0,            # High risk tolerance for extreme measures
        time_horizon_preference="short"
    )
}


def get_stakeholder_profile(stakeholder_type: StakeholderType) -> StakeholderProfile:
    """Get stakeholder profile by type."""
    return STAKEHOLDER_PROFILES[stakeholder_type]


def list_stakeholder_types() -> List[StakeholderType]:
    """List all available stakeholder types."""
    return list(StakeholderType)


def get_stakeholder_summary() -> Dict[str, str]:
    """Get summary of all stakeholder priorities."""
    return {
        stakeholder.name: stakeholder.description
        for stakeholder in STAKEHOLDER_PROFILES.values()
    }


def validate_stakeholder_weights(profile: StakeholderProfile) -> List[str]:
    """Validate that stakeholder weights are reasonable."""
    issues = []
    weights = profile.objectives

    # Check for negative weights
    if any(getattr(weights, attr) < 0 for attr in weights.__dataclass_fields__):
        issues.append("Negative weights detected")

    # Check for all-zero categories
    ux_sum = weights.a1_fee_stability + weights.a2_fee_jumpiness + weights.a3_high_fee_penalty
    safety_sum = weights.b1_deficit_duration + weights.b2_max_deficit_depth + weights.b3_recovery_time
    eff_sum = weights.c1_capital_cost + weights.c2_vault_deviation + weights.c3_capital_efficiency

    if ux_sum <= 0:
        issues.append("UX weights sum to zero - stakeholder needs some UX consideration")
    if safety_sum <= 0:
        issues.append("Safety weights sum to zero - stakeholder needs some safety consideration")
    if eff_sum <= 0:
        issues.append("Efficiency weights sum to zero - stakeholder needs some efficiency consideration")

    # Check constraint reasonableness
    if profile.objectives.fee_tolerance_gwei <= 0:
        issues.append("Fee tolerance must be positive")
    if not (0 < profile.crr_tolerance < 0.5):
        issues.append("CRR tolerance should be between 0 and 50%")
    if not (0 < profile.max_ruin_probability < 0.1):
        issues.append("Max ruin probability should be between 0 and 10%")

    return issues


def demonstrate_stakeholder_differences():
    """Demonstrate how stakeholder profiles differ in practice."""
    print("=== STAKEHOLDER PROFILE COMPARISON ===")

    for stakeholder_type, profile in STAKEHOLDER_PROFILES.items():
        print(f"\n{profile.name}:")
        print(f"  Description: {profile.description}")

        weights = profile.objectives.get_normalized_weights()
        print(f"  UX Focus: {weights['ux_total_weight']:.1f} (stability={weights['a1_fee_stability']:.2f}, jumps={weights['a2_fee_jumpiness']:.2f}, high_fees={weights['a3_high_fee_penalty']:.2f})")
        print(f"  Safety Focus: {weights['safety_total_weight']:.1f} (duration={weights['b1_deficit_duration']:.2f}, depth={weights['b2_max_deficit_depth']:.2f}, recovery={weights['b3_recovery_time']:.2f})")
        print(f"  Efficiency Focus: {weights['efficiency_total_weight']:.1f} (capital={weights['c1_capital_cost']:.2f}, deviation={weights['c2_vault_deviation']:.2f}, efficiency={weights['c3_capital_efficiency']:.2f})")
        print(f"  Fee Tolerance: {profile.objectives.fee_tolerance_gwei:.0f} gwei")
        print(f"  Risk Profile: {profile.risk_tolerance:.1f}x, CRR±{profile.crr_tolerance*100:.1f}%, Ruin<{profile.max_ruin_probability*100:.1f}%")


if __name__ == "__main__":
    demonstrate_stakeholder_differences()