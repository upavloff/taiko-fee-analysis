"""
Optimization Status and Recalibration Requirements

This module tracks the status of optimization results and provides
guidance for recalibration after the specification consolidation.
"""

import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .unified_fee_mechanism import (
    FeeParameters,
    ParameterCalibrationStatus,
    UnifiedFeeCalculator
)


class OptimizationValidityStatus(Enum):
    """Status of optimization results validity."""
    VALID = "valid"                      # Based on correct calibrated parameters
    INVALID_PARAMETERS = "invalid_parameters"  # Based on wrong α_data, Q̄
    INVALID_FORMULA = "invalid_formula"         # Based on deprecated formula
    UNCALIBRATED = "uncalibrated"              # Never properly optimized


@dataclass
class OptimizationResult:
    """Container for optimization results with validity tracking."""
    mu: float
    nu: float
    H: int
    lambda_B: float = 0.3

    # Validity tracking
    status: OptimizationValidityStatus = OptimizationValidityStatus.UNCALIBRATED
    description: str = "Uncalibrated parameters"
    based_on_alpha_data: Optional[float] = None
    based_on_Q_bar: Optional[float] = None
    optimization_date: Optional[str] = None

    def to_fee_parameters(self) -> FeeParameters:
        """Convert to FeeParameters with appropriate calibration status."""
        if self.status == OptimizationValidityStatus.VALID:
            optimization_status = ParameterCalibrationStatus.CALIBRATED
        else:
            optimization_status = ParameterCalibrationStatus.UNCALIBRATED

        return FeeParameters(
            mu=self.mu,
            nu=self.nu,
            H=self.H,
            lambda_B=self.lambda_B,
            optimization_status=optimization_status
        )


class LegacyOptimizationResults:
    """
    Legacy optimization results that are NO LONGER VALID.

    These were based on incorrect parameters and deprecated formulas.
    All results require recalibration with the unified specification.
    """

    @staticmethod
    def get_all_legacy_results() -> Dict[str, OptimizationResult]:
        """Get all legacy optimization results marked as invalid."""

        return {
            "claude_md_optimal": OptimizationResult(
                mu=0.0, nu=0.27, H=492, lambda_B=0.3,
                status=OptimizationValidityStatus.INVALID_PARAMETERS,
                description="CLAUDE.md 'optimal' - based on wrong α_data=0.5",
                based_on_alpha_data=0.5,
                based_on_Q_bar=690000.0,
                optimization_date="Pre-consolidation"
            ),

            "claude_md_balanced": OptimizationResult(
                mu=0.0, nu=0.48, H=492, lambda_B=0.3,
                status=OptimizationValidityStatus.INVALID_PARAMETERS,
                description="CLAUDE.md 'balanced' - based on wrong parameters",
                based_on_alpha_data=0.5,
                based_on_Q_bar=690000.0,
                optimization_date="Pre-consolidation"
            ),

            "claude_md_crisis": OptimizationResult(
                mu=0.0, nu=0.88, H=120, lambda_B=0.3,
                status=OptimizationValidityStatus.INVALID_PARAMETERS,
                description="CLAUDE.md 'crisis' - based on wrong parameters",
                based_on_alpha_data=0.5,
                based_on_Q_bar=690000.0,
                optimization_date="Pre-consolidation"
            ),

            "get_optimal_parameters": OptimizationResult(
                mu=0.0, nu=0.369, H=1794, lambda_B=0.365,
                status=OptimizationValidityStatus.INVALID_PARAMETERS,
                description="Python get_optimal_parameters() - based on wrong α_data=0.5",
                based_on_alpha_data=0.5,
                based_on_Q_bar=690000.0,
                optimization_date="2024 NSGA-II (invalid inputs)"
            ),

            "canonical_spec_optimal": OptimizationResult(
                mu=0.0, nu=0.1, H=36, lambda_B=0.3,
                status=OptimizationValidityStatus.INVALID_FORMULA,
                description="CANONICAL_FEE_MECHANISM_SPEC.md - based on deprecated per-tx formula",
                based_on_alpha_data=None,  # Different formula
                based_on_Q_bar=None,
                optimization_date="Deprecated spec"
            ),

            "ui_defaults": OptimizationResult(
                mu=0.0, nu=0.27, H=492, lambda_B=0.3,
                status=OptimizationValidityStatus.INVALID_PARAMETERS,
                description="UI slider defaults - not derived from any optimization",
                based_on_alpha_data=None,
                based_on_Q_bar=None,
                optimization_date="Never optimized"
            )
        }

    @staticmethod
    def emit_deprecation_warnings():
        """Emit warnings about deprecated optimization results."""
        warnings.warn(
            "🚨 OPTIMIZATION WARNING: All previous optimization results are INVALID. "
            "They were based on incorrect parameters (α_data=0.5 vs expected ~0.22, "
            "Q̄=690k vs expected ~200k) or deprecated formulas. "
            "Re-optimization required with unified specification.",
            UserWarning,
            stacklevel=2
        )


class RecalibrationGuidance:
    """Guidance for recalibrating optimization results."""

    @staticmethod
    def get_recalibration_requirements() -> Dict[str, str]:
        """Get requirements for valid recalibration."""
        return {
            "real_alpha_data": (
                "α_data must be calibrated from real Taiko proposeBlock transactions. "
                "Current theoretical estimate (0.22) is placeholder."
            ),
            "real_Q_bar": (
                "Q̄ must be measured from real Taiko L2 batch sizes. "
                "Current conservative estimate (200k) is placeholder."
            ),
            "unified_formula": (
                "Optimization must use unified specification: "
                "F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t) with UX wrapper."
            ),
            "historical_validation": (
                "Results must be validated on historical Ethereum L1 data "
                "with realistic vault dynamics."
            ),
            "invariant_testing": (
                "Must satisfy cost-recovery, solvency, and UX constraints "
                "as defined in AUTHORITATIVE_SPECIFICATION.md."
            )
        }

    @staticmethod
    def get_conservative_parameters() -> OptimizationResult:
        """Get conservative parameter set for immediate deployment."""
        return OptimizationResult(
            mu=0.0,     # Pure deficit correction (no L1 pass-through)
            nu=0.3,     # Moderate vault healing
            H=144,      # ~4.8 minute horizon at 2-second blocks
            lambda_B=0.2,  # Conservative L1 smoothing

            status=OptimizationValidityStatus.UNCALIBRATED,
            description="Conservative deployment parameters - NOT OPTIMIZED",
            based_on_alpha_data=0.22,  # Theoretical estimate
            based_on_Q_bar=200_000.0,  # Conservative estimate
            optimization_date=None
        )

    @staticmethod
    def get_experimental_parameters() -> OptimizationResult:
        """Get experimental parameter set for testing."""
        return OptimizationResult(
            mu=0.1,     # Small L1 component for testing
            nu=0.7,     # Aggressive vault healing
            H=72,       # ~2.4 minute horizon
            lambda_B=0.5,  # Responsive L1 smoothing

            status=OptimizationValidityStatus.UNCALIBRATED,
            description="Experimental test parameters - NOT OPTIMIZED",
            based_on_alpha_data=0.22,  # Theoretical estimate
            based_on_Q_bar=200_000.0,  # Conservative estimate
            optimization_date=None
        )


def validate_optimization_claims():
    """Validate any optimization claims in the codebase."""
    print("🔍 OPTIMIZATION VALIDATION")
    print("=" * 50)

    # Check legacy results
    legacy_results = LegacyOptimizationResults.get_all_legacy_results()

    print(f"📊 Found {len(legacy_results)} legacy optimization results:")
    for name, result in legacy_results.items():
        status_icon = "❌" if result.status != OptimizationValidityStatus.VALID else "✅"
        print(f"   {status_icon} {name}: {result.description}")

    print(f"\n🚨 VALIDITY STATUS:")
    valid_results = [r for r in legacy_results.values() if r.status == OptimizationValidityStatus.VALID]
    invalid_results = [r for r in legacy_results.values() if r.status != OptimizationValidityStatus.VALID]

    print(f"   Valid results: {len(valid_results)}")
    print(f"   Invalid results: {len(invalid_results)}")

    if len(invalid_results) > 0:
        print(f"\n⚠️ RECALIBRATION REQUIRED:")
        requirements = RecalibrationGuidance.get_recalibration_requirements()
        for req, desc in requirements.items():
            print(f"   • {req}: {desc}")

    return len(valid_results) == 0  # True if recalibration needed


if __name__ == "__main__":
    # Emit deprecation warnings
    LegacyOptimizationResults.emit_deprecation_warnings()

    # Validate optimization claims
    needs_recalibration = validate_optimization_claims()

    if needs_recalibration:
        print(f"\n🔄 RECOMMENDED ACTION:")
        print(f"   1. Use conservative parameters for immediate deployment")
        print(f"   2. Calibrate α_data and Q̄ from real Taiko data")
        print(f"   3. Re-run optimization with correct parameters")
        print(f"   4. Validate results on historical data")
        print(f"   5. Update all 'optimal' parameter claims")
    else:
        print(f"\n✅ OPTIMIZATION RESULTS VALID")