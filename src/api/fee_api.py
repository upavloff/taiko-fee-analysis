"""
Fee Mechanism API Endpoints

This module provides REST endpoints for the canonical fee mechanism functionality.
The web interface calls these endpoints to perform fee calculations and simulations
using the authoritative Python implementations.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import numpy as np
import logging
import time
import traceback
import sys
import os

# Add project paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from core.canonical_fee_mechanism import (
    CanonicalTaikoFeeCalculator,
    FeeParameters,
    VaultInitMode as CoreVaultInitMode,
    create_default_calculator,
    create_balanced_calculator,
    create_crisis_calculator,
    get_optimal_parameters
)
from core.canonical_metrics import (
    CanonicalMetricsCalculator,
    calculate_basic_metrics,
    validate_metric_thresholds
)
from api.models import (
    FeeCalculationRequest,
    FeeCalculationResponse,
    SimulationRequest,
    SimulationResponse,
    SimulationStepResult,
    MetricsCalculationRequest,
    ComprehensiveMetricsResponse,
    ParameterValidationResponse,
    PresetsResponse,
    L1DataGenerationRequest,
    L1DataGenerationResponse,
    VaultInitMode,
    validate_simulation_data,
    validate_l1_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/fee", tags=["fee_mechanism"])

# Cache for fee calculators (to avoid repeated initialization)
_calculator_cache: Dict[str, CanonicalTaikoFeeCalculator] = {}


def _get_or_create_calculator(params: FeeParameters) -> CanonicalTaikoFeeCalculator:
    """Get cached calculator or create new one."""
    # Create cache key from parameters
    cache_key = f"{params.mu}_{params.nu}_{params.H}_{params.target_balance}"

    if cache_key not in _calculator_cache:
        _calculator_cache[cache_key] = CanonicalTaikoFeeCalculator(params)

    return _calculator_cache[cache_key]


def _convert_vault_init_mode(mode: VaultInitMode) -> CoreVaultInitMode:
    """Convert API enum to core enum."""
    mapping = {
        VaultInitMode.target: CoreVaultInitMode.TARGET,
        VaultInitMode.deficit: CoreVaultInitMode.DEFICIT,
        VaultInitMode.surplus: CoreVaultInitMode.SURPLUS,
        VaultInitMode.custom: CoreVaultInitMode.CUSTOM
    }
    return mapping[mode]


def _generate_synthetic_l1_data(steps: int,
                               initial_basefee: float = 15e9,
                               volatility: float = 0.3,
                               include_spike: bool = False,
                               seed: Optional[int] = 42) -> List[float]:
    """Generate synthetic L1 basefee data using GBM."""
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / (24 * 3600 / 2)  # 2-second steps
    basefees = [initial_basefee]

    for i in range(steps - 1):
        # Base GBM
        dW = np.random.normal(0, np.sqrt(dt))
        drift = 0.0
        dS = drift * basefees[-1] * dt + volatility * basefees[-1] * dW
        new_basefee = max(basefees[-1] + dS, 1e6)  # Floor at 0.001 gwei

        # Add spike if requested
        if include_spike and 0.4 * steps <= i <= 0.6 * steps:
            spike_factor = 1.0 + 2.0 * np.sin((i - 0.4 * steps) / (0.2 * steps) * np.pi)
            new_basefee *= spike_factor

        basefees.append(new_basefee)

    return basefees


@router.post("/calculate", response_model=FeeCalculationResponse)
async def calculate_fee(request: FeeCalculationRequest) -> FeeCalculationResponse:
    """
    Calculate estimated fee for given parameters and conditions.

    This endpoint provides single-point fee calculations using the canonical
    fee mechanism implementation.
    """
    try:
        # Create fee parameters
        fee_params = FeeParameters(
            mu=request.parameters.mu,
            nu=request.parameters.nu,
            H=request.parameters.H,
            target_balance=request.parameters.target_balance,
            min_fee=request.parameters.min_fee,
            gas_per_tx=request.parameters.gas_per_tx,
            base_tx_demand=request.parameters.base_tx_demand or 100.0,
            fee_elasticity=request.parameters.fee_elasticity or 2.0,
            max_tx_demand=request.parameters.max_tx_demand or 1000.0,
            guaranteed_recovery=request.parameters.guaranteed_recovery or False,
            min_deficit_rate=request.parameters.min_deficit_rate or 1e-3
        )

        # Get calculator
        calculator = _get_or_create_calculator(fee_params)

        # Calculate L1 cost per transaction
        l1_cost = calculator.calculate_l1_cost_per_tx(
            request.l1_basefee_wei,
            apply_smoothing=request.apply_smoothing
        )

        # Calculate estimated fee
        estimated_fee = calculator.calculate_estimated_fee(l1_cost, request.vault_deficit)

        # Calculate components
        l1_component = fee_params.mu * l1_cost
        deficit_component = fee_params.nu * request.vault_deficit / fee_params.H

        # Check if guaranteed recovery was applied
        standard_deficit = deficit_component
        if fee_params.guaranteed_recovery and request.vault_deficit > 0:
            guaranteed_deficit = fee_params.min_deficit_rate
            guaranteed_applied = guaranteed_deficit > standard_deficit
        else:
            guaranteed_applied = False

        return FeeCalculationResponse(
            estimated_fee_eth=estimated_fee,
            estimated_fee_gwei=estimated_fee * 1e9,
            l1_cost_component=l1_component,
            deficit_component=deficit_component,
            l1_cost_per_tx=l1_cost,
            guaranteed_recovery_applied=guaranteed_applied
        )

    except Exception as e:
        logger.error(f"Fee calculation error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Fee calculation failed: {str(e)}")


@router.post("/simulate", response_model=SimulationResponse)
async def simulate_fee_mechanism(request: SimulationRequest) -> SimulationResponse:
    """
    Run complete fee mechanism simulation.

    This endpoint runs a full time-series simulation of the fee mechanism
    and returns detailed results for analysis and visualization.
    """
    try:
        start_time = time.time()

        # Create fee parameters
        fee_params = FeeParameters(
            mu=request.parameters.mu,
            nu=request.parameters.nu,
            H=request.parameters.H,
            target_balance=request.parameters.target_balance,
            min_fee=request.parameters.min_fee,
            gas_per_tx=request.parameters.gas_per_tx,
            base_tx_demand=request.parameters.base_tx_demand or 100.0,
            fee_elasticity=request.parameters.fee_elasticity or 2.0,
            max_tx_demand=request.parameters.max_tx_demand or 1000.0,
            guaranteed_recovery=request.parameters.guaranteed_recovery or False,
            min_deficit_rate=request.parameters.min_deficit_rate or 1e-3
        )

        # Get calculator
        calculator = _get_or_create_calculator(fee_params)

        # Create vault with specified initialization
        vault_mode = _convert_vault_init_mode(request.vault_init)
        vault = calculator.create_vault(
            vault_mode,
            deficit_ratio=request.deficit_ratio,
            custom_balance=request.custom_balance
        )

        # Prepare L1 data
        if request.l1_data and len(request.l1_data) > 0:
            validate_l1_data(request.l1_data)
            l1_basefees = request.l1_data
        elif request.use_synthetic_l1:
            l1_basefees = _generate_synthetic_l1_data(
                request.simulation_steps,
                initial_basefee=request.initial_basefee_wei or 15e9,
                volatility=request.volatility or 0.3,
                include_spike=request.include_spike or False
            )
        else:
            raise ValueError("No L1 data provided and synthetic data disabled")

        # Ensure we have enough L1 data
        if len(l1_basefees) < request.simulation_steps:
            # Extend by repeating
            repeats = (request.simulation_steps // len(l1_basefees)) + 1
            l1_basefees = (l1_basefees * repeats)[:request.simulation_steps]

        # Run simulation
        simulation_results = []
        total_fees_collected = 0.0
        total_l1_costs = 0.0

        for step in range(request.simulation_steps):
            l1_basefee_wei = l1_basefees[step]

            # Calculate L1 cost and fee
            l1_cost = calculator.calculate_l1_cost_per_tx(l1_basefee_wei)
            estimated_fee = calculator.calculate_estimated_fee(l1_cost, vault.deficit)

            # Calculate transaction volume
            tx_volume = calculator.calculate_transaction_volume(estimated_fee)

            # Collect fees (every step)
            fees_collected = estimated_fee * tx_volume
            vault.collect_fees(fees_collected)
            total_fees_collected += fees_collected

            # Pay L1 costs (every batch_interval steps)
            l1_costs_paid = 0.0
            if step % fee_params.batch_interval_steps == 0:
                l1_costs_paid = calculator.calculate_l1_batch_cost(l1_basefee_wei)
                vault.pay_l1_costs(l1_costs_paid)
                total_l1_costs += l1_costs_paid

            # Record step result
            step_result = SimulationStepResult(
                time_step=step,
                l1_basefee_gwei=l1_basefee_wei / 1e9,
                estimated_fee_eth=estimated_fee,
                estimated_fee_gwei=estimated_fee * 1e9,
                transaction_volume=tx_volume,
                vault_balance=vault.balance,
                vault_deficit=vault.deficit,
                fees_collected=fees_collected,
                l1_costs_paid=l1_costs_paid
            )
            simulation_results.append(step_result)

        # Calculate summary statistics
        fees = [step.estimated_fee_eth for step in simulation_results]
        deficits = [step.vault_deficit for step in simulation_results]

        average_fee_gwei = np.mean(fees) * 1e9
        fee_cv = np.std(fees) / (np.mean(fees) + 1e-12)
        underfunded_steps = sum(1 for d in deficits if d > fee_params.target_balance * 0.01)
        time_underfunded_pct = (underfunded_steps / len(deficits)) * 100.0
        max_deficit_ratio = max(deficits) / fee_params.target_balance

        # Log simulation completion
        simulation_time = time.time() - start_time
        logger.info(f"Simulation completed in {simulation_time:.2f}s: {len(simulation_results)} steps")

        return SimulationResponse(
            parameters={
                "mu": fee_params.mu,
                "nu": fee_params.nu,
                "H": fee_params.H,
                "target_balance": fee_params.target_balance,
                "vault_init": request.vault_init,
                "simulation_steps": request.simulation_steps
            },
            steps=simulation_results,
            total_steps=len(simulation_results),
            total_fees_collected=total_fees_collected,
            total_l1_costs=total_l1_costs,
            net_revenue=total_fees_collected - total_l1_costs,
            average_fee_gwei=average_fee_gwei,
            fee_stability_cv=fee_cv,
            time_underfunded_pct=time_underfunded_pct,
            max_deficit_ratio=max_deficit_ratio
        )

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Simulation failed: {str(e)}")


@router.post("/metrics", response_model=ComprehensiveMetricsResponse)
async def calculate_metrics(request: MetricsCalculationRequest) -> ComprehensiveMetricsResponse:
    """
    Calculate comprehensive performance metrics from simulation results.

    This endpoint provides detailed performance analysis using the canonical
    metrics calculation system.
    """
    try:
        # Validate simulation data
        validate_simulation_data(request.simulation_results)

        # Calculate comprehensive metrics
        calculator = CanonicalMetricsCalculator()
        metrics = calculator.calculate_comprehensive_metrics(request.simulation_results)

        # Calculate threshold-based grades
        metric_grades = validate_metric_thresholds(
            average_fee_gwei=metrics.average_fee_gwei,
            fee_cv=metrics.fee_stability_cv,
            underfunded_pct=metrics.time_underfunded_pct,
            tracking_error=metrics.l1_tracking_error
        )

        # Convert to response model
        return ComprehensiveMetricsResponse(
            # User Experience
            average_fee_gwei=metrics.average_fee_gwei,
            fee_stability_cv=metrics.fee_stability_cv,
            fee_affordability_score=metrics.fee_affordability_score,
            fee_predictability_1h=metrics.fee_predictability_1h,
            fee_predictability_6h=metrics.fee_predictability_6h,
            fee_rate_of_change_p95=metrics.fee_rate_of_change_p95,

            # Protocol Safety
            time_underfunded_pct=metrics.time_underfunded_pct,
            max_deficit_ratio=metrics.max_deficit_ratio,
            insolvency_protection_score=metrics.insolvency_protection_score,
            vault_stress_resilience=metrics.vault_stress_resilience,
            deficit_recovery_rate=metrics.deficit_recovery_rate,
            underfunding_resistance=metrics.underfunding_resistance,

            # Economic Efficiency
            vault_utilization_score=metrics.vault_utilization_score,
            capital_efficiency=metrics.capital_efficiency,
            cost_coverage_ratio=metrics.cost_coverage_ratio,
            revenue_efficiency=metrics.revenue_efficiency,
            deficit_correction_rate=metrics.deficit_correction_rate,

            # System Performance
            l1_tracking_error=metrics.l1_tracking_error,
            correlation_with_l1=metrics.correlation_with_l1,
            transaction_throughput=metrics.transaction_throughput,
            fee_revenue_total=metrics.fee_revenue_total,
            l1_cost_total=metrics.l1_cost_total,
            net_revenue=metrics.net_revenue,

            # Validation
            simulation_length=metrics.simulation_length,
            data_completeness=metrics.data_completeness,
            outlier_percentage=metrics.outlier_percentage,

            # Overall Scores
            user_experience_score=metrics.user_experience_score,
            protocol_safety_score=metrics.protocol_safety_score,
            economic_efficiency_score=metrics.economic_efficiency_score,
            overall_performance_score=metrics.overall_performance_score,

            # Grades
            metric_grades=metric_grades
        )

    except Exception as e:
        logger.error(f"Metrics calculation error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Metrics calculation failed: {str(e)}")


@router.post("/validate-parameters", response_model=ParameterValidationResponse)
async def validate_parameters(request: FeeCalculationRequest) -> ParameterValidationResponse:
    """
    Validate fee mechanism parameters and provide suggestions.

    This endpoint checks parameter validity and can suggest corrections
    or improvements based on research findings.
    """
    try:
        errors = []
        warnings = []
        suggested = None

        # Validate individual parameters
        params = request.parameters

        if not (0.0 <= params.mu <= 1.0):
            errors.append(f"mu must be between 0.0 and 1.0, got {params.mu}")
        elif params.mu > 0.1:
            warnings.append("Research suggests μ=0.0 is optimal for most scenarios")

        if not (0.0 <= params.nu <= 1.0):
            errors.append(f"nu must be between 0.0 and 1.0, got {params.nu}")
        elif not (0.1 <= params.nu <= 0.9):
            warnings.append("ν should typically be between 0.1 and 0.9")

        if params.H % 6 != 0:
            errors.append(f"H must be multiple of 6 for batch alignment, got {params.H}")

        if params.H < 24 or params.H > 1800:
            errors.append(f"H must be between 24 and 1800, got {params.H}")

        # Suggest optimal parameters if there are issues
        if errors or (params.mu != 0.0 or params.nu not in [0.27, 0.48, 0.88]):
            optimal_params = get_optimal_parameters()
            suggested = request.parameters.copy(update={
                "mu": optimal_params["mu"],
                "nu": optimal_params["nu"],
                "H": optimal_params["H"]
            })

        # Try to create calculator to catch any other validation issues
        if not errors:
            try:
                fee_params = FeeParameters(
                    mu=params.mu,
                    nu=params.nu,
                    H=params.H,
                    target_balance=params.target_balance
                )
                _get_or_create_calculator(fee_params)
            except Exception as e:
                errors.append(f"Parameter validation failed: {str(e)}")

        return ParameterValidationResponse(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggested_parameters=suggested
        )

    except Exception as e:
        logger.error(f"Parameter validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Parameter validation failed: {str(e)}")


@router.get("/presets", response_model=PresetsResponse)
async def get_presets() -> PresetsResponse:
    """
    Get predefined parameter presets based on research findings.

    Returns the optimal, balanced, and crisis-resilient parameter sets
    validated through comprehensive research.
    """
    try:
        optimal_params = get_optimal_parameters()

        presets = {
            "optimal": {
                "mu": optimal_params["mu"],
                "nu": optimal_params["nu"],
                "H": optimal_params["H"],
                "target_balance": 1000.0,
                "min_fee": 1e-8,
                "gas_per_tx": 2000.0,
                "guaranteed_recovery": False
            },
            "balanced": {
                "mu": 0.0,
                "nu": 0.27,  # Same as optimal for current research
                "H": 492,
                "target_balance": 1000.0,
                "min_fee": 1e-8,
                "gas_per_tx": 2000.0,
                "guaranteed_recovery": False
            },
            "crisis_resilient": {
                "mu": 0.0,
                "nu": 0.88,
                "H": 120,
                "target_balance": 1000.0,
                "min_fee": 1e-8,
                "gas_per_tx": 2000.0,
                "guaranteed_recovery": True,
                "min_deficit_rate": 1e-2
            },
            "conservative": {
                "mu": 0.0,
                "nu": 0.48,
                "H": 492,
                "target_balance": 1000.0,
                "min_fee": 1e-8,
                "gas_per_tx": 2000.0,
                "guaranteed_recovery": False
            }
        }

        return PresetsResponse(presets=presets)

    except Exception as e:
        logger.error(f"Presets error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load presets: {str(e)}")


@router.post("/generate-l1-data", response_model=L1DataGenerationResponse)
async def generate_l1_data(request: L1DataGenerationRequest) -> L1DataGenerationResponse:
    """
    Generate synthetic L1 basefee data for testing and simulation.

    This endpoint creates realistic L1 basefee time series data using
    geometric Brownian motion with optional volatility spikes.
    """
    try:
        # Generate synthetic data
        basefees_wei = _generate_synthetic_l1_data(
            steps=request.steps,
            initial_basefee=request.initial_basefee_wei,
            volatility=request.volatility,
            include_spike=request.include_spike,
            seed=request.seed
        )

        # Convert to gwei
        basefees_gwei = [bf / 1e9 for bf in basefees_wei]

        # Calculate statistics
        stats = {
            "min_gwei": min(basefees_gwei),
            "max_gwei": max(basefees_gwei),
            "mean_gwei": np.mean(basefees_gwei),
            "std_gwei": np.std(basefees_gwei),
            "cv": np.std(basefees_gwei) / np.mean(basefees_gwei),
            "total_steps": len(basefees_wei)
        }

        return L1DataGenerationResponse(
            l1_basefees_wei=basefees_wei,
            l1_basefees_gwei=basefees_gwei,
            parameters_used=request,
            statistics=stats
        )

    except Exception as e:
        logger.error(f"L1 data generation error: {e}")
        raise HTTPException(status_code=400, detail=f"L1 data generation failed: {str(e)}")


# Utility endpoints

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test basic functionality
        test_calculator = create_default_calculator()
        test_fee = test_calculator.calculate_estimated_fee(0.001, 0.0)

        return {
            "status": "healthy",
            "canonical_fee_mechanism": "operational",
            "test_fee_calculation": f"{test_fee:.2e} ETH"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.post("/clear-cache")
async def clear_calculator_cache():
    """Clear the calculator cache."""
    global _calculator_cache
    cache_size = len(_calculator_cache)
    _calculator_cache.clear()
    return {"message": f"Cleared {cache_size} cached calculators"}