"""
API Data Models for Taiko Fee Mechanism Service

Pydantic models for request/response validation and serialization.
These models define the interface between the JavaScript web frontend
and the Python canonical implementations.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import numpy as np


class VaultInitMode(str, Enum):
    """Vault initialization strategies."""
    target = "target"
    deficit = "deficit"
    surplus = "surplus"
    custom = "custom"


class OptimizationStrategy(str, Enum):
    """Optimization strategies."""
    user_centric = "user_centric"
    protocol_centric = "protocol_centric"
    balanced = "balanced"
    crisis_resilient = "crisis_resilient"
    capital_efficient = "capital_efficient"


class FeeParametersRequest(BaseModel):
    """Request model for fee mechanism parameters."""
    mu: float = Field(0.0, ge=0.0, le=1.0, description="L1 weight parameter")
    nu: float = Field(0.27, ge=0.0, le=1.0, description="Deficit weight parameter")
    H: int = Field(492, ge=24, le=1800, description="Prediction horizon (steps)")

    target_balance: float = Field(1000.0, gt=0.0, description="Target vault balance (ETH)")
    min_fee: float = Field(1e-8, gt=0.0, description="Minimum fee floor (ETH)")
    gas_per_tx: float = Field(2000.0, gt=0.0, description="Gas cost per transaction")

    # Optional advanced parameters
    base_tx_demand: Optional[float] = Field(100.0, gt=0.0)
    fee_elasticity: Optional[float] = Field(2.0, gt=0.0)
    max_tx_demand: Optional[float] = Field(1000.0, gt=0.0)

    guaranteed_recovery: Optional[bool] = Field(False)
    min_deficit_rate: Optional[float] = Field(1e-3, gt=0.0)

    @validator('H')
    def validate_H_alignment(cls, v):
        """Ensure H is aligned to batch intervals."""
        if v % 6 != 0:
            # Round to nearest 6-step alignment
            v = round(v / 6) * 6
        return max(24, min(1800, v))


class SimulationRequest(BaseModel):
    """Request model for fee mechanism simulation."""
    parameters: FeeParametersRequest
    vault_init: VaultInitMode = VaultInitMode.target
    deficit_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Initial deficit ratio")
    custom_balance: Optional[float] = Field(None, gt=0.0)

    # L1 data configuration
    l1_data: Optional[List[float]] = Field(None, description="L1 basefee data in wei")
    simulation_steps: int = Field(1800, ge=100, le=10800, description="Simulation length")
    use_synthetic_l1: bool = Field(True, description="Use synthetic L1 data if l1_data not provided")

    # Synthetic L1 data parameters
    initial_basefee_wei: Optional[float] = Field(15e9, gt=0.0)
    volatility: Optional[float] = Field(0.3, gt=0.0, le=2.0)
    include_spike: Optional[bool] = Field(False)


class FeeCalculationRequest(BaseModel):
    """Request model for single fee calculation."""
    parameters: FeeParametersRequest
    l1_basefee_wei: float = Field(..., gt=0.0, description="L1 basefee in wei")
    vault_deficit: float = Field(0.0, ge=0.0, description="Current vault deficit (ETH)")
    apply_smoothing: bool = Field(True, description="Apply L1 cost smoothing")


class OptimizationRequest(BaseModel):
    """Request model for parameter optimization."""
    strategy: OptimizationStrategy = OptimizationStrategy.balanced
    population_size: int = Field(50, ge=20, le=200, description="GA population size")
    generations: int = Field(25, ge=10, le=100, description="GA generations")

    # Simulation configuration for evaluation
    vault_init: VaultInitMode = VaultInitMode.target
    simulation_steps: int = Field(1800, ge=100, le=3600)
    l1_data: Optional[List[float]] = Field(None)

    # Optional parameter bounds
    mu_min: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    mu_max: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    nu_min: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    nu_max: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    H_min: Optional[int] = Field(24, ge=6)
    H_max: Optional[int] = Field(1800, le=3600)


class MetricsCalculationRequest(BaseModel):
    """Request model for metrics calculation."""
    simulation_results: Dict[str, List[float]] = Field(..., description="Simulation time series data")


# Response Models

class FeeCalculationResponse(BaseModel):
    """Response model for fee calculation."""
    estimated_fee_eth: float = Field(..., description="Estimated fee in ETH")
    estimated_fee_gwei: float = Field(..., description="Estimated fee in gwei")
    l1_cost_component: float = Field(..., description="L1 cost component (ETH)")
    deficit_component: float = Field(..., description="Deficit correction component (ETH)")
    l1_cost_per_tx: float = Field(..., description="L1 cost per transaction (ETH)")
    guaranteed_recovery_applied: bool = Field(..., description="Whether guaranteed recovery was used")


class SimulationStepResult(BaseModel):
    """Single time step simulation result."""
    time_step: int
    l1_basefee_gwei: float
    estimated_fee_eth: float
    estimated_fee_gwei: float
    transaction_volume: float
    vault_balance: float
    vault_deficit: float
    fees_collected: float
    l1_costs_paid: float


class SimulationResponse(BaseModel):
    """Response model for fee mechanism simulation."""
    parameters: Dict[str, Any] = Field(..., description="Simulation parameters used")
    steps: List[SimulationStepResult] = Field(..., description="Time series results")

    # Summary statistics
    total_steps: int
    total_fees_collected: float
    total_l1_costs: float
    net_revenue: float

    # Basic metrics
    average_fee_gwei: float
    fee_stability_cv: float
    time_underfunded_pct: float
    max_deficit_ratio: float


class IndividualSolution(BaseModel):
    """Individual solution in optimization results."""
    mu: float
    nu: float
    H: int

    # Objective scores (higher is better)
    user_experience_score: float
    protocol_safety_score: float
    economic_efficiency_score: float
    overall_score: float

    # Detailed metrics
    average_fee_gwei: float
    fee_stability_cv: float
    time_underfunded_pct: float
    l1_tracking_error: float

    # Optimization metadata
    pareto_rank: int
    crowding_distance: float
    simulation_time: float


class OptimizationResponse(BaseModel):
    """Response model for optimization results."""
    strategy: str = Field(..., description="Optimization strategy used")
    pareto_solutions: List[IndividualSolution] = Field(..., description="Pareto optimal solutions")

    # Algorithm results
    total_generations: int
    total_evaluations: int
    optimization_time: float

    # Quality metrics
    hypervolume: float
    spread: float
    n_pareto_solutions: int

    # Convergence data
    convergence_history: List[Dict[str, float]] = Field(..., description="Generation-by-generation progress")


class ComprehensiveMetricsResponse(BaseModel):
    """Response model for comprehensive metrics."""

    # User Experience Metrics
    average_fee_gwei: float
    fee_stability_cv: float
    fee_affordability_score: float
    fee_predictability_1h: float
    fee_predictability_6h: float
    fee_rate_of_change_p95: float

    # Protocol Safety Metrics
    time_underfunded_pct: float
    max_deficit_ratio: float
    insolvency_protection_score: float
    vault_stress_resilience: float
    deficit_recovery_rate: float
    underfunding_resistance: float

    # Economic Efficiency Metrics
    vault_utilization_score: float
    capital_efficiency: float
    cost_coverage_ratio: float
    revenue_efficiency: float
    deficit_correction_rate: float

    # System Performance Metrics
    l1_tracking_error: float
    correlation_with_l1: float
    transaction_throughput: float
    fee_revenue_total: float
    l1_cost_total: float
    net_revenue: float

    # Validation Metrics
    simulation_length: int
    data_completeness: float
    outlier_percentage: float

    # Overall Scores
    user_experience_score: float
    protocol_safety_score: float
    economic_efficiency_score: float
    overall_performance_score: float

    # Qualitative assessments
    metric_grades: Dict[str, str] = Field(..., description="Threshold-based grades (excellent/good/poor)")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str = "healthy"
    version: str = "1.0.0"
    uptime_seconds: float
    canonical_modules_loaded: bool
    memory_usage_mb: float


class ErrorResponse(BaseModel):
    """Response model for API errors."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error description")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")


# Utility models for common operations

class ParameterValidationResponse(BaseModel):
    """Response for parameter validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    suggested_parameters: Optional[FeeParametersRequest]


class PresetsResponse(BaseModel):
    """Response model for available parameter presets."""
    presets: Dict[str, FeeParametersRequest] = Field(..., description="Named parameter presets")


class L1DataGenerationRequest(BaseModel):
    """Request for synthetic L1 data generation."""
    steps: int = Field(1800, ge=100, le=10800)
    initial_basefee_wei: float = Field(15e9, gt=0.0)
    volatility: float = Field(0.3, gt=0.0, le=2.0)
    include_spike: bool = Field(False)
    spike_delay_pct: float = Field(50.0, ge=0.0, le=90.0)
    spike_height: float = Field(0.3, ge=0.1, le=0.8)
    seed: Optional[int] = Field(42, ge=0)


class L1DataGenerationResponse(BaseModel):
    """Response with synthetic L1 data."""
    l1_basefees_wei: List[float] = Field(..., description="Generated L1 basefee data in wei")
    l1_basefees_gwei: List[float] = Field(..., description="Generated L1 basefee data in gwei")
    parameters_used: L1DataGenerationRequest = Field(..., description="Generation parameters")
    statistics: Dict[str, float] = Field(..., description="Data statistics")


# Custom validation functions

def validate_simulation_data(data: Dict[str, List[float]]) -> bool:
    """Validate that simulation data has required fields and consistent lengths."""
    required_fields = ['estimatedFee', 'vaultBalance', 'l1Basefee', 'transactionVolume']

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        if not data[field]:
            raise ValueError(f"Empty data for required field: {field}")

    lengths = [len(data[field]) for field in required_fields]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent data lengths: {dict(zip(required_fields, lengths))}")

    return True


def validate_l1_data(l1_data: List[float]) -> bool:
    """Validate L1 basefee data."""
    if not l1_data:
        raise ValueError("L1 data cannot be empty")

    for i, value in enumerate(l1_data):
        if value <= 0:
            raise ValueError(f"L1 basefee must be positive, got {value} at index {i}")
        if value > 1e15:  # 1000 ETH worth of wei - sanity check
            raise ValueError(f"L1 basefee suspiciously high: {value} wei at index {i}")

    return True