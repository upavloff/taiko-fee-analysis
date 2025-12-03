"""Core simulation components for Taiko fee mechanism."""

# Import canonical modules (single source of truth)
from .canonical_fee_mechanism import (
    CanonicalTaikoFeeCalculator,
    FeeParameters,
    VaultInitMode,
    create_default_calculator,
    create_balanced_calculator,
    create_crisis_calculator,
    get_optimal_parameters,
    validate_fee_parameters
)

from .canonical_metrics import (
    CanonicalMetricsCalculator,
    calculate_basic_metrics,
    validate_metric_thresholds
)

from .canonical_optimization import (
    CanonicalOptimizer,
    OptimizationStrategy,
    validate_parameter_set
)

__all__ = [
    # Canonical fee mechanism
    'CanonicalTaikoFeeCalculator',
    'FeeParameters',
    'VaultInitMode',
    'create_default_calculator',
    'create_balanced_calculator',
    'create_crisis_calculator',
    'get_optimal_parameters',
    'validate_fee_parameters',
    # Canonical metrics
    'CanonicalMetricsCalculator',
    'calculate_basic_metrics',
    'validate_metric_thresholds',
    # Canonical optimization
    'CanonicalOptimizer',
    'OptimizationStrategy',
    'validate_parameter_set'
]