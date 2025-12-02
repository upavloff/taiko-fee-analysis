"""
Runtime validation and assertion system

Provides defensive programming utilities to catch unit mismatches and
unrealistic values during simulation execution.
"""

import logging
from typing import Union, Optional, List
from functools import wraps
import numpy as np

logger = logging.getLogger(__name__)


class UnitMismatchError(ValueError):
    """Raised when unit validation fails"""
    pass


class UnrealisticValueError(ValueError):
    """Raised when values are outside realistic ranges"""
    pass


# === VALIDATION FUNCTIONS ===

def validate_wei_amount(value: Union[int, float], name: str,
                       min_wei: int = 0, max_wei: int = int(1e21)) -> int:
    """
    Validate a wei amount is reasonable

    Args:
        value: Value to validate
        name: Name for error messages
        min_wei: Minimum allowed value (default: 0)
        max_wei: Maximum allowed value (default: 1000 ETH)

    Returns:
        Validated value as integer

    Raises:
        UnitMismatchError: If value is outside bounds
    """
    if not isinstance(value, (int, float)):
        raise UnitMismatchError(f"{name} must be numeric, got {type(value)}")

    if not (min_wei <= value <= max_wei):
        raise UnitMismatchError(
            f"{name} {value:,.0f} wei outside valid range [{min_wei:,}, {max_wei:,}]"
        )

    return int(value)


def validate_gwei_amount(value: float, name: str,
                        min_gwei: float = 0.0001, max_gwei: float = 10000) -> float:
    """
    Validate a gwei amount is reasonable

    Args:
        value: Value in gwei
        name: Name for error messages
        min_gwei: Minimum reasonable fee (default: 0.0001 gwei)
        max_gwei: Maximum reasonable fee (default: 10000 gwei)

    Returns:
        Validated value

    Raises:
        UnrealisticValueError: If value is unrealistic
    """
    if not isinstance(value, (int, float)):
        raise UnitMismatchError(f"{name} must be numeric, got {type(value)}")

    if not (min_gwei <= value <= max_gwei):
        raise UnrealisticValueError(
            f"{name} {value:.6f} gwei outside realistic range [{min_gwei}, {max_gwei}]"
        )

    return float(value)


def validate_eth_amount(value: float, name: str,
                       min_eth: float = 0, max_eth: float = 1000) -> float:
    """
    Validate an ETH amount is reasonable

    Args:
        value: Value in ETH
        name: Name for error messages
        min_eth: Minimum allowed value
        max_eth: Maximum allowed value

    Returns:
        Validated value

    Raises:
        UnrealisticValueError: If value is unrealistic
    """
    if not isinstance(value, (int, float)):
        raise UnitMismatchError(f"{name} must be numeric, got {type(value)}")

    if not (min_eth <= value <= max_eth):
        raise UnrealisticValueError(
            f"{name} {value:.6f} ETH outside realistic range [{min_eth}, {max_eth}]"
        )

    return float(value)


def validate_fee_calculation_inputs(l1_cost: Union[int, float], deficit: Union[int, float],
                                  q_bar: int, function_name: str) -> tuple:
    """
    Validate inputs to fee calculation functions

    Args:
        l1_cost: L1 cost (should be in wei)
        deficit: Deficit amount (should be in wei or ETH)
        q_bar: Gas amount per batch
        function_name: Name of calling function

    Returns:
        (validated_l1_cost, validated_deficit)

    Raises:
        UnitMismatchError: If inputs suggest wrong units
    """
    # Check if l1_cost looks like it's in ETH (too small)
    if 0 < l1_cost < 1e12:  # Less than 0.000001 ETH in wei
        logger.warning(
            f"{function_name}: L1 cost {l1_cost} looks suspiciously small. "
            f"Are you passing ETH instead of wei?"
        )
        # Don't raise error, just warn (might be legitimate small cost)

    # Check if l1_cost is unrealistically large
    if l1_cost > 1e21:  # More than 1000 ETH in wei
        raise UnitMismatchError(
            f"{function_name}: L1 cost {l1_cost:,.0f} wei is unrealistically large"
        )

    # Validate deficit
    if abs(deficit) > 1e21:  # More than 1000 ETH
        raise UnrealisticValueError(
            f"{function_name}: Deficit {deficit:,.0f} is unrealistically large"
        )

    # Validate q_bar
    if not (1e4 <= q_bar <= 1e7):  # 10k to 10M gas
        raise UnrealisticValueError(
            f"{function_name}: q_bar {q_bar:,} outside realistic range [10k, 10M]"
        )

    return l1_cost, deficit


def validate_fee_output(fee_wei: Union[int, float], function_name: str,
                       inputs_summary: str = "") -> float:
    """
    Validate fee calculation output

    Args:
        fee_wei: Calculated fee in wei per gas
        function_name: Name of calling function
        inputs_summary: Summary of inputs for debugging

    Returns:
        Validated fee

    Raises:
        UnrealisticValueError: If fee is unrealistic
    """
    fee_gwei = fee_wei / 1e9

    # Check for suspiciously small fees (suggests unit mismatch)
    if 0 < fee_gwei < 1e-6:  # Less than 0.000001 gwei
        raise UnrealisticValueError(
            f"{function_name}: Fee {fee_gwei:.9f} gwei suspiciously small. "
            f"Unit mismatch? Inputs: {inputs_summary}"
        )

    # Check for unrealistically large fees
    if fee_gwei > 50000:  # More than 50k gwei
        raise UnrealisticValueError(
            f"{function_name}: Fee {fee_gwei:.1f} gwei unrealistically large. "
            f"Inputs: {inputs_summary}"
        )

    # Check for negative fees
    if fee_wei < 0:
        logger.warning(f"{function_name}: Negative fee {fee_gwei:.6f} gwei")

    return float(fee_wei)


# === DECORATORS ===

def validate_fee_function(func):
    """Decorator to add validation to fee calculation functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to extract common parameters for validation (skip self)
        if len(args) >= 3:  # self, l1_cost, deficit
            l1_cost, deficit = args[1], args[2]
            validate_fee_calculation_inputs(l1_cost, deficit, 690000, func.__name__)

        # Call original function
        result = func(*args, **kwargs)

        # Validate output if it looks like a fee
        if isinstance(result, (int, float)) and result >= 0:
            inputs_str = f"args={args[:3]}..." if len(args) > 3 else f"args={args}"
            validate_fee_output(result, func.__name__, inputs_str)

        return result

    return wrapper


def validate_simulation_inputs(func):
    """Decorator to validate simulation input data"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Look for array-like inputs that might be L1 costs
        for i, arg in enumerate(args):
            if hasattr(arg, '__len__') and len(arg) > 0:
                first_val = arg[0] if hasattr(arg, '__getitem__') else next(iter(arg))
                if isinstance(first_val, (int, float)):
                    # Check if this looks like L1 cost data
                    max_val = max(arg) if hasattr(arg, '__iter__') else first_val
                    min_val = min(arg) if hasattr(arg, '__iter__') else first_val

                    # Warn if values look like ETH (too small for wei)
                    if 0 < max_val < 1e12:
                        logger.warning(
                            f"{func.__name__}: Input array {i} values range {min_val:.6f}-{max_val:.6f}. "
                            f"These look small for wei amounts. Using ETH instead of wei?"
                        )

        return func(*args, **kwargs)

    return wrapper


# === ASSERTION UTILITIES ===

def assert_reasonable_fee(fee_gwei: float, context: str = ""):
    """Assert that a fee is in reasonable range"""
    assert 0.00001 <= fee_gwei <= 10000, \
        f"Unreasonable fee {fee_gwei:.6f} gwei{' in ' + context if context else ''}"


def assert_no_unit_mismatch(value: float, expected_range: tuple, unit: str, context: str = ""):
    """Assert value is in expected range for its unit"""
    min_val, max_val = expected_range
    assert min_val <= value <= max_val, \
        f"Value {value:.6f} {unit} outside expected range [{min_val}, {max_val}]" \
        f"{' in ' + context if context else ''}"


def assert_proportional_relationship(x: float, y: float, expected_ratio: float,
                                   tolerance: float = 0.1, context: str = ""):
    """Assert that y ≈ x * expected_ratio within tolerance"""
    actual_ratio = y / (x + 1e-20)  # Avoid div by zero
    error = abs(actual_ratio - expected_ratio) / expected_ratio

    assert error <= tolerance, \
        f"Ratio {actual_ratio:.6f} not close to expected {expected_ratio:.6f} " \
        f"(error: {error:.2%}){' in ' + context if context else ''}"


# === LOGGING SETUP ===

def setup_validation_logging(level=logging.WARNING):
    """Setup logging for validation warnings"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# === USAGE EXAMPLES ===

if __name__ == "__main__":
    # Example usage
    setup_validation_logging()

    # Test validations
    try:
        validate_gwei_amount(0.05, "test_fee")  # Should pass
        validate_gwei_amount(50000, "huge_fee")  # Should fail
    except UnrealisticValueError as e:
        print(f"Caught unrealistic value: {e}")

    try:
        validate_fee_calculation_inputs(0.001, 0, 690000, "test_function")  # Should warn
    except UnitMismatchError as e:
        print(f"Caught unit mismatch: {e}")