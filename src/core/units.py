"""
Type-Safe Unit System for Fee Mechanism Calculations

This module provides a comprehensive type-safe unit system to prevent
the critical unit conversion bugs that have affected the fee mechanism.

Key Features:
- Type-safe unit classes (Wei, Gwei, ETH) with automatic validation
- Conversion functions with bounds checking and overflow protection
- Runtime assertion helpers for detecting unit mismatches
- Clear naming conventions and documentation

Critical Safety Goals:
1. Prevent ETH/Wei confusion (10^18 factor errors)
2. Catch unreasonable fee values immediately
3. Ensure consistent units across Python/JavaScript
4. Provide clear error messages for debugging

Example Usage:
    from core.units import Wei, Gwei, ETH, gwei_to_wei, validate_fee_range

    # Type-safe conversions
    basefee = Gwei(50.0)
    basefee_wei = gwei_to_wei(basefee)
    l1_cost = calculate_l1_cost(basefee_wei, gas_per_tx=20000)

    # Automatic validation
    fee_result = Wei(estimated_fee_wei)  # Validates reasonable range
    validate_fee_range(fee_result.to_gwei(), "final fee calculation")
"""

import math
from typing import Union, Optional, Any
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import warnings


# Unit conversion constants
WEI_PER_GWEI = 10**9
WEI_PER_ETH = 10**18
GWEI_PER_ETH = 10**9

# Reasonable bounds for validation (in gwei)
MIN_REASONABLE_FEE_GWEI = 0.001      # 0.001 gwei minimum
MAX_REASONABLE_FEE_GWEI = 10000.0    # 10,000 gwei maximum
MIN_REASONABLE_BASEFEE_GWEI = 0.001  # Historical minimum ~0.055 gwei
MAX_REASONABLE_BASEFEE_GWEI = 2000.0 # Historical maximum ~1,352 gwei


class UnitValidationError(ValueError):
    """Raised when unit validation fails."""
    pass


class UnitOverflowError(OverflowError):
    """Raised when unit conversion would cause overflow."""
    pass


@dataclass
class Wei:
    """Type-safe wei unit with automatic validation."""

    value: int

    def __post_init__(self):
        """Validate wei value on creation."""
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Wei value must be numeric, got {type(self.value)}")

        # Convert to int if float (with validation)
        if isinstance(self.value, float):
            if not self.value.is_finite():
                raise UnitValidationError(f"Wei value must be finite, got {self.value}")
            if self.value != int(self.value):
                warnings.warn(f"Converting float wei {self.value} to int {int(self.value)}")
            self.value = int(self.value)

        if self.value < 0:
            raise UnitValidationError(f"Wei value cannot be negative: {self.value}")

        # Check for reasonable upper bound (prevent overflow)
        if self.value > 10**30:  # ~10^12 ETH
            raise UnitOverflowError(f"Wei value too large: {self.value}")

    def to_gwei(self) -> 'Gwei':
        """Convert to Gwei with precision handling."""
        gwei_value = self.value / WEI_PER_GWEI
        return Gwei(gwei_value)

    def to_eth(self) -> 'ETH':
        """Convert to ETH with precision handling."""
        eth_value = self.value / WEI_PER_ETH
        return ETH(eth_value)

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.value >= WEI_PER_ETH:
            return f"{self.value / WEI_PER_ETH:.6f} ETH ({self.value:,} wei)"
        elif self.value >= WEI_PER_GWEI:
            return f"{self.value / WEI_PER_GWEI:.6f} gwei ({self.value:,} wei)"
        else:
            return f"{self.value:,} wei"

    def __repr__(self) -> str:
        return f"Wei({self.value})"

    # Arithmetic operations
    def __add__(self, other: 'Wei') -> 'Wei':
        if not isinstance(other, Wei):
            raise TypeError(f"Can only add Wei to Wei, got {type(other)}")
        return Wei(self.value + other.value)

    def __sub__(self, other: 'Wei') -> 'Wei':
        if not isinstance(other, Wei):
            raise TypeError(f"Can only subtract Wei from Wei, got {type(other)}")
        return Wei(self.value - other.value)

    def __mul__(self, factor: Union[int, float]) -> 'Wei':
        if not isinstance(factor, (int, float)):
            raise TypeError(f"Can only multiply Wei by number, got {type(factor)}")
        return Wei(int(self.value * factor))

    def __truediv__(self, divisor: Union[int, float]) -> 'Wei':
        if not isinstance(divisor, (int, float)):
            raise TypeError(f"Can only divide Wei by number, got {type(divisor)}")
        if divisor == 0:
            raise ZeroDivisionError("Cannot divide Wei by zero")
        return Wei(int(self.value / divisor))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Wei):
            return self.value == other.value
        return False

    def __lt__(self, other: 'Wei') -> bool:
        if not isinstance(other, Wei):
            raise TypeError(f"Can only compare Wei with Wei, got {type(other)}")
        return self.value < other.value


@dataclass
class Gwei:
    """Type-safe gwei unit with automatic validation."""

    value: float

    def __post_init__(self):
        """Validate gwei value on creation."""
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Gwei value must be numeric, got {type(self.value)}")

        self.value = float(self.value)

        if not math.isfinite(self.value):
            raise UnitValidationError(f"Gwei value must be finite, got {self.value}")

        if self.value < 0:
            raise UnitValidationError(f"Gwei value cannot be negative: {self.value}")

        # Check for reasonable bounds
        if self.value > MAX_REASONABLE_FEE_GWEI * 10:  # Allow some headroom
            warnings.warn(f"Unusually large gwei value: {self.value:.6f}")

    def to_wei(self) -> Wei:
        """Convert to Wei with overflow checking."""
        wei_value = self.value * WEI_PER_GWEI
        if wei_value > 10**30:
            raise UnitOverflowError(f"Gwei to Wei conversion would overflow: {self.value}")
        return Wei(int(wei_value))

    def to_eth(self) -> 'ETH':
        """Convert to ETH."""
        eth_value = self.value / GWEI_PER_ETH
        return ETH(eth_value)

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"{self.value:.6f} gwei"

    def __repr__(self) -> str:
        return f"Gwei({self.value})"

    # Arithmetic operations
    def __add__(self, other: 'Gwei') -> 'Gwei':
        if not isinstance(other, Gwei):
            raise TypeError(f"Can only add Gwei to Gwei, got {type(other)}")
        return Gwei(self.value + other.value)

    def __sub__(self, other: 'Gwei') -> 'Gwei':
        if not isinstance(other, Gwei):
            raise TypeError(f"Can only subtract Gwei from Gwei, got {type(other)}")
        return Gwei(self.value - other.value)

    def __mul__(self, factor: Union[int, float]) -> 'Gwei':
        if not isinstance(factor, (int, float)):
            raise TypeError(f"Can only multiply Gwei by number, got {type(factor)}")
        return Gwei(self.value * factor)

    def __truediv__(self, divisor: Union[int, float]) -> 'Gwei':
        if not isinstance(divisor, (int, float)):
            raise TypeError(f"Can only divide Gwei by number, got {type(divisor)}")
        if divisor == 0:
            raise ZeroDivisionError("Cannot divide Gwei by zero")
        return Gwei(self.value / divisor)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Gwei):
            return abs(self.value - other.value) < 1e-9  # Float comparison tolerance
        return False

    def __lt__(self, other: 'Gwei') -> bool:
        if not isinstance(other, Gwei):
            raise TypeError(f"Can only compare Gwei with Gwei, got {type(other)}")
        return self.value < other.value


@dataclass
class ETH:
    """Type-safe ETH unit with automatic validation."""

    value: float

    def __post_init__(self):
        """Validate ETH value on creation."""
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"ETH value must be numeric, got {type(self.value)}")

        self.value = float(self.value)

        if not math.isfinite(self.value):
            raise UnitValidationError(f"ETH value must be finite, got {self.value}")

        if self.value < 0:
            raise UnitValidationError(f"ETH value cannot be negative: {self.value}")

        # Check for reasonable bounds (fee context)
        if self.value > 1.0:  # Fees shouldn't exceed 1 ETH
            warnings.warn(f"Unusually large ETH value (fee context): {self.value:.6f}")

    def to_wei(self) -> Wei:
        """Convert to Wei with overflow checking."""
        wei_value = self.value * WEI_PER_ETH
        if wei_value > 10**30:
            raise UnitOverflowError(f"ETH to Wei conversion would overflow: {self.value}")
        return Wei(int(wei_value))

    def to_gwei(self) -> Gwei:
        """Convert to Gwei."""
        gwei_value = self.value * GWEI_PER_ETH
        return Gwei(gwei_value)

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"{self.value:.6f} ETH"

    def __repr__(self) -> str:
        return f"ETH({self.value})"

    # Arithmetic operations
    def __add__(self, other: 'ETH') -> 'ETH':
        if not isinstance(other, ETH):
            raise TypeError(f"Can only add ETH to ETH, got {type(other)}")
        return ETH(self.value + other.value)

    def __sub__(self, other: 'ETH') -> 'ETH':
        if not isinstance(other, ETH):
            raise TypeError(f"Can only subtract ETH from ETH, got {type(other)}")
        return ETH(self.value - other.value)

    def __mul__(self, factor: Union[int, float]) -> 'ETH':
        if not isinstance(factor, (int, float)):
            raise TypeError(f"Can only multiply ETH by number, got {type(factor)}")
        return ETH(self.value * factor)

    def __truediv__(self, divisor: Union[int, float]) -> 'ETH':
        if not isinstance(divisor, (int, float)):
            raise TypeError(f"Can only divide ETH by number, got {type(divisor)}")
        if divisor == 0:
            raise ZeroDivisionError("Cannot divide ETH by zero")
        return ETH(self.value / divisor)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ETH):
            return abs(self.value - other.value) < 1e-12  # Float comparison tolerance
        return False

    def __lt__(self, other: 'ETH') -> bool:
        if not isinstance(other, ETH):
            raise TypeError(f"Can only compare ETH with ETH, got {type(other)}")
        return self.value < other.value


# Conversion functions with validation
def gwei_to_wei(gwei: Gwei) -> Wei:
    """Convert Gwei to Wei with validation."""
    if not isinstance(gwei, Gwei):
        raise TypeError(f"Expected Gwei, got {type(gwei)}")
    return gwei.to_wei()


def wei_to_gwei(wei: Wei) -> Gwei:
    """Convert Wei to Gwei with validation."""
    if not isinstance(wei, Wei):
        raise TypeError(f"Expected Wei, got {type(wei)}")
    return wei.to_gwei()


def eth_to_wei(eth: ETH) -> Wei:
    """Convert ETH to Wei with validation."""
    if not isinstance(eth, ETH):
        raise TypeError(f"Expected ETH, got {type(eth)}")
    return eth.to_wei()


def wei_to_eth(wei: Wei) -> ETH:
    """Convert Wei to ETH with validation."""
    if not isinstance(wei, Wei):
        raise TypeError(f"Expected Wei, got {type(wei)}")
    return wei.to_eth()


# Legacy conversion helpers (for gradual migration)
def safe_gwei_to_wei(value: Union[float, int]) -> Wei:
    """Safely convert numeric gwei to Wei with validation."""
    try:
        gwei_val = Gwei(value)
        return gwei_val.to_wei()
    except (UnitValidationError, UnitOverflowError) as e:
        raise UnitValidationError(f"Failed to convert {value} gwei to wei: {e}")


def safe_wei_to_gwei(value: Union[int, float]) -> Gwei:
    """Safely convert numeric wei to Gwei with validation."""
    try:
        wei_val = Wei(value)
        return wei_val.to_gwei()
    except (UnitValidationError, UnitOverflowError) as e:
        raise UnitValidationError(f"Failed to convert {value} wei to gwei: {e}")


# Validation and assertion helpers
def validate_fee_range(fee_gwei: Union[Gwei, float], context: str = "fee calculation") -> None:
    """Validate that fee is in reasonable range."""
    if isinstance(fee_gwei, Gwei):
        fee_value = fee_gwei.value
    else:
        fee_value = float(fee_gwei)

    if fee_value < MIN_REASONABLE_FEE_GWEI:
        raise UnitValidationError(
            f"Fee too low in {context}: {fee_value:.6f} gwei "
            f"(minimum reasonable: {MIN_REASONABLE_FEE_GWEI} gwei). "
            f"This might indicate a unit conversion error."
        )

    if fee_value > MAX_REASONABLE_FEE_GWEI:
        raise UnitValidationError(
            f"Fee too high in {context}: {fee_value:.6f} gwei "
            f"(maximum reasonable: {MAX_REASONABLE_FEE_GWEI} gwei). "
            f"This might indicate a unit conversion error."
        )


def validate_basefee_range(basefee_gwei: Union[Gwei, float], context: str = "basefee") -> None:
    """Validate that L1 basefee is in reasonable range."""
    if isinstance(basefee_gwei, Gwei):
        basefee_value = basefee_gwei.value
    else:
        basefee_value = float(basefee_gwei)

    if basefee_value < MIN_REASONABLE_BASEFEE_GWEI:
        warnings.warn(
            f"L1 basefee very low in {context}: {basefee_value:.6f} gwei "
            f"(historical minimum ~0.055 gwei)"
        )

    if basefee_value > MAX_REASONABLE_BASEFEE_GWEI:
        warnings.warn(
            f"L1 basefee very high in {context}: {basefee_value:.6f} gwei "
            f"(historical maximum ~1,352 gwei)"
        )


def assert_no_unit_mismatch(value: Union[float, int], expected_range: tuple,
                           expected_unit: str, context: str) -> None:
    """Assert that a value is in expected range for its alleged unit."""
    min_val, max_val = expected_range

    if not (min_val <= value <= max_val):
        raise UnitValidationError(
            f"Value {value} outside expected range for {expected_unit} "
            f"in {context}: expected [{min_val}, {max_val}]. "
            f"This might indicate a unit mismatch (e.g., passing ETH when wei expected)."
        )


def assert_reasonable_fee(fee_gwei: Union[float, Gwei], context: str = "fee result") -> None:
    """Assert that a fee value is reasonable (not suspiciously near zero)."""
    if isinstance(fee_gwei, Gwei):
        fee_value = fee_gwei.value
    else:
        fee_value = float(fee_gwei)

    # Check for suspiciously small fees (likely unit errors)
    if 0 < fee_value < 0.0001:  # Between 0 and 0.0001 gwei is suspicious
        raise UnitValidationError(
            f"Fee suspiciously small in {context}: {fee_value:.8f} gwei. "
            f"This likely indicates a unit conversion error (ETH passed as wei?)."
        )

    # Check for exactly zero fees (might be intentional or error)
    if fee_value == 0:
        warnings.warn(f"Fee is exactly zero in {context}. Verify this is intentional.")


# Debug helpers
def diagnose_unit_mismatch(value: Union[int, float], alleged_unit: str) -> str:
    """Diagnose potential unit mismatches and suggest corrections."""
    msg = f"Diagnosing value {value} alleged to be {alleged_unit}:\n"

    if alleged_unit.lower() == "gwei":
        msg += f"  - As gwei: {value:.6f} (reasonable: {MIN_REASONABLE_FEE_GWEI}-{MAX_REASONABLE_FEE_GWEI})\n"
        msg += f"  - If actually wei: {value/1e9:.6f} gwei\n"
        msg += f"  - If actually ETH: {value*1e9:.6f} gwei\n"

        if value < 0.001:
            msg += "  ⚠️  LIKELY ISSUE: Too small for gwei, might be ETH\n"
        elif value > 10000:
            msg += "  ⚠️  LIKELY ISSUE: Too large for gwei, might be wei\n"

    elif alleged_unit.lower() == "wei":
        msg += f"  - As wei: {value:,}\n"
        msg += f"  - If actually gwei: {value*1e9:,} wei\n"
        msg += f"  - If actually ETH: {value*1e18:,} wei\n"

        if value < 1e15:  # Less than 0.001 ETH in wei
            msg += "  ⚠️  POSSIBLE ISSUE: Very small for wei, might be ETH or gwei\n"

    elif alleged_unit.lower() == "eth":
        msg += f"  - As ETH: {value:.6f}\n"
        msg += f"  - If actually wei: {value/1e18:.6f} ETH\n"
        msg += f"  - If actually gwei: {value/1e9:.6f} ETH\n"

        if value > 1.0:
            msg += "  ⚠️  POSSIBLE ISSUE: Very large for ETH (fee context)\n"
        elif value < 1e-10:
            msg += "  ⚠️  LIKELY ISSUE: Too small for ETH, might be wei\n"

    return msg


def create_unit_safety_report() -> str:
    """Create a report showing unit safety measures status."""
    report = "🛡️  Unit Safety System Status\n"
    report += "=" * 40 + "\n\n"

    report += "✅ Type-safe unit classes (Wei, Gwei, ETH)\n"
    report += "✅ Automatic validation on creation\n"
    report += "✅ Overflow protection in conversions\n"
    report += "✅ Reasonable range checking\n"
    report += "✅ Clear error messages for debugging\n"
    report += "✅ Arithmetic operations with type safety\n"
    report += "✅ Legacy conversion helpers for migration\n"
    report += "✅ Assertion helpers for runtime validation\n"
    report += "✅ Diagnostic tools for troubleshooting\n\n"

    report += f"Configured ranges:\n"
    report += f"  - Fee range: {MIN_REASONABLE_FEE_GWEI}-{MAX_REASONABLE_FEE_GWEI} gwei\n"
    report += f"  - Basefee range: {MIN_REASONABLE_BASEFEE_GWEI}-{MAX_REASONABLE_BASEFEE_GWEI} gwei\n"

    return report


# Export key functions and classes
__all__ = [
    # Core unit classes
    'Wei', 'Gwei', 'ETH',

    # Conversion functions
    'gwei_to_wei', 'wei_to_gwei', 'eth_to_wei', 'wei_to_eth',
    'safe_gwei_to_wei', 'safe_wei_to_gwei',

    # Validation helpers
    'validate_fee_range', 'validate_basefee_range',
    'assert_no_unit_mismatch', 'assert_reasonable_fee',

    # Debug helpers
    'diagnose_unit_mismatch', 'create_unit_safety_report',

    # Exceptions
    'UnitValidationError', 'UnitOverflowError'
]