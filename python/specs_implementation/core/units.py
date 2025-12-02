"""
Type-safe unit system to prevent unit mismatch bugs

This module provides strongly-typed wrappers for different units used in the fee mechanism.
All financial calculations should use these types to prevent unit confusion.
"""

from typing import Union, NewType
from dataclasses import dataclass
import numpy as np


# === BASE UNIT TYPES ===
# These prevent accidental mixing of different units

Wei = NewType('Wei', int)           # Base unit: wei (10^-18 ETH)
Gwei = NewType('Gwei', float)       # 10^9 wei = 1 gwei
EthAmount = NewType('EthAmount', float)  # 1 ETH = 10^18 wei
GasAmount = NewType('GasAmount', int)    # Gas units


@dataclass(frozen=True)
class WeiPerGas:
    """Fee rate: wei per gas unit"""
    value: int

    def to_gwei_per_gas(self) -> float:
        """Convert to gwei per gas for display"""
        return self.value / 1e9

    def to_eth_per_gas(self) -> float:
        """Convert to ETH per gas (rarely used)"""
        return self.value / 1e18

    def __mul__(self, gas: GasAmount) -> Wei:
        """Calculate total cost: (wei/gas) * gas = wei"""
        return Wei(self.value * gas)

    def __str__(self) -> str:
        return f"{self.to_gwei_per_gas():.6f} gwei/gas"


@dataclass(frozen=True)
class WeiPerBatch:
    """L1 cost: wei per batch"""
    value: int

    def to_eth(self) -> EthAmount:
        """Convert to ETH"""
        return EthAmount(self.value / 1e18)

    def to_wei_per_gas(self, q_bar: GasAmount) -> WeiPerGas:
        """Convert to wei per gas using Q̄"""
        return WeiPerGas(self.value // q_bar)

    def __str__(self) -> str:
        return f"{self.value:,} wei/batch ({self.to_eth():.6f} ETH/batch)"


# === CONVERSION UTILITIES ===

def gwei_to_wei(gwei: Gwei) -> Wei:
    """Convert gwei to wei"""
    return Wei(int(gwei * 1e9))

def wei_to_gwei(wei: Wei) -> Gwei:
    """Convert wei to gwei"""
    return Gwei(wei / 1e9)

def eth_to_wei(eth: EthAmount) -> Wei:
    """Convert ETH to wei"""
    return Wei(int(eth * 1e18))

def wei_to_eth(wei: Wei) -> EthAmount:
    """Convert wei to ETH"""
    return EthAmount(wei / 1e18)

def l1_basefee_to_batch_cost(basefee_wei: Wei, gas_per_tx: int, txs_per_batch: int = 100) -> WeiPerBatch:
    """
    Convert L1 basefee to L1 cost per batch

    Args:
        basefee_wei: L1 basefee in wei per gas
        gas_per_tx: Gas consumed per transaction
        txs_per_batch: Transactions per batch (default: 100)

    Returns:
        L1 cost in wei per batch
    """
    total_gas = gas_per_tx * txs_per_batch
    total_cost_wei = basefee_wei * total_gas
    return WeiPerBatch(total_cost_wei)


# === VALIDATION FUNCTIONS ===

def validate_fee_range(fee: WeiPerGas, min_gwei: float = 0.0001, max_gwei: float = 1000) -> bool:
    """Validate fee is in reasonable range"""
    gwei_fee = fee.to_gwei_per_gas()
    return min_gwei <= gwei_fee <= max_gwei

def validate_l1_cost(cost: WeiPerBatch, max_eth: float = 0.1) -> bool:
    """Validate L1 cost is reasonable"""
    eth_cost = cost.to_eth()
    return 0 <= eth_cost <= max_eth


# === EXAMPLE USAGE ===

if __name__ == "__main__":
    # Example: Convert 50 gwei L1 basefee to batch cost
    l1_basefee = gwei_to_wei(Gwei(50.0))
    batch_cost = l1_basefee_to_batch_cost(l1_basefee, gas_per_tx=2000)

    print(f"L1 basefee: {wei_to_gwei(l1_basefee)} gwei")
    print(f"Batch cost: {batch_cost}")

    # Example: Calculate fee per gas
    q_bar = GasAmount(690_000)
    fee_rate = batch_cost.to_wei_per_gas(q_bar)
    print(f"Fee component: {fee_rate}")

    # Validation
    print(f"Fee valid: {validate_fee_range(fee_rate)}")
    print(f"L1 cost valid: {validate_l1_cost(batch_cost)}")