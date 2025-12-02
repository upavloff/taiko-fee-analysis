"""
Taiko Fee Mechanism Implementation - Mathematical Specifications

This package implements the Taiko Fee Mechanism as specified in SPECS.md,
providing exact mathematical implementations of:

- Section 3: Fee Controller (raw basefee, clipping, rate limiting)
- Section 4: Vault Dynamics (subsidy rules, vault updates, deficit calculations)

All mathematical formulas follow the canonical specification exactly.
"""

__version__ = "1.0.0"

from .core.simulation_engine import SimulationEngine
from .core.fee_controller import FeeController
from .core.vault_dynamics import VaultDynamics
from .core.l1_cost_smoother import L1CostSmoother
from .data.loader import DataLoader

__all__ = [
    "SimulationEngine",
    "FeeController",
    "VaultDynamics",
    "L1CostSmoother",
    "DataLoader",
]