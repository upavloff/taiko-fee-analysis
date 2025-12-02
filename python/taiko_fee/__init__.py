"""
Taiko Fee Mechanism Analysis Package

A comprehensive toolkit for analyzing and optimizing Taiko's EIP-1559 based
fee mechanism using real Ethereum L1 data.

Main components:
- core: Fee mechanism simulation engine
- data: Data fetching and processing utilities
- analysis: Performance metrics and evaluation tools
- utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "Taiko Fee Research Team"

# Import main classes for convenient access
from .core.fee_mechanism_simulator import GeometricBrownianMotion

__all__ = [
    "GeometricBrownianMotion"
]