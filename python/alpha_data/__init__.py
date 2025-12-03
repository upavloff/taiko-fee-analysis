"""
Alpha-Data Based Fee Vault Implementation

This module implements empirically-measured α_data (L1 DA gas per L2 gas) to replace
the crude Q̄ constant in the Taiko fee mechanism.

Key Components:
- taiko_da_fetcher: Fetch Taiko L1 contract data for DA transactions
- alpha_calculator: Calculate α_data statistics from transaction data
- validation: Validate the new model against historical scenarios

Background:
The current Q̄ = 690,000 gas/batch parameter conflates proof gas with DA gas,
leading to 0.00 gwei fees (hitting minimum bounds). This module replaces it with
principled empirical measurement of actual DA costs.
"""

from .taiko_da_fetcher import TaikoDAFetcher
from .alpha_calculator import AlphaCalculator
from .validation import AlphaDataValidator

__all__ = [
    'TaikoDAFetcher',
    'AlphaCalculator',
    'AlphaDataValidator'
]