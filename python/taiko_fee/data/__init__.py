"""
Data management modules for Taiko fee analysis.

This module provides utilities for:
- Fetching blockchain data from RPC endpoints
- Validating data contiguity and completeness
- Managing real historical data
"""

from .rpc_data_fetcher import RPCBasefeeeFetcher, ImprovedRealDataIntegrator
from .real_data_fetcher import *
from .fetchers import EthereumRPCClient, BlockFetcher
from .validators import ContiguityAnalyzer

__all__ = [
    'RPCBasefeeeFetcher',
    'ImprovedRealDataIntegrator',
    'EthereumRPCClient',
    'BlockFetcher',
    'ContiguityAnalyzer',
]