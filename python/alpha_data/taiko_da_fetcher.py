"""
Taiko L1 Data Availability Fetcher

Fetches and analyzes Taiko L1 proposeBlock transactions to extract empirical
DA cost data for α_data calculation.

Key Functions:
- Fetch proposeBlock transactions from Taiko L1 contract
- Separate DA posting from proof verification transactions
- Extract gas usage for DA transactions only
- Map L1 DA transactions to corresponding L2 batch data
"""

import asyncio
import time
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
from web3 import Web3
from web3.middleware import geth_poa_middleware
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DATransaction:
    """Data structure for a DA transaction"""
    tx_hash: str
    block_number: int
    timestamp: int
    gas_used: int
    gas_price: int
    l1_cost_wei: int
    data_size: int
    is_blob_mode: bool
    l2_batch_id: Optional[int] = None
    l2_gas_used: Optional[int] = None


@dataclass
class AlphaDataPoint:
    """Data structure for a calculated alpha data point"""
    batch_id: int
    alpha_value: float  # L1 DA gas per L2 gas
    l1_da_gas: int
    l2_gas: int
    timestamp: int
    is_blob_mode: bool
    confidence: float  # Confidence in the measurement


class TaikoDAFetcher:
    """
    Fetcher for Taiko L1 DA transactions and alpha calculation
    """

    # Taiko L1 contract address
    TAIKO_L1_CONTRACT = "0xe84dc8e2a21e59426542ab040d77f81d6db881ee"

    # proposeBlock function signature
    PROPOSE_BLOCK_SIG = "0x092bfe76"  # proposeBlock(bytes,bytes)

    # RPC endpoints (add more for redundancy)
    DEFAULT_RPC_ENDPOINTS = [
        "https://eth.llamarpc.com",
        "https://rpc.ankr.com/eth",
        "https://ethereum.publicnode.com"
    ]

    def __init__(
        self,
        rpc_endpoints: Optional[List[str]] = None,
        rate_limit_delay: float = 0.1,
        max_retries: int = 3,
        cache_dir: str = "./data_cache"
    ):
        """
        Initialize the Taiko DA fetcher

        Args:
            rpc_endpoints: List of Ethereum RPC endpoints
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum number of retries for failed requests
            cache_dir: Directory for caching fetched data
        """
        self.rpc_endpoints = rpc_endpoints or self.DEFAULT_RPC_ENDPOINTS
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.cache_dir = cache_dir

        # Initialize Web3 connections
        self.w3_connections = []
        for endpoint in self.rpc_endpoints:
            try:
                w3 = Web3(Web3.HTTPProvider(endpoint))
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                if w3.is_connected():
                    self.w3_connections.append(w3)
                    logger.info(f"Connected to RPC endpoint: {endpoint}")
                else:
                    logger.warning(f"Failed to connect to: {endpoint}")
            except Exception as e:
                logger.warning(f"Error connecting to {endpoint}: {e}")

        if not self.w3_connections:
            raise ConnectionError("No RPC endpoints available")

        # Current connection index for round-robin
        self.current_connection = 0

    def get_web3(self) -> Web3:
        """Get current Web3 connection with round-robin failover"""
        if not self.w3_connections:
            raise ConnectionError("No Web3 connections available")

        w3 = self.w3_connections[self.current_connection]
        self.current_connection = (self.current_connection + 1) % len(self.w3_connections)
        return w3

    async def fetch_da_transactions(
        self,
        start_block: int,
        end_block: int,
        save_progress: bool = True
    ) -> List[DATransaction]:
        """
        Fetch DA transactions from Taiko L1 contract

        Args:
            start_block: Starting block number
            end_block: Ending block number
            save_progress: Whether to save progress for resume capability

        Returns:
            List of DA transactions
        """
        logger.info(f"Fetching DA transactions from blocks {start_block} to {end_block}")

        transactions = []
        blocks_processed = 0
        total_blocks = end_block - start_block + 1

        for block_num in range(start_block, end_block + 1):
            try:
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)

                # Get block transactions
                block_txs = await self._fetch_block_transactions(block_num)

                # Filter for proposeBlock transactions
                da_txs = self._filter_da_transactions(block_txs, block_num)
                transactions.extend(da_txs)

                blocks_processed += 1
                if blocks_processed % 100 == 0:
                    progress = (blocks_processed / total_blocks) * 100
                    logger.info(f"Progress: {progress:.1f}% ({blocks_processed}/{total_blocks} blocks)")

                # Save checkpoint every 1000 blocks
                if save_progress and blocks_processed % 1000 == 0:
                    self._save_checkpoint(transactions, block_num)

            except Exception as e:
                logger.error(f"Error processing block {block_num}: {e}")
                continue

        logger.info(f"Fetched {len(transactions)} DA transactions from {blocks_processed} blocks")
        return transactions

    async def _fetch_block_transactions(self, block_number: int) -> List[Dict]:
        """Fetch all transactions from a specific block"""
        for attempt in range(self.max_retries):
            try:
                w3 = self.get_web3()
                block = w3.eth.get_block(block_number, full_transactions=True)
                return block.transactions
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for block {block_number}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

    def _filter_da_transactions(self, transactions: List[Dict], block_number: int) -> List[DATransaction]:
        """Filter transactions for Taiko proposeBlock calls"""
        da_transactions = []

        for tx in transactions:
            # Check if transaction is to Taiko L1 contract
            if (tx.to and
                tx.to.lower() == self.TAIKO_L1_CONTRACT.lower() and
                tx.input.startswith(self.PROPOSE_BLOCK_SIG)):

                # Parse transaction data
                da_tx = self._parse_da_transaction(tx, block_number)
                if da_tx:
                    da_transactions.append(da_tx)

        return da_transactions

    def _parse_da_transaction(self, tx: Dict, block_number: int) -> Optional[DATransaction]:
        """Parse a proposeBlock transaction to extract DA data"""
        try:
            # Get transaction receipt for gas usage
            w3 = self.get_web3()
            receipt = w3.eth.get_transaction_receipt(tx.hash)

            # Determine if this is blob mode (EIP-4844) or calldata mode
            is_blob_mode = self._is_blob_transaction(tx)

            # Calculate L1 cost
            l1_cost_wei = receipt.gasUsed * tx.gasPrice

            # Get data size (for calldata mode)
            data_size = len(tx.input) if tx.input else 0

            # Create DA transaction object
            da_tx = DATransaction(
                tx_hash=tx.hash.hex(),
                block_number=block_number,
                timestamp=0,  # Will be filled later
                gas_used=receipt.gasUsed,
                gas_price=tx.gasPrice,
                l1_cost_wei=l1_cost_wei,
                data_size=data_size,
                is_blob_mode=is_blob_mode
            )

            return da_tx

        except Exception as e:
            logger.warning(f"Error parsing transaction {tx.hash.hex()}: {e}")
            return None

    def _is_blob_transaction(self, tx: Dict) -> bool:
        """Determine if transaction uses EIP-4844 blob mode"""
        # Check for blob-related fields (EIP-4844)
        return (hasattr(tx, 'maxFeePerBlobGas') and tx.maxFeePerBlobGas is not None) or \
               (hasattr(tx, 'blobVersionedHashes') and tx.blobVersionedHashes is not None)

    def _save_checkpoint(self, transactions: List[DATransaction], current_block: int):
        """Save progress checkpoint for resume capability"""
        checkpoint_file = f"{self.cache_dir}/taiko_da_checkpoint_{current_block}.json"

        checkpoint_data = {
            'current_block': current_block,
            'transactions_count': len(transactions),
            'timestamp': datetime.now().isoformat(),
            'transactions': [
                {
                    'tx_hash': tx.tx_hash,
                    'block_number': tx.block_number,
                    'gas_used': tx.gas_used,
                    'l1_cost_wei': tx.l1_cost_wei,
                    'is_blob_mode': tx.is_blob_mode
                } for tx in transactions
            ]
        }

        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_file: str) -> Tuple[List[DATransaction], int]:
        """Load progress from checkpoint file"""
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            transactions = []
            for tx_data in checkpoint_data['transactions']:
                tx = DATransaction(
                    tx_hash=tx_data['tx_hash'],
                    block_number=tx_data['block_number'],
                    timestamp=0,
                    gas_used=tx_data['gas_used'],
                    gas_price=0,
                    l1_cost_wei=tx_data['l1_cost_wei'],
                    data_size=0,
                    is_blob_mode=tx_data['is_blob_mode']
                )
                transactions.append(tx)

            current_block = checkpoint_data['current_block']
            logger.info(f"Loaded checkpoint with {len(transactions)} transactions from block {current_block}")
            return transactions, current_block

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return [], 0

    async def fetch_recent_data(self, days: int = 30) -> List[DATransaction]:
        """Fetch recent DA transactions for the specified number of days"""
        w3 = self.get_web3()
        latest_block = w3.eth.block_number

        # Estimate blocks for the specified days (assuming ~12 second block time)
        blocks_per_day = 24 * 60 * 60 // 12
        start_block = latest_block - (days * blocks_per_day)

        logger.info(f"Fetching {days} days of recent data from block {start_block} to {latest_block}")
        return await self.fetch_da_transactions(start_block, latest_block)

    def export_to_csv(self, transactions: List[DATransaction], filename: str):
        """Export transactions to CSV for further analysis"""
        data = []
        for tx in transactions:
            data.append({
                'tx_hash': tx.tx_hash,
                'block_number': tx.block_number,
                'timestamp': tx.timestamp,
                'gas_used': tx.gas_used,
                'gas_price': tx.gas_price,
                'l1_cost_wei': tx.l1_cost_wei,
                'l1_cost_eth': tx.l1_cost_wei / 1e18,
                'data_size': tx.data_size,
                'is_blob_mode': tx.is_blob_mode,
                'l2_batch_id': tx.l2_batch_id,
                'l2_gas_used': tx.l2_gas_used
            })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(data)} transactions to {filename}")


# Utility functions for quick analysis
async def quick_alpha_analysis(days: int = 7) -> Dict[str, Any]:
    """Quick analysis of recent alpha data"""
    fetcher = TaikoDAFetcher()

    logger.info(f"Running quick alpha analysis for last {days} days...")
    transactions = await fetcher.fetch_recent_data(days)

    if not transactions:
        logger.warning("No transactions found")
        return {}

    # Basic statistics
    gas_usage = [tx.gas_used for tx in transactions]
    l1_costs = [tx.l1_cost_wei for tx in transactions]
    blob_mode_count = sum(1 for tx in transactions if tx.is_blob_mode)

    analysis = {
        'total_transactions': len(transactions),
        'blob_mode_percentage': (blob_mode_count / len(transactions)) * 100,
        'avg_gas_per_tx': sum(gas_usage) / len(gas_usage),
        'avg_l1_cost_wei': sum(l1_costs) / len(l1_costs),
        'avg_l1_cost_eth': (sum(l1_costs) / len(l1_costs)) / 1e18,
        'days_analyzed': days
    }

    logger.info(f"Analysis complete: {analysis}")
    return analysis


if __name__ == "__main__":
    # Example usage
    async def main():
        # Quick 7-day analysis
        analysis = await quick_alpha_analysis(7)
        print(f"Recent Alpha Analysis: {analysis}")

    asyncio.run(main())