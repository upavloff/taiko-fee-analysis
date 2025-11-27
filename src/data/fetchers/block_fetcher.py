"""
High-level block data fetching utilities.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional
import logging

from .ethereum_rpc_client import EthereumRPCClient

logger = logging.getLogger(__name__)


class BlockFetcher:
    """
    High-level interface for fetching Ethereum block data.
    """

    # Default reliable public RPC endpoints
    DEFAULT_RPC_URLS = [
        "https://eth.llamarpc.com",
        "https://ethereum-rpc.publicnode.com",
        "https://ethereum.blockpi.network/v1/rpc/public",
        "https://rpc.ankr.com/eth",
        "https://cloudflare-eth.com"
    ]

    def __init__(self, rpc_urls: Optional[List[str]] = None):
        """
        Initialize the block fetcher.

        Args:
            rpc_urls: Custom RPC URLs. Uses defaults if None.
        """
        if rpc_urls is None:
            rpc_urls = self.DEFAULT_RPC_URLS

        self.rpc_client = EthereumRPCClient(rpc_urls)

    def fetch_contiguous_blocks(
        self,
        start_block: int,
        count: int,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Fetch a contiguous range of blocks.

        Args:
            start_block: Starting block number
            count: Number of blocks to fetch
            progress_callback: Optional progress callback

        Returns:
            DataFrame with columns: timestamp, basefee_wei, basefee_gwei, block_number
        """
        logger.info(f"Fetching {count} contiguous blocks starting from {start_block}")

        end_block = start_block + count - 1
        blocks_data = self.rpc_client.batch_get_blocks(
            start_block,
            end_block,
            progress_callback=progress_callback
        )

        if not blocks_data:
            logger.error("Failed to fetch any block data")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(blocks_data)

        # Add derived columns
        df['basefee_gwei'] = df['basefee_wei'] / 1e9
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        logger.info(f"Successfully fetched {len(df)} blocks")
        return df

    def fetch_recent_blocks(
        self,
        hours: int,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Fetch recent blocks for the specified number of hours.

        Args:
            hours: Number of recent hours of blocks to fetch
            progress_callback: Optional progress callback

        Returns:
            DataFrame with recent block data
        """
        # Ethereum average block time is ~12 seconds
        # blocks_per_hour = 3600 / 12 = 300
        blocks_per_hour = 300
        total_blocks = hours * blocks_per_hour

        latest_block = self.rpc_client.get_latest_block_number()
        if latest_block is None:
            logger.error("Failed to get latest block number")
            return pd.DataFrame()

        start_block = latest_block - total_blocks + 1

        return self.fetch_contiguous_blocks(
            start_block,
            total_blocks,
            progress_callback
        )

    def save_to_csv(
        self,
        df: pd.DataFrame,
        filepath: str,
        description: str = ""
    ) -> bool:
        """
        Save block data to CSV file.

        Args:
            df: DataFrame to save
            filepath: Output file path
            description: Optional description for logging

        Returns:
            True if successful, False otherwise
        """
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} blocks to {filepath}")
            if description:
                logger.info(f"Dataset: {description}")
            return True
        except Exception as e:
            logger.error(f"Failed to save to {filepath}: {e}")
            return False