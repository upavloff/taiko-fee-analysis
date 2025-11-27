"""
Robust Ethereum RPC client with retry logic and URL rotation.
"""

import requests
import time
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EthereumRPCClient:
    """
    Robust Ethereum RPC client with automatic failover and retry logic.

    Features:
    - Multiple RPC URL support with automatic rotation
    - Configurable retry logic with exponential backoff
    - Request rate limiting
    - Comprehensive error handling
    """

    def __init__(
        self,
        rpc_urls: List[str],
        request_timeout: int = 30,
        rate_limit_delay: float = 0.1
    ):
        """
        Initialize the RPC client.

        Args:
            rpc_urls: List of Ethereum RPC endpoints
            request_timeout: Timeout for individual requests in seconds
            rate_limit_delay: Delay between requests in seconds
        """
        self.rpc_urls = rpc_urls
        self.current_url_index = 0
        self.request_timeout = request_timeout
        self.rate_limit_delay = rate_limit_delay

    def _get_current_rpc_url(self) -> str:
        """Get the current RPC URL."""
        return self.rpc_urls[self.current_url_index % len(self.rpc_urls)]

    def _rotate_rpc_url(self):
        """Rotate to the next RPC URL."""
        self.current_url_index = (self.current_url_index + 1) % len(self.rpc_urls)
        logger.debug(f"Rotated to RPC URL: {self._get_current_rpc_url()}")

    def make_rpc_call(
        self,
        method: str,
        params: list,
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ) -> Optional[Dict]:
        """
        Make an RPC call with retry logic and URL rotation.

        Args:
            method: RPC method name
            params: RPC method parameters
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier

        Returns:
            RPC response result or None if all attempts failed
        """
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }

        total_attempts = max_retries * len(self.rpc_urls)

        for attempt in range(total_attempts):
            try:
                url = self._get_current_rpc_url()

                response = requests.post(
                    url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=self.request_timeout
                )
                response.raise_for_status()

                result = response.json()

                if 'error' in result:
                    logger.warning(f"RPC error from {url}: {result['error']}")
                    self._rotate_rpc_url()
                    time.sleep(backoff_factor * (attempt + 1))
                    continue

                # Rate limiting
                time.sleep(self.rate_limit_delay)
                return result.get('result')

            except Exception as e:
                logger.warning(f"Request failed to {url}: {e}")
                self._rotate_rpc_url()
                delay = backoff_factor * (2 ** min(attempt, 5))  # Cap exponential growth
                time.sleep(delay)

        logger.error(f"All RPC attempts failed for method {method}")
        return None

    def get_block(self, block_number: int) -> Optional[Dict]:
        """
        Fetch block data by block number.

        Args:
            block_number: Block number to fetch

        Returns:
            Block data dictionary or None if failed
        """
        hex_block = hex(block_number)
        return self.make_rpc_call("eth_getBlockByNumber", [hex_block, False])

    def get_latest_block_number(self) -> Optional[int]:
        """
        Get the latest block number.

        Returns:
            Latest block number or None if failed
        """
        result = self.make_rpc_call("eth_getBlockByNumber", ["latest", False])
        if result:
            return int(result['number'], 16)
        return None

    def batch_get_blocks(
        self,
        start_block: int,
        end_block: int,
        batch_size: int = 50,
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Fetch multiple blocks efficiently with progress tracking.

        Args:
            start_block: Starting block number
            end_block: Ending block number (inclusive)
            batch_size: Number of blocks to process in each batch
            progress_callback: Optional callback for progress updates

        Returns:
            List of block data dictionaries
        """
        blocks = []
        total_blocks = end_block - start_block + 1

        for i in range(start_block, end_block + 1, batch_size):
            batch_end = min(i + batch_size - 1, end_block)

            logger.info(f"Fetching blocks {i} to {batch_end}")

            batch_blocks = []
            for block_num in range(i, batch_end + 1):
                block_data = self.get_block(block_num)
                if block_data:
                    batch_blocks.append({
                        'block_number': block_num,
                        'timestamp': int(block_data['timestamp'], 16),
                        'basefee_wei': int(block_data.get('baseFeePerGas', '0x0'), 16)
                    })
                else:
                    logger.warning(f"Failed to fetch block {block_num}")

            blocks.extend(batch_blocks)

            # Progress callback
            if progress_callback:
                progress = (batch_end - start_block + 1) / total_blocks * 100
                progress_callback(progress, batch_end - start_block + 1, total_blocks)

            # Inter-batch delay for rate limiting
            time.sleep(0.5)

        return blocks