#!/usr/bin/env python3
"""
Robust Ethereum Block Data Fetcher for July 2022 Fee Spike Analysis

This script fetches complete block-by-block Ethereum data for blocks 15055000-15064900
(July 2022 fee spike period) with robust error handling, multiple RPC endpoints,
retry logic, and resume capability.

Usage:
    python fetch_ethereum_blocks.py --start-block 15055000 --end-block 15064900
    python fetch_ethereum_blocks.py --test  # Test mode with small range
    python fetch_ethereum_blocks.py --resume  # Resume from last checkpoint
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class EthereumBlockFetcher:
    """Robust Ethereum block data fetcher with multiple RPC endpoints and retry logic."""

    # Multiple RPC endpoints for redundancy
    RPC_ENDPOINTS = [
        "https://eth.llamarpc.com",
        "https://rpc.ankr.com/eth",
        "https://ethereum.publicnode.com",
        "https://eth.rpc.blxrbdn.com",
        "https://cloudflare-eth.com",
        "https://nodes.mewapi.io/rpc/eth",
        "https://main-rpc.linkpool.io",
        "https://rpc.flashbots.net"
    ]

    def __init__(self, output_file: str, checkpoint_file: str = None):
        """Initialize the fetcher with output file and checkpoint handling."""
        self.output_file = Path(output_file)
        self.checkpoint_file = Path(checkpoint_file) if checkpoint_file else self.output_file.with_suffix('.checkpoint.json')

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_file.with_suffix('.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Setup HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Rate limiting
        self.min_delay = 0.1  # Minimum delay between requests
        self.max_delay = 5.0  # Maximum delay for backoff
        self.current_delay = self.min_delay

        # Current RPC endpoint index
        self.current_rpc_index = 0

    def get_current_rpc_url(self) -> str:
        """Get the current RPC endpoint URL."""
        return self.RPC_ENDPOINTS[self.current_rpc_index]

    def rotate_rpc_endpoint(self):
        """Rotate to the next RPC endpoint."""
        self.current_rpc_index = (self.current_rpc_index + 1) % len(self.RPC_ENDPOINTS)
        self.logger.info(f"Rotated to RPC endpoint: {self.get_current_rpc_url()}")

    def make_rpc_call(self, method: str, params: List, max_retries: int = 3) -> Optional[Dict]:
        """Make an RPC call with retry logic and endpoint rotation."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }

        for attempt in range(max_retries):
            try:
                # Rate limiting
                time.sleep(self.current_delay)

                response = self.session.post(
                    self.get_current_rpc_url(),
                    json=payload,
                    timeout=30,
                    headers={'Content-Type': 'application/json'}
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'result' in data:
                        # Success - reset delay
                        self.current_delay = self.min_delay
                        return data['result']
                    elif 'error' in data:
                        self.logger.warning(f"RPC error: {data['error']}")
                        if 'rate limit' in str(data['error']).lower():
                            self.current_delay = min(self.current_delay * 2, self.max_delay)

                elif response.status_code == 429:  # Rate limited
                    self.logger.warning(f"Rate limited by {self.get_current_rpc_url()}")
                    self.current_delay = min(self.current_delay * 2, self.max_delay)
                    time.sleep(self.current_delay)

                else:
                    self.logger.warning(f"HTTP {response.status_code} from {self.get_current_rpc_url()}")

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed to {self.get_current_rpc_url()}: {e}")

            # If we've tried multiple times with current endpoint, rotate
            if attempt == max_retries - 1:
                self.rotate_rpc_endpoint()
                # Add exponential backoff
                backoff_time = min(2 ** attempt, 30)
                self.logger.info(f"Backing off for {backoff_time} seconds...")
                time.sleep(backoff_time)

        return None

    def get_block_data(self, block_number: int) -> Optional[Tuple[str, str, str, str]]:
        """Fetch block data and return (timestamp, basefee_wei, basefee_gwei, block_number_hex)."""
        # Convert to hex
        block_hex = hex(block_number)

        # Fetch block data
        block_data = self.make_rpc_call("eth_getBlockByNumber", [block_hex, False])

        if not block_data:
            self.logger.error(f"Failed to fetch block {block_number}")
            return None

        try:
            # Extract timestamp and base fee
            timestamp_unix = int(block_data['timestamp'], 16)
            timestamp_str = datetime.fromtimestamp(timestamp_unix).strftime('%Y-%m-%d %H:%M:%S')

            # Base fee (EIP-1559 introduced in London hard fork)
            if 'baseFeePerGas' in block_data:
                basefee_wei = int(block_data['baseFeePerGas'], 16)
                basefee_gwei = basefee_wei / 1e9
            else:
                # Pre-EIP-1559 blocks (shouldn't happen for July 2022, but just in case)
                basefee_wei = 0
                basefee_gwei = 0.0

            return timestamp_str, str(basefee_wei), str(basefee_gwei), block_hex

        except (KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse block {block_number}: {e}")
            return None

    def save_checkpoint(self, last_block: int, total_blocks: int):
        """Save progress checkpoint."""
        checkpoint_data = {
            'last_completed_block': last_block,
            'total_blocks': total_blocks,
            'timestamp': datetime.now().isoformat(),
            'output_file': str(self.output_file)
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_checkpoint(self) -> Optional[Dict]:
        """Load progress checkpoint."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.logger.warning("Invalid checkpoint file, starting fresh")
        return None

    def fetch_blocks(self, start_block: int, end_block: int, resume: bool = False) -> bool:
        """Fetch blocks from start_block to end_block (inclusive)."""
        total_blocks = end_block - start_block + 1

        # Handle resume functionality
        actual_start_block = start_block
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                actual_start_block = checkpoint['last_completed_block'] + 1
                self.logger.info(f"Resuming from block {actual_start_block}")

        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Determine if we're creating a new file or appending
        write_header = not self.output_file.exists() or not resume

        self.logger.info(f"Starting to fetch blocks {actual_start_block} to {end_block}")
        self.logger.info(f"Total blocks to fetch: {end_block - actual_start_block + 1}")
        self.logger.info(f"Output file: {self.output_file}")

        # Open output file
        mode = 'w' if write_header else 'a'
        with open(self.output_file, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header if creating new file
            if write_header:
                writer.writerow(['timestamp', 'basefee_wei', 'basefee_gwei', 'block_number'])

            # Fetch blocks
            blocks_processed = 0
            last_progress_time = time.time()

            for block_num in range(actual_start_block, end_block + 1):
                # Fetch block data
                block_data = self.get_block_data(block_num)

                if block_data:
                    writer.writerow(block_data)
                    csvfile.flush()  # Ensure data is written immediately
                    blocks_processed += 1

                    # Progress reporting
                    if blocks_processed % 100 == 0 or time.time() - last_progress_time > 30:
                        progress = (block_num - start_block + 1) / total_blocks * 100
                        blocks_per_sec = blocks_processed / (time.time() - last_progress_time) if time.time() - last_progress_time > 0 else 0
                        eta_seconds = (end_block - block_num) / blocks_per_sec if blocks_per_sec > 0 else 0
                        eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"

                        self.logger.info(f"Progress: {progress:.1f}% | Block {block_num} | "
                                       f"Rate: {blocks_per_sec:.1f} blocks/sec | ETA: {eta_str}")
                        last_progress_time = time.time()
                        blocks_processed = 0

                    # Save checkpoint every 500 blocks
                    if block_num % 500 == 0:
                        self.save_checkpoint(block_num, total_blocks)

                else:
                    self.logger.error(f"Failed to fetch block {block_num}, stopping...")
                    return False

            # Final checkpoint
            self.save_checkpoint(end_block, total_blocks)

        self.logger.info(f"Successfully fetched all blocks from {start_block} to {end_block}")
        self.logger.info(f"Output saved to: {self.output_file}")

        # Clean up checkpoint file on successful completion
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

        return True


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description='Fetch Ethereum block data for July 2022 fee spike analysis')
    parser.add_argument('--start-block', type=int, default=15055000,
                       help='Starting block number (default: 15055000)')
    parser.add_argument('--end-block', type=int, default=15064900,
                       help='Ending block number (default: 15064900)')
    parser.add_argument('--output', type=str,
                       default='/Users/ulyssepavloff/Desktop/Nethermind/taiko-fee-analysis/data/data_cache/real_july_2022_spike_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: fetch only 20 blocks for testing')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')

    args = parser.parse_args()

    # Test mode - override block range
    if args.test:
        args.start_block = 15055000
        args.end_block = 15055019  # Just 20 blocks for testing
        args.output = str(Path(args.output).with_name('test_blocks.csv'))
        print(f"TEST MODE: Fetching blocks {args.start_block} to {args.end_block}")

    # Create fetcher and run
    fetcher = EthereumBlockFetcher(args.output)

    try:
        success = fetcher.fetch_blocks(args.start_block, args.end_block, args.resume)
        if success:
            print(f"\n✅ Successfully fetched blocks {args.start_block} to {args.end_block}")
            print(f"📁 Output saved to: {args.output}")
        else:
            print(f"\n❌ Failed to fetch all blocks")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⏹️ Interrupted by user. Progress saved to checkpoint.")
        print(f"   Resume with: python {sys.argv[0]} --resume")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()