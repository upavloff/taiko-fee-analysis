#!/usr/bin/env python3
"""
Robust Ethereum Block Data Fetcher for May 2023 PEPE Memecoin Crisis Analysis

This script fetches complete block-by-block Ethereum data for blocks 17,180,000-17,220,000
(May 3-7, 2023 PEPE memecoin crisis period) with robust error handling, multiple RPC endpoints,
retry logic, and resume capability.

The PEPE memecoin crisis on May 5, 2023 saw basefees spike to 184+ gwei, providing an excellent
test case for the Taiko fee mechanism under extreme conditions.

Usage:
    python fetch_may_2023_pepe_crisis.py --start-block 17180000 --end-block 17220000
    python fetch_may_2023_pepe_crisis.py --test  # Test mode with small range
    python fetch_may_2023_pepe_crisis.py --resume  # Resume from last checkpoint
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


class EthereumPepeCrisisFetcher:
    """Robust Ethereum block data fetcher for May 2023 PEPE crisis with multiple RPC endpoints and retry logic."""

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

        # Statistics tracking
        self.max_basefee_gwei = 0.0
        self.max_basefee_block = 0
        self.total_high_fee_blocks = 0  # Count blocks > 100 gwei

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

            # Base fee (EIP-1559 - all blocks in May 2023 should have this)
            if 'baseFeePerGas' in block_data:
                basefee_wei = int(block_data['baseFeePerGas'], 16)
                basefee_gwei = basefee_wei / 1e9

                # Track statistics for PEPE crisis analysis
                if basefee_gwei > self.max_basefee_gwei:
                    self.max_basefee_gwei = basefee_gwei
                    self.max_basefee_block = block_number
                    self.logger.info(f"📈 New max basefee: {basefee_gwei:.2f} gwei at block {block_number}")

                if basefee_gwei > 100:
                    self.total_high_fee_blocks += 1
                    if basefee_gwei > 150:
                        self.logger.info(f"🔥 EXTREME fee detected: {basefee_gwei:.2f} gwei at block {block_number}")
            else:
                # This shouldn't happen for May 2023 blocks
                self.logger.warning(f"Missing baseFeePerGas in block {block_number}")
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
            'output_file': str(self.output_file),
            'max_basefee_gwei': self.max_basefee_gwei,
            'max_basefee_block': self.max_basefee_block,
            'total_high_fee_blocks': self.total_high_fee_blocks
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_checkpoint(self) -> Optional[Dict]:
        """Load progress checkpoint."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    # Restore statistics
                    self.max_basefee_gwei = checkpoint.get('max_basefee_gwei', 0.0)
                    self.max_basefee_block = checkpoint.get('max_basefee_block', 0)
                    self.total_high_fee_blocks = checkpoint.get('total_high_fee_blocks', 0)
                    return checkpoint
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

        self.logger.info("="*80)
        self.logger.info("🚀 STARTING MAY 2023 PEPE MEMECOIN CRISIS DATA FETCH")
        self.logger.info(f"📅 Target period: May 3-7, 2023 (PEPE crisis)")
        self.logger.info(f"🔢 Fetching blocks {actual_start_block} to {end_block}")
        self.logger.info(f"📊 Total blocks to fetch: {end_block - actual_start_block + 1}")
        self.logger.info(f"💾 Output file: {self.output_file}")
        self.logger.info(f"🎯 Expected peak: ~184 gwei on May 5, 2023")
        self.logger.info("="*80)

        # Open output file
        mode = 'w' if write_header else 'a'
        with open(self.output_file, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header if creating new file
            if write_header:
                writer.writerow(['timestamp', 'basefee_wei', 'basefee_gwei', 'block_number'])

            # Fetch blocks
            blocks_processed = 0
            start_time = time.time()
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
                        elapsed_time = time.time() - start_time
                        blocks_per_sec = (block_num - actual_start_block + 1) / elapsed_time if elapsed_time > 0 else 0
                        eta_seconds = (end_block - block_num) / blocks_per_sec if blocks_per_sec > 0 else 0
                        eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"

                        self.logger.info(f"📈 Progress: {progress:.1f}% | Block {block_num} | "
                                       f"Rate: {blocks_per_sec:.1f} blocks/sec | ETA: {eta_str}")
                        self.logger.info(f"🔥 Max fee so far: {self.max_basefee_gwei:.2f} gwei "
                                       f"(block {self.max_basefee_block}) | High fee blocks: {self.total_high_fee_blocks}")
                        last_progress_time = time.time()

                    # Save checkpoint every 500 blocks
                    if block_num % 500 == 0:
                        self.save_checkpoint(block_num, total_blocks)

                else:
                    self.logger.error(f"Failed to fetch block {block_num}, stopping...")
                    return False

            # Final checkpoint
            self.save_checkpoint(end_block, total_blocks)

        # Final statistics
        self.logger.info("="*80)
        self.logger.info("🎉 PEPE CRISIS DATA FETCH COMPLETED!")
        self.logger.info(f"📊 Fetched blocks: {start_block} to {end_block}")
        self.logger.info(f"📈 Maximum basefee: {self.max_basefee_gwei:.2f} gwei (block {self.max_basefee_block})")
        self.logger.info(f"🔥 High fee blocks (>100 gwei): {self.total_high_fee_blocks}")
        self.logger.info(f"💾 Output saved to: {self.output_file}")
        self.logger.info("="*80)

        # Clean up checkpoint file on successful completion
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

        return True


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description='Fetch Ethereum block data for May 2023 PEPE memecoin crisis analysis')
    parser.add_argument('--start-block', type=int, default=17180000,
                       help='Starting block number (default: 17180000 - May 3, 2023)')
    parser.add_argument('--end-block', type=int, default=17220000,
                       help='Ending block number (default: 17220000 - May 7, 2023)')
    parser.add_argument('--output', type=str,
                       default='/Users/ulyssepavloff/Desktop/Nethermind/taiko-fee-analysis/data/data_cache/may_2023_pepe_crisis_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: fetch only 20 blocks for testing')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')

    args = parser.parse_args()

    # Validate block range for PEPE crisis period
    if not args.test:
        if args.start_block < 17180000 or args.end_block > 17220000:
            print("⚠️  WARNING: Block range is outside the PEPE crisis period (17,180,000-17,220,000)")
            print("   This script is optimized for May 3-7, 2023 data.")

    # Test mode - override block range
    if args.test:
        args.start_block = 17200000  # Around peak crisis time
        args.end_block = 17200019   # Just 20 blocks for testing
        args.output = str(Path(args.output).with_name('test_pepe_crisis_blocks.csv'))
        print(f"🧪 TEST MODE: Fetching blocks {args.start_block} to {args.end_block}")
        print(f"   This should include some high-fee blocks from the PEPE crisis")

    # Create fetcher and run
    fetcher = EthereumPepeCrisisFetcher(args.output)

    try:
        success = fetcher.fetch_blocks(args.start_block, args.end_block, args.resume)
        if success:
            print(f"\n✅ Successfully fetched PEPE crisis blocks {args.start_block} to {args.end_block}")
            print(f"📁 Output saved to: {args.output}")
            print(f"🔥 Maximum basefee observed: {fetcher.max_basefee_gwei:.2f} gwei")
            print(f"🎯 Ready for Taiko fee mechanism stress testing!")
        else:
            print(f"\n❌ Failed to fetch all blocks")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⏹️  Interrupted by user. Progress saved to checkpoint.")
        print(f"   Resume with: python {sys.argv[0]} --resume")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()