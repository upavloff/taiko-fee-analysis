"""
RPC-based L1 Basefee Data Fetcher

This module fetches historical Ethereum basefee data directly from RPC nodes,
which is more reliable than API-based approaches and has no rate limits.

Supports multiple RPC providers:
1. Public RPC endpoints (free)
2. Infura (with project ID)
3. Alchemy (with API key)
4. Custom RPC endpoints

Usage:
    python rpc_data_fetcher.py --period may_crash --provider infura --project-id YOUR_ID
"""

import json
import time
import requests
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class RPCProvider:
    """RPC provider configuration."""
    name: str
    url: str
    headers: Dict[str, str]
    rate_limit_delay: float  # Seconds between requests


class RPCBasefeeeFetcher:
    """Fetches basefee data directly from Ethereum RPC nodes."""

    # Public RPC endpoints (free but may have limitations)
    PUBLIC_PROVIDERS = {
        'ethereum_public': RPCProvider(
            name='Ethereum Public RPC',
            url='https://eth.llamarpc.com',
            headers={'Content-Type': 'application/json'},
            rate_limit_delay=0.1
        ),
        'cloudflare': RPCProvider(
            name='Cloudflare Ethereum',
            url='https://cloudflare-eth.com',
            headers={'Content-Type': 'application/json'},
            rate_limit_delay=0.1
        ),
        'ankr': RPCProvider(
            name='Ankr Public',
            url='https://rpc.ankr.com/eth',
            headers={'Content-Type': 'application/json'},
            rate_limit_delay=0.2
        )
    }

    def __init__(self, provider: str = 'ethereum_public',
                 api_key: Optional[str] = None,
                 project_id: Optional[str] = None,
                 custom_url: Optional[str] = None):

        self.provider_config = self._setup_provider(provider, api_key, project_id, custom_url)
        self.session = requests.Session()
        self.session.headers.update(self.provider_config.headers)

    def _setup_provider(self, provider: str, api_key: Optional[str],
                       project_id: Optional[str], custom_url: Optional[str]) -> RPCProvider:
        """Setup RPC provider configuration."""

        if custom_url:
            return RPCProvider(
                name='Custom RPC',
                url=custom_url,
                headers={'Content-Type': 'application/json'},
                rate_limit_delay=0.1
            )

        if provider == 'infura' and project_id:
            return RPCProvider(
                name='Infura',
                url=f'https://mainnet.infura.io/v3/{project_id}',
                headers={'Content-Type': 'application/json'},
                rate_limit_delay=0.1
            )

        if provider == 'alchemy' and api_key:
            return RPCProvider(
                name='Alchemy',
                url=f'https://eth-mainnet.g.alchemy.com/v2/{api_key}',
                headers={'Content-Type': 'application/json'},
                rate_limit_delay=0.1
            )

        if provider in self.PUBLIC_PROVIDERS:
            return self.PUBLIC_PROVIDERS[provider]

        # Default to public provider
        return self.PUBLIC_PROVIDERS['ethereum_public']

    def _make_rpc_call(self, method: str, params: list) -> dict:
        """Make an RPC call and return the result."""

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }

        try:
            response = self.session.post(self.provider_config.url,
                                       json=payload,
                                       timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'error' in data:
                raise Exception(f"RPC error: {data['error']}")

            return data.get('result')

        except Exception as e:
            print(f"RPC call failed: {e}")
            raise

    def get_latest_block_number(self) -> int:
        """Get the latest block number."""
        result = self._make_rpc_call("eth_blockNumber", [])
        return int(result, 16)

    def get_block_by_number(self, block_number: int, include_txs: bool = False) -> dict:
        """Get block data by block number."""
        block_hex = hex(block_number)
        result = self._make_rpc_call("eth_getBlockByNumber", [block_hex, include_txs])
        return result

    def estimate_block_by_timestamp(self, target_timestamp: int) -> int:
        """Estimate block number for a given timestamp using binary search."""

        # Get latest block for upper bound
        latest_block = self.get_latest_block_number()
        latest_block_data = self.get_block_by_number(latest_block)
        latest_timestamp = int(latest_block_data['timestamp'], 16)

        # Estimate average block time (12 seconds for post-merge)
        avg_block_time = 12

        # Initial estimate
        blocks_back = max(0, int((latest_timestamp - target_timestamp) / avg_block_time))
        estimated_block = max(0, latest_block - blocks_back)

        # Binary search to find exact block
        low, high = max(0, estimated_block - 1000), min(latest_block, estimated_block + 1000)

        print(f"Searching for block near timestamp {target_timestamp} between blocks {low}-{high}")

        while low <= high:
            mid = (low + high) // 2

            try:
                block_data = self.get_block_by_number(mid)
                block_timestamp = int(block_data['timestamp'], 16)

                if abs(block_timestamp - target_timestamp) <= avg_block_time:
                    return mid
                elif block_timestamp < target_timestamp:
                    low = mid + 1
                else:
                    high = mid - 1

                # Rate limiting
                time.sleep(self.provider_config.rate_limit_delay)

            except Exception as e:
                print(f"Error fetching block {mid}: {e}")
                break

        # Return closest block if exact match not found
        return (low + high) // 2

    def fetch_basefee_range(self, start_block: int, end_block: int,
                           step: int = 1) -> List[Tuple[datetime, float]]:
        """Fetch basefee data for a range of blocks."""

        basefee_data = []
        total_blocks = end_block - start_block

        print(f"Fetching basefee data from block {start_block} to {end_block} (step={step})")

        for i, block_num in enumerate(range(start_block, end_block, step)):
            try:
                block_data = self.get_block_by_number(block_num)

                if 'baseFeePerGas' in block_data and 'timestamp' in block_data:
                    timestamp = datetime.fromtimestamp(int(block_data['timestamp'], 16))
                    basefee_wei = int(block_data['baseFeePerGas'], 16)
                    basefee_data.append((timestamp, basefee_wei))

                # Progress indicator
                if i % 50 == 0:
                    progress = (i * step) / total_blocks * 100
                    print(f"Progress: {progress:.1f}% ({len(basefee_data)} blocks fetched)")

                # Rate limiting
                time.sleep(self.provider_config.rate_limit_delay)

            except Exception as e:
                print(f"Error fetching block {block_num}: {e}")
                continue

        print(f"Successfully fetched {len(basefee_data)} blocks")
        return sorted(basefee_data)

    def fetch_period_data(self, start_date: str, end_date: str,
                         samples_per_hour: int = 5) -> pd.DataFrame:
        """Fetch basefee data for a date range."""

        # Convert dates to timestamps
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp())

        print(f"Fetching data from {start_date} to {end_date}")
        print(f"Using RPC provider: {self.provider_config.name}")

        # Find block numbers for date range
        try:
            start_block = self.estimate_block_by_timestamp(start_timestamp)
            end_block = self.estimate_block_by_timestamp(end_timestamp)
        except Exception as e:
            print(f"Error finding blocks for date range: {e}")
            raise

        # Calculate step size for desired sampling rate
        total_blocks = end_block - start_block
        blocks_per_hour = 300  # ~5 minutes per block = 12 blocks/hour, but variable
        desired_blocks = int((end_dt - start_dt).total_seconds() / 3600 * samples_per_hour)
        step = max(1, total_blocks // desired_blocks)

        print(f"Sampling every {step} blocks for {samples_per_hour} samples/hour")

        # Fetch the data
        basefee_data = self.fetch_basefee_range(start_block, end_block, step)

        if not basefee_data:
            raise Exception("No basefee data retrieved")

        # Convert to DataFrame
        df = pd.DataFrame(basefee_data, columns=['timestamp', 'basefee_wei'])
        df['basefee_gwei'] = df['basefee_wei'] / 1e9
        df['block_number'] = range(start_block, start_block + len(df) * step, step)

        return df


class ImprovedRealDataIntegrator:
    """Enhanced real data integrator with proper vault initialization."""

    def __init__(self):
        self.cache_dir = "data_cache"
        import os
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_real_basefee_data(self, start_date: str, end_date: str,
                             provider: str = 'ethereum_public',
                             api_key: Optional[str] = None,
                             project_id: Optional[str] = None,
                             use_cache: bool = True) -> pd.DataFrame:
        """Get real basefee data using RPC."""

        cache_file = f"{self.cache_dir}/basefee_{start_date}_{end_date}_{provider}.csv"

        # Try cache first
        if use_cache:
            try:
                df = pd.read_csv(cache_file, parse_dates=['timestamp'])
                print(f"Loaded {len(df)} records from cache")
                return df
            except FileNotFoundError:
                pass

        # Fetch from RPC
        fetcher = RPCBasefeeeFetcher(provider, api_key, project_id)
        df = fetcher.fetch_period_data(start_date, end_date)

        # Cache the results
        df.to_csv(cache_file, index=False)
        print(f"Cached {len(df)} records to {cache_file}")

        return df

    def create_enhanced_l1_model(self, df: pd.DataFrame, name: str = "Real Data"):
        """Create L1 model from real data with better naming."""

        class EnhancedRealDataL1Model:
            def __init__(self, basefee_sequence: np.ndarray, name: str):
                self.basefee_sequence = basefee_sequence
                self.name = name

            def generate_sequence(self, steps: int, initial_basefee: float = None) -> np.ndarray:
                """Return real basefee sequence, repeated/truncated as needed."""
                if steps <= len(self.basefee_sequence):
                    return self.basefee_sequence[:steps]
                else:
                    # Repeat pattern if more steps needed
                    repeats = (steps // len(self.basefee_sequence)) + 1
                    extended = np.tile(self.basefee_sequence, repeats)
                    return extended[:steps]

            def get_name(self) -> str:
                return self.name

        return EnhancedRealDataL1Model(df['basefee_wei'].values, name)


def test_rpc_fetcher():
    """Test the RPC fetcher with a small date range."""

    print("Testing RPC-based basefee fetching...")

    # Test with a small recent range (last week)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')

    try:
        integrator = ImprovedRealDataIntegrator()

        # Try multiple providers in order of preference
        providers = ['ethereum_public', 'cloudflare', 'ankr']

        for provider in providers:
            try:
                print(f"\nTrying {provider}...")
                df = integrator.get_real_basefee_data(
                    start_date, end_date,
                    provider=provider,
                    use_cache=False
                )

                print(f"✓ Successfully fetched {len(df)} records")
                print(f"Basefee range: {df['basefee_gwei'].min():.1f} - {df['basefee_gwei'].max():.1f} gwei")
                print(f"Sample data:")
                print(df[['timestamp', 'basefee_gwei']].head())

                return df

            except Exception as e:
                print(f"✗ {provider} failed: {e}")
                continue

        print("All RPC providers failed")
        return None

    except Exception as e:
        print(f"Test failed: {e}")
        return None


def main():
    """Command line interface for RPC data fetching."""
    import argparse

    parser = argparse.ArgumentParser(description='Fetch basefee data via RPC')
    parser.add_argument('--start-date', help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', help='End date YYYY-MM-DD')
    parser.add_argument('--provider', default='ethereum_public',
                       choices=['ethereum_public', 'cloudflare', 'ankr', 'infura', 'alchemy'],
                       help='RPC provider')
    parser.add_argument('--api-key', help='API key for Alchemy')
    parser.add_argument('--project-id', help='Project ID for Infura')
    parser.add_argument('--test', action='store_true', help='Run connection test')

    args = parser.parse_args()

    if args.test:
        test_rpc_fetcher()
        return

    if not args.start_date or not args.end_date:
        parser.error('--start-date and --end-date are required unless using --test')

    integrator = ImprovedRealDataIntegrator()

    try:
        df = integrator.get_real_basefee_data(
            args.start_date, args.end_date,
            provider=args.provider,
            api_key=args.api_key,
            project_id=args.project_id
        )

        print(f"\n=== BASEFEE DATA SUMMARY ===")
        print(f"Period: {args.start_date} to {args.end_date}")
        print(f"Records: {len(df)}")
        print(f"Basefee range: {df['basefee_gwei'].min():.1f} - {df['basefee_gwei'].max():.1f} gwei")
        print(f"Average: {df['basefee_gwei'].mean():.1f} gwei")
        print(f"Median: {df['basefee_gwei'].median():.1f} gwei")
        print(f"95th percentile: {df['basefee_gwei'].quantile(0.95):.1f} gwei")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()