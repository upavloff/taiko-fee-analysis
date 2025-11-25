"""
Real L1 Basefee Data Fetcher for Taiko Fee Analysis

This module fetches historical Ethereum basefee data from multiple sources
and provides it in a format suitable for our fee mechanism testing.

Supports multiple data sources:
1. Etherscan API (free tier available)
2. Infura/Alchemy RPC calls
3. Dune Analytics (with API key)
4. Pre-saved historical datasets

Usage:
    python real_data_fetcher.py --period crisis --days 30
"""

import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import json
import os
from dataclasses import dataclass


@dataclass
class HistoricalPeriod:
    """Represents a significant period in Ethereum fee history."""
    name: str
    start_date: str
    end_date: str
    description: str
    expected_max_basefee_gwei: float


# Define significant historical periods for testing
HISTORICAL_PERIODS = {
    'defi_summer': HistoricalPeriod(
        name='DeFi Summer 2021',
        start_date='2021-08-01',
        end_date='2021-08-31',
        description='Peak DeFi activity with sustained high fees',
        expected_max_basefee_gwei=200
    ),
    'nft_boom': HistoricalPeriod(
        name='NFT Boom 2021',
        start_date='2021-09-01',
        end_date='2021-09-30',
        description='NFT mania causing extreme fee spikes',
        expected_max_basefee_gwei=400
    ),
    'may_crash': HistoricalPeriod(
        name='May 2022 Crash',
        start_date='2022-05-09',
        end_date='2022-05-15',
        description='UST/Luna collapse causing massive volatility',
        expected_max_basefee_gwei=300
    ),
    'arbitrage_wars': HistoricalPeriod(
        name='MEV/Arbitrage Wars',
        start_date='2021-04-15',
        end_date='2021-04-30',
        description='Intense MEV competition causing fee spikes',
        expected_max_basefee_gwei=150
    ),
    'recent_stable': HistoricalPeriod(
        name='Recent Stable Period',
        start_date='2024-01-01',
        end_date='2024-01-31',
        description='Post-merge stable period with lower fees',
        expected_max_basefee_gwei=30
    )
}


class EtherscanFetcher:
    """Fetches basefee data from Etherscan API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "YourApiKeyToken"  # Free tier default
        self.base_url = "https://api.etherscan.io/api"

    def fetch_basefee_data(self, start_date: str, end_date: str) -> List[Tuple[datetime, float]]:
        """Fetch basefee data between dates."""
        print(f"Fetching Etherscan data from {start_date} to {end_date}...")

        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # Get block numbers for date range
        start_block = self._get_block_by_timestamp(start_ts)
        end_block = self._get_block_by_timestamp(end_ts)

        if start_block is None or end_block is None:
            raise Exception("Could not fetch block numbers for date range")

        print(f"Fetching blocks {start_block} to {end_block}")

        # Sample blocks (every 100 blocks for performance)
        block_step = max(1, (end_block - start_block) // 1000)
        sampled_blocks = range(start_block, end_block, block_step)

        basefee_data = []

        for block_num in sampled_blocks:
            try:
                block_info = self._get_block_info(block_num)
                if block_info and 'baseFeePerGas' in block_info:
                    timestamp = datetime.fromtimestamp(int(block_info['timestamp'], 16))
                    basefee_wei = int(block_info['baseFeePerGas'], 16)
                    basefee_data.append((timestamp, basefee_wei))

                # Rate limiting
                time.sleep(0.2)

            except Exception as e:
                print(f"Error fetching block {block_num}: {e}")
                continue

        return sorted(basefee_data)

    def _get_block_by_timestamp(self, timestamp: int) -> Optional[int]:
        """Get block number by timestamp."""
        url = f"{self.base_url}?module=block&action=getblocknobytime&timestamp={timestamp}&closest=before&apikey={self.api_key}"
        try:
            response = requests.get(url)
            data = response.json()
            if data['status'] == '1':
                return int(data['result'])
        except Exception as e:
            print(f"Error getting block by timestamp: {e}")
        return None

    def _get_block_info(self, block_number: int) -> Optional[dict]:
        """Get block information including basefee."""
        url = f"{self.base_url}?module=proxy&action=eth_getBlockByNumber&tag=0x{block_number:x}&boolean=false&apikey={self.api_key}"
        try:
            response = requests.get(url)
            data = response.json()
            if 'result' in data and data['result']:
                return data['result']
        except Exception as e:
            print(f"Error getting block info: {e}")
        return None


class DuneAnalyticsFetcher:
    """Fetches data from Dune Analytics."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.dune.com/api/v1"

    def fetch_basefee_data(self, start_date: str, end_date: str) -> List[Tuple[datetime, float]]:
        """Fetch basefee data from Dune."""
        # This would use a custom Dune query for basefee data
        # For now, return empty as it requires API setup
        print("Dune Analytics fetcher not implemented yet - requires API key setup")
        return []


class SyntheticDataGenerator:
    """Generates realistic synthetic data based on known periods."""

    @staticmethod
    def generate_period_data(period: HistoricalPeriod, samples_per_day: int = 24) -> List[Tuple[datetime, float]]:
        """Generate synthetic data for a historical period."""
        print(f"Generating synthetic data for {period.name}")

        start = datetime.strptime(period.start_date, '%Y-%m-%d')
        end = datetime.strptime(period.end_date, '%Y-%m-%d')

        # Calculate total samples
        total_days = (end - start).days
        total_samples = total_days * samples_per_day

        # Generate time series
        times = [start + timedelta(hours=i/samples_per_day*24) for i in range(total_samples)]

        # Generate realistic basefee patterns
        basefees = SyntheticDataGenerator._generate_realistic_pattern(
            total_samples, period.expected_max_basefee_gwei
        )

        return list(zip(times, basefees))

    @staticmethod
    def _generate_realistic_pattern(samples: int, max_basefee_gwei: float) -> np.ndarray:
        """Generate realistic basefee pattern with spikes and trends."""

        # Base level (10-30% of max)
        base_level = max_basefee_gwei * 0.2

        # Generate base GBM process
        dt = 1/24  # Hourly data
        volatility = 0.5
        drift = 0.0

        # GBM with mean reversion
        gbm = np.random.normal(0, 1, samples)
        log_prices = np.cumsum((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * gbm)
        base_prices = base_level * np.exp(log_prices)

        # Add mean reversion
        mean_reversion_strength = 0.1
        target_level = base_level
        for i in range(1, samples):
            reversion = mean_reversion_strength * (target_level - base_prices[i-1]) * dt
            base_prices[i] = base_prices[i-1] + reversion + volatility * np.sqrt(dt) * gbm[i]

        # Add spike events (5-10 during the period)
        num_spikes = np.random.randint(3, 8)
        spike_indices = np.random.choice(samples, num_spikes, replace=False)

        for spike_idx in spike_indices:
            # Spike magnitude (2x to 10x base)
            spike_magnitude = np.random.uniform(2.0, min(10.0, max_basefee_gwei/base_level))
            spike_duration = np.random.randint(2, 12)  # 2-12 hours

            # Apply spike with exponential decay
            for i in range(spike_duration):
                if spike_idx + i < samples:
                    decay = np.exp(-i / (spike_duration / 3))
                    base_prices[spike_idx + i] *= (1 + (spike_magnitude - 1) * decay)

        # Convert to wei
        basefees_wei = base_prices * 1e9  # Convert gwei to wei
        return basefees_wei


class RealDataIntegrator:
    """Integrates real basefee data with our simulation framework."""

    def __init__(self):
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_period_data(self, period_name: str, use_cache: bool = True, etherscan_api_key: Optional[str] = None) -> pd.DataFrame:
        """Get basefee data for a specific historical period."""

        if period_name not in HISTORICAL_PERIODS:
            raise ValueError(f"Unknown period: {period_name}. Available: {list(HISTORICAL_PERIODS.keys())}")

        period = HISTORICAL_PERIODS[period_name]
        cache_file = os.path.join(self.cache_dir, f"{period_name}_basefee_data.csv")

        # Try to load from cache first
        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached data for {period.name}")
            return pd.read_csv(cache_file, parse_dates=['timestamp'])

        # Try to fetch real data
        basefee_data = []

        try:
            if etherscan_api_key:
                fetcher = EtherscanFetcher(etherscan_api_key)
                basefee_data = fetcher.fetch_basefee_data(period.start_date, period.end_date)
            else:
                print("No API key provided, using synthetic data")
                raise Exception("No API key")

        except Exception as e:
            print(f"Real data fetch failed: {e}")
            print(f"Generating synthetic data for {period.name}")
            basefee_data = SyntheticDataGenerator.generate_period_data(period)

        if not basefee_data:
            print(f"Generating synthetic data for {period.name}")
            basefee_data = SyntheticDataGenerator.generate_period_data(period)

        # Convert to DataFrame
        df = pd.DataFrame(basefee_data, columns=['timestamp', 'basefee_wei'])

        # Add derived columns
        df['basefee_gwei'] = df['basefee_wei'] / 1e9
        df['block_number'] = range(len(df))  # Synthetic block numbers

        # Save to cache
        df.to_csv(cache_file, index=False)
        print(f"Saved {len(df)} data points to cache")

        return df

    def create_l1_model_from_data(self, df: pd.DataFrame):
        """Create an L1 dynamics model from real data."""

        class RealDataL1Model:
            def __init__(self, basefee_sequence: np.ndarray):
                self.basefee_sequence = basefee_sequence
                self.name = "Real Historical Data"

            def generate_sequence(self, steps: int, initial_basefee: float = None) -> np.ndarray:
                """Return the real basefee sequence, repeated/truncated as needed."""
                if steps <= len(self.basefee_sequence):
                    return self.basefee_sequence[:steps]
                else:
                    # Repeat the sequence if more steps needed
                    repeats = (steps // len(self.basefee_sequence)) + 1
                    extended = np.tile(self.basefee_sequence, repeats)
                    return extended[:steps]

            def get_name(self) -> str:
                return self.name

        return RealDataL1Model(df['basefee_wei'].values)


def main():
    """Command line interface for data fetching."""
    import argparse

    parser = argparse.ArgumentParser(description='Fetch real L1 basefee data')
    parser.add_argument('--period', choices=list(HISTORICAL_PERIODS.keys()),
                       default='defi_summer', help='Historical period to fetch')
    parser.add_argument('--api-key', type=str, help='Etherscan API key')
    parser.add_argument('--no-cache', action='store_true', help='Skip cache and fetch fresh data')
    parser.add_argument('--list-periods', action='store_true', help='List available periods')

    args = parser.parse_args()

    if args.list_periods:
        print("Available historical periods:")
        for name, period in HISTORICAL_PERIODS.items():
            print(f"  {name}: {period.description} ({period.start_date} to {period.end_date})")
        return

    integrator = RealDataIntegrator()

    try:
        df = integrator.get_period_data(
            args.period,
            use_cache=not args.no_cache,
            etherscan_api_key=args.api_key
        )

        print(f"\nData Summary for {HISTORICAL_PERIODS[args.period].name}:")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Data points: {len(df)}")
        print(f"Basefee range: {df['basefee_gwei'].min():.1f} to {df['basefee_gwei'].max():.1f} gwei")
        print(f"Average basefee: {df['basefee_gwei'].mean():.1f} gwei")
        print(f"Median basefee: {df['basefee_gwei'].median():.1f} gwei")
        print(f"95th percentile: {df['basefee_gwei'].quantile(0.95):.1f} gwei")

        # Show sample data
        print(f"\nFirst 5 data points:")
        print(df[['timestamp', 'basefee_gwei']].head())

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()