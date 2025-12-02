"""
Data Loader for Taiko Fee Mechanism Simulation

Loads and processes Ethereum L1 data in the canonical CSV format for
use with the Taiko fee mechanism simulation.

Canonical CSV Format (from CLAUDE.md):
timestamp,basefee_wei,basefee_gwei,block_number
2022-07-01 08:46:46,12999038238,12.999038238,0xe5b8ec

Key Features:
- Validates data format and continuity
- Converts basefee to L1 costs using gas calibration
- Handles missing data and edge cases
- Provides data statistics and quality checks
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import warnings


class DataLoader:
    """
    Data loader for Ethereum L1 basefee data in canonical CSV format.
    """

    def __init__(self, gas_per_batch: float = 6.9e5):
        """
        Initialize data loader.

        Args:
            gas_per_batch: Gas used per batch for L1 cost calculation (default from Taiko data)
        """
        self.gas_per_batch = gas_per_batch

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load canonical CSV file with validation.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with validated and processed data

        Raises:
            ValueError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            # Load with expected columns
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

        # Validate required columns
        required_columns = ['timestamp', 'basefee_wei', 'basefee_gwei', 'block_number']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate data types and ranges
        if not pd.api.types.is_numeric_dtype(df['basefee_wei']):
            raise ValueError("basefee_wei column must be numeric")

        if (df['basefee_wei'] < 0).any():
            raise ValueError("basefee_wei cannot contain negative values")

        # Convert timestamp to datetime
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            raise ValueError(f"Invalid timestamp format: {e}")

        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate L1 costs
        df = self._calculate_l1_costs(df)

        return df

    def _calculate_l1_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate L1 costs from basefee data.

        L1 Cost = BaseFee (wei) * Gas_per_batch / 1e18 (convert to ETH)

        Args:
            df: DataFrame with basefee data

        Returns:
            DataFrame with added l1_cost_eth column
        """
        # Convert wei to ETH and multiply by gas usage
        df['l1_cost_eth'] = df['basefee_wei'] * self.gas_per_batch / 1e18

        return df

    def validate_data_continuity(self, df: pd.DataFrame, max_gap_seconds: int = 60) -> Dict[str, Any]:
        """
        Validate that data represents continuous blocks with no major gaps.

        Args:
            df: DataFrame with timestamp data
            max_gap_seconds: Maximum allowed gap between consecutive timestamps

        Returns:
            Dictionary with validation results and statistics
        """
        if len(df) < 2:
            return {'is_continuous': True, 'gaps': [], 'total_gaps': 0}

        # Calculate time differences
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        gaps = time_diffs[time_diffs > max_gap_seconds]

        validation_result = {
            'is_continuous': len(gaps) == 0,
            'total_gaps': len(gaps),
            'gaps': gaps.tolist() if len(gaps) > 0 else [],
            'max_gap_seconds': gaps.max() if len(gaps) > 0 else 0,
            'avg_interval_seconds': time_diffs[1:].mean(),  # Skip first NaN
            'data_span_hours': (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 3600,
        }

        return validation_result

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary statistics.

        Args:
            df: Processed DataFrame

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'record_count': len(df),
            'time_range': {
                'start': df['timestamp'].iloc[0],
                'end': df['timestamp'].iloc[-1],
                'duration_hours': (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 3600
            },
            'basefee_stats_gwei': {
                'min': df['basefee_gwei'].min(),
                'max': df['basefee_gwei'].max(),
                'mean': df['basefee_gwei'].mean(),
                'median': df['basefee_gwei'].median(),
                'std': df['basefee_gwei'].std(),
                'p95': df['basefee_gwei'].quantile(0.95),
                'p99': df['basefee_gwei'].quantile(0.99),
            },
            'l1_cost_stats_eth': {
                'min': df['l1_cost_eth'].min(),
                'max': df['l1_cost_eth'].max(),
                'mean': df['l1_cost_eth'].mean(),
                'median': df['l1_cost_eth'].median(),
                'std': df['l1_cost_eth'].std(),
            },
        }

        return summary

    def load_multiple_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Load and concatenate multiple CSV files.

        Args:
            file_paths: List of file paths to load

        Returns:
            Combined DataFrame, sorted by timestamp

        Raises:
            ValueError: If files have inconsistent formats or overlapping data
        """
        if not file_paths:
            raise ValueError("No file paths provided")

        dataframes = []
        for file_path in file_paths:
            df = self.load_csv(file_path)
            dataframes.append(df)

        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Sort by timestamp and remove duplicates
        combined_df = combined_df.sort_values('timestamp').drop_duplicates(
            subset=['timestamp'], keep='first'
        ).reset_index(drop=True)

        return combined_df

    def extract_l1_cost_series(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract L1 cost time series for simulation.

        Args:
            df: Processed DataFrame

        Returns:
            NumPy array of L1 costs in ETH
        """
        if 'l1_cost_eth' not in df.columns:
            raise ValueError("DataFrame missing l1_cost_eth column. Run load_csv first.")

        return df['l1_cost_eth'].values

    def simulate_realistic_timing(
        self,
        df: pd.DataFrame,
        l2_block_time_seconds: int = 2,
        l1_batch_interval_seconds: int = 12
    ) -> pd.DataFrame:
        """
        Simulate realistic Taiko timing by resampling data to L1 batch intervals.

        Taiko specific timing:
        - L2 blocks every 2 seconds
        - L1 batches every 12 seconds (6 L2 blocks)

        Args:
            df: Input DataFrame with high-frequency data
            l2_block_time_seconds: L2 block time (default: 2s)
            l1_batch_interval_seconds: L1 batch posting interval (default: 12s)

        Returns:
            DataFrame resampled to L1 batch intervals
        """
        if len(df) == 0:
            raise ValueError("Cannot resample empty DataFrame")

        # Set timestamp as index for resampling
        df_indexed = df.set_index('timestamp')

        # Resample to L1 batch intervals using mean of interval
        resampled = df_indexed.resample(f'{l1_batch_interval_seconds}S').agg({
            'basefee_wei': 'mean',
            'basefee_gwei': 'mean',
            'l1_cost_eth': 'mean',
            'block_number': 'first',  # Keep first block number in interval
        }).dropna()

        # Reset index to get timestamp back as column
        resampled = resampled.reset_index()

        return resampled

    def apply_data_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data quality filters to remove obvious outliers or errors.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame with quality issues removed
        """
        initial_count = len(df)
        filtered_df = df.copy()

        # Remove rows with zero basefee (likely errors)
        filtered_df = filtered_df[filtered_df['basefee_wei'] > 0]

        # Remove extreme outliers (>1000 gwei, likely errors in historical data)
        max_reasonable_gwei = 1000
        filtered_df = filtered_df[filtered_df['basefee_gwei'] <= max_reasonable_gwei]

        # Report filtering results
        removed_count = initial_count - len(filtered_df)
        if removed_count > 0:
            warnings.warn(
                f"Removed {removed_count} rows ({removed_count/initial_count*100:.1f}%) "
                f"due to quality issues"
            )

        return filtered_df

    def __str__(self) -> str:
        """String representation of data loader."""
        return f"DataLoader(gas_per_batch={self.gas_per_batch:.1e})"

    def __repr__(self) -> str:
        """Detailed representation of data loader."""
        return self.__str__()