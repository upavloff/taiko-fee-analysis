"""
Advanced data contiguity analysis for blockchain datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ContiguityAnalysis:
    """Container for contiguity analysis results."""

    def __init__(
        self,
        dataset_name: str,
        is_contiguous: bool,
        total_blocks: int,
        expected_blocks: int,
        missing_blocks: int,
        gaps: List[Tuple[int, int]],
        time_range: Optional[Tuple[datetime, datetime]] = None,
        block_range: Optional[Tuple[int, int]] = None
    ):
        self.dataset_name = dataset_name
        self.is_contiguous = is_contiguous
        self.total_blocks = total_blocks
        self.expected_blocks = expected_blocks
        self.missing_blocks = missing_blocks
        self.gaps = gaps
        self.time_range = time_range
        self.block_range = block_range

    def __str__(self) -> str:
        status = "✅ Contiguous" if self.is_contiguous else "❌ Has gaps"
        return f"{self.dataset_name}: {status} ({self.total_blocks}/{self.expected_blocks} blocks)"

    def get_summary(self) -> Dict:
        """Get a summary dictionary of the analysis."""
        return {
            'dataset_name': self.dataset_name,
            'is_contiguous': self.is_contiguous,
            'total_blocks': self.total_blocks,
            'expected_blocks': self.expected_blocks,
            'missing_blocks': self.missing_blocks,
            'num_gaps': len(self.gaps),
            'block_range': self.block_range,
            'time_range': self.time_range
        }


class ContiguityAnalyzer:
    """
    Advanced analyzer for checking block-by-block data contiguity.

    Provides comprehensive analysis of blockchain datasets including:
    - Block contiguity validation
    - Gap detection and analysis
    - Data completeness metrics
    - Timestamp consistency checks
    """

    @staticmethod
    def analyze_dataset(
        file_path: str,
        dataset_name: str
    ) -> ContiguityAnalysis:
        """
        Perform comprehensive contiguity analysis on a dataset.

        Args:
            file_path: Path to the CSV file
            dataset_name: Human-readable name for the dataset

        Returns:
            ContiguityAnalysis object with detailed results
        """
        logger.info(f"Analyzing dataset: {dataset_name}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return ContiguityAnalysis(
                dataset_name, False, 0, 0, 0, [], None, None
            )

        return ContiguityAnalyzer._analyze_dataframe(df, dataset_name)

    @staticmethod
    def analyze_dataframe(
        df: pd.DataFrame,
        dataset_name: str
    ) -> ContiguityAnalysis:
        """
        Analyze a DataFrame for contiguity.

        Args:
            df: DataFrame with block data
            dataset_name: Human-readable name for the dataset

        Returns:
            ContiguityAnalysis object with detailed results
        """
        return ContiguityAnalyzer._analyze_dataframe(df, dataset_name)

    @staticmethod
    def _analyze_dataframe(
        df: pd.DataFrame,
        dataset_name: str
    ) -> ContiguityAnalysis:
        """Internal method to analyze DataFrame."""

        if df.empty:
            logger.warning(f"Dataset {dataset_name} is empty")
            return ContiguityAnalysis(
                dataset_name, False, 0, 0, 0, [], None, None
            )

        total_blocks = len(df)
        logger.info(f"Dataset has {total_blocks} rows")

        # Analyze timestamps
        time_range = None
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                time_range = (df['timestamp'].iloc[0], df['timestamp'].iloc[-1])
                duration = time_range[1] - time_range[0]
                logger.info(f"Time range: {time_range[0]} to {time_range[1]} (Duration: {duration})")
            except Exception as e:
                logger.warning(f"Error parsing timestamps: {e}")

        # Analyze block numbers
        if 'block_number' not in df.columns:
            logger.warning("No block_number column found")
            return ContiguityAnalysis(
                dataset_name, False, total_blocks, total_blocks, 0, [], time_range, None
            )

        # Handle hex block numbers
        try:
            if df['block_number'].dtype == 'object':
                # Check if any values are hex strings
                sample = df['block_number'].iloc[0]
                if isinstance(sample, str) and sample.startswith('0x'):
                    df['block_number'] = df['block_number'].apply(
                        lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else int(x)
                    )
                else:
                    df['block_number'] = pd.to_numeric(df['block_number'])
            else:
                df['block_number'] = pd.to_numeric(df['block_number'])
        except Exception as e:
            logger.error(f"Error converting block numbers: {e}")
            return ContiguityAnalysis(
                dataset_name, False, total_blocks, total_blocks, 0, [], time_range, None
            )

        # Sort by block number for analysis
        df_sorted = df.sort_values('block_number').copy()

        block_min = int(df_sorted['block_number'].min())
        block_max = int(df_sorted['block_number'].max())
        block_range = (block_min, block_max)

        logger.info(f"Block range: {block_min} to {block_max}")

        # Check contiguity
        expected_blocks = block_max - block_min + 1
        missing_blocks = expected_blocks - total_blocks

        logger.info(f"Expected blocks: {expected_blocks}")
        logger.info(f"Actual blocks: {total_blocks}")
        logger.info(f"Missing blocks: {missing_blocks}")

        # Find gaps
        gaps = ContiguityAnalyzer._find_gaps(df_sorted)

        is_contiguous = missing_blocks == 0 and len(gaps) == 0

        # Validate basefee data
        ContiguityAnalyzer._validate_basefee_data(df)

        logger.info(f"Analysis complete: {'✅ Contiguous' if is_contiguous else '❌ Has gaps'}")

        return ContiguityAnalysis(
            dataset_name,
            is_contiguous,
            total_blocks,
            expected_blocks,
            missing_blocks,
            gaps,
            time_range,
            block_range
        )

    @staticmethod
    def _find_gaps(df_sorted: pd.DataFrame) -> List[Tuple[int, int]]:
        """Find gaps in block sequence."""
        gaps = []
        block_numbers = df_sorted['block_number'].values

        for i in range(len(block_numbers) - 1):
            current_block = block_numbers[i]
            next_block = block_numbers[i + 1]

            if next_block - current_block > 1:
                gap_start = current_block + 1
                gap_end = next_block - 1
                gaps.append((gap_start, gap_end))

        if gaps:
            logger.info(f"Found {len(gaps)} gaps:")
            for i, (start, end) in enumerate(gaps[:10]):  # Show first 10 gaps
                gap_size = end - start + 1
                logger.info(f"  Gap {i+1}: blocks {start} to {end} ({gap_size} missing blocks)")

        return gaps

    @staticmethod
    def _validate_basefee_data(df: pd.DataFrame) -> None:
        """Validate basefee data completeness."""
        basefee_cols = [col for col in df.columns if 'basefee' in col.lower()]
        logger.info(f"Base fee columns: {basefee_cols}")

        for col in basefee_cols:
            non_null = df[col].notna().sum()
            total = len(df)
            logger.info(f"  {col}: {non_null}/{total} non-null values ({non_null/total:.1%})")

    @staticmethod
    def analyze_multiple_datasets(
        datasets: List[Tuple[str, str]]
    ) -> List[ContiguityAnalysis]:
        """
        Analyze multiple datasets and return results.

        Args:
            datasets: List of (file_path, dataset_name) tuples

        Returns:
            List of ContiguityAnalysis objects
        """
        results = []

        for file_path, name in datasets:
            try:
                result = ContiguityAnalyzer.analyze_dataset(file_path, name)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {name}: {e}")
                results.append(ContiguityAnalysis(name, False, 0, 0, 0, [], None, None))

        return results

    @staticmethod
    def print_summary(analyses: List[ContiguityAnalysis]) -> None:
        """Print a summary of multiple analyses."""
        logger.info("\n=== CONTIGUITY ANALYSIS SUMMARY ===")

        for analysis in analyses:
            logger.info(str(analysis))

            if analysis.gaps and len(analysis.gaps) <= 5:
                logger.info(f"  Gaps: {analysis.gaps}")
            elif len(analysis.gaps) > 5:
                logger.info(f"  {len(analysis.gaps)} gaps detected (showing first 3)")
                for i, (start, end) in enumerate(analysis.gaps[:3]):
                    gap_size = end - start + 1
                    logger.info(f"    Gap {i+1}: blocks {start}-{end} ({gap_size} blocks)")

        logger.info("=" * 50)