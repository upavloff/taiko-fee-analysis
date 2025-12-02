#!/usr/bin/env python3
"""
Script to analyze data contiguity in blockchain datasets.

This script analyzes CSV files containing blockchain data to check
if they have block-by-block contiguous data with no gaps.
"""

import sys
import os
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.validators import ContiguityAnalyzer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def main():
    """Main function to analyze data contiguity."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=== Blockchain Data Contiguity Analysis ===")

    # Define datasets to analyze
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data_cache')

    datasets = [
        (os.path.join(data_dir, 'may_crash_basefee_data.csv'), 'May Crash Data'),
        (os.path.join(data_dir, 'recent_low_fees_3hours.csv'), 'Recent Low Fees (3 hours)'),
        (os.path.join(data_dir, 'real_july_2022_spike_data.csv'), 'July 2022 Spike Data'),
        (os.path.join(data_dir, 'real_1hour_contiguous.csv'), 'Real 1-Hour Contiguous'),
        (os.path.join(data_dir, 'real_3hour_contiguous.csv'), 'Real 3-Hour Contiguous'),
    ]

    # Filter to only existing files
    existing_datasets = []
    for file_path, name in datasets:
        if os.path.exists(file_path):
            existing_datasets.append((file_path, name))
        else:
            logger.warning(f"File not found: {file_path}")

    if not existing_datasets:
        logger.error("No dataset files found for analysis")
        sys.exit(1)

    # Analyze all datasets
    try:
        analyses = ContiguityAnalyzer.analyze_multiple_datasets(existing_datasets)

        # Print detailed results
        for analysis in analyses:
            logger.info(f"\n=== {analysis.dataset_name} ===")
            summary = analysis.get_summary()

            logger.info(f"Status: {'✅ Contiguous' if analysis.is_contiguous else '❌ Has gaps'}")
            logger.info(f"Blocks: {analysis.total_blocks}/{analysis.expected_blocks}")

            if analysis.missing_blocks > 0:
                logger.info(f"Missing blocks: {analysis.missing_blocks}")

            if analysis.block_range:
                logger.info(f"Block range: {analysis.block_range[0]} to {analysis.block_range[1]}")

            if analysis.time_range:
                duration = analysis.time_range[1] - analysis.time_range[0]
                logger.info(f"Time range: {analysis.time_range[0]} to {analysis.time_range[1]}")
                logger.info(f"Duration: {duration}")

            if analysis.gaps:
                logger.info(f"Gaps found: {len(analysis.gaps)}")
                if len(analysis.gaps) <= 5:
                    for i, (start, end) in enumerate(analysis.gaps):
                        gap_size = end - start + 1
                        logger.info(f"  Gap {i+1}: blocks {start} to {end} ({gap_size} blocks)")
                else:
                    logger.info("  (Showing first 3 gaps)")
                    for i, (start, end) in enumerate(analysis.gaps[:3]):
                        gap_size = end - start + 1
                        logger.info(f"  Gap {i+1}: blocks {start} to {end} ({gap_size} blocks)")

        # Print summary
        ContiguityAnalyzer.print_summary(analyses)

        # Return appropriate exit code
        all_contiguous = all(analysis.is_contiguous for analysis in analyses)
        if all_contiguous:
            logger.info("🎉 All datasets are contiguous!")
            sys.exit(0)
        else:
            logger.warning("⚠️  Some datasets have gaps")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()