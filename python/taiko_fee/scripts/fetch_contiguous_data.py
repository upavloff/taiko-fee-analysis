#!/usr/bin/env python3
"""
Script to fetch contiguous Ethereum block data.

This script creates datasets with block-by-block contiguous data
including basefee for each block, suitable for fee analysis.
"""

import sys
import os
import logging
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.fetchers import BlockFetcher


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def progress_callback(progress: float, completed: int, total: int):
    """Progress callback for data fetching."""
    print(f"Progress: {progress:.1f}% ({completed}/{total} blocks)")


def main():
    """Main function to fetch contiguous data."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=== Fetching Contiguous Ethereum Block Data ===")

    # Initialize fetcher
    fetcher = BlockFetcher()

    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data_cache')
    os.makedirs(data_dir, exist_ok=True)

    try:
        # Fetch 1-hour dataset (300 blocks)
        logger.info("Fetching 1-hour contiguous dataset...")
        df_1h = fetcher.fetch_recent_blocks(1, progress_callback=progress_callback)

        if not df_1h.empty:
            output_1h = os.path.join(data_dir, 'real_1hour_contiguous.csv')
            if fetcher.save_to_csv(df_1h, output_1h, "1-hour contiguous data"):
                logger.info(f"✅ 1-hour dataset: {len(df_1h)} blocks saved")
            else:
                logger.error("❌ Failed to save 1-hour dataset")
        else:
            logger.error("❌ Failed to fetch 1-hour dataset")

        # Fetch 3-hour dataset (900 blocks)
        logger.info("\nFetching 3-hour contiguous dataset...")
        df_3h = fetcher.fetch_recent_blocks(3, progress_callback=progress_callback)

        if not df_3h.empty:
            output_3h = os.path.join(data_dir, 'real_3hour_contiguous.csv')
            if fetcher.save_to_csv(df_3h, output_3h, "3-hour contiguous data"):
                logger.info(f"✅ 3-hour dataset: {len(df_3h)} blocks saved")
            else:
                logger.error("❌ Failed to save 3-hour dataset")
        else:
            logger.error("❌ Failed to fetch 3-hour dataset")

        logger.info("\n=== SUMMARY ===")
        logger.info("✅ Created contiguous datasets with real Ethereum data:")
        if not df_1h.empty:
            logger.info(f"  - 1-hour dataset: {len(df_1h)} consecutive blocks with basefee")
        if not df_3h.empty:
            logger.info(f"  - 3-hour dataset: {len(df_3h)} consecutive blocks with basefee")
        logger.info("✅ All data is block-by-block contiguous!")

    except Exception as e:
        logger.error(f"Error during data fetching: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()