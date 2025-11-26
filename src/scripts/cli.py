#!/usr/bin/env python3
"""
Command-line interface for Taiko fee analysis tools.

This script provides a unified interface for all data fetching and analysis tasks.
"""

import sys
import os
import argparse
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data import BlockFetcher, ContiguityAnalyzer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def cmd_fetch(args):
    """Handle fetch command."""
    fetcher = BlockFetcher()
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data_cache')
    os.makedirs(data_dir, exist_ok=True)

    def progress_callback(progress: float, completed: int, total: int):
        if not args.quiet:
            print(f"Progress: {progress:.1f}% ({completed}/{total} blocks)")

    if args.hours:
        # Fetch recent blocks by hours
        df = fetcher.fetch_recent_blocks(args.hours, progress_callback=progress_callback)
        if not df.empty:
            output_file = os.path.join(data_dir, f'real_{args.hours}hour_contiguous.csv')
            if fetcher.save_to_csv(df, output_file, f"{args.hours}-hour contiguous data"):
                print(f"✅ Saved {len(df)} blocks to {output_file}")
            else:
                print("❌ Failed to save dataset")
                sys.exit(1)
        else:
            print("❌ Failed to fetch data")
            sys.exit(1)

    elif args.blocks:
        # Fetch specific number of recent blocks
        latest = fetcher.rpc_client.get_latest_block_number()
        if latest is None:
            print("❌ Failed to get latest block number")
            sys.exit(1)

        start_block = latest - args.blocks + 1
        df = fetcher.fetch_contiguous_blocks(start_block, args.blocks, progress_callback=progress_callback)

        if not df.empty:
            output_file = os.path.join(data_dir, f'real_{args.blocks}blocks_contiguous.csv')
            if fetcher.save_to_csv(df, output_file, f"{args.blocks} contiguous blocks"):
                print(f"✅ Saved {len(df)} blocks to {output_file}")
            else:
                print("❌ Failed to save dataset")
                sys.exit(1)
        else:
            print("❌ Failed to fetch data")
            sys.exit(1)

    elif args.range:
        # Fetch specific block range
        start, end = map(int, args.range.split('-'))
        count = end - start + 1
        df = fetcher.fetch_contiguous_blocks(start, count, progress_callback=progress_callback)

        if not df.empty:
            output_file = os.path.join(data_dir, f'real_blocks_{start}_{end}.csv')
            if fetcher.save_to_csv(df, output_file, f"Blocks {start} to {end}"):
                print(f"✅ Saved {len(df)} blocks to {output_file}")
            else:
                print("❌ Failed to save dataset")
                sys.exit(1)
        else:
            print("❌ Failed to fetch data")
            sys.exit(1)


def cmd_analyze(args):
    """Handle analyze command."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data_cache')

    if args.file:
        # Analyze specific file
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            sys.exit(1)

        filename = os.path.basename(args.file)
        analysis = ContiguityAnalyzer.analyze_dataset(args.file, filename)

        print(f"\n=== {analysis.dataset_name} ===")
        print(f"Status: {'✅ Contiguous' if analysis.is_contiguous else '❌ Has gaps'}")
        print(f"Blocks: {analysis.total_blocks}/{analysis.expected_blocks}")

        if analysis.missing_blocks > 0:
            print(f"Missing blocks: {analysis.missing_blocks}")

        if analysis.block_range:
            print(f"Block range: {analysis.block_range[0]} to {analysis.block_range[1]}")

        if analysis.gaps and not args.quiet:
            print(f"Gaps: {len(analysis.gaps)}")
            for i, (start, end) in enumerate(analysis.gaps[:5]):
                gap_size = end - start + 1
                print(f"  Gap {i+1}: blocks {start} to {end} ({gap_size} blocks)")

    else:
        # Analyze all files in data directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

        if not csv_files:
            print(f"❌ No CSV files found in {data_dir}")
            sys.exit(1)

        datasets = [(os.path.join(data_dir, f), f) for f in csv_files]
        analyses = ContiguityAnalyzer.analyze_multiple_datasets(datasets)

        if not args.quiet:
            ContiguityAnalyzer.print_summary(analyses)
        else:
            for analysis in analyses:
                status = "✅" if analysis.is_contiguous else "❌"
                print(f"{status} {analysis.dataset_name}: {analysis.total_blocks}/{analysis.expected_blocks}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Taiko fee analysis data tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s fetch --hours 1           # Fetch 1 hour of recent blocks
  %(prog)s fetch --hours 3           # Fetch 3 hours of recent blocks
  %(prog)s fetch --blocks 300        # Fetch latest 300 blocks
  %(prog)s fetch --range 18000000-18000100  # Fetch specific block range
  %(prog)s analyze                   # Analyze all datasets
  %(prog)s analyze --file data.csv   # Analyze specific file
        """
    )

    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet output')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch blockchain data')
    fetch_group = fetch_parser.add_mutually_exclusive_group(required=True)
    fetch_group.add_argument('--hours', type=int, help='Fetch N hours of recent blocks')
    fetch_group.add_argument('--blocks', type=int, help='Fetch N recent blocks')
    fetch_group.add_argument('--range', type=str, help='Fetch block range (start-end)')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze data contiguity')
    analyze_parser.add_argument('--file', type=str, help='Analyze specific CSV file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)

    try:
        if args.command == 'fetch':
            cmd_fetch(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()