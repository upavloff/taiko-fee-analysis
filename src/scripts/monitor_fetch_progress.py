#!/usr/bin/env python3
"""
Monitor the progress of Ethereum block data fetching.

This script monitors the fetch progress by checking the output file and checkpoint.
"""

import json
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def monitor_progress():
    """Monitor the fetching progress."""
    output_file = Path("/Users/ulyssepavloff/Desktop/Nethermind/taiko-fee-analysis/data/data_cache/real_july_2022_spike_data.csv")
    checkpoint_file = output_file.with_suffix('.checkpoint.json')
    log_file = output_file.with_suffix('.log')

    print(f"📊 Monitoring Ethereum Block Fetch Progress")
    print(f"=" * 60)

    # Check if files exist
    if not output_file.exists():
        print("❌ Output file doesn't exist yet. Fetching hasn't started.")
        return

    # Count rows in CSV (subtract 1 for header)
    try:
        df = pd.read_csv(output_file)
        current_blocks = len(df)
        total_blocks = 9901

        if current_blocks > 0:
            latest_block_hex = df.iloc[-1]['block_number']
            latest_block_num = int(latest_block_hex, 16)
            start_block = 15055000
            progress_pct = (latest_block_num - start_block + 1) / total_blocks * 100

            print(f"✅ Blocks fetched: {current_blocks:,} / {total_blocks:,}")
            print(f"📍 Latest block: {latest_block_num:,} ({latest_block_hex})")
            print(f"📈 Progress: {progress_pct:.2f}%")
            print(f"📅 Latest timestamp: {df.iloc[-1]['timestamp']}")

            # Estimate completion
            if current_blocks > 100:  # Need enough data for estimate
                remaining_blocks = total_blocks - current_blocks
                # Assume average rate of 1.4 blocks/sec (from progress report)
                remaining_seconds = remaining_blocks / 1.4
                remaining_hours = remaining_seconds / 3600
                print(f"⏰ Estimated time remaining: {remaining_hours:.1f} hours")
        else:
            print("📁 CSV file exists but is empty")

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

    # Check checkpoint file
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"\n📋 Checkpoint Info:")
            print(f"   Last completed block: {checkpoint.get('last_completed_block', 'N/A'):,}")
            print(f"   Checkpoint time: {checkpoint.get('timestamp', 'N/A')}")
        except Exception as e:
            print(f"❌ Error reading checkpoint: {e}")

    # Check recent log entries
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Get last 5 lines
            recent_logs = lines[-5:] if len(lines) >= 5 else lines
            print(f"\n📝 Recent Log Entries:")
            for line in recent_logs:
                if line.strip():
                    print(f"   {line.strip()}")

        except Exception as e:
            print(f"❌ Error reading log: {e}")

    print(f"\n" + "=" * 60)
    print(f"💡 To resume if interrupted: python src/scripts/fetch_ethereum_blocks.py --resume")
    print(f"📊 Run this monitor again: python src/scripts/monitor_fetch_progress.py")


if __name__ == "__main__":
    monitor_progress()