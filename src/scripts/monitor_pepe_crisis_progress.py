#!/usr/bin/env python3
"""
Monitor the progress of May 2023 PEPE memecoin crisis data fetching.

This script monitors the fetch progress by checking the output file and checkpoint,
with specific focus on tracking the extreme basefee spikes during the PEPE crisis.
"""

import json
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def monitor_pepe_progress():
    """Monitor the PEPE crisis data fetching progress."""
    output_file = Path("/Users/ulyssepavloff/Desktop/Nethermind/taiko-fee-analysis/data/data_cache/may_2023_pepe_crisis_data.csv")
    checkpoint_file = output_file.with_suffix('.checkpoint.json')
    log_file = output_file.with_suffix('.log')

    print(f"🔥 Monitoring May 2023 PEPE Memecoin Crisis Data Fetch")
    print(f"📅 Target Period: May 3-7, 2023 (Blocks 17,180,000 - 17,220,000)")
    print(f"🎯 Expected Peak: ~184 gwei on May 5, 2023")
    print(f"=" * 80)

    # Check if files exist
    if not output_file.exists():
        print("❌ Output file doesn't exist yet. PEPE crisis fetching hasn't started.")
        return

    # Count rows in CSV (subtract 1 for header)
    try:
        df = pd.read_csv(output_file)
        current_blocks = len(df)
        total_blocks = 40001  # 17,220,000 - 17,180,000 + 1

        if current_blocks > 0:
            latest_block_hex = df.iloc[-1]['block_number']
            latest_block_num = int(latest_block_hex, 16)
            start_block = 17180000
            progress_pct = (latest_block_num - start_block + 1) / total_blocks * 100

            # Calculate basefee statistics
            df['basefee_gwei'] = df['basefee_gwei'].astype(float)
            max_basefee = df['basefee_gwei'].max()
            max_basefee_idx = df['basefee_gwei'].idxmax()
            max_basefee_block = df.iloc[max_basefee_idx]['block_number']
            high_fee_blocks = len(df[df['basefee_gwei'] > 100])
            extreme_fee_blocks = len(df[df['basefee_gwei'] > 150])
            avg_basefee = df['basefee_gwei'].mean()

            print(f"✅ Blocks fetched: {current_blocks:,} / {total_blocks:,}")
            print(f"📍 Latest block: {latest_block_num:,} ({latest_block_hex})")
            print(f"📈 Progress: {progress_pct:.2f}%")
            print(f"📅 Latest timestamp: {df.iloc[-1]['timestamp']}")
            print()
            print(f"🔥 PEPE Crisis Fee Statistics:")
            print(f"   📊 Maximum basefee: {max_basefee:.2f} gwei (block {max_basefee_block})")
            print(f"   📊 Average basefee: {avg_basefee:.2f} gwei")
            print(f"   🔥 High fee blocks (>100 gwei): {high_fee_blocks:,}")
            print(f"   💥 Extreme fee blocks (>150 gwei): {extreme_fee_blocks:,}")

            # Estimate completion
            if current_blocks > 100:  # Need enough data for estimate
                remaining_blocks = total_blocks - current_blocks
                # Estimate rate based on checkpoint if available
                estimated_rate = 1.0  # Conservative estimate: 1 block/sec
                if checkpoint_file.exists():
                    try:
                        with open(checkpoint_file, 'r') as f:
                            checkpoint = json.load(f)
                        # Could calculate actual rate from checkpoint timestamp
                        estimated_rate = 1.2  # Slightly optimistic with checkpoint data
                    except:
                        pass

                remaining_seconds = remaining_blocks / estimated_rate
                remaining_hours = remaining_seconds / 3600
                print(f"   ⏰ Estimated time remaining: {remaining_hours:.1f} hours")

            # Check if we've hit the expected peak
            if max_basefee > 180:
                print(f"   🎯 PEAK DETECTED! We've captured the expected ~184 gwei spike!")
            elif max_basefee > 150:
                print(f"   🔥 High spike detected! Getting close to the expected peak.")
            elif max_basefee > 100:
                print(f"   📈 Significant fees detected! PEPE crisis patterns emerging.")
        else:
            print("📁 CSV file exists but is empty")

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

    # Check checkpoint file with PEPE-specific stats
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"\n📋 Checkpoint Info:")
            print(f"   Last completed block: {checkpoint.get('last_completed_block', 'N/A'):,}")
            print(f"   Checkpoint time: {checkpoint.get('timestamp', 'N/A')}")
            if 'max_basefee_gwei' in checkpoint:
                print(f"   Max basefee in checkpoint: {checkpoint.get('max_basefee_gwei', 0):.2f} gwei")
                print(f"   High fee blocks in checkpoint: {checkpoint.get('total_high_fee_blocks', 0):,}")
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

    print(f"\n" + "=" * 80)
    print(f"💡 To resume if interrupted: python src/scripts/fetch_may_2023_pepe_crisis.py --resume")
    print(f"📊 Run this monitor again: python src/scripts/monitor_pepe_crisis_progress.py")
    print(f"🎯 Expected outcome: ~40,000 blocks with 184+ gwei peak for Taiko testing")


if __name__ == "__main__":
    monitor_pepe_progress()