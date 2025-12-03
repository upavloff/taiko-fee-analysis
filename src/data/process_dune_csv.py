
#!/usr/bin/env python3
"""
Process Dune CSV Export for Alpha-Data Calculation

Usage:
1. Export Taiko proposeBlock data from Dune as CSV
2. Run: python3 process_dune_csv.py taiko_proposeblock_data.csv
3. Get alpha_data statistics and recommendations
"""

import pandas as pd
import numpy as np
import sys

def process_dune_csv(csv_file):
    """Process Dune CSV export to calculate alpha_data"""

    print(f"📊 Processing Dune CSV: {csv_file}")

    # Load data
    df = pd.read_csv(csv_file)

    # Expected columns from Dune query:
    # - block_number, block_time, tx_hash, gas_used, l1_cost_wei, da_mode

    print(f"📈 Data loaded: {len(df)} transactions")
    print(f"📅 Date range: {df['block_time'].min()} to {df['block_time'].max()}")

    # Calculate alpha for each transaction
    # Note: We need L2 gas data - might use fixed estimate initially
    l2_gas_per_batch = 690_000  # Current calibration

    df['alpha_data'] = df['gas_used'] / l2_gas_per_batch

    # Calculate statistics
    alpha_stats = {
        'mean': df['alpha_data'].mean(),
        'median': df['alpha_data'].median(),
        'std': df['alpha_data'].std(),
        'min': df['alpha_data'].min(),
        'max': df['alpha_data'].max(),
        'p5': df['alpha_data'].quantile(0.05),
        'p95': df['alpha_data'].quantile(0.95)
    }

    print("🎯 ALPHA-DATA STATISTICS:")
    for key, value in alpha_stats.items():
        print(f"   {key}: {value:.4f}")

    # Regime analysis if da_mode column exists
    if 'da_mode' in df.columns:
        blob_data = df[df['da_mode'] == 'blob']['alpha_data']
        calldata_data = df[df['da_mode'] == 'calldata']['alpha_data']

        print("\n🔬 REGIME ANALYSIS:")
        if len(blob_data) > 0:
            print(f"   Blob mode: α = {blob_data.mean():.3f} ({len(blob_data)} samples)")
        if len(calldata_data) > 0:
            print(f"   Calldata mode: α = {calldata_data.mean():.3f} ({len(calldata_data)} samples)")

    # Deployment recommendation
    print("\n🚀 DEPLOYMENT RECOMMENDATION:")
    print(f"   Recommended α_data: {alpha_stats['median']:.3f}")

    if 0.15 <= alpha_stats['median'] <= 0.28:
        print("   ✅ Within expected range (0.15-0.28)")
    else:
        print("   ⚠️  Outside expected range - verify data")

    return alpha_stats

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 process_dune_csv.py <taiko_data.csv>")
        sys.exit(1)

    process_dune_csv(sys.argv[1])
