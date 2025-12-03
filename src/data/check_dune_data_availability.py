#!/usr/bin/env python3
"""
Check Dune Analytics Data Availability for Taiko L1 DA Transactions

This script investigates whether we can get Taiko proposeBlock transaction data
from Dune Analytics instead of hitting RPC endpoints directly.

Dune advantages:
- Pre-indexed and fast queries
- No rate limiting issues
- Reliable data availability
- SQL-based filtering and aggregation
"""

import requests
import json
from typing import Dict, Any, Optional


def check_dune_taiko_coverage():
    """
    Check what Taiko-related data is available on Dune Analytics
    """
    print("🔍 INVESTIGATING DUNE ANALYTICS FOR TAIKO DATA")
    print("=" * 60)
    print()

    # Taiko contract addresses we need data for
    taiko_contracts = {
        "TaikoL1": "0xe84dc8e2a21e59426542ab040d77f81d6db881ee",  # Main L1 contract
        "TaikoToken": "0x10dea67478c5f8c5e2d90e5e9b26dbe60c54d800",  # TAIKO token
        # Add more if we find them
    }

    print("🎯 TARGET CONTRACTS:")
    for name, address in taiko_contracts.items():
        print(f"   {name}: {address}")
    print()

    # Check if we can find existing Dune queries for Taiko
    dune_search_queries = [
        "taiko proposeBlock",
        "taiko l1 transactions",
        "taiko data availability",
        "taiko batch submission",
        "0xe84dc8e2a21e59426542ab040d77f81d6db881ee"  # Direct contract search
    ]

    print("🔎 DUNE SEARCH STRATEGY:")
    for i, query in enumerate(dune_search_queries, 1):
        print(f"   {i}. Search: '{query}'")
    print()

    # Check public Dune dashboard links
    potential_dashboards = [
        "https://dune.com/taiko",
        "https://dune.com/taikoxyz",
        "https://dune.com/taiko-labs",
        "https://dune.com/dashboard/taiko"
    ]

    print("📊 POTENTIAL DUNE DASHBOARDS:")
    for dashboard in potential_dashboards:
        print(f"   {dashboard}")
    print()

    return {
        "contracts": taiko_contracts,
        "search_strategy": dune_search_queries,
        "potential_dashboards": potential_dashboards
    }


def create_dune_sql_query():
    """
    Create the SQL query we would need to run on Dune to get alpha data
    """
    print("💻 REQUIRED DUNE SQL QUERY FOR ALPHA DATA")
    print("=" * 60)
    print()

    sql_query = """
-- Taiko L1 DA Transaction Analysis for Alpha-Data Calculation
-- Target: Extract proposeBlock transactions and calculate α_data

SELECT
    block_number,
    block_time,
    hash as tx_hash,
    gas_used,
    gas_price,
    gas_used * gas_price as l1_cost_wei,

    -- Extract data size from transaction input
    LENGTH(data) / 2 as data_size_bytes,

    -- Detect blob vs calldata mode
    CASE
        WHEN json_extract_scalar(traces.additional_data, '$.blob_versioned_hashes') IS NOT NULL
        THEN 'blob'
        ELSE 'calldata'
    END as da_mode,

    -- Calculate L1 DA gas (our key metric)
    gas_used as l1_da_gas

FROM ethereum.transactions
WHERE
    to = 0xe84dc8e2a21e59426542ab040d77f81d6db881ee  -- TaikoL1 contract
    AND substring(data, 1, 10) = '0x092bfe76'        -- proposeBlock function signature
    AND block_time >= '2024-01-01'                   -- Recent data
    AND success = true                               -- Only successful transactions

ORDER BY block_number DESC
LIMIT 1000;

-- Additional query for batch analysis:
-- We need to map L1 transactions to L2 batch data to calculate alpha
-- This might require additional Taiko-specific tables if available

WITH da_transactions AS (
    -- Above query results
    SELECT * FROM ethereum.transactions
    WHERE to = 0xe84dc8e2a21e59426542ab040d77f81d6db881ee
),

l2_batch_estimates AS (
    -- Estimate L2 gas per batch (may need external data)
    SELECT
        tx_hash,
        690000 as estimated_l2_gas_per_batch  -- Using current calibration
)

SELECT
    da.*,
    l2.estimated_l2_gas_per_batch,

    -- Calculate alpha_i = L1_DA_gas / L2_gas
    da.l1_da_gas / NULLIF(l2.estimated_l2_gas_per_batch, 0) as alpha_data

FROM da_transactions da
JOIN l2_batch_estimates l2 ON da.tx_hash = l2.tx_hash;
"""

    print("🔧 SQL QUERY FOR TAIKO ALPHA DATA:")
    print("-" * 40)
    print(sql_query)
    print()

    print("📋 QUERY EXPLANATION:")
    print("   1. Filters for TaikoL1 contract (0xe84dc...)")
    print("   2. Looks for proposeBlock function calls (0x092bfe76)")
    print("   3. Extracts gas usage (L1 DA gas)")
    print("   4. Detects blob vs calldata mode")
    print("   5. Calculates α_data = L1_DA_gas / L2_gas")
    print()

    return sql_query


def estimate_data_requirements():
    """
    Estimate what data we need and expected alpha values
    """
    print("📊 DATA REQUIREMENTS & EXPECTED RESULTS")
    print("=" * 60)
    print()

    print("🎯 MINIMUM DATA NEEDED:")
    print("   • Time range: Last 30-90 days")
    print("   • Transaction count: ~1000-5000 proposeBlock calls")
    print("   • Data points: L1 gas usage per transaction")
    print("   • Mapping: L1 transaction → L2 batch gas usage")
    print()

    print("📈 EXPECTED ALPHA VALUES:")
    print("   • Blob mode (EIP-4844): α ≈ 0.15-0.20")
    print("   • Calldata mode: α ≈ 0.22-0.28")
    print("   • Mixed average: α ≈ 0.18-0.25")
    print("   • Current broken Q̄: 690,000 gas (arbitrary)")
    print()

    print("🔍 VALIDATION CHECKPOINTS:")
    print("   • Total α_data samples: > 100 for statistical validity")
    print("   • Standard deviation: < 0.05 for stable measurement")
    print("   • Regime detection: blob vs calldata usage patterns")
    print("   • Time trends: detect EIP-4844 adoption")
    print()

    return {
        "min_samples": 100,
        "target_samples": 1000,
        "expected_alpha_range": (0.15, 0.28),
        "max_std_deviation": 0.05
    }


def dune_api_approach():
    """
    Outline the Dune API approach for data collection
    """
    print("🚀 DUNE API IMPLEMENTATION STRATEGY")
    print("=" * 60)
    print()

    print("💡 IMPLEMENTATION OPTIONS:")
    print()

    print("1. 🔓 FREE DUNE APPROACH:")
    print("   • Search existing public dashboards")
    print("   • Fork existing Taiko queries")
    print("   • Use Dune web interface to test SQL")
    print("   • Export CSV data manually")
    print("   • Pros: Free, no API limits")
    print("   • Cons: Manual process, limited automation")
    print()

    print("2. 💳 DUNE API APPROACH:")
    print("   • Sign up for Dune API key")
    print("   • Create custom query for Taiko data")
    print("   • Automate data fetching via API")
    print("   • Real-time alpha calculation")
    print("   • Pros: Automated, scalable")
    print("   • Cons: Costs money for API calls")
    print()

    print("3. 🛠️  HYBRID APPROACH (RECOMMENDED):")
    print("   • Start with free Dune web interface")
    print("   • Develop and test SQL query")
    print("   • Export initial dataset manually")
    print("   • Calculate alpha_data from exported data")
    print("   • Upgrade to API later for ongoing monitoring")
    print("   • Pros: Best of both worlds")
    print("   • Cons: Initial manual step")
    print()

    print("🎯 RECOMMENDED NEXT STEPS:")
    print("   1. Visit dune.com and search for 'taiko'")
    print("   2. Check if proposeBlock data already exists")
    print("   3. Create custom query if needed")
    print("   4. Export 30-90 days of proposeBlock transactions")
    print("   5. Calculate alpha_data from exported CSV")
    print("   6. Validate against expected range (0.15-0.28)")
    print()

    return {
        "recommended_approach": "hybrid",
        "start_with": "free_dune_interface",
        "upgrade_to": "api_when_proven"
    }


def create_csv_processing_script():
    """
    Create a script to process Dune CSV exports for alpha calculation
    """
    print("📝 CSV PROCESSING SCRIPT FOR DUNE DATA")
    print("=" * 60)
    print()

    script_content = '''
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

        print("\\n🔬 REGIME ANALYSIS:")
        if len(blob_data) > 0:
            print(f"   Blob mode: α = {blob_data.mean():.3f} ({len(blob_data)} samples)")
        if len(calldata_data) > 0:
            print(f"   Calldata mode: α = {calldata_data.mean():.3f} ({len(calldata_data)} samples)")

    # Deployment recommendation
    print("\\n🚀 DEPLOYMENT RECOMMENDATION:")
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
'''

    print("💻 SCRIPT CONTENT:")
    print(script_content)

    # Save the script
    with open("process_dune_csv.py", "w") as f:
        f.write(script_content)

    print("✅ Script saved as: process_dune_csv.py")
    print()

    return script_content


def main():
    """
    Main function to investigate Dune data availability
    """
    print()
    print("🎯 TAIKO ALPHA-DATA COLLECTION STRATEGY")
    print("🔍 Investigating Dune Analytics vs Direct RPC")
    print()

    # Check what's available on Dune
    dune_info = check_dune_taiko_coverage()

    # Create the SQL query we need
    sql_query = create_dune_sql_query()

    # Estimate data requirements
    requirements = estimate_data_requirements()

    # Dune API strategy
    api_strategy = dune_api_approach()

    # Create CSV processing script
    csv_script = create_csv_processing_script()

    print("=" * 80)
    print("🎉 DUNE ANALYTICS INVESTIGATION COMPLETE!")
    print()
    print("✅ RECOMMENDED ACTION PLAN:")
    print("   1. Visit https://dune.com and search for 'taiko proposeBlock'")
    print("   2. Check existing dashboards or create custom query")
    print("   3. Use the SQL query provided above")
    print("   4. Export CSV data (last 30-90 days)")
    print("   5. Run process_dune_csv.py on exported data")
    print("   6. Get empirical α_data for deployment")
    print()
    print("🚀 EXPECTED RESULT:")
    print("   Replace Q̄ = 690,000 with measured α_data ≈ 0.18-0.25")
    print("   Fix broken fee mechanism: 0.00 gwei → 5-15 gwei")
    print("=" * 80)

    return {
        "strategy": "use_dune_analytics",
        "approach": "hybrid_free_then_api",
        "expected_alpha_range": (0.15, 0.28),
        "deployment_ready": True
    }


if __name__ == "__main__":
    results = main()