#!/usr/bin/env python3
"""
Fetch Taiko Data using Etherscan API

This script uses Etherscan API to get Taiko proposeBlock transactions
more efficiently than scanning blocks directly.
"""

import requests
import time
import pandas as pd
import sys

# Etherscan API configuration
API_KEY = "2X66SY2WQRS6R7I8WCCEWA5CITGWWY7HK2"
TAIKO_CONTRACT = "0xe84dc8e2a21e59426542ab040d77f81d6db881ee"
PROPOSE_BLOCK_SIG = "0x092bfe76"

def get_contract_transactions(max_results=100):
    """Get transactions to Taiko contract using Etherscan API."""
    print(f"📡 Fetching Taiko contract transactions from Etherscan...")

    url = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "txlist",
        "address": TAIKO_CONTRACT,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": max_results,
        "sort": "desc",  # Most recent first
        "apikey": API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if data["status"] != "1":
            print(f"❌ Etherscan API error: {data.get('message', 'Unknown error')}")
            return []

        transactions = data["result"]
        print(f"📊 Retrieved {len(transactions)} total transactions to Taiko contract")

        # Filter for proposeBlock transactions
        propose_block_txs = []
        for tx in transactions:
            if tx.get("input", "").startswith(PROPOSE_BLOCK_SIG):
                propose_block_txs.append({
                    "hash": tx["hash"],
                    "block_number": int(tx["blockNumber"]),
                    "block_time": pd.to_datetime(int(tx["timeStamp"]), unit='s'),
                    "gas_used": int(tx["gasUsed"]) if tx.get("gasUsed") else 0,
                    "gas_price": int(tx["gasPrice"]) if tx.get("gasPrice") else 0,
                    "gas_limit": int(tx["gas"]) if tx.get("gas") else 0,
                    "tx_fee_wei": int(tx["gasUsed"]) * int(tx["gasPrice"]) if tx.get("gasUsed") and tx.get("gasPrice") else 0
                })

        print(f"🎯 Found {len(propose_block_txs)} proposeBlock transactions")
        return propose_block_txs

    except Exception as e:
        print(f"❌ Etherscan API request failed: {e}")
        return []

def calculate_alpha_data_from_etherscan(transactions):
    """Calculate α_data from Etherscan transaction data."""
    if not transactions:
        print("❌ No transactions to analyze")
        return None

    print(f"📊 Calculating α_data from {len(transactions)} transactions")

    # Convert to DataFrame
    df = pd.DataFrame(transactions)

    # Use current conservative estimate for Q̄
    Q_bar = 200_000  # L2 gas per batch

    # Calculate alpha_data for each transaction
    df['alpha_data'] = df['gas_used'] / Q_bar

    # Calculate statistics
    stats = {
        'count': len(df),
        'mean': df['alpha_data'].mean(),
        'median': df['alpha_data'].median(),
        'std': df['alpha_data'].std(),
        'min': df['alpha_data'].min(),
        'max': df['alpha_data'].max(),
        'p5': df['alpha_data'].quantile(0.05),
        'p25': df['alpha_data'].quantile(0.25),
        'p75': df['alpha_data'].quantile(0.75),
        'p95': df['alpha_data'].quantile(0.95)
    }

    print(f"\n🎯 ETHERSCAN α_data STATISTICS:")
    print(f"   Sample size: {stats['count']}")
    print(f"   Mean: {stats['mean']:.4f}")
    print(f"   Median: {stats['median']:.4f}")
    print(f"   Std Dev: {stats['std']:.4f}")
    print(f"   Range: {stats['min']:.4f} - {stats['max']:.4f}")
    print(f"   5th-95th percentile: {stats['p5']:.4f} - {stats['p95']:.4f}")

    # Validation against theoretical range
    theoretical_range = (0.15, 0.28)
    measured_alpha = stats['median']

    print(f"\n🔬 THEORETICAL VALIDATION:")
    print(f"   Expected range: {theoretical_range[0]:.3f} - {theoretical_range[1]:.3f}")
    print(f"   Measured α_data: {measured_alpha:.3f}")

    if theoretical_range[0] <= measured_alpha <= theoretical_range[1]:
        print(f"   ✅ WITHIN EXPECTED RANGE")
    else:
        print(f"   ⚠️  OUTSIDE EXPECTED RANGE")

    # Save results
    if len(df) > 0:
        csv_file = "taiko_etherscan_data.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Data saved to: {csv_file}")

    return {
        'statistics': stats,
        'transactions': df,
        'recommended_alpha': measured_alpha,
        'data_source': 'etherscan'
    }

def main():
    """Main function."""
    print("🎯 TAIKO DATA FETCHING VIA ETHERSCAN")
    print("=" * 50)

    # Get transactions from Etherscan
    transactions = get_contract_transactions(max_results=50)

    if not transactions:
        print("❌ No Taiko transactions found via Etherscan")
        print("🔄 This could mean:")
        print("   - Taiko activity is very low recently")
        print("   - Contract address might be different")
        print("   - API rate limits hit")
        return None

    # Analyze the data
    results = calculate_alpha_data_from_etherscan(transactions)

    if results:
        print(f"\n🎉 ETHERSCAN DATA COLLECTION SUCCESS!")
        print(f"✅ Recommended α_data: {results['recommended_alpha']:.4f}")
        print(f"📊 Based on {results['statistics']['count']} real transactions")

        # Compare with current wrong estimate
        current_wrong = 0.5
        improvement = results['recommended_alpha'] / current_wrong
        print(f"🔧 Current wrong estimate: {current_wrong}")
        print(f"📈 Improvement factor: {improvement:.2f}x")

    return results

if __name__ == "__main__":
    results = main()