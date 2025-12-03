#!/usr/bin/env python3
"""
Fetch Real Taiko L1 Data for Alpha-Data Calculation

This script attempts to fetch actual proposeBlock transactions from Taiko L1 contract
to calculate empirical α_data values.
"""

import asyncio
import sys
import os
import requests
import time
from typing import List, Dict, Any
import pandas as pd

# Add python directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

# RPC endpoints (free tier)
RPC_ENDPOINTS = [
    "https://eth.llamarpc.com",
    "https://rpc.ankr.com/eth",
    "https://ethereum.publicnode.com",
    "https://eth.drpc.org",
    "https://eth-pokt.nodies.app"
]

# Taiko contracts
TAIKO_L1_CONTRACT = "0xe84dc8e2a21e59426542ab040d77f81d6db881ee"
PROPOSE_BLOCK_SIG = "0x092bfe76"  # proposeBlock function signature


def get_latest_block_number(rpc_url: str) -> int:
    """Get the latest Ethereum block number"""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_blockNumber",
        "params": [],
        "id": 1
    }

    try:
        response = requests.post(rpc_url, json=payload, timeout=10)
        result = response.json()
        if "result" in result:
            return int(result["result"], 16)
    except Exception as e:
        print(f"Error getting block number from {rpc_url}: {e}")

    return None


def get_block_transactions(rpc_url: str, block_number: int) -> List[Dict]:
    """Get all transactions from a specific block"""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [hex(block_number), True],  # True = include full transaction objects
        "id": 1
    }

    try:
        response = requests.post(rpc_url, json=payload, timeout=15)
        result = response.json()
        if "result" in result and result["result"]:
            return result["result"].get("transactions", [])
    except Exception as e:
        print(f"Error getting block {block_number} from {rpc_url}: {e}")

    return []


def filter_taiko_transactions(transactions: List[Dict]) -> List[Dict]:
    """Filter transactions for Taiko proposeBlock calls"""
    taiko_txs = []

    for tx in transactions:
        if (tx.get("to") and
            tx["to"].lower() == TAIKO_L1_CONTRACT.lower() and
            tx.get("input") and
            tx["input"].startswith(PROPOSE_BLOCK_SIG)):

            taiko_txs.append({
                "hash": tx["hash"],
                "blockNumber": int(tx["blockNumber"], 16),
                "gasUsed": None,  # Need to get from receipt
                "gasPrice": int(tx["gasPrice"], 16) if tx.get("gasPrice") else 0,
                "input": tx["input"],
                "dataSize": len(tx["input"]) // 2 if tx.get("input") else 0
            })

    return taiko_txs


def get_transaction_receipt(rpc_url: str, tx_hash: str) -> Dict:
    """Get transaction receipt for gas usage"""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getTransactionReceipt",
        "params": [tx_hash],
        "id": 1
    }

    try:
        response = requests.post(rpc_url, json=payload, timeout=10)
        result = response.json()
        if "result" in result and result["result"]:
            receipt = result["result"]
            return {
                "gasUsed": int(receipt["gasUsed"], 16) if receipt.get("gasUsed") else 0,
                "status": receipt.get("status") == "0x1"
            }
    except Exception as e:
        print(f"Error getting receipt for {tx_hash}: {e}")

    return {"gasUsed": 0, "status": False}


def scan_recent_blocks(rpc_url: str, num_blocks: int = 1000) -> List[Dict]:
    """Scan recent blocks for Taiko transactions"""
    print(f"🔍 Scanning last {num_blocks} blocks using {rpc_url}")

    # Get latest block
    latest_block = get_latest_block_number(rpc_url)
    if not latest_block:
        print(f"❌ Could not get latest block from {rpc_url}")
        return []

    print(f"📊 Latest block: {latest_block}")
    start_block = latest_block - num_blocks

    taiko_transactions = []
    blocks_scanned = 0

    # Scan blocks in reverse order (most recent first)
    for block_num in range(latest_block, start_block, -1):
        try:
            # Rate limiting
            time.sleep(0.1)

            transactions = get_block_transactions(rpc_url, block_num)
            taiko_txs = filter_taiko_transactions(transactions)

            if taiko_txs:
                print(f"✅ Block {block_num}: Found {len(taiko_txs)} Taiko transactions")

                # Get receipts for gas usage
                for tx in taiko_txs:
                    time.sleep(0.05)  # Rate limiting
                    receipt = get_transaction_receipt(rpc_url, tx["hash"])
                    tx["gasUsed"] = receipt["gasUsed"]
                    tx["status"] = receipt["status"]

                    if tx["status"] and tx["gasUsed"] > 0:
                        taiko_transactions.append(tx)

            blocks_scanned += 1
            if blocks_scanned % 100 == 0:
                progress = (blocks_scanned / num_blocks) * 100
                print(f"📈 Progress: {progress:.1f}% ({blocks_scanned}/{num_blocks} blocks)")
                print(f"🎯 Found {len(taiko_transactions)} valid Taiko transactions so far")

            # Early exit if we have enough data
            if len(taiko_transactions) >= 100:
                print(f"✅ Collected {len(taiko_transactions)} transactions, stopping early")
                break

        except Exception as e:
            print(f"❌ Error scanning block {block_num}: {e}")
            continue

    print(f"🎉 Scan complete: {len(taiko_transactions)} valid Taiko transactions found")
    return taiko_transactions


def calculate_alpha_data(taiko_transactions: List[Dict]) -> Dict[str, Any]:
    """Calculate alpha-data from Taiko transactions"""
    if not taiko_transactions:
        print("❌ No Taiko transactions to analyze")
        return {}

    print(f"📊 Calculating α_data from {len(taiko_transactions)} transactions")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(taiko_transactions)

    # Current Q̄ calibration for comparison
    l2_gas_per_batch = 690_000

    # Calculate alpha for each transaction
    df['alpha_data'] = df['gasUsed'] / l2_gas_per_batch

    # Calculate statistics
    alpha_stats = {
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

    print("🎯 EMPIRICAL ALPHA-DATA STATISTICS:")
    print("-" * 50)
    print(f"Sample size: {alpha_stats['count']}")
    print(f"Mean: {alpha_stats['mean']:.4f}")
    print(f"Median: {alpha_stats['median']:.4f}")
    print(f"Std Dev: {alpha_stats['std']:.4f}")
    print(f"Range: {alpha_stats['min']:.4f} - {alpha_stats['max']:.4f}")
    print(f"5th-95th percentile: {alpha_stats['p5']:.4f} - {alpha_stats['p95']:.4f}")
    print()

    # Compare with theoretical expectations
    expected_range = (0.15, 0.28)
    measured_alpha = alpha_stats['median']

    print("🔬 VALIDATION AGAINST THEORETICAL EXPECTATIONS:")
    print(f"Expected range: {expected_range[0]:.3f} - {expected_range[1]:.3f}")
    print(f"Measured α_data: {measured_alpha:.3f}")

    if expected_range[0] <= measured_alpha <= expected_range[1]:
        print("✅ WITHIN EXPECTED RANGE - Ready for deployment!")
    else:
        print("⚠️  OUTSIDE EXPECTED RANGE - Needs investigation")

    print()

    # Deployment recommendation
    print("🚀 DEPLOYMENT RECOMMENDATION:")
    if alpha_stats['count'] >= 50:
        print(f"✅ Sufficient data: Deploy α_data = {measured_alpha:.3f}")
        print(f"🔄 Replace: Q̄ = 690,000 → α_data = {measured_alpha:.3f}")

        # Calculate expected fee improvement
        l1_basefee_gwei = 20  # Example L1 basefee
        l1_basefee_wei = l1_basefee_gwei * 1e9

        # Alpha model fee
        da_component = measured_alpha * l1_basefee_wei
        proof_component = 180_000 * l1_basefee_wei / l2_gas_per_batch
        alpha_fee_gwei = (da_component + proof_component) / 1e9

        print(f"📈 Expected fees at {l1_basefee_gwei} gwei L1:")
        print(f"   Alpha model: {alpha_fee_gwei:.2f} gwei")
        print(f"   vs Q̄ model: ~4 gwei (broken)")

    else:
        print(f"⚠️  Insufficient data: Only {alpha_stats['count']} samples")
        print("🔄 Recommend collecting more data or using theoretical α = 0.22")

    return {
        'statistics': alpha_stats,
        'transactions': df,
        'deployment_ready': alpha_stats['count'] >= 50,
        'recommended_alpha': measured_alpha
    }


def save_results(taiko_transactions: List[Dict], alpha_results: Dict[str, Any]):
    """Save results for further analysis"""
    if taiko_transactions:
        # Save raw transaction data
        df = pd.DataFrame(taiko_transactions)
        csv_file = "taiko_l1_transactions.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 Raw data saved: {csv_file}")

        # Save analysis results
        if 'statistics' in alpha_results:
            stats_df = pd.DataFrame([alpha_results['statistics']])
            stats_file = "alpha_data_statistics.csv"
            stats_df.to_csv(stats_file, index=False)
            print(f"📊 Statistics saved: {stats_file}")


def main():
    """Main function to fetch and analyze Taiko data"""
    print("🎯 TAIKO L1 DATA FETCHING & ALPHA-DATA CALCULATION")
    print("=" * 60)
    print()

    print("📋 TARGET:")
    print(f"   Contract: {TAIKO_L1_CONTRACT}")
    print(f"   Function: proposeBlock ({PROPOSE_BLOCK_SIG})")
    print("   Goal: Calculate empirical α_data for fee mechanism")
    print()

    # Try each RPC endpoint until we find one that works
    for i, rpc_url in enumerate(RPC_ENDPOINTS):
        print(f"🌐 Trying RPC endpoint {i+1}/{len(RPC_ENDPOINTS)}: {rpc_url}")

        try:
            # Test connectivity
            latest_block = get_latest_block_number(rpc_url)
            if not latest_block:
                print(f"❌ Cannot connect to {rpc_url}")
                continue

            print(f"✅ Connected! Latest block: {latest_block}")

            # Scan for Taiko transactions
            taiko_transactions = scan_recent_blocks(rpc_url, num_blocks=2000)

            if taiko_transactions:
                print(f"🎉 Successfully collected {len(taiko_transactions)} Taiko transactions!")

                # Calculate alpha-data
                alpha_results = calculate_alpha_data(taiko_transactions)

                # Save results
                save_results(taiko_transactions, alpha_results)

                print("=" * 60)
                print("🎉 TAIKO ALPHA-DATA COLLECTION COMPLETE!")

                if alpha_results.get('deployment_ready'):
                    alpha = alpha_results['recommended_alpha']
                    print(f"✅ READY FOR DEPLOYMENT: α_data = {alpha:.3f}")
                    print("🔄 Replace broken Q̄ = 690,000 with empirical measurement")
                    print("📈 Expected: 0.00 gwei → 5-15 gwei realistic fees")
                else:
                    print("⚠️  More data collection recommended")
                    print("🔄 Consider using theoretical α = 0.22 for now")

                return alpha_results

            else:
                print(f"😞 No Taiko transactions found with {rpc_url}")
                print("🔄 Trying next RPC endpoint...")
                continue

        except Exception as e:
            print(f"❌ Error with {rpc_url}: {e}")
            continue

    print("❌ All RPC endpoints failed or no Taiko data found")
    print("🔄 FALLBACK RECOMMENDATION:")
    print("   1. Use Dune Analytics approach (see check_dune_data_availability.py)")
    print("   2. Deploy with theoretical α_data = 0.22")
    print("   3. Monitor and update with real data later")

    return {"deployment_ready": False, "recommended_fallback": 0.22}


if __name__ == "__main__":
    results = main()