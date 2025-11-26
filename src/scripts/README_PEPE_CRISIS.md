# May 2023 PEPE Memecoin Crisis Data Fetching

## Overview
This directory contains scripts to fetch Ethereum basefee data from the May 2023 PEPE memecoin crisis period, when basefees spiked to 184+ gwei on May 5, 2023. This data is crucial for testing the Taiko fee mechanism under extreme conditions.

## Target Data
- **Period**: May 3-7, 2023
- **Block Range**: 17,180,000 - 17,220,000 (~40,000 blocks)
- **Peak Crisis**: May 5, 2023 with basefees reaching 184+ gwei
- **Expected Output**: `data_cache/may_2023_pepe_crisis_data.csv`

## Scripts

### 1. Main Fetching Script: `fetch_may_2023_pepe_crisis.py`
Robust data fetcher with multiple RPC endpoints, retry logic, and progress tracking.

```bash
# Activate Python environment
source ./venv/bin/activate

# Test run (20 blocks around peak crisis)
python src/scripts/fetch_may_2023_pepe_crisis.py --test

# Full run (all 40,000 blocks)
python src/scripts/fetch_may_2023_pepe_crisis.py --start-block 17180000 --end-block 17220000

# Resume from checkpoint (if interrupted)
python src/scripts/fetch_may_2023_pepe_crisis.py --resume
```

### 2. Progress Monitoring: `monitor_pepe_crisis_progress.py`
Tracks fetching progress and provides PEPE crisis-specific statistics.

```bash
# Monitor progress (run in separate terminal)
python src/scripts/monitor_pepe_crisis_progress.py
```

## Key Features

### Error Handling & Reliability
- **Multiple RPC endpoints** for redundancy
- **Automatic retry logic** with exponential backoff
- **Rate limiting** to respect RPC provider limits
- **Checkpoint/resume capability** for long runs

### PEPE Crisis Specific Tracking
- **Real-time max basefee detection** with alerts
- **High fee block counting** (>100 gwei, >150 gwei)
- **Crisis-specific progress reporting**
- **Expected peak detection** (~184 gwei)

### Data Format
Follows the standard project CSV format:
```csv
timestamp,basefee_wei,basefee_gwei,block_number
2023-05-05 14:23:47,184567123456,184.567123456,0x106a7b2
```

## Estimated Runtime
- **Full range**: ~8-12 hours (depending on RPC performance)
- **Rate**: ~1-1.5 blocks/second average
- **Data size**: ~6-8 MB for complete dataset

## Usage Tips

### For Development/Testing
```bash
# Quick test to verify setup
python src/scripts/fetch_may_2023_pepe_crisis.py --test

# Monitor test results
python src/scripts/monitor_pepe_crisis_progress.py
```

### For Production Data Collection
```bash
# Start the full fetch (run in background or screen session)
python src/scripts/fetch_may_2023_pepe_crisis.py --start-block 17180000 --end-block 17220000

# Monitor progress from another terminal
watch -n 30 "python src/scripts/monitor_pepe_crisis_progress.py"
```

### If Interrupted
```bash
# Resume from last checkpoint
python src/scripts/fetch_may_2023_pepe_crisis.py --resume
```

## Expected Output Statistics
Based on historical research, the complete dataset should contain:
- **Maximum basefee**: ~184+ gwei (May 5, 2023)
- **High fee blocks** (>100 gwei): Several hundred blocks
- **Extreme fee blocks** (>150 gwei): Dozens of blocks
- **Average basefee**: Significantly higher than normal periods

## Integration with Taiko Analysis
Once collected, this data can be used with:
- `src/core/fee_mechanism_simulator.py` - Test Taiko mechanism under extreme conditions
- Web interface - Compare PEPE crisis vs normal periods
- Analysis notebooks - Validate fee mechanism performance

## Troubleshooting

### RPC Issues
- Script automatically rotates between 8 RPC endpoints
- If all endpoints are rate-limited, script will backoff and retry
- Check logs for specific RPC errors

### Disk Space
- Ensure ~10+ MB free space for complete dataset
- Logs and checkpoints add additional ~1-2 MB

### Network Issues
- Script is designed to handle temporary network interruptions
- Use `--resume` to continue after network issues

## Files Generated
- `may_2023_pepe_crisis_data.csv` - Main dataset
- `may_2023_pepe_crisis_data.log` - Detailed fetch logs
- `may_2023_pepe_crisis_data.checkpoint.json` - Resume checkpoint (deleted on completion)