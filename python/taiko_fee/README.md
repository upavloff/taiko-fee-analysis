# Taiko Fee Analysis - Source Code Architecture

This directory contains the organized source code for the Taiko fee analysis project, structured following Python best practices.

## 📁 Directory Structure

```
src/
├── core/                          # Core simulation logic
│   ├── fee_mechanism_simulator.py # Main fee mechanism simulator
│   ├── improved_simulator.py      # Enhanced simulator with real data
│   └── __init__.py
├── data/                          # Data management modules
│   ├── fetchers/                  # Blockchain data fetching
│   │   ├── ethereum_rpc_client.py # Robust RPC client with retry logic
│   │   ├── block_fetcher.py       # High-level block data fetcher
│   │   └── __init__.py
│   ├── validators/                # Data validation and analysis
│   │   ├── contiguity_analyzer.py # Block contiguity analysis
│   │   └── __init__.py
│   ├── rpc_data_fetcher.py        # Legacy RPC fetcher
│   ├── real_data_fetcher.py       # Legacy data fetcher
│   └── __init__.py
├── analysis/                      # Analysis and metrics
│   ├── mechanism_metrics.py       # Fee mechanism analysis
│   └── __init__.py
├── utils/                         # Utility functions
│   ├── vault_initialization_demo.py
│   └── __init__.py
├── scripts/                       # Command-line tools
│   ├── cli.py                     # Unified CLI interface
│   ├── fetch_contiguous_data.py   # Data fetching script
│   ├── analyze_data_contiguity.py # Data analysis script
│   └── __init__.py
└── __init__.py
```

## 🚀 Quick Start

### Using the CLI

The unified CLI provides all data management functionality:

```bash
# Fetch 1 hour of recent contiguous block data
python src/scripts/cli.py fetch --hours 1

# Fetch 3 hours of recent contiguous block data
python src/scripts/cli.py fetch --hours 3

# Fetch latest 500 blocks
python src/scripts/cli.py fetch --blocks 500

# Fetch specific block range
python src/scripts/cli.py fetch --range 18000000-18000100

# Analyze all datasets for contiguity
python src/scripts/cli.py analyze

# Analyze specific file
python src/scripts/cli.py analyze --file data/data_cache/my_data.csv

# Verbose output
python src/scripts/cli.py -v fetch --hours 1
```

### Using Individual Scripts

```bash
# Fetch contiguous data
python src/scripts/fetch_contiguous_data.py

# Analyze data contiguity
python src/scripts/analyze_data_contiguity.py
```

### Using as Python Modules

```python
from src.data import BlockFetcher, ContiguityAnalyzer, EthereumRPCClient

# Fetch data programmatically
fetcher = BlockFetcher()
df = fetcher.fetch_recent_blocks(hours=1)

# Analyze data contiguity
analysis = ContiguityAnalyzer.analyze_dataframe(df, "My Dataset")
print(f"Contiguous: {analysis.is_contiguous}")

# Use RPC client directly
client = EthereumRPCClient(['https://eth.llamarpc.com'])
block = client.get_block(18000000)
```

## 🏗️ Architecture Principles

### 1. **Separation of Concerns**
- **Core**: Pure simulation logic
- **Data**: All data management (fetching, validation, storage)
- **Analysis**: Metrics and analysis algorithms
- **Scripts**: CLI tools and automation
- **Utils**: Shared utilities

### 2. **Robust Data Fetching**
- Multiple RPC endpoint support with automatic failover
- Retry logic with exponential backoff
- Rate limiting to respect API limits
- Comprehensive error handling and logging

### 3. **Data Validation**
- Block-by-block contiguity analysis
- Gap detection and reporting
- Data completeness validation
- Timestamp consistency checks

### 4. **Modular Design**
- Each module has a single responsibility
- Clean interfaces between components
- Easy to test and extend
- Follows Python packaging conventions

## 📊 Data Management

### Fetching Real Data

The `BlockFetcher` class provides high-level data fetching:

```python
from src.data import BlockFetcher

fetcher = BlockFetcher()

# Fetch recent data by time period
df_1h = fetcher.fetch_recent_blocks(1)    # 1 hour (~300 blocks)
df_3h = fetcher.fetch_recent_blocks(3)    # 3 hours (~900 blocks)

# Fetch specific block ranges
df_range = fetcher.fetch_contiguous_blocks(18000000, 1000)

# Save to CSV
fetcher.save_to_csv(df_1h, 'data/1hour_data.csv', '1-hour dataset')
```

### Data Validation

The `ContiguityAnalyzer` ensures data quality:

```python
from src.data import ContiguityAnalyzer

# Analyze a dataset
analysis = ContiguityAnalyzer.analyze_dataset('data/my_data.csv', 'My Dataset')

print(f"Contiguous: {analysis.is_contiguous}")
print(f"Blocks: {analysis.total_blocks}/{analysis.expected_blocks}")
print(f"Missing: {analysis.missing_blocks}")
print(f"Gaps: {len(analysis.gaps)}")

# Analyze multiple datasets
datasets = [
    ('data/dataset1.csv', 'Dataset 1'),
    ('data/dataset2.csv', 'Dataset 2')
]
analyses = ContiguityAnalyzer.analyze_multiple_datasets(datasets)
```

## 🔧 Configuration

### RPC Endpoints

The system uses multiple public RPC endpoints for reliability:

- `https://eth.llamarpc.com`
- `https://ethereum-rpc.publicnode.com`
- `https://ethereum.blockpi.network/v1/rpc/public`
- `https://rpc.ankr.com/eth`
- `https://cloudflare-eth.com`

You can customize endpoints:

```python
custom_rpcs = ['https://my-rpc.com', 'https://backup-rpc.com']
fetcher = BlockFetcher(rpc_urls=custom_rpcs)
```

### Logging

All modules use Python's standard logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure specifically
logging.getLogger('src.data.fetchers').setLevel(logging.INFO)
```

## 📈 Data Format

All datasets follow a consistent CSV format:

```csv
timestamp,basefee_wei,basefee_gwei,block_number
2025-11-26 15:00:00,1000000000,1.0,18000000
2025-11-26 15:00:12,1100000000,1.1,18000001
...
```

Where:
- `timestamp`: Block timestamp (UTC)
- `basefee_wei`: Base fee in wei (integer)
- `basefee_gwei`: Base fee in gwei (float)
- `block_number`: Ethereum block number (integer)

## 🧪 Testing

Run contiguity analysis on your data:

```bash
# Quick test
python src/scripts/cli.py analyze

# Detailed analysis
python src/scripts/cli.py -v analyze --file data/data_cache/my_data.csv
```

## 📝 Migration from Old Scripts

The old scattered scripts have been replaced:

- ❌ `analyze_data_contiguity.py` → ✅ `src/scripts/analyze_data_contiguity.py`
- ❌ `fetch_missing_blocks.py` → ✅ `src/data/fetchers/`
- ❌ `quick_contiguous_fetch.py` → ✅ `src/scripts/cli.py fetch`

All functionality is preserved with improved architecture, error handling, and usability.