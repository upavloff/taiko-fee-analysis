# Taiko Fee Analysis - Data Management

This directory contains all datasets used in the Taiko fee mechanism analysis project.

## 📁 Structure

```
data/
├── cache/              # Processed cache files (.csv)
├── raw/               # Original raw data files
├── external/          # External data sources
└── results/           # Generated analysis outputs
```

## 📊 Datasets

### Cached Data (`cache/`)

**Core Historical Datasets:**
- **`recent_low_fees_3hours.csv`**: Nov 2025 low fee period (0.055-0.092 gwei)
- **`luna_crash_true_peak_contiguous.csv`**: May 2022 UST/Luna crash (53-533 gwei)
- **`real_july_2022_spike_data.csv`**: July 2022 market volatility (7-88 gwei)

**Specialized Datasets:**
- **`may_2023_pepe_crisis_data.csv`**: PEPE memecoin crisis data (60-184 gwei)
- **Additional crisis scenarios for stress testing**

### CSV Format Standard
All datasets use this exact format:
```csv
timestamp,basefee_wei,basefee_gwei,block_number
2022-07-01 08:46:46,12999038238,12.999038238,0xe5b8ec
```

### Data Sources (`external/`)
- **Ethereum RPC endpoints**: Historical basefee data
- **External APIs**: Market data, network statistics
- **Third-party datasets**: Academic research data

## 🔄 Data Pipeline

### Data Fetching
```bash
# Fetch new historical data
cd python
python -m taiko_fee.data.rpc_data_fetcher --start-block X --end-block Y

# Process and cache data
python -m taiko_fee.data.real_data_fetcher --process-range "2023-11-20" "2023-11-23"
```

### Data Validation
- **Continuity checks**: Ensure no missing blocks
- **Range validation**: Verify realistic basefee values
- **Timestamp consistency**: Check chronological ordering

## 📈 Usage in Analysis

### Python Package
```python
from taiko_fee.data import load_cached_data

# Load specific dataset
df = load_cached_data('recent_low_fees_3hours')

# Load multiple scenarios
datasets = load_cached_data(['july_2022', 'luna_crash', 'recent_low'])
```

### Web Interface
Data is automatically loaded by the historical data loader:
```javascript
// Loads from data_cache/*.csv (symlinked to data/cache/)
const datasets = await loadHistoricalData();
```

### Research Notebooks
```python
import pandas as pd

# Direct loading
df = pd.read_csv('../data/cache/luna_crash_true_peak_contiguous.csv')

# Processed loading with validation
from taiko_fee.data import DataValidator
validator = DataValidator()
df = validator.load_and_validate('luna_crash_true_peak_contiguous.csv')
```

## 🗃️ Data Standards

### Block Range Conventions
- **Hex format**: Use `0xe5b8ec` format for block numbers
- **Contiguous ranges**: No gaps in block sequences
- **Post-EIP-1559 only**: Data from August 5, 2021+

### Quality Standards
- **Minimum duration**: 1 hour continuous data
- **Maximum gaps**: No more than 5 missing blocks per 1000
- **Validation required**: All datasets pass continuity checks

### File Naming
- **Format**: `{scenario}_{detail}_{duration}.csv`
- **Examples**:
  - `luna_crash_true_peak_contiguous.csv`
  - `recent_low_fees_3hours.csv`
  - `july_2022_spike_data.csv`

## 🚀 Data Collection Scripts

### RPC Data Fetcher
```bash
python -m taiko_fee.data.rpc_data_fetcher \
  --start-block 15055000 \
  --end-block 15064900 \
  --output july_2022_spike_data.csv
```

### Real Data Integrator
```bash
python -m taiko_fee.data.real_data_fetcher \
  --date-range "2023-11-20" "2023-11-23" \
  --cache-only \
  --validate
```

### Monitoring Script
```bash
python scripts/monitor_data_health.py --check-all
```

## 🔒 Data Governance

### Version Control
- **Large files**: Use Git LFS for files > 100MB
- **Critical datasets**: Always tracked (exceptions in .gitignore)
- **Generated data**: Excluded from git (results/, processed/)

### Access Control
- **Public datasets**: Available via GitHub
- **Sensitive data**: Environment variable configuration
- **API keys**: Never committed to repository

### Backup Strategy
- **Critical datasets**: Backed up to multiple locations
- **Reproduction**: All data can be regenerated from scripts
- **Documentation**: Data lineage clearly documented

## 📋 Data TODO

### Priority Tasks
- [ ] Implement automated data validation pipeline
- [ ] Add data freshness monitoring
- [ ] Create data quality dashboard
- [ ] Set up automated backups

### Research Needs
- [ ] More crisis scenario datasets
- [ ] Long-term historical data (multi-year)
- [ ] Cross-chain comparative data
- [ ] MEV-aware transaction data

## 🆘 Data Issues

### Troubleshooting
```bash
# Check data integrity
python scripts/validate_data.py --dataset all

# Repair corrupted cache
python scripts/repair_cache.py --force-rebuild

# Update stale data
python scripts/update_data.py --check-freshness
```

### Common Issues
- **Missing blocks**: Use `scripts/fill_gaps.py`
- **Timestamp errors**: Use `scripts/fix_timestamps.py`
- **Format inconsistencies**: Use `scripts/normalize_format.py`

## 📚 Data Sources

### Primary Sources
- **Ethereum RPC**: Alchemy, Infura, QuickNode
- **Historical APIs**: Etherscan, The Graph
- **Academic sources**: Research institution datasets

### Data Attribution
All external data sources are properly attributed and licensed according to their terms of use.

---

*Ensuring high-quality, reproducible data foundation for Taiko fee mechanism research.*