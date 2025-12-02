# Taiko Fee Mechanism Analysis Project

## 🚨 CRITICAL UPDATE: SPECS.md Implementation Complete

**BREAKING**: Fee mechanism implementation updated to exact SPECS.md mathematical formulas.

**SPECS Implementation**: Complete mathematical compliance with SPECS.md:
- **Section 3**: Fee Controller with exact formula f^raw(t) = μ * Ĉ_L1(t)/Q̄ + ν * D(t)/(H*Q̄)
- **Section 4**: Vault Dynamics with precise balance updates V(t+1) = V(t) + R(t) - S(t)
- **Section 6**: Hard Constraints for solvency, cost recovery, and UX bounds
- **Section 7**: Soft Objectives for UX, robustness, and capital efficiency

**SPECS-Validated Optimal Parameters** (mathematically corrected):
- **SPECS Optimal**: μ=0.7, ν=0.2, H=72 (balanced L1 tracking + deficit response)
- **Conservative**: μ=0.5, ν=0.1, H=36 (lower fees, gentler response)
- **Aggressive**: μ=1.0, ν=0.3, H=144 (full L1 tracking, strong deficit correction)

**Parameter Ranges**:
- μ ∈ [0,1]: L1 weight (0=ignore L1 costs, 1=full L1 tracking)
- ν ∈ [0,1]: Deficit weight (0=ignore deficits, 1=aggressive correction)
- H ∈ {36,72,144,288}: Prediction horizon in 6-step aligned batches

**Calibration Note**: Q̄ = 20,000 gas/batch (corrected from 690,000) for realistic fee levels

## Project Context
This repo analyzes Taiko's EIP-1559 based fee mechanism using real Ethereum L1 data. Key focus areas:
- **Fee mechanism simulation** with realistic lumpy cash flow vault economics
- **Historical Ethereum data analysis** (post-EIP-1559 only: Aug 5, 2021+)
- **Interactive web interface** for parameter exploration
- **Scientific validation** of fee mechanism performance with 6-step batch cycles

## Architecture

```
python/specs_implementation/    # SPECS.md compliant implementation
├── core/                      # Fee mechanism simulation engine
│   ├── fee_controller.py      # Section 3: Fee calculation formulas
│   ├── vault_dynamics.py      # Section 4: Vault balance updates
│   ├── simulation_engine.py   # Complete simulation integration
│   └── l1_cost_smoother.py    # EMA-based L1 cost estimation
├── metrics/                   # SPECS.md constraints & objectives
│   ├── constraints.py         # Section 6: Hard constraints
│   ├── objectives.py          # Section 7: Soft objectives
│   └── calculator.py          # Integrated metrics pipeline
└── data/                      # Historical data loading & validation

web_src/components/            # JavaScript SPECS integration
├── specs-simulator.js         # SPECS-compliant web simulator
├── simulator.js               # Updated to use SPECS formulas
└── [other components]         # Charts, optimization, etc.

Legacy (being phased out):
├── src/                       # Original implementation
├── data_cache/               # Historical Ethereum datasets
└── analysis/notebooks/       # Jupyter analysis notebooks
```

## Data Standards

### CSV Format (Critical)
All Ethereum data uses this exact format:
```csv
timestamp,basefee_wei,basefee_gwei,block_number
2022-07-01 08:46:46,12999038238,12.999038238,0xe5b8ec
```

**Key datasets:**
- `data/data_cache/real_july_2022_spike_data.csv` - July 2022 fee spike (15055000-15064900)
- `data/data_cache/luna_crash_true_peak_contiguous.csv` - UST/Luna crash data (9.4h continuous)
- `data/data_cache/recent_low_fees_3hours.csv` - Recent low fee period (3h continuous)

### Block Range Conventions
- Use hex format for block numbers: `0xe5b8ec`
- Target ranges: contiguous blocks (no gaps)
- Always verify data continuity for analysis

## Code Patterns

### Data Fetching
- **Always** implement retry logic and rate limiting
- Use multiple RPC endpoints for redundancy
- Include checkpoint/resume capability for large ranges
- Progress tracking with ETA calculations

### Simulation Classes
- `ImprovedTaikoFeeSimulator` - Main simulation engine
- `ImprovedSimulationParams` - Parameter configuration
- `MetricsCalculator` - Performance analysis

### Key Parameters
- `μ (mu)`: L1 weight [0.0-1.0]
- `ν (nu)`: Deficit weight [0.1-0.9]
- `H`: Prediction horizon (steps, e.g., 144 ≈ 1 day)

## Development Guidelines

### When Working on Data Scripts
- Always test on small ranges first (e.g., 10 blocks)
- Implement graceful error handling for RPC failures
- Use background execution for large data fetches
- Include real-time progress monitoring

### When Modifying Simulators
- Maintain compatibility with existing CSV data format
- Test with all 3 historical datasets
- Validate metrics calculations match web interface

### Web Interface
- Pure static files (no backend required)
- JavaScript simulation must match Python implementation
- Data symlinked from `../data/data_cache`

## Testing Approach
- Small block ranges for development/testing
- Full historical datasets for production analysis
- Cross-validate Python vs JavaScript implementations
- Verify data contiguity before analysis

## Dependencies
- Python environment in `./venv/`
- Key packages: pandas, requests, web3, matplotlib
- No additional installs needed for core functionality

## Common Tasks

### Fetch New Historical Data
```bash
source ./venv/bin/activate
python src/scripts/fetch_ethereum_blocks.py --start-block X --end-block Y
```

### Monitor Background Fetching
```bash
python src/scripts/monitor_fetch_progress.py
```

### Run Analysis
```bash
jupyter notebook analysis/notebooks/taiko_fee_analysis.ipynb
```

## Important Notes
- **Only post-EIP-1559 data** (Aug 5, 2021+) for scientific validity
- **Contiguous block ranges** required for accurate analysis
- **Multiple RPC endpoints** essential for reliability
- **Static web deployment** to GitHub Pages via symlinks