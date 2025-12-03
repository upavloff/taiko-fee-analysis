# Taiko Fee Mechanism Analysis Project

## 🏗️ CANONICAL MODULE ARCHITECTURE

**SINGLE SOURCE OF TRUTH**: All fee mechanism logic is now centralized in canonical modules.

**🔧 CRITICAL BUG FIXES APPLIED** (Dec 2024):
- **Gas Calculation**: Fixed 30x JavaScript overestimate and 10x Python underestimate → Now uses correct 20,000 gas/tx
- **Basefee Floor**: Removed artificial 1 gwei minimum → Uses realistic 0.001 gwei floor
- **Time Units**: Corrected documentation H=144 ≈ 4.8 minutes (was incorrectly stated as 1 day)
- **Consistency**: All implementations now use identical, scientifically accurate formulas

**Canonical Modules**:
- **Python**: `src/core/canonical_*.py` - Authoritative implementations for research
- **JavaScript**: `canonical-*.js` - Mirror implementations for GitHub Pages web interface

**Research-Validated Optimal Parameters**:
- Optimal: μ=0.0, ν=0.27, H=492 (NSGA-II multi-objective optimization result)
- Balanced: μ=0.0, ν=0.48, H=492 (Conservative approach)
- Crisis: μ=0.0, ν=0.88, H=120 (Aggressive deficit recovery)

## Project Context
This repo analyzes Taiko's EIP-1559 based fee mechanism using real Ethereum L1 data. Key focus areas:
- **Fee mechanism simulation** with realistic lumpy cash flow vault economics
- **Historical Ethereum data analysis** (post-EIP-1559 only: Aug 5, 2021+)
- **Interactive web interface** for parameter exploration
- **Scientific validation** of fee mechanism performance with 6-step batch cycles

## Architecture

**Canonical Module System**:
```
# PYTHON (Research & Analysis)
src/core/
├── canonical_fee_mechanism.py     # Fee calculation logic
├── canonical_metrics.py           # Performance metrics
└── canonical_optimization.py      # NSGA-II optimization

# JAVASCRIPT (Web Interface)
├── canonical-fee-mechanism.js     # Mirror of Python fee logic
├── canonical-metrics.js           # Mirror of Python metrics
├── canonical-optimization.js      # Mirror of Python optimization
└── test-canonical-consistency.html # Python-JS consistency tests
```

**Legacy Structure** (gradually being migrated):
```
src/
├── data/                  # RPC data fetching & caching
├── analysis/              # Performance metrics calculation (updated to use canonical)
├── scripts/               # Data fetching scripts (robust, resumable)
└── utils/                 # Helper functions (updated to use canonical)
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

### Canonical Module Usage

**Python Research/Analysis**:
```python
from core.canonical_fee_mechanism import create_default_calculator
from core.canonical_metrics import calculate_basic_metrics

calculator = create_default_calculator()  # Uses optimal parameters
fee = calculator.calculate_estimated_fee(l1_cost, vault_deficit)
```

**JavaScript Web Interface**:
```javascript
import { createDefaultCalculator } from './canonical-fee-mechanism.js';
import { calculateBasicMetrics } from './canonical-metrics.js';

const calculator = createDefaultCalculator();  // Uses optimal parameters
const fee = calculator.calculateEstimatedFee(l1Cost, vaultDeficit);
```

**Legacy Classes** (being phased out):
- `ImprovedTaikoFeeSimulator` → Use `CanonicalTaikoFeeCalculator`
- `ImprovedSimulationParams` → Use `FeeParameters`
- `MetricsCalculator` → Use `CanonicalMetricsCalculator`

### Key Parameters
- `μ (mu)`: L1 weight [0.0-1.0]
- `ν (nu)`: Deficit weight [0.1-0.9]
- `H`: Prediction horizon (steps, e.g., 144 ≈ 4.8 minutes, 492 ≈ 16.4 minutes)

## Development Guidelines

### When Working on Data Scripts
- Always test on small ranges first (e.g., 10 blocks)
- Implement graceful error handling for RPC failures
- Use background execution for large data fetches
- Include real-time progress monitoring

### When Modifying Fee Mechanism Logic
- **ALWAYS** update both Python AND JavaScript canonical modules
- Run consistency tests: `test-canonical-consistency.html`
- Test with all 3 historical datasets
- Maintain exact formula implementations: `F_estimated(t) = max(μ × C_L1(t) + ν × D(t)/H, F_min)`

### 🚫 MOCK DATA PROHIBITION POLICY

**CRITICAL**: This project maintains strict scientific accuracy standards. Mock data usage is **STRICTLY PROHIBITED** except where explicitly documented and warned.

#### Prohibited Practices:
- **Hardcoded fee floors** (e.g., 1.5 gwei injection) - Use configurable parameters with defaults ≤ 0.001 gwei
- **Arbitrary constants** (e.g., Q̄ = 690,000) - Use real historical data or configurable parameters
- **Random number generation** in fee calculations - Must reflect real market conditions
- **Silent fallbacks** to mock data - All fallbacks must issue clear warnings
- **Mock optimization results** - Use real historical data evaluation only

#### Required Practices:
- **Real data enforcement**: Use `src/utils/mock_data_detector.py` to audit codebase
- **Runtime warnings**: All canonical modules detect and warn about mock data usage
- **Explicit fallbacks**: When real data fails, fail with clear error messages or issue prominent warnings
- **Data provenance**: Track and validate data sources throughout calculation pipeline
- **Configurable parameters**: Replace hardcoded values with explicit, documented parameters

#### Detection and Enforcement:
```bash
# Run mock data detection audit
python3 src/utils/mock_data_detector.py

# Validate canonical modules
python3 -m http.server 8001 → visit test-canonical-consistency.html
```

**Violation Consequences**: Code using undisclosed mock data violates scientific integrity and produces unreliable fee mechanism analysis. All mock data usage must be eliminated or clearly documented with prominent warnings.

### Web Interface (GitHub Pages Compatible)
- Pure static files (no backend required)
- Uses JavaScript canonical modules for consistency
- Test page at `test-canonical-consistency.html` verifies Python-JS equivalence
- All existing JS files should migrate to import from canonical modules

## Testing Approach

### Canonical Module Testing
- **Consistency Tests**: `python3 -m http.server 8000` → visit `test-canonical-consistency.html`
- **Python Module Tests**: `python src/utils/updated_quick_validation.py`
- **Cross-validation**: Verify Python and JavaScript produce identical results

### Data Analysis Testing
- Small block ranges for development/testing
- Full historical datasets for production analysis
- Verify data contiguity before analysis

## Dependencies
- Python environment in `./venv/`
- Key packages: pandas, requests, web3, matplotlib
- No additional installs needed for core functionality

## Common Tasks

### Using Canonical Modules

**Quick Validation (Python)**:
```bash
source ./venv/bin/activate
python3 src/utils/updated_quick_validation.py
```

**Comprehensive Analysis (Python)**:
```bash
source ./venv/bin/activate
python3 src/analysis/updated_alpha_validation_demo.py
```

**Consistency Testing (Web)**:
```bash
python3 -m http.server 8001
# Visit: http://localhost:8001/test-canonical-consistency.html
```

**Bug Fix Validation (Web)**:
```bash
python3 -m http.server 8001
# Visit: http://localhost:8001/test-gas-calculation-fix.html
```

**Demo Usage (JavaScript)**:
```bash
python3 -m http.server 8001
# Visit: http://localhost:8001/canonical-demo.js (in console)
```

### Data Fetching

**Fetch New Historical Data**:
```bash
source ./venv/bin/activate
python3 src/scripts/fetch_ethereum_blocks.py --start-block X --end-block Y
```

**Monitor Background Fetching**:
```bash
python3 src/scripts/monitor_fetch_progress.py
```

## Important Notes

### Canonical Module Consistency
- **ALWAYS** maintain identical logic between Python and JavaScript canonical modules
- **Test consistency** after any changes using `test-canonical-consistency.html`
- **Use canonical imports** in all new code - avoid duplicating fee mechanism logic

### Data Analysis
- **Only post-EIP-1559 data** (Aug 5, 2021+) for scientific validity
- **Contiguous block ranges** required for accurate analysis
- **Multiple RPC endpoints** essential for reliability

### Deployment
- **GitHub Pages compatible** - no server required
- **Static web deployment** using `python3 -m http.server`
- All canonical modules work in browser without build step
- Use python3 and pip3