# Taiko Fee Analysis

A professional-grade analysis framework for Taiko's fee mechanism design and optimization.

## 🏗️ Architecture Overview

This repository follows enterprise-level architecture patterns with clear separation of concerns:

```
taiko-fee-analysis/
├── src/                          # Source code (modular architecture)
│   ├── core/                     # Core simulation components
│   │   ├── fee_mechanism_simulator.py    # Base simulator engine
│   │   └── improved_simulator.py         # Enhanced simulator with optimizations
│   ├── data/                     # Data fetching and caching
│   │   ├── rpc_data_fetcher.py          # RPC-based Ethereum data fetching
│   │   └── real_data_fetcher.py         # Legacy data integration
│   ├── analysis/                 # Analytics and metrics
│   │   └── mechanism_metrics.py         # Performance metrics calculation
│   └── utils/                    # Utility functions
│       └── vault_initialization_demo.py # Demo utilities
├── notebooks/                    # Jupyter analysis notebooks
│   └── taiko_fee_analysis.ipynb         # Main research notebook
├── data_cache/                   # Cached RPC data (auto-created)
├── tests/                        # Unit tests (future)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

```bash
# Setup environment
pip install -r requirements.txt

# Run main analysis
jupyter notebook notebooks/taiko_fee_analysis.ipynb
```

## 📋 Component Descriptions

### Core Components (`src/core/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `fee_mechanism_simulator.py` | Base simulation engine | `TaikoFeeSimulator`, `SimulationParams`, `FeeVault` |
| `improved_simulator.py` | Enhanced simulator with optimizations | `ImprovedTaikoFeeSimulator`, `ImprovedSimulationParams` |

**Key Features:**
- Monte Carlo simulation framework
- Multiple L1 dynamics models (GBM, real data)
- Configurable fee mechanism parameters (μ, ν, H)
- Proper vault initialization strategies

### Data Layer (`src/data/`)

| File | Purpose | Key Features |
|------|---------|-------------|
| `rpc_data_fetcher.py` | **Primary data source** - RPC-based Ethereum basefee fetching | CSV caching, multiple providers, rate limiting |
| `real_data_fetcher.py` | Legacy API-based data integration | Historical data analysis |

**RPC Data Fetcher Features:**
- ✅ **Automatic CSV caching** - Fetches once, caches forever
- ✅ **Multiple RPC providers** - Public, Infura, Alchemy support
- ✅ **Intelligent rate limiting** - Respects provider limits
- ✅ **Error handling** - Graceful fallbacks

### Analysis Engine (`src/analysis/`)

| File | Purpose | Key Metrics |
|------|---------|-------------|
| `mechanism_metrics.py` | Performance evaluation framework | Fee stability, vault management, L1 tracking accuracy |

**Metrics Calculated:**
- Fee volatility (coefficient of variation)
- Vault underfunding percentage
- L1 cost tracking error
- Response lag to L1 changes

### Utilities (`src/utils/`)

| File | Purpose |
|------|---------|
| `vault_initialization_demo.py` | Demonstration utilities for proper vault setup |

## 🔬 Research Framework

### Core Research Questions
1. **μ=0 Viability**: Can Taiko use only deficit correction without L1 cost tracking?
2. **Parameter Optimization**: Optimal values for (μ, ν, H) parameters
3. **Real Data Performance**: Mechanism behavior under actual Ethereum conditions
4. **Vault Initialization**: Impact of starting vault balance on performance

### Key Findings
- ✅ **Vault initialization is critical** - Empty vault creates extreme initial fees
- ✅ **μ=0 is viable** but has slower L1 response (higher lag)
- ✅ **Optimal parameters**: μ=0.3-0.5, ν=0.3, H=144 blocks
- ✅ **Current Ethereum**: ~0.1 gwei basefee (very stable post-merge)

## 💻 Usage Examples

### Basic Simulation
```python
from src.core import ImprovedTaikoFeeSimulator, ImprovedSimulationParams, GeometricBrownianMotion

# Create parameters with proper vault initialization
params = ImprovedSimulationParams(
    mu=0.5, nu=0.3, H=144,
    target_balance=1000,
    vault_initialization_mode="target",  # Critical!
    total_steps=500
)

# Run simulation
l1_model = GeometricBrownianMotion(mu=0.0, sigma=0.3)
simulator = ImprovedTaikoFeeSimulator(params, l1_model)
results = simulator.run_simulation()
```

### Real Data Analysis with Caching
```python
from src.data import ImprovedRealDataIntegrator

# Fetches once, caches to CSV automatically
integrator = ImprovedRealDataIntegrator()
df = integrator.get_real_basefee_data(
    '2023-11-20', '2023-11-23',
    provider='ethereum_public',
    use_cache=True  # Default
)
```

### Performance Metrics
```python
from src.analysis import MetricsCalculator

calc = MetricsCalculator(target_balance=1000)
metrics = calc.calculate_all_metrics(results)

print(f"Average fee: {metrics.avg_fee:.2e} ETH")
print(f"Fee stability (CV): {metrics.fee_cv:.3f}")
print(f"Time underfunded: {metrics.time_underfunded_pct:.1f}%")
```

## 🗄️ Data Management

### Automatic Caching System
The RPC data fetcher implements intelligent caching:

```python
# First call: fetches from RPC and caches
df1 = integrator.get_real_basefee_data('2023-11-20', '2023-11-23')

# Second call: loads instantly from cache
df2 = integrator.get_real_basefee_data('2023-11-20', '2023-11-23')
```

**Cache Location:** `data_cache/basefee_{start_date}_{end_date}_{provider}.csv`

### Multiple RPC Providers
- **ethereum_public**: Public RPC (free, rate limited)
- **cloudflare**: Cloudflare Ethereum Gateway
- **infura**: Infura (requires project ID)
- **alchemy**: Alchemy (requires API key)

## 🧪 Development Best Practices

### Code Organization
- **Separation of Concerns**: Clear module boundaries
- **Dependency Injection**: Configurable components
- **Error Handling**: Graceful degradation
- **Caching Strategy**: Minimize external API calls
- **Type Hints**: Enhanced code clarity

### Import Structure
```python
# Clean imports using package structure
from src.core import *                    # All simulation components
from src.analysis import MetricsCalculator # Specific imports
from src.data import ImprovedRealDataIntegrator
```

## 🔧 Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/taiko_fee_analysis.ipynb
```

## 📊 Performance Benchmarks

### Simulation Performance
- **Standard run** (500 steps): ~2-3 seconds
- **Parameter sweep** (5x5 grid): ~45-60 seconds
- **Real data integration**: First fetch ~30-60s, cached <1s

### Memory Usage
- **Base simulation**: ~10-20 MB
- **Large dataset** (1000+ blocks): ~50-100 MB
- **Multiple scenarios**: Scales linearly

## 🔮 Future Enhancements

### Planned Features
- [ ] **Unit test suite** (`tests/` directory)
- [ ] **CLI interface** for automated analysis
- [ ] **Docker containerization** for reproducibility
- [ ] **Enhanced visualization** with Plotly
- [ ] **Parameter optimization** with scipy

### Research Extensions
- [ ] **Arbitrum mechanism comparison**
- [ ] **MEV impact analysis**
- [ ] **Multi-chain data integration**
- [ ] **Production deployment guide**

## 📈 Research Status

- ✅ **Core Framework**: Production-ready simulation engine
- ✅ **Data Infrastructure**: Robust RPC integration with caching
- ✅ **Analysis Tools**: Comprehensive metrics calculation
- ✅ **Research Findings**: Key insights documented
- 🔄 **Optimization**: Parameter tuning in progress
- 📋 **Documentation**: Architecture fully documented

## 🤝 Contributing

This is a research-grade framework. Contributions should focus on:

1. **Code Quality**: Following established architecture patterns
2. **Performance**: Optimizing simulation speed and memory usage
3. **Analysis**: New metrics or evaluation approaches
4. **Data**: Additional RPC providers or data sources
5. **Testing**: Unit tests and validation frameworks

## 📄 License

MIT License - See LICENSE file for details.