# Taiko Fee Mechanism Analysis

A comprehensive scientific analysis framework for the Taiko protocol's fee mechanism, implementing EIP-1559 based fee estimation with vault deficit correction.

## 🎯 Overview

This repository contains a complete analysis of Taiko's fee mechanism, including:

- **Mathematical modeling** of the fee estimation formula
- **Real-time simulation** with historical Ethereum L1 data
- **Interactive web interface** for parameter exploration
- **Comprehensive metrics** for mechanism evaluation

## 📊 Fee Mechanism Formula

The Taiko fee mechanism implements a dual-component pricing model:

$$F_E(t) = \max\left(\mu \times C_{L1}(t) + \nu \times \frac{D(t)}{H}, F_{\text{min}}\right)$$

**Component Definitions:**
- $F_E(t)$: Estimated fee at time $t$ (in ETH)
- $\mu \in [0,1]$: L1 weight parameter controlling L1 cost influence
- $\nu \in [0,1]$: Deficit weight parameter controlling vault correction strength
- $C_{L1}(t)$: L1 cost per transaction at time $t$
- $D(t)$: Vault deficit at time $t$ (target balance - current balance)
- $H$: Prediction horizon (number of steps, e.g., 144 = 288s ≈ 4.8 min)
- $F_{\text{min}}$: Minimum fee threshold (1e-8 ETH)

**L1 Cost Calculation:**

$$C_{L1}(t) = \frac{\text{BaseFee}_{L1}(t) \times \text{Gas}_{\text{per tx}}}{10^{18}}$$

Where:
$$\text{Gas}_{\text{per tx}} = \max\left(\frac{200{,}000}{\text{Expected Tx Volume}}, 2{,}000\right)$$

This implements economies of scale: higher transaction volume reduces per-transaction L1 cost due to batch efficiency, with a 2,000 gas minimum for overhead.

## 🚀 Quick Start

### Web Interface (Recommended)
```bash
open index.html  # Open in browser - fully static!
```

### Python Analysis
```bash
# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run Jupyter analysis
jupyter notebook analysis/notebooks/taiko_fee_analysis.ipynb
```

## 📁 Repository Structure

```
taiko-fee-analysis/
├── src/                           # Core simulation engine
│   ├── core/                     # Fee mechanism simulation
│   │   ├── fee_mechanism_simulator.py
│   │   └── improved_simulator.py
│   ├── data/                     # Data fetching & caching
│   │   ├── rpc_data_fetcher.py
│   │   └── real_data_fetcher.py
│   ├── analysis/                 # Performance metrics
│   │   └── mechanism_metrics.py
│   └── utils/                    # Utility functions
│       └── vault_initialization_demo.py
├── web/                          # Interactive web interface (static)
│   ├── index.html               # Main application
│   ├── simulator.js             # JavaScript simulator
│   ├── charts.js               # Visualization engine
│   ├── styles.css              # UI styling
│   └── data_cache/             # → ../data/data_cache (symlink)
├── analysis/                     # Scientific analysis
│   └── notebooks/               # Jupyter notebooks
│       ├── taiko_fee_analysis.ipynb
│       └── updated_taiko_analysis.ipynb
├── data/                        # Historical L1 data
│   └── data_cache/             # Cached basefee datasets
│       ├── recent_low_fees_3hours.csv       # Nov 2025 low fee period (0.055-0.092 gwei)
│       ├── may_crash_basefee_data.csv       # May 2022 UST/Luna crash (53-533 gwei)
│       └── real_july_2022_spike_data.csv    # July 2022 market volatility (7-88 gwei)
├── docs/                        # Documentation
│   └── README.md               # Research findings & methodology
├── tests/                       # Test suite (future)
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🔬 Features

### Web Interface
- **Real-time parameter adjustment** with instant feedback
- **Multiple data sources**: Historical Ethereum data & simulated scenarios
- **Interactive visualizations**: Fee trends, vault dynamics, L1 tracking
- **Preset configurations**: Optimized parameter sets for different use cases
- **Performance metrics**: Comprehensive analysis dashboard

### Python Framework
- **Modular architecture** with clear separation of concerns
- **Data caching** for efficient historical data retrieval
- **Flexible simulation** with customizable parameters
- **Extensive metrics** for mechanism evaluation

## 📈 Analysis Capabilities

### Historical Data Analysis
- **May 2022 Crypto Crash**: Real UST/Luna collapse data (53-533 gwei)
- **July 2022 Market Volatility**: Real Ethereum network spikes (7-88 gwei)
- **Recent Low Fee Period**: Nov 2025 market conditions (0.055-0.092 gwei)

### Simulation Scenarios
- **Geometric Brownian Motion** for realistic L1 basefee modeling
- **Volatility spikes** with configurable timing and intensity
- **Various vault initialization** states for comprehensive testing

### Performance Metrics
- **Average Fee**: Mean transaction cost over simulation period
- **Fee Variability (CV)**: Coefficient of variation for stability analysis
- **Time Underfunded**: Percentage of time below vault target
- **L1 Tracking Error**: Deviation from actual L1 costs

## 🎛️ Key Parameters

| Parameter | Range | Description | Impact |
|-----------|-------|-------------|---------|
| μ (mu) | 0.0-1.0 | L1 weight | Higher = more L1 cost tracking |
| ν (nu) | 0.1-0.9 | Deficit weight | Higher = faster vault correction |
| H | 24-576 | Horizon (steps) | Longer = smoother adjustments |

## 📖 Scientific Validation

All analysis uses **post-EIP-1559 data only** (August 5, 2021+) to ensure compatibility with Ethereum's current base fee mechanism. Pre-EIP-1559 gas auction data is excluded for methodological accuracy.

## 🌐 Live Demo

The web interface is fully static and can be deployed to any hosting platform:
- **GitHub Pages** (for public repos)
- **Netlify/Vercel** (supports private repos)
- **Local hosting** (open `web/index.html`)

## 💻 Usage Examples

### Basic Simulation
```python
from src.core import ImprovedTaikoFeeSimulator, ImprovedSimulationParams, GeometricBrownianMotion

# Create parameters with proper vault initialization
params = ImprovedSimulationParams(
    mu=0.5, nu=0.3, H=144,
    target_balance=100,
    vault_initialization_mode="target",
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
    use_cache=True
)
```

### Performance Metrics
```python
from src.analysis import MetricsCalculator

calc = MetricsCalculator(target_balance=100)
metrics = calc.calculate_all_metrics(results)

print(f"Average fee: {metrics.avg_fee:.2e} ETH")
print(f"Fee stability (CV): {metrics.fee_cv:.3f}")
print(f"Time underfunded: {metrics.time_underfunded_pct:.1f}%")
```

## 🔧 Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name taiko-analysis --display-name "Taiko Analysis"

# Launch analysis
jupyter notebook analysis/notebooks/taiko_fee_analysis.ipynb
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see our contributing guidelines for details.

---

*Built for the Nethermind research team - Advancing Ethereum's Layer 2 ecosystem through rigorous analysis.*
