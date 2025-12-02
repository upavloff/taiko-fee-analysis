# Taiko Fee Mechanism Analysis - Python Package

A comprehensive Python toolkit for analyzing and optimizing Taiko's EIP-1559 based fee mechanism.

## 📦 Installation

### Development Installation
```bash
# From the repository root
cd python
pip install -e .
```

### Production Installation
```bash
pip install taiko-fee-analysis
```

## 🚀 Quick Start

### Basic Simulation
```python
from taiko_fee import GeometricBrownianMotion
from taiko_fee.core.fee_mechanism_simulator import SimulationParams, TaikoFeeSimulator

# Create simulation parameters
params = SimulationParams(
    mu=0.0,           # L1 weight (pure deficit correction)
    nu=0.1,           # Deficit weight (gentle correction)
    H=36,             # Horizon in steps (72 seconds)
    target_balance=100 # Target vault balance in ETH
)

# Create L1 dynamics model
l1_model = GeometricBrownianMotion(mu=0.0, sigma=0.3)

# Run simulation
simulator = TaikoFeeSimulator(params, l1_model)
results = simulator.run_simulation(steps=1000)

print(f"Average fee: {results['avg_fee']:.2e} ETH")
print(f"Vault stability: {results['vault_stability']:.3f}")
```

### Historical Data Analysis
```python
from taiko_fee.data import RealDataFetcher

# Fetch real Ethereum data
fetcher = RealDataFetcher()
df = fetcher.get_basefee_data(
    start_date='2023-11-20',
    end_date='2023-11-23'
)

# Run simulation with real data
results = simulator.run_with_real_data(df)
```

## 🏗️ Package Structure

```
python/
├── taiko_fee/                 # Main package
│   ├── __init__.py           # Package initialization
│   ├── core/                 # Fee mechanism simulation engine
│   │   ├── fee_mechanism_simulator.py
│   │   └── improved_simulator.py
│   ├── data/                 # Data fetching and processing
│   │   ├── rpc_data_fetcher.py
│   │   └── real_data_fetcher.py
│   ├── analysis/             # Performance metrics and evaluation
│   │   └── mechanism_metrics.py
│   └── utils/                # Utility functions
│       └── vault_initialization_demo.py
├── tests/                    # Test suite
├── pyproject.toml           # Package configuration
└── README.md               # This file
```

## 🧪 Testing

```bash
# Run tests
cd python
pytest tests/

# Run with coverage
pytest --cov=taiko_fee tests/
```

## 📊 Key Components

### Core Simulation Engine
- **GeometricBrownianMotion**: L1 basefee modeling
- **TaikoFeeSimulator**: Main fee mechanism simulation
- **SimulationParams**: Parameter configuration

### Data Processing
- **RealDataFetcher**: Ethereum historical data retrieval
- **RPCDataFetcher**: Real-time RPC data access
- **DataCache**: Efficient data caching

### Analysis Tools
- **MetricsCalculator**: Performance evaluation
- **VaultAnalyzer**: Vault dynamics analysis
- **OptimizationFramework**: Parameter optimization

## ⚙️ Configuration

### Environment Variables
```bash
export ETHEREUM_RPC_URL="https://your-rpc-endpoint"
export DATA_CACHE_DIR="/path/to/cache"
```

### Python Requirements
- Python >= 3.8
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Matplotlib >= 3.5.0
- SciPy >= 1.7.0

## 📈 Advanced Usage

### Custom L1 Models
```python
from taiko_fee.core import L1DynamicsModel

class CustomL1Model(L1DynamicsModel):
    def generate_sequence(self, steps: int) -> np.ndarray:
        # Your custom L1 basefee generation logic
        return custom_basefees

simulator = TaikoFeeSimulator(params, CustomL1Model())
```

### Parameter Optimization
```python
from taiko_fee.analysis import ParameterOptimizer

optimizer = ParameterOptimizer()
optimal_params = optimizer.optimize(
    objective='minimize_fees',
    constraints={'stability': 0.8, 'safety': 0.9}
)
```

## 🔬 Research Integration

This package integrates seamlessly with Jupyter notebooks and research workflows:

```python
import matplotlib.pyplot as plt
from taiko_fee.analysis import plot_simulation_results

# Generate and plot results
results = simulator.run_simulation()
plot_simulation_results(results)
plt.show()
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

## 🆘 Support

- **Issues**: Report bugs at https://github.com/taikoxyz/taiko-fee-analysis/issues
- **Documentation**: https://docs.taiko.xyz/fee-analysis
- **Discord**: #research channel in Taiko Discord

---

*Part of the Taiko Fee Mechanism Analysis suite - advancing Ethereum Layer 2 research through rigorous analysis.*