# Taiko Fee Mechanism Analysis

Comprehensive analysis and simulation framework for Taiko L2's proposed fee mechanism design.

## Overview

This project implements a detailed simulation and analysis framework to evaluate Taiko's proposed fee mechanism:

```
estimated_fee = basefee_L2 + μ × C_L1 + ν × D/H
```

Where:
- `μ ∈ [0,1]`: Weight on L1 cost estimate
- `ν ∈ [0,1]`: Weight on deficit correction
- `D = T - X`: Vault deficit (target minus current balance)
- `H`: Time horizon for deficit correction

## Key Research Questions

1. **μ=0 Viability**: Can we set μ=0 and rely only on deficit-based control?
2. **Parameter Optimization**: What are optimal values for (μ, ν, H)?
3. **L1 Dynamics Impact**: How do different L1 fee patterns affect the mechanism?
4. **Fee Caps**: How do caps affect performance and vault solvency?
5. **Extreme Scenarios**: How robust is the mechanism under stress conditions?

## Quick Start

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install numpy pandas matplotlib seaborn scipy jupyter

# Clone or download the project files
```

### Run Complete Analysis

```bash
# Full analysis (may take 10-15 minutes)
python main.py

# Quick analysis (faster, reduced parameters)
python main.py --quick

# Only generate visualizations from existing results
python main.py --visualize-only
```

### Interactive Exploration

```bash
# Create Jupyter notebook for interactive analysis
python main.py --create-notebook

# Then open the notebook
jupyter notebook taiko_fee_analysis.ipynb
```

## Project Structure

```
taiko-fee-analysis/
├── fee_mechanism_simulator.py      # Core simulation framework
├── mechanism_metrics.py            # Comprehensive metrics calculation
├── advanced_mechanisms.py          # Enhanced mechanism variants
├── run_analysis.py                 # Complete analysis suite
├── create_visualizations.py        # Publication-quality charts
├── main.py                         # Main execution script
├── taiko_fee_analysis.ipynb        # Interactive notebook (generated)
├── results/                        # Analysis outputs
│   ├── *.csv                       # Detailed metrics data
│   ├── *.png                       # Visualizations
│   └── summary_dashboard.png       # Key findings dashboard
└── README.md                       # This file
```

## Core Components

### 1. Simulation Framework (`fee_mechanism_simulator.py`)

- **TaikoFeeSimulator**: Main simulation class
- **L1 Dynamics Models**:
  - `GeometricBrownianMotion`: Baseline L1 fee dynamics
  - `RegimeSwitchingModel`: Low/medium/high fee regimes
  - `SpikeEventsModel`: Sudden fee spikes overlaid on baseline
- **FeeVault**: Manages vault balance and fee collection

### 2. Advanced Mechanisms (`advanced_mechanisms.py`)

- **MultiTimescaleSimulator**: Fast/slow deficit correction
- **DemandModel**: Fee elasticity with saturation effects
- **DynamicTargetManager**: Target adjustment based on L1 volatility
- **OptimalControlBenchmark**: Theoretical performance bounds

### 3. Metrics Framework (`mechanism_metrics.py`)

Comprehensive evaluation across four dimensions:

- **Vault Stability**: Balance variance, underfunding frequency
- **User Experience**: Fee volatility, tail percentiles
- **System Efficiency**: L1 tracking, response lags
- **Risk Management**: Insolvency probability, VaR

### 4. Analysis Suite (`run_analysis.py`)

- μ=0 viability analysis across L1 scenarios
- Parameter sweeps over (μ, ν, H) space
- Fee cap impact assessment
- Extreme scenario stress testing
- Optimal control comparison

## Key Results Summary

### μ=0 Viability

**Finding**: μ=0 (pure deficit-based control) is viable under most conditions but shows increased fee volatility during rapid L1 changes.

**Tradeoffs**:
- ✅ Simpler mechanism design
- ✅ Good long-term stability
- ❌ Higher short-term fee volatility
- ❌ Slower response to L1 spikes

### Optimal Parameters

**Recommended baseline**: μ=0.5, ν=0.3, H=144 steps (1 day)

**Key insights**:
- Higher ν improves vault stability but increases fee volatility
- Longer H smooths fees but slows deficit correction
- μ>0 provides better responsiveness to L1 changes

### Fee Caps

**Recommendation**: 5x cap (5× recent average fee) balances user protection with vault solvency.

**Findings**:
- No cap: Extreme fee spikes possible
- 2x cap: Risk of chronic underfunding
- 5x+ cap: Good balance of protection and solvency

### Extreme Scenario Robustness

The mechanism remains stable under:
- 10x L1 fee spikes lasting hours
- High volatility regimes (σ=1.0)
- Regime switching between fee levels

## Detailed Usage

### Single Simulation Example

```python
from fee_mechanism_simulator import *

# Set parameters
params = SimulationParams(
    mu=0.5,           # L1 cost weight
    nu=0.3,           # Deficit correction weight
    H=144,            # Correction horizon (1 day)
    target_balance=1000,
    total_steps=2000
)

# Choose L1 dynamics
l1_model = GeometricBrownianMotion(mu=0.0, sigma=0.3)

# Run simulation
simulator = TaikoFeeSimulator(params, l1_model)
df = simulator.run_simulation()

# Analyze results
from mechanism_metrics import MetricsCalculator
metrics_calc = MetricsCalculator(target_balance=1000)
metrics = metrics_calc.calculate_all_metrics(df)

print(f"Average fee: {metrics.avg_fee:.4f} ETH")
print(f"Fee volatility: {metrics.fee_cv:.3f}")
```

### Parameter Sweep

```python
from mechanism_metrics import ParameterSweepAnalyzer

analyzer = ParameterSweepAnalyzer(target_balance=1000)

param_ranges = {
    'mu': [0.0, 0.25, 0.5, 0.75, 1.0],
    'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
    'H': [72, 144, 288, 576]
}

results = analyzer.run_parameter_sweep(
    TaikoFeeSimulator, l1_model, param_ranges, base_params
)
```

### Advanced Mechanisms

```python
from advanced_mechanisms import *

# Multi-timescale control
advanced_params = AdvancedSimulationParams(
    mu=0.5, nu=0.3,
    dynamic_target=True,        # Adjust target based on volatility
    use_predictive_l1=True,     # Use L1 cost prediction
    fast_horizon=24,            # Fast response (2 hours)
    slow_horizon=288            # Slow response (1 day)
)

simulator = MultiTimescaleSimulator(advanced_params, l1_model)
```

## Extending the Framework

### Adding New L1 Dynamics

```python
class CustomL1Model(L1DynamicsModel):
    def generate_sequence(self, steps, initial_basefee=20e9):
        # Implement your L1 dynamics
        return basefee_sequence

    def get_name(self):
        return "Custom L1 Model"
```

### Custom Metrics

```python
def custom_metric(df):
    """Calculate custom performance metric."""
    # Your metric calculation
    return metric_value

# Add to metrics calculation
metrics_calc = MetricsCalculator(target_balance=1000)
# Extend with custom metrics
```

## Research Applications

This framework supports research in:

- **Mechanism Design**: Testing alternative fee formulas
- **Parameter Optimization**: Finding optimal control parameters
- **Risk Analysis**: Stress testing under extreme conditions
- **User Experience**: Analyzing fee predictability and fairness
- **Economic Modeling**: Understanding incentive alignment

## Technical Notes

### Time Scales
- Default time step = 12 seconds (L2 block time)
- H=144 ≈ 1 day at 12s blocks
- Simulations typically run 1000-5000 steps (3.3-16.7 hours)

### Computational Performance
- Single simulation: ~0.1-1 seconds
- Parameter sweep: ~30-300 seconds
- Full analysis suite: ~5-15 minutes

### Numerical Stability
- All calculations use 64-bit floating point
- Minimum fee/balance floors prevent numerical issues
- Exponential smoothing for stable L1 cost estimation

## Contributing

To extend or improve the analysis:

1. **Add scenarios**: Create new L1DynamicsModel subclasses
2. **Enhance metrics**: Add domain-specific evaluation metrics
3. **Improve mechanisms**: Implement advanced control algorithms
4. **Optimize performance**: Profile and optimize computation bottlenecks

## References

- [Taiko Protocol Documentation](https://taiko.xyz)
- [EIP-1559: Fee Market](https://eips.ethereum.org/EIPS/eip-1559)
- Control Theory for Blockchain Fee Markets
- Mechanism Design in Cryptocurrency Systems