# Documentation

This directory contains comprehensive documentation for the Taiko Fee Mechanism Analysis framework.

## 📚 Documentation Structure

- **API Reference**: Auto-generated API documentation
- **Research Notes**: Detailed analysis findings and methodology
- **Configuration Guide**: Setup and parameter tuning instructions
- **Deployment Guide**: Instructions for web interface deployment

## 🔬 Key Research Findings

### Parameter Optimization
- **Optimal μ (L1 weight)**: 0.3-0.5 for balanced L1 tracking
- **Optimal ν (deficit weight)**: 0.3 for stable vault management
- **Optimal H (horizon)**: 144 steps (~1 day) for smooth adjustments

### Performance Insights
- **Vault initialization is critical**: Empty vault causes 47x higher initial fees
- **μ=0 is viable**: Pure deficit-based mechanism works but has higher lag
- **Current Ethereum conditions**: ~0.1 gwei basefee (post-merge stability)

### Data Validation
- **EIP-1559 compliance**: Only post-August 5, 2021 data used
- **Historical periods validated**: May 2022 crash, June 2022 volatility, recent low fees
- **Pre-EIP-1559 data excluded**: DeFi summer data removed for methodological accuracy

## 📈 Methodology

### Simulation Framework
1. **Monte Carlo approach** with configurable L1 dynamics
2. **Historical data integration** with automatic caching
3. **Performance metrics** across multiple dimensions
4. **Parameter sensitivity analysis** for optimization

### Validation Approach
1. **Historical backtesting** against real Ethereum data
2. **Stress testing** with extreme volatility scenarios
3. **Comparative analysis** across different parameter sets
4. **Cross-validation** with alternative L1 models

## 🎯 Use Cases

### Research Applications
- **Academic studies** of Layer 2 fee mechanisms
- **Protocol optimization** for Taiko development team
- **Comparative analysis** with other L2 solutions
- **Policy evaluation** for different economic parameters

### Practical Applications
- **Parameter tuning** for production deployment
- **Risk assessment** under various market conditions
- **Performance monitoring** for live systems
- **Economic modeling** for protocol sustainability

---

For detailed technical documentation, see the individual files in this directory.