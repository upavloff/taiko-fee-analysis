# Taiko Fee Mechanism Research Workspace

This directory contains the research components of the Taiko fee mechanism analysis project, including notebooks, experiments, papers, and results.

## 📁 Structure

```
research/
├── notebooks/           # Jupyter notebooks for analysis
├── experiments/         # Ad-hoc research scripts
├── papers/             # Research documents and findings
└── results/            # Generated outputs and data
```

## 📓 Notebooks

### Core Analysis Notebooks
- **`01_fee_mechanism_analysis.ipynb`**: Primary fee mechanism analysis
- **`02_optimal_parameters.ipynb`**: Parameter optimization research
- **`03_historical_validation.ipynb`**: Validation against real data

### Development Notebooks
- **`dev_*.ipynb`**: Development and debugging notebooks
- **`test_*.ipynb`**: Test and validation notebooks

## 🧪 Experiments

Ad-hoc research scripts for specific investigations:

```bash
# Run parameter sweep experiment
python experiments/parameter_sweep.py --scenario crisis

# Stress test specific configurations
python experiments/stress_test.py --config optimal

# Generate adversarial scenarios
python experiments/adversarial_scenarios.py
```

## 📄 Papers

Research documents and findings:
- **`taiko_fee_mechanism.md`**: Main research paper
- **`optimization_framework.md`**: Optimization methodology
- **`empirical_validation.md`**: Real-world validation results

## 📊 Results

Generated research outputs:
- **`optimal_parameters.json`**: Final parameter recommendations
- **`validation_results.csv`**: Cross-validation results
- **`figures/`**: Generated plots and visualizations

## 🔧 Setup

### Python Environment
```bash
# From repository root
cd python
pip install -e .

# Install Jupyter dependencies
pip install jupyter matplotlib seaborn plotly

# Launch Jupyter
jupyter notebook research/notebooks/
```

### Data Access
Notebooks expect data in `../data_cache/`:
```
../data_cache/
├── recent_low_fees_3hours.csv
├── luna_crash_true_peak_contiguous.csv
└── real_july_2022_spike_data.csv
```

## 📈 Research Workflow

### 1. Exploratory Analysis
```bash
jupyter notebook research/notebooks/01_fee_mechanism_analysis.ipynb
```

### 2. Parameter Optimization
```bash
python research/experiments/comprehensive_optimization.py
```

### 3. Validation
```bash
jupyter notebook research/notebooks/03_historical_validation.ipynb
```

### 4. Documentation
Update findings in `research/papers/` directory.

## 🧮 Key Research Questions

### Primary Questions
1. **Optimal Parameters**: What are the best μ, ν, H values across scenarios?
2. **Robustness**: How do parameters perform under stress conditions?
3. **Trade-offs**: What are the UX vs safety vs efficiency trade-offs?

### Research Methodology
- **Multi-scenario optimization** across normal/spike/crash/crisis conditions
- **Historical validation** with real Ethereum data
- **Monte Carlo robustness** testing with synthetic scenarios

## 🔬 Research Standards

### Reproducibility
- All notebooks include random seeds
- Data preprocessing steps documented
- Environment requirements specified

### Scientific Rigor
- Cross-validation across multiple datasets
- Statistical significance testing
- Peer review of methodology and results

### Documentation
- Clear methodology explanations
- Assumptions explicitly stated
- Limitations acknowledged

## 📋 Research TODO

### Active Research
- [ ] Long-term parameter adaptation strategies
- [ ] Cross-L2 mechanism comparison
- [ ] MEV resistance analysis
- [ ] Governance parameter update mechanisms

### Validation Tasks
- [ ] Live network parameter monitoring
- [ ] A/B testing framework development
- [ ] Real-world stress testing

## 🤝 Research Collaboration

### Contributing Research
1. Fork repository
2. Create research branch: `research/your-investigation`
3. Add notebooks with clear methodology
4. Document findings in `papers/`
5. Submit pull request with research summary

### Review Process
- Research findings undergo peer review
- Methodology validation required
- Reproducibility testing performed
- Integration with main findings

## 📚 References

- **EIP-1559**: Ethereum fee market mechanism
- **Taiko Protocol**: Layer 2 architecture documentation
- **Research Papers**: Previous L2 fee mechanism studies

---

*Advancing Ethereum Layer 2 research through rigorous scientific analysis of fee mechanisms.*