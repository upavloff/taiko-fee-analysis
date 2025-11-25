# 🚀 Quick Start Guide

## Tested Setup Instructions

### 1. Validate Your Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Run validation test (THIS MUST PASS)
python test_setup.py
```

You should see:
```
✅ ALL TESTS PASSED!
🎯 Your setup is working correctly.
```

### 2. Run Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook notebooks/taiko_fee_analysis.ipynb

# Make sure to select "Taiko Analysis" kernel
# Kernel → Change Kernel → "Taiko Analysis"
```

### 3. Run the First Cell
The first cell should output:
```
✅ ALL MODULES IMPORTED SUCCESSFULLY!
✅ Taiko Fee Analysis ready to use
✅ RPC data integration with CSV caching available
✅ Enhanced vault initialization enabled

🎯 You can now run all analysis cells below...
```

## If Something Goes Wrong

### Problem: `test_setup.py` fails
**Solution:** Check your virtual environment
```bash
# Make sure you're in the right directory
cd /path/to/taiko-fee-analysis

# Activate venv
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Re-register kernel
python -m ipykernel install --user --name taiko-analysis --display-name "Taiko Analysis"
```

### Problem: Jupyter imports fail
**Solution:**
1. **Restart Jupyter kernel** (Kernel → Restart)
2. **Select correct kernel** (Kernel → Change Kernel → "Taiko Analysis")
3. **Run first cell again**

### Problem: "Module not found" errors
**Solution:** The `test_setup.py` validates this. If test passes but notebook fails, ensure:
- You're using the "Taiko Analysis" kernel
- You've restarted the kernel after changing it

## Architecture Quick Reference

```
src/
├── core/           # Simulation engines
├── data/           # RPC data fetching with CSV caching
├── analysis/       # Performance metrics
└── utils/          # Utility functions

notebooks/          # Jupyter analysis
data_cache/         # Cached RPC data (auto-created)
```

## Usage Examples (After Imports Work)

### Basic Simulation
```python
params = ImprovedSimulationParams(
    mu=0.5, nu=0.3, H=144,
    target_balance=1000,
    vault_initialization_mode="target",
    total_steps=200
)

l1_model = GeometricBrownianMotion(mu=0.0, sigma=0.3)
simulator = ImprovedTaikoFeeSimulator(params, l1_model)
df = simulator.run_simulation()
```

### Real Data with Caching
```python
integrator = ImprovedRealDataIntegrator()
df = integrator.get_real_basefee_data('2023-11-20', '2023-11-23')  # Cached automatically!
```

### Metrics Analysis
```python
calc = MetricsCalculator(target_balance=1000)
metrics = calc.calculate_all_metrics(df)
print(f"Average fee: {metrics.avg_fee:.2e} ETH")
```

---

## ✅ Success Criteria
- [ ] `python test_setup.py` passes all tests
- [ ] Jupyter notebook first cell imports without errors
- [ ] You see "✅ ALL MODULES IMPORTED SUCCESSFULLY!"

Once these work, you're ready for full analysis!