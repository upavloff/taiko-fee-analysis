# Unit Conventions & Anti-Bug Guidelines

## 🚨 Critical: Unit Mismatch Prevention

**THE BUG**: Fee mechanism was passing ETH amounts to functions expecting WEI, causing fees to be 10^18 times too small (effectively zero).

**THE FIX**: Strict unit conventions + validation + type safety.

---

## 📏 **Standard Unit Conventions**

### **Always Use Wei for Internal Calculations**
```python
# ✅ CORRECT: Internal calculations in wei
l1_cost_wei = 100_000_000_000_000  # 0.0001 ETH
fee_wei = calculate_fee(l1_cost_wei, deficit_wei)

# ❌ WRONG: Mixing units
l1_cost_eth = 0.0001
fee_wei = calculate_fee(l1_cost_eth, deficit_wei)  # BUG!
```

### **Display Units by Context**
- **Internal**: Always wei
- **User Display**: gwei for fees, ETH for large amounts
- **Configuration**: Explicit unit suffix (`fee_min_gwei`, `balance_eth`)

### **Parameter Naming Convention**
```python
# ✅ GOOD: Unit is clear from name
def calculate_fee(l1_cost_wei: int, deficit_wei: int) -> int:
    pass

def display_fee(fee_wei: int) -> str:
    return f"{fee_wei / 1e9:.6f} gwei"

# ❌ BAD: Unit unclear
def calculate_fee(l1_cost: float, deficit: float) -> float:
    pass  # What units? ETH? Wei? Gwei?
```

---

## 🔒 **Mandatory Safety Measures**

### **1. Use Type-Safe Unit System**
```python
from core.units import Wei, Gwei, WeiPerGas, WeiPerBatch
from core.units import gwei_to_wei, wei_to_gwei

# Type system prevents unit confusion
basefee_wei: Wei = gwei_to_wei(Gwei(50.0))
batch_cost: WeiPerBatch = l1_basefee_to_batch_cost(basefee_wei, 2000)
fee_rate: WeiPerGas = batch_cost.to_wei_per_gas(690_000)
```

### **2. Add Validation Decorators**
```python
from core.validation import validate_fee_function, validate_simulation_inputs

@validate_fee_function
def calculate_fee(l1_cost_wei: int, deficit_wei: int) -> int:
    # Decorator validates inputs/outputs automatically
    return fee_calculation_logic(l1_cost_wei, deficit_wei)

@validate_simulation_inputs
def run_simulation(l1_costs_wei: np.ndarray) -> dict:
    # Decorator warns if inputs look like wrong units
    return simulation_logic(l1_costs_wei)
```

### **3. Golden Value Tests**
```python
def test_fee_calculation_golden():
    """Test with pre-calculated expected values"""
    # Specific scenario with known result
    l1_basefee_gwei = 50.0
    expected_fee_gwei = 0.072463768  # Manually calculated

    # Calculate actual result
    basefee_wei = gwei_to_wei(l1_basefee_gwei)
    batch_cost = l1_basefee_to_batch_cost(basefee_wei, 2000)
    fee_wei = controller.calculate_fee(batch_cost.value, 0)
    fee_gwei = fee_wei / 1e9

    # Any change in behavior will fail this test
    assert abs(fee_gwei - expected_fee_gwei) < 1e-6
```

### **4. Runtime Assertions**
```python
from core.validation import assert_reasonable_fee, assert_no_unit_mismatch

def process_fee_result(fee_wei: int):
    fee_gwei = fee_wei / 1e9

    # Catch unreasonable fees immediately
    assert_reasonable_fee(fee_gwei, "final fee calculation")

    # Catch unit mismatches
    assert_no_unit_mismatch(fee_gwei, (0.001, 1000), "gwei", "fee output")
```

---

## 📋 **Development Checklist**

### **Before Writing Fee Code**
- [ ] Define input/output units explicitly in function signatures
- [ ] Add unit suffixes to variable names (`_wei`, `_gwei`, `_eth`)
- [ ] Plan conversion points between units
- [ ] Add validation decorators to functions

### **Before Committing**
- [ ] Run unit tests (especially `test_unit_safety.py`)
- [ ] Run golden value tests (`test_integration_golden.py`)
- [ ] Check no fees are suspiciously close to zero
- [ ] Verify realistic fee ranges in sample scenarios

### **Before Production**
- [ ] Test with real historical L1 data
- [ ] Validate fee mechanism produces reasonable ranges
- [ ] Check optimization results show non-zero μ values
- [ ] Verify stakeholder profiles produce different parameters

---

## 🔧 **Common Conversion Patterns**

### **L1 Basefee → Batch Cost → Fee Rate**
```python
# Step 1: L1 basefee (gwei) → wei
basefee_wei = gwei_to_wei(Gwei(50.0))

# Step 2: Wei per gas → wei per batch
total_gas = gas_per_tx * txs_per_batch
batch_cost_wei = basefee_wei * total_gas

# Step 3: Wei per batch → wei per gas (fee rate)
fee_component_wei_per_gas = batch_cost_wei // q_bar
```

### **Fee Display**
```python
# Internal calculation in wei
fee_wei = calculate_fee(l1_cost_wei, deficit_wei)

# Display conversion
fee_gwei = fee_wei / 1e9
print(f"Fee: {fee_gwei:.6f} gwei")

# Type-safe version
fee_rate = WeiPerGas(fee_wei)
print(f"Fee: {fee_rate}")  # Auto-formats as gwei
```

---

## 🚨 **Red Flags to Watch For**

### **Suspicious Values**
```python
# 🚨 RED FLAG: Fees near zero
if fee_gwei < 0.001:
    # Probably unit mismatch!

# 🚨 RED FLAG: μ=0 in optimization
if optimal_mu < 0.01:
    # L1 tracking broken, investigate units

# 🚨 RED FLAG: Perfect scores
if ux_score > 0.999:
    # Probably artificial (zero fees)
```

### **Code Smells**
```python
# 🚨 SMELL: Unexplained division by 1e18
result = value / 1e18  # Why? Document the conversion!

# 🚨 SMELL: Magic numbers
fee = cost * 0.000001449  # What is this factor?

# 🚨 SMELL: Ambiguous parameters
def calc_fee(cost, deficit):  # What units?
```

---

## 📚 **References & Resources**

### **Key Files**
- `core/units.py` - Type-safe unit system
- `core/validation.py` - Runtime validation
- `tests/test_unit_safety.py` - Unit conversion tests
- `tests/test_integration_golden.py` - Golden value tests

### **Debugging Commands**
```bash
# Run unit safety tests
pytest tests/test_unit_safety.py -v

# Run golden value tests
pytest tests/test_integration_golden.py -v

# Check for zero fees in optimization
python diagnose_unit_mismatch.py
```

### **Emergency Unit Check**
```python
# Quick test for unit mismatch
controller = FeeController(mu=0.5, nu=0.2, horizon_h=144, q_bar=690000)

# Test reasonable L1 cost
basefee_wei = 50e9 * 2000  # 50 gwei * 2000 gas = 1e14 wei
fee_wei = controller.calculate_fee(basefee_wei, 0)
fee_gwei = fee_wei / 1e9

print(f"Fee: {fee_gwei:.6f} gwei")
# Should be ~0.072 gwei, NOT ~0.000 gwei!
```

---

## ✅ **Success Metrics**

After implementing these measures:
- ✅ No fees are exactly 0.000000 gwei
- ✅ Optimization finds μ > 0 (L1 tracking viable)
- ✅ Stakeholder profiles show different optimal parameters
- ✅ Fees scale proportionally with L1 costs
- ✅ Unit tests catch conversion errors immediately
- ✅ Golden tests detect any behavioral changes

**Remember: This bug cost us weeks of investigation. These measures ensure it never happens again.** 🛡️