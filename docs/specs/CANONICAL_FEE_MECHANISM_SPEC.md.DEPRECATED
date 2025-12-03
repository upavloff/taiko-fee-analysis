# Canonical Taiko Fee Mechanism Specification

## 🎯 Purpose

This document serves as the **single source of truth** for the Taiko fee mechanism implementation, ensuring consistency across all Python and JavaScript implementations, documentation, and research.

## 📐 Mathematical Formula

### Core Fee Calculation
```
F(t) = μ × C_L1(t) + ν × D(t) / H
```

**Where:**
- `F(t)`: Estimated fee at time t (in ETH)
- `μ ∈ [0,1]`: L1 weight parameter (dimensionless)
- `ν ∈ [0,1]`: Deficit weight parameter (dimensionless)
- `C_L1(t)`: L1 cost per transaction at time t (in ETH)
- `D(t)`: Vault deficit at time t (in ETH)
- `H`: Prediction horizon (integer number of steps)

### L1 Cost Calculation
```
C_L1(t) = BaseFee_L1(t) × Gas_per_tx / 10^18
```

### Gas Per Transaction Formula
```
Gas_per_tx = max(200,000 / Expected_Tx_Volume, 2,000)
```

**Default Values:**
- Expected_Tx_Volume: 100 transactions per batch
- Gas_per_tx: max(200,000 / 100, 2,000) = max(2,000, 2,000) = **2,000 gas**

## 🔧 Implementation Standards

### Basefee Floor Policy
- **No artificial 1 gwei floor enforcement**
- **Technical floor**: 0.001 gwei (1e6 wei) to prevent numerical issues
- **Rationale**: Allow natural sub-gwei periods as documented in real datasets (0.055-0.092 gwei)

### Parameter Constraints
- `μ ∈ [0.0, 1.0]`: L1 weight (0.0 = pure deficit correction, 1.0 = pure L1 tracking)
- `ν ∈ [0.1, 0.9]`: Deficit weight (0.1 = gentle correction, 0.9 = aggressive correction)
- `H ∈ [6, 576]`: Horizon in steps, must be divisible by 6 for batch alignment

## 🎯 Canonical Optimal Parameters

Based on post-timing-fix analysis with realistic lumpy cash flows:

### **Primary Recommendation (Optimal)**
```
μ = 0.0    # Pure deficit correction
ν = 0.1    # Gentle deficit correction
H = 36     # 72 seconds prediction horizon
```

### **Balanced Alternative**
```
μ = 0.0    # Pure deficit correction
ν = 0.2    # Moderate deficit correction
H = 72     # 144 seconds prediction horizon
```

### **Crisis-Resilient Alternative**
```
μ = 0.0    # Pure deficit correction
ν = 0.7    # Aggressive deficit correction
H = 288    # 576 seconds prediction horizon
```

## 🔄 Timing Parameters

### Taiko L2 Block Timing
- **L2 Block Time**: 2 seconds
- **L1 Batch Interval**: 12 seconds (every 6 L2 blocks)
- **Step Definition**: 1 step = 2 seconds = 1 Taiko L2 block

### Vault Cash Flow Timing (Critical for Realism)
- **Fee Collection**: Every 2 seconds (L2 block)
- **L1 Cost Payment**: Every 12 seconds (L1 batch submission)
- **Result**: Realistic lumpy cash flows instead of artificial smoothness

## 💻 Implementation Requirements

### JavaScript Implementations
All JavaScript files must implement exactly:
```javascript
// Fee calculation
const l1Component = this.mu * l1CostEstimate;
const deficitComponent = this.nu * vaultDeficit / this.H;
const estimatedFee = l1Component + deficitComponent;

// Gas calculation
const expectedTxVolume = this.txsPerBatch || 100;
const gasPerTx = Math.max(200000 / expectedTxVolume, 2000);

// Basefee floor (technical only)
const basefee = Math.max(calculatedBasefee, 1e6); // 0.001 gwei floor
```

### Python Implementation
The Python implementation must match:
```python
# Fee calculation
l1_component = self.mu * l1_cost_estimate
deficit_component = self.nu * vault_deficit / self.H
estimated_fee = l1_component + deficit_component

# Gas calculation
expected_tx_volume = self.txs_per_batch or 100
gas_per_tx = max(200000 / expected_tx_volume, 2000)

# Basefee floor (technical only)
basefees = np.maximum(calculated_basefees, 1e6)  # 0.001 gwei floor
```

## 📊 Validation Requirements

### Cross-Implementation Consistency
- Python and JavaScript simulators must produce **identical results** for identical inputs
- All gas calculations must use the same formula
- All basefee floor policies must be identical
- All parameter defaults must match

### Parameter Validation
- All implementations must validate parameter ranges
- All implementations must enforce H divisible by 6
- All implementations must use the canonical optimal parameters as defaults

## 📋 Implementation Checklist

### ✅ Completed (Latest Consistency Restoration)
- [x] Fixed taiko-simulator-js.js formula to include H term
- [x] Standardized gas-per-tx calculations across all JavaScript files
- [x] Removed artificial 1 gwei basefee floors in Python and JavaScript
- [x] Updated README.md to reflect actual repository structure
- [x] Corrected documentation claims about bug fixes

### 🔄 Validation Tasks
- [ ] Cross-validate Python vs JavaScript implementations produce identical results
- [ ] Verify all presets use canonical parameters
- [ ] Test edge cases (sub-gwei periods, high volatility)
- [ ] Confirm all gas calculations produce expected values

## 🔒 Governance

This specification is the **authoritative reference**. Any changes to:
- Mathematical formulas
- Parameter defaults
- Implementation standards
- Optimal parameter recommendations

Must be reflected in **this document first**, then propagated to all implementations.

## 📅 Version History

- **v1.0** (2025-01-01): Initial canonical specification established
- **Post-Consistency-Restoration**: All major discrepancies resolved

---

*This specification ensures scientific rigor and implementation consistency across the Taiko fee mechanism research and deployment.*