# Taiko Fee Mechanism - Authoritative Specification

**⚠️ SINGLE SOURCE OF TRUTH - This document supersedes all other specifications**

This specification is derived from SUMMARY.md and serves as the definitive reference for all implementations.

## Formula: L2 Sustainability Basefee

### Core Mechanism

The **raw L2 sustainability basefee** is calculated as:

```
F_L2_raw(t) = μ × C_DA(t) + ν × C_vault(t)
```

Where:
- **C_DA(t) = α_data × B̂_L1(t)**: Smoothed marginal DA cost per L2 gas
- **C_vault(t) = D(t)/(H × Q̄)**: Full-strength vault healing surcharge per L2 gas

### Component Definitions

#### 1. Smoothed L1 Basefee
```
B̂_L1(t) = (1 - λ_B) × B̂_L1(t-1) + λ_B × B_L1(t)
```

#### 2. Vault Deficit
```
D(t) = T - V(t)
```
Where D(t) > 0 means vault is underfunded.

#### 3. UX Wrapper (Clipping and Rate Limiting)

**Step 1 - Clipping:**
```
F_clip(t) = min(max(F_L2_raw(t), F_min), F_max)
```

**Step 2 - Rate Limiting:**
```
F_L2(t) = min(
    F_L2(t-1) × (1 + κ_↑),
    max(
        F_L2(t-1) × (1 - κ_↓),
        F_clip(t)
    )
)
```

## Parameters

### Mechanism Parameters
- **μ ∈ [0,1]**: DA cost pass-through coefficient
- **ν ∈ [0,1]**: Vault healing intensity coefficient
- **H > 0**: Recovery horizon (batches under typical load)
- **λ_B ∈ (0,1]**: EMA smoothing parameter for L1 basefee

### System Constants (Require Real Data Calibration)
- **α_data > 0**: Expected L1 DA gas per 1 L2 gas (**NEEDS CALIBRATION FROM REAL TAIKO DATA**)
- **Q̄ > 0**: Typical L2 gas per batch (**NEEDS CALIBRATION FROM REAL TAIKO DATA**)
- **T > 0**: Fee vault target balance (ETH)

### UX Parameters
- **F_min, F_max**: Min/max sustainability basefee bounds
- **κ_↑, κ_↓ ∈ [0,1]**: Max relative fee increase/decrease per batch

## Data Calibration Requirements

**🚨 CRITICAL: The following parameters MUST be calibrated from real Taiko network data:**

1. **α_data**: Extract from actual proposeBlock transaction gas usage:
   ```
   α_data ≈ E[L1_DA_gas_per_batch] / Q̄
   ```

2. **Q̄**: Measure from actual L2 batch gas consumption:
   ```
   Q̄ ≈ long_run_average(L2_gas_per_batch)
   ```

**⚠️ NO MOCK VALUES**: If real data is unavailable, parameters must be marked as **UNCALIBRATED** with prominent warnings.

## Implementation Requirements

### 1. Formula Consistency
All Python and JavaScript implementations MUST implement the identical mathematical formulas above.

### 2. Parameter Validation
- All parameters must be within documented ranges
- α_data and Q̄ must have data provenance documentation
- Any uncalibrated parameters must trigger warnings

### 3. Cross-Language Testing
- Python and JavaScript must produce identical results for identical inputs
- CI gates must prevent deployment of inconsistent implementations

## Optimization Objectives

Parameters should be chosen to optimize:
1. **Cost Recovery**: Long-run revenues ≈ L1 DA costs (CRR ≈ 1.0)
2. **Vault Solvency**: Low probability of vault depletion
3. **UX Quality**: Stable, predictable fee trajectory
4. **Capital Efficiency**: Minimal required vault size

## Governance

**This specification is immutable.** Any changes require:
1. Update this document first
2. Propagate to all implementations
3. Re-run consistency tests
4. Re-validate optimization results

---

**Status**: Implementation in progress
**Last Updated**: December 2024