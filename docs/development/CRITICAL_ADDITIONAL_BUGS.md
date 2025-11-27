# 🚨 CRITICAL: Additional Bugs Identified in External Analysis

## Summary
External analysis revealed **5 additional critical bugs** beyond the initial gas calculation issue. These affect parameter interpretation, metrics accuracy, and documentation consistency.

## 🐛 Bug #1: Gas Per TX Implementation vs Documentation

### Issue
- **Code**: `gasPerTx = 200` (hardcoded)
- **README**: `gasPerTx = max(200,000 / Expected Tx Volume, 2,000)`
- **Impact**: 10x mismatch when expectedTxVolume = 10

### Current Broken State
```javascript
// simulator.js:142 - WRONG
this.gasPerTx = Math.max(this.batchGas / realisticBatchSize, 200);
// With batchGas=200000, realisticBatchSize=1000 → gasPerTx = 200

// README.md:34 - DOCUMENTED FORMULA
gasPerTx = max(200,000 / expectedTxVolume, 2,000);
// With expectedTxVolume=10 → gasPerTx = max(20000, 2000) = 20000
```

### Fix Required
```javascript
updateGasPerTx() {
    // Use documented formula from README
    this.gasPerTx = Math.max(this.batchGas / Math.max(this.baseTxVolume, 1), 2000);
    console.log(`gasPerTx = max(${this.batchGas} / ${this.baseTxVolume}, 2000) = ${this.gasPerTx}`);
}
```

## 🐛 Bug #2: L1 Tracking Error Wrong Unit Scale

### Issue
- **Fee calculation**: Uses actual `gasPerTx` (200 gas)
- **L1 tracking error**: Uses hardcoded 2000 gas
- **Impact**: Metrics measure different unit scale than simulator

### Current Broken State
```javascript
// simulator.js:192 - Fee calculation
return (basefeeForCost * this.gasPerTx) / 1e18; // uses ~200 gas

// simulator.js:371 - Metrics calculation
const l1Costs = l1Basefees.map(basefee => (basefee * 2000) / 1e18); // uses 2000 gas!
```

### Fix Required
```javascript
// Use consistent gasPerTx in metrics
const l1Costs = l1Basefees.map(basefee => (basefee * this.gasPerTx) / 1e18);
```

## 🐛 Bug #3: Basefee Floor Too High

### Issue
- **Code**: Floors at 1 gwei (`1e9` wei)
- **Real data**: Shows 0.055-0.092 gwei periods
- **Impact**: Simulated runs 10x+ higher than real low-fee periods

### Current Broken State
```javascript
// simulator.js:51
this.currentBaseFee = Math.max(newBaseFee, 1e9); // 1 gwei floor
```

### Fix Required
```javascript
// Allow sub-gwei basefees like real data
this.currentBaseFee = Math.max(newBaseFee, 1e6); // 0.001 gwei floor
```

## 🐛 Bug #4: Time Units Documentation Error

### Issue
- **README**: "H=144 ≈ 1 day"
- **Code**: H=144 steps × 2s/step = 288s = 4.8 minutes
- **Impact**: Horizon 300x shorter than documented

### Analysis
```
README claim: 144 steps ≈ 1 day = 86,400 seconds
Code reality: 144 steps × 2s = 288 seconds
Error factor: 86,400 / 288 = 300x difference
```

### Fix Required
Either:
1. Fix documentation: "H=144 ≈ 4.8 minutes"
2. Fix code: Use steps = minutes/blocks, not 2s blocks

## 🐛 Bug #5: Directory Structure Documentation

### Issue
- **README**: References `web/` directory
- **Reality**: Files at repo root
- **Impact**: Users can't find files

### Fix Required
Update README.md paths:
```bash
# OLD (WRONG)
cd web
open index.html

# NEW (CORRECT)
open index.html  # Files at repo root
```

## 🎯 Impact Assessment

### High Priority (Breaks Core Functionality)
1. **Bug #1**: Gas per TX wrong by 10x
2. **Bug #2**: L1 tracking metrics meaningless
3. **Bug #3**: Can't simulate realistic low-fee periods

### Medium Priority (Documentation/UX)
4. **Bug #4**: Time horizon completely misunderstood
5. **Bug #5**: Users can't find files

## 🚨 Critical Realization

These bugs mean our "corrected" analysis is **STILL WRONG**! The gas per TX should be:
- **Current**: 200 gas
- **Documented**: 20,000 gas (for baseTxVolume=10)
- **Impact**: L1 costs are 100x LOWER than they should be

**This completely invalidates our breakthrough conclusions again!**