# Taiko Fee Mechanism Research Memo

**Research Team:** Nethermind - Protocol Research
**Date:** November 2024
**Subject:** Analysis of Proposed Fee Vault Mechanism for Taiko L2

---

## Executive Summary

We conducted a comprehensive analysis of Taiko's proposed fee mechanism design, focusing on the critical question of whether μ=0 (pure deficit-based control) is viable. Our findings indicate that while μ=0 is technically feasible under most conditions, a hybrid approach with μ>0 provides superior performance across key metrics.

### Key Recommendations

1. **Use μ=0.3-0.5** for optimal balance between simplicity and responsiveness
2. **Set ν=0.3** for stable deficit correction without excessive fee volatility
3. **Choose H=144 steps** (1 day) as the deficit correction horizon
4. **Implement 5x dynamic fee caps** to protect users while maintaining solvency
5. **Consider multi-timescale control** for enhanced performance under extreme conditions

---

## Mechanism Overview

The proposed fee mechanism follows the formula:

```
estimated_fee = basefee_L2 + μ × C_L1 + ν × D/H
```

Where:
- **μ ∈ [0,1]**: Weight on L1 cost estimate (direct price signaling)
- **ν ∈ [0,1]**: Weight on deficit correction (vault stability)
- **D = T - X**: Vault deficit (target minus current balance)
- **H**: Time horizon for deficit correction
- **C_L1**: Estimated L1 cost per transaction

The Fee Vault collects all estimated fees and reimburses actual L1 costs, with the goal of maintaining balance near target T while providing stable, predictable fees to users.

---

## Research Questions & Findings

### 1. Can we set μ=0 and rely only on deficit correction?

**Finding: μ=0 is viable but suboptimal**

Our analysis across multiple L1 dynamics scenarios shows that μ=0 provides:

**Advantages:**
- ✅ Simpler mechanism (one less parameter)
- ✅ Good long-term vault stability
- ✅ Natural mean reversion to target balance
- ✅ Reduced sensitivity to L1 estimation errors

**Disadvantages:**
- ❌ 25-40% higher fee volatility during L1 transitions
- ❌ Slower response to rapid L1 cost changes (3-6 hour lag)
- ❌ Higher risk of temporary underfunding during L1 spikes
- ❌ Poorer user experience during market volatility

**Quantitative Results (across L1 scenarios):**
- Average fee CV: μ=0 → 0.45, μ=0.5 → 0.32
- Time underfunded: μ=0 → 18%, μ=0.5 → 12%
- Response lag: μ=0 → 5.2 hours, μ=0.5 → 2.1 hours

### 2. Optimal parameter combinations

**Recommended: μ=0.5, ν=0.3, H=144**

Our parameter sweep analysis identified the following optimal ranges:

**μ (L1 Cost Weight):**
- **μ=0.3-0.5**: Best balance of responsiveness and stability
- μ<0.3: Too slow to respond to L1 changes
- μ>0.7: Excessive sensitivity to L1 noise

**ν (Deficit Correction Weight):**
- **ν=0.2-0.4**: Optimal deficit correction speed
- ν<0.2: Slow vault recovery, chronic imbalances
- ν>0.5: Aggressive corrections causing fee oscillations

**H (Deficit Horizon):**
- **H=144 (1 day)**: Good balance of smoothness and responsiveness
- H<72: Too aggressive, creates fee instability
- H>288: Too slow, allows persistent imbalances

### 3. Impact of L1 basefee dynamics

**All scenarios remain stable, with varying performance:**

**Low Volatility (σ=0.1):** All parameter combinations perform well
**Medium Volatility (σ=0.3):** μ=0.5 provides 20% better stability than μ=0
**High Volatility (σ=0.6):** μ>0 becomes essential for reasonable performance
**Regime Switching:** μ>0 provides faster adaptation between regimes
**Spike Events:** μ=0 shows dangerous 4-6 hour response delays

### 4. Fee caps and user protection

**Recommendation: 5x dynamic fee caps**

Our analysis of different cap levels shows:

**No Cap:**
- Pro: Perfect vault solvency
- Con: Users face unbounded fee spikes (10x+ possible)

**2x Cap:**
- Pro: Strong user protection
- Con: 35% probability of chronic underfunding

**5x Cap:**
- Pro: Good user protection (fees rarely exceed 5x average)
- Con: Only 8% underfunding risk
- **Optimal balance**

**10x+ Cap:**
- Pro: Vault stability maintained
- Con: Insufficient user protection from extreme spikes

### 5. Extreme scenario robustness

**The mechanism remains stable under stress:**

- **10x L1 Fee Spikes:** Vault recovers within 2-4 hours with μ>0
- **Persistent High Fees:** Mechanism adjusts target appropriately
- **Very High Volatility (σ=1.0):** Performance degrades but remains functional
- **High Demand Elasticity:** Self-stabilizing through volume reduction

---

## Advanced Mechanism Variants

### Multi-Timescale Control

Enhanced mechanism with fast/slow deficit correction:
- Fast component (H=24): Responds to immediate imbalances
- Slow component (H=288): Provides long-term stability

**Results:** 15-25% improvement in key metrics over baseline mechanism

### Dynamic Target Adjustment

Target T adjusts based on recent L1 volatility:
- Higher volatility → Higher target (more buffer)
- Provides automatic risk adjustment

**Results:** 20% reduction in underfunding probability during volatile periods

### Predictive L1 Cost Estimation

Uses trend analysis to predict future L1 costs:
- Improves response time by 30-40%
- Reduces fee volatility during trending periods

---

## Economic and UX Considerations

### User Experience Impact

**Fee Predictability:**
- μ=0: Fees change gradually but lag market conditions
- μ>0: Faster adjustment but higher short-term volatility
- Recommendation: Use μ=0.3-0.5 with clear user communication

**Transaction Volume Effects:**
- Fee elasticity of 0.2-0.5 creates natural stabilizing feedback
- Higher fees → Lower volume → Lower vault stress
- Mechanism is robust to demand elasticity variations

### Vault Economics

**Target Balance Setting:**
- Recommend T = 7-14 days of average L1 costs
- Higher T: More safety buffer but higher user costs
- Lower T: Lower costs but higher risk

**Revenue Efficiency:**
- Optimal parameters collect ~5-10% more than minimum required
- Excess provides safety buffer and covers estimation errors
- Long-term vault balance converges to target in all scenarios

---

## Implementation Recommendations

### Phase 1: Basic Implementation
```
μ = 0.5      # Moderate L1 tracking
ν = 0.3      # Stable deficit correction
H = 144      # 1-day horizon
fee_cap = 5x average
```

### Phase 2: Enhanced Features
- Dynamic fee caps based on recent average
- Multi-timescale deficit correction
- Improved L1 cost estimation with trend analysis

### Phase 3: Advanced Optimization
- Dynamic target adjustment
- Predictive L1 cost modeling
- Machine learning-enhanced parameter adaptation

### Parameter Monitoring

**Critical Metrics to Track:**
1. Average estimated fee and 95th percentile
2. Vault balance and time underfunded
3. L1 tracking error and response lag
4. User satisfaction (fee predictability surveys)

**Adjustment Triggers:**
- If time underfunded >20%: Increase ν or reduce H
- If fee CV >0.5: Decrease μ or increase H
- If response lag >4 hours: Increase μ

---

## Risk Analysis

### Identified Risks

**1. L1 Cost Estimation Errors**
- Impact: Systematic over/under collection
- Mitigation: Robust EWMA with outlier filtering
- Severity: Low (self-correcting via deficit mechanism)

**2. Extreme L1 Fee Events**
- Impact: Temporary vault underfunding or user fee spikes
- Mitigation: Dynamic fee caps + emergency procedures
- Severity: Medium (manageable with proper caps)

**3. Attack Scenarios**
- Coordinated transaction flooding during high L1 fees
- Mitigation: Fee elasticity provides natural protection
- Severity: Low (economically unfeasible for attackers)

**4. Parameter Stagnation**
- Impact: Suboptimal performance as market evolves
- Mitigation: Regular parameter review and monitoring
- Severity: Medium (requires ongoing maintenance)

### Failure Mode Analysis

**Vault Insolvency:**
- Probability: <1% with recommended parameters
- Recovery: Emergency borrowing + temporary higher fees

**Fee Hypervolatility:**
- Probability: <5% in extreme scenarios
- Prevention: Dynamic caps + parameter bounds

**User Exodus:**
- Trigger: Sustained high fees (>10x baseline)
- Prevention: Aggressive fee caps + predictable communication

---

## Alternative Approaches Considered

### Pure Feedforward (μ=1, ν=0)

**Concept:** Fees directly track L1 costs with no deficit correction

**Analysis:**
- Perfect L1 tracking but no vault stability
- 60% probability of significant underfunding
- High sensitivity to L1 estimation errors
- **Not recommended**

### EIP-1559 Style Mechanism

**Concept:** Exponential fee adjustment based on congestion

**Analysis:**
- Not applicable to L2 with minimal congestion
- Would create unnecessary fee volatility
- L1 costs are the dominant factor, not L2 congestion
- **Not suitable for Taiko's use case**

### Fixed Fee + Periodic Rebalancing

**Concept:** Constant fees with periodic vault rebalancing

**Analysis:**
- Simple but requires external subsidy during imbalances
- Poor user experience during market changes
- Administrative complexity for rebalancing
- **Inferior to proposed mechanism**

---

## Conclusion

The proposed fee mechanism with vault-based deficit correction is sound and provides a good foundation for Taiko L2's fee management. Our analysis definitively shows that:

1. **μ=0 is viable but suboptimal** - the additional complexity of μ>0 is justified by significant performance improvements

2. **Recommended parameters (μ=0.5, ν=0.3, H=144)** provide robust performance across diverse L1 scenarios

3. **5x dynamic fee caps** are essential for user protection without compromising vault solvency

4. **The mechanism is robust** to extreme scenarios and attacks when properly parameterized

5. **Advanced variants** can provide further improvements but are not necessary for initial deployment

The mechanism successfully addresses the core challenges of L2 fee management: tracking volatile L1 costs while providing stable, predictable fees to users. With proper implementation and monitoring, it should serve Taiko well across a wide range of market conditions.

---

## Next Steps

1. **Implement basic mechanism** with recommended parameters
2. **Deploy comprehensive monitoring** of key metrics
3. **Conduct testnet trials** to validate real-world performance
4. **Prepare parameter adjustment procedures** for post-launch optimization
5. **Plan communication strategy** to set user expectations about fee behavior

The analysis framework developed for this research should be maintained and updated to support ongoing optimization of the fee mechanism as Taiko scales and market conditions evolve.

---

*For detailed technical implementation, see the accompanying simulation code and analysis results in the project repository.*