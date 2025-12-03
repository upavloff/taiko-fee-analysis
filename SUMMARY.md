# Taiko Fee Mechanism and Parameter Optimization

This document defines:

1. The **L2 sustainability basefee** mechanism, which transforms volatile L1 DA costs into smooth L2 per-gas fees.
2. The **parameter optimization problem**, i.e., how to choose the mechanism parameters subject to solvency and UX constraints.

---

## 1. Fee Mechanism: L2 Sustainability Basefee

### 1.1 Notation

We work in discrete time indexed by batches or L2 blocks, denoted by \(t\).

#### State / exogenous variables

- \(B_{L1}(t)\)  
  L1 basefee at time \(t\), in **ETH per L1-gas**.

- \(\hat B_{L1}(t)\)  
  Smoothed L1 basefee (EMA of \(B_{L1}\)), in **ETH per L1-gas**.

- \(V(t)\)  
  Fee Vault balance before processing batch \(t\), in **ETH**.

- \(Q(t)\)  
  L2 gas used in batch \(t\), in **L2-gas units**.

- \(C_{L1}(t)\)  
  L1 DA cost paid to post batch \(t\), in **ETH**.

#### Governance / mechanism parameters

- \(\lambda_B \in (0,1]\)
  **EMA smoothing parameter** for L1 basefee. Controls the responsiveness vs stability trade-off in the smoothed basefee \(\hat{B}_{L1}(t)\). Lower values (e.g., 0.1) provide more stability but slower response to L1 market changes; higher values (e.g., 0.5) provide faster response but more volatility pass-through.

- \(\alpha_{\text{data}} > 0\)
  **Expected L1 DA gas per 1 L2-gas**. This critical parameter represents the data availability gas overhead ratio, empirically estimated from real Taiko proposeBlock transactions as:
  $$\alpha_{\text{data}} \approx \mathbb{E}\left[\frac{\text{L1 DA gas used per batch}}{\bar{Q}}\right]$$
  where \(\bar{Q}\) serves as the reference unit for typical L2 gas per batch. Typical values range from 15,000-25,000, representing the L1 gas overhead per unit of L2 gas processed.

- \(T > 0\)
  Fee Vault **target balance** in ETH.

- \(\bar Q > 0\)
  **Typical L2 gas per batch**, estimated from real Taiko throughput data (e.g., long-run average gas per L2 batch) and then **fixed**. This governance constant serves dual purposes: (1) as the denominator in vault healing calculations, and (2) as the reference unit for \(\alpha_{\text{data}}\) estimation from empirical Taiko transaction data.

- \(H > 0\)  
  **Recovery horizon** measured in batches under typical load \(\bar Q\).

- \(\mu \in [0,1]\)  
  **DA cost pass-through coefficient**.

- \(\nu \in [0,1]\)  
  **Vault healing intensity coefficient**.

- \(F_{\min} > 0\), \(F_{\max} > 0\)  
  Minimum and maximum sustainability basefee (ETH per L2-gas).

- \(\kappa_\uparrow \in [0,1]\), \(\kappa_\downarrow \in [0,1]\)  
  Maximum **relative** upward / downward change in basefee per batch.

---

### 1.2 Derived quantities

#### 1.2.1 Smoothed L1 gas price

We smooth the L1 basefee via an exponential moving average (EMA) to reduce fee volatility while maintaining responsiveness to genuine L1 market trends:

\[
\hat B_{L1}(t)
= (1 - \lambda_B) \hat B_{L1}(t-1) + \lambda_B B_{L1}(t),
\]

where:
- \(B_{L1}(t)\) is the instantaneous L1 basefee at batch \(t\) (in ETH per L1-gas),
- \(\lambda_B \in (0,1]\) controls the smoothing responsiveness,
- \(\hat B_{L1}(0) = B_{L1}(0)\) (initialization with first observation).

**Parameter estimation**: \(\lambda_B\) is calibrated through multi-objective optimization to balance fee stability (lower \(\lambda_B\)) against L1 market responsiveness (higher \(\lambda_B\)). Optimal values typically range from 0.1-0.5 depending on the desired stability-responsiveness trade-off.

#### 1.2.2 Vault deficit

Define the **vault deficit** as:

\[
D(t) := T - V(t),
\]

where:
- \(D(t) > 0\) means the vault is **under target**,
- \(D(t) < 0\) means the vault holds more than its target.

#### 1.2.3 Smoothed marginal DA cost per L2-gas

Define:

\[
C_{\text{DA}}(t)
:= \alpha_{\text{data}} \, \hat B_{L1}(t),
\]

which has units **ETH per L2-gas** and represents the **smoothed marginal DA cost** of one unit of L2-gas, given current L1 prices and empirically-measured DA gas usage.

**Components**:
- \(\hat B_{L1}(t)\): smoothed L1 gas price (ETH per L1-gas),
- \(\alpha_{\text{data}}\): expected L1 DA gas per 1 L2-gas.

**Parameter estimation**: \(\alpha_{\text{data}}\) is empirically calibrated from real Taiko L1 proposeBlock transactions:
\[
\alpha_{\text{data}} \approx \frac{1}{N} \sum_{i=1}^{N} \frac{\text{gasUsed}_{\text{L1,DA}}^{(i)}}{\bar{Q}}
\]
where \(N\) is the number of observed batches, \(\text{gasUsed}_{\text{L1,DA}}^{(i)}\) is the L1 data availability gas consumed by batch \(i\), and \(\bar{Q}\) is the reference L2 gas per batch. This captures the actual L1 gas overhead per unit of L2 processing, accounting for batch compression and data structure efficiency.

#### 1.2.4 Full-strength vault healing surcharge per L2-gas

Define:

\[
C_{\text{vault}}(t)
:= \frac{D(t)}{H \, \bar Q},
\]

which also has units **ETH per L2-gas**.

Interpretation:

- Under **typical load**, \(Q(t) \approx \bar Q\), the ETH collected **per batch** from this term (with intensity \(\nu = 1\)) is:
  \[
  C_{\text{vault}}(t) \cdot \bar Q = \frac{D(t)}{H}.
  \]
- Thus, for \(\nu = 1\), the healing term would amortize the current deficit \(D(t)\) **linearly over \(H\) batches** assuming batches carry \(\bar Q\) L2-gas.

The parameter \(\bar Q\) is chosen from **real throughput data** (e.g. long-run average gas used per L2 block), then **frozen** so that:
- \(H\) has a stable meaning (“recovery horizon in typical batches”),
- users are **not penalized** when actual \(Q(t)\) is temporarily low (no thin-market tax).

---

### 1.3 Sustainability basefee: raw formula

We define the **raw L2 sustainability basefee** as:

\[
F_{L2}^{\text{raw}}(t)
:= \mu \, C_{\text{DA}}(t)
+ \nu \, C_{\text{vault}}(t)
\quad \text{[ETH per L2-gas]}.
\]

Semantics:

- **DA cost coverage term**:
  \[
  \mu \, C_{\text{DA}}(t)
  = \mu \, \alpha_{\text{data}} \, \hat B_{L1}(t).
  \]
  - Passes through a fraction \(\mu\) of the smoothed marginal DA cost per L2-gas.
  - \(\mu = 1\): full pass-through of DA cost volatility to users.
  - \(\mu = 0\): vault absorbs all short-run DA volatility.

- **Vault healing term**:
  \[
  \nu \, C_{\text{vault}}(t)
  = \nu \, \frac{D(t)}{H \bar Q}.
  \]
  - Applies a fraction \(\nu\) of the **full-strength** per-gas surcharge that would amortize the current deficit over \(H\) typical batches.
  - Under typical load \(Q(t)\approx \bar Q\), deficit reduction per batch from this term is approximately:
    \[
    \nu \, C_{\text{vault}}(t) \, Q(t)
    \approx \frac{\nu}{H} D(t).
    \]
  - \(\nu\) directly controls the **aggressiveness** of deficit correction.

**No thin-market tax:**  
Because \(C_{\text{vault}}(t)\) uses the constant \(\bar Q\) rather than the realized \(Q(t)\), the **per-gas** surcharge \(\nu C_{\text{vault}}(t)\) does not increase when the batch is small. Instead, low-activity periods simply heal the deficit more slowly in total ETH, but individual users are not penalized.

---

### 1.4 UX wrapper: clipping and rate limiting

To avoid violent fee jumps and make the fee path more predictable, we apply:

1. **Global clipping** to the raw basefee:
   \[
   F_{\text{clip}}(t)
   := \min\big(\max(F_{L2}^{\text{raw}}(t), F_{\min}), F_{\max}\big).
   \]

2. **Rate limits** on changes from one batch to the next:
   \[
   F_{L2}(t)
   := \min\!\left(
        F_{L2}(t-1)(1 + \kappa_\uparrow),
        \max\big(
             F_{L2}(t-1)(1 - \kappa_\downarrow),
             F_{\text{clip}}(t)
        \big)
      \right).
   \]

Interpretation:

- \(F_{\min}\), \(F_{\max}\): absolute min/max sustainability basefee.
- \(\kappa_\uparrow\): max allowed **relative increase** per batch (e.g. +20%).
- \(\kappa_\downarrow\): max allowed **relative decrease** per batch.

The **user-facing** sustainability basefee is \(F_{L2}(t)\), in ETH per L2-gas.

---

### 1.5 Transaction fee

For a transaction \(i\) with gas \(g_i\) included in batch \(t\), the fee paid is:

\[
\text{fee}_i(t)
= g_i \cdot F_{L2}(t)
+ \text{priority fee}_i(t),
\]

where:

- \(F_{L2}(t)\): sustainability basefee defined above,
- \(\text{priority fee}_i(t)\): optional tip chosen by the user, paid to the proposer.

---

## 2. Parameter Optimization Problem

We collect mechanism parameters in:

\[
\theta =
(\mu, \nu, H, \lambda_B, F_{\min}, F_{\max}, \kappa_\uparrow, \kappa_\downarrow, T),
\]

with \(\alpha_{\text{data}}\) and \(\bar Q\) treated as inputs calibrated from data (or included in \(\theta\) if we also want to optimize them).

We aim to choose \(\theta\) such that:

1. **Long-run L2 revenues match L1 DA costs** (cost recovery).  
2. The Fee Vault remains **solvent** with high probability (safety).  
3. Subject to (1) and (2), the fee trajectory is **smooth and predictable**, and the required capital is **as low as possible**.

All metrics below are evaluated via simulation or historical replay under candidate \(\theta\).

---

### 2.1 Cost recovery (budget neutrality)

Let:

- \(F_{L2}(t;\theta)\): sustainability basefee produced by the mechanism under parameters \(\theta\).
- \(Q(t)\): realized L2-gas per batch.
- \(C_{L1}(t)\): realized L1 DA cost (ETH) per batch.

Define:

- **Total L2 revenue** from sustainability basefee:
  \[
  R(\theta) := \sum_t F_{L2}(t;\theta) \, Q(t).
  \]

- **Total L1 DA cost**:
  \[
  C_{L1} := \sum_t C_{L1}(t).
  \]

**Cost Recovery Ratio (CRR):**

\[
\text{CRR}(\theta) := \frac{R(\theta)}{C_{L1}}.
\]

We require:

\[
1 - \varepsilon_{\text{CRR}}
\;\le\;
\text{CRR}(\theta)
\;\le\;
1 + \varepsilon_{\text{CRR}},
\]

for some small tolerance \(\varepsilon_{\text{CRR}}\) (e.g. 5–10%), ensuring the protocol is approximately **budget-neutral** over the long run.

---

### 2.2 Vault solvency and ruin probability

Let \(V(t;\theta)\) be the Fee Vault balance under parameter set \(\theta\) and a given scenario \(s\).

Define:

- \(V_{\text{crit}}\): critical threshold below which the vault is considered **underfunded/unsafe**.

Over a distribution of scenarios \(\mathcal{S}\) (e.g. historical L1 traces, demand paths, stress tests), define:

**Ruin probability:**

\[
\rho_{\text{ruin}}(\theta)
:= \Pr_{s \sim \mathcal{S}}\left[
  \exists t :
  V_s(t;\theta) < V_{\text{crit}}
\right].
\]

We require:

\[
\rho_{\text{ruin}}(\theta) \le \varepsilon_{\text{ruin}},
\]

for a governance-chosen bound \(\varepsilon_{\text{ruin}}\) (e.g. maximum acceptable annual probability of underfunding).

Optionally, additional safety-related constraints can be imposed, such as:

- **Max deficit depth**:
  \[
  D_{\max}(\theta)
  := \max_t (T - V(t;\theta))_+
  \le D_{\max}^{\text{allowed}},
  \]
  where \((x)_+ := \max(x, 0)\).

- **Deficit-weighted duration**:
  \[
  \text{DD}(\theta)
  := \sum_t (T - V(t;\theta))_+
  \le \text{DD}_{\max}^{\text{allowed}}.
  \]

---

### 2.3 UX and capital efficiency objectives

Within the set of \(\theta\) satisfying **cost recovery** and **solvency** constraints, we trade off:

- Fee **stability/predictability** (UX),
- Safety **robustness** (beyond bare ruin probability),
- **Capital efficiency** of the vault.

We define three objective functions:

- \(J_{\text{UX}}(\theta)\): UX cost,
- \(J_{\text{safe}}(\theta)\): safety cost,
- \(J_{\text{eff}}(\theta)\): capital efficiency cost.

#### 2.3.1 UX objective \(J_{\text{UX}}(\theta)\)

We consider statistics of the fee trajectory \(F_{L2}(t;\theta)\):

- Median fee level:
  \[
  m_F(\theta) := \text{median}_t\big(F_{L2}(t;\theta)\big).
  \]

- High-percentile fee level (e.g. p95):
  \[
  F_{95}(\theta) := \text{p95}_t\big(F_{L2}(t;\theta)\big).
  \]

- Fee volatility (e.g. variance or coefficient of variation):
  \[
  \text{Var}_F(\theta)
  := \operatorname{Var}_t\big(F_{L2}(t;\theta)\big),
  \]
  \[
  \text{CV}_F(\theta)
  := \frac{\sqrt{\text{Var}_F(\theta)}}{m_F(\theta)}.
  \]

- Jumpiness (e.g. p95 of relative changes):
  \[
  J_{\Delta F}(\theta)
  := \text{p95}_t\left(
      \frac{\big|F_{L2}(t;\theta) - F_{L2}(t-1;\theta)\big|}
           {F_{L2}(t-1;\theta)}
    \right).
  \]

A simple composite UX objective could be:

\[
J_{\text{UX}}(\theta)
= a_1 \cdot \text{CV}_F(\theta)
+ a_2 \cdot J_{\Delta F}(\theta)
+ a_3 \cdot \max\big(0, F_{95}(\theta) - F_{\text{UX-cap}}\big),
\]

with weights \(a_i\) and a soft UX cap \(F_{\text{UX-cap}}\).

#### 2.3.2 Safety robustness objective \(J_{\text{safe}}(\theta)\)

Beyond ruin probability, we may penalize **how bad** deficits get.

Metrics:

- **Deficit-weighted duration**:
  \[
  \text{DD}(\theta)
  := \sum_t (T - V(t;\theta))_+.
  \]

- **Maximum deficit depth**:
  \[
  D_{\max}(\theta)
  := \max_t (T - V(t;\theta))_+.
  \]

- **Recovery time** after stress spikes, e.g. the number of batches needed for \(V(t;\theta)\) to return within a band around \(T\) after a synthetic L1 fee spike.

A composite could be:

\[
J_{\text{safe}}(\theta)
= b_1 \cdot \text{DD}(\theta)
+ b_2 \cdot D_{\max}(\theta)
+ b_3 \cdot \text{RecoveryTime}_{\text{shock}}(\theta),
\]

with weights \(b_i\).

#### 2.3.3 Capital efficiency objective \(J_{\text{eff}}(\theta)\)

We penalize excessive capital locked in the Fee Vault.

Metrics:

- Target size \(T\) required to satisfy the solvency constraint.
- Average deviation from target:
  \[
  \overline{|V-T|}(\theta)
  := \mathbb{E}_t\big[ |V(t;\theta) - T| \big].
  \]

- Capital per unit throughput:
  \[
  \text{CapEff}(\theta)
  := \frac{T}{\mathbb{E}_t[Q(t)]}.
  \]

A composite could be:

\[
J_{\text{eff}}(\theta)
= c_1 \cdot T
+ c_2 \cdot \overline{|V-T|}(\theta)
+ c_3 \cdot \text{CapEff}(\theta),
\]

with weights \(c_i\).

---

### 2.4 Overall optimization problem

Given the above, the parameter selection problem can be written as:

\[
\begin{aligned}
\min_{\theta} \quad
  & w_{\text{UX}} J_{\text{UX}}(\theta)
  + w_{\text{safe}} J_{\text{safe}}(\theta)
  + w_{\text{eff}} J_{\text{eff}}(\theta) \\
\text{s.t.} \quad
  & 1 - \varepsilon_{\text{CRR}}
    \le \text{CRR}(\theta)
    \le 1 + \varepsilon_{\text{CRR}},\\
  & \rho_{\text{ruin}}(\theta) \le \varepsilon_{\text{ruin}}, \\
  & \text{(and any additional safety / UX constraints).}
\end{aligned}
\]

Interpretation:

- The **mechanism** is fixed by
  \[
  F_{L2}^{\text{raw}}(t) = \mu C_{\text{DA}}(t) + \nu C_{\text{vault}}(t),
  \]
  plus clipping and rate-limiting to obtain \(F_{L2}(t)\).

- The **optimization** chooses \(\theta\) so that:
  1. Long-run revenues from the sustainability basefee match L1 DA costs (cost recovery),
  2. The Fee Vault remains solvent with high probability (safety),
  3. Among all such safe, budget-neutral configurations, the fee path is as smooth and predictable as possible, and the required vault capital is minimized.
