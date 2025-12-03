# L2 Fee Vault Mechanism for Taiko – Mechanism & Objectives

This document summarizes the **Fee Vault + L2 basefee** mechanism, the control formula, the vault dynamics, and the objectives/constraints we will use to optimize parameters, with Taiko-specific calibration for the first iteration.

---

## 1. Time Indexing and Context

We model at the **batch level**, where each batch corresponds to an L2→L1 posting (e.g. one L1 block that includes Taiko data).

* (t = 0,1,2,\dots): batch index (approximately one per Ethereum L1 block).
* Ethereum L1 block time: (\approx 12) seconds.

For Taiko Alethia, current **Total Gas Used (24h)** is roughly (4.94 \times 10^9) gas, and network utilization is very low (~0.1%), meaning actual usage is far below configured capacity. ([Taiko Alethia Blockchain Explorer][1])

An approximate **average gas per L1 block (batch)** is:

[
\bar Q
\approx \frac{4.94 \times 10^{9} \ \text{gas/day}}{7{,}200\ \text{L1 blocks/day}}
\approx 6.9 \times 10^5 \ \text{gas per batch}.
]

For the **first modeling pass**, we take:

[
Q(t) \equiv \bar Q \approx 6.9 \times 10^5 \ \text{gas per batch},
]

and ignore demand elasticity (fees do not affect (Q) yet).

Later we can generalize to (Q(t)) stochastic or demand-dependent.

---

## 2. State Variables and Notation

### 2.1 Core variables

* (C_{L1}(t)): *actual* L1 data cost (in ETH or some unit) to post Taiko batch (t).
* (\hat C_{L1}(t)): smoothed/estimated L1 cost used by the controller.
* (Q(t)): total L2 gas in batch (t) (for now (Q(t) = \bar Q)).
* (F_{L2}(t)): **per-gas** L2 basefee that funds the Fee Vault in batch (t) (in ETH/gas).
* (V(t)): Fee Vault balance *just before* processing batch (t).
* (S(t)): subsidy paid *from* the vault to the proposer for batch (t).

### 2.2 Target and deficit

* (T > 0): target vault level (desired buffer).
* Deficit (positive when below target, negative in surplus):

[
D(t) := T - V(t).
]

So:

* (D(t) > 0) ⇒ vault below target (deficit).
* (D(t) < 0) ⇒ vault above target (surplus).

### 2.3 Smoothing of L1 cost

We use an EMA for L1 posting cost:

[
\hat C_{L1}(t)
==============

(1-\lambda_C),\hat C_{L1}(t-1)
+
\lambda_C,C_{L1}(t),
\quad
\lambda_C \in (0,1].
]

With (Q(t) \equiv \bar Q), we do **not** initially smooth (Q); it is taken as constant.

---

## 3. Fee Rule (Per-Gas Basefee)

The controller is designed around a **desired per-batch revenue** consisting of:

1. A share (\mu) of the *current* smoothed L1 cost.
2. A term that amortizes the current deficit over a horizon (H).

We divide by (\bar Q) to obtain a per-gas basefee.

### 3.1 Raw per-gas basefee

Parameters:

* (\mu \in [0,1]): weight for **instantaneous pass-through** of L1 cost.
* (\nu \in [0,1]): weight for **deficit correction**.
* (H > 0): amortization horizon (in batches).

Raw basefee:

[
f^{\mathrm{raw}}(t)
===================

\mu,\frac{\hat C_{L1}(t)}{\bar Q}
+
\nu,\frac{D(t)}{H,\bar Q}.
]

Interpretation:

* First term: “this batch should contribute a fraction (\mu) of its expected L1 posting cost.”
* Second term: “this batch should pay (\nu/H) of the current deficit (D(t)) (or receive a discount if (D(t) < 0)).”

### 3.2 Static min/max bounds

To enforce absolute UX bounds:

* (F_{\min} \ge 0): minimum basefee.
* (F_{\max} > F_{\min}): maximum basefee.

We clamp:

[
f^{\mathrm{clip}}(t)
====================

\min\big(F_{\max},\ \max(F_{\min},\ f^{\mathrm{raw}}(t))\big).
]

### 3.3 Rate-limiting fee jumps (optional)

To limit per-batch fee jumps, define:

* (F_{L2}(t-1)): last per-gas basefee.
* (\kappa_\uparrow, \kappa_\downarrow \in [0,1]): max relative up/down moves per batch.

Then:

[
F_{L2}(t) =
\min\Big(
F_{L2}(t-1)(1+\kappa_\uparrow),\
\max\big(F_{L2}(t-1)(1-\kappa_\downarrow),\ f^{\mathrm{clip}}(t)\big)
\Big).
]

This enforces:

[
F_{L2}(t-1)(1-\kappa_\downarrow)
;\le;
F_{L2}(t)
;\le;
F_{L2}(t-1)(1+\kappa_\uparrow).
]

For the **core analysis**, we can initially set (\kappa_\uparrow = \kappa_\downarrow = 1) (no rate limiting), then re-introduce them in a second phase.

---

## 4. Vault Dynamics

### 4.1 Subsidy rule

We assume **full reimbursement while solvent**:

[
S(t) = \min\big(C_{L1}(t),\ V(t)\big).
]

* If (V(t) \ge C_{L1}(t)): proposer fully reimbursed, (S(t) = C_{L1}(t)).
* If (V(t) < C_{L1}(t)): vault is emptied, (S(t) = V(t)), proposer eats the shortfall.

More complex “partial reimbursement under stress” rules are possible but not modeled here.

### 4.2 Vault balance update

Total fees collected in batch (t):

[
R(t) = F_{L2}(t),Q(t) = F_{L2}(t),\bar Q.
]

Vault update:

[
V(t+1) = V(t) + R(t) - S(t).
]

Deficit update:

[
D(t+1)
= T - V(t+1)
= T - \big(V(t) + R(t) - S(t)\big).
]

### 4.3 Approximate deficit recursion (no smoothing, solvent regime)

To understand the structure, consider a simplification:

* Ignore clamping and rate limits.
* Ignore smoothing ((\hat C_{L1}(t) = C_{L1}(t))).
* Assume vault is solvent so (S(t) = C_{L1}(t)).
* Assume we hit the intended revenue exactly:

[
R(t) = F_{L2}(t),\bar Q
=======================

\mu,C_{L1}(t) + \nu,\frac{D(t)}{H}.
]

Then:

[
\begin{aligned}
D(t+1)
&= T - V(t+1) \
&= T - \big(V(t) + R(t) - C_{L1}(t)\big) \
&= (T - V(t)) - R(t) + C_{L1}(t) \
&= D(t) - \Big(\mu,C_{L1}(t) + \nu,\frac{D(t)}{H}\Big) + C_{L1}(t) \
&= \Big(1 - \frac{\nu}{H}\Big) D(t)

* (1-\mu),C_{L1}(t).
  \end{aligned}
  ]

So (D(t)) behaves like an AR(1) process:

[
D(t+1)
======

\phi D(t) + (1-\mu),C_{L1}(t),
\quad
\phi := 1 - \frac{\nu}{H}.
]

Key qualitative points (still informative when we reintroduce smoothing and clamping):

* (\nu/H) controls **mean reversion speed** of the deficit.
* ((1-\mu)) controls **how much of L1 volatility** is absorbed by the buffer vs passed through to fees.

---

## 5. Parameter Sets: Core vs Extended

We conceptually split parameters into:

### 5.1 Core controller parameters

These shape the fundamental economics:

[
\theta_{\mathrm{core}} = (\mu,\ \nu,\ H).
]

* (\mu): instant pass-through vs buffering.
* (\nu/H): deficit correction speed.
* (H): intertemporal sharing horizon.

### 5.2 Extended parameters (UX & smoothing)

These refine UX and add guardrails:

* (\lambda_C): smoothing for L1 cost in (\hat C_{L1}(t)).
* (F_{\min}, F_{\max}): absolute min/max fee.
* (\kappa_\uparrow,\kappa_\downarrow): rate limits on fee jumps.
* (Later) a non-constant (Q(t)) or a smoothed (\hat Q(t)).

### 5.3 Buffer size (T)

(T) is the vault target (buffer size). There are two approaches:

* **First pass**: treat (T) as fixed by governance (e.g. “buffer ≈ X days of average L1 costs”), not part of optimization.
* **Second pass**: include (T) in the search to study capital efficiency (“How small can (T) be while still meeting constraints and UX targets?”).

The full parameter vector, if we decide to optimize everything, is:

[
\theta
======

\big(
\mu,\ \nu,\ H,\
\lambda_C,\
F_{\min}, F_{\max},\
\kappa_\uparrow, \kappa_\downarrow,\
T
\big).
]

---

## 6. Hard Constraints

We **discard** any (\theta) that violates these.

### 6.1 Solvency / ruin probability

Let:

* (V_{\min}): critical threshold (e.g. (0) or (0.1T)).
* (T_{\mathrm{horizon}}): evaluation horizon (e.g. 1 year of batches).

Ruin probability:

[
p_{\mathrm{insolv}}(\theta)
===========================

\Pr\big[\exists t \le T_{\mathrm{horizon}}:\ V(t) < V_{\min}\big].
]

**Constraint:**

[
p_{\mathrm{insolv}}(\theta) \le \varepsilon,
\quad
\varepsilon \text{ small (e.g. } 10^{-2} \text{ or } 10^{-3}).
]

---

### 6.2 Long-run cost recovery (budget balance)

Cost recovery ratio:

[
\text{CRR}(\theta)
==================

# \frac{\sum_t F_{L2}(t),Q(t)}{\sum_t C_{L1}(t)}

\frac{\sum_t F_{L2}(t),\bar Q}{\sum_t C_{L1}(t)}.
]

**Constraint:**

[
\text{CRR}(\theta)
\in [1-\delta_{\mathrm{cr}},,1+\delta_{\mathrm{cr}}],
]

with (\delta_{\mathrm{cr}}) small (e.g. (\delta_{\mathrm{cr}} = 0.05), i.e. ±5%).

---

### 6.3 Extreme fee bound (UX sanity)

Let:

[
F_{0.99}(\theta)
================

\text{p99 of } F_{L2}(t).
]

**Constraint:**

[
F_{0.99}(\theta) \le F_{\max}^{\mathrm{UX}},
]

where (F_{\max}^{\mathrm{UX}}) is an upper bound on fees still acceptable for UX (e.g. a multiple of some baseline L2 gas price).

---

### 6.4 (Optional) Fairness guardrail

We do **not** optimize primarily for fairness, but we may want to rule out obviously pathological intertemporal cross-subsidies.

Define cohorts (k) (e.g. monthly windows). For each (k):

[
F_{\mathrm{paid}}(k) = \sum_{t \in k} F_{L2}(t),Q(t),
\quad
C_{\mathrm{resp}}(k) = \sum_{t \in k} C_{L1}(t),
]

markup:

[
\phi(k) = \frac{F_{\mathrm{paid}}(k)}{C_{\mathrm{resp}}(k)}.
]

Weak fairness constraint (example):

[
\phi(k) \in [1-\delta_{\mathrm{fair}},,1+\delta_{\mathrm{fair}}]
\quad
\text{for at least } 95% \text{ of cohorts }k,
]

with (\delta_{\mathrm{fair}}) relatively loose (e.g. 0.5).

---

## 7. Main Objective Axes (for Pareto Optimization)

Subject to the hard constraints, we search for Pareto-optimal (\theta) along three main axes:

1. **User UX** – fee level and predictability.
2. **Protocol robustness** – depth/length of stressed vault episodes.
3. **Capital efficiency** – buffer size relative to throughput.

### 7.1 UX objective: fee level and predictability

We want fees that are:

* Cheap on average.
* Globally stable.
* Locally smooth (few big jumps).
* Predictable over 1h / 6h horizons.

**Metrics:**

* Average fee:

  [
  \bar F(\theta) = \mathbb{E}[F_{L2}(t)].
  ]

* Global coefficient of variation:

  [
  \text{CV}*F(\theta) =
  \frac{\sqrt{\mathrm{Var}[F*{L2}(t)]}}{\bar F(\theta)}.
  ]

* 95th percentile relative jump:

  [
  J_{0.95}(\theta)
  ================

  \text{p95 of }
  \left|
  \frac{F_{L2}(t+1) - F_{L2}(t)}{F_{L2}(t)}
  \right|.
  ]

* Rolling 1h CV, (\text{CV}_{1h}(\theta)): average coefficient of variation computed over sliding 1-hour windows.

* Rolling 6h CV, (\text{CV}_{6h}(\theta)): same over 6-hour windows.

**Aggregated UX objective (to minimize):**

[
J_{\mathrm{UX}}(\theta)
=======================

w_1,\bar F(\theta)
+
w_2,\text{CV}*F(\theta)
+
w_3,J*{0.95}(\theta)
+
w_4,\text{CV}*{1h}(\theta)
+
w_5,\text{CV}*{6h}(\theta),
]

with weights (w_i > 0) chosen to balance scales.

---

### 7.2 Robustness objective: how bad “bad times” are

Even with small ruin probability, we care about how deep and how long the vault sits in deficit when stressed.

**Metrics:**

* Deficit-weighted duration (severity of underfunding):

  [
  \text{DWD}(\theta)
  ==================

  \mathbb{E}
  \left[
  \sum_t (V_{\min} - V(t))_+
  \right].
  ]

* Max continuous underfunding streak:

  [
  L_{\max}(\theta)
  ================

  \max
  {\text{length of consecutive batches with } V(t) < V_{\min}}.
  ]

**Aggregated robustness objective (to minimize):**

[
J_{\mathrm{Robust}}(\theta)
===========================

u_1,\text{DWD}(\theta)
+
u_2,L_{\max}(\theta),
]

with (u_i > 0).

---

### 7.3 Capital efficiency objective

Given the same solvency guarantees and UX, a smaller average buffer is preferable.

**Metrics:**

* Average vault size:

  [
  \bar V(\theta) = \mathbb{E}[V(t)].
  ]

* Capital per unit throughput:

  [
  \text{CapEff}(\theta)
  =====================

  # \frac{\bar V(\theta)}{\mathbb{E}[Q(t)]}

  \frac{\bar V(\theta)}{\bar Q}
  \quad
  \text{(in first-pass model)}.
  ]

**Capital efficiency objective (to minimize):**

[
J_{\mathrm{CapEff}}(\theta) = \text{CapEff}(\theta).
]

---

## 8. Pareto Optimization Pipeline

Given:

* A stochastic or empirical process for (C_{L1}(t)),
* (Q(t) \equiv \bar Q) in the first pass,
* Parameter space for (\theta_{\mathrm{core}} = (\mu,\nu,H)) (and optionally extended parameters),

we proceed as follows:

1. **Simulation**:
   For each (\theta), simulate trajectories

   [
   {F_{L2}(t),\ V(t),\ D(t),\ Q(t)}*{t=0}^{T*{\mathrm{horizon}}}
   ]

   over many scenarios.

2. **Metric computation**:
   For each (\theta), compute:

   * Hard-constraint metrics: (p_{\mathrm{insolv}}, \text{CRR}, F_{0.99}), and fairness metrics if used.
   * Objectives: (J_{\mathrm{UX}}(\theta), J_{\mathrm{Robust}}(\theta), J_{\mathrm{CapEff}}(\theta)).

3. **Constraint filtering**:
   Discard any (\theta) violating a hard constraint.

4. **Pareto frontier**:
   Among the remaining (\theta), construct the Pareto frontier in:

   [
   \big(
   J_{\mathrm{UX}}(\theta),\
   J_{\mathrm{Robust}}(\theta),\
   J_{\mathrm{CapEff}}(\theta)
   \big).
   ]

5. **Selection**:
   Governance / research then chooses a point on this frontier according to priorities:

   * UX-first (very smooth/cheap fees, accepting larger (T), slower recovery).
   * Safety-first (more responsive fees, tighter vault behavior).
   * Balanced (moderate fee volatility, vault staying near (T) with high probability, reasonable capital commitment).

In a **first phase**, we focus on (\theta_{\mathrm{core}} = (\mu,\nu,H)) with fixed (T), (\bar Q), and simple/no rate limits. In a **second phase**, we expand to the full (\theta) to refine UX and capital efficiency and possibly explore different choices of (T).

[1]: https://taikoscan.io/charts?utm_source=chatgpt.com "Taiko Alethia Charts and Statistics"
