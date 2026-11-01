# PackServe — Performance Model

## 1. Prefill/Decode Interference

Prefill and decode share GPU execution resources.

Prefill consumes a fraction of available GPU service time, reducing the capacity available for decode operations.

PackServe models this relationship as:
$TPOT_{\text{real}} = T_{\text{iter}}(b) / (1 - \lambda * T_{\text{pre}})$

### Variables

| Variable | Definition | Units |
|---|---|---|
| $(T_{\text{iter}}(b))$ | Pure decode iteration time for batch size \(b\) | Seconds/iteration |
| \(b\) | Decode batch size | Requests |
| \(\lambda\) | Incoming request rate | Requests/second |
| \(T_{\text{pre}}\) | Average prefill service time per request | Seconds/request |
| \(\lambda T_{\text{pre}}\) | Fraction of service time occupied by prefill | Dimensionless |
| \(TPOT_{\text{real}}\) | Effective time per output token | Seconds/token |

## 2. Derivation From First Principles

Consider a 1,000 ms wall-clock observation interval.

Assume:

- Arrival rate: 3 requests/second
- Prefill service time: 100 ms/request
- Pure decode iteration time: 35 ms

### Step 1 — Calculate prefill occupancy

\[
\lambda T_{\text{pre}}=3(0.1)=0.3
\]

Prefill occupies 30% of GPU service time.

### Step 2 — Calculate available decode time

\[
T_{\text{decode available}}=1000(1-0.3)=700\text{ ms}
\]

### Step 3 — Calculate completed decode iterations

\[
N_{\text{iterations}}=\frac{700}{35}=20
\]

### Step 4 — Calculate effective TPOT

\[
TPOT_{\text{real}}=\frac{1000}{20}=50\text{ ms}
\]

Substitute the iteration count:

\[
TPOT_{\text{real}}=\frac{1000}{700/35}
\]

Rearrange:

\[
TPOT_{\text{real}}=\frac{35}{700/1000}
\]

Since \(700/1000=1-\lambda T_{\text{pre}}\):

\[
\boxed{
TPOT_{\text{real}}\approx
\frac{T_{\text{iter}}(b)}
{1-\lambda T_{\text{pre}}}
}
\]

## 3. Why This Formulation Matters

The equation separates two performance factors.

**Intrinsic decode efficiency**

\[
T_{\text{iter}}(b)
\]

Represents the GPU's execution cost for a decode iteration at a given batch size.

**Prefill interference**

\[
1-\lambda T_{\text{pre}}
\]

Represents the fraction of service time remaining for decoding.

This separation allows the scheduler to reason about batching efficiency and interference independently.

## 4. Numerical Sensitivity

Assuming a constant pure decode iteration time of 40 ms:

| Prefill occupancy | Effective TPOT |
|---|---|
| 0% | 40 ms |
| 20% | 50 ms |
| 50% | 80 ms |
| 60% | 100 ms |
| 80% | 200 ms |

The relationship is nonlinear.

As prefill occupancy approaches 100%, predicted TPOT grows without bound.

## 5. Connection to KV-Cache Reuse

Increasing KV-cache reuse reduces the amount of new prefill computation.

Holding arrival rate and decode iteration time constant:

\[
T_{\text{pre}}\downarrow
\Rightarrow
\lambda T_{\text{pre}}\downarrow
\Rightarrow
TPOT_{\text{real}}\downarrow
\]

Therefore, KV-cache reuse can benefit not only the incoming request's TTFT but also the TPOT of requests already decoding.

## 6. Model Assumptions and Limitations

This is an approximate steady-state service-time model.

It assumes:

- Prefill and decode consume disjoint portions of modeled service time.
- Arrival rate and average prefill cost are sufficiently stable.
- Decode iteration time can be characterized by batch size.
- Prefill occupancy remains below 100%.

The model does not independently capture burstiness, queueing delays, heterogeneous request costs, or detailed GPU execution overlap.

## 7. Engineering Takeaway

Effective latency depends on both intrinsic operation cost and the fraction of execution capacity available to that operation.

This provides a useful framework for analyzing interference in shared serving systems.

## Next

Derive Equation 2 using Little's Law, then connect Equations 1–3 to throughput, batching, and SLO-constrained scheduling.  

