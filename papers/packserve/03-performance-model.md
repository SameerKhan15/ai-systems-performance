## Mental Model: Per-Request Execution vs. Aggregate GPU Execution

### 1. Individual Request Lifecycle

Each LLM inference request consists of two phases:

1. **Prefill:** Processes the input prompt, constructs the KV cache, and produces the first output token.
2. **Decode:** Generates the remaining output tokens autoregressively.

For a request generating \(N\) output tokens, our simplified model assumes one prefill operation followed by \(N-1\) decode iterations.

### 2. Continuous Request Arrivals

In a production serving system, multiple requests share the same GPU.

As existing requests transition from prefill to decode, new requests continue arriving and introduce additional prefill work.

Meanwhile, requests that finish generating their output tokens leave the system.

In statistical steady state, assuming the system is stable:
$\text{Average arrival rate}=\text{Average completion rate}$  

The GPU continuously services a mixture of prefill and decode operations belonging to different requests.

### 3. Aggregate Prefill Occupancy

Let:

- $(\lambda)$: Average request arrival rate (requests/second).
- $(T_{\text{pre}})$: Average prefill service time (seconds/request).

The aggregate prefill workload introduced per second is:  
$\lambda T_{\text{pre}}$  

For example:  
$\lambda=3\text{ requests/sec}$   
$T_{\text{pre}}=0.1\text{ sec/request}$

Therefore:  
$\lambda T_{\text{pre}}=3(0.1)=0.3$  

Under the simplified non-overlapping service model, prefill consumes 30% of GPU service capacity, leaving 70% available for decoding.

### 4. Why Existing Requests Experience Higher TPOT

Consider Request A, which has already completed prefill and is generating output tokens.

While Request A decodes, Requests B, C, and D arrive and execute their prefill operations.

Request A does not perform additional prefill work. However, its decode iterations can be delayed because the GPU is servicing prefill operations belonging to other requests.

Consequently, its effective wall-clock TPOT increases even though its intrinsic decode execution cost may remain unchanged.

### 5. Connection to Equation 1

The effective decode service fraction is:  
$f_{\text{decode}}=1-\lambda T_{\text{pre}}$  

If the intrinsic decode iteration time is $T_{\text{iter}}(b)$, then:  
$TPOT_{\text{real}} = T_{\text{iter}} / f_{\text{decode}}$  
$
{ TPOT_{\text{real}}\approx \frac{T_{\text{iter}}(b)} {f_{\text{decode}}}
= \frac{T_{\text{iter}}(b)} {1-\lambda T_{\text{pre}}} }
$  

The equation relates intrinsic decode execution time to the fraction of wall-clock service capacity available for decoding.

### 6. Modeling Assumptions

This interpretation assumes:

- Prefill and decode occupy disjoint portions of modeled GPU service time.
- The system operates in a stable statistical steady state.
- Average arrival rate and prefill cost are representative.
- Decode batch size and iteration time can be approximated as stable.
- Prefill utilization remains below 100%.

Actual serving systems may interleave, chunk, or partially overlap prefill and decode operations. Resource contention and scheduling variability can therefore cause deviations from this approximation.

### Key Engineering Insight

**The prefill work of newly arriving requests can increase the TPOT of requests already decoding.**

Equation 1 captures this cross-request interference by modeling the GPU's aggregate prefill occupancy rather than the prefill cost of any individual decoding request.  

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

