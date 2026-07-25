# Hand Calculations: Prefill Latency vs. Prompt Length

## 1. What prefill means

Suppose the prompt contains `T` tokens. During **prefill**, the transformer receives the entire prompt in one forward pass:

```text
Token 1, Token 2, ..., Token T
```

For one attention-only transformer layer, the input tensor has shape:

$$
X \in \mathbb{R}^{B \times T \times D}
$$

where:

- $B$ = batch size
- $T$ = prompt length
- $D$ = embedding dimension
- $H$ = number of attention heads
- $D_h = D/H$ = dimension of each head

This lab uses causal attention. A token can attend to itself and earlier tokens, but not later tokens.

---

## 2. Projection work

The layer performs four square projections:

```text
Q = XWq
K = XWk
V = XWv
O = Attention(Q, K, V)Wo
```

Each weight matrix has shape:

$$
[D, D]
$$

For one projection, the output has $B \times T \times D$ elements. Each output element is a dot product of length $D$.

Therefore, one projection costs:

$$
B \times T \times D \times D
$$

MACs.

There are four projections: Q, K, V, and O. Thus:

$$
\boxed{\text{Projection MACs} = 4BTD^2}
$$

For fixed $B$ and $D$, projection work grows linearly with prompt length:

$$
O(T)
$$

---

## 3. Attention-score work

After splitting into heads:

$$
Q, K, V \in \mathbb{R}^{B \times H \times T \times D_h}
$$

The attention-score operation is:

$$
QK^T
$$

For each batch item and head, this produces a $T \times T$ score matrix. Every score is a dot product of length $D_h$.

Therefore:

$$
\text{Score MACs} = BHT^2D_h
$$

Because $HD_h = D$:

$$
\boxed{\text{Score MACs} = BT^2D}
$$

---

## 4. Attention-value work

After softmax, the second attention matrix multiplication is:

$$
\operatorname{softmax}(QK^T)V
$$

It has the same MAC count as score calculation:

$$
\boxed{\text{Value MACs} = BT^2D}
$$

Therefore, total attention matrix-multiplication work is:

$$
\boxed{\text{Attention MACs} = 2BT^2D}
$$

For fixed $B$ and $D$, attention work grows quadratically with prompt length:

$$
O(T^2)
$$

---

## 5. Total work for one attention-only layer

Combining projection and attention work:

$$
\boxed{\text{Total MACs} = 4BTD^2 + 2BT^2D}
$$

This equation contains two different scaling terms:

```text
Projection term: 4BTD²  → linear in T
Attention term:  2BT²D  → quadratic in T
```

At shorter prompts, the projection term may dominate. At sufficiently long prompts, the quadratic attention term dominates.

---

## 6. Where projection and attention work are equal

Set the two terms equal for batch size 1:

$$
4TD^2 = 2T^2D
$$

Divide both sides by $2TD$:

$$
2D = T
$$

Therefore:

$$
\boxed{T_{\text{crossover}} = 2D}
$$

For `embed_dim = 512`:

$$
T_{\text{crossover}} = 2 \times 512 = 1024
$$

At approximately 1,024 tokens, projection and attention MACs are equal in this simplified one-layer model.

This is a theoretical compute crossover, not a guarantee that measured latency will split exactly the same way. Kernel efficiency, memory traffic, launch overhead, and hardware utilization also matter.

---

## 7. Logical attention-score memory

The logical attention-score tensor has shape:

$$
[B, H, T, T]
$$

Therefore, it contains:

$$
BHT^2
$$

scalars.

If every scalar consumes `dtype_bytes` bytes:

$$
\boxed{\text{Score bytes} = BHT^2 \times \text{dtype bytes}}
$$

For fixed batch size, head count, and dtype, logical score memory grows quadratically:

$$
O(T^2)
$$

The lab's `manual` backend physically materializes this score matrix. An optimized SDPA or FlashAttention-style kernel may avoid materializing the entire matrix, even though the logical attention relationship remains $T \times T$.

---

## 8. Numerical example

Use:

```text
batch_size = 1
embed_dim  = 512
num_heads  = 8
head_dim   = 64
dtype      = float16 = 2 bytes
```

### Prompt length T = 512

Projection MACs:

$$
4 \times 1 \times 512 \times 512^2
= 536{,}870{,}912
$$

Attention MACs:

$$
2 \times 1 \times 512^2 \times 512
= 268{,}435{,}456
$$

Total:

$$
805{,}306{,}368\ \text{MACs}
$$

Logical score bytes:

$$
1 \times 8 \times 512^2 \times 2
= 4{,}194{,}304\ \text{bytes}
= 4\ \text{MiB}
$$

### Prompt length T = 1024

Projection MACs:

$$
4 \times 1024 \times 512^2
= 1{,}073{,}741{,}824
$$

Attention MACs:

$$
2 \times 1024^2 \times 512
= 1{,}073{,}741{,}824
$$

At $T=1024$, the two values are equal because $T=2D$.

Total:

$$
2{,}147{,}483{,}648\ \text{MACs}
$$

Logical score bytes:

$$
1 \times 8 \times 1024^2 \times 2
= 16{,}777{,}216\ \text{bytes}
= 16\ \text{MiB}
$$

### Prompt length T = 2048

Projection MACs:

$$
4 \times 2048 \times 512^2
= 2{,}147{,}483{,}648
$$

Attention MACs:

$$
2 \times 2048^2 \times 512
= 4{,}294{,}967{,}296
$$

Total:

$$
6{,}442{,}450{,}944\ \text{MACs}
$$

Logical score bytes:

$$
1 \times 8 \times 2048^2 \times 2
= 67{,}108{,}864\ \text{bytes}
= 64\ \text{MiB}
$$

---

## 9. What happens when prompt length doubles?

Replace $T$ with $2T$.

Projection work:

$$
4B(2T)D^2 = 2 \times 4BTD^2
$$

Therefore, projection work doubles.

Attention work:

$$
2B(2T)^2D = 4 \times 2BT^2D
$$

Therefore, attention work quadruples.

Logical score memory also quadruples because it contains a $T \times T$ dimension.

```text
When T doubles:

Projection MACs:  2×
Attention MACs:   4×
Score memory:     4×
```

---

## 10. Connecting the formulas to measured latency

The experiment measures median and p95 prefill latency over 30 runs for each prompt length.

A power law is fitted to the medians:

$$
\text{Latency} \approx cT^\alpha
$$

Interpretation:

- $\alpha \approx 1$: approximately linear scaling
- $\alpha \approx 2$: approximately quadratic scaling
- $1 < \alpha < 2$: both projection and attention terms, plus hardware effects, are visible

Do not expect the empirical exponent to exactly equal the theoretical operation-count exponent. Measured runtime also depends on:

- GPU kernel launch overhead
- matrix size and tensor-core utilization
- memory bandwidth
- kernel fusion
- causal masking implementation
- dtype
- batch size
- thermal and system noise

The goal is to explain the shape of the curve—not to force latency to match MAC counts perfectly.