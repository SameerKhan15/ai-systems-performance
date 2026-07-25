# Hand-worked example of how KV Cache optimizes memory and compute for inference 
The confusing part is usually this:  
Without cache at step 5:  
* Q has 5 rows  
* K has 5 rows  
So `Q @ K^T` produces a `5 × 5` score matrix.  

With KV cache at step 5:
* Q has only 1 row, for the newest token
* K_cache has 5 rows
So `Q_new @ K_cache^T` produces a 1 × 5 score row

## Tiny hand example  
Suppose for one attention head:  
`head_dim = 2`  
`num_heads = 1`  

We have 3 tokens:  
`Token 1 = The`  
`Token 2 = cat`  
`Token 3 = sat`  

Assume after projection we got these Q and K vectors:  
`Q1 = [1, 0]`  
`Q2 = [0, 1]`  
`Q3 = [1, 1]`  
  
`K1 = [1, 2]`  
`K2 = [3, 1]`  
`K3 = [2, 2]`  

So:  
$$
Q =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}
$$

where:

| Token | Query vector |
|---|---|
| Token 1 | $[1, 0]$ |
| Token 2 | $[0, 1]$ |
| Token 3 | $[1, 1]$ |

$$
K =
\begin{bmatrix}
1 & 2 \\
3 & 1 \\
2 & 2
\end{bmatrix}
$$

where:

| Token | Key vector |
|---|---|
| Token 1 | $[1, 2]$ |
| Token 2 | $[3, 1]$ |
| Token 3 | $[2, 2]$ |  

Now attention scores are:  
`Q @ K^T`  
Meaning every query compares against every key.  

### Without cache: full-prefix attention  
Recompute all Qs and all Ks:  
`Q1, Q2, Q3`  
`K1, K2, K3`  

The score matrix is:  
|      | K1      | K2      | K3      |  
|------|---------|---------|---------|  
| Q1   | Q1 · K1 | Q1 · K2 | Q1 · K3 |  
| Q2   | Q2 · K1 | Q2 · K2 | Q2 · K3 |  
| Q3   | Q3 · K1 | Q3 · K2 | Q3 · K3 |  

Note: For causal/autoregressive attention, a token is never allowed to attend to a future token.  
Therefore, the model may either still calculate the entire matrix (given above) and then mask the future tokens OR skip that compute entirely.  

So the actual full causal attention pattern is:  
|      | K1  | K2  | K3  | K4  | K5  |  
|------|-----|-----|-----|-----|-----|  
| Q1   | yes | no  | no  | no  | no  |  
| Q2   | yes | yes | no  | no  | no  |  
| Q3   | yes | yes | yes | no  | no  |  
| Q4   | yes | yes | yes | yes | no  |  
| Q5   | yes | yes | yes | yes | yes |  

So these are forbidden:  
```text
Q1 against K5
Q2 against K5
Q3 against K5
Q4 against K5
```
Because that would mean older tokens are looking into the future.
Think of each row as asking:  
`What tokens is this query position allowed to look at?`  
So:  
`Q1 = token 1 asking: what can I attend to?`  
`Answer: only token 1.`

`Q2 = token 2 asking: what can I attend to?`  
`Answer: token 1 and token 2.`  

`Q3 = token 3 asking: what can I attend to?`  
`Answer: token 1, token 2, token 3.`  

`Q5 = token 5 asking: what can I attend to?`  
`Answer: token 1, token 2, token 3, token 4, token 5.`  
The attention flow is past-to-current, not future-to-past.  

### Now the key KV-cache insight:  
For token#5, we only need the output for that particular token.  
Because token 5’s output is what we use to predict token 6.  

```text
The cat sat on the
                 ↑
          use this final position to predict next token
```
We are not trying to re-predict what came after token 1, token 2, token 3, or token 4. Those steps already happened.  
So during cached decoding:  
```text
Need:        Q5 against K1, K2, K3, K4, K5

Do not need: Q1, Q2, Q3, Q4 rows
```  
Now here is the subtle but beautiful point.  
Suppose we recompute the whole prefix at step 5 without cache. Because of the causal mask, the old rows produce the same results they already produced earlier.  
For example, Q2 at step 2 saw:  
`K1, K2`  
At step 5, Q2 is still only allowed to see:  
`K1, K2`  
It is not allowed to see:  
`K3, K4, K5`  
So recomputing Q2 at step 5 gives you the same result as before. That is wasted work.  
Same for Q1, Q3, and Q4.  

So without cache at step 5, the naive full-prefix computation contains lots of unnecessary work:  
|      | K1   | K2   | K3   | K4   | K5   |  
|------|------|------|------|------|------|  
| Q1   | old  | mask | mask | mask | mask |  
| Q2   | old  | old  | mask | mask | mask |  
| Q3   | old  | old  | old  | mask | mask |  
| Q4   | old  | old  | old  | old  | mask |  
| Q5   | need | need | need | need | need |   

The only row we actually need for next-token prediction is:  
|      | K1   | K2   | K3   | K4   | K5   |  
|------|------|------|------|------|------|  
| Q5   | need | need | need | need | need |  

That is why cached decoding computes only:  
$$
Q_5 \times K_{\text{cache}}^T
$$  

Another way to say it:  
`Old tokens do **not** get updated when new tokens arrive.`  
`New tokens look **back** at old tokens.`  
`Old tokens do **not** look **forward** at new tokens.`  
That is the causal/autoregressive rule.  

So K5 is useful, but not for Q1, Q2, Q3, or Q4.  
K5 is useful for:  
```text
Q5 now
Q6 later
Q7 later
Q8 later
...
```
That is why we cache K5 and V5.  
But Q1 is not useful later, because no future token attends using Q1. Future tokens use their own Q:  
```text
Q6 attends to K1..K6
Q7 attends to K1..K7
Q8 attends to K1..K8
```
So the cache stores keys and values from old tokens because future queries need them.  
It does not store old queries because future tokens do not need old queries.  

**The above is true for both training and inference in a causal/autoregressive Transformer.**  

But it shows up differently in each case.  

**During training**  
Training often processes the whole sequence at once:  
`The cat sat on the mat`  

So the model may build full Q/K/V matrices for all positions:  
```text
Q1 Q2 Q3 Q4 Q5 Q6
K1 K2 K3 K4 K5 K6
V1 V2 V3 V4 V5 V6
``` 
If we allowed full attention, then Q1 could see K5, which would leak future information.  
That would be cheating, because token 1 should not know token 5 yet.  
So training uses a causal mask:  
|      | K1  | K2  | K3  | K4  | K5  | K6  |  
|------|-----|-----|-----|-----|-----|-----|  
| Q1   | yes | no  | no  | no  | no  | no  |  
| Q2   | yes | yes | no  | no  | no  | no  |  
| Q3   | yes | yes | yes | no  | no  | no  |  
| Q4   | yes | yes | yes | yes | no  | no  |  
| Q5   | yes | yes | yes | yes | yes | no  |  
| Q6   | yes | yes | yes | yes | yes | yes |  

So in training:  
$Q_1$ against $K_5$ is present in the full matrix shape, but it is **masked out**.  
Usually the score is set to -inf before softmax.  

**During inference without cache**  
If we implement inference naively by recomputing the whole prefix every step, then at step 5 you may again build:  
`Q1 Q2 Q3 Q4 Q5`  
`K1 K2 K3 K4 K5`  
The causal rule still applies:  
`Q1 cannot attend to K5`  
`Q2 cannot attend to K5`  
`Q3 cannot attend to K5`  
`Q4 cannot attend to K5`  
`Q5 can attend to K5`  
So yes, even during no-cache inference:  
`Q1 against K5 is not allowed.`  
The old rows are either masked or unnecessary.  

**During inference with KV cache**  
This is where the shape changes.  

At step 5, cached inference usually does not even compute:  
`Q1, Q2, Q3, Q4`  

It computes only:  
`Q5`  

And attends against:  
`K1, K2, K3, K4, K5`  

So the score matrix is just:  
|      | K1  | K2  | K3  | K4  | K5  |  
|------|-----|-----|-----|-----|-----|  
| Q5   | yes | yes | yes | yes | yes |  
There is no Q1 against K5 entry, because Q1 is not recomputed at all.  

So in cached inference:  
`Q1 against K5 is not allowed conceptually, and it is not computed physically.`  

The clean rule is:  
`In causal language modeling:`  
`A position can attend to itself and earlier positions.`  
`A position cannot attend to later positions.`  
That rule is true in both training and inference.  

The difference is implementation:  
`Training:`  
`Process whole sequence together, use causal mask.`  

`No-cache inference:`  
`May recompute whole prefix, still use causal mask.`  

`KV-cache inference:`  
`Only compute newest query, so future-looking rows never arise.`  

**One exception: this is specifically for causal/autoregressive models, like GPT-style decoder-only LLMs.   
It is not true for bidirectional encoder models like BERT, where tokens are allowed to attend both left and right.**  

### Quantifying Performance Optimization  
Without cache, at generation step t, we recompute the full prefix:  
`tokens 1..t`  
`Q1..Qt`  
`K1..Kt`  
`V1..Vt`  
`full attention over t tokens`  

So across all generation steps:  
`projection work ≈ 1 + 2 + 3 + ... + T`  
`attention work  ≈ 1² + 2² + 3² + ... + T²`  

That means:  
`projection work: O(T²)`  
`attention work:  O(T³)`  

Btw: even if a model does not compute attention scores for future tokens, the overall cost would still be quadratic.  
For 5 token example -  
descending series:  
$$
5 + 4 + 3 + 2 + 1 = 15
$$  
the formula is:  
$$
\frac{n(n+1)}{2}
$$  

More generally:  
$$
1 + 2 + 3 + \cdots + n = \frac{n(n+1)}{2}
$$  

This is directly relevant to the KV-cache / causal attention math we’ve been discussing.  
For example, if token 1 attends to 1 key, token 2 to 2 keys, ..., token 5 to 5 keys, the total number of query-key comparisons is: 15  

#### Formula Derivation  
For:

$$
1 + 2 + 3 + \cdots + n
$$

Call the sum:

$$
S = 1 + 2 + 3 + \cdots + n
$$

Write it again backward:

$$
S = n + (n - 1) + (n - 2) + \cdots + 1
$$

Now add the two rows:

$$
2S = (1 + n) + (2 + n - 1) + (3 + n - 2) + \cdots + (n + 1)
$$

Each pair equals:

$$
n + 1
$$

There are $n$ such pairs, so:

$$
2S = n(n + 1)
$$

Divide by 2:

$$
S = \frac{n(n + 1)}{2}
$$

So:

$$
\boxed{1 + 2 + 3 + \cdots + n = \frac{n(n + 1)}{2}}
$$  

#### Back to the 'Without cache: full-prefix attention toy example'  
At step 3 without cache, we recompute all Qs and all Ks:  
```text
Q1, Q2, Q3
K1, K2, K3
```
The score matrix is:  
| .... |    K1   |    K2   |    K3   |  
|------|---------|---------|---------|  
| Q1   | Q1 · K1 | Q1 · K2 | Q1 · K3 |  
| Q2   | Q2 · K1 | Q2 · K2 | Q2 · K3 |  
| Q3   | Q3 · K1 | Q3 · K2 | Q3 · K3 |  
Now compute each dot product.  
$$
Q_1 \cdot K_1 = [1,0] \cdot [1,2] = 1 \times 1 + 0 \times 2 = 1
$$

$$
Q_1 \cdot K_2 = [1,0] \cdot [3,1] = 1 \times 3 + 0 \times 1 = 3
$$

$$
Q_1 \cdot K_3 = [1,0] \cdot [2,2] = 1 \times 2 + 0 \times 2 = 2
$$  

So first row is:  
`[1, 3, 2]`  

Second row:  
$$
Q_2 \cdot K_1 = [0,1] \cdot [1,2] = 0 \times 1 + 1 \times 2 = 2
$$

$$
Q_2 \cdot K_2 = [0,1] \cdot [3,1] = 0 \times 3 + 1 \times 1 = 1
$$

$$
Q_2 \cdot K_3 = [0,1] \cdot [2,2] = 0 \times 2 + 1 \times 2 = 2
$$  

Second row:  
`[2, 1, 2]`  

Third row:  
$$
Q_3 \cdot K_1 = [1,1] \cdot [1,2] = 1 \times 1 + 1 \times 2 = 3
$$

$$
Q_3 \cdot K_2 = [1,1] \cdot [3,1] = 1 \times 3 + 1 \times 1 = 4
$$

$$
Q_3 \cdot K_3 = [1,1] \cdot [2,2] = 1 \times 2 + 1 \times 2 = 4
$$  

Third row:  
`[3, 4, 4]`  

So the full attention score matrix is:  
$$
QK^T =
\begin{bmatrix}
1 & 3 & 2 \\
2 & 1 & 2 \\
3 & 4 & 4
\end{bmatrix}
$$

|      | K1 | K2 | K3 |
|------|----|----|----|
| Q1   | 1  | 3  | 2  |
| Q2   | 2  | 1  | 2  |
| Q3   | 3  | 4  | 4  |

Shape:  
`[query_len, key_len] = [3, 3]`  

There are:  
`3 × 3 = 9 dot products`  

Each dot product has length:  
`head_dim = 2`  

So MACs:  
`9 × 2 = 18 MACs`  

Using the formula:  
`num_heads × query_len × key_len × head_dim`  
`= 1 × 3 × 3 × 2`  
`= 18 MACs`  

### With cache: only newest query  
Now suppose we are doing autoregressive decoding with KV cache.  
At step 3, we already cached:  
`K1, K2`  
`V1, V2`  

Now token 3 arrives. We compute only:  
`Q3, K3, V3`  

Then we append:  
`K_cache = [K1, K2, K3]`  
`V_cache = [V1, V2, V3]`  

The key moment is:  
`Q3 @ K_cache^T`  

Not:  
`[Q1, Q2, Q3] @ [K1, K2, K3]^T`  

Only the newest token needs a new output. So we only need the newest query.  
So the score row is:  
|      | K1      | K2      | K3      |  
|------|---------|---------|---------|  
| Q3   | Q3 · K1 | Q3 · K2 | Q3 · K3 |  

We already computed those above:  
`Q3 · K1 = 3`  
`Q3 · K2 = 4`  
`Q3 · K3 = 4`  

So with cache, the score output is only:  
`Q3 @ K_cache^T = [3, 4, 4]`  

Shape:  
`[query_len, key_len] = [1, 3]`  

There are:  
`1 × 3 = 3 dot products`  

Each dot product has length:  
`head_dim = 2`  

So MACs:  
`3 × 2 = 6 MACs`  

Using the formula:  
`num_heads × query_len × key_len × head_dim`  
`= 1 × 1 × 3 × 2`  
`= 6 MACs`  

So at this step:  
`without cache: 18 score MACs`  
`with cache:     6 score MACs`  

That is a 3× reduction at step 3.  

### Optimization numbers calculated on another example  
num_heads = 2  
head_dim = 2048  

At step 5, prefix length is:  
`The cat sat on the`  

So:  
`key_len = 5`  

#### Without cache  
Without cache, we recompute full-prefix attention.  
So:  
`query_len = 5`  
`key_len   = 5`  
`num_heads = 2`  
`head_dim  = 2048`  

**Projection:**  

The script uses one full projection matrix for each of Q, K, and V:

```text
X  = [5, 4096]
Wq = [4096, 4096]
Wk = [4096, 4096]
Wv = [4096, 4096]
```

For Q:

```text
Q = XWq = [5, 4096]
```

This output contains:

```text
5 × 4096 = 20,480 scalar output elements
```

Each scalar output is a dot product of length 4096. Therefore:

$$
5 \times 4096 \times 4096
= 83{,}886{,}080\ \text{MACs}
$$

So:

```text
Q projection = 83,886,080 MACs ≈ 83.89M
K projection = 83,886,080 MACs ≈ 83.89M
V projection = 83,886,080 MACs ≈ 83.89M
```

Total Q/K/V projection work is:

$$
3 \times 83{,}886{,}080
= 251{,}658{,}240\ \text{MACs}
\approx 251.66\text{M MACs}
$$

Equivalently, the 4096 output features can be viewed as two head-sized output blocks of 2048 features each. That per-head view produces the same total MAC count.

**Attention:**  

The script counts two attention matrix multiplications:

1. `Q @ K^T` to produce attention scores
2. `attention_weights @ V` to aggregate values

##### Attention-score MACs: `Q @ K^T`

For each head:

```text
Q head shape   = [5, 2048]
K head shape   = [5, 2048]
K^T shape      = [2048, 5]
score shape    = [5, 5]
```

There are:

```text
5 × 5 = 25 dot products per head
```

Each dot product has length:

```text
head_dim = 2048
```

Therefore, for one head:

```text
25 × 2048 = 51,200 MACs
```

For two heads:

```text
2 × 51,200 = 102,400 attention-score MACs
```

Using the script formula:

$$
\begin{aligned}
\text{score MACs}
&= \text{num\_heads} \times \text{query\_len} \times \text{key\_len} \times \text{head\_dim} \\
&= 2 \times 5 \times 5 \times 2048 \\
&= 102{,}400
\end{aligned}
$$

##### Attention-value MACs: `attention_weights @ V`

For each head:

```text
attention_weights shape = [5, 5]
V head shape             = [5, 2048]
output shape             = [5, 2048]
```

There are:

```text
5 × 2048 = 10,240 output elements per head
```

Each output element combines five values, so:

```text
10,240 × 5 = 51,200 MACs per head
```

For two heads:

```text
2 × 51,200 = 102,400 attention-value MACs
```

This matches the script formula:

$$
\begin{aligned}
\text{value MACs}
&= \text{num\_heads} \times \text{query\_len} \times \text{key\_len} \times \text{head\_dim} \\
&= 2 \times 5 \times 5 \times 2048 \\
&= 102{,}400
\end{aligned}
$$

Therefore, total attention work without cache at step 5 is:

$$
102{,}400 + 102{,}400 = 204{,}800\ \text{MACs}
$$

##### Output projection: `Wo`

The script also applies the output projection to all five prefix positions:

```text
merged attention output = [5, 4096]
Wo                      = [4096, 4096]
output                  = [5, 4096]
```

Therefore:

$$
5 \times 4096 \times 4096
= 83{,}886{,}080\ \text{MACs}
$$

##### Total work without cache at step 5

| Component | MACs |
|---|---:|
| Q projection | 83,886,080 |
| K projection | 83,886,080 |
| V projection | 83,886,080 |
| O projection | 83,886,080 |
| Attention scores: `Q @ K^T` | 102,400 |
| Attention values: `weights @ V` | 102,400 |
| **Total** | **335,749,120** |

So:

```text
Total projection MACs = 4 × 83,886,080
                      = 335,544,320

Total attention MACs  = 102,400 + 102,400
                      = 204,800

Total step MACs       = 335,544,320 + 204,800
                      = 335,749,120
                      ≈ 335.75M MACs
```

The attention score tensor contains:

```text
2 heads × 5 queries × 5 keys = 50 score elements
```

For `float32`, this uses:

```text
50 × 4 bytes = 200 bytes
```

#### With KV cache

At step 5, only the newest token is projected:

```text
query_len = 1
key_len   = 5
num_heads = 2
head_dim  = 2048
embed_dim = 4096
```

##### Q, K, and V projections

For each projection:

```text
X_new = [1, 4096]
W     = [4096, 4096]
result = [1, 4096]
```

Therefore, each projection costs:

$$
1 \times 4096 \times 4096
= 16{,}777{,}216\ \text{MACs}
$$

For Q, K, and V together:

$$
3 \times 16{,}777{,}216
= 50{,}331{,}648\ \text{MACs}
$$

##### Attention-score MACs: `Q5 @ K_cache^T`

For each head:

```text
Q5 head shape       = [1, 2048]
K_cache head shape  = [5, 2048]
score-row shape     = [1, 5]
```

There are five dot products per head, each of length 2048:

```text
5 × 2048 = 10,240 MACs per head
```

For two heads:

```text
2 × 10,240 = 20,480 attention-score MACs
```

Using the script formula:

$$
2 \times 1 \times 5 \times 2048
= 20{,}480\ \text{MACs}
$$

##### Attention-value MACs: `attention_weights @ V_cache`

The attention weights have shape `[1, 5]` per head and aggregate five cached value vectors, each of length 2048.

Therefore:

$$
2 \times 1 \times 5 \times 2048
= 20{,}480\ \text{MACs}
$$

Total attention work with cache at step 5 is:

$$
20{,}480 + 20{,}480
= 40{,}960\ \text{MACs}
$$

##### Output projection: `Wo`

Only the newest token's attention output is projected:

$$
1 \times 4096 \times 4096
= 16{,}777{,}216\ \text{MACs}
$$

##### Total work with KV cache at step 5

| Component | MACs |
|---|---:|
| Q projection | 16,777,216 |
| K projection | 16,777,216 |
| V projection | 16,777,216 |
| O projection | 16,777,216 |
| Attention scores: `Q_new @ K_cache^T` | 20,480 |
| Attention values: `weights @ V_cache` | 20,480 |
| **Total** | **67,149,824** |

So:

```text
Total projection MACs = 4 × 16,777,216
                      = 67,108,864

Total attention MACs  = 20,480 + 20,480
                      = 40,960

Total step MACs       = 67,108,864 + 40,960
                      = 67,149,824
                      ≈ 67.15M MACs
```

The attention score tensor contains:

```text
2 heads × 1 query × 5 keys = 10 score elements
```

For `float32`, this uses:

```text
10 × 4 bytes = 40 bytes
```

### Step-5 compute comparison

| Metric | Without cache | With KV cache | Improvement |
|---|---:|---:|---:|
| Q projected tokens | 5 | 1 | 5× less work |
| K projected tokens | 5 | 1 | 5× less work |
| V projected tokens | 5 | 1 | 5× less work |
| O projected tokens | 5 | 1 | 5× less work |
| Projection MACs | 335,544,320 | 67,108,864 | 5× lower; 80% reduction |
| Attention-score MACs | 102,400 | 20,480 | 5× lower; 80% reduction |
| Attention-value MACs | 102,400 | 20,480 | 5× lower; 80% reduction |
| Total attention MACs | 204,800 | 40,960 | 5× lower; 80% reduction |
| **Total MACs at step 5** | **335,749,120** | **67,149,824** | **5× lower; 80% reduction** |
| Attention score elements | 50 | 10 | 5× lower; 80% reduction |

A 5× reduction means the cached version performs:

$$
\frac{1}{5} = 20\%
$$

of the original work, or an:

$$
1 - \frac{1}{5} = 80\%
$$

reduction in MACs at step 5.

This exact 5× relationship is not accidental. At generation step $t$, this script's no-cache path does exactly $t$ times the projection and attention work of its cached path.

This is a MAC-count comparison, not a guarantee of exactly 5× lower wall-clock time; kernel-launch overhead, memory behavior, and implementation efficiency also affect measured runtime.

### Step-5 memory accounting from the script

Assume:

```text
dtype       = float32
bytes/value = 4
embed_dim   = 4096
num_heads   = 2
```

#### Without cache

The script retains no persistent KV cache:

```text
persistent KV cache = 0 bytes
```

Its temporary Q/K/V storage at this step is:

$$
3 \times 5 \times 4096 \times 4
= 245{,}760\ \text{bytes}
= 240\ \text{KiB}
$$

Its temporary attention-score storage is:

$$
2 \times 5 \times 5 \times 4
= 200\ \text{bytes}
$$

#### With KV cache

The active persistent K and V cache through token 5 is:

$$
2 \times 5 \times 4096 \times 4
= 163{,}840\ \text{bytes}
= 160\ \text{KiB}
$$

The leading factor of 2 represents:

```text
one K cache + one V cache
```

Temporary Q/K/V storage for only the new token is:

$$
3 \times 1 \times 4096 \times 4
= 49{,}152\ \text{bytes}
= 48\ \text{KiB}
$$

Temporary attention-score storage is:

$$
2 \times 1 \times 5 \times 4
= 40\ \text{bytes}
$$

| Memory recorded at step 5 | Without cache | With KV cache |
|---|---:|---:|
| Persistent active KV cache | 0 B | 163,840 B = 160 KiB |
| Temporary Q/K/V tensors | 245,760 B = 240 KiB | 49,152 B = 48 KiB |
| Temporary attention scores | 200 B | 40 B |

The cached path deliberately trades persistent linear KV-cache memory for much lower recomputation and smaller temporary tensors.

> The implementation preallocates storage for `max_steps`, but `persistent_kv_cache_bytes` records the logically active portion through the current step. At the final step of a five-token run, the allocated and active sizes are the same.

### Cumulative calculations across all five generation steps

The script's counters accumulate work across steps 1 through 5.

#### Useful sums

$$
1 + 2 + 3 + 4 + 5 = 15
$$

$$
1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 55
$$

#### Without cache: cumulative projections

At each successive step, the script projects 1, 2, 3, 4, and 5 tokens.

Therefore, each of Q, K, V, and O processes:

```text
15 projected token vectors
```

MACs for each projection type:

$$
15 \times 4096 \times 4096
= 251{,}658{,}240
$$

Across Q, K, V, and O:

$$
4 \times 251{,}658{,}240
= 1{,}006{,}632{,}960\ \text{projection MACs}
$$

#### Without cache: cumulative attention

The score work across the five steps is proportional to:

$$
1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 55
$$

Therefore:

$$
\begin{aligned}
\text{attention-score MACs}
&= 2 \times 2048 \times 55 \\
&= 225{,}280
\end{aligned}
$$

Attention-value MACs are the same:

$$
225{,}280
$$

Total cumulative attention MACs:

$$
225{,}280 + 225{,}280
= 450{,}560
$$

Total cumulative MACs without cache:

$$
1{,}006{,}632{,}960 + 450{,}560
= 1{,}007{,}083{,}520
$$

#### With KV cache: cumulative projections

The cached path projects exactly one new token at each of five steps.

Therefore, each of Q, K, V, and O processes:

```text
5 projected token vectors
```

MACs for each projection type:

$$
5 \times 4096 \times 4096
= 83{,}886{,}080
$$

Across Q, K, V, and O:

$$
4 \times 83{,}886{,}080
= 335{,}544{,}320\ \text{projection MACs}
$$

#### With KV cache: cumulative attention

At steps 1 through 5, the newest query attends to 1, 2, 3, 4, and 5 cached keys.

Therefore:

$$
\begin{aligned}
\text{attention-score MACs}
&= 2 \times 2048 \times (1+2+3+4+5) \\
&= 2 \times 2048 \times 15 \\
&= 61{,}440
\end{aligned}
$$

Attention-value MACs are the same:

$$
61{,}440
$$

Total cumulative attention MACs:

$$
61{,}440 + 61{,}440
= 122{,}880
$$

Total cumulative MACs with KV cache:

$$
335{,}544{,}320 + 122{,}880
= 335{,}667{,}200
$$

### Cumulative five-step counter summary

| Metric | Without cache | With KV cache | Reduction |
|---|---:|---:|---:|
| Q projected token-count | 15 | 5 | 66.7% |
| K projected token-count | 15 | 5 | 66.7% |
| V projected token-count | 15 | 5 | 66.7% |
| O projected token-count | 15 | 5 | 66.7% |
| Projection MACs | 1,006,632,960 | 335,544,320 | 66.7% |
| Attention-score MACs | 225,280 | 61,440 | 72.7% |
| Attention-value MACs | 225,280 | 61,440 | 72.7% |
| Attention score elements | 110 | 30 | 72.7% |
| **Total MACs** | **1,007,083,520** | **335,667,200** | **66.7%** |
| Final persistent KV cache | 0 B | 163,840 B | additional persistent memory |

The cumulative total-work speedup is:

$$
\frac{1{,}007{,}083{,}520}{335{,}667{,}200}
\approx 3.00024\times
$$

So over the complete five-step run, KV caching performs approximately one-third of the total MACs, which is approximately a 66.7% reduction.

The attention-only cumulative speedup is larger:

$$
\frac{450{,}560}{122{,}880}
= 3.6667\times
$$

or a 72.7% reduction in attention MACs.

The total is very close to 3× rather than 3.67× because, at these dimensions and this short sequence, the large projection matrices dominate the total operation count.

### General formulas represented by the script

Let:

```text
D  = embed_dim
H  = num_heads
Dh = head_dim
t  = current generation step
T  = total number of generation steps
```

Since:

$$
H \times D_h = D
$$

#### Work at a single step `t`

Without cache:

$$
\text{projection MACs} = 4tD^2
$$

$$
\text{attention MACs} = 2Ht^2D_h = 2t^2D
$$

$$
\boxed{\text{total no-cache MACs at step }t = 4tD^2 + 2t^2D}
$$

With KV cache:

$$
\text{projection MACs} = 4D^2
$$

$$
\text{attention MACs} = 2H(1)(t)D_h = 2tD
$$

$$
\boxed{\text{total cached MACs at step }t = 4D^2 + 2tD}
$$

The ratio is exactly:

$$
\frac{4tD^2 + 2t^2D}{4D^2 + 2tD}
= t
$$

Therefore, for this implementation:

$$
\boxed{\text{step-}t\text{ compute speedup from KV cache} = t\times}
$$

#### Work across all steps `1..T`

Without cache:

$$
\text{projection MACs}
= 4D^2\sum_{t=1}^{T}t
= 4D^2\frac{T(T+1)}{2}
= O(T^2)
$$

$$
\text{attention MACs}
= 2D\sum_{t=1}^{T}t^2
= 2D\frac{T(T+1)(2T+1)}{6}
= O(T^3)
$$

With KV cache:

$$
\text{projection MACs}
= 4D^2T
= O(T)
$$

$$
\text{attention MACs}
= 2D\sum_{t=1}^{T}t
= 2D\frac{T(T+1)}{2}
= O(T^2)
$$

Thus KV caching changes the cumulative decoding work in this lab from:

```text
Projections: O(T²) → O(T)
Attention:   O(T³) → O(T²)
```

It does not make attention constant-time, because each new query must still compare with all previously cached keys.

#### Active KV-cache memory at step `t`

For one layer and batch size 1:

$$
\boxed{\text{KV bytes} = 2 \times t \times D \times \text{dtype bytes}}
$$

The factor of 2 is for the K cache plus the V cache. Therefore, active KV-cache memory grows linearly with sequence length:

$$
O(T)
$$

