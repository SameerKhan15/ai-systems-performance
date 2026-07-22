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

For each head:  
**Projection:**  
X = [5,4096] 
Wq = [4096,2048]  
Wk = [4096,2048]  
Wv = [4096,2048]  

Q = XWq = [5,2048]  
This is 5 x 2048 = 10,240 dot products per head  
Each dot product length is: 4096  
So per head:  
10,240 × 4096 = 41.9M MACs  

For 2 heads, if each head has its own Wq:  
`2×41.9M = ~83M MACs`  

**~83M Q-projection MACs total**  

Same calculation would apply to K and V projections.  
`Hence, total Q/K/V projection MACs = 3 × ~83.9M = ~251.7M MACs`    

**Attention:**  
