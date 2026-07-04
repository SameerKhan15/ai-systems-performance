# Why have multiple heads?  
Imagine we are reading "The animal didn't cross the street because it was too tired."  

Different heads may naturally learn to focus on different aspects:  
Head 1: Which noun does "it" refer to?  
Head 2: Where is the verb?  
Head 3: Is this sentence negated?  
Head 4 What is the subject?  

Every head performs the same attention computation, but because they have different learned projection matrices,  
they produce different Queries, Keys, and Values, which leads them to attend to different patterns.  

# An analogy  
Imagine five detectives looking at the same crime scene. They all see the exact same scene. But each wears a different pair of glasses.  
* One pair highlights fingerprints  
* Another highlights footprints  
* Another highlights blood  
* Another highlights fibers  
The crime scene never changed. Only the lenses changed. Those lenses are exactly what `WQ`, `WK`, `WV` are.  

The only thing that changes between heads is the learned projection matrices.  
Let's use a real example.  

## Same input for every head  
Suppose the token embedding is  
`x = [2, 1, 3, 4]`  
Every head receives this same vector. Head 1 does not receive one part and Head 2 another. Both see the entire embedding.   

## Head 1 has its own learned matrices  
Head 1 learns  
`WQ₁`  
`WK₁`  
`WV₁`  

Suppose:  
WQ₁ =  
`[1 0]`  
`[0 1]`  
`[0 0]`  
`[0 0]`  
This is a 4×2 matrix.  
Compute 
Q1 = xWQ1 will produce `[1x4] [4x2] = [1x2]` matrix.  
`Q₁ = [2,1]`  
Notice something interesting. Head 1 reduced the token dimension from 4 to 2.  

## Head 2 has DIFFERENT weights  
Now suppose  
WQ₂ =  
`[0 0]`  
`[0 0]`  
`[1 0]`  
`[0 1]`  

Compute `Q2 = xWQ2`  
Now `Q₂ = [3,4]`  
Same input. Different learned projection. Different Query. Same is true for K and V.
So now we have  
Input  
      [2,1,3,4]  
  
          │  
          │  
     ┌────┴────┐  
     │         │  
  
 Head 1     Head 2  
 
Q=[2,1]    Q=[3,4]  

Neither head saw a different input. They simply learned different "views" of the same input. The same happens for K and V.  
Head 1 has  
`WK₁`  
`WV₁`  

Head 2 has  
`WK₂`  
`WV₂`  
All six matrices are independent.  
* So Head 1 may learn "I care about syntax."  
* Head 2 learns "I care about long-distance dependencies."  
No one programmed this. Training discovered it.  

# By-Hand Exercise  
Objective is to compute the entire "pipeline" of multi-headed attention mechanism, using a toy example:  
`Input Embeddings`  
      ↓  
`Q/K/V Projection`  
      ↓  
`Split into Heads`  
      ↓  
`Attention per Head`  
      ↓  
`Concatenate Heads`  
      ↓  
`Output Projection`  

We'll use:  
* `embed_dim = 4`
* `num_heads = 2`
* `head_dim = 2`  

We choose different, simple projection matrices for Head 1 and Head 2 (with small integers like 0s and 1s).  
That way, we can compute everything by hand, and actually see the two heads produce different Q, K, V vectors and different attention patterns.  
It will still be manageable on paper, but it will accurately reflect how real multi-head attention works. It will make the role of multiple heads much more intuitive.  

`Each token -> 4 numbers -> Q,K,V -> reshape -> 2 heads -> 2 numbers/head`  

## Step 1 — Input Embeddings  
Let's make the two tokens intentionally different.  
**Token 1**  
`x₁ = [1, 2, 0, 1]`  
**Token 2**  
`x₂ = [0, 1, 2, 1]`  

Therefore  
$$
X =
\begin{bmatrix}
1 & 2 & 0 & 1 \\
0 & 1 & 2 & 1
\end{bmatrix}
$$  
Notice each token has 4 features.  

## Head-specific Projection Matrices  
### Head 1  
A particular "view" of the embedding.  
$$
W_Q^{(1)} =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 0 \\
0 & 0
\end{bmatrix}
$$  
This means: Head 1 mostly pays attention to the first two embedding features.  
For simplicity, let:  
`WK₁ = WQ₁`  
`WV₁ = WQ₁`  
Right now we're only trying to understand heads.  

### Head 2  
Now Head 2 gets a completely different projection.  
$$
W_Q^{(2)} =
\begin{bmatrix}
0 & 0 \\
0 & 0 \\
1 & 0 \\
0 & 1
\end{bmatrix}
$$  
Notice Head 2 ignores the first two embedding features. Instead it extracts information from the last two. Again,  
`WK₂ = WQ₂`  
`WV₂ = WQ₂`  
Both heads receive `x = [1 2 0 1]` but they learn different projections. Think of them as wearing different glasses.  

### Compute Q for Every Head  
Let's compute Head 1 first.  
Token 1  
$$
Q_1^{(1)} = x_1 W_Q^{(1)}
$$  

Multiply  
`[1 2 0 1]`  
by  
$$
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 0 \\
0 & 0
\end{bmatrix}
$$  
Result  
Head1: `Q(token1) = [1,2]`  

Token 2  
`[0 1 2 1]`  
times the same matrix.  
Result  
Head1: `Q(token2) = [0,1]`  

Now Head 2  
Use  
$$
W_Q^{(2)} =
\begin{bmatrix}
0 & 0 \\
0 & 0 \\
1 & 0 \\
0 & 1
\end{bmatrix}
$$  

Token 1  
`[1 2 0 1]`  
Result  
Head2: `Q(token1) = [0,1]`  

Token 2  
`[0 1 2 1]`  
Result  
Head2: `Q(token2) = [2,1]` 

Both heads saw the same original embeddings, yet they produced different Queries:  
| Token   | Head 1 Q | Head 2 Q |  
| ------- | -------- | -------- |  
| Token 1 | `[1,2]`  | `[0,1]`  |  
| Token 2 | `[0,1]`  | `[2,1]`  |  
This is exactly what happens in a real transformer. Each head learned a different projection matrix.  

At this point, a major insight is:  
* We started with the same input embeddings  
* We applied different learned projection matrices for each head  
* The heads now have different representations of the same tokens  

In the next step, we'll compute the Keys and Values (using the same simple matrices for now), then perform the attention calculation (QKᵀ, scaling, softmax, weighted sum) for each head.  
That's where we will see the two heads attend differently, even though they started from the same tokens.  

One important reminder: because we deliberately chose  
`WK₁ = WQ₁`  
`WV₁ = WQ₁`  
 
`WK₂ = WQ₂`  
`WV₂ = WQ₂`  

for this first exercise, within each head we will have:  
`Q = K = V`
That is not generally true in a trained transformer. We are doing it only to keep this first full hand computation manageable.  

At this point, we have:  
Input embeddings:  
$$
X =
\begin{bmatrix}
1 & 2 & 0 & 1 \\
0 & 1 & 2 & 1
\end{bmatrix}
$$  

So:  
`Token 1: x₁ = [1, 2, 0, 1]`  
`Token 2: x₂ = [0, 1, 2, 1]`  

For Head 1, we already computed:  
$$
Q^{(1)} =
\begin{bmatrix}
1 & 2 \\
0 & 1
\end{bmatrix}
$$  

Because:  
`WK₁ = WQ₁`  
`WV₁ = WQ₁`
we also get:  
$$
K^{(1)} =
\begin{bmatrix}
1 & 2 \\
0 & 1
\end{bmatrix}
$$

$$
V^{(1)} =
\begin{bmatrix}
1 & 2 \\
0 & 1
\end{bmatrix}
$$  

For Head 2, we computed:  
$$
Q^{(2)} =
\begin{bmatrix}
0 & 1 \\
2 & 1
\end{bmatrix}
$$  
Therefore:  
$$
K^{(2)} =
\begin{bmatrix}
0 & 1 \\
2 & 1
\end{bmatrix}
$$

$$
V^{(2)} =
\begin{bmatrix}
0 & 1 \\
2 & 1
\end{bmatrix}
$$  
Now we have everything needed for attention.  

### Compute Head 1 attention scores  
The formula is:  
$$
S^{(1)} =
\frac{Q^{(1)}\left(K^{(1)}\right)^T}{\sqrt{d_h}}
$$  
where:  
`head_dim = d_h = 2`  
Therefore:  
$$
\sqrt{d_h} = \sqrt{2} \approx 1.414
$$  

First compute:  
$$
Q^{(1)} \left(K^{(1)}\right)^T
$$  
We have:  
$$
Q^{(1)} =
\begin{bmatrix}
1 & 2 \\
0 & 1
\end{bmatrix}
$$  
and:  
$$
\left(K^{(1)}\right)^T =
\begin{bmatrix}
1 & 0 \\
2 & 1
\end{bmatrix}
$$  
Multiply:  
$$
\begin{bmatrix}
1 & 2 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 \\
2 & 1
\end{bmatrix}
$$  
$$
Q^{(1)}\left(K^{(1)}\right)^T =
\begin{bmatrix}
5 & 2 \\
2 & 1
\end{bmatrix}
$$  
This matrix is extremely important. Its interpretation is:  
|                 | Key token 1 | Key token 2 |  
|-----------------|-------------|-------------|  
| Query token 1   | 5           | 2           |  
| Query token 2   | 2           | 1           |  

Each row asks:  
For this query token, how compatible am I with every key token?  

#### Scale the scores  
Divide by:  
$$
\sqrt{2} \approx 1.414
$$  
So:  
$$
S^{(1)} =
\begin{bmatrix}
5 / 1.414 & 2 / 1.414 \\
2 / 1.414 & 1 / 1.414
\end{bmatrix}
$$

Approximately:

$$
S^{(1)} =
\begin{bmatrix}
3.536 & 1.414 \\
1.414 & 0.707
\end{bmatrix}
$$  

#### Apply causal mask  
Because we are modeling autoregressive generation:  
`Token 1 cannot see Token 2`  
`Token 2 can see Token 1 and Token 2`
So before softmax:  
$$
S_{\text{masked}}^{(1)} =
\begin{bmatrix}
3.536 & -\infty \\
1.414 & 0.707
\end{bmatrix}
$$  
This is the causal mask.  

#### Softmax Head 1  
Softmax is applied row by row.  
For Token 1:  
`[3.536, -∞]`  
Softmax gives:  
`[1.0, 0.0]`  
Of course: token 1 can only attend to itself.  

Now Token 2:  
`[1.414, 0.707]`  

Compute exponentials:  
$$
e^{1.414} \approx 4.113
$$

$$
e^{0.707} \approx 2.028
$$

Sum:

$$
4.113 + 2.028 = 6.141
$$

Therefore:

$$
\frac{4.113}{6.141} \approx 0.670
$$

$$
\frac{2.028}{6.141} \approx 0.330
$$

So Head 1 attention weights are:

$$
A^{(1)} =
\begin{bmatrix}
1 & 0 \\
0.670 & 0.330
\end{bmatrix}
$$ 

Interpretation:  
Head 1:

Token 1 attends:  
    100% → Token 1  

Token 2 attends:  
     67% → Token 1  
     33% → Token 2  

#### Multiply by Values  
Attention output formula:

$$
O^{(1)} = A^{(1)}V^{(1)}
$$

We have:

$$
A^{(1)} =
\begin{bmatrix}
1 & 0 \\
0.670 & 0.330
\end{bmatrix}
$$

and:

$$
V^{(1)} =
\begin{bmatrix}
1 & 2 \\
0 & 1
\end{bmatrix}
$$  

Multiply.

Token 1 output:

$$
1[1, 2] + 0[0, 1]
$$

Therefore:

$$
o_1^{(1)} = [1, 2]
$$

Token 2 output:

$$
0.670[1, 2] + 0.330[0, 1]
$$

First dimension:

$$
0.670(1) + 0.330(0) = 0.670
$$

Second dimension:

$$
0.670(2) + 0.330(1)
$$

$$
= 1.340 + 0.330 = 1.670
$$

Therefore:

$$
o_2^{(1)} = [0.670, 1.670]
$$

Full Head 1 output:

$$
O^{(1)} =
\begin{bmatrix}
1 & 2 \\
0.670 & 1.670
\end{bmatrix}
$$  

### Compute Head 2 attention scores  
This is where the multi-head idea becomes visible.  
Recall:

$$
Q^{(2)} = K^{(2)} = V^{(2)} =
\begin{bmatrix}
0 & 1 \\
2 & 1
\end{bmatrix}
$$

Compute:

$$
Q^{(2)}\left(K^{(2)}\right)^T
$$  

$$
Q^{(2)}\left(K^{(2)}\right)^T =
\begin{bmatrix}
1 & 1 \\
1 & 5
\end{bmatrix}
$$  
Already notice the difference.  
Head 1 produced:

$$
\begin{bmatrix}
5 & 2 \\
2 & 1
\end{bmatrix}
$$

Head 2 produced:

$$
\begin{bmatrix}
1 & 1 \\
1 & 5
\end{bmatrix}
$$
Same original tokens. Different learned projections. Completely different similarity patterns.  

#### Scale Head 2  
Divide by:

$$
\sqrt{2}
$$

Therefore:

$$
S^{(2)} =
\begin{bmatrix}
0.707 & 0.707 \\
0.707 & 3.536
\end{bmatrix}
$$  

#### Apply causal mask  
$$
S_{\text{masked}}^{(2)} =
\begin{bmatrix}
0.707 & -\infty \\
0.707 & 3.536
\end{bmatrix}
$$  

#### Softmax Head 2  
$$
A^{(2)} =
\begin{bmatrix}
1 & 0 \\
0.056 & 0.944
\end{bmatrix}
$$

Interpretation:  
Head 2:  

Token 1 attends:  
    100% → Token 1  
 
Token 2 attends:  
      5.6% → Token 1  
     94.4% → Token 2  
Compare this with Head 1:  
Head 1, Token 2:  

67% → Token 1  
33% → Token 2  
This is the essence of multi-head attention.  

#### Head 2 weighted Values  
$$
O^{(2)} = A^{(2)}V^{(2)}
$$

We have:

$$
V^{(2)} =
\begin{bmatrix}
0 & 1 \\
2 & 1
\end{bmatrix}
$$

Token 1:

$$
1[0, 1] + 0[2, 1]
$$

Therefore:

$$
o_1^{(2)} = [0, 1]
$$

Token 2:

$$
0.056[0, 1] + 0.944[2, 1]
$$

First dimension:

$$
0.056(0) + 0.944(2) = 1.888
$$

Second dimension:

$$
0.056(1) + 0.944(1) = 1.000
$$

Therefore:

$$
o_2^{(2)} = [1.888, 1.000]
$$  

Full Head 2 output:  
$$
O^{(2)} =
\begin{bmatrix}
0 & 1 \\
1.888 & 1
\end{bmatrix}
$$  

### Additional explanation of the computation of the weighted values  
For  
$$
V^{(2)} =
\begin{bmatrix}
0 & 1 \\
2 & 1
\end{bmatrix}
$$
The rows correspond to tokens, and the columns correspond to the 2 dimensions of the Value vector for Head 2:  
 
|         | Value dim 1 | Value dim 2 |  
|---------|-------------|-------------|  
| Token 1 | 0           | 1           |  
| Token 2 | 2           | 1           |  

So:  
`Row 1 = Token 1's Value vector for Head 2 = [0, 1]`  
`Row 2 = Token 2's Value vector for Head 2 = [2, 1]`  
And this is why, when computing the Head 2 output for Token 2, we used its attention weights:  
$$
[0.056,\ 0.944]
$$

to combine the two rows of $V^{(2)}$:

$$
0.056
\underbrace{[0, 1]}_{\text{Token 1's value}}
+
0.944
\underbrace{[2, 1]}_{\text{Token 2's value}}
$$

giving:

$$
[1.888,\ 1.000]
$$  

So the mental model is:  
Rows = tokens; columns = dimensions/features within that head.  
This row/token convention applies to $Q^{(2)}$, $K^{(2)}$, and $V^{(2)}$ in our example.  

### Concatenate the heads  
Now we concatenate Head 1 and Head 2 outputs.  
For Token 1:  
`Head 1: [1, 2]`  
`Head 2: [0, 1]`  

Concatenate:  
`[1, 2, 0, 1]`  

For Token 2:  
`Head 1: [0.670, 1.670]`  
`Head 2: [1.888, 1.000]`  

Concatenate:  
[0.670, 1.670, 1.888, 1.000]  

Therefore:  
$$
O_{\text{concat}} =
\begin{bmatrix}
1 & 2 & 0 & 1 \\
0.670 & 1.670 & 1.888 & 1
\end{bmatrix}
$$  

## The additional trained matrix Wo?  
In our toy model:  
`embed_dim = 4`  
`num_heads = 2`  
`head_dim = 2`  

After attention, each head produces 2 dimensions:  
`Head 1 output = [a, b]`  
`Head 2 output = [c, d]`  

We concatenate them:  
`O_concat = [a, b, c, d]`  

Then apply another learned matrix:  
$$
Y = O_{\text{concat}} W_O
$$

where:

$$
W_O \in \mathbb{R}^{4 \times 4}
$$  

So the full real architecture is:  
```text
Input X
   │
   ├──→ learned WQ ──→ Q
   ├──→ learned WK ──→ K
   └──→ learned WV ──→ V

          ↓

    split/reshape into heads

          ↓

    attention per head

          ↓

    concatenate head outputs

          ↓

      learned WO

          ↓

     final MHA output  
```
### Why do we need Wo?  
This is subtle and important. Before Wo, the output dimensions are still arranged as separate head results:  
`[ Head 1 information | Head 2 information ]`  

For our Token 2:  
`[0.670, 1.670 | 1.888, 1.000]`  

A learned Wo can mix information across heads.  
For example, suppose:  
$$
W_O =
\begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}
$$  

Then:  
$$
[0.670,\ 1.670,\ 1.888,\ 1.000] W_O
$$
produces a new 4-dimensional vector whose dimensions combine information from different head outputs.  

So conceptually:  
Wq, Wk, Wv let the heads develop different views of the input; Wo learns how to recombine those views into the model’s embedding space.  

Wo is also trained through backpropagation along with Wq, Wk, Wv., and the rest of the transformer.  

