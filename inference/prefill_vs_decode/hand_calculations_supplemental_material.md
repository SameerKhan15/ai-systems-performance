# What “prefill” means  
Suppose the user provides this prompt:  
`The cat sat on the mat`  
Assume the tokenizer produces six tokens:  
`Token 1: The`  
`Token 2: cat`  
`Token 3: sat`  
`Token 4: on `  
`Token 5: the`    
`Token 6: mat`  
So:  
`T = 6`  

During **prefill**, the model processes all six known prompt tokens together:  
`[The, cat, sat, on, the, mat]`  

This is different from the later decode phase, where the model generates one new token at a time.  

**Prefill:**  
process the entire existing prompt  

**Decode:**  
generate token 7  
generate token 8  
generate token 9  
...  

Prefill has two important jobs:  
1. Compute the transformer representations for the prompt  
2. Populate the KV cache with the keys and values for all prompt tokens  

After prefill, the model uses the final prompt position to predict the first generated token.  

# Before X: token IDs  
The model does not initially receive words or strings. It receives integer token IDs.  
For example:  
`"The" → 101`  
`"cat" → 205`  
`"sat" → 417`  
`"on"  → 88`  
`"the" → 33`  
`"mat" → 902`  

The token-ID tensor might be:  
`input_ids = [[101, 205, 417, 88, 33, 902]]`  

Its shape is:  
`[B,T]=[1,6]`  
because:  
* There is one prompt in the batch  
* The prompt has six tokens
Token IDs are just integers. They are not yet the X tensor.  

# Each token ID becomes an embedding vector  
The embedding layer converts every token ID into a vector of D real numbers.  
Suppose:  
`D = 4`  
Then each token becomes a vector with four components.  
For example:  
`The → [ 0.2, -0.1,  0.7,  0.4]`  
`cat → [ 0.8,  0.3, -0.2,  0.1]`  
`sat → [-0.4,  0.9,  0.5, -0.3]`  
`on  → [ 0.1,  0.6, -0.7,  0.2]`  
`the → [ 0.3, -0.2,  0.8,  0.5]`  
`mat → [-0.1,  0.4,  0.2,  0.9]`  
Stacking those token vectors gives:  
$$
X =
\begin{bmatrix}
0.2 & -0.1 & 0.7 & 0.4 \\
0.8 & 0.3 & -0.2 & 0.1 \\
-0.4 & 0.9 & 0.5 & -0.3 \\
0.1 & 0.6 & -0.7 & 0.2 \\
0.3 & -0.2 & 0.8 & 0.5 \\
-0.1 & 0.4 & 0.2 & 0.9
\end{bmatrix}
$$
Ignoring the batch dimension for a moment, this matrix has shape:  
`[T,D]=[6,4]`  
There are:  
* Six rows: one per token  
* Four columns: one per embedding feature  

# Why the full shape is [B,T,D]  
Deep-learning frameworks normally preserve a batch dimension, even when the batch contains only one prompt.  
Therefore, the actual tensor has shape:  
$$
X \in \mathbb{R}^{B \times T \times D}
$$
For our example:  
`B=1, T=6, D=4`  

$$
X \in \mathbb{R}^{1 \times 6 \times 4}
$$

In PyTorch:  
`X.shape`  
`# torch.Size([1, 6, 4])`  

Conceptually, the tensor looks like:  
```
X = [  
    [   # Batch item 0  
        [ 0.2, -0.1,  0.7,  0.4],  # Token 0: The  
        [ 0.8,  0.3, -0.2,  0.1],  # Token 1: cat  
        [-0.4,  0.9,  0.5, -0.3],  # Token 2: sat  
        [ 0.1,  0.6, -0.7,  0.2],  # Token 3: on  
        [ 0.3, -0.2,  0.8,  0.5],  # Token 4: the  
        [-0.1,  0.4,  0.2,  0.9],  # Token 5: mat  
    ]  
]  
```
There are three levels of brackets:  
```
batch  
  sequence of tokens    
    vector components  
```

# Meaning of each axis  
For:  
`X[b,t,d]`  
the indices mean:  
* b: which prompt in the batch  
* t: which token position in that prompt  
* d: which embedding component  

For example:  
`X[0, 2, :]`  
means:  
`Give me the entire embedding vector`  
`for token position 2`  
`in batch item 0`  

In our example, that is the token `sat`:  
`[-0.4, 0.9, 0.5, -0.3]`  

Meanwhile:  
`X[0, 2, 1]`  

means:  
`Batch item 0`  
`Token position 2`  
`Embedding component 1`  

The result is one scalar: 0.9  

A useful mental model is:  
`X[which prompt, which token, which feature]`  

# What does $\mathbb{R}$ mean? 
The notation:  
$$
X \in \mathbb{R}^{B \times T \times D}
$$
X is a tensor whose entries are real-valued numbers.  

For example:  
`0.2`  
`-0.1`  
`0.7`  
`0.4`  

In an actual model, these may be represented as:  
* `float32`  
* `float16`  
* `bfloat16`  
The notation does not mean that the tensor is one enormous real number. It means that every cell in the B×T×D structure contains a real-valued scalar.  

# Example with multiple prompts  
Suppose the batch contains two prompts:  
`Prompt 1: The cat sat`  
`Prompt 2: A dog ran`  
Assume:  
`B=2`, `T=3`, `D=4`  
Then:  
$$
X \in \mathbb{R}^{2 \times 3 \times 4}
$$
Conceptually:  
```
X = [
    [   # Prompt 1
        embedding_of("The"),
        embedding_of("cat"),
        embedding_of("sat"),
    ],

    [   # Prompt 2
        embedding_of("A"),
        embedding_of("dog"),
        embedding_of("ran"),
    ],
]
```
The shape is:  
```
2 prompts  
× 3 tokens per prompt  
× 4 numbers per token  
```
or:  
`X.shape == [2, 3, 4]`  

# What does the embedding dimension D represent?  
The embedding dimension is the number of numerical features used to represent each token.  
In our toy example:  
`D=16`  
Each token is represented by a vector of 16 numbers:  
```
The → [x₁, x₂, x₃, ..., x₁₆]
cat → [x₁, x₂, x₃, ..., x₁₆]
sat → [x₁, x₂, x₃, ..., x₁₆]
```
For a six-token prompt:  
`X shape=[1,6,16]`  

The model does not assign a simple human-readable meaning to every dimension such as:  
`dimension 1 = noun`  
`dimension 2 = animal`  
`dimension 3 = plural`  
Instead, the representation is distributed. Meaning is encoded through combinations of many dimensions.  

# How X becomes Q, K, and V  
The attention layer applies three learned projection matrices:  
$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$
Suppose:  
`X:[B,T,D]`  
and each projection matrix has shape:  
$$
W_Q, W_K, W_V : [D, D]
$$
Then:  
$$
Q,K,V:[B,T,D]  
$$
The projection changes the numbers but not the overall tensor shape.  
For example:   
`X: [1, 6, 16]`  
`WQ: [16, 16]`  
`WK: [16, 16]`  
`WV: [16, 16]`  
produces:  
`Q: [1, 6, 16]`  
`K: [1, 6, 16]`  
`V: [1, 6, 16]`  
Every one of the six token vectors is projected independently through the same learned matrix.  

# Where the attention heads appear  
Suppose:  
`D=16`  
and:  
`H=2`  
Then each head has dimension:  
$$
D_h = \frac{D}{H} = \frac{16}{2} = 8
$$
Initially:  
`Q:[B,T,D]`  
so in our lab:  
`Q:[1,6,16]`  
It is then reshaped into separate heads:  
$$
Q : [B, H, T, D_h]
$$
Therefore:  
`Q:[1,2,6,8]`  
The total amount of information has not changed:  
`2×8=16`  
So:  
```  
Before splitting:
1 batch × 6 tokens × 16 dimensions  

After splitting:  
1 batch × 2 heads × 6 tokens × 8 dimensions per head  
```
The same transformation happens to K and V.  

# Why prefill can process all tokens simultaneously  
If the model is causal, how can it process the entire prompt simultaneously?  
The answer is that it computes representations for all tokens in parallel, but applies a causal mask to attention.  

For four prompt tokens:  
`The cat sat down`  
the allowed attention pattern is:  

| Query   | Token 1 | Token 2 | Token 3 | Token 4 |  
|---------|--------:|--------:|--------:|--------:|  
| Token 1 | Yes      | No      | No      | No      |  
| Token 2 | Yes      | Yes     | No      | No      |  
| Token 3 | Yes      | Yes     | Yes     | No      |  
| Token 4 | Yes      | Yes     | Yes     | Yes     |  

All four query rows may be calculated in one large operation, but forbidden future positions are masked before softmax.  
Thus:  
`Token 1 cannot see tokens 2–4`  
`Token 2 cannot see tokens 3–4`  
`Token 3 cannot see token 4`  
`Token 4 can see tokens 1–4`  
Parallel computation does not violate causality because the mask prevents information from flowing backward from future positions.  

# The attention-score tensor during prefill  
After splitting into heads:  
$$
Q : [B, H, T, D_h]
$$

$$
K : [B, H, T, D_h]
$$
Attention computes:  
$$
QK^T
$$
The resulting shape is:  
`[B,H,T,T]`  
For:  
$$
B=1, H=2, T=6, D_h=6  
$$ 
the score tensor has shape:  
`[1,2,6,6]`  
Why `6×6`?  
Because each of the six query positions compares against each of the six key positions:  
`6 queries × 6 keys`  
This happens independently for each of the two heads.  
The upper-right portion is then causally masked.  

## Additional notes on attention-score matrix operations  
After splitting into heads:  
$$
Q \in \mathbb{R}^{[B,H,T,D_h]}
$$

$$
K \in \mathbb{R}^{[B,H,T,D_h]}
$$
Attention scores are computed as:  
$$
S = QK^{\mathsf{T}}
$$
The transpose applies to the last two dimensions of K:  
$$
K^{\mathsf{T}} \in \mathbb{R}^{[B,H,D_h,T]}
$$
Now consider the matrix multiplication independently for every batch item and every head:  
$$
[T, D_h] \times [D_h, T] = [T, T]
$$
The inner D_h dimensions are multiplied and summed over, leaving two T dimensions:  
$$
[B, H, T, D_h] \times [B, H, D_h, T]
\rightarrow
[B, H, T, T]
$$
Each dimension means:  
$$
\boxed{
[B, H, T_{\text{query}}, T_{\text{key}}]
}
$$
* B: batch items  
* H: attention heads  
* First T: which token is issuing the query  
* Second T: which token’s key that query is compared against  

For one batch item and one head, the result is a T×T matrix:  

| Query \ Key | Token 1 | Token 2 | Token 3 | Token 4 |  
| ------------| -------:| -------:| -------:| -------:|  
| **Token 1** |  score  |  score  |  score  |  score  |  
| **Token 2** |  score  |  score  |  score  |  score  |  
| **Token 3** |  score  |  score  |  score  |  score  |  
| **Token 4** |  score  |  score  |  score  |  score  |  

Every cell is a dot product:
$$
\[
S_{b,h,i,j} = Q_{b,h,i,:} \cdot K_{b,h,j,:}
\]
$$
For batch item b and head h, how strongly does query token i match key token j  
For example:  
`Q:[2,8,128,64]`  
`K^⊤:[2,8,64,128]`  

Therefore:  
`QK⊤:[2,8,128,128]`  
The output has one 128×128 attention-score matrix for each of the 8 heads in each of the 2 batch items.  

# Why process old prompt positions if only the last one predicts the next token?  
For the first generated token, the model normally uses the final prompt position.  
However, the earlier prompt positions must still be processed because:  
1. Their keys and values provide context to later tokens  
2. Their keys and values must be stored in the KV cache  
3. The final prompt token must attend to representations derived from all prior prompt tokens  

At the end of prefill, each layer has cache tensors approximately shaped as:  
`K_cache:[B,H,T,Dh]`  
`V_cache:[B,H,T,Dh]`

For your toy configuration:  
`K_cache: [1, 2, 6, 8]`  
`V_cache: [1, 2, 6, 8]`  
These contain the keys and values for all six prompt tokens.  

# What changes during decode?  
During prefill:  
`X:[B,T,D]`  
For example:  
`[1, 6, 16]`  

During the next decode step, only one new token is processed:  
`Xnew:[B,1,D]`  
For example:  
`[1, 1, 16]`  
The new token produces:  
`Q_new: [1, 2, 1, 8]`  
`K_new: [1, 2, 1, 8]`  
`V_new: [1, 2, 1, 8]`  

The new key and value are appended to the cache:  
**Before decode:**  
`K_cache = [1, 2, 6, 8]`  

**After adding one token:**  
`K_cache = [1, 2, 7, 8]`  

The new query attends to all seven cached keys:  
`Q_new * K^T_cache`  

with score shape:  
`[1,2,1,7]`  

That is the key shape difference:  
**Prefill attention:**  
`[B, H, T, T]`  

**Decode attention:**  
`[B, H, 1, current_cache_length]`  

# Compact mental model  
Think of X as a collection of token vectors:  

Batch  
└── Prompt  
    ├── Token 1 → D numbers  
    ├── Token 2 → D numbers  
    ├── Token 3 → D numbers  
    └── ...  

Therefore:  
`X:[B,T,D]`  

means:  
````
B prompts  
× T tokens per prompt  
× D numbers representing each token  
````
For your lab configuration:  
````
B = 1  
T = prompt length  
D = 16  
H = 2  
Dh = 8  
````
So for a six-token prompt:  
````
X         [1, 6, 16]  
Q/K/V     [1, 6, 16]  
Q/K/V     [1, 2, 6, 8] after splitting into heads  
scores    [1, 2, 6, 6]  
K cache   [1, 2, 6, 8]  
V cache   [1, 2, 6, 8]  
````

The most important interpretation is:  
`X[b,t,:] is the complete D-dimensional representation of token position t in prompt b.`  



