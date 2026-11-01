# Lab 1: Build a A-MEM Write Path  
## Objective  
Build one complete, real A-MEM memory insertion:  
`Interaction -> Metadata (via LLM) -> Embedding (via Embedding Model) -> ANN (via Vector DB) -> Link Reasoning (via LLM) -> Graph`  

Tech stack:  
* A100 80 GB  
* Python  
* local ~8B instruct LLM  
* BGE embedding model  
* Qdrant  (Vector DB)  
* Neo4j (Graph DB)  

By the end, we should be able to insert an interaction and inspect the resulting memory in both *Qdrant* and *Neo4j*.  

## Keep the first version extremely small  
1. No concurrency  
2. No batching  
3. No million-memory dataset  
4. No Nsight  
5. No optimization  
6. No memory evolution yet  

Start with perhaps 20 - 50 manually understandable seed memories.  

That lets us inspect results and answer:  
`Did the system actually create sensible semantic relationships`  

Performance comes in the later versions of the lab.  

## The Write Path  
Suppose out input is:  
`During autoregressive decoding, KV caching avoids recomputing keys and values for previously generated tokens.`  

### Step1: Metadata generation  
Send the interaction to the local LLM.  
Require structured output:  
````
{
  "keywords": [
    "KV cache",
    "autoregressive decoding",
    "keys",
    "values"
  ],
  "tags": [
    "LLM inference",
    "transformers"
  ],
  "description":
    "Explains how KV caching avoids recomputation
     of historical key and value tensors during
     autoregressive decoding."
}
````
And we record `T_metadata`  

### Step2: Construct canonical memory text  
Construct:  
````
Interaction:
During autoregressive decoding...

Keywords:
KV cache, autoregressive decoding, keys, values

Tags:
LLM inference, transformers

Description:
Explains how KV caching avoids...
````

Call this $X_i$  

````
Raw interaction = historical event

Canonical memory text = representation used for semantic retrieval
````

### Step3: Generate the embedding  
$e_i = BGE(X_i)$  
Print the following during this lab, for manual study 
````
Embedding shape: [1024]
dtype: float32
device: cuda:0
L2 norm: 1.000
````
`text -> token -> transformer -> vector`  
Measure:  
`T_measure`  

### Step4: Query Qdrant BEFORE inserting  
We have `e_new` but it is not in the vector database yet.  
Ask Qdrant:  
`TopK(e_new)`  

For Lab 1:  
`k=5`  

Return:  
````
Candidate     similarity

M17           0.89
M4            0.84
M11           0.78
M2            0.71
M9            0.66
````

Measure:  
`T_ANN`  

*Print candidates memory text. We want to examine whether the embedding model's choices make semantic sense.*  

### Step5: Let the LLM decide links  
Now construct a second prompt:  
````
NEW MEMORY

<new memory>


CANDIDATES

M17
<candidate text>

M4
<candidate text>

...


Determine which memories have a meaningful
conceptual relationship to the new memory.
````

Require structured output:  
````
{
  "links": [
    {
      "memory_id": "M17",
      "reason":
        "Both describe KV caching during
         autoregressive decoding."
    },
    {
      "memory_id": "M4",
      "reason":
        "The candidate discusses the memory
         consequences of KV caching."
    }
  ]
}
````

Measure:  
`T_link`  

### Step6: Now write the memory to Qdrant  
Qdrant gets something conceptually like:  
````
ID
embedding
interaction
keywords
tags
description
timestamp
````

This becomes our vector-side representation:  
````
             QDRANT

M101 ─── embedding
  │
  ├── interaction
  ├── keywords
  ├── tags
  ├── description
  └── timestamp
````

Measure:  
`T_vector-write`  

### Step7: Create the graph node  
Now create:  
````
(:Memory {
    id: "M101",
    timestamp: ...
})
````

Keep the graph DB primarily responsible for relationships.  
````
(M101)-[:RELATED_TO {
    reason: "..."
}]->(M17)
````
and:  
````
(M101)-[:RELATED_TO {
    reason: "..."
}]->(M4)
````

Now physically:  
````
                 M17
                  ▲
                  │
                M101
                  │
                  ▼
                 M4
````
Measure:  
`T_graph-write`  

### Observability  
````
Memory insertion M101

Metadata LLM       74.2 ms
Embedding           8.7 ms
ANN                  3.1 ms
Link LLM            81.6 ms
Vector write         2.8 ms
Graph write          4.2 ms
---------------------------
Pipeline            177.9 ms
````
And calculate:  
`T_accounted = Summation over i for T_i`  versus actual `T_wall`  

The difference:  
`T_overhead = T_wall - T_accounted`  is itself interesting.  

### Seed Data  
````
Transformer attention
KV cache
prefill
decode
MHA
MQA
GQA
PagedAttention
GPU memory
HBM bandwidth
batching
continuous batching
CUDA execution
SMs
warps
occupancy
shared memory
````

This is much better than random Wikipedia text because we can independently judge whether A-MEM created a sensible relationship.  

## Lab 1 validation  
1. Storage correctness: Can we retrieve the exact memory from Qdrant by ID?  
2. Vector correctness: Does ANN return semantically plausible candidates?  
3. LLM correctness: Does the linking model reject superficially similar but conceptually unrelated memories and accept meaningful relationships?  
4. Graph correctness: Starting at the new memory, can Neo4j return exactly the relationships selected by the LLM?  

Then perform the most important validation manually:  
`Draw the resulting graph and ask whether it makes conceptual sense.`  
With only 30–50 nodes, we can actually inspect it.  

## Repository structure  
````
amem-performance-lab/
│
├── README.md
├── requirements.txt
├── config/
│
├── data/
│   └── seed_memories.json
│
├── src/
│   ├── llm.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── graph_store.py
│   ├── memory.py
│   └── pipeline.py
│
├── experiments/
│   └── lab01_write_path.py
│
├── results/
│   └── lab01/
│
└── docs/
    └── lab01.md
````

