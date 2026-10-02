# A-MEM  
Ordinary RAG stores documents. A-MEM tries to build an evolving network of memories.  

## What is the problem A-MEM is trying to solve  
Imagine an AI agent that interacts with you repeatedly.  
Over time it sees interactions such as:  
* "The user prefers deep technical explanations."  
* "The user learns GPU architecture by working from first principles."  
* "The user prefers hands-on experiments after understanding the theory."  

A simple memory system could store these as three independent chunks in a vector database.  

When a new query arrives:  
`Query → embedding → vector similarity → top-k chunks`  
That's basically memory implemented as conventional RAG.  

But humans don't seem to remember things as completely isolated chunks. One memory activates related memories.  
You might think:
````
GPU architecture
      ↓
performance engineering
      ↓
memory hierarchy
      ↓
shared memory
      ↓
bank conflicts
````
One thought leads to another.  

A-MEM tries to give an agent something closer to that interconnected structure.  

## Zettelkasten gives A-MEM its conceptual model  
The idea says:  
`Don't build giant notes. Build small ideas and connect them.`  

Suppose you're studying GPU architecture.  
Instead of:  
````
GPU Architecture Notes
-----------------------
50 pages of everything I learned
````
you create:  
````
Note 101
GPU warps contain 32 threads.

Note 102
An SM maintains multiple resident warps.

Note 103
Warp schedulers select ready warps.

Note 104
Occupancy measures resident warps relative
to the hardware maximum.
````
Each note represents approximately one idea.  
That's atomicity.  

But then you add links:  
````
101 → 102
102 → 103
102 → 104
103 → 104
````
Now we don't merely have notes. We have knowledge graph-like structure.  

That gives us the first **mental model**:  
````
Atomic knowledge
       +
Connections between knowledge
       =
Network of knowledge
````
A-MEM applies essentially this idea to an agent's experiences.  

## What is an "atom" in A-MEM?  
This is particularly important.

In A-MEM, the atomic unit isn't necessarily a fact like:  
`"A100 has 108 SMs."`  
The atomic unit is an interaction/experience.  

Imagine this agent interaction:  
````
User:
Explain why increasing prompt length makes
attention increasingly expensive.

Assistant:
Because QK^T creates a T × T attention matrix...
````
A-MEM turns that interaction into a memory note.  
Conceptually:  
````
Memory M17

Original interaction:
    User asked about attention scaling...

Timestamp:
    2026-08-24

Keywords:
    attention, prompt length,
    quadratic scaling, QK^T

Tags:
    transformers, performance

Description:
    Discussion explaining why attention
    computation grows quadratically with
    sequence length.
````
Notice something interesting.  
Only one piece came directly from the environment:  
**the original interaction.**  

The LLM manufactures the additional semantic metadata:  
````
interaction
     │
     ├── LLM → keywords
     ├── LLM → tags
     └── LLM → contextual description
````
So A-MEM isn't simply recording memory.  
It is interpreting the experience before storing it.  
That's an important agentic element.  

## Now we need embeddings  
Suppose our memory contains:  
````
interaction
keywords
tags
description
````
A-MEM concatenates these.  

Conceptually:  
````
TEXT =
interaction
+ keywords
+ tags
+ contextual description
````
Then: 
`e=Embedding(TEXT)`  
where \(e\) might be something like a 1,536-dimensional vector:  
`e=[0.17,−0.42,0.08,…,0.31]`  
The exact dimensionality depends on the embedding model.  

The point is that the entire memory now gets represented as a location in semantic vector space.  

So memories about:  
````
attention scaling
transformer performance
quadratic complexity
long context
````
should lie relatively close together.  

## Here is where A-MEM becomes more interesting than ordinary vector RAG  
Suppose a new memory M_new arrives.  
We calculate:  
$e_{\text{new}} = Embed(M_{\text{new}})$  
Then compare it against existing memories.  

For example, using cosine similarity:  
$
sim(e_{\text{new}}, e_i)
=
\frac{e_{\text{new}} \cdot e_i}
{\lVert e_{\text{new}} \rVert \lVert e_i \rVert}
$  

Suppose the database contains 10,000 memories.  
We retrieve the most similar:  
````
New memory
    │
    │ vector similarity search
    ▼
┌────────────┐
│ Memory 23  │  0.91
│ Memory 91  │  0.87
│ Memory 17  │  0.84
│ Memory 66  │  0.81
└────────────┘
````
So far this is basically *normal vector search*.  

**But A-MEM adds another step**  

## Similarity does NOT automatically mean "create a link"  
This distinction is important.  

The embedding model says:  
`These memories appear semantically similar`  
But similarity alone doesn't tell us whether there is a **meaningful conceptual relationship**.  

So A-MEM asks an LLM to examine the candidates.  
For example:  
````
New memory:
Long prompts increase attention computation
quadratically.

Candidate memory:
Attention scores form a T × T matrix.

Should these memories be linked?
````
The LLM might decide:  
````
YES

Reason:
The T × T attention matrix explains the
quadratic scaling described in the new memory.
````
So we have two stages:  
`Vector search → candidate relationships`  
followed by  
`LLM reasoning→actual relationships`  
This is one reason it is called **agentic memory** rather than merely vector retrieval.  

## Now the memory system starts becoming a graph  
Suppose we have memories:  
````
M1: Self-attention computes QK^T

M2: QK^T produces a T × T matrix

M3: Attention therefore has quadratic
    sequence-length complexity

M4: KV caching avoids recomputing historical
    keys and values during decoding

M5: Prefill and decoding have different
    computational characteristics
````
A-MEM might create:  

             M1
              │
              ▼
             M2
              │
              ▼
             M3
              │
              ▼
             M5
              │
              ▼
             M4
But there could also be cross-links:  
````
M2 ─────────→ M5
M3 ─────────→ M4
````
Eventually:  

      M1 ───── M2
       │      / │
       │     /  │
       ▼    ▼   ▼
      M3 ───── M5
       │       /
       │      /
       ▼     ▼
         M4
Now the system contains **two forms of structure simultaneously:**  

**Vector space**  
$Mi -> ei$  
and **explicit graph relationships**  
$Mi → Mj$  

That's a powerful mental model for A-MEM:  
**A-MEM is approximately a vector memory store with an LLM-maintained relationship graph layered on top**  

## The "evolutionary" part is especially interesting  
Suppose an older memory says:  
````
Memory M20

Keywords:
cache, transformer

Tags:
LLM

Description:
Discussion about caching transformer state.
````
Months later, the agent learns much more:  
````
Memory M183

Interaction:
KV caching stores previously computed K and V
vectors during autoregressive decoding.

Keywords:
KV cache, autoregressive decoding,
keys, values

Tags:
transformer inference, performance
````
A-MEM may realize:  
`M20 ↔ M183`  
But it can do something beyond simply creating the link.  
The new memory provides **context for understanding the old memory.**  
So M20's metadata might evolve:  
````
BEFORE

keywords:
cache, transformer

tags:
LLM

description:
Caching transformer state.
````
to:  
````
AFTER

keywords:
KV cache, transformer,
autoregressive decoding

tags:
LLM inference, performance

description:
Memory concerning caching transformer
key/value state during autoregressive inference.
````
The historical event itself hasn't changed.  
But the system's **interpretation of the event has changed.**  
That is what it means for **A-MEM to be an ever-evolving memory database**  

## Retrieval therefore becomes richer than top-k vector search  
Suppose the user asks:  
`Why does KV caching improve decoding performance?`  

Normal RAG might do:  
````
query
  ↓
embedding
  ↓
vector search
  ↓
M183
````
And return the three nearest chunks.  
A-MEM can instead do:  
````
query
   ↓
vector search
   ↓
M183
   ↓
follow memory links
   ↓
M20 ─ M4 ─ M5 ─ M91
````
This means:  
`Retrieval can start semantically and then expand structurally`  

That's a very useful distinction.  
We can think of it as:  
`retrieved context=semantic neighbors+linked conceptual neighbors`  
The second set might contain useful memories that weren't especially close to the query embedding.  

## Why might graph traversal find something vector search misses?  
Imagine:  
`A → B → C`  
where:  
* A: Long-context inference is expensive  
* B: Long context increases KV-cache memory requirements  
* C: KV-cache pressure can cause memory-bandwidth bottlenecks  

Your query might be:  
`Why is long-context inference expensive?`  
Vector search strongly retrieves A.  
Maybe B is somewhat similar.  
But C might have relatively weak direct embedding similarity to the query because it talks mostly about memory bandwidth.  

Yet the graph knows:  
`A → B → C`  
Therefore retrieval can discover C through the chain.  
This is similar to associative human recall:  
````
long context
    ↓
KV cache
    ↓
GPU memory
    ↓
bandwidth
````
One memory activates another.  

## Now compare ordinary RAG and A-MEM conceptually  
| Ordinary RAG                    | A-MEM                                       |
| ------------------------------- | ------------------------------------------- |
| Stores chunks                   | Stores atomic experiences                   |
| Embeds chunks                   | Embeds enriched memories                    |
| Similarity determines retrieval | Similarity helps both linking and retrieval |
| Chunks mostly independent       | Memories explicitly connected               |
| Metadata often static           | Metadata can evolve                         |
| Retrieval = nearest chunks      | Retrieval can expand through links          |
| Mostly passive storage          | LLM actively organizes memory               |

This doesn't mean A-MEM replaces RAG.  
It's better understood as an agentic extension of the RAG idea for long-term memory.  

## There's actually a beautiful systems architecture hiding underneath  
Strip away the terminology and we get roughly:  
````
                NEW INTERACTION
                      │
                      ▼
              ┌────────────────┐
              │ Memory Creator │
              │      LLM       │
              └────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       keywords      tags     description
          │           │           │
          └───────────┼───────────┘
                      ▼
                 concatenate
                      │
                      ▼
                  embedding
                      │
                      ▼
               VECTOR SEARCH
                      │
                      ▼
               candidate memories
                      │
                      ▼
                ┌──────────┐
                │   LLM    │
                │ linking  │
                └──────────┘
                      │
                      ▼
              MEMORY GRAPH
             /      │       \
           M1 ───── M2 ───── M3
                      │
                      ▼
                update metadata
````
Then retrieval runs approximately in reverse:  
````
USER QUERY
    │
    ▼
embedding
    │
    ▼
semantic retrieval
    │
    ▼
seed memories
    │
    ▼
follow links
    │
    ▼
expanded context
    │
    ▼
AGENT / LLM
````

## The deepest idea to retain  
A-MEM is not fundamentally about embeddings.  
Embeddings already existed in RAG.  
And it's not fundamentally about storing conversation history. Agents already did that.  

The deeper idea is:  
`Memory should be organized continuously as experience accumulates`  

A new experience doesn't merely get appended:  
````
M1
M2
M3
M4
M5
````
Instead, the system asks:  
````
What does M5 mean?

What older memories does M5 relate to?

Does M5 change how I understand M2?

Should M2's metadata evolve?

What network of knowledge is emerging?
````
So the memory store gradually moves from a database of past interactions toward an organized model of accumulated experience.  

## Why do we need explicit links at all if embeddings already encode semantic relationships?  
That question gets us into the fundamental difference between vector similarity and graph relationships, and understanding it will make the architecture of A-MEM much more intuitive.  

### Start with a simple example  
Suppose we have three memories:  
````
M1: Increasing sequence length increases the size
    of the attention matrix.

M2: Attention creates a T × T score matrix.

M3: GPU memory bandwidth can become the bottleneck
    during autoregressive decoding.
````
Embedding similarity might give:  
`sim(M1,M2)=0.91`  
`sim(M1,M3)=0.48`  
That's perfectly reasonable. M1 and M2 use very similar concepts: attention, sequence length, matrices.  
But now imagine another memory:  
````
M4: Larger KV caches increase HBM traffic and can
    make decoding memory-bandwidth bound.
````
We might establish relationships:  
````
M1 → M2

M1 → M4 → M3
````
M1 and M3 might not be particularly similar linguistically or semantically, but there is an important reasoning chain connecting them:  
````
longer sequence
      ↓
larger KV cache
      ↓
more memory traffic
      ↓
memory-bandwidth bottleneck
````
That's something pure nearest-neighbor retrieval can have trouble preserving.  

### Think of embeddings as coordinates and links as roads  
This is probably the best mental model.  
An embedding gives every memory a coordinate:  
$M_i \rightarrow \mathbf{e}_i \in \mathbb{R}^d$  

So imagine:  
````
                     M7

          M2  M3

      M1

                              M8
                         M9
````
Distance tells us:  
`M2 and M3 are close`   
But it doesn't necessarily tell us why.  
A link adds information:  
`M2 ──explains──→ M3`  
or more generally in A-MEM:  
`M2 ──────────── M3`  
The system has decided that there is a relationship worth remembering.  
So:  
`embedding ≈ where is this memory semantically?`  
whereas:  
`link ≈ what other memories should I associate with it?`  

### There's another major difference: links persist  
Suppose today we calculate:  
`sim(M_a,M_b)=0.72`  

That's just the result of a similarity calculation.  

Tomorrow we search for something else and may never compare those memories.  

But if the LLM examines them and decides:  
`These two memories belong together`  

A-MEM can persist:  
`MA ↔ MB`  
Now the system has learned an association.  

It doesn't have to rediscover that relationship from scratch every time.  

This starts to resemble a graph:  
````
             M7
            /
M1 ── M2 ── M3
      │     │
      M4 ── M5
             \
              M8
````
The links become part of the memory system's accumulated structure.  

### And this changes retrieval  
Suppose the query \(Q\) is embedded:  
$e_q = Embed(Q)$  
Vector search finds:  
````
Q
│
▼
M2        ← very similar
````
Ordinary RAG might retrieve:  
````
M2
M7
M11
````
because those have the highest cosine similarities.  

A-MEM can say:  
````
Q
│
▼
M2
│
├──── M1
├──── M3
└──── M4
````
So **vector search gives us the entry point into memory.**  
The links let us explore the neighborhood around that memory.  

That's a very useful way to remember the architecture:  
`Query -> Vector Search -> Seed Memories -> Graph Expansion`  

### Why not just retrieve more vector neighbors?  
`Question: Instead of retrieving top-3, why not retrieve top-20?`  

Because you increase **recall at the expense of precision**  

**Precision = “Of what I returned, how much was actually relevant?”**  
**Recall = “Of everything relevant that existed, how much did I successfully find?”**  
Formally:  
$\text{Precision} = \frac{TP}{TP + FP}$  

$\text{Recall} = \frac{TP}{TP + FN}$  

Suppose a vector search system has 100 truly relevant documents in the database. It returns 20 documents, and 18 of those are relevant.  
Then:  
$\text{Precision} = \frac{18}{20} = 90\%$  
Very clean results—few irrelevant documents.  

But:  
$\text{Recall} = \frac{18}{100} = 18\%$  
It missed 82 relevant documents.  

So the intuition is:  
**Precision cares about false positives**  
**Recall cares about false negatives**  

By increasing recall at the expense of precision, we can start dumping lots of vaguely similar material into the LLM's context.  

Suppose:  
Top-10 vector results
````
M1  0.93  useful
M2  0.89  useful
M3  0.86  useful
M4  0.83  vaguely useful
M5  0.81  noise
M6  0.80  noise
M7  0.79  noise
...
````
Whereas graph expansion might retrieve:  
````
M1  ← semantic match
 │
 ├── M17   important consequence
 │
 └── M42   relevant previous experience
````
M42 might have only:  
`sim(Q,M42)=0.54`  
and therefore would never make top-k.  

But it may be highly useful because there's an established relationship:  
`M1 → M17 → M42`  
This is where the graph adds something that nearest-neighbor vector search doesn't naturally provide.  

### One subtle correction  
We shouldn't think:  
* embeddings = dumb  
* links = intelligent  
Embeddings themselves capture remarkably rich semantic relationships.  
Nor does A-MEM guarantee that explicit links are better. The LLM can create bad links, and graph expansion can introduce irrelevant memories.  

Rather, the two mechanisms are complementary:  
````
Embedding space
     │
     │ discovers likely relationships
     ▼
Candidate memories
     │
     │ LLM evaluates
     ▼
Explicit links
     │
     │ persist over time
     ▼
Memory graph
````

So vector similarity is effectively being used as a cheap candidate generator, while the LLM performs the more expensive semantic judgment.  

And that leads the most interesting part of A-MEM:  
`The graph isn't manually designed. It emerges as the agent accumulates experiences`  

## Worked Example: Simulate A-MEM from an empty database  
### Starting state: nothing exists  
At \(t=0\):  
````
MEMORY DATABASE
(empty)
````
There are no embeddings and no links. Then the agent starts interacting with the user.  

### Interaction 1  
`Attention becomes more expensive as sequence length increases because the attention matrix is T * T`  
A-MEM treats this **one interaction as one atomic memory**  

#### Step 1 — LLM enriches the interaction  
The LLM might generate:  
````
M1

Original:
Attention becomes more expensive as sequence
length increases because the attention matrix
is T × T.

Timestamp:
10:00

Keywords:
attention, sequence length, T×T matrix,
quadratic complexity

Tags:
transformers, attention, performance

Description:
Explains that attention computation scales
quadratically with sequence length.
````
Then A-MEM concatenates most of this:  
$X_1 = Original + Keywords + Tags + Description$  
and embeds it:  
$e_1 = Embed(X_1)$  
There are no previous memories, so there is nothing to link.  
Database:  
`M1`  

### Interaction 2  
Later the user says:  
$The (QK^T) multiplication produces the (T  \times T) attention score matrix.$  

A-MEM creates another atomic memory:  
````
M2

Keywords:
QK^T, attention scores, T×T matrix

Tags:
transformers, attention

Description:
Explains how QK^T creates the quadratic-sized
attention score matrix.
````
and:  
$e_2 = Embed(M2)$  
Now something new happens.  
A-MEM searches the existing memories using e  
There is currently only M1:  
$sim(e_2, e_1) = 0.91$  

So M1 becomes a candidate.  
The LLM examines:  
````
NEW MEMORY
M2: QK^T produces a T×T attention matrix.

CANDIDATE
M1: Attention becomes more expensive because
    the attention matrix is T×T.
````
The LLM determines these are meaningfully related.  
So we create:  
`M1 ←────→ M2`  

Conceptually:  
`M2 explains WHY M1 occurs.`  

Our memory system has now moved from a collection to a graph.  

### Interaction 3  
Now the user says:  
`During autoregressive decoding, KV caching prevents recomputation of keys and values for previous tokens.`  

A-MEM generates:  
````
M3

Keywords:
KV cache, keys, values, decoding,
autoregressive inference

Tags:
transformers, inference, optimization

Description:
KV caching reuses previously computed keys
and values during autoregressive decoding.
````

Then:  
$e_3=Embed(M3)$  

A similarity search might produce:  
````
Candidate     Similarity

M2              0.64
M1              0.57
````
Notice that these similarities aren't nearly as strong as M1 ↔ M2.  
The LLM examines them.  
It might decide:  
````
M3 and M2:
related through transformer attention.

M3 and M1:
not sufficiently direct.
````

Perhaps we end up with:  
`M1 ───── M2 ───── M3`  

Now something interesting is happening.  
M1 and M3 aren't especially similar:  
$sim(M1,M3)=0.57$  
But they are connected through M2:  
`M1→M2→M3`  

We've started accumulating structure that isn't represented simply by nearest-neighbor distance.  

### Interaction 4  
Now suppose the user says:  
`As context length grows, the KV cache becomes larger and increases GPU memory traffic.`  
A-MEM creates:  
````
M4

Keywords:
KV cache, context length, GPU memory,
memory traffic

Tags:
LLM inference, GPU performance, memory

Description:
Longer context increases KV-cache size,
resulting in increased GPU memory traffic.
````
Embed:  
$e_4=Embed(M4)$  

Similarity search might return:  
````
M3    0.89
M1    0.68
M2    0.65
````
The LLM evaluates those relationships.  

It might decide:  
````
M4 ↔ M3
M4 ↔ M1
````

Now look at our graph:  
````````````````
       M1
      /  \
     M2   M4
      \   /
       M3
````````````````
We're beginning to get an interconnected conceptual structure.  
But now comes one of A-MEM's most distinctive ideas.  

**Existing memories can evolve**  
When M4 arrived, M3 originally looked like:  
````
M3

Keywords:
KV cache, decoding, keys, values

Tags:
transformers, inference

Description:
KV caching avoids recomputing keys and
values during decoding.
````
But M4 gives us additional context.  
The system now understands that KV caching isn't merely a computational optimization—it also has memory-system consequences.  

So the LLM might update M3's generated metadata:  
````
M3 — UPDATED

Keywords:
KV cache, decoding, keys, values,
memory footprint, context length

Tags:
transformers, inference,
GPU memory

Description:
KV caching avoids recomputation during
autoregressive decoding but its memory
footprint grows with context length.
````

Notice what didn't change.  
The original interaction remains:  
`During autoregressive decoding, KV caching prevents recomputation...`  

A-MEM isn't rewriting history.  
Instead:  
`Experience stays fixed. interpretation can involve`  

### Interaction 5  
Now the user says:  
`Large KV caches can make decoding memory-bandwidth bound because the GPU repeatedly reads keys and values from HBM.`  
A-MEM constructs:  
````
M5

Keywords:
KV cache, HBM, memory bandwidth,
decoding, GPU

Tags:
GPU performance, LLM inference,
memory bandwidth

Description:
Large KV caches increase HBM traffic and
can cause autoregressive decoding to become
memory-bandwidth bound.
````

Embedding:  
$e_5=Embed(M5)$  

Similarity search might produce:  
````
M4    0.94
M3    0.82
M1    0.49
M2    0.45
````
The LLM decides:  
````
M5 ↔ M4
M5 ↔ M3
````
Now our memory network becomes:  
``````````````````````````````````
                 M1
                /  \
               /    \
              M2     M4
               \    / \
                \  /   \
                 M3 ─── M5
``````````````````````````````````

Look at what has emerged from five independent interactions.  

We never manually created a taxonomy like:  
````````````````````````````````````
Transformers
 ├── Attention
 │    └── Complexity
 └── Inference
      └── KV Cache
           └── GPU Memory
                └── Bandwidth
````````````````````````````````````
Instead, the structure emerged from experience.  
That's the "evolutionary" aspect of A-MEM.  

## Now let's ask the agent a question  
`Why can long-context decoding become slow?`  

Create query embedding:  
$e_q=Embed(Q)$  
Vector search might return:  
````
M4    0.88
M1    0.74
M5    0.69
````
Suppose \(k=2\).  
Pure vector RAG retrieves:  
````
M4
M1
````
And stops.  
But A-MEM now has another source of information.  

It can start from M4:  
````````````````````````````
                  QUERY
                    │
                    ▼
                   M4
                 / | \
                /  |  \
              M1   M3  M5
`````````````````````````````
Now M5 becomes available because:  
`M4 ↔ M5`  
even though M5 wasn't in our top-2 vector search.  
And M5 happens to contain the crucial insight:  
`Large KV caches can make decoding memory-bandwidth bound.`  
So retrieval becomes:  
$Q ⟶ M4 -> M5$  
That's much closer to **associative recall**.  











