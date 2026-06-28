# Indexing Algorithm: Hierarchical Navigable Small World (HNSW)  
## Introduction  
HNSW, is a graph-based algorithm for approximate nearest neighbor search.  
It is commonly used in vector databases to quickly find vectors that are similar to a query vector, such as embeddings for text, images, documents, or user behavior.  

In a **brute-force search**, the query vector is compared against every vector in the dataset. This gives exact results, but it becomes expensive when the dataset contains millions of high-dimensional vectors.  
HNSW speeds this up by building a multi-layer graph index. Each vector is represented as a node, and nearby vectors are connected by edges. The upper layers contain fewer nodes and allow long-distance jumps across the vector space, while the bottom layer contains all vectors and supports fine-grained local search.  

At query time, HNSW starts from an entry point in the top layer and greedily moves toward nodes that are closer to the query. It then descends layer by layer, carrying forward the best node found so far. At the bottom layer, it performs a broader local search to return the top-k nearest neighbors. This allows HNSW to examine only a small fraction of the dataset while still achieving high recall.  

The main tuning parameters are M, which controls how many graph neighbors each node keeps; efConstruction, which controls how carefully the graph is built; and efSearch, which controls how broadly the graph is searched at query time. Increasing these values generally improves recall but also increases memory usage, build time, or query latency.  

In short, HNSW trades exact exhaustive search for a carefully constructed navigable graph, enabling fast and accurate vector similarity search at large scale.  

## Notes on visualizations  
The main learning points are visible in the output:  
`Layer 2: A → G  
 Layer 1: stay near G  
 Layer 0: expand around G, discover F and H  
Return top-2: H, F`  

![](plots/00_vectors_and_query.png "vectors_and_query")  
![](plots/01_layer_2_express_graph.png "layer2_express_qraph")
![](plots/02_layer_1_regional_graph.png "layer1_regional_qraph")
![](plots/03_layer_0_local_graph.png "layer0_local_qraph")

## Algorithm Details via Toy Example(s)  
Let: 
M = 2 // each node keeps at most 2 neighbors per layer  
efConstruction = 3 // each node considers up to 3 promising candidates during insertion  

Inter-node Euclidean distance (**eDist**) formula:  
`(B, A) = sqrt{(b1 - a1)^2 + (b2 - a2)^2} + ... = ... `  
`e.g. A = (1,1), B = (2,2)`  
`(B, A) = sqrt{(1-2)^2 + (1-2)^2} = sqrt(2) = 1.41`  

To be inserted Vectors:  
### Vector, Coordinates, Maximum Layer  
`A, (1,1), 2`  
`B, (2,2), 0`  
`C, (1,4), 1`  
`D, (4,4), 0`  
`E, (5,1), 1`  
`F, (7,2), 0`  
`G, (8,4), 2`  
`H, (9,1), 0`  

A vector exists from layer 0 to its maximum layer. 

### INSERT A = (1,1), max_layer = 2  
`Layer2: A`  
`Layer1: A`  
`Layer0: A`  

### INSERT B = (2,2), max_layer = 0  
Start at layer 2. The layer has only a single node. Descend from A to the same node at Layer 1. Layer 1 also has only a single node. Descend from A to the same node at Layer 0.  
Layer 0 also has a single node. Connect B to A.  

`Layer2: A`  
`Layer1: A`  
`Layer0: A - B`  

### INSERT C = (1,4), max_layer = 1  
Start at layer 2. The layer has only a single node. Descend from A to the same node at Layer 1. Layer 1 also has only a single node. Connect this node to A. 
`Layer1: A - C`  
Move to A and descend to Layer 0's node A.  
Layer 0 has node A and B.  
`eDist(C,A) = 3`  
`eDist(C,B) = 2.24`  

M=2, so connect C to both nodes.  
```text
Layer 2:  A

Layer 1:  A --- C

Layer 0:  A --- B
           \   /
             C
```


### INSERT D = (4,4), max_layer = 0  
Start at layer 2. Since the layer only has a single node, descend to Layer 1's node A.  
At Layer 1:  
`eDist(D,A) = 4.24`  
`eDist(D,C) =3`  

Since C is closer, move to that node and descend to Layer 0's node C.  

At Layer 0:  
`eDist(D,A) = 4.24`  
`eDist(D,C) = 3`  
`eDist(D,B) = 2.83`  

`efConstruction = 3` - All three will be held in the candidate list.  

With M=2, initially select [B,C]  

```text
Layer 0:

  A --- B --- D
   \   |    /
    \  |   /
       C
```
Note that B now has 3 neighbours. Since M = 2, it violates that property. B should keep its closest two.  
`eDist(B,A) = 1.41`  
`eDist(B,C) = 2.24`  
`eDist(B,D) = 2.83`  

B prunes D. C also has 3 neighbours now: A, B, D  
`eDist(C,A) = 2.24`  
`eDist(C,B) = 3.0`  
`eDist(C,D) = 3.0`  

Tie between B and D. Retain D.  
`Layer2: A`  
`Layer1: A - C`  
`Layer0: B - A - C - D`  

### INSERT E = (5,1), max_layer = 1  
Start at layer 2. Descend to Layer 1's node A.  
At Layer 1:  
`eDist(E,A) = 4  
 eDist(E,C) = 5`  

Since M=2, connect to both nodes.  
`Layer1: A - C  
         -  -  
           E  `  
Since node E is the nearest from node A, move to the node and descend to Layer 0's node A.  
At Layer 0:  
`Layer 0: A - B - C - D`  

**Note that efConstruction = 3. This does NOT mean "only consider the first three nodes encountered: A,B,C". It means the algorithm maintains a search frontier / candidate set of roughly size 3 while 
exploring the graph. It can still discover D if the search expands through C**  

`eDist(E,A) = 4`  
Now explore A's neighbour(s). Calculate distance from B  
`eDist(E,B) = 3.16`  

Now explore B's neighbours(s). A is already computed. Calculate distance from C  
`eDist(E,C) = 5`  

Does it stop before D? This is the subtle part. The algorithm does not simply say: "I have seen 3 nodes, so stop". Instead C is still in the candidate frontier (as the 3rd node). So it will explore its neighbours.  
`eDist(E,D) = 3.16`  

efConstruction = 3, best candidates are: [B,D,A]  
`At Layer 0: A - B - C- D  
                 -      -  
                    E  `  

Now B has 3 nodes. One need to be pruned.  
`eDist(B,A) = 1.41  
 eDist(B,C) = 2.24  
 eDist(B,E) = 3.16`  
B - E gets pruned  
`At Layer 0: A - B - C - D - E` 

`Layer 2: A  
 Layer 1: A - C  
          -   -  
            E  
 Layer 0: A - B - C - D - E`  

### INSERT F = (7,2), max_layer = 0  
At Layer 2: only a single node. Descend to Layer 1's node A.  
At Layer1:  
`eDist(F,A) = 6.08  
 eDist(F,C) = 6.32  
 eDist(F,E) = 2.24`
Move to E and descend to Layer 0.  

At Layer 0:  
Since efConstruction = 3, consider 3 candidate nodes from E.  
eDist(F,E) = 2.24  
eDist(F,D) = 3.61  
eDist(F,C) = 6.32  
eDist(F,B) = // might compute this as well but we will skip for this example  

select [E,D]  

`Layer 0: A - B - C - D - E  
                      -   -  
                        F`  
Node D has 3 neighbours, which is a violation.  
`eDist(D,C) = 3   
 eDist(D,E) = 3.16  
 eDist(D,F) = 3.61`  
D-F gets pruned. 
`Layer 0: A - B - C - D - E - F`  

`Layer 2: A  
 Layer 1: A - C  
          -   -  
            E  
 Layer 0: A - B - C - D - E - F`  

### INSERT G = (8,4), max_layer = 2  
At Layer 2: since there is only a single node, connect G to A.  
`Layer 2: A - G`  
Move to A and descend to Layer 1.  

At Layer 1:  
`eDist(G,A) = 7.62  
 eDist(G,C) = 7  
 eDist(G,E) = 4.24`  
efConstruction = 3 so we consider all three distances. M = 2, so we pick the best two: [E,C]  

`Layer 1: 
          A 
       -     -  
       C  -  E  
       -     -  
          G`  
Node E is in violation. Need to prune one edge.  
`eDist(E,G) = 4.24  
 eDist(E,C) = 5   
 eDist(E,A) = 4`  
Prune [E,C]  

`Layer 1: 
          A 
       -     -  
       C     E  
       -     -  
          G`  

eDist(G,E) = 4.24  
eDist(G,C) = 7  
eDist(G,A) = // lets assume > 7  

Move to node E and descend to Layer 0  

At Layer 0:  
`A - B - C - D - E - F`  
`eDist(G,E) = 4.24  
 eDist(G,F) = 2.24  
 eDist(G,D) = 4  
 eDist(G,C) = // lets assume > 4.24`
M=2, pick [F,D]  

Layer 0: 
`A - B - C - D - E - F
             -       -  
                 G`    
Node D is in violation.  
D:[C,E,G]  
eDist(D,C) = .. 
eDist(D,E) = .. 
eDist(D,G) = .. // prunes this  

Layer 0: `A - B - C - D - E - F - G`  

Current graph state:  
`Layer 2: A - G
 Layer 1: 
          A 
       -     -  
       C     E  
       -     -  
          G
 Layer 0: 
 A - B - C - D - E - F - G`   

### INSERT H = (9,1), max_layer = 0  
At Layer 2:  
`eDist(H,A) = 8  
 eDist(H,G) = 3.16`  
Move to node G and descend to Layer 1 G node  

At Layer 1:  
`eDist(H,G) = 3.16  
 eDist(H,C) = 8.54  
 eDist(H,E) = 4  
 eDist(H,A) = ... // assume its > 8.54` 

Move to G and descend to Layer 0's G  

At Layer 0:  
`eDist(H,G) = 3.16  
eDist(H,E) = 4  
eDist(H,F) = 2.24`  
...  
Pick [G,F]  
`Layer 0: A - B - C - D - E - F - G 
                             -   - 
                               H`  
F is in violation.  
`eDist(F,E) = 2.24  
 eDist(F,G) = 2.24  
 eDist(F,H) = 2.24`  
They al tier. Pruned E.  

`Layer 0: A - B - C - D - E  F - G 
                              -   - 
                               H`  
Notice: Disconnected left and right chains. 
Artifact of our simplified toy rules.  
M=2 is very small. Real HNSW uses better neighbor-selection heuristics.  
Real layer 0 often allow more neighbors than upper layers.  
A more stable version would be: A - B - C - D - E - F - G - H  

## Overall  
* INSERT Vector → use upper layers to navigate close  
* Descend by some node ID  
* Search locally on the target layer  
* Connect to selected neighbors  
* Prune if neighbor list exceed capacity  

# Query Search  
`Q = (8.5,1.5)`  
`Layer 2: A - G 
 Layer 1: A - E 
         -   -  
         C - G  
 Layer 0: A - B - C - D - E - F - G - H`  

Step1: Layer 2 greedy search   
eDist(Q,A) = 7.52  
eDist(Q,G) = 2.55  
Move to G and descend to Layer 1.  

Layer 1:  
`eDist(Q,G) = 2.55  
 eDist(Q,E) = 3.54  
 eDist(Q,C) = 7.91`  

Now descend to Layer 0 from G  

Layer 0:  
`Layer 0: A - B - C - D - E - F - G - H`  
Expanded search starting at node G.  
efSearch = 4 // Explore about 4 promising candidates  
k=2 // return the best 2  
`eDist(Q,G) = 2.55  
eDist(Q,F) = 1.58  
eDist(Q,H) = 0.71`  
1 more ...  
Expand H neighbor, already explored. Expand F neighbor: G already explored.  
`eDist(Q,E) = 3.54`  

Return top_k = 2 = [H,F]  
**At production scale, the same also becomes: "checks 100s of nodes instead of millions"**  
ß