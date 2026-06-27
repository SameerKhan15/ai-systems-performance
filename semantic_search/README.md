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