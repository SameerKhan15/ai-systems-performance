"""
HNSW Toy Visualization Lab
--------------------------

This script recreates the hand-worked HNSW example with 8 two-dimensional vectors.

It shows:
1. The vectors A-H in 2D space.
2. The HNSW graph at Layer 2, Layer 1, and Layer 0.
3. Brute-force nearest-neighbor search for Q = (8.5, 1.5).
4. HNSW-style search using upper-layer greedy navigation and Layer-0 expanded search.
5. A search visualization showing visited nodes and the final top-k answer.

Run:
    python hnsw_toy_visualization.py

Install dependencies:
    pip install numpy matplotlib
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


Point = Tuple[float, float]
Graph = Dict[int, Dict[str, List[str]]]


# ---------------------------------------------------------------------
# 1. Toy vectors from our hand-worked example
# ---------------------------------------------------------------------

VECTORS: Dict[str, Point] = {
    "A": (1, 1),
    "B": (2, 2),
    "C": (1, 4),
    "D": (4, 4),
    "E": (5, 1),
    "F": (7, 2),
    "G": (8, 4),
    "H": (9, 1),
}

QUERY: Point = (8.5, 1.5)

# We use the stable graph from the lesson.
# Each layer has its own adjacency list.
# A node with max layer L exists at all layers 0..L, but each layer has separate edges.
GRAPH: Graph = {
    2: {
        "A": ["G"],
        "G": ["A"],
    },
    1: {
        "A": ["C", "E"],
        "C": ["A", "G"],
        "E": ["A", "G"],
        "G": ["C", "E"],
    },
    0: {
        "A": ["B"],
        "B": ["A", "C"],
        "C": ["B", "D"],
        "D": ["C", "E"],
        "E": ["D", "F"],
        "F": ["E", "G", "H"],
        "G": ["F", "H"],
        "H": ["F", "G"],
    },
}

ENTRY_POINT = "A"


# ---------------------------------------------------------------------
# 2. Distance and search helpers
# ---------------------------------------------------------------------

def euclidean_distance(p: Point, q: Point) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def distance_to_query(node: str, query: Point = QUERY) -> float:
    return euclidean_distance(VECTORS[node], query)


def brute_force_search(query: Point, k: int = 2) -> List[Tuple[str, float]]:
    """Exact nearest neighbors by scanning all vectors."""
    distances = [(node, euclidean_distance(point, query)) for node, point in VECTORS.items()]
    return sorted(distances, key=lambda x: x[1])[:k]


def greedy_search_layer(
    start_node: str,
    layer: int,
    query: Point,
) -> Tuple[str, List[str]]:
    """
    Greedy HNSW navigation on one upper layer.

    Starting from start_node, repeatedly move to a neighbor if that neighbor
    is closer to the query. Stop when no neighbor improves distance.
    """
    current = start_node
    path = [current]

    improved = True
    while improved:
        improved = False
        current_distance = distance_to_query(current, query)

        for neighbor in GRAPH[layer].get(current, []):
            neighbor_distance = distance_to_query(neighbor, query)
            if neighbor_distance < current_distance:
                current = neighbor
                path.append(current)
                improved = True
                break

    return current, path


def expanded_search_layer0(
    start_node: str,
    query: Point,
    ef_search: int = 4,
    k: int = 2,
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Simplified Layer-0 expanded search.

    This is not a full production HNSW implementation. It is intentionally simple
    for learning:
      - Maintain a visited set.
      - Repeatedly expand the closest unexpanded candidate.
      - Stop after roughly ef_search visited nodes.
      - Return top-k by distance among visited nodes.
    """
    visited: Set[str] = set()
    expanded: Set[str] = set()
    candidates: List[str] = [start_node]
    visit_order: List[str] = []

    while candidates and len(visited) < ef_search:
        # Pick the currently closest unexpanded candidate.
        candidates = sorted(candidates, key=lambda node: distance_to_query(node, query))
        current = candidates.pop(0)

        if current in expanded:
            continue

        expanded.add(current)

        if current not in visited:
            visited.add(current)
            visit_order.append(current)

        for neighbor in GRAPH[0].get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                visit_order.append(neighbor)
                candidates.append(neighbor)

                if len(visited) >= ef_search:
                    break

    ranked = sorted(
        [(node, distance_to_query(node, query)) for node in visited],
        key=lambda x: x[1],
    )

    return ranked[:k], visit_order


def hnsw_search(query: Point, ef_search: int = 4, k: int = 2):
    """
    HNSW-style search:
      1. Start at top entry point.
      2. Greedily navigate Layer 2.
      3. Descend by same node ID to Layer 1.
      4. Greedily navigate Layer 1.
      5. Descend by same node ID to Layer 0.
      6. Expanded local search on Layer 0.
    """
    current = ENTRY_POINT
    full_path = []

    # Upper layers: greedy search.
    for layer in [2, 1]:
        current, path = greedy_search_layer(current, layer, query)
        full_path.extend([(layer, node) for node in path])

    # Bottom layer: expanded search.
    top_k, layer0_visit_order = expanded_search_layer0(
        start_node=current,
        query=query,
        ef_search=ef_search,
        k=k,
    )

    full_path.extend([(0, node) for node in layer0_visit_order])

    return top_k, full_path, current, layer0_visit_order


# ---------------------------------------------------------------------
# 3. Visualization helpers
# ---------------------------------------------------------------------

def draw_layer(
    layer: int,
    output_path: Path,
    title: str,
    query: Point | None = None,
    highlight_nodes: Set[str] | None = None,
    answer_nodes: Set[str] | None = None,
):
    """Draw one graph layer."""
    highlight_nodes = highlight_nodes or set()
    answer_nodes = answer_nodes or set()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw edges.
    for node, neighbors in GRAPH[layer].items():
        x1, y1 = VECTORS[node]
        for neighbor in neighbors:
            # Avoid drawing each undirected edge twice.
            if node < neighbor:
                x2, y2 = VECTORS[neighbor]
                ax.plot([x1, x2], [y1, y2], linewidth=1.5, alpha=0.7)

    # Draw nodes that exist on this layer.
    for node in GRAPH[layer].keys():
        x, y = VECTORS[node]

        if node in answer_nodes:
            marker = "*"
            size = 300
        elif node in highlight_nodes:
            marker = "s"
            size = 180
        else:
            marker = "o"
            size = 120

        ax.scatter(x, y, s=size, marker=marker)
        ax.text(x + 0.08, y + 0.08, node, fontsize=12)

    # Draw query if provided.
    if query is not None:
        ax.scatter(query[0], query[1], s=220, marker="X")
        ax.text(query[0] + 0.08, query[1] + 0.08, "Q", fontsize=12)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def draw_all_vectors(output_path: Path):
    """Draw all vectors and the query without graph edges."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for node, point in VECTORS.items():
        x, y = point
        ax.scatter(x, y, s=120)
        ax.text(x + 0.08, y + 0.08, node, fontsize=12)

    ax.scatter(QUERY[0], QUERY[1], s=220, marker="X")
    ax.text(QUERY[0] + 0.08, QUERY[1] + 0.08, "Q", fontsize=12)

    ax.set_title("Toy vector space: A-H plus query Q")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_results(ef_search: int = 4, k: int = 2):
    print("\n=== Brute-force exact search ===")
    for rank, (node, dist) in enumerate(brute_force_search(QUERY, k=8), start=1):
        print(f"{rank:>2}. {node}  distance={dist:.3f}")

    print(f"\nExact top-{k}: {[node for node, _ in brute_force_search(QUERY, k=k)]}")

    print("\n=== HNSW-style search ===")
    top_k, full_path, layer0_start, layer0_visit_order = hnsw_search(
        QUERY,
        ef_search=ef_search,
        k=k,
    )

    print(f"Query Q = {QUERY}")
    print(f"efSearch = {ef_search}")
    print(f"Layer-0 start after upper-layer navigation: {layer0_start}")

    print("\nSearch path / visited nodes:")
    for layer, node in full_path:
        print(f"  Layer {layer}: {node}  distance={distance_to_query(node):.3f}")

    print(f"\nLayer-0 visit order: {layer0_visit_order}")
    print(f"HNSW top-{k}: {[node for node, _ in top_k]}")
    print("\nDistances for HNSW top-k:")
    for node, dist in top_k:
        print(f"  {node}: {dist:.3f}")


def main():
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    ef_search = 4
    k = 2

    top_k, full_path, layer0_start, layer0_visit_order = hnsw_search(
        QUERY,
        ef_search=ef_search,
        k=k,
    )

    visited_nodes = {node for _, node in full_path}
    answer_nodes = {node for node, _ in top_k}

    draw_all_vectors(output_dir / "00_vectors_and_query.png")

    draw_layer(
        layer=2,
        output_path=output_dir / "01_layer_2_express_graph.png",
        title="HNSW Layer 2: express graph",
        query=QUERY,
        highlight_nodes={node for layer, node in full_path if layer == 2},
        answer_nodes=set(),
    )

    draw_layer(
        layer=1,
        output_path=output_dir / "02_layer_1_regional_graph.png",
        title="HNSW Layer 1: regional graph",
        query=QUERY,
        highlight_nodes={node for layer, node in full_path if layer == 1},
        answer_nodes=set(),
    )

    draw_layer(
        layer=0,
        output_path=output_dir / "03_layer_0_local_graph.png",
        title="HNSW Layer 0: local graph with search result",
        query=QUERY,
        highlight_nodes=visited_nodes,
        answer_nodes=answer_nodes,
    )

    print_results(ef_search=ef_search, k=k)

    print("\nSaved plots:")
    for path in sorted(output_dir.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()