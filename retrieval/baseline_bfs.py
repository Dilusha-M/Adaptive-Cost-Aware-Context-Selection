"""Baseline breadth-first search retrieval.

Implements a simple BFS traversal starting from query-matched
nodes, retrieving all reachable nodes within a maximum depth.
This serves as the baseline for comparison with the adaptive algorithm.

Uses node-level token estimation for accurate context sizing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from graph.graph_builder import DependencyGraph
from graph.graph_search import GraphSearch
from metrics.token_estimator import TokenEstimator, TokenEstimate


@dataclass
class RetrievalResult:
    """Result of a retrieval operation.

    Attributes:
        algorithm_name: Name of the retrieval algorithm used.
        start_nodes: Query start node IDs.
        retrieved_nodes: Set of retrieved node IDs.
        retrieved_files: Set of file paths retrieved.
        traversal_order: List of node IDs in traversal order.
        skipped_nodes: Dict mapping skipped node IDs to reason strings.
        token_estimate: Token cost estimate for the retrieved set.
        execution_time_ms: Wall-clock execution time in milliseconds.
        query: The original user query.
    """
    algorithm_name: str
    start_nodes: list[str]
    retrieved_nodes: set[str]
    retrieved_files: set[str]
    traversal_order: list[str]
    skipped_nodes: dict[str, str]
    token_estimate: TokenEstimate
    execution_time_ms: float
    query: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "algorithm_name": self.algorithm_name,
            "query": self.query,
            "start_nodes": self.start_nodes,
            "retrieved_files": sorted(self.retrieved_files),
            "retrieved_nodes": sorted(self.retrieved_nodes),
            "traversal_order": self.traversal_order,
            "skipped_nodes": self.skipped_nodes,
            "token_estimate": self.token_estimate.to_dict(),
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


class BaselineBFS:
    """Breadth-first search retrieval algorithm.

    Performs a level-by-level BFS from start nodes, retrieving
    all reachable nodes up to the specified maximum depth.
    No budget constraints or scoring - retrieves everything
    within the depth limit.

    Uses node-level token estimation for accurate context sizing.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        max_depth: int = 3,
    ) -> None:
        """Initialize the BFS retriever.

        Args:
            graph: The DependencyGraph to traverse.
            max_depth: Maximum BFS depth from start nodes.
        """
        self.graph = graph
        self.max_depth = max_depth
        self._query = ""

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> RetrievalResult:
        """Execute BFS retrieval for a query.

        Args:
            query: User's request string (e.g., "Modify LoginService").
            top_k: Number of top start nodes to use.

        Returns:
            RetrievalResult with all retrieved nodes and metadata.
        """
        self._query = query
        start_time = time.perf_counter()

        # Find start nodes via keyword matching
        start_nodes = GraphSearch.find_start_nodes(self.graph, query, top_k=top_k)

        if not start_nodes:
            # Fallback: use all file nodes
            start_nodes = self.graph.get_all_files()[:1]

        # BFS traversal
        retrieved: set[str] = set()
        traversal_order: list[str] = []
        skipped: dict[str, str] = {}

        # Queue entries: (node_id, depth)
        queue: list[tuple[str, int]] = [(nid, 0) for nid in start_nodes]

        for start in start_nodes:
            retrieved.add(start)
            traversal_order.append(start)

        while queue:
            current, depth = queue.pop(0)

            if depth >= self.max_depth:
                continue

            current_node = self.graph.get_node_data(current)
            if not current_node:
                continue

            # Get neighbors (successors)
            neighbors = list(self.graph.graph.successors(current))

            for neighbor in neighbors:
                if neighbor in retrieved:
                    continue

                neighbor_node = self.graph.get_node_data(neighbor)

                # Skip import-module-name nodes that don't have file nodes
                if neighbor_node and neighbor_node.node_type == "file":
                    retrieved.add(neighbor)
                    traversal_order.append(neighbor)
                    # Add neighbors of this neighbor to queue
                    if depth + 1 < self.max_depth:
                        queue.append((neighbor, depth + 1))
                else:
                    # It's a module name reference, not a real node
                    # Try to find the actual file node
                    retrieved.add(neighbor)
                    traversal_order.append(neighbor)
                    skipped[neighbor] = "module reference (resolved via import)"

        # Compute token estimate using node-level estimation
        token_estimate = TokenEstimator.estimate_nodes(retrieved, self.graph)

        # Collect unique file paths
        retrieved_files: set[str] = set()
        for nid in retrieved:
            nd = self.graph.get_node_data(nid)
            if nd:
                retrieved_files.add(nd.file_path)

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000

        return RetrievalResult(
            algorithm_name="Baseline BFS",
            start_nodes=start_nodes,
            retrieved_nodes=retrieved,
            retrieved_files=retrieved_files,
            traversal_order=traversal_order,
            skipped_nodes=skipped,
            token_estimate=token_estimate,
            execution_time_ms=execution_time,
            query=query,
        )
